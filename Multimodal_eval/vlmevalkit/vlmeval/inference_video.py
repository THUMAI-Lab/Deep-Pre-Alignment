import torch
import torch.distributed as dist
from vlmeval.config import supported_VLM
from vlmeval.utils import track_progress_rich
from vlmeval.smp import *
import sys

FAIL_MSG = 'Failed to obtain answer via API.'


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, nargs='+', required=True)
    parser.add_argument('--model', type=str, nargs='+', required=True)
    parser.add_argument('--nproc', type=int, default=4, required=True)
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()
    return args


# Only API model is accepted
def infer_data_api(work_dir, model_name, dataset, nframe=8, pack=False, samples_dict={}, api_nproc=4, batch_size=4):
    rank, world_size = get_rank_and_world_size()
    assert rank == 0 and world_size == 1
    dataset_name = dataset.dataset_name
    data = dataset.data
    if index_set is not None:
        data = data[data['index'].isin(index_set)]

    model = supported_VLM[model_name]() if isinstance(model_name, str) else model_name
    assert getattr(model, 'is_api', False)

    indices = list(samples_dict.keys())
    structs = [dataset.build_prompt(samples_dict[idx], num_frames=nframe,
                                    video_llm=getattr(model, 'VIDEO_LLM', False)) for idx in indices]

    packstr = 'pack' if pack else 'nopack'
    out_file = f'{work_dir}/{model_name}_{dataset_name}_{nframe}frame_{packstr}_supp.pkl'
    res = load(out_file) if osp.exists(out_file) else {}

    structs = [s for i, s in zip(indices, structs) if i not in res]
    indices = [i for i in indices if i not in res]

    gen_func = model.generate
    structs = [dict(message=struct, dataset=dataset_name) for struct in structs]

    if len(structs):
        track_progress_rich(gen_func, structs, nproc=api_nproc, chunksize=api_nproc, save=out_file, keys=indices)

    res = load(out_file)
    return res

def infer_data(model_name, work_dir, dataset, out_file, nframe=8, verbose=False, api_nproc=4, batch_size=4):
    print(f"processing batch_size {batch_size}", flush=True, file=sys.stderr)
    dataset_name = dataset.dataset_name
    res = {}
    if osp.exists(out_file):
        res = load(out_file)

    data = dataset.data
    data_indices = list(data['index'])

    all_finished = all(idx in res for idx in data_indices)
    if all_finished:
        print("All tasks are already finished.")
        return

    # 筛选出需要进行推理的数据
    data_to_infer = data[~data['index'].isin(res)]
    lt = len(data_to_infer)

    model = supported_VLM[model_name]() if isinstance(model_name, str) else model_name

    is_api = getattr(model, 'is_api', False)
    if is_api:
        indices_to_infer = list(data_to_infer['index'])
        supp = infer_data_api(
            work_dir=work_dir,
            model_name=model_name,
            dataset=dataset,
            index_set=set(indices_to_infer),
            api_nproc=api_nproc,
            nframe=nframe)
        res.update(supp)
        for idx in indices_to_infer:
            assert idx in supp
        res = {k: res[k] for k in data_indices if k in res}
        dump(res, out_file)
        return model_name
    
    # if hasattr(model, 'set_dump_image'):
    #     model.set_dump_image(dataset.dump_image)

    for i in tqdm(range(0, lt, batch_size)):
        # 获取当前批次的数据
        mini_batch_data = data_to_infer.iloc[i:i + batch_size]
        
        prompts = []
        indices = []
        for j in range(len(mini_batch_data)):
            idx = mini_batch_data.iloc[j]['index']
            
            # 优先使用模型自定义的帧数
            current_nframe = getattr(model, 'nframe', 0) or nframe
            
            # 构建prompt
            struct = dataset.build_prompt(mini_batch_data.iloc[j], video_llm=getattr(model, 'VIDEO_LLM', False))
            
            prompts.append(struct)
            indices.append(idx)

        if not prompts:
            continue

        # 使用 generate_batch 进行批处理
        try:
            if batch_size > 1:
                print(f"Processing batch {i // batch_size + 1} with {len(prompts)} prompts.", flush=True)
                responses = model.generate_batch(messages=prompts, dataset=dataset_name)
            else:
                 responses = [model.generate(message=prompts[0], dataset=dataset_name)]
        except Exception as e:
            print(f"Error during batch generation: {e}. Falling back to single-item generation for this batch.")
            # 如果批处理失败，则回退到单个处理
            responses = []
            for prompt in prompts:
                try:
                    responses.append(model.generate(message=prompt, dataset=dataset_name))
                except Exception as e_single:
                    print(f"Error during single-item fallback: {e_single}. Skipping this item.")
                    responses.append(None)  # 为失败的项添加占位符

        torch.cuda.empty_cache()

        # 将响应映射回索引
        for idx, response in zip(indices, responses):
            if response is not None:
                res[idx] = response
            else:
                res[idx] = None  # 处理失败的响应
            if verbose:
                print(f"Index: {idx}\nResponse: {response}", flush=True)

        # 每个批次处理后保存一次中间结果
        dump(res, out_file)

    dump(res, out_file)
    return model


def infer_data_job_video(
        model,
        work_dir,
        model_name,
        dataset,
        nframe=8,
        pack=False,  # pack 参数保留，以备将来使用或兼容旧配置
        verbose=False,
        subtitle=False,
        api_nproc=4,
        batch_size=4):
    """
    修改后的视频推理封装函数，支持丰富的预测格式并简化了文件处理。
    """
    dataset_name = dataset.dataset_name
    packstr = 'pack' if pack else 'nopack'
    
    # 简化文件名，移除分布式信息
    result_file_name = f'{model_name}_{dataset_name}_{nframe}frame_{packstr}'
    if dataset_name == 'Video-MME':
        subtitle_str = 'subs' if subtitle else 'nosubs'
        result_file_name = f'{result_file_name}_{subtitle_str}'
    
    result_file = osp.join(work_dir, f'{result_file_name}.xlsx')

    if osp.exists(result_file):
        return model_name

    tmp_pkl_file = osp.join(work_dir, f'{result_file_name}.pkl')

    model = infer_data(
        model,
        work_dir=work_dir,
        dataset=dataset,
        nframe=nframe,
        out_file=tmp_pkl_file,
        verbose=verbose,
        api_nproc=api_nproc,
        batch_size=batch_size)

    data_all = load(tmp_pkl_file)

    data = dataset.data
    for x in data['index']:
        if x not in data_all:
            print(f"Warning: Index {x} not found in inference results, will be filled with empty.")
            data_all[x] = {} 

    predictions = []
    descriptions = []
    detailed_predictions = []

    for x in data['index']:
        result_dict = data_all.get(x, {})  # 安全地获取结果
        if isinstance(result_dict, dict):
            predictions.append(str(result_dict.get('prediction', '')))
            descriptions.append(str(result_dict.get('description', '')))
            detailed_predictions.append(str(result_dict.get('detailed_prediction', '')))
        else:  # 兼容非字典格式的旧结果
            predictions.append(str(result_dict))
            descriptions.append('')
            detailed_predictions.append('')

    data['prediction'] = predictions
    data['description'] = descriptions
    data['detailed_prediction'] = detailed_predictions

    if 'image' in data and 'image' in data.columns:
        data.pop('image')

    dump(data, result_file)

    if osp.exists(tmp_pkl_file):
        os.remove(tmp_pkl_file)

    return model