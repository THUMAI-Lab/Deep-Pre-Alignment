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
def infer_data_api(work_dir, model_name, dataset, index_set=None, api_nproc=4, ignore_failed=False, batch_size=4):
    rank, world_size = get_rank_and_world_size()
    assert rank == 0 and world_size == 1
    dataset_name = dataset.dataset_name
    data = dataset.data
    if index_set is not None:
        data = data[data['index'].isin(index_set)]

    model = supported_VLM[model_name]() if isinstance(model_name, str) else model_name
    assert getattr(model, 'is_api', False)
    if hasattr(model, 'set_dump_image'):
        model.set_dump_image(dataset.dump_image)

    lt, indices = len(data), list(data['index'])

    structs = []
    for i in range(lt):
        item = data.iloc[i]
        if hasattr(model, 'use_custom_prompt') and model.use_custom_prompt(dataset_name):
            assert hasattr(model, 'build_prompt')
            struct = model.build_prompt(item, dataset=dataset_name)
        else:
            struct = dataset.build_prompt(item)
        structs.append(struct)

    # structs = [dataset.build_prompt(data.iloc[i]) for i in range(lt)]

    out_file = f'{work_dir}/{model_name}_{dataset_name}_supp.pkl'
    res = {}
    if osp.exists(out_file):
        res = load(out_file)
        if ignore_failed:
            res = {k: v for k, v in res.items() if FAIL_MSG not in v}

    structs = [s for i, s in zip(indices, structs) if i not in res]
    indices = [i for i in indices if i not in res]

    gen_func = model.generate
    structs = [dict(message=struct, dataset=dataset_name) for struct in structs]

    if len(structs):
        track_progress_rich(gen_func, structs, nproc=api_nproc, chunksize=api_nproc, save=out_file, keys=indices)

    res = load(out_file)
    if index_set is not None:
        res = {k: v for k, v in res.items() if k in index_set}
    os.remove(out_file)
    return res


def infer_data(model_name, work_dir, dataset, out_file, verbose=False, api_nproc=4, batch_size=4):
    print(f"processing batch_size {batch_size}", flush=True, file=sys.stderr)
    dataset_name = dataset.dataset_name
    res = {}
    if osp.exists(out_file):
        res = load(out_file)

    data = dataset.data

    lt = len(data)
    data_indices = list(data['index'])

    all_finished = all(idx in res for idx in data_indices)
    if all_finished:
        print("All tasks are already finished.")
        return

    # Data need to be inferred
    data = data[~data['index'].isin(res)]
    lt = len(data)

    model = supported_VLM[model_name]() if isinstance(model_name, str) else model_name

    is_api = getattr(model, 'is_api', False)
    if is_api:
        lt, indices = len(data), list(data['index'])
        supp = infer_data_api(
            work_dir=work_dir,
            model_name=model_name,
            dataset=dataset,
            index_set=set(indices),
            api_nproc=api_nproc)
        for idx in indices:
            assert idx in supp
        res.update(supp)
        res = {k: res[k] for k in data_indices}
        dump(res, out_file)
        return model_name
    else:
        if hasattr(model, 'set_dump_image'):
            model.set_dump_image(dataset.dump_image)

    for i in tqdm(range(0, lt, batch_size)):
        # Get the mini-batch of data
        mini_batch_data = data.iloc[i:i + batch_size]
        
        # Prepare prompts and indices for the mini-batch
        prompts = []
        indices = []
        for j in range(len(mini_batch_data)):
            idx = mini_batch_data.iloc[j]['index']
            
            if hasattr(model, 'use_custom_prompt') and model.use_custom_prompt(dataset_name):
                struct = model.build_prompt(mini_batch_data.iloc[j], dataset=dataset_name)
            else:
                struct = dataset.build_prompt(mini_batch_data.iloc[j])
            
            prompts.append(struct)
            indices.append(idx)

        if not prompts:
            continue
        

        # Call generate_batch for the mini-batch
        if batch_size > 1:
            try:
                print(f"Processing batch {i // batch_size + 1} with {len(prompts)} prompts.", flush=True)
                responses = model.generate_batch(messages=prompts, dataset=dataset_name)
            except Exception as e:
                print(f"Error during batch generation: {e}. Falling back to single-item generation for this batch.")
                # Fallback to single-item generation if batch fails
                responses = []
                for prompt in prompts:
                    try:
                        responses.append(model.generate(message=prompt, dataset=dataset_name))
                    except Exception as e_single:
                        print(f"Error during single-item fallback: {e_single}. Skipping this item.")
                        responses.append(None) # Add a placeholder for the failed item
                    # responses.append(model.generate(message=prompt, dataset=dataset_name))

        else:
            # try:
            #     responses = [model.generate(message=prompts[0], dataset=dataset_name)]
            # except Exception as e_single:
            #     print(f"Error during single-item generation: {e_single}. Skipping this item.")
            #     responses = [None]
            print(f"prompts: {prompts}")
            responses = [model.generate(message=prompts[0], dataset=dataset_name)]

        torch.cuda.empty_cache()

        # Map responses back to indices
        for idx, response in zip(indices, responses):
            if response is not None:
                res[idx] = response
            else:
                res[idx] = None  # Handle failed responses
            if verbose:
                print(f"Index: {idx}\nResponse: {response}", flush=True)

        # Save intermediate results after each batch
        dump(res, out_file)

    dump(res, out_file)
    return model


# A wrapper for infer_data, do the pre & post processing
def infer_data_job(model, work_dir, model_name, dataset, verbose=False, api_nproc=4, ignore_failed=False, batch_size=4):
    dataset_name = dataset.dataset_name
    result_file = osp.join(work_dir, f'{model_name}_{dataset_name}.xlsx')

    if osp.exists(result_file):
        return model_name

    tmp_pkl_file = osp.join(work_dir, f'{model_name}_{dataset_name}.pkl')

    out_file = tmp_pkl_file

    model = infer_data(
        model, work_dir=work_dir, dataset=dataset, out_file=out_file, verbose=verbose, api_nproc=api_nproc, batch_size=batch_size)

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
        result_dict = data_all.get(x, {}) # Use .get for safety
        if isinstance(result_dict, dict):
            predictions.append(str(result_dict.get('prediction', '')))
            descriptions.append(str(result_dict.get('description', '')))
            detailed_predictions.append(str(result_dict.get('detailed_prediction', '')))
        else: # Fallback for non-dict results (like FAIL_MSG from older versions)
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
