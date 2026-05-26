from mb_internal_scripts.core_lists import models_map, datasets_map
import os

task_list1 = [
]

task_list2 = [
]

task_list3 = [
]


def task_list_parser(raw_task_list):
    """解析任务列表，支持数据集重复语法如 dataset*3"""
    task_list = []
    for task in raw_task_list:
        infer_type_temp, datasets_str, model_config = task[0], task[1], task[2]

        # 处理数据集重复语法 dataset*3
        repeated_datasets = []
        for dataset in datasets_str.split(','):
            dataset = dataset.strip()
            if '*' in dataset:
                parts = dataset.split('*')
                assert len(parts) == 2, '重复数据集定义出错'
                repeated_datasets.extend([parts[0]] * int(parts[1]))
            else:
                repeated_datasets.append(dataset)

        # 原有逻辑保持不变
        model_configs = models_map.get(
            model_config, [[infer_type_temp, model_config]])

        for dataset in repeated_datasets:
            datasets = datasets_map.get(dataset, [dataset])
            for _dataset in datasets:
                for model_config in model_configs:
                    infer_type, model_path = model_config
                    model_path = os.path.normpath(model_path)
                    if infer_type_temp:  # 强行绑定特定 infer_type
                        infer_type = infer_type_temp
                    if not infer_type:
                        infer_type = 'vllm_general_sft'
                    if 'with_chat_template_gen_e8a78d' in _dataset:
                        infer_type = 'vllm_general_base'
                    if 'benbench' in _dataset and 'sft' in infer_type:
                        print('=' * 100)
                        print(
                            'WARNING: contamination dataset is not supported for sft model')
                        print('=' * 100)
                        infer_type = infer_type.replace('sft', 'base')
                    if len(task) > 3 and task[3] is not None:
                        task_list.append(
                            [infer_type, _dataset, model_path, task[3]])
                    else:
                        task_list.append([infer_type, _dataset, model_path])

    # for task in task_list:
        # print(task)
    return task_list


task_list1 = task_list_parser(task_list1)
task_list2 = task_list_parser(task_list2)
task_list3 = task_list_parser(task_list3)

if __name__ == '__main__':
    raw_task_list = [
        ['', '6h*2', 'qwen3'],
    ]
    for task in task_list_parser(raw_task_list):
        print(task)
