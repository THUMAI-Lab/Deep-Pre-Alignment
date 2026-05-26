import os.path as osp
from copy import deepcopy

from mmengine.config import read_base

with read_base():
    from ..dataset_collections.base_scaling_bench_loss import base_datasets_total
    from ..summarizers.scaling_bench_loss import summarizer
    from ..models.small_models.scaling_bench_model import base_scaling_model
    from ..cluster.local_cluster import eval, infer_num_worker as infer

models_path = [
{HF_MODEL_PATH}
]

# NOTE: num_gpus_per_infer should equal to the GPU nums
# when the task is begin, cannot change it!
num_gpus_per_infer = {NUM_GPUS}
num_gpus_per_model = 1

infer['runner']['max_num_workers'] = num_gpus_per_infer
infer['runner']['max_workers_per_gpu'] = num_gpus_per_model


infer['partitioner']['num_worker'] = num_gpus_per_infer // num_gpus_per_model

scaling_models = []

for model_path in models_path:
    if model_path.endswith('/'):
        model_path = model_path[:-1]
    # assume dir name is /{model_size}/{iter}
    abbr = osp.basename(osp.dirname(model_path))
    ckpt_iter = osp.basename(model_path)

    summarizer_abbr = f'{abbr}@{ckpt_iter}'

    tmp_scaling_model_cfg = deepcopy(base_scaling_model)
    tmp_scaling_model_cfg['abbr'] = f'{abbr}@{ckpt_iter}-scaling'
    tmp_scaling_model_cfg['summarizer_abbr'] = summarizer_abbr
    tmp_scaling_model_cfg['path'] = model_path
    scaling_models.append(tmp_scaling_model_cfg)

model_dataset_combinations = []
models = []
datasets = []

model_dataset_combinations.append(dict(models=scaling_models, datasets=base_datasets_total))
models.extend(scaling_models)
datasets.extend(base_datasets_total)
