import argparse
import os
import os.path as osp
from copy import deepcopy

from mmengine.config import Config, ConfigDict

from opencompass.partitioners import NaivePartitioner, NumWorkerPartitioner
from opencompass.runners import LocalRunner
from opencompass.tasks import OpenICLEvalTask, OpenICLInferTask
from opencompass.utils import get_logger
from opencompass.utils.run import get_config_type, match_cfg_file

logger = get_logger()

MODEL_TYPE_MAP = {'vllm': 'vllm_model', 'hf': 'hf_model'}


def parse_args():
    parser = argparse.ArgumentParser(description='Process run config')
    parser.add_argument('config', type=str, help='Path to the config file')
    parser.add_argument('--models',
                        nargs='+',
                        help='Path to the model(s)',
                        default=None)
    parser.add_argument('--datasets',
                        nargs='+',
                        help='Dataset(s) path or file name',
                        default=None)
    parser.add_argument('--num-gpus',
                        type=int,
                        default=8,
                        help='Number of GPUs')
    parser.add_argument('--num-gpus-per-model',
                        type=int,
                        default=1,
                        help='Each model use how many GPUs')
    parser.add_argument('--model-type',
                        choices=['hf', 'vllm'],
                        type=str,
                        default='vllm',
                        help='Model type')
    parser.add_argument('--model-abbr-type',
                        choices=['ckpt_iter', 'dir_name'],
                        type=str,
                        default='ckpt_iter',
                        help='Model abbr type')
    parser.add_argument('--save-cfg-name',
                        type=str,
                        default='current_config.py',
                        help='Name of the saved config file')
    parser.add_argument('--dataset-check',
                        action='store_true',
                        help='Ignore dataset check')
    args = parser.parse_args()
    return args


def initial_infer_cfg(cfg: ConfigDict) -> ConfigDict:
    """Initial infer config."""
    infer_cfg = dict(
        partitioner=dict(type=get_config_type(NumWorkerPartitioner),
                         num_worker=8),
        runner=dict(type=get_config_type(LocalRunner),
                    max_num_workers=16,
                    max_workers_per_gpu=1,
                    task=dict(type=get_config_type(OpenICLInferTask))),
    )
    cfg['infer'] = infer_cfg
    return cfg


def initial_eval_cfg(cfg: ConfigDict) -> ConfigDict:
    """Initial eval config."""
    eval_cfg = dict(
        partitioner=dict(type=get_config_type(NaivePartitioner), n=10),
        runner=dict(type=get_config_type(LocalRunner),
                    max_num_workers=64,
                    max_workers_per_gpu=1,
                    task=dict(type=get_config_type(OpenICLEvalTask))),
    )
    cfg['eval'] = eval_cfg
    return cfg


def initial_summarizer_cfg(cfg):
    """Initial summarizer config."""
    summarizer_cfg_path_list = [
        'configs/mb_internal/summarizers/mb_example.py',
        'opencompass/configs/summarizers/example.py',
        'configs/summarizers/example.py',
    ]
    summarizer_path = None
    for summarizer_cfg_path in summarizer_cfg_path_list:
        if osp.exists(summarizer_cfg_path):
            summarizer_path = summarizer_cfg_path
            break
    if summarizer_path is not None:
        summarizer_cfg = Config.fromfile(summarizer_path)
        cfg['summarizer'] = summarizer_cfg
    return cfg


def process_infer_cfg(cfg: ConfigDict, args: argparse.Namespace) -> ConfigDict:
    """Process and calculate necessary values in infer config."""
    assert 'infer' in cfg
    num_gpus_per_infer = args.num_gpus
    num_gpus_per_model = args.num_gpus_per_model
    cfg['infer']['runner']['max_num_workers'] = num_gpus_per_infer
    cfg['infer']['runner']['max_workers_per_gpu'] = num_gpus_per_model

    cfg['infer']['partitioner'][
        'num_worker'] = num_gpus_per_infer // num_gpus_per_model
    return cfg


def process_models(cfg: ConfigDict, args: argparse.Namespace) -> ConfigDict:
    model_type = args.model_type
    model_cfg = cfg.get(MODEL_TYPE_MAP[model_type], None)
    assert model_cfg is not None, \
        f'Model config {MODEL_TYPE_MAP[model_type]} is not found ' \
        'in the config file'
    models_path = args.models

    processed_models = []
    model_abbr_type = args.model_abbr_type
    for model_path in models_path:
        if model_path.endswith('/'):
            model_path = model_path[:-1]
        dir_name = osp.basename(model_path)
        if model_abbr_type == 'dir_name':
            summarizer_abbr = dir_name
        elif model_abbr_type == 'ckpt_iter':
            abbr = dir_name.split('-')[0]
            ckpt_iter = dir_name.split('-')[-1]
            summarizer_abbr = f'{abbr}@{ckpt_iter}'
        else:
            raise ValueError(f'Unsupported model abbr type {model_abbr_type}')
        tmp_model_cfg = deepcopy(model_cfg)
        tmp_model_cfg['abbr'] = summarizer_abbr
        # tmp_model_cfg['summarizer_abbr'] = summarizer_abbr
        tmp_model_cfg['path'] = model_path
        processed_models.append(tmp_model_cfg)
    cfg['models'] = processed_models
    return cfg


def process_datasets(cfg: ConfigDict, args: argparse.Namespace) -> ConfigDict:
    parent_dir = os.environ.get('OPENCOMPASS_RUN_DIR', os.getcwd())

    # the dataset path may be in different directories
    datasets_dir = [
        osp.join(parent_dir, 'configs', 'mb_internal', 'datasets'),
        osp.join(parent_dir, 'configs', 'mb_internal', 'dataset_collections'),
        osp.join(parent_dir, 'configs', 'datasets'),
        osp.join(parent_dir, 'configs', 'dataset_collections'),
        osp.join(parent_dir, 'opencompass', 'configs', 'datasets'),
        osp.join(parent_dir, 'opencompass', 'configs', 'dataset_collections'),
    ]
    processed_dataset_list = []

    for dataset_arg in args.datasets:
        dataset_arg_with_py = dataset_arg + '.py' if not dataset_arg.endswith(
            '.py') else dataset_arg
        if osp.exists(dataset_arg_with_py):
            dataset_cfg = Config.fromfile(dataset_arg_with_py)
            for k in dataset_cfg.keys():
                if k.endswith('_datasets'):
                    processed_dataset_list += dataset_cfg[k]
            continue
        elif '/' in dataset_arg:
            dataset_name, dataset_suffix = dataset_arg.split('/', 1)
            dataset_key_suffix = dataset_suffix
        else:
            dataset_name = dataset_arg
            dataset_key_suffix = '_datasets'

        for dataset in match_cfg_file(datasets_dir, [dataset_name]):
            logger.info(f'Loading {dataset[0]}: {dataset[1]}')
            dataset_cfg = Config.fromfile(dataset[1])
            for k in dataset_cfg.keys():
                if k.endswith(dataset_key_suffix):
                    processed_dataset_list += dataset_cfg[k]

    set_datasets = []
    set_datasets_dict = {}
    dataset_abbrs = []
    dataset_check = args.dataset_check
    for dataset in processed_dataset_list:
        curr_dataset_abbr = dataset['abbr']
        if curr_dataset_abbr in dataset_abbrs:
            check_dataset = set_datasets_dict[curr_dataset_abbr]
            if check_dataset == dataset:
                continue
            else:
                if dataset_check:
                    raise ValueError(
                        f'The dataset abbr {curr_dataset_abbr} is '
                        'duplicated, but not equal')
                logger.warning(f'The dataset abbr {curr_dataset_abbr} is '
                               'duplicated, but not equal, ignore it')
        else:
            dataset_abbrs.append(curr_dataset_abbr)
            set_datasets.append(dataset)
            set_datasets_dict[curr_dataset_abbr] = dataset

    cfg['datasets'] = set_datasets
    return cfg


def main():
    args = parse_args()
    models_path = args.models
    datasets = args.datasets
    save_cfg_name = args.save_cfg_name

    assert osp.exists(args.config)

    raw_cfg_root = osp.dirname(args.config)
    raw_cfg_name = osp.basename(args.config)
    if not save_cfg_name.endswith('.py'):
        logger.warning(
            f'Save config name {save_cfg_name} does not end with .py,'
            ' appending .py')
        save_cfg_name += '.py'
    assert raw_cfg_name != save_cfg_name, \
        f'Save config name {save_cfg_name} is the same as the raw config' \
        f' name {raw_cfg_name}'
    save_cfg_name = osp.basename(save_cfg_name)

    assert models_path is not None, 'Models path is required'
    for model_path in models_path:
        assert osp.exists(
            model_path), f'Model path {model_path} does not exist'

    assert datasets is not None, 'Datasets is required'

    cfg = Config.fromfile(args.config, format_python_code=False)
    if 'infer' not in cfg:
        logger.info('infer config not found, initial it')
        cfg['infer'] = initial_infer_cfg()
    if 'eval' not in cfg:
        logger.info('eval config not found, initial it')
        cfg = initial_eval_cfg(cfg)
    if 'summarizer' not in cfg:
        logger.info('summarizer config not found, initial it')
        cfg = initial_summarizer_cfg(cfg)

    # process infer cfg, calculate necessary values
    cfg = process_infer_cfg(cfg, args)

    # process models
    cfg = process_models(cfg, args)

    # process datasets
    cfg = process_datasets(cfg, args)

    save_path = osp.join(raw_cfg_root, save_cfg_name)
    cfg.dump(save_path)

    logger.info('Config:\n\n')
    logger.info(cfg.text)

    logger.info(f'Save config to {save_path}')


if __name__ == '__main__':
    main()
