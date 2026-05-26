import argparse
import os


def parse_args():
    parser = argparse.ArgumentParser(description='Process run config')
    parser.add_argument('config', type=str, help='Path to the config file')
    parser.add_argument('--models',
                        nargs='+',
                        help='Path to the model(s)',
                        default=None)
    parser.add_argument('--num-gpus',
                        type=int,
                        default=8,
                        help='Number of GPUs')
    parser.add_argument('--save-cfg-name',
                        type=str,
                        default='current_config.py',
                        help='Name of the saved config file')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    cfg_path = args.config
    models_path = args.models
    num_gpus = args.num_gpus
    save_cfg_name = args.save_cfg_name

    save_cfg_name = os.path.basename(save_cfg_name)

    assert os.path.exists(cfg_path), \
        f'Config file {cfg_path} does not exist'

    raw_cfg_root = os.path.dirname(cfg_path)
    raw_cfg_name = os.path.basename(cfg_path)

    if not save_cfg_name.endswith('.py'):
        print(f'Save config name {save_cfg_name} does not end with .py,'
              ' appending .py')
        save_cfg_name += '.py'
    assert raw_cfg_name != save_cfg_name, \
        f'Save config name {save_cfg_name} is the same as the raw config' \
        f' name {raw_cfg_name}'
    save_cfg_name = os.path.basename(save_cfg_name)

    with open(cfg_path, 'r') as f:
        raw_cfg = f.read()
    assert '{HF_MODEL_PATH}' in raw_cfg, \
        'HF_MODEL_PATH not found in config file'
    assert '{NUM_GPUS}' in raw_cfg, \
        'NUM_GPUS not found in config file'

    models_path_str = ''
    for model_path in models_path:
        assert os.path.exists(model_path), \
            f'Model file {model_path} does not exist'
        models_path_str += f'    "{model_path}",\n'
    models_path_str = models_path_str.rstrip()
    new_cfg = raw_cfg.replace('{HF_MODEL_PATH}', models_path_str)
    new_cfg = new_cfg.replace('{NUM_GPUS}', str(num_gpus))
    save_path = os.path.join(raw_cfg_root, save_cfg_name)
    with open(save_path, 'w') as f:
        f.write(new_cfg)

    print('Run Config:\n\n')
    print(new_cfg)


if __name__ == '__main__':
    main()
