from copy import deepcopy

import torch
from tqdm import tqdm

path_list = [
    'zh_seed_exp_v2_rc6_eqpos_eqneg-iter_latest/pytorch_model.bin',
    'zh_seed_exp_v2_rc9_weightpos_eqneg-iter_latest/pytorch_model.bin',
    'zh_seed_exp_v2_rc6_weightpos_eqneg-iter_latest/pytorch_model.bin',
    'zh_seed_exp_v2_rc9_eqpos_eqneg-iter_latest/pytorch_model.bin',
]

save_path = 'zh_seed_exp_v2_avg3-iter_latest/pytorch_model.bin'

tmp_model = torch.load(path_list[0], map_location='cpu')
model_keys = tmp_model.keys()

all_model_dict = {}
for path in path_list:
    print('Loading', path)
    tmp_model = torch.load(path, map_location='cpu')
    all_model_dict[path] = tmp_model

# average the model
avg_model = deepcopy(all_model_dict[path_list[0]])
for key in tqdm(model_keys):
    for path in path_list[1:]:
        avg_model[key] += all_model_dict[path][key]
    avg_model[key] /= len(path_list)

# save the averaged model
print('Saving the averaged model to', save_path)
torch.save(avg_model, save_path)
