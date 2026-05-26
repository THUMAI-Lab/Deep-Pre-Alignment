#!/usr/bin/env bash
#

SELF_DIR=$(cd $(dirname ${BASH_SOURCE});pwd)

PROJECT_ROOT=${SELF_DIR}/..

conda activate texteval

if [[ ${SKIP_INSTALL} == "false" ]]; then
  pip install -e .
fi

echo "cp eval data to ~/cache_eval_data"
[[ -d ~/cache_eval_data ]] && rm -rf ~/cache_eval_data
mkdir -p ~/cache_eval_data
cp -r /your/cache_eval_data/ ~/cache_eval_data/

# Your cache_eval_data should be in this form:
# ~/cache_eval_data/data/gpqa
# ~/cache_eval_data/data/math
# ~/cache_eval_data/data/mmlu-redux

export ROOT_CACHE_PREFIX=~/cache_eval_data/
export OPENCOMPASS_RUN_DIR="${OPENCOMPASS_RUN_DIR}"

num_gpus=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
python ${SELF_DIR}/run_model_example.py $@ --hf-num-gpus ${num_gpus}
