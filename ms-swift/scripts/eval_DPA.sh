cd $(cd $(dirname ${BASH_SOURCE});pwd)
cd ../../Multimodal_eval
conda init
conda activate vlmevalkit

poll_interval=0
benchmarks=$1
# SEEDBench2_Plus MMVet MMStar MMMU_DEV_VAL MathVista_MINI MathVision OCRBench AI2D_TEST
model_path=$2
eval_step=$3
start_step=$4
poll_interval=${5:-0}

echo "Evaluating model $model_path on benchmarks $benchmarks"
echo "Poll interval: $poll_interval"
echo "OPENAI_API_KEY: $OPENAI_API_KEY"

echo python path: $(which python)
echo python version: $(python --version)

python auto_evaluator.py --model_path $model_path --eval_step $eval_step --start_step $start_step --model_name DuplexThinkerS2ForwardvLLMPrefixCustomLLaVA --benchmarks $benchmarks --poll_interval $poll_interval --gpus 0,1,2,3 --hidden_size 2560

echo "Evaluation completed"

