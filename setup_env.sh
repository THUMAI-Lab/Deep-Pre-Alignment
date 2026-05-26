cd $(cd $(dirname ${BASH_SOURCE});pwd)
conda init
conda env create -f sft_msswift.yml
conda env create -f vlmevalkit.yml
conda env create -f texteval.yml
conda activate sft_msswift
pip install -e ./ms-swift
conda activate texteval
pip install -e ./opencompass