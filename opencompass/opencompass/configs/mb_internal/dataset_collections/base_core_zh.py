from mmengine.config import read_base

with read_base():
    # cmmlu
    from ...datasets.cmmlu.cmmlu_ppl_041cbf import cmmlu_datasets

    # ceval
    from ...datasets.ceval.ceval_internal_ppl_93e5ce import ceval_datasets

    # commonsenseqa cn
    from ...datasets.commonsenseqa_cn.commonsenseqacn_ppl_971f48 import commonsenseqacn_datasets

    # csl
    from ...datasets.FewCLUE_csl.FewCLUE_csl_ppl_841b62 import csl_datasets

    # nq cn
    from ..datasets.nq_cn.nqcn_gen_len100_141737 import nqcn_datasets

    # ocnli
    from ...datasets.CLUE_ocnli.CLUE_ocnli_gen_c4cb6c import ocnli_datasets

    # cmnli
    from ...datasets.CLUE_cmnli.CLUE_cmnli_gen_1abf97 import cmnli_datasets

    # chid
    from ...datasets.FewCLUE_chid.FewCLUE_chid_ppl_8f2872 import chid_datasets

    # c3
    from ...datasets.CLUE_C3.CLUE_C3_ppl_e24a31 import C3_datasets

HF_INFER_DATASET_NAMES = [
    'mmlu_datasets',
]

# base datasets
base_datasets_total = sum((v for k, v in locals().items() if k.endswith('_datasets') and k not in HF_INFER_DATASET_NAMES), [])
# mmlu may got OOM error when using vllm, so we need to split it and use HF Wrapper
hf_infer_datasets_total = sum((v for k, v in locals().items() if k.endswith('_datasets') and k in HF_INFER_DATASET_NAMES), [])
