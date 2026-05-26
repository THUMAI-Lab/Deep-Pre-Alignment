from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever, TopkRetriever, FixKRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import BleuEvaluator, BleuFloresEvaluator
from opencompass.utils.text_postprocessors import general_cn_postprocess
from opencompass.datasets import WmtDataset

wmt_name_list = [
    # "wmt20_en_zh",
    # "wmt20_zh_en"
    'wmt20_en_zh_raw',
    'wmt20_zh_en_raw',
]

wmt_datasets = []

for _name in wmt_name_list:
    wmt_infer_cfg = dict(
        ice_template=dict(
            type=PromptTemplate,
            template='</E> 请把下面的句子由英文翻译成中文,只回答你翻译的句子:\n{input}\n译文:\n {golden}' if 'en_zh' in _name
            else '</E> Please translate the following sentence from Chineses to English:\n{input} \nTranslation:\n{golden}',
            # template='</E>{input}',
            # template=dict(
            # begin='</E>',
            #     round=[
            #         dict(
            #             role='HUMAN',
            #             prompt=
            #             '{input}'
            #         ),
            #         dict(role='BOT', prompt='{golden}'),
            #     ]),
            ice_token='</E>',
        ),
        retriever=dict(type=ZeroRetriever),
        # retriever=dict(type=FixKRetriever, fix_id_list=[0]),
        # retriever=dict(type=TopkRetriever, ice_num=1),
        inferencer=dict(type=GenInferencer, max_out_len=512),
    )

    # wmt_eval_cfg = dict(
    #     evaluator=dict(type=BleuEvaluator),
    #     pred_role='BOT',
    #     )

    # wmt_eval_cfg = dict(
    #     evaluator=dict(type=BleuFloresEvaluator),
    #     pred_role='BOT',
    #     pred_postprocessor=dict(type='flores-chinese'),
    #     dataset_postprocessor=dict(type='flores-chinese'),
    # )
    
    wmt_eval_cfg = dict(
        evaluator=dict(type=BleuEvaluator),
        pred_role='BOT',
        pred_postprocessor=dict(type=general_cn_postprocess),
        dataset_postprocessor=dict(type=general_cn_postprocess)
    )

    # if 'en_zh' in _name:
    #     wmt_eval_cfg['pred_postprocessor'] = dict(type='flores')
    #     wmt_eval_cfg['dataset_postprocessor'] = dict(type='flores')

    # if _tgt == 'zho_simpl':
    # if "_en_zh" in _name:
    # wmt_eval_cfg['pred_postprocessor'] = dict(type='flores-chinese')
    # wmt_eval_cfg['dataset_postprocessor'] = dict(type='flores-chinese')

    wmt_datasets.append(
        dict(
            type=WmtDataset,
            path='./data/wmt_mb',
            name=_name,
            abbr=_name,
            reader_cfg=dict(
                input_columns=['input'],
                output_column='golden',
                train_split='dev',
                test_split='test'),
            infer_cfg=wmt_infer_cfg,
            eval_cfg=wmt_eval_cfg,
        ))

del _name
