from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import TopkRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import BleuEvaluator
from opencompass.datasets import FloresFirst100Dataset

"""
​​英语​​ - eng_Latn
​​中文（简体）​​ - zho_Hans
​​中文（繁体）​​ - zho_Hant
​​印地语​​ - hin_Deva
​​西班牙语​​ - spa_Latn
​​法语​​ - fra_Latn
​​阿拉伯语（现代标准）​​ - arb_Arab
​​俄语​​ - rus_Cyrl
​​葡萄牙语​​ - por_Latn
​​德语​​ - deu_Latn
​​日语​​ - jpn_Jpan
​​韩语​​ - kor_Hang
​​土耳其语​​ - tur_Latn
​​意大利语​​ - ita_Latn
​​荷兰语​​ - nld_Latn
​​波兰语​​ - pol_Latn
​​越南语​​ - vie_Latn
​​泰语​​ - tha_Thai
​​波斯语（西方）​​ - pes_Arab
​​乌尔都语​​ - urd_Arab
​​泰米尔语​​ - tam_Taml
​​泰卢固语​​ - tel_Telu
​​马拉地语​​ - mar_Deva
​​孟加拉语​​ - ben_Beng
​​旁遮普语（东）​​ - pan_Guru
​​古吉拉特语​​ - guj_Gujr
​​印尼语​​ - ind_Latn
​​菲律宾语（他加禄语）​​ - tgl_Latn
​​斯瓦希里语​​ - swh_Latn
​​豪萨语​​ - hau_Latn
​​约鲁巴语​​ - yor_Latn
​​乌克兰语​​ - ukr_Cyrl
​​罗马尼亚语​​ - ron_Latn
​​希腊语​​ - ell_Grek
​​瑞典语​​ - swe_Latn
​​捷克语​​ - ces_Latn
​​匈牙利语​​ - hun_Latn
​​保加利亚语​​ - bul_Cyrl
​​塞尔维亚语​​ - srp_Cyrl
​​克罗地亚语​​ - hrv_Latn
​​丹麦语​​ - dan_Latn
​​挪威语（书面）​​ - nob_Latn
​​芬兰语​​ - fin_Latn
​​希伯来语​​ - heb_Hebr
​​马来语（标准）​​ - zsm_Latn
​​缅甸语​​ - mya_Mymr
​​尼泊尔语​​ - npi_Deva
​​僧伽罗语​​ - sin_Sinh
​​高棉语​​ - khm_Khmr
​​阿姆哈拉语​​ - amh_Ethi"""

_flores_lang_map = [
    ['eng', 'eng_Latn', 'English', 'Indo-European-Germanic'],
    # ['afr', 'afr_Latn', 'Afrikaans', 'Indo-European-Germanic'],
    ['dan', 'dan_Latn', 'Danish', 'Indo-European-Germanic'],
    ['deu', 'deu_Latn', 'German', 'Indo-European-Germanic'],
    # ['isl', 'isl_Latn', 'Icelandic', 'Indo-European-Germanic'],
    # ['ltz', 'ltz_Latn', 'Luxembourgish', 'Indo-European-Germanic'],
    ['nld', 'nld_Latn', 'Dutch', 'Indo-European-Germanic'],
    ['nob', 'nob_Latn', 'Norwegian', 'Indo-European-Germanic'],
    ['swe', 'swe_Latn', 'Swedish', 'Indo-European-Germanic'],
    # ['ast', 'ast_Latn', 'Asturian', 'Indo-European-Romance'],
    # ['cat', 'cat_Latn', 'Catalan', 'Indo-European-Romance'],
    ['fra', 'fra_Latn', 'French', 'Indo-European-Romance'],
    # ['glg', 'glg_Latn', 'Galician', 'Indo-European-Romance'],
    # ['oci', 'oci_Latn', 'Occitan', 'Indo-European-Romance'],
    ['por', 'por_Latn', 'Portuguese', 'Indo-European-Romance'],
    ['ron', 'ron_Latn', 'Romanian', 'Indo-European-Romance'],
    ['spa', 'spa_Latn', 'Spanish', 'Indo-European-Romance'],
    # ['bel', 'bel_Cyrl', 'Belarusian', 'Indo-European-Slavic'],
    # ['bos', 'bos_Latn', 'Bosnian', 'Indo-European-Slavic'],
    ['bul', 'bul_Cyrl', 'Bulgarian', 'Indo-European-Slavic'],
    ['ces', 'ces_Latn', 'Czech', 'Indo-European-Slavic'],
    ['hrv', 'hrv_Latn', 'Croatian', 'Indo-European-Slavic'],
    # ['mkd', 'mkd_Cyrl', 'Macedonian', 'Indo-European-Slavic'],
    ['pol', 'pol_Latn', 'Polish', 'Indo-European-Slavic'],
    ['rus', 'rus_Cyrl', 'Russian', 'Indo-European-Slavic'],
    # ['slk', 'slk_Latn', 'Slovak', 'Indo-European-Slavic'],
    # ['slv', 'slv_Latn', 'Slovenian', 'Indo-European-Slavic'],
    ['srp', 'srp_Cyrl', 'Serbian', 'Indo-European-Slavic'],
    ['ukr', 'ukr_Cyrl', 'Ukrainian', 'Indo-European-Slavic'],
    # ['asm', 'asm_Beng', 'Assamese', 'Indo-European-Indo-Aryan'],
    ['ben', 'ben_Beng', 'Bengali', 'Indo-European-Indo-Aryan'],
    ['guj', 'guj_Gujr', 'Gujarati', 'Indo-European-Indo-Aryan'],
    ['hin', 'hin_Deva', 'Hindi', 'Indo-European-Indo-Aryan'],
    ['mar', 'mar_Deva', 'Marathi', 'Indo-European-Indo-Aryan'],
    ['npi', 'npi_Deva', 'Nepali', 'Indo-European-Indo-Aryan'],
    # ['ory', 'ory_Orya', 'Oriya', 'Indo-European-Indo-Aryan'],
    ['pan', 'pan_Guru', 'Punjabi', 'Indo-European-Indo-Aryan'],
    # ['snd', 'snd_Arab', 'Sindhi', 'Indo-European-Indo-Aryan'],
    ['urd', 'urd_Arab', 'Urdu', 'Indo-European-Indo-Aryan'],
    # ['ckb', 'ckb_Arab', 'Kurdish', 'Indo-European-Other'],
    # ['cym', 'cym_Latn', 'Welsh', 'Indo-European-Other'],
    ['ell', 'ell_Grek', 'Greek', 'Indo-European-Other'],
    ['fas', 'pes_Arab', 'Persian', 'Indo-European-Other'],
    # ['gle', 'gle_Latn', 'Irish', 'Indo-European-Other'],
    # ['hye', 'hye_Armn', 'Armenian', 'Indo-European-Other'],
    ['ita', 'ita_Latn', 'Italian', 'Indo-European-Other'],
    # ['lav', 'lvs_Latn', 'Latvian', 'Indo-European-Other'],
    # ['lit', 'lit_Latn', 'Lithuanian', 'Indo-European-Other'],
    # ['pus', 'pbt_Arab', 'Pashto', 'Indo-European-Other'],
    # ['tgk', 'tgk_Cyrl', 'Tajik', 'Indo-European-Other'],
    # ['ceb', 'ceb_Latn', 'Cebuano', 'Austronesian'],
    ['ind', 'ind_Latn', 'Indonesian', 'Austronesian'],
    # ['jav', 'jav_Latn', 'Javanese', 'Austronesian'],
    # ['mri', 'mri_Latn', 'Maori', 'Austronesian'],
    ['msa', 'zsm_Latn', 'Malay', 'Austronesian'],
    ['tgl', 'tgl_Latn', 'Tagalog', 'Austronesian'],
    # ['ibo', 'ibo_Latn', 'Igbo', 'Atlantic-Congo'],
    # ['kam', 'kam_Latn', 'Kamba', 'Atlantic-Congo'],
    # ['kea', 'kea_Latn', 'Kabuverdianu', 'Atlantic-Congo'],
    # ['lin', 'lin_Latn', 'Lingala', 'Atlantic-Congo'],
    # ['lug', 'lug_Latn', 'Luganda', 'Atlantic-Congo'],
    # ['nso', 'nso_Latn', 'Northern Sotho', 'Atlantic-Congo'],
    # ['nya', 'nya_Latn', 'Nyanja', 'Atlantic-Congo'],
    # ['sna', 'sna_Latn', 'Shona', 'Atlantic-Congo'],
    ['swh', 'swh_Latn', 'Swahili', 'Atlantic-Congo'],
    # ['umb', 'umb_Latn', 'Umbundu', 'Atlantic-Congo'],
    # ['wol', 'wol_Latn', 'Wolof', 'Atlantic-Congo'],
    # ['xho', 'xho_Latn', 'Xhosa', 'Atlantic-Congo'],
    ['yor', 'yor_Latn', 'Yoruba', 'Atlantic-Congo'],
    # ['zul', 'zul_Latn', 'Zulu', 'Atlantic-Congo'],
    ['amh', 'amh_Ethi', 'Amharic', 'Afro-Asiatic'],
    ['ara', 'arb_Arab', 'Arabic', 'Afro-Asiatic'],
    # ['ful', 'fuv_Latn', 'Fulah', 'Afro-Asiatic'],
    # ['mlt', 'mlt_Latn', 'Maltese', 'Afro-Asiatic'],
    # ['orm', 'gaz_Latn', 'Oromo', 'Afro-Asiatic'],
    # ['som', 'som_Latn', 'Somali', 'Afro-Asiatic'],
    # ['azj', 'azj_Latn', 'Azerbaijani', 'Turkic'],
    # ['kaz', 'kaz_Cyrl', 'Kazakh', 'Turkic'],
    # ['kir', 'kir_Cyrl', 'Kyrgyz', 'Turkic'],
    ['tur', 'tur_Latn', 'Turkish', 'Turkic'],
    # ['uzb', 'uzn_Latn', 'Uzbek', 'Turkic'],
    # ['kan', 'kan_Knda', 'Kannada', 'Dravidian'],
    # ['mal', 'mal_Mlym', 'Malayalam', 'Dravidian'],
    ['tam', 'tam_Taml', 'Tamil', 'Dravidian'],
    ['tel', 'tel_Telu', 'Telugu', 'Dravidian'],
    ['mya', 'mya_Mymr', 'Burmese', 'Sino-Tibetan'],
    ['zho_simpl', 'zho_Hans', 'Chinese (Simpl)', 'Sino-Tibetan'],
    ['zho_trad', 'zho_Hant', 'Chinese (Trad)', 'Sino-Tibetan'],
    # ['est', 'est_Latn', 'Estonian', 'Other'],
    ['fin', 'fin_Latn', 'Finnish', 'Other'],
    ['hau', 'hau_Latn', 'Hausa', 'Other'],
    ['heb', 'heb_Hebr', 'Hebrew', 'Other'],
    ['hun', 'hun_Latn', 'Hungarian', 'Other'],
    ['jpn', 'jpn_Jpan', 'Japanese', 'Other'],
    # ['kat', 'kat_Geor', 'Georgian', 'Other'],
    ['khm', 'khm_Khmr', 'Khmer', 'Other'],
    ['kor', 'kor_Hang', 'Korean', 'Other'],
    # ['lao', 'lao_Laoo', 'Lao', 'Other'],
    # ['luo', 'luo_Latn', 'Luo', 'Other'],
    # ['mon', 'khk_Cyrl', 'Mongolian', 'Other'],
    ['tha', 'tha_Thai', 'Thai', 'Other'],
    ['vie', 'vie_Latn', 'Vietnamese', 'Other'],
    # ['sin', 'sin_Sinh', 'Sinhala', 'Other'],
]
flores_lang_map = {i[0]: i for i in _flores_lang_map}
_flores_subtasks = [f'eng-{i}' for i in flores_lang_map if i != 'eng'
                    ] + [f'{i}-eng' for i in flores_lang_map if i != 'eng']

flores_datasets = []
for _flores_subtask in _flores_subtasks:
    _src, _tgt = _flores_subtask.split('-')
    _, _flores_source, _src_inst, _ = flores_lang_map[_src]
    _, _flores_target, _tgt_inst, _ = flores_lang_map[_tgt]

    flores_reader_cfg = dict(
        input_columns=f'sentence_{_flores_source}',
        output_column=f'sentence_{_flores_target}',
        train_split='dev',
        test_split='devtest'
    )
    flores_infer_cfg = dict(
        ice_template=dict(
            type=PromptTemplate,
            template=dict(
                begin='</E>',
                round=[
                    dict(
                        role='HUMAN',
                        prompt=f'Translate the following {_src_inst} statements to {_tgt_inst}.\n{{sentence_{_flores_source}}}'
                    ),
                    dict(role='BOT', prompt=f'{{sentence_{_flores_target}}}'),
                ],
            ),
            ice_token='</E>',
        ),
        retriever=dict(type=TopkRetriever, ice_num=8),
        inferencer=dict(type=GenInferencer),
    )
    flores_eval_cfg = dict(
        evaluator=dict(type=BleuEvaluator),
        pred_role='BOT',
    )
    if _tgt == 'zho_simpl':
        flores_eval_cfg['pred_postprocessor'] = dict(type='flores')
        flores_eval_cfg['dataset_postprocessor'] = dict(type='flores')
    flores_datasets.append(
        dict(
            abbr=f'flores_100_{_src}-{_tgt}',
            type=FloresFirst100Dataset,
            path='opencompass/flores',
            name=f'{_flores_source}-{_flores_target}',
            reader_cfg=flores_reader_cfg.copy(),
            infer_cfg=flores_infer_cfg.copy(),
            eval_cfg=flores_eval_cfg.copy(),
        ))
