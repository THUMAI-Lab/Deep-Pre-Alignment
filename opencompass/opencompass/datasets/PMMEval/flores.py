import json
import multiprocessing as mp
import os
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Tuple

import numpy as np
from datasets import Dataset
from sacrebleu.metrics import BLEU
from sacrebleu.tokenizers.tokenizer_13a import Tokenizer13a
from sacrebleu.tokenizers.tokenizer_zh import TokenizerZh

from opencompass.datasets.base import BaseDataset
from opencompass.openicl.icl_evaluator import BaseEvaluator
from opencompass.registry import LOAD_DATASET, TEXT_POSTPROCESSORS
from opencompass.utils import get_data_path


def wmt_postprocess(text: str, lang: str) -> str:
    text = text.strip()
    texts = list(x.strip() for x in text.split('\n'))
    texts = list(x for x in texts if x != '')
    text = '\n'.join(texts)
    text = tokenize(text, lang)
    return text


def _compute_bleu_for_range(args):
    """并发计算辅助函数：计算特定i值范围内的最大BLEU分数."""
    i_start, i_end, gens_tokens, ref_tokens, max_consecutive_decreases = args

    scorer = BLEU(tokenize='13a', effective_order=True)
    local_maximum_bleu_value = -100.0
    local_maximum_bleu_object = None

    for i in range(i_start, i_end):
        consecutive_decreases = 0  # 连续下降计数器
        last_score = -100.0  # 记录上一次的分数

        for j in range(i, len(gens_tokens)):
            gens_tokens_region = ' '.join(gens_tokens[i:j + 1])
            sentence_bleu = scorer.sentence_score(gens_tokens_region,
                                                  [ref_tokens])

            if sentence_bleu.score > local_maximum_bleu_value:
                local_maximum_bleu_value = sentence_bleu.score
                local_maximum_bleu_object = sentence_bleu
                consecutive_decreases = 0  # 重置计数器
            else:
                # 检查是否连续下降
                if sentence_bleu.score < last_score:
                    consecutive_decreases += 1
                    if consecutive_decreases >= max_consecutive_decreases:
                        # 连续max_consecutive_decreases次下降，启动早期停止
                        break
                else:
                    consecutive_decreases = 0  # 分数没有下降，重置计数器

            last_score = sentence_bleu.score

    return local_maximum_bleu_object


def compute_maximum_bleu_value(gen: str,
                               ref: str,
                               lang: str,
                               use_multiprocessing: bool = True,
                               num_workers: int = None):
    gens = list(x.strip() for x in gen.split('\n'))
    gens = list(x for x in gens if x != '')

    gens_tokens = list(wmt_postprocess(x, lang) for x in gens)
    ref_tokens = wmt_postprocess(ref, lang)

    scorer = BLEU(tokenize='13a', effective_order=True)

    maximum_bleu_value = -100.0
    maximum_bleu_object = None
    """
    internal fix: 原代码采用O(n^2)复杂度, 耗时太长, 需要减少计算量
    1. 截断生成长度, 避免过长的序列
    2. 使用早期停止, 如果连续max_consecutive_decreases次下降, 启动早期停止
    3. 使用多进程并发计算, 充分利用CPU资源
    """
    gens_tokens = gens_tokens[:min(1024, 5 * len(ref_tokens))]

    max_consecutive_decreases = 32

    # 使用多进程并发计算
    if num_workers is None:
        num_workers = min(mp.cpu_count(), 32)  # 限制最大进程数，避免资源竞争

    n = len(gens_tokens)
    chunk_size = max(1, n // num_workers)  # 每个进程处理的i值数量

    # 准备任务参数
    tasks = []
    for i in range(0, n, chunk_size):
        i_end = min(i + chunk_size, n)
        tasks.append(
            (i, i_end, gens_tokens, ref_tokens, max_consecutive_decreases))

    print(f'使用 {num_workers} 个进程并行计算，共 {len(tasks)} 个任务')

    # 执行并发计算
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        future_to_task = {
            executor.submit(_compute_bleu_for_range, task): task
            for task in tasks
        }

        for future in as_completed(future_to_task):
            try:
                local_result = future.result()
                if local_result and local_result.score > maximum_bleu_value:
                    maximum_bleu_value = local_result.score
                    maximum_bleu_object = local_result
            except Exception as exc:
                print(f'任务执行出错: {exc}')

    if maximum_bleu_object is None:
        sentence_bleu = scorer.sentence_score('', [ref_tokens])
        return sentence_bleu
    else:
        return maximum_bleu_object


def trim_multiple_space(tokes):
    return ''.join(tokes).strip().split()


class SpaceTokenizer(object):

    def __call__(self, sent):
        if type(sent) == list:
            print(sent)
            raise ValueError()
        return ' '.join(sent.strip().split())


class NonASCIITokenizer(object):

    def __init__(self):
        self.is_cjk = re.compile('([\u2e80-\u9fff]|'  # 中日韩
                                 '[\ua960-\ua97f]|'  # 谚文字母扩展A
                                 '[\uac00-\ud7ff]|'  # 谚文音节+谚文字母扩展B
                                 '[\u0E00-\u0E7F]'  # 泰文
                                 ')')

    def __call__(self, sent):
        sent = sent.strip()
        chs = list(sent)
        line_chtok = []
        for ch in chs:
            if self.is_cjk.match(ch):
                line_chtok.append(' ')
                line_chtok.append(ch)
                line_chtok.append(' ')
            else:
                line_chtok.append(ch)
        line_chtok = trim_multiple_space(line_chtok)
        return ' '.join(line_chtok)


def build_tokenizer(lang: str):
    if lang == 'Chinese':
        return TokenizerZh()
    elif lang in {'Japanese', 'Korean', 'Thai'}:
        return NonASCIITokenizer()
    else:
        return SpaceTokenizer()


def tokenize(sent, lang):
    tokenizer = build_tokenizer(lang)
    final_tokenizer = Tokenizer13a()
    return final_tokenizer(tokenizer(sent))


@TEXT_POSTPROCESSORS.register_module('pmmeval_flores')
def pmmeval_flores_postprocess(text: str, lang_fullname: str) -> Tuple[str]:
    return text, lang_fullname


@LOAD_DATASET.register_module()
class PMMEvalFloresDataset(BaseDataset):

    @staticmethod
    def load(path: str, lang_fullname: str):
        data_path = get_data_path(path)

        if os.environ.get('DATASET_SOURCE') == 'ModelScope':
            from modelscope import MsDataset
            dataset = MsDataset.load(dataset_name=data_path,
                                     subset_name='flores',
                                     split=f'test/{lang_fullname}')
        else:
            dataset = list()
            filename = os.path.join(data_path,
                                    f'flores/test/{lang_fullname}.jsonl')
            with open(filename, mode='r', encoding='utf-8') as f:
                for line in f:
                    line = json.loads(line.strip())
                    dataset.append(line)
            dataset = Dataset.from_list(dataset)

        return dataset


class PMMEvalFloresEvaluator(BaseEvaluator):

    def score(self, predictions, references):
        maximum_bleu_results = list()
        for (pred, tgt_lang), ref in zip(predictions, references):
            maximum_bleu_results.append(
                compute_maximum_bleu_value(pred, ref, tgt_lang))

        maximum_corpus_bleu_counts = sum(
            np.array(x.counts) for x in maximum_bleu_results).tolist()
        maximum_corpus_bleu_totals = sum(
            np.array(x.totals) for x in maximum_bleu_results).tolist()
        maximum_corpus_bleu_sys_len = sum(x.sys_len
                                          for x in maximum_bleu_results)
        maximum_corpus_bleu_ref_len = sum(x.ref_len
                                          for x in maximum_bleu_results)
        print(f'maximum_corpus_bleu_sys_len, {maximum_corpus_bleu_sys_len}')
        print(f'maximum_corpus_bleu_ref_len, {maximum_corpus_bleu_ref_len}')
        maximum_bleu_result = BLEU.compute_bleu(
            correct=maximum_corpus_bleu_counts,
            total=maximum_corpus_bleu_totals,
            sys_len=maximum_corpus_bleu_sys_len,
            ref_len=maximum_corpus_bleu_ref_len)

        result = {'BLEU': round(maximum_bleu_result.score, 2)}
        return result
