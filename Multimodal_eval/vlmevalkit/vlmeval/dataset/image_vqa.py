from functools import partial

from .image_base import ImageBaseDataset
from .utils import build_judge, DEBUG_MESSAGE
from ..smp import *
from ..utils import track_progress_rich


class ImageVQADataset(ImageBaseDataset):
    TYPE = 'VQA'

    DATASET_URL = {
        'OCRVQA_TEST': 'https://opencompass.openxlab.space/utils/VLMEval/OCRVQA_TEST.tsv',
        'OCRVQA_TESTCORE': 'https://opencompass.openxlab.space/utils/VLMEval/OCRVQA_TESTCORE.tsv',
        'TextVQA_VAL': 'https://opencompass.openxlab.space/utils/VLMEval/TextVQA_VAL.tsv',
        'DocVQA_VAL': 'https://opencompass.openxlab.space/utils/VLMEval/DocVQA_VAL.tsv',
        'DocVQA_TEST': 'https://opencompass.openxlab.space/utils/VLMEval/DocVQA_TEST.tsv',
        'InfoVQA_VAL': 'https://opencompass.openxlab.space/utils/VLMEval/InfoVQA_VAL.tsv',
        'InfoVQA_TEST': 'https://opencompass.openxlab.space/utils/VLMEval/InfoVQA_TEST.tsv',
        'ChartQA_TEST': 'https://opencompass.openxlab.space/utils/VLMEval/ChartQA_TEST.tsv',
        'GQA_TestDev_Balanced': 'https://opencompass.openxlab.space/utils/VLMEval/GQA_TestDev_Balanced.tsv',
    }

    DATASET_MD5 = {
        'OCRVQA_TEST': 'ca46a6d74b403e9d6c0b670f6fc00db9',
        'OCRVQA_TESTCORE': 'c5239fe77db8bdc1f2ad8e55e0d1fe97',
        'TextVQA_VAL': 'b233b31f551bbf4056f2f955da3a92cd',
        'DocVQA_VAL': 'd5ee77e1926ff10690d469c56b73eabf',
        'DocVQA_TEST': '6a2f28cac26ef2d3447374e8c6f6c8e9',
        'InfoVQA_VAL': '2342e9c225222f0ef4dec545ebb126fe',
        'InfoVQA_TEST': 'df535bf51b88dc9718252c34131a6227',
        'ChartQA_TEST': 'c902e0aa9be5582a7aad6dcf52734b42',
        'GQA_TestDev_Balanced': 'fead7df22befc1ed3ca2b62ea26fa17b',
    }

    def build_prompt(self, line):
        msgs = super().build_prompt(line)
        assert msgs[-1]['type'] == 'text'
        msgs[-1]['value'] += '\nAnswer the question using a single word or phrase.'
        return msgs

    # It returns a DataFrame
    def evaluate(self, eval_file, **judge_kwargs):
        from .utils.vqa_eval import hit_calculate, process_line

        suffix = eval_file.split('.')[-1]
        result_file = eval_file.replace(f'.{suffix}', '_acc.csv')
        if osp.exists(result_file):
            return load(result_file)

        data = load(eval_file)
        dataset = self.dataset_name
        assert 'answer' in data and 'prediction' in data
        data['prediction'] = [str(x) for x in data['prediction']]
        data['answer'] = [str(x) for x in data['answer']]
        lt = len(data)
        pool = mp.Pool(16)
        lines = [data.iloc[i] for i in range(lt)]
        if listinstr(['TextVQA'], dataset):
            res = pool.map(partial(process_line, method='vqa_score'), lines)
        elif listinstr(['ChartQA'], dataset):
            res = pool.map(partial(process_line, method='relaxed_accuracy'), lines)
        elif listinstr(['OCRVQA', 'GQA'], dataset):
            res = pool.map(partial(process_line, method='accuracy'), lines)
        elif listinstr(['DocVQA', 'InfoVQA'], dataset):
            res = pool.map(partial(process_line, method='anls'), lines)
        else:  # default using vqa_score to calculate score
            res = pool.map(process_line, lines)
        hit = hit_calculate(res, dataset)
        ret = dict()
        if 'split' in data:
            splits = set(data['split'])
            for sp in splits:
                sub = [r for l, r in zip(lines, res) if l['split'] == sp]
                # [np.mean(x['match']) >= full_score_weight for x in sub]
                hit = hit_calculate(sub, dataset)
                ret[sp] = np.mean(hit) * 100
            sub = [r for l, r in zip(lines, res)]
            hit = hit_calculate(sub, dataset)
            ret['Overall'] = np.mean(hit) * 100
        else:
            ret['Overall'] = np.mean(hit) * 100
            if 'category' in data:
                cates = list(set(data['category']))
                cates.sort()
                for c in cates:
                    sub = [r for l, r in zip(lines, res) if l['category'] == c]
                    # [np.mean(x['match']) >= full_score_weight for x in sub]
                    hit = hit_calculate(sub, dataset)
                    ret[c] = np.mean(hit) * 100
        ret = d2df(ret)
        ret.round(2)

        suffix = eval_file.split('.')[-1]
        result_file = eval_file.replace(f'.{suffix}', '_acc.csv')
        dump(ret, result_file)
        return ret


class OCRBench(ImageBaseDataset):
    TYPE = 'VQA'
    DATASET_URL = {
        'OCRBench': 'https://opencompass.openxlab.space/utils/VLMEval/OCRBench.tsv'
    }
    DATASET_MD5 = {'OCRBench': 'e953d98a987cc6e26ef717b61260b778'}

    # It returns a dictionary
    @classmethod
    def evaluate(self, eval_file, **judge_kwargs):
        score_pth = eval_file.replace('.xlsx', '_score.json')
        if osp.exists(score_pth):
            return load(score_pth)

        OCRBench_score = {
            'Regular Text Recognition': 0,
            'Irregular Text Recognition': 0,
            'Artistic Text Recognition': 0,
            'Handwriting Recognition': 0,
            'Digit String Recognition': 0,
            'Non-Semantic Text Recognition': 0,
            'Scene Text-centric VQA': 0,
            'Doc-oriented VQA': 0,
            'Key Information Extraction': 0,
            'Handwritten Mathematical Expression Recognition': 0,
        }

        data = load(eval_file)
        lt = len(data)
        lines = [data.iloc[i] for i in range(lt)]
        for i in tqdm(range(len(lines))):
            line = lines[i]
            predict = str(line['prediction'])
            answers = eval(line['answer'])
            category = line['category']
            if category == 'Handwritten Mathematical Expression Recognition':
                for j in range(len(answers)):
                    answer = answers[j].strip().replace('\n', ' ').replace(' ', '')
                    predict = predict.strip().replace('\n', ' ').replace(' ', '')
                    if answer in predict:
                        OCRBench_score[category] += 1
                        break
            else:
                for j in range(len(answers)):
                    answer = answers[j].lower().strip().replace('\n', ' ')
                    predict = predict.lower().strip().replace('\n', ' ')
                    if answer in predict:
                        OCRBench_score[category] += 1
                        break

        final_score_dict = {}
        final_score_dict['Text Recognition'] = \
            (OCRBench_score['Regular Text Recognition'] + OCRBench_score['Irregular Text Recognition']
             + OCRBench_score['Artistic Text Recognition'] + OCRBench_score['Handwriting Recognition']
             + OCRBench_score['Digit String Recognition'] + OCRBench_score['Non-Semantic Text Recognition'])
        final_score_dict['Scene Text-centric VQA'] = OCRBench_score['Scene Text-centric VQA']
        final_score_dict['Doc-oriented VQA'] = OCRBench_score['Doc-oriented VQA']
        final_score_dict['Key Information Extraction'] = OCRBench_score['Key Information Extraction']
        final_score_dict['Handwritten Mathematical Expression Recognition'] = \
            (OCRBench_score['Handwritten Mathematical Expression Recognition'])
        final_score_dict['Final Score'] = \
            (final_score_dict['Text Recognition'] + final_score_dict['Scene Text-centric VQA']
             + final_score_dict['Doc-oriented VQA'] + final_score_dict['Key Information Extraction']
             + final_score_dict['Handwritten Mathematical Expression Recognition'])
        final_score_dict['Final Score Norm'] = (float(final_score_dict['Final Score']) / 10)
        score_pth = eval_file.replace('.xlsx', '_score.json')
        dump(final_score_dict, score_pth)
        return final_score_dict


class MathVista(ImageBaseDataset):
    TYPE = 'VQA'
    DATASET_URL = {
        'MathVista_MINI': 'https://opencompass.openxlab.space/utils/VLMEval/MathVista_MINI.tsv'
    }
    DATASET_MD5 = {'MathVista_MINI': 'f199b98e178e5a2a20e7048f5dcb0464'}

    # It returns a DataFrame
    @classmethod
    def evaluate(self, eval_file, **judge_kwargs):
        from .utils.mathvista import MathVista_auxeval, MathVista_acc

        model = judge_kwargs['model']
        suffix = eval_file.split('.')[-1]
        storage = eval_file.replace(f'.{suffix}', f'_{model}.xlsx')
        score_pth = storage.replace('.xlsx', '_score.csv')
        if osp.exists(score_pth):
            return load(score_pth)

        tmp_file = eval_file.replace(f'.{suffix}', f'_{model}.pkl')
        nproc = judge_kwargs.pop('nproc', 4)

        if not osp.exists(storage):
            data = load(eval_file)
            model = build_judge(max_tokens=128, **judge_kwargs)
            assert model.working(), ('MathVista evaluation requires a working OPENAI API\n' + DEBUG_MESSAGE)
            lt = len(data)
            lines = [data.iloc[i] for i in range(lt)]
            tups = [(model, line) for line in lines]
            indices = [line['index'] for line in lines]

            ans = {}
            if osp.exists(tmp_file):
                ans = load(tmp_file)
            tups = [x for x, i in zip(tups, indices) if i not in ans]
            indices = [i for i in indices if i not in ans]

            if len(indices):
                new_results = track_progress_rich(
                    MathVista_auxeval,
                    tups,
                    nproc=nproc,
                    chunksize=nproc,
                    keys=indices,
                    save=tmp_file,
                )
                ans = load(tmp_file)
                for k, v in zip(indices, new_results):
                    assert k in ans
                    assert ans[k]['log'] == v['log'] and ans[k]['res'] == v['res']

            data['res'] = [ans[idx]['res'] for idx in data['index']]
            data['log'] = [ans[idx]['log'] for idx in data['index']]
            dump(data, storage)

        score = MathVista_acc(storage)
        score_pth = storage.replace('.xlsx', '_score.csv')
        dump(score, score_pth)
        return score


class MathVerse(ImageBaseDataset):
    TYPE = 'VQA'
    DATASET_URL = {
        'MathVerse_MINI':
        'http://opencompass.openxlab.space/utils/benchmarks/MathVerse/MathVerse_MINIV.tsv',  # noqa
        'MathVerse_MINI_Vision_Only':
        'http://opencompass.openxlab.space/utils/benchmarks/MathVerse/MathVerse_MINIVOnly.tsv',  # noqa
        'MathVerse_MINI_Vision_Only_cot':
        'http://opencompass.openxlab.space/utils/benchmarks/MathVerse/MathVerse_MINIVOnly.tsv',  # noqa
        'MathVerse_MINI_Vision_Dominant':
        'http://opencompass.openxlab.space/utils/benchmarks/MathVerse/MathVerse_MINIVDom.tsv',  # noqa
        'MathVerse_MINI_Vision_Intensive':
        'http://opencompass.openxlab.space/utils/benchmarks/MathVerse/MathVerse_MINIVInt.tsv',  # noqa
        'MathVerse_MINI_Text_Lite':
        'http://opencompass.openxlab.space/utils/benchmarks/MathVerse/MathVerse_MINITLite.tsv',  # noqa
        'MathVerse_MINI_Text_Dominant':
        'http://opencompass.openxlab.space/utils/benchmarks/MathVerse/MathVerse_MINITDom.tsv',  # noqa
    }
    DATASET_MD5 = {
        'MathVerse_MINI': '5017caca32b7fa110c350a1bea861b65',
        'MathVerse_MINI_Vision_Only': '68a11d4680014ac881fa37adeadea3a4',
        'MathVerse_MINI_Vision_Only_cot': '68a11d4680014ac881fa37adeadea3a4',
        'MathVerse_MINI_Vision_Dominant': 'b8fb63852d261ab2aaefba29cc2414d3',
        'MathVerse_MINI_Vision_Intensive': '01cbd35be202bb0c4873a4186a63bc19',
        'MathVerse_MINI_Text_Lite': '19e4b13bdd30b89a03b2e358bcfefa04',
        'MathVerse_MINI_Text_Dominant': '4f5cd2fa6630ea00bb11d6fde1f6fe6a',
    }

    # Given one data record, return the built prompt (a multi-modal message), can override
    def build_prompt(self, line):
        if isinstance(line, int):
            line = self.data.iloc[line]

        if self.meta_only:
            tgt_path = toliststr(line['image_path'])
        else:
            tgt_path = self.dump_image(line)
        if 'cot' in self.dataset_name:
            question = line['query_cot']
        else:
            question = line['question']

        msgs = []
        if isinstance(tgt_path, list):
            msgs.extend([dict(type='image', value=p) for p in tgt_path])
        else:
            msgs = [dict(type='image', value=tgt_path)]
        msgs.append(dict(type='text', value=question))
        return msgs

    # It returns a DataFrame
    @classmethod
    def evaluate(self, eval_file, **judge_kwargs):
        from .utils.mathverse import MathVerse_auxeval_extract, MathVerse_auxeval_score, MathVerse_acc

        model = judge_kwargs['model']
        storage_extract = get_intermediate_file_path(eval_file, f'_{model}_extract')
        tmp_file_extract = get_intermediate_file_path(eval_file, f'_{model}_extract', 'pkl')
        storage_score = get_intermediate_file_path(eval_file, f'_{model}_score')
        tmp_file_score = get_intermediate_file_path(eval_file, f'_{model}_score', 'pkl')
        nproc = judge_kwargs.pop('nproc', 4)
        # stage1: extract the answer
        if not osp.exists(storage_extract):
            data = load(eval_file)
            model = build_judge(max_tokens=128, **judge_kwargs)
            assert model.working(), 'MathVerse evaluation requires a working OPENAI API\n' + DEBUG_MESSAGE
            lt = len(data)
            lines = [data.iloc[i] for i in range(lt)]
            tups = [(model, line) for line in lines]
            indices = [line['index'] for line in lines]

            ans = {}
            if osp.exists(tmp_file_extract):
                ans = load(tmp_file_extract)
            tups = [x for x, i in zip(tups, indices) if i not in ans]
            indices = [i for i in indices if i not in ans]

            if len(indices):
                new_results = track_progress_rich(
                    MathVerse_auxeval_extract,
                    tups,
                    nproc=nproc,
                    chunksize=nproc,
                    keys=indices,
                    save=tmp_file_extract,
                )
                ans = load(tmp_file_extract)
                for k, v in zip(indices, new_results):
                    assert k in ans
                    assert ans[k]['log_extract'] == v['log_extract'] and ans[k][
                        'extract'] == v['extract']

            data['extract'] = [ans[idx]['extract'] for idx in data['index']]
            data['log_extract'] = [
                ans[idx]['log_extract'] for idx in data['index']
            ]
            dump(data, storage_extract)

        # stage2: score the answer
        if not osp.exists(storage_score):
            data = load(storage_extract)
            model = build_judge(max_tokens=128, **judge_kwargs)
            assert model.working(), 'MathVerse evaluation requires a working OPENAI API\n' + DEBUG_MESSAGE
            lt = len(data)
            lines = [data.iloc[i] for i in range(lt)]
            tups = [(model, line) for line in lines]
            indices = [line['index'] for line in lines]

            ans = {}
            if osp.exists(tmp_file_score):
                ans = load(tmp_file_score)
            tups = [x for x, i in zip(tups, indices) if i not in ans]
            indices = [i for i in indices if i not in ans]

            if len(indices):
                new_results = track_progress_rich(
                    MathVerse_auxeval_score,
                    tups,
                    nproc=nproc,
                    chunksize=nproc,
                    keys=indices,
                    save=tmp_file_score,
                )
                ans = load(tmp_file_score)
                for k, v in zip(indices, new_results):
                    assert k in ans
                    assert ans[k]['log_score'] == v['log_score'] and ans[k][
                        'score'] == v['score']

            data['score'] = [ans[idx]['score'] for idx in data['index']]
            data['log_score'] = [
                ans[idx]['log_score'] for idx in data['index']
            ]
            dump(data, storage_score)

        score = MathVerse_acc(storage_score)
        score_pth = get_intermediate_file_path(storage_score, '', 'csv')
        dump(score, score_pth)
        return score

class MathVision(ImageBaseDataset):
    TYPE = 'VQA'
    DATASET_URL = {
        'MathVision':
        'https://opencompass.openxlab.space/utils/VLMEval/MathVision.tsv',
        'MathVision_MINI':
        'https://opencompass.openxlab.space/utils/VLMEval/MathVision_MINI.tsv'
    }
    DATASET_MD5 = {
        'MathVision': '93f6de14f7916e598aa1b7165589831e',
        'MathVision_MINI': '060fe4fa5d868987ce179307bd5f8a33'
    }

    def evaluate(self, eval_file, **judge_kwargs):
        if judge_kwargs.get('use_verifier', False):
            return self.evaluate_verifier(eval_file, **judge_kwargs)
        else:
            return self.evaluate_heuristic(eval_file, **judge_kwargs)

    def evaluate_heuristic(self, eval_file, **judge_kwargs):
        from .utils.mathv import MATH_V_auxeval, MATH_V_acc

        if 'model' in judge_kwargs:
            model = judge_kwargs['model']
        else:
            model = os.path.basename(os.environ.get('LOCAL_LLM'))
        storage = get_intermediate_file_path(eval_file, f'_{model}')
        tmp_file = get_intermediate_file_path(eval_file, f'_{model}', 'pkl')
        nproc = judge_kwargs.pop('nproc', 4)

        if not osp.exists(storage):
            data = load(eval_file)
            model = build_judge(max_tokens=128, **judge_kwargs)
            assert model.working(), 'MATH-Vision evaluation requires a working OPENAI API\n' + DEBUG_MESSAGE
            lt = len(data)
            lines = [data.iloc[i] for i in range(lt)]
            tups = [(model, line) for line in lines]
            indices = [line['index'] for line in lines]

            ans = {}
            if osp.exists(tmp_file):
                ans = load(tmp_file)
            tups = [x for x, i in zip(tups, indices) if i not in ans]
            indices = [i for i in indices if i not in ans]

            if len(indices):
                new_results = track_progress_rich(
                    MATH_V_auxeval,
                    tups,
                    nproc=nproc,
                    chunksize=nproc,
                    keys=indices,
                    save=tmp_file,
                )
                ans = load(tmp_file)
                for k, v in zip(indices, new_results):
                    assert k in ans
                    assert ans[k]['log'] == v['log'] and ans[k]['res'] == v[
                        'res']

            data['res'] = [ans[idx]['res'] for idx in data['index']]
            data['log'] = [ans[idx]['log'] for idx in data['index']]
            dump(data, storage)

        score, hit_score = MATH_V_acc(storage)
        data = load(storage)
        data['hit_score'] = hit_score
        dump(data, storage)
        score_pth = get_intermediate_file_path(storage, '_score', 'csv')
        dump(score, score_pth)
        return score

    # It returns a DataFrame
    @classmethod
    def evaluate_verifier(self, eval_file, **judge_kwargs):
        # Add verifier evaluation for MathVision
        data = load(eval_file)
        if 'verifier_score' not in data.columns:
            from .utils.verifier import Verifier
            verifier = Verifier(use_vllm=judge_kwargs.get('use_vllm', False))

            verifier_scores = []
            verifier_matches = []
            for idx, row in tqdm(data.iterrows(), total=len(data), desc="Verifier Evaluation Progress"):
                question_text = row['question'] if 'question' in row else ""
                prediction_text = row['prediction'] if 'prediction' in row else ""
                answer_text = row['answer'] if 'answer' in row else ""

                score = verifier.evaluate(question_text, prediction_text, answer_text)
                verifier_scores.append(score)
                verifier_matches.append(1.0 if score else 0.0)

            data['verifier_score'] = verifier_scores
            data['verifier_match'] = verifier_matches

            detailed_result_file = get_intermediate_file_path(eval_file, '_detailed_results')
            dump(data, detailed_result_file)

        else:
            detailed_result_file = get_intermediate_file_path(eval_file, '_detailed_results')
            if not osp.exists(detailed_result_file):
                dump(data, detailed_result_file)

        def MathVision_acc_verifier(result_file):
            from collections import defaultdict
            data = load(result_file)
            tot = defaultdict(lambda: 0)
            hit = defaultdict(lambda: 0)
            lt = len(data)

            for i in range(lt):
                item = data.iloc[i]
                cate = item['category'] if 'category' in item else 'Overall'
                tot['Overall'] += 1
                tot[cate] += 1

                if item['verifier_score'] is True:
                    hit['Overall'] += 1
                    hit[cate] += 1

            res = defaultdict(list)
            for k in tot.keys():
                res['Subject'].append(k)
                res['tot'].append(tot[k])
                res['hit'].append(hit[k])
                res['acc'].append(hit[k] / tot[k] * 100)
            res = pd.DataFrame(res).sort_values('Subject', ignore_index=True)
            return res

        score = MathVision_acc_verifier(detailed_result_file)
        score_pth = get_intermediate_file_path(eval_file, '_score', 'csv')
        dump(score, score_pth)
        return score


class LLaVABench(ImageBaseDataset):
    TYPE = 'VQA'
    DATASET_URL = {'LLaVABench': 'https://opencompass.openxlab.space/utils/VLMEval/LLaVABench.tsv'}
    DATASET_MD5 = {'LLaVABench': 'd382a093f749a697820d3dadd61c8428'}

    # It returns a DataFrame
    @classmethod
    def evaluate(self, eval_file, **judge_kwargs):
        from .utils.llavabench import (
            build_prompt,
            LLaVABench_atomeval,
            LLaVABench_score,
        )

        suffix = '.' + eval_file.split('.')[-1]
        record_file = eval_file.replace(suffix, '_openai_result' + suffix)
        score_file = eval_file.replace(suffix, '_score.csv')
        if osp.exists(score_file):
            return load(score_file)

        nproc = judge_kwargs.pop('nproc', 4)
        system_prompt = 'You are a helpful and precise assistant for checking the quality of the answer.'

        if not osp.exists(record_file):
            data = load(eval_file)
            lines = [data.iloc[i] for i in range(len(data))]
            model = build_judge(temperature=0.2, system_prompt=system_prompt, **judge_kwargs)
            assert model.working(), ('LLaVABench evaluation requires a working OPENAI API\n' + DEBUG_MESSAGE)

            prompts = [build_prompt(line) for line in lines]
            tups = [(model, prompt) for prompt in prompts]
            scores = track_progress_rich(LLaVABench_atomeval, tups, nproc=nproc, chunksize=nproc)
            data['gpt4_score'] = [x[0] for x in scores]
            data['score'] = [x[1] for x in scores]
            dump(data, record_file)

        data = load(record_file)
        ret = LLaVABench_score(data).round(1)
        dump(ret, score_file)
        return ret


class MMVet(ImageBaseDataset):
    TYPE = 'VQA'
    DATASET_URL = {
        'MMVet': 'https://opencompass.openxlab.space/utils/VLMEval/MMVet.tsv'
    }
    DATASET_MD5 = {'MMVet': '748aa6d4aa9d4de798306a63718455e3'}

    # It returns a DataFrame
    @classmethod
    def evaluate(self, eval_file, **judge_kwargs):
        from .utils.mmvet import MMVet_auxeval, MMVet_acc

        suffix = eval_file.split('.')[-1]
        model = judge_kwargs['model']
        storage = eval_file.replace(f'.{suffix}', f'_{model}.xlsx')
        score_pth = storage.replace('.xlsx', '_score.csv')
        if osp.exists(score_pth):
            return load(score_pth)

        tmp_file = eval_file.replace(f'.{suffix}', f'_{model}.pkl')
        nproc = judge_kwargs.pop('nproc', 4)
        if not osp.exists(storage):
            data = load(eval_file)
            model = build_judge(**judge_kwargs)
            assert model.working(), ('MMVet evaluation requires a working OPENAI API\n' + DEBUG_MESSAGE)

            lt = len(data)
            lines = [data.iloc[i] for i in range(lt)]
            tups = [(model, line) for line in lines]
            indices = [line['index'] for line in lines]

            ans = load(tmp_file) if osp.exists(tmp_file) else {}
            tups = [x for x, i in zip(tups, indices) if i not in ans]
            indices = [i for i in indices if i not in ans]

            if len(indices):
                new_results = track_progress_rich(
                    MMVet_auxeval,
                    tups,
                    nproc=nproc,
                    chunksize=nproc,
                    keys=indices,
                    save=tmp_file,
                )
                ans = load(tmp_file)
                for k, v in zip(indices, new_results):
                    assert k in ans
                    assert ans[k]['log'] == v['log'] and ans[k]['score'] == v['score']
            data['score'] = [ans[idx]['score'] for idx in data['index']]
            data['log'] = [ans[idx]['log'] for idx in data['index']]
            dump(data, storage)

        score, score_fine = MMVet_acc(storage)
        score_pth = storage.replace('.xlsx', '_score.csv')
        score_fine_pth = storage.replace('.xlsx', '_score_fine.csv')
        dump(score, score_pth)
        dump(score_fine, score_fine_pth)
        return score


class MTVQADataset(ImageBaseDataset):
    TYPE = 'VQA'
    DATASET_URL = {'MTVQA_TEST': 'https://opencompass.openxlab.space/utils/VLMEval/MTVQA_TEST.tsv'}
    DATASET_MD5 = {'MTVQA_TEST': 'd87c17dbab934b7cd89c0a3c1c5657f4'}

    @classmethod
    def evaluate(self, eval_file, **judge_kwargs):
        suffix = eval_file.split('.')[-1]
        result_file = eval_file.replace(f'.{suffix}', '_acc.json')
        if osp.exists(result_file):
            return load(result_file)

        data = load(eval_file)
        assert 'answer' in data and 'prediction' in data and 'category' in data
        data['prediction'] = [str(x) for x in data['prediction']]
        data['answer'] = [str(x) for x in data['answer']]
        if 'split' in data:
            assert np.all([x.lower() == 'test' for x in data['split']]), 'We only support MTVQA_TEST for now. '
        lt = len(data)
        category_scores = defaultdict(list)
        for i in range(lt):
            line = data.iloc[i]
            ans = line['answer'].strip().lower().replace('.', '')
            pred = line['prediction'].strip().lower().replace('.', '')
            cate = line['category']
            score = 1.0 if ans in pred else 0.0
            category_scores[cate].append(score)
            category_scores['Average'].append(score)
        # Calculate the average score for each category, the score is normalized to [0, 100]
        category_averages = {category: np.mean(scores) * 100 for category, scores in category_scores.items()}

        suffix = eval_file.split('.')[-1]
        result_file = eval_file.replace(f'.{suffix}', '_acc.json')
        dump(category_averages, result_file)

        return category_averages

    # MT-VQA adopts a custom prompt
    def build_prompt(self, line):
        msgs = super().build_prompt(line)
        assert sum([x['type'] == 'text' for x in msgs]) == 1
        for item in msgs:
            if item['type'] == 'text':
                item['value'] += '\nAnswer the question using a word or phrase in the language of the question.'
        return msgs


class TableVQABench(ImageBaseDataset):
    TYPE = 'VQA'
    DATASET_URL = {
        'TableVQABench': 'https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/mentor-vil/datasets/tablevqa-bench.tsv'
    }
    DATASET_MD5 = {'TableVQABench': '2550adc61bdc82d8e62f3b003de7c62d'}

    from .utils.tablevqabench import FINTABNETQA_PROMPT, VTABFACT_PROMPT, VWTQ_PROMPT

    # It returns a DataFrame
    @classmethod
    def evaluate(self, eval_file, **judge_kwargs):
        import pandas as pd
        from .utils.tablevqabench import evaluate_fintabnet, evaluate_tabfact, evaluate_wtq

        suffix = eval_file.split('.')[-1]
        result_file = eval_file.replace(f'.{suffix}', '_acc.csv')
        if osp.exists(result_file):
            return load(result_file)

        data = load(eval_file)
        assert 'answer' in data and 'prediction' in data

        data['prediction'] = data['prediction'].str.replace('^Answer: ', '', regex=True)
        data_group = dict(tuple(data.groupby('split')))
        eval_result = {'split': [], 'average_scores': []}
        for split in ['fintabnetqa', 'vtabfact', 'vwtq', 'vwtq_syn']:
            data_split = data_group[split].to_dict(orient='records')
            if split == 'fintabnetqa':
                split_eval_meta = evaluate_fintabnet(data_split, ['accuracy'])
            elif split == 'vtabfact':
                split_eval_meta = evaluate_tabfact(data_split, ['accuracy'])
            elif split == 'vwtq' or split == 'vwtq_syn':
                split_eval_meta = evaluate_wtq(data_split, ['accuracy'])
            eval_result['split'].append(split)
            eval_result['average_scores'].append(split_eval_meta['average_scores'])

        suffix = eval_file.split('.')[-1]
        result_file = eval_file.replace(f'.{suffix}', '_acc.csv')
        eval_result = pd.DataFrame(eval_result)
        dump(eval_result, result_file)

        return eval_result

    # TableVQABench adopts a custom prompt
    def build_prompt(self, line):
        msgs = super().build_prompt(line)
        assert sum([x['type'] == 'text' for x in msgs]) == 1
        for item in msgs:
            if item['type'] == 'text':
                if line['split'] == 'fintabnetqa':
                    item['value'] = self.FINTABNETQA_PROMPT.format_map({'question': item['value']})
                elif line['split'] == 'vtabfact':
                    item['value'] = self.VTABFACT_PROMPT.format_map({'question': item['value']})
                elif line['split'] == 'vwtq_syn' or line['split'] == 'vwtq':
                    item['value'] = self.VWTQ_PROMPT.format_map({'question': item['value']})
        return msgs


class CustomVQADataset(ImageBaseDataset):
    TYPE = 'VQA'

    def load_data(self, dataset):
        data_path = osp.join(LMUDataRoot(), f'{dataset}.tsv')

        if file_size(data_path, 'GB') > 1:
            local_path = data_path.replace('.tsv', '_local.tsv')
            if not osp.exists(local_path) or os.environ.get('FORCE_LOCAL', None):
                from ..tools import LOCALIZE

                LOCALIZE(data_path, local_path)
            data_path = local_path
        return load(data_path)

    def evaluate(self, eval_file, **judge_kwargs):
        raise NotImplementedError
