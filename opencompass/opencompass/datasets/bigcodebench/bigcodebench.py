# Copyright (c) 2024, BigCodeBench and its contributors.
# Copyright (c) 2023, OpenCompass and its contributors.

import os
import random
import time
from concurrent.futures._base import CancelledError

import httpx
from datasets import Dataset, DatasetDict
from gradio_client import Client, handle_file

from opencompass.openicl.icl_evaluator import BaseEvaluator
from opencompass.utils import JSONToolkit, get_data_path, get_logger

# from opencompass.utils import check_url_accessibility,
from ..base import BaseDataset
from .extractor import extract_code_generation


# 在文件顶部添加工作函数（在类定义之前）
def _extract_worker_function(prediction, entrypoint, enable_cache,
                             prediction_hash):
    """独立的工作进程函数 - 必须在模块级别定义以支持pickle"""
    try:
        # 在工作进程中重新导入
        from .extractor import extract_code_generation

        if enable_cache:
            from functools import lru_cache

            @lru_cache(maxsize=100)  # 进程级别的小缓存
            def cached_extract(pred_hash, pred, ep):
                return extract_code_generation(pred, entrypoint=ep)

            return cached_extract(prediction_hash, prediction, entrypoint)
        else:
            return extract_code_generation(prediction, entrypoint=entrypoint)
    except Exception as e:
        # 在工作进程中无法直接使用logger，使用print
        print(f'Worker process extraction failed: {e}')
        return ''


class BigCodeBenchDataset(BaseDataset):

    @staticmethod
    def load(path: str = 'opencompass/bigcodebench',
             local_mode: bool = False,
             release_version: str = 'v0.1.2',
             dataset_version: str = 'full'):
        """
        Args:
            path (str): The path to the dataset.
            local_mode (bool): Whether to use local give path or use
                automatically download.
            release_version (str): The release version of the dataset.
            dataset_version (str): The data version of the dataset.
                only support ['full', 'hard']
        """
        assert dataset_version in ['full', 'hard'], \
            'dataset_version should be one of ["full", "hard"], '
        f'but got {dataset_version}'
        path = get_data_path(path, local_mode=local_mode)
        dataset = DatasetDict()
        # Valid Keys:
        # 'task_id', 'complete_prompt', 'instruct_prompt',
        # 'canonical_solution', 'code_prompt', 'test',
        # 'entry_point', 'doc_struct', 'libs'
        if dataset_version == 'full':
            items = JSONToolkit.read_jsonl(
                os.path.join(path, f'BigCodeBench-{release_version}.jsonl'))
        else:
            items = JSONToolkit.read_jsonl(
                os.path.join(path,
                             f'BigCodeBench-Hard-{release_version}.jsonl'))

        dataset['train'] = Dataset.from_list(items)
        dataset['test'] = Dataset.from_list(items)

        return dataset


class BigCodeBenchEvaluator(BaseEvaluator):
    """针对多进程环境优化的BigCodeBench评估器.

    设计原则:
    1. 减少进程内线程竞争，适配已有的多进程架构
    2. 保留缓存优化和超时机制
    3. 使用进程安全的进度监控
    4. 线程安全的超时处理
    """

    def __init__(
            self,
            release_version='v0.1.2',
            eval_type='instruct',
            remote_execute_api='https://bigcode-bigcodebench-evaluator.hf.space/',  # noqa: E501
            backup_apis=None,  # 新增：备用API地址列表
            dataset_version: str = 'full',
            local_mode: bool = False,
            path: str = 'opencompass/bigcodebench',
            pass_k: str = '1,5,10',
            parallel: int = -1,
            min_time_limit: float = 1,
            max_as_limit: int = 30 * 1024,
            max_data_limit: int = 30 * 1024,
            max_stack_limit: int = 10,
            check_gt_only: bool = False,
            no_gt: bool = False,
            # 新增多进程优化参数
            enable_cache: bool = True,
            cache_size: int = 500,  # 减少缓存大小避免内存竞争
            timeout_seconds: int = 10,
            use_threading: bool = False,  # 默认关闭多线程，适配多进程
            max_workers: int = 32):  # 如果启用线程，使用较少线程数
        super().__init__()
        self.dataset = BigCodeBenchDataset.load(
            release_version=release_version,
            dataset_version=dataset_version,
            local_mode=local_mode,
            path=path)['test']
        self.eval_type = eval_type

        # 处理主API和备用API地址
        self.remote_execute_api = remote_execute_api

        # 设置默认备用地址
        if backup_apis is None:
            backup_apis = [
                'https://bigcode-bigcodebench-evaluator.hf.space/',
                'https://mingzhong-bigcodebench-evaluator.hf.space/',
                'https://bigcode-bigcodebench-evaluator-1.hf.space/',
                'https://tingting-bigcodebench-evaluator.hf.space/',
                'https://tingting-bigcodebench-evaluator22.hf.space/',
                # 'http://pytorchjob-eval-lby-57783-master-0:7860/',
            ]

        # 构建完整的API地址列表（主地址 + 备用地址）
        self.api_endpoints = [remote_execute_api] + backup_apis

        # # 为所有API地址修复网络配置
        # for api in self.api_endpoints:
        #     self._fix_network_config(api)

        # 优化参数
        self.enable_cache = enable_cache
        self.cache_size = cache_size
        self.timeout_seconds = timeout_seconds
        self.use_threading = use_threading
        self.max_workers = max_workers

        self.eval_kwargs = dict(subset=dataset_version,
                                pass_k=pass_k,
                                parallel=parallel,
                                min_time_limit=min_time_limit,
                                max_as_limit=max_as_limit,
                                max_data_limit=max_data_limit,
                                max_stack_limit=max_stack_limit,
                                check_gt_only=check_gt_only,
                                no_gt=no_gt)

    def _create_cached_extractor(self):
        """创建带缓存的提取器（进程级别）"""
        if not self.enable_cache:
            return extract_code_generation

        from functools import lru_cache

        @lru_cache(maxsize=self.cache_size)
        def cached_extract_code_generation(prediction_hash, prediction,
                                           entrypoint):
            """进程级别的缓存提取函数."""
            return extract_code_generation(prediction, entrypoint=entrypoint)

        return cached_extract_code_generation

    def _extract_with_signal_timeout(self, cached_extractor, prediction,
                                     entrypoint):
        """使用signal实现的超时机制（仅限Linux/Unix系统）"""
        import hashlib
        import signal

        logger = get_logger()

        def timeout_handler(signum, frame):
            raise TimeoutError(
                f'Code extraction timeout after {self.timeout_seconds} seconds'
            )

        try:
            # 创建哈希用于缓存
            prediction_hash = hashlib.md5(
                f'{prediction}_{entrypoint}'.encode()).hexdigest()

            def _extract():
                if self.enable_cache:
                    return cached_extractor(prediction_hash, prediction,
                                            entrypoint)
                else:
                    return cached_extractor(prediction, entrypoint=entrypoint)

            # 设置信号处理器
            old_handler = signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(self.timeout_seconds)

            try:
                result = _extract()
                return result
            finally:
                # 恢复原来的信号处理器
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)

        except TimeoutError:
            logger.warning(
                f'Code extraction timeout for entrypoint {entrypoint}: '
                f'{self.timeout_seconds}s timeout (signal-based)')
            return ''
        except Exception as e:
            logger.warning(
                f'Code extraction failed for entrypoint {entrypoint}: {str(e)}'
            )
            return ''

    def _extract_with_timeout_safe(self, cached_extractor, prediction,
                                   entrypoint):
        """进程安全的超时提取（优先使用signal，回退到进程池）"""
        import hashlib
        import platform
        from concurrent.futures import (
            ProcessPoolExecutor,
            TimeoutError as FuturesTimeoutError,
        )

        logger = get_logger()

        # 在Unix-like系统上优先使用signal超时
        if platform.system() in ['Linux', 'Darwin']:  # Linux或macOS
            try:
                return self._extract_with_signal_timeout(
                    cached_extractor, prediction, entrypoint)
            except Exception as e:
                logger.warning(f'Signal-based timeout failed: {e}')

        # 回退到进程池方案
        try:
            # 创建哈希用于缓存
            prediction_hash = hashlib.md5(
                f'{prediction}_{entrypoint}'.encode()).hexdigest()

            # 使用进程池实现真正的超时
            try:
                with ProcessPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(_extract_worker_function,
                                             prediction, entrypoint,
                                             self.enable_cache,
                                             prediction_hash)
                    try:
                        result = future.result(timeout=self.timeout_seconds)
                        return result
                    except FuturesTimeoutError:
                        logger.warning('Code extraction timeout '
                                       f'for entrypoint {entrypoint}: '
                                       f'{self.timeout_seconds}s timeout')

                        # 强制取消和终止
                        future.cancel()

                        # 尝试强制关闭executor中的进程
                        try:
                            executor.shutdown(wait=False)  # 不等待进程完成
                        except Exception as e:
                            print(f'Error: {e}')
                            pass

                        return ''
            except Exception as proc_e:
                logger.warning(f'Process pool execution failed: {proc_e}')
                # 回退到原来的线程方式，但增加更短的超时
                return self._extract_with_thread_fallback(
                    cached_extractor, prediction, entrypoint)

        except Exception as e:
            logger.warning(
                f'Code extraction failed for entrypoint {entrypoint}: {str(e)}'
            )
            return ''

    def _extract_with_thread_fallback(self, cached_extractor, prediction,
                                      entrypoint):
        """线程回退方案（缩短超时时间）"""
        import hashlib
        from concurrent.futures import (
            ThreadPoolExecutor,
            TimeoutError as FuturesTimeoutError,
        )

        logger = get_logger()

        try:
            prediction_hash = hashlib.md5(
                f'{prediction}_{entrypoint}'.encode()).hexdigest()

            def _extract():
                if self.enable_cache:
                    return cached_extractor(prediction_hash, prediction,
                                            entrypoint)
                else:
                    return cached_extractor(prediction, entrypoint=entrypoint)

            # 使用更短的超时时间作为回退
            fallback_timeout = min(self.timeout_seconds // 2, 5)  # 最多5秒
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(_extract)
                try:
                    result = future.result(timeout=fallback_timeout)
                    return result
                except FuturesTimeoutError:
                    logger.warning(
                        f'Thread fallback timeout for entrypoint {entrypoint}:'
                        f'{fallback_timeout}s timeout')
                    return ''
        except Exception as e:
            logger.warning(f'Thread fallback failed: {e}')
            return ''

    def _fix_network_config(self, ip_address):
        """修复网络配置，确保内网地址绕过代理."""
        import os

        # 提取主机名
        hostname = ip_address.replace('http://', '').replace('https://',
                                                             '').split(':')[0]

        # 如果是内网地址，添加到NO_PROXY
        if any(pattern in hostname for pattern in [
                'pytorchjob-', 'localhost', '127.0.0.1', '192.168.', '10.',
                '172.'
        ]):
            current_no_proxy = os.environ.get('NO_PROXY',
                                              os.environ.get('no_proxy', ''))

            # 添加必要的绕过模式
            no_proxy_patterns = [
                'localhost', '127.0.0.1', hostname, 'pytorchjob-*', '*.local'
            ]

            if current_no_proxy:
                existing = [
                    p.strip() for p in current_no_proxy.split(',')
                    if p.strip()
                ]
                all_patterns = list(set(existing + no_proxy_patterns))
            else:
                all_patterns = no_proxy_patterns

            new_no_proxy = ','.join(all_patterns)
            os.environ['NO_PROXY'] = new_no_proxy
            os.environ['no_proxy'] = new_no_proxy

    def score(self, predictions, references):
        logger = get_logger()
        entrypoints = [item['entry_point'] for item in self.dataset]

        # Append content for completion mode
        if self.eval_type == 'complete':
            content = [item['complete_prompt'] for item in self.dataset]
            predictions = [
                content[idx] + item for idx, item in enumerate(predictions)
            ]
        elif self.eval_type == 'instruct':
            pass
        else:
            raise ValueError(f'Unknown eval_type: {self.eval_type}')

        # 代码提取优化 (多进程友好版本)
        logger.info(
            'Start to extract code from predictions (multiprocess-optimized)')

        # 创建缓存提取器
        cached_extractor = self._create_cached_extractor()

        # 获取初始缓存状态
        if self.enable_cache:
            cache_info_before = cached_extractor.cache_info()
            logger.info(f'Cache状态 (开始前): '
                        f'hits={cache_info_before.hits}, '
                        f'misses={cache_info_before.misses}')

        sanitized_predictions = []

        if self.use_threading and len(predictions) > 10:
            # 线程模式：仅在数据量大时使用少量线程
            from concurrent.futures import ThreadPoolExecutor, as_completed

            from tqdm import tqdm

            logger.info(f'Using {self.max_workers} threads')

            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # 预分配结果
                results = [''] * len(predictions)

                # 提交任务
                future_to_index = {
                    executor.submit(self._extract_with_timeout_safe,
                                    cached_extractor, pred, ep): idx
                    for idx, (pred,
                              ep) in enumerate(zip(predictions, entrypoints))
                }

                # 处理结果 (进程安全的进度条)
                with tqdm(total=len(predictions),
                          desc='Extracting (MP)',
                          unit='pred',
                          disable=not logger.isEnabledFor(20)
                          ) as pbar:  # 只在INFO级别显示
                    for future in as_completed(future_to_index):
                        idx = future_to_index[future]
                        try:
                            result = future.result()
                            results[idx] = result
                        except Exception as e:
                            logger.warning(
                                f'Thread execution failed for index {idx}: {e}'
                            )
                            results[idx] = ''
                        pbar.update(1)

                sanitized_predictions = results
        else:
            # 单线程模式：适合多进程环境，避免资源竞争
            logger.info('Using single-thread mode (multiprocess-optimized)')

            from tqdm import tqdm

            with tqdm(total=len(predictions),
                      desc='Extracting (MP-Single)',
                      unit='pred',
                      disable=not logger.isEnabledFor(20)) as pbar:
                for pred, ep in zip(predictions, entrypoints):
                    result = self._extract_with_timeout_safe(
                        cached_extractor, pred, ep)
                    sanitized_predictions.append(result)
                    pbar.update(1)

        # 最终统计
        if self.enable_cache:
            cache_info_final = cached_extractor.cache_info()
            total_requests = cache_info_final.hits + cache_info_final.misses
            hit_rate = cache_info_final.hits / \
                total_requests * 100 if total_requests > 0 else 0

            logger.info('Code extraction completed!')
            logger.info(f'Cache统计 (最终): '
                        f'hits={cache_info_final.hits}, '
                        f'misses={cache_info_final.misses}')
            logger.info(f'Cache命中率: {hit_rate:.1f}% '
                        f'({cache_info_final.hits}/{total_requests})')

        successful_extractions = sum(1 for pred in sanitized_predictions
                                     if pred.strip())
        logger.info(f'成功提取代码: '
                    f'{successful_extractions}/{len(predictions)} '
                    f'({successful_extractions/len(predictions)*100:.1f}%)')

        # 准备提交内容
        submitted_contents = []
        task_ids = [item['task_id'] for item in self.dataset]
        for task_id, sanitized_prediction in zip(task_ids,
                                                 sanitized_predictions):
            submitted_content = {
                'task_id': task_id,
                'solution': sanitized_prediction
            }
            submitted_contents.append(submitted_content)

        submitted_contents_path = os.path.join(
            self._out_dir, 'bigcodebench_submitted_contents.jsonl')
        JSONToolkit.save_jsonl(submitted_contents, submitted_contents_path)
        logger.info(f'Dump submitted contents to {submitted_contents_path}')

        # 远程评估
        logger.info('Start to connect to remote APIs for evaluating')
        logger.info(f'Available API endpoints: {self.api_endpoints}')

        # 依次尝试所有API端点
        results = None
        pass_at_k = None
        successful_api = None

        for api_index, api_endpoint in enumerate(self.api_endpoints):
            logger.info(f'尝试API端点 {api_index + 1}/{len(self.api_endpoints)}: '
                        f'{api_endpoint}')

            # 连接重试机制（每个API地址重试3次）
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    logger.info(
                        f'连接到 {api_endpoint} (第{attempt + 1}/{max_retries}次)')
                    eval_client = Client(
                        api_endpoint,
                        httpx_kwargs=dict(timeout=httpx.Timeout(300.0)))
                    logger.info(f'建立连接成功 {eval_client}')
                    results, pass_at_k = eval_client.predict(
                        split=self.eval_type,
                        samples=handle_file(submitted_contents_path),
                        api_name='/predict',
                        **self.eval_kwargs)
                    logger.info(f'连接成功！使用API: {results}')
                    successful_api = api_endpoint
                    break
                except (httpx.ReadTimeout, CancelledError) as e:
                    logger.warning(f'API {api_endpoint} 连接超时或被取消: {e}')
                    if attempt == max_retries - 1:
                        logger.warning(f'API {api_endpoint} 达到最大重试次数，尝试下一个API')
                    else:
                        retry_delay = random.uniform(2, 5)
                        logger.info(f'等待{retry_delay:.1f}秒后重试...')
                        time.sleep(retry_delay)
                except Exception as e:
                    logger.warning(f'API {api_endpoint} 连接失败: {e}')
                    if attempt == max_retries - 1:
                        logger.warning(f'API {api_endpoint} 达到最大重试次数，尝试下一个API')
                    else:
                        retry_delay = random.uniform(1, 3)
                        logger.info(f'等待{retry_delay:.1f}秒后重试...')
                        time.sleep(retry_delay)

            # 如果当前API成功连接，跳出循环
            if successful_api:
                break

        # 检查是否有任何API成功连接
        if not successful_api:
            error_msg = f'所有API端点都连接失败: {self.api_endpoints}'
            logger.error(error_msg)
            raise ConnectionError(error_msg)

        if 'pass@1' in pass_at_k.keys():
            pass_at_k['pass@1'] *= 100
        dump_results = {'details': self._results_processor(results)}
        dump_results.update(pass_at_k)

        return dump_results

    def _results_processor(self, results):
        details = []
        for key, value in results['eval'].items():
            if value[0]['status'] == 'pass':
                value[0]['correct'] = True
            else:
                value[0]['correct'] = False
            details.append(value[0])
        return details
