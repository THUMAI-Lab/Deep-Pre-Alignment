import openai
import time
import os
import random
import logging
import traceback
from typing import Dict, Any, Optional, Tuple
import requests  # 引入 requests 库以处理超时异常

DEFAULT_BASE_URL = "https://your_api_url/"

DEFAULT_API_KEY = 'sk-xxxxxxxx'

# Configure logger
logger = logging.getLogger('OpenAIChatClient')

# 模型名称常量
MODEL_GPT4 = "gpt-4"
MODEL_GPT4_TURBO = "gpt-4-turbo"
MODEL_GPT4o = "gpt-4o"
MODEL_GPT4o_mini = "gpt-4o-mini"
MODEL_GPT35_TURBO = "gpt-3.5-turbo"
MODEL_GPT41 = "gpt-4.1"

# Connection error types that need exponential backoff
CONNECTION_ERRORS = (
    requests.exceptions.ConnectionError,
    requests.exceptions.Timeout,
    ConnectionResetError,
    ConnectionError,
    TimeoutError,
)
class OpenAIChatClient:
    """
    OpenAI 聊天客户端封装类，支持超时机制。
    """

    def __init__(self, api_key: str = DEFAULT_API_KEY, base_url: Optional[str] = DEFAULT_BASE_URL,
                 organization: Optional[str] = None):
        """
        初始化 OpenAI 聊天客户端。

        Args:
            api_key: OpenAI API Key
            base_url: 可选的 base_url（如使用代理或自建代理）
            organization: 可选的 organization ID
        """
        openai.api_key = api_key
        if base_url:
            openai.base_url = base_url
        if organization:
            openai.organization = organization

    def chat_sync(self,
                  user_prompt: str,
                  system_prompt: str = 'You are a helpful assistant.',
                  model: str = MODEL_GPT4o,
                  max_tokens: int = 1024,
                  temperature: float = 0.7,
                  tools: Optional[list] = None,
                  timeout: float = 60.0) -> Tuple[str, Dict[str, Any]]:
        """
        同步聊天请求，添加超时机制。

        Args:
            user_prompt: 用户输入
            system_prompt: 系统提示
            model: 使用的模型名称
            max_tokens: 最大回复长度
            temperature: 控制生成的随机程度
            tools: 可选，传入 tool_calls 时使用
            timeout: 请求超时时间（秒）

        Returns:
            Tuple[str, Dict[str, Any]]: 返回模型回复与完整响应

        Raises:
            TimeoutError: 如果请求超时
            RuntimeError: 其他 OpenAI API 错误
        """
        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]

            response = openai.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                tools=tools or [],
                tool_choice="auto" if tools else None,
                timeout=timeout  # 设置请求超时
            )
            reply = response.choices[0].message.content
            return reply, response.model_dump()

        except requests.exceptions.Timeout:
            raise TimeoutError(f"OpenAI API request timed out after {timeout} seconds")
        except Exception as e:
            traceback.print_exc()
            raise RuntimeError(f"OpenAI chat failed: {e}")

    def chat_sync_retry(self,
                        user_prompt: str,
                        system_prompt: str = 'You are a helpful assistant.',
                        model: str = MODEL_GPT4o,
                        max_tokens: int = 1024,
                        temperature: float = 0.1,
                        max_retry: int = 10,
                        timeout: float = 60.0) -> Optional[Tuple[str, Dict[str, Any]]]:
        """
        带重试机制的聊天，支持超时和指数退避。

        Args:
            user_prompt: 用户输入
            system_prompt: 系统提示
            model: 使用的模型名称
            max_tokens: 最大回复长度
            temperature: 控制生成的随机程度
            max_retry: 最大重试次数
            timeout: 单次请求超时时间（秒）

        Returns:
            成功返回 (回复, 完整响应)，失败返回 None
        """
        for attempt in range(max_retry):
            try:
                return self.chat_sync(
                    user_prompt=user_prompt,
                    system_prompt=system_prompt,
                    model=model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    timeout=timeout
                )
            except CONNECTION_ERRORS as err:
                # Connection errors need exponential backoff
                base_delay = min(2 ** (attempt + 1), 60)  # 2, 4, 8, 16, 32, 60, 60...
                delay = base_delay * (0.5 + random.random())  # Add jitter
                logger.warning(
                    f"{model} 连接错误 {attempt + 1}/{max_retry}: {type(err).__name__}: {str(err)[:80]}"
                )
                logger.info(f"等待 {delay:.1f}s 后重试...")
                if attempt < max_retry - 1:
                    time.sleep(delay)
            except Exception as err:
                # Other errors use shorter delay
                logger.error(f"{model} 尝试 {attempt + 1}/{max_retry} 失败: {err}")
                if attempt < max_retry - 1:
                    time.sleep(2 + random.random() * 2)
        return None


client = OpenAIChatClient()

# 示例用法
if __name__ == "__main__":
    try:
        reply, raw = client.chat_sync("What is your model series id?", model='gpt-4o-mini', timeout=10.0)
        print("模型回复：", reply)
    except TimeoutError as e:
        print(f"请求超时: {e}")
    except RuntimeError as e:
        print(f"请求失败: {e}")