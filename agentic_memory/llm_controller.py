import os
import logging
from typing import Optional, Literal
from abc import ABC, abstractmethod
from tenacity import retry, stop_after_attempt, wait_fixed

logger = logging.getLogger(__name__)


class BaseLLMController(ABC):
    @abstractmethod
    def get_completion(self, prompt: str) -> str:
        pass


class OpenAIController(BaseLLMController):
    def __init__(self, model: str = "gpt-4", api_key: Optional[str] = None, base_url: Optional[str] = None):
        try:
            from openai import OpenAI
            self.model = model
            if api_key is None:
                api_key = os.getenv('OPENAI_API_KEY')
            if api_key is None:
                raise ValueError("OpenAI API key not found.")
            self.client = OpenAI(api_key=api_key, base_url=base_url)
        except ImportError:
            raise ImportError("OpenAI package not found. Install it with: pip install openai")

    @retry(stop=stop_after_attempt(8), wait=wait_fixed(5))
    def get_completion(self, prompt: str, response_format: dict = None, temperature: float = 0.7) -> str:
        # 构造基础参数
        kwargs = {
            "model": self.model,
            "messages": [
                # 关键点1：DeepSeek开启JSON模式要求System Prompt里必须包含"JSON"字样
                {"role": "system",
                 "content": "You are a helpful assistant. You must respond with a valid JSON object."},
                {"role": "user", "content": prompt}
            ],
            "temperature": temperature,
            "max_tokens": 4000
        }

        # 关键点2：强制降级 response_format
        # 不管传入什么复杂的 schema，只要有 format，就只给 DeepSeek 传 {"type": "json_object"}
        if response_format:
            kwargs["response_format"] = {"type": "json_object"}

        try:
            response = self.client.chat.completions.create(**kwargs)
            return response.choices[0].message.content
        except Exception as e:
            # 打印错误日志，然后抛出异常触发重试
            logger.warning(f"LLM call failed, retrying... Error: {e}")
            raise e


class LLMController:
    def __init__(self,
                 backend: Literal["openai"] = "openai",
                 model: str = "gpt-4",
                 api_key: Optional[str] = None,
                 base_url: Optional[str] = None):
        if backend == "openai":
            self.llm = OpenAIController(model=model, api_key=api_key, base_url=base_url)
        else:
            raise ValueError("Only 'openai' backend is supported.")

    def get_completion(self, prompt: str, response_format: dict = None, temperature: float = 0.7) -> str:
        return self.llm.get_completion(prompt, response_format, temperature)