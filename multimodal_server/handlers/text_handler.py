"""文本模型处理器"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, TextIteratorStreamer
from threading import Thread
from typing import List, Dict, Any, AsyncIterator, Optional
import asyncio
from dataclasses import dataclass

from .base import BaseModelHandler, GenerationConfig as GenConfig
from ..utils.message_parser import format_prompt_with_template


# ============ DeepSeek-R1 Reasoning Parser ============

@dataclass
class ReasoningParserResult:
    """解析结果：包含推理内容和最终答案"""
    content: str = ""
    reasoning_content: str = ""


class DeepSeekR1Parser:
    """
    DeepSeek-R1 推理解析器
    格式: <think>推理过程</think>最终答案
    """

    def __init__(self, reasoning_at_start: bool = True):
        self.reasoning_start = "<think>"
        self.reasoning_end = "</think>"
        self.reasoning_at_start = reasoning_at_start
        self.in_reasoning = self.reasoning_at_start
        self._buffer = ""

    def parse(self, text: str) -> ReasoningParserResult:
        """非流式解析：一次性解析完整文本"""
        if not self.reasoning_at_start:
            splits = text.partition(self.reasoning_start)
            if splits[1] == "":
                return ReasoningParserResult(content=text)
            text = splits[2]

        splits = text.partition(self.reasoning_end)
        reasoning_content, content = splits[0], splits[2]
        return ReasoningParserResult(
            content=content,
            reasoning_content=reasoning_content
        )

    def parse_delta(self, delta_text: str) -> ReasoningParserResult:
        """流式解析：增量解析文本片段"""
        self._buffer += delta_text
        delta_text = self._buffer
        reasoning_content = None
        content = None

        if (self.reasoning_start.startswith(delta_text) or
            self.reasoning_end.startswith(delta_text)):
            return ReasoningParserResult()

        if not self.in_reasoning:
            begin_idx = delta_text.find(self.reasoning_start)
            if begin_idx == -1:
                self._buffer = ""
                return ReasoningParserResult(content=delta_text)
            self.in_reasoning = True
            reasoning_content = delta_text[begin_idx + len(self.reasoning_start):]

        if self.in_reasoning:
            delta_text = reasoning_content if reasoning_content is not None else delta_text
            end_idx = delta_text.find(self.reasoning_end)
            if end_idx == -1:
                last_idx = delta_text.rfind(self.reasoning_end[0])
                if last_idx != -1 and self.reasoning_end.startswith(delta_text[last_idx:]):
                    self._buffer = delta_text[last_idx:]
                    reasoning_content = delta_text[:last_idx]
                else:
                    self._buffer = ""
                    reasoning_content = delta_text
                return ReasoningParserResult(reasoning_content=reasoning_content)

            reasoning_content = delta_text[:end_idx]
            content = delta_text[end_idx + len(self.reasoning_end):]
            self.in_reasoning = False
            self._buffer = ""
            return ReasoningParserResult(
                content=content,
                reasoning_content=reasoning_content
            )

        raise RuntimeError("Unreachable code in DeepSeekR1Parser.parse_delta")


class TextModelHandler(BaseModelHandler):
    """
    文本模型处理器

    支持标准的因果语言模型（如 DeepSeek-R1, Qwen, LLaMA 等）
    """

    def load_model(self):
        """加载文本模型"""
        print(f"正在加载文本模型: {self.model_path}")

        # 加载 tokenizer
        print("  - 加载 tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            padding_side="left"
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # 加载模型
        print(f"  - 加载模型到 {self.device}...")
        device_available = self.device == "cuda" and torch.cuda.is_available()

        try:
            import bitsandbytes
            if device_available:
                print("  - 使用 4bit 量化...")
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    trust_remote_code=True,
                    load_in_4bit=True,
                    device_map="auto"
                )
            else:
                raise ImportError("CPU 不支持量化")
        except ImportError:
            if device_available:
                print("  - 使用 float16...")
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    trust_remote_code=True,
                    torch_dtype=torch.float16,
                    device_map="auto"
                )
            else:
                print("  - 使用 float32 (CPU)...")
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    trust_remote_code=True,
                    torch_dtype=torch.float32
                ).to(self.device)

        self.model.eval()
        print("✓ 文本模型加载完成")

    async def generate(
        self,
        messages: List[Dict[str, Any]],
        config: GenConfig,
        images: Optional[List[Any]] = None
    ) -> str:
        """非流式生成"""
        if images:
            print("警告: 文本模型不支持图像输入，将忽略图像")

        # 格式化提示
        prompt = format_prompt_with_template(messages, self.tokenizer)

        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        # 生成配置
        gen_config = GenerationConfig(
            max_new_tokens=config.max_new_tokens,
            temperature=config.temperature,
            top_p=config.top_p,
            top_k=config.top_k,
            repetition_penalty=config.repetition_penalty,
            do_sample=config.do_sample,
            pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id
        )

        # 生成
        outputs = self.model.generate(
            **inputs,
            generation_config=gen_config
        )

        # 解码
        response_text = self.tokenizer.decode(
            outputs[0][len(inputs["input_ids"][0]):],
            skip_special_tokens=True
        )

        # 使用 reasoning parser 解析
        parser = DeepSeekR1Parser(reasoning_at_start=True)
        result = parser.parse(response_text)

        return result

    async def generate_stream(
        self,
        messages: List[Dict[str, Any]],
        config: GenConfig,
        images: Optional[List[Any]] = None
    ) -> AsyncIterator[ReasoningParserResult]:
        """流式生成"""
        if images:
            print("警告: 文本模型不支持图像输入，将忽略图像")

        # 格式化提示
        prompt = format_prompt_with_template(messages, self.tokenizer)

        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        # 创建 streamer
        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True
        )

        # 生成配置
        gen_config = GenerationConfig(
            max_new_tokens=config.max_new_tokens,
            temperature=config.temperature,
            top_p=config.top_p,
            top_k=config.top_k,
            repetition_penalty=config.repetition_penalty,
            do_sample=config.do_sample,
            pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id
        )

        # 在后台线程中生成
        generation_kwargs = dict(
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask"),
            generation_config=gen_config,
            streamer=streamer
        )

        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()

        # 初始化 reasoning parser
        parser = DeepSeekR1Parser(reasoning_at_start=True)

        # 流式返回
        for text in streamer:
            if text:
                result = parser.parse_delta(text)
                yield result
                await asyncio.sleep(0)

        thread.join()

    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        return {
            "type": "text",
            "model_path": self.model_path,
            "device": self.device,
            "supports_images": False,
        }

    def supports_images(self) -> bool:
        return False
