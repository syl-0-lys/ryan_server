"""多模态模型处理器"""
import torch
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer, TextIteratorStreamer
from threading import Thread
from typing import List, Dict, Any, AsyncIterator, Optional
from PIL import Image
import asyncio

from .base import BaseModelHandler, GenerationConfig as GenConfig


class MultimodalModelHandler(BaseModelHandler):
    """
    多模态模型处理器

    支持视觉-语言模型（如 DeepSeek-VL, Qwen2-VL, LLaVA 等）
    """

    def __init__(self, model_path: str, device: str = "cuda"):
        super().__init__(model_path, device)
        self.processor = None

    def load_model(self):
        """加载多模态模型"""
        print(f"正在加载多模态模型: {self.model_path}")

        # 加载 processor（包含 tokenizer 和 image processor）
        print("  - 加载 processor...")
        try:
            self.processor = AutoProcessor.from_pretrained(
                self.model_path,
                trust_remote_code=True
            )
            # processor 通常包含 tokenizer
            self.tokenizer = self.processor.tokenizer if hasattr(self.processor, 'tokenizer') else None
        except Exception as e:
            print(f"  ! 加载 processor 失败: {e}")
            print("  - 尝试分别加载 tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True
            )
            self.processor = None

        # 确保 tokenizer 有 pad_token
        if self.tokenizer and self.tokenizer.pad_token is None:
            print("  - 设置 pad_token...")
            if self.tokenizer.eos_token:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            else:
                self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})

        # 加载模型
        print(f"  - 加载模型到 {self.device}...")
        device_available = self.device == "cuda" and torch.cuda.is_available()

        # 加载模型参数
        model_kwargs = {
            "trust_remote_code": True,
        }

        try:
            import bitsandbytes
            if device_available:
                print("  - 使用 4bit 量化...")
                model_kwargs.update({
                    "load_in_4bit": True,
                    "device_map": "auto"
                })
            else:
                raise ImportError("CPU 不支持量化")
        except ImportError:
            if device_available:
                print("  - 使用 float16...")
                model_kwargs.update({
                    "torch_dtype": torch.float16,
                    "device_map": "auto"
                })
            else:
                print("  - 使用 float32 (CPU)...")
                model_kwargs["torch_dtype"] = torch.float32

        # 尝试加载模型
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                **model_kwargs
            )
        except (ValueError, KeyError) as e:
            # 如果 AutoModelForCausalLM 失败，尝试直接导入自定义模型类
            print(f"  ! AutoModel 加载失败: {e}")
            print("  - 尝试导入自定义模型类...")

            import sys
            from pathlib import Path

            # 添加模型路径到 sys.path
            model_dir = Path(self.model_path)
            if str(model_dir) not in sys.path:
                sys.path.insert(0, str(model_dir))

            try:
                # 尝试导入常见的自定义模型类名
                model_class = None
                for module_name in ['modeling_vlm', 'modeling']:
                    try:
                        module = __import__(module_name)
                        # 尝试常见的类名
                        for class_name in ['MultiModalityCausalLM', 'VLChatProcessor', 'VisionLanguageModel']:
                            if hasattr(module, class_name):
                                model_class = getattr(module, class_name)
                                print(f"  - 找到自定义类: {module_name}.{class_name}")
                                break
                        if model_class:
                            break
                    except ImportError:
                        continue

                if model_class:
                    self.model = model_class.from_pretrained(
                        self.model_path,
                        **model_kwargs
                    )
                else:
                    raise RuntimeError("无法找到合适的模型类")
            finally:
                # 清理 sys.path
                if str(model_dir) in sys.path:
                    sys.path.remove(str(model_dir))

        # 如果不使用 device_map，需要手动移动到设备
        if not device_available or "device_map" not in model_kwargs:
            self.model = self.model.to(self.device)

        self.model.eval()
        print("模型加载完成！")

    def _format_conversation(
        self,
        messages: List[Dict[str, str]],
        images: Optional[List[Image.Image]] = None
    ) -> str:
        """
        格式化对话（处理图像占位符）

        Args:
            messages: 消息列表（可能包含 <image> 占位符）
            images: 图像列表

        Returns:
            格式化后的文本
        """
        # 如果有 processor 且支持 apply_chat_template
        if self.processor and hasattr(self.processor, 'apply_chat_template'):
            try:
                # 某些 VL 模型的 processor 支持直接传入 images
                return self.processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
            except Exception as e:
                print(f"  processor.apply_chat_template 失败: {e}")

        # 回退到 tokenizer
        if self.tokenizer and hasattr(self.tokenizer, 'apply_chat_template'):
            try:
                return self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
            except Exception as e:
                print(f"  tokenizer.apply_chat_template 失败: {e}")

        # 手动格式化
        prompt = ""
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                prompt += f"<|system|>\n{content}\n"
            elif role == "user":
                prompt += f"<|user|>\n{content}\n"
            elif role == "assistant":
                prompt += f"<|assistant|>\n{content}\n"
        prompt += "<|assistant|>\n"

        return prompt

    async def generate(
        self,
        messages: List[Dict[str, Any]],
        config: GenConfig,
        images: Optional[List[Image.Image]] = None
    ) -> str:
        """非流式生成"""
        # 格式化对话
        prompt = self._format_conversation(messages, images)

        # 处理输入
        if images and self.processor:
            # 使用 processor 处理文本+图像
            inputs = self.processor(
                text=prompt,
                images=images,
                return_tensors="pt",
                padding=True
            ).to(self.model.device)
        else:
            # 仅文本
            if self.processor:
                inputs = self.processor(text=prompt, return_tensors="pt").to(self.model.device)
            else:
                inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        # 生成
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=config.max_new_tokens,
                temperature=config.temperature,
                top_p=config.top_p,
                top_k=config.top_k,
                repetition_penalty=config.repetition_penalty,
                do_sample=config.do_sample,
                pad_token_id=self.tokenizer.pad_token_id if self.tokenizer else None,
                eos_token_id=self.tokenizer.eos_token_id if self.tokenizer else None,
            )

        # 解码
        if self.processor and hasattr(self.processor, 'batch_decode'):
            response_text = self.processor.batch_decode(
                outputs[:, inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            )[0]
        else:
            response_text = self.tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True
            )

        return response_text

    async def generate_stream(
        self,
        messages: List[Dict[str, Any]],
        config: GenConfig,
        images: Optional[List[Image.Image]] = None
    ) -> AsyncIterator[str]:
        """流式生成"""
        # 格式化对话
        prompt = self._format_conversation(messages, images)

        # 处理输入
        if images and self.processor:
            inputs = self.processor(
                text=prompt,
                images=images,
                return_tensors="pt",
                padding=True
            ).to(self.model.device)
        else:
            if self.processor:
                inputs = self.processor(text=prompt, return_tensors="pt").to(self.model.device)
            else:
                inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        # 创建 streamer
        tokenizer_for_stream = self.tokenizer if self.tokenizer else self.processor.tokenizer
        streamer = TextIteratorStreamer(
            tokenizer_for_stream,
            skip_prompt=True,
            skip_special_tokens=True
        )

        # 在后台线程中生成
        generation_kwargs = dict(
            **inputs,
            max_new_tokens=config.max_new_tokens,
            temperature=config.temperature,
            top_p=config.top_p,
            top_k=config.top_k,
            repetition_penalty=config.repetition_penalty,
            do_sample=config.do_sample,
            streamer=streamer,
            pad_token_id=tokenizer_for_stream.pad_token_id,
            eos_token_id=tokenizer_for_stream.eos_token_id,
        )

        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()

        # 流式返回
        for text in streamer:
            if text:
                yield text
                await asyncio.sleep(0)

        thread.join()

    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        return {
            "type": "multimodal",
            "model_path": self.model_path,
            "device": self.device,
            "supports_images": True,
            "has_processor": self.processor is not None,
        }

    def supports_images(self) -> bool:
        return True
