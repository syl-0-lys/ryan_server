"""DeepSeek-VL 专用处理器"""
import torch
from transformers import AutoModelForCausalLM, TextIteratorStreamer
from threading import Thread
from typing import List, Dict, Any, AsyncIterator, Optional
from PIL import Image
import asyncio

from .base import BaseModelHandler, GenerationConfig as GenConfig


class DeepSeekVLHandler(BaseModelHandler):
    """
    DeepSeek-VL 专用模型处理器

    DeepSeek-VL 使用特殊的调用方式：
    1. 使用 VLChatProcessor 处理对话
    2. 调用 model.prepare_inputs_embeds() 准备输入
    3. 使用 model.language_model.generate() 生成
    """

    def __init__(self, model_path: str, device: str = "cuda"):
        super().__init__(model_path, device)
        self.vl_chat_processor = None

    def load_model(self):
        """加载 DeepSeek-VL 模型"""
        print(f"正在加载 DeepSeek-VL 模型: {self.model_path}")

        # 导入 DeepSeek-VL 的模块
        import sys
        from pathlib import Path
        model_dir = Path(self.model_path)
        if str(model_dir) not in sys.path:
            sys.path.insert(0, str(model_dir))

        try:
            from processing_vlm import VLChatProcessor

            # 加载 VLChatProcessor
            print("  - 加载 VLChatProcessor...")
            self.vl_chat_processor = VLChatProcessor.from_pretrained(self.model_path)
            self.tokenizer = self.vl_chat_processor.tokenizer

            # 确保 tokenizer 有 pad_token
            if self.tokenizer.pad_token is None:
                print("  - 设置 pad_token...")
                if self.tokenizer.eos_token:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                else:
                    self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})

            # 加载模型
            print(f"  - 加载模型到 {self.device}...")
            device_available = self.device == "cuda" and torch.cuda.is_available()

            model_kwargs = {
                "trust_remote_code": True,
            }

            if device_available:
                print("  - 使用 bfloat16...")
                model_kwargs["torch_dtype"] = torch.bfloat16
            else:
                print("  - 使用 float32 (CPU)...")
                model_kwargs["torch_dtype"] = torch.float32

            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                **model_kwargs
            )

            # 显式转换模型数据类型以确保一致性
            if device_available:
                self.model = self.model.cuda()
            else:
                print("  - 转换模型为 float32...")
                self.model = self.model.to(self.device).float()

            self.model.eval()
            print("模型加载完成！")

        finally:
            # 清理 sys.path
            if str(model_dir) in sys.path:
                sys.path.remove(str(model_dir))

    def _prepare_conversation(
        self,
        messages: List[Dict[str, Any]],
        images: Optional[List[Image.Image]] = None
    ) -> List[Dict[str, Any]]:
        """
        将 OpenAI 格式的消息转换为 DeepSeek-VL 格式

        OpenAI 格式: [{"role": "user", "content": "..."}]
        DeepSeek-VL 格式: [{"role": "User", "content": "...", "images": [...]}]
        """
        conversation = []

        for msg in messages:
            role = msg["role"]
            content = msg["content"]

            # 转换角色名称
            if role == "user":
                vl_role = "User"
            elif role == "assistant":
                vl_role = "Assistant"
            elif role == "system":
                vl_role = "System"
            else:
                vl_role = role

            # 如果有图像，添加图像占位符
            if images:
                # 为每个图像添加占位符
                image_placeholders = ""
                for _ in images:
                    image_placeholders += "<image_placeholder>"

                # 将占位符添加到内容前面
                content = image_placeholders + content

                conversation.append({
                    "role": vl_role,
                    "content": content,
                    "images": images  # PIL Image 对象列表
                })
            else:
                conversation.append({
                    "role": vl_role,
                    "content": content
                })

        # 添加空的 Assistant 响应
        conversation.append({"role": "Assistant", "content": ""})

        return conversation

    async def generate(
        self,
        messages: List[Dict[str, Any]],
        config: GenConfig,
        images: Optional[List[Image.Image]] = None
    ) -> str:
        """非流式生成"""
        # 准备对话格式
        conversation = self._prepare_conversation(messages, images)

        # 使用 VLChatProcessor 处理
        prepare_inputs = self.vl_chat_processor(
            conversations=conversation,
            images=images or [],
            force_batchify=True
        ).to(self.model.device)

        # CPU 模式下，确保所有输入都是 float32
        if self.device == "cpu":
            # 转换 prepare_inputs 中的所有张量为 float32
            for key in prepare_inputs:
                if hasattr(prepare_inputs[key], 'float'):
                    prepare_inputs[key] = prepare_inputs[key].float()

        # 准备输入 embeddings
        inputs_embeds = self.model.prepare_inputs_embeds(**prepare_inputs)

        # 生成
        with torch.no_grad():
            outputs = self.model.language_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=prepare_inputs.attention_mask,
                pad_token_id=self.tokenizer.pad_token_id,
                bos_token_id=self.tokenizer.bos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                max_new_tokens=config.max_new_tokens,
                temperature=config.temperature if config.do_sample else 1.0,
                top_p=config.top_p if config.do_sample else 1.0,
                do_sample=config.do_sample,
                use_cache=True,
            )

        # 解码
        response_text = self.tokenizer.decode(
            outputs[0].cpu().tolist(),
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
        # 准备对话格式
        conversation = self._prepare_conversation(messages, images)

        # 使用 VLChatProcessor 处理
        prepare_inputs = self.vl_chat_processor(
            conversations=conversation,
            images=images or [],
            force_batchify=True
        ).to(self.model.device)

        # CPU 模式下，确保所有输入都是 float32
        if self.device == "cpu":
            # 转换 prepare_inputs 中的所有张量为 float32
            for key in prepare_inputs:
                if hasattr(prepare_inputs[key], 'float'):
                    prepare_inputs[key] = prepare_inputs[key].float()

        # 准备输入 embeddings
        inputs_embeds = self.model.prepare_inputs_embeds(**prepare_inputs)

        # 创建 streamer
        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True
        )

        # 在后台线程中生成
        generation_kwargs = dict(
            inputs_embeds=inputs_embeds,
            attention_mask=prepare_inputs.attention_mask,
            pad_token_id=self.tokenizer.pad_token_id,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            max_new_tokens=config.max_new_tokens,
            temperature=config.temperature if config.do_sample else 1.0,
            top_p=config.top_p if config.do_sample else 1.0,
            do_sample=config.do_sample,
            streamer=streamer,
            use_cache=True,
        )

        thread = Thread(target=self.model.language_model.generate, kwargs=generation_kwargs)
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
            "type": "deepseek-vl",
            "model_path": self.model_path,
            "device": self.device,
            "supports_images": True,
        }

    def supports_images(self) -> bool:
        return True
