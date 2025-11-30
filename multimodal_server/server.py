"""FastAPI 服务器实现"""
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse
from typing import List, Dict, Any, Optional
import time
import json
import os

from .handlers import BaseModelHandler
from .utils.message_parser import parse_chat_messages
from .utils.image_utils import process_images
from .handlers.base import GenerationConfig


class MultimodalServer:
    """
    OpenAI 兼容的多模态服务器

    支持:
    - 文本模型 (TextModelHandler)
    - 多模态模型 (MultimodalModelHandler)
    """

    def __init__(self, handler: BaseModelHandler):
        """
        初始化服务器

        Args:
            handler: 模型处理器（TextModelHandler 或 MultimodalModelHandler）
        """
        self.handler = handler
        self.app = FastAPI(title="Multimodal Server")
        self.model_name = os.path.basename(handler.model_path)

        # 注册路由
        self._register_routes()

    def _register_routes(self):
        """注册 API 路由"""
        self.app.add_api_route("/v1/models", self.list_models, methods=["GET"])
        self.app.add_api_route("/v1/chat/completions", self.chat_completions, methods=["POST"])
        self.app.add_api_route("/health", self.health_check, methods=["GET"])

    async def health_check(self):
        """健康检查"""
        return JSONResponse({
            "status": "healthy",
            "model": self.model_name,
            "supports_images": self.handler.supports_images()
        })

    async def list_models(self):
        """列出可用模型"""
        return JSONResponse({
            "object": "list",
            "data": [{
                "id": self.model_name,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "user",
                "supports_images": self.handler.supports_images()
            }]
        })

    async def chat_completions(self, request: Request):
        """
        OpenAI 兼容的聊天补全接口

        支持:
        - 纯文本消息
        - 图像 + 文本消息（多模态模型）
        - 流式和非流式生成
        """
        req = await request.json()

        # 解析请求参数
        messages = req.get("messages", [])
        stream = req.get("stream", False)
        max_tokens = req.get("max_tokens", 512)
        temperature = req.get("temperature", 0.7)
        top_p = req.get("top_p", 0.9)
        top_k = req.get("top_k", 50)

        # 解析消息（提取文本和图像 URL）
        text_messages, image_urls = parse_chat_messages(messages)

        # 下载并处理图像
        images = None
        if image_urls:
            if not self.handler.supports_images():
                return JSONResponse(
                    status_code=400,
                    content={"error": "当前模型不支持图像输入"}
                )
            try:
                images = await process_images(image_urls)
                print(f"已处理 {len(images)} 张图像")
            except Exception as e:
                return JSONResponse(
                    status_code=400,
                    content={"error": f"图像处理失败: {str(e)}"}
                )

        # 生成配置
        config = GenerationConfig(
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            stream=stream
        )

        # 流式 vs 非流式
        if stream:
            return StreamingResponse(
                self._generate_stream(text_messages, config, images),
                media_type="text/event-stream"
            )
        else:
            return await self._generate_completion(text_messages, config, images)

    async def _generate_completion(
        self,
        messages: List[Dict[str, str]],
        config: GenerationConfig,
        images: Optional[List[Any]]
    ):
        """非流式生成"""
        try:
            result = await self.handler.generate(messages, config, images)

            # 处理结果（可能是 ReasoningParserResult 或 str）
            if hasattr(result, 'content') and hasattr(result, 'reasoning_content'):
                # DeepSeek-R1 格式
                message = {"role": "assistant"}
                if result.reasoning_content:
                    message["reasoning_content"] = result.reasoning_content
                if result.content:
                    message["content"] = result.content
                else:
                    message["content"] = ""
            else:
                # 普通格式
                message = {
                    "role": "assistant",
                    "content": str(result)
                }

            return JSONResponse({
                "id": f"chatcmpl-{int(time.time())}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": self.model_name,
                "choices": [{
                    "index": 0,
                    "message": message,
                    "finish_reason": "stop"
                }],
                "usage": {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0
                }
            })

        except Exception as e:
            import traceback
            traceback.print_exc()
            return JSONResponse(
                status_code=500,
                content={"error": f"生成失败: {str(e)}"}
            )

    async def _generate_stream(
        self,
        messages: List[Dict[str, str]],
        config: GenerationConfig,
        images: Optional[List[Any]]
    ):
        """流式生成"""
        request_id = f"chatcmpl-{int(time.time())}"

        try:
            async for chunk in self.handler.generate_stream(messages, config, images):
                # 处理结果（ReasoningParserResult 或 str）
                if hasattr(chunk, 'content') or hasattr(chunk, 'reasoning_content'):
                    # DeepSeek-R1 格式
                    delta = {}
                    if hasattr(chunk, 'reasoning_content') and chunk.reasoning_content:
                        delta["reasoning_content"] = chunk.reasoning_content
                    if hasattr(chunk, 'content') and chunk.content:
                        delta["content"] = chunk.content

                    if not delta:
                        continue
                else:
                    # 普通格式
                    delta = {"content": str(chunk)}

                chunk_data = {
                    "id": request_id,
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": self.model_name,
                    "choices": [{
                        "index": 0,
                        "delta": delta,
                        "finish_reason": None
                    }]
                }

                yield f"data: {json.dumps(chunk_data, ensure_ascii=False)}\n\n"

            # 结束标志
            final_chunk = {
                "id": request_id,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": self.model_name,
                "choices": [{
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop"
                }]
            }
            yield f"data: {json.dumps(final_chunk, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"

        except Exception as e:
            import traceback
            traceback.print_exc()
            error_chunk = {
                "id": request_id,
                "object": "error",
                "error": {
                    "message": str(e),
                    "type": "server_error",
                    "code": 500
                }
            }
            yield f"data: {json.dumps(error_chunk, ensure_ascii=False)}\n\n"

    def get_app(self) -> FastAPI:
        """获取 FastAPI 应用实例"""
        return self.app
