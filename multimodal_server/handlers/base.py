"""基础模型处理器抽象类"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any, AsyncIterator, Optional
from dataclasses import dataclass


@dataclass
class GenerationConfig:
    """生成配置"""
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.05
    do_sample: bool = True
    stream: bool = False


class BaseModelHandler(ABC):
    """
    模型处理器基类

    定义统一的接口，用于处理文本和多模态模型
    """

    def __init__(self, model_path: str, device: str = "cuda"):
        """
        初始化模型处理器

        Args:
            model_path: 模型路径
            device: 设备 (cuda/cpu)
        """
        import os
        # 规范化路径，处理 Windows 路径问题
        self.model_path = os.path.normpath(model_path)
        self.device = device
        self.model = None
        self.tokenizer = None

    @abstractmethod
    def load_model(self):
        """加载模型和分词器"""
        pass

    @abstractmethod
    async def generate(
        self,
        messages: List[Dict[str, Any]],
        config: GenerationConfig,
        images: Optional[List[Any]] = None
    ) -> str:
        """
        非流式生成

        Args:
            messages: 消息列表
            config: 生成配置
            images: 图像列表（可选，多模态模型使用）

        Returns:
            生成的文本
        """
        pass

    @abstractmethod
    async def generate_stream(
        self,
        messages: List[Dict[str, Any]],
        config: GenerationConfig,
        images: Optional[List[Any]] = None
    ) -> AsyncIterator[str]:
        """
        流式生成

        Args:
            messages: 消息列表
            config: 生成配置
            images: 图像列表（可选，多模态模型使用）

        Yields:
            生成的文本片段
        """
        pass

    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """
        获取模型信息

        Returns:
            模型信息字典
        """
        pass

    def supports_images(self) -> bool:
        """
        是否支持图像输入

        Returns:
            True 表示支持图像
        """
        return False

    def __repr__(self):
        return f"{self.__class__.__name__}(model_path={self.model_path}, device={self.device})"
