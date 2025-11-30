"""模型配置检测工具"""
import json
import os
from enum import Enum
from pathlib import Path
from typing import Dict, Any


class ModelType(Enum):
    """模型类型枚举"""
    TEXT = "text"           # 纯文本模型
    MULTIMODAL = "multimodal"  # 多模态模型
    UNKNOWN = "unknown"


# 已知的多模态模型类型
MULTIMODAL_MODEL_TYPES = {
    "multi_modality",    # DeepSeek-VL
    "qwen2_vl",          # Qwen2-VL
    "llava",             # LLaVA
    "llava_next",        # LLaVA-NeXT
    "intern_vl",         # InternVL
    "cogvlm",            # CogVLM
    "minicpmv",          # MiniCPM-V
}


def detect_model_type(model_path: str) -> tuple[ModelType, Dict[str, Any]]:
    """
    自动检测模型类型

    Args:
        model_path: 模型路径

    Returns:
        (model_type, config_dict)
    """
    config_path = Path(model_path) / "config.json"

    if not config_path.exists():
        print(f"警告: 未找到 config.json，假定为文本模型")
        return ModelType.TEXT, {}

    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)

        model_type_str = config.get("model_type", "").lower()

        # 判断是否为多模态模型
        if model_type_str in MULTIMODAL_MODEL_TYPES:
            print(f"检测到多模态模型: {model_type_str}")
            return ModelType.MULTIMODAL, config

        # 额外检查：是否有 vision_config
        if "vision_config" in config or "visual" in config:
            print(f"检测到视觉配置，判定为多模态模型")
            return ModelType.MULTIMODAL, config

        print(f"检测到文本模型: {model_type_str}")
        return ModelType.TEXT, config

    except Exception as e:
        print(f"读取配置文件失败: {e}，假定为文本模型")
        return ModelType.TEXT, {}


def get_model_info(model_path: str) -> Dict[str, Any]:
    """
    获取模型详细信息

    Args:
        model_path: 模型路径

    Returns:
        模型信息字典
    """
    model_type, config = detect_model_type(model_path)

    return {
        "model_type": model_type,
        "model_name": os.path.basename(model_path),
        "config": config,
        "supports_multimodal": model_type == ModelType.MULTIMODAL,
    }
