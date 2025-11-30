"""服务器启动脚本"""
import argparse
import torch
import uvicorn
import sys
import os

# 添加父目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from multimodal_server.handlers import TextModelHandler, MultimodalModelHandler
from multimodal_server.server import MultimodalServer
from multimodal_server.utils.config_utils import detect_model_type, ModelType


def main():
    parser = argparse.ArgumentParser(description="Multimodal Server - 支持文本和多模态模型")

    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="模型路径"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="服务器地址（默认: 0.0.0.0）"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="服务器端口（默认: 8000）"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cuda", "cpu"],
        help="设备类型（默认: 自动检测）"
    )
    parser.add_argument(
        "--force-type",
        type=str,
        default=None,
        choices=["text", "multimodal"],
        help="强制指定模型类型（默认: 自动检测）"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("Multimodal Server - 启动中...")
    print("=" * 60)

    # 检测模型类型
    if args.force_type:
        print(f"\n强制使用模型类型: {args.force_type}")
        model_type = ModelType.MULTIMODAL if args.force_type == "multimodal" else ModelType.TEXT
    else:
        print("\n正在检测模型类型...")
        model_type, config = detect_model_type(args.model_path)

    # 创建处理器
    print(f"\n创建模型处理器...")
    if model_type == ModelType.MULTIMODAL:
        print("  → 使用 MultimodalModelHandler")
        handler = MultimodalModelHandler(args.model_path, args.device)
    else:
        print("  → 使用 TextModelHandler")
        handler = TextModelHandler(args.model_path, args.device)

    # 加载模型
    print(f"\n开始加载模型...")
    handler.load_model()

    # 创建服务器
    print(f"\n创建服务器...")
    server = MultimodalServer(handler)

    # 打印信息
    print("\n" + "=" * 60)
    print("✓ 服务器准备就绪！")
    print("=" * 60)
    print(f"模型路径:     {args.model_path}")
    print(f"模型类型:     {model_type.value}")
    print(f"设备:         {args.device}")
    print(f"支持图像:     {'是' if handler.supports_images() else '否'}")
    print(f"服务器地址:   http://{args.host}:{args.port}")
    print("=" * 60)
    print("\nAPI 端点:")
    print(f"  - GET  http://{args.host}:{args.port}/v1/models")
    print(f"  - POST http://{args.host}:{args.port}/v1/chat/completions")
    print(f"  - GET  http://{args.host}:{args.port}/health")
    print("\n按 Ctrl+C 停止服务器\n")

    # 启动服务器
    uvicorn.run(
        server.get_app(),
        host=args.host,
        port=args.port,
        log_level="info"
    )


if __name__ == "__main__":
    main()
