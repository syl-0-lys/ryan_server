"""测试客户端 - 演示如何使用 multimodal_server"""
from openai import OpenAI
import sys


def test_text_chat():
    """测试纯文本对话"""
    print("\n" + "=" * 60)
    print("测试 1: 纯文本对话")
    print("=" * 60)

    client = OpenAI(
        base_url="http://localhost:8000/v1",
        api_key="dummy"
    )

    try:
        response = client.chat.completions.create(
            model="model",
            messages=[
                {"role": "user", "content": "你好，请简单介绍一下自己"}
            ],
            max_tokens=100
        )

        print("\n✓ 请求成功!")
        print(f"回复: {response.choices[0].message.content}")

    except Exception as e:
        print(f"\n✗ 请求失败: {e}")


def test_stream_chat():
    """测试流式对话"""
    print("\n" + "=" * 60)
    print("测试 2: 流式对话")
    print("=" * 60)

    client = OpenAI(
        base_url="http://localhost:8000/v1",
        api_key="dummy"
    )

    try:
        stream = client.chat.completions.create(
            model="model",
            messages=[
                {"role": "user", "content": "用一句话介绍人工智能"}
            ],
            max_tokens=50,
            stream=True
        )

        print("\n✓ 流式生成:")
        print("回复: ", end="", flush=True)

        for chunk in stream:
            if chunk.choices[0].delta.content:
                print(chunk.choices[0].delta.content, end="", flush=True)

        print("\n")

    except Exception as e:
        print(f"\n✗ 请求失败: {e}")


def test_multimodal_chat():
    """测试多模态对话（图像 + 文本）"""
    print("\n" + "=" * 60)
    print("测试 3: 多模态对话（图像 + 文本）")
    print("=" * 60)

    client = OpenAI(
        base_url="http://localhost:8000/v1",
        api_key="dummy"
    )

    # 使用一个公开的测试图像
    image_url = "https://picsum.photos/200"

    try:
        response = client.chat.completions.create(
            model="model",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": "描述这张图片"},
                    {"type": "image_url", "image_url": {"url": image_url}}
                ]
            }],
            max_tokens=100
        )

        print(f"\n✓ 请求成功!")
        print(f"图像 URL: {image_url}")
        print(f"回复: {response.choices[0].message.content}")

    except Exception as e:
        print(f"\n✗ 请求失败: {e}")
        print("提示: 如果是文本模型，此测试会失败（正常现象）")


def test_models_list():
    """测试模型列表"""
    print("\n" + "=" * 60)
    print("测试 4: 获取模型列表")
    print("=" * 60)

    client = OpenAI(
        base_url="http://localhost:8000/v1",
        api_key="dummy"
    )

    try:
        models = client.models.list()

        print("\n✓ 请求成功!")
        for model in models.data:
            print(f"\n模型信息:")
            print(f"  ID: {model.id}")
            print(f"  类型: {model.object}")
            # 注意：OpenAI SDK 可能不会返回自定义字段，需要直接请求 API

    except Exception as e:
        print(f"\n✗ 请求失败: {e}")


def main():
    print("\n" + "=" * 60)
    print("Multimodal Server - 测试客户端")
    print("=" * 60)
    print("\n确保服务器已启动: http://localhost:8000")
    print("按任意键开始测试...")
    input()

    # 运行测试
    test_models_list()
    test_text_chat()
    test_stream_chat()

    # 询问是否测试多模态
    print("\n是否测试多模态功能？(如果是文本模型请跳过) [y/N]: ", end="")
    if input().lower() == 'y':
        test_multimodal_chat()

    print("\n" + "=" * 60)
    print("测试完成!")
    print("=" * 60)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n测试已取消")
        sys.exit(0)
