"""OpenAI 消息格式解析器"""
from typing import List, Dict, Any, Tuple, Optional


def parse_chat_messages(messages: List[Dict[str, Any]]) -> Tuple[List[Dict[str, str]], List[str]]:
    """
    解析 OpenAI 格式的聊天消息

    支持两种格式:
    1. 纯文本: {"role": "user", "content": "Hello"}
    2. 多模态: {"role": "user", "content": [
           {"type": "text", "text": "描述图像"},
           {"type": "image_url", "image_url": {"url": "..."}}
       ]}

    Args:
        messages: OpenAI 格式消息列表

    Returns:
        (text_messages, image_urls)
        - text_messages: 纯文本消息列表 [{"role": "user", "content": "..."}]
        - image_urls: 图像 URL 列表
    """
    text_messages = []
    image_urls = []

    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")

        # 情况 1: 纯文本消息
        if isinstance(content, str):
            text_messages.append({
                "role": role,
                "content": content
            })
            continue

        # 情况 2: 多模态消息（列表格式）
        if isinstance(content, list):
            text_parts = []

            for part in content:
                part_type = part.get("type", "")

                if part_type == "text":
                    text_parts.append(part.get("text", ""))

                elif part_type == "image_url":
                    image_url_data = part.get("image_url", {})
                    if isinstance(image_url_data, dict):
                        url = image_url_data.get("url", "")
                    else:
                        url = image_url_data  # 直接是字符串

                    if url:
                        image_urls.append(url)
                        # 添加图像占位符
                        text_parts.append("<image>")

                elif part_type == "video_url":
                    # 视频支持（可选）
                    print("警告: 当前不支持视频输入")

                else:
                    print(f"警告: 未知的内容类型: {part_type}")

            # 合并文本部分
            merged_text = " ".join(text_parts)
            text_messages.append({
                "role": role,
                "content": merged_text
            })
            continue

        # 其他情况：转为字符串
        text_messages.append({
            "role": role,
            "content": str(content)
        })

    return text_messages, image_urls


def format_prompt_with_template(
    messages: List[Dict[str, str]],
    tokenizer: Any,
    add_generation_prompt: bool = True
) -> str:
    """
    使用 tokenizer 的 chat_template 格式化提示

    Args:
        messages: 消息列表
        tokenizer: HuggingFace tokenizer
        add_generation_prompt: 是否添加生成提示

    Returns:
        格式化后的提示文本
    """
    # 优先使用 tokenizer 的 apply_chat_template
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=add_generation_prompt
            )
        except Exception as e:
            print(f"apply_chat_template 失败: {e}，使用手动格式化")

    # 备用方案：手动格式化
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

    if add_generation_prompt:
        prompt += "<|assistant|>\n"

    return prompt
