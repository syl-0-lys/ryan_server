"""图像处理工具"""
import base64
import io
import re
from typing import List, Union, Optional
from PIL import Image
import aiohttp


async def download_image(image_url: str, timeout: float = 30.0) -> Image.Image:
    """
    异步下载图像

    支持格式:
    - HTTP/HTTPS URL: http://example.com/image.jpg
    - Data URL: data:image/jpeg;base64,/9j/4AAQ...
    - 本地文件: file:///path/to/image.jpg 或 /path/to/image.jpg

    Args:
        image_url: 图像 URL
        timeout: 超时时间（秒）

    Returns:
        PIL Image 对象
    """
    # 处理 data URL
    if image_url.startswith("data:"):
        match = re.match(r"data:image/[^;]+;base64,(.+)", image_url)
        if match:
            image_data = base64.b64decode(match.group(1))
            return Image.open(io.BytesIO(image_data))
        else:
            raise ValueError(f"不支持的 data URL 格式: {image_url[:50]}...")

    # 处理本地文件
    if image_url.startswith("file://"):
        image_url = image_url[7:]  # 移除 file:// 前缀

    if image_url.startswith("/") or image_url[1:3] == ":\\":  # Unix 路径或 Windows 路径
        return Image.open(image_url)

    # 处理 HTTP/HTTPS URL
    if image_url.startswith(("http://", "https://")):
        async with aiohttp.ClientSession() as session:
            async with session.get(image_url) as response:
                response.raise_for_status()
                content = await response.read()
                return Image.open(io.BytesIO(content))

    raise ValueError(f"不支持的 URL 格式: {image_url}")


async def process_images(image_urls: List[str]) -> List[Image.Image]:
    """
    批量处理图像

    Args:
        image_urls: 图像 URL 列表

    Returns:
        PIL Image 对象列表
    """
    images = []
    for url in image_urls:
        try:
            image = await download_image(url)
            # 转换为 RGB（如果是 RGBA 或其他格式）
            if image.mode != "RGB":
                image = image.convert("RGB")
            images.append(image)
        except Exception as e:
            print(f"警告: 下载图像失败 ({url}): {e}")
            # 创建占位图像（可选）
            # images.append(Image.new("RGB", (224, 224), color='gray'))
            raise

    return images


def resize_image(image: Image.Image, max_size: int = 1024) -> Image.Image:
    """
    调整图像大小（保持宽高比）

    Args:
        image: PIL Image
        max_size: 最大边长

    Returns:
        调整后的图像
    """
    width, height = image.size

    if width <= max_size and height <= max_size:
        return image

    if width > height:
        new_width = max_size
        new_height = int(height * max_size / width)
    else:
        new_height = max_size
        new_width = int(width * max_size / height)

    return image.resize((new_width, new_height), Image.Resampling.LANCZOS)
