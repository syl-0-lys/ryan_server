# Multimodal Server

基于 TensorRT-LLM 设计的多模态服务器，支持文本和视觉-语言模型。

## 功能特性

✅ **自动模型类型检测**
- 从 `config.json` 自动识别模型类型
- 支持文本模型和多模态模型

✅ **OpenAI 兼容 API**
- `/v1/chat/completions` - 聊天补全
- `/v1/models` - 模型列表
- 支持流式和非流式生成

✅ **多模态支持**
- 图像 + 文本输入
- 支持 HTTP URL、Data URL、本地文件
- 自动图像下载和处理

✅ **DeepSeek-R1 推理解析**
- 自动解析 `<think>` 标签
- 分离推理内容和最终答案

## 架构设计

```
multimodal_server/
├── handlers/              # 模型处理器
│   ├── base.py           # 抽象基类
│   ├── text_handler.py   # 文本模型处理器
│   └── multimodal_handler.py  # 多模态处理器
├── utils/                # 工具模块
│   ├── config_utils.py   # 模型类型检测
│   ├── message_parser.py # 消息解析
│   └── image_utils.py    # 图像处理
├── server.py             # FastAPI 服务器
└── run.py                # 启动脚本
```

## 快速开始

### 1. 启动服务器

```bash
# 文本模型（自动检测）
python multimodal_server/run.py --model-path C:\syl_file\DeepSeek-R1-Distill-Qwen-1.5B

# 多模态模型（自动检测）
python multimodal_server/run.py --model-path C:\syl_file\DeepSeek-VL

# 指定端口和设备
python multimodal_server/run.py \
    --model-path C:\syl_file\DeepSeek-VL \
    --port 8000 \
    --device cuda
```

### 2. 测试 API

#### 纯文本对话
```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="dummy"
)

response = client.chat.completions.create(
    model="model",
    messages=[
        {"role": "user", "content": "你好，介绍一下自己"}
    ]
)

print(response.choices[0].message.content)
```

#### 图像 + 文本（多模态）
```python
response = client.chat.completions.create(
    model="model",
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "描述这张图片"},
            {"type": "image_url", "image_url": {"url": "http://example.com/image.jpg"}}
        ]
    }]
)

print(response.choices[0].message.content)
```

#### 流式生成
```python
stream = client.chat.completions.create(
    model="model",
    messages=[{"role": "user", "content": "讲个故事"}],
    stream=True
)

for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
```

## 支持的模型类型

### 文本模型
- DeepSeek-R1 系列
- Qwen 系列
- LLaMA 系列
- 其他 Causal LM 模型

### 多模态模型
- DeepSeek-VL
- Qwen2-VL
- LLaVA
- InternVL
- CogVLM
- MiniCPM-V

## API 参数

### `/v1/chat/completions`

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `messages` | List | 必填 | 消息列表 |
| `stream` | bool | false | 是否流式生成 |
| `max_tokens` | int | 512 | 最大生成 token 数 |
| `temperature` | float | 0.7 | 温度参数 |
| `top_p` | float | 0.9 | Top-p 采样 |
| `top_k` | int | 50 | Top-k 采样 |

### 消息格式

#### 纯文本
```json
{
  "messages": [
    {"role": "user", "content": "Hello"}
  ]
}
```

#### 多模态（图像 + 文本）
```json
{
  "messages": [{
    "role": "user",
    "content": [
      {"type": "text", "text": "描述图片"},
      {"type": "image_url", "image_url": {"url": "http://..."}}
    ]
  }]
}
```

## 依赖项

```bash
pip install fastapi uvicorn transformers torch pillow httpx
```

## 与 simple_server.py 的区别

| 特性 | simple_server.py | multimodal_server |
|------|------------------|-------------------|
| 模型类型支持 | 仅文本 | 文本 + 多模态 |
| 自动检测 | ❌ | ✅ |
| 图像输入 | ❌ | ✅ |
| 模块化设计 | ❌ | ✅ |
| 可扩展性 | 低 | 高 |

## 故障排查

### Q: 提示 "模型不支持图像输入"
**A:** 确认模型是多模态模型，或使用 `--force-type multimodal` 强制指定。

### Q: 图像下载失败
**A:** 检查网络连接，或使用本地文件路径 `file:///path/to/image.jpg`。

### Q: CPU 加载太慢
**A:** 使用 GPU (`--device cuda`) 或使用量化模型。

## 贡献

欢迎提交 Issue 和 Pull Request！

## 许可证

MIT License
