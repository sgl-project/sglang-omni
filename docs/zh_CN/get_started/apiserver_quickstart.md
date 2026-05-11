# API 服务器快速入门

本页面展示了从“克隆仓库”到“让 API 服务器响应请求”的最短路径。

`sglang-omni` 在其多阶段流水线运行时之上，暴露了一个兼容 OpenAI 格式的 API 服务器。该服务器是以下功能的主要 HTTP 入口点：

- 聊天补全 (chat completions)
- 流式响应 (streaming responses)
- 模型列表获取 (model listing)
- 健康检查 (health checks)
- 文本转语音 (text-to-speech)

如果你想了解内部设计而不是使用流程，请参阅 [API 服务器设计](../developer_reference/apiserver_design.md)。

## 这个服务器是什么

API 服务器是 HTTP 客户端与内部流水线运行时之间的适配器：

`HTTP 请求` → `FastAPI 应用` → `Client 对象` → `Coordinator (协调器)` → `Pipeline 阶段`

换句话说，它不直接运行模型逻辑。它的工作是将 OpenAI 风格的请求转换为内部请求，并将结果格式化回 HTTP 响应。

## 启动服务器

安装后可用的命令行入口点是 `sgl-omni`。

启动服务器最简单的方法是提供一个模型路径，让 `sglang-omni` 为你自动构建流水线配置：

```bash
sgl-omni serve \
  --model-path Qwen/Qwen3-Omni-30B-A3B-Instruct \
  --host 0.0.0.0 \
  --port 8000
```

最常用的参数包括：

- `--model-path`: Hugging Face 模型 ID 或本地模型目录
- `--host`: 绑定地址，默认为 `0.0.0.0`
- `--port`: 绑定端口，默认为 `8000`
- `--model-name`: 覆盖 `/v1/models` 返回的模型名称
- `--log-level`: 服务器进程的日志级别

如果你已经有一个流水线配置文件，你也可以传递 `--config path/to/config.yaml`。在当前的 CLI 中，即使提供了 `--config`，`--model-path` 仍然是必需的。

## 验证服务器是否工作

### 健康检查

```bash
curl -s http://localhost:8000/health
```

响应示例：

```json
{
  "status": "healthy",
  "running": true
}
```

服务器会返回：

- `200`: 当运行时健康时
- `503`: 当 HTTP 服务器已启动，但底层运行时报告处于不健康状态时

### 获取已部署模型列表

```bash
curl -s http://localhost:8000/v1/models
```

该端点返回一个包含单一模型的列表。模型 ID 如果你设置了 `--model-name` 则取该值，否则取自流水线名称。

## 发送最简聊天请求

核心端点是 `POST /v1/chat/completions`。

```bash
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-omni",
    "messages": [
      {"role": "user", "content": "Hello!"}
    ],
    "max_tokens": 128,
    "stream": false
  }'
```

响应格式遵循 OpenAI 聊天补全的结构。通常情况下，生成的文本位于 `choices[0].message.content` 中。

除了 `model` 和 `messages` 之外，最常用的请求字段还有：

- `temperature`
- `top_p`
- `max_tokens`
- `stop`
- `seed`
- `stream`

## 流式与多模态请求

### 流式输出 (Streaming)

将 `stream` 设置为 `true` 即可接收服务器发送事件 (SSE)：

```bash
curl -N http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-omni",
    "messages": [
      {"role": "user", "content": "Write a short greeting."}
    ],
    "stream": true
  }'
```

这里有几个细节需要注意：

- 响应类型为 `text/event-stream`
- 第一个数据块（chunk）可能只包含 `role="assistant"`
- 数据流以 `data: [DONE]` 结束
- `usage`（用量信息）附加在最后一个补全数据块上

### 多模态输入与输出

`sglang-omni` 通过几个额外的字段扩展了标准的 OpenAI 聊天请求结构：

- `images` (图像)
- `audios` (音频)
- `videos` (视频)
- `modalities` (模态类型)
- `audio` (音频配置)
- `stage_sampling` (阶段采样)
- `stage_params` (阶段参数)

例如，一个输入视频并要求输出文本的请求如下所示：

```bash
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-omni",
    "messages": [
      {"role": "user", "content": "What is happening in this video?"}
    ],
    "videos": ["/absolute/path/to/demo.mp4"],
    "modalities": ["text"],
    "max_tokens": 128,
    "stream": false
  }'
```

`videos`、`images` 和 `audios` 字段均接受本地文件路径或 HTTP(S) URL。

## 文本转语音 (Text-to-Speech)

服务器还暴露了 `POST /v1/audio/speech` 端点。

```bash
curl -s http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-omni",
    "input": "Hello from SGLang-Omni.",
    "voice": "default",
    "response_format": "wav"
  }' \
  -o speech.wav
```

两点注意事项：

- 响应主体是音频字节流，而不是 JSON
- 实际输出的格式可能与请求的格式不同，前提是编码器回退到了其他支持的编解码器

## 常见错误

当请求失败时，服务器会返回标准的 HTTP 错误代码：

- `400 Bad Request`: 请求体格式错误或参数无效
- `500 Internal Server Error`: 生成过程中的运行时错误（请检查服务器日志以获取详情）
- `503 Service Unavailable`: 运行时不健康（可通过 `/health` 验证）

如果你看到 500 错误，请检查服务器日志以查看完整的追踪信息 (traceback)。常见问题包括：
- 不支持的媒体格式
- 内存不足 (OOM) 错误
- 模型文件丢失

## 下一步阅读

- [API 服务器设计](../developer_reference/apiserver_design.md)
- [架构设计](../developer_reference/architecture.md)
