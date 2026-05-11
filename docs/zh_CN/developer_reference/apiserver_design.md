# API 服务器设计 (API Server Design)

本页面从最有助于代码维护的层面来解释 API 服务器：它在系统中的位置、哪些文件最重要，以及请求是如何映射到运行时的。

如果你只是想启动并调用服务器，请先阅读 [API 服务器快速入门](../get_started/apiserver_quickstart.md)。

## 在系统中的角色 (Role in the System)

API 服务器是位于 `sglang-omni` 流水线运行时之上的一层外部协议层。

从高层来看，请求的处理路径为：

`CLI / Python 入口点` → `PipelineConfig (流水线配置)` → `Pipeline Startup (流水线启动)` → `Coordinator (协调器)` → `Client (客户端)` → `FastAPI`

这种拆分保持了职责的清晰：

- 流水线运行时负责编排和执行
- `Client` 层负责提交请求并组装结果
- API 服务器负责在 HTTP/OpenAI 风格的数据包与内部抽象之间进行翻译转换

## 核心文件 (Key Files)

对于当前的服务器实现来说，以下是最重要的几个文件。

| 文件 | 作用 |
| --- | --- |
| `sglang_omni/serve/openai_api.py` | 定义 FastAPI 应用、路由、请求转换以及响应格式化 |
| `sglang_omni/serve/protocol.py` | 定义请求和响应的数据模式 (schemas) |
| `sglang_omni/serve/launcher.py` | 编译流水线、启动运行时、挂载应用并运行 Uvicorn |
| `sglang_omni/client/client.py` | 向协调器提交请求并聚合文本、音频及流式结果 |
| `sglang_omni/cli/serve.py` | 定义了当前 `sgl-omni serve` 的命令行界面 (CLI) |

如果你需要追踪某个端点 (endpoint) 的行为，通常从 `openai_api.py` 和 `client.py` 开始看是最好的选择。

## `create_app()` vs `launch_server()`

这是服务端代码中最重要的区分。

### `create_app(client, model_name=...)`

`create_app()` 仅仅构建 FastAPI 应用并注册核心路由。

它**不会**执行以下操作：

- 编译流水线
- 启动运行时
- 创建协调器
- 挂载性能分析 (profiling) 路由
- 运行 Uvicorn

当你已经拥有一个存活的 `Client` 实例，并且想自己嵌入 HTTP 层时，请使用它。

### `launch_server(pipeline_config, ...)`

`launch_server()` 代表了完整的内置服务器生命周期。

它会：

- 编译流水线配置
- 启动 runner
- 创建 `Client`
- 创建 FastAPI 应用
- 挂载性能分析路由
- 运行 Uvicorn
- 在关机时停止运行时

当你需要标准的、开箱即用的服务器启动流程时，请使用它。

## 路由接口 (Route Surface)

当前的服务器暴露了以下主要路由：

| 方法 | 路径 | 备注 |
| --- | --- | --- |
| `GET` | `/health` | 返回来自 `client.health()` 的健康状态 |
| `GET` | `/v1/models` | 返回当前活跃流水线的单一模型列表 |
| `POST` | `/v1/chat/completions` | 聊天补全，支持流式传输和可选的音频输出 |
| `POST` | `/v1/audio/speech` | 文本转语音 (TTS) |
| `POST` | `/start_profile` | 由内置的 launcher 自动添加 |
| `POST` | `/stop_profile` | 由内置的 launcher 自动添加 |

性能分析路由仅在通过 `launch_server()` 启动服务器时才会存在。

## 请求映射 (Request Mapping)

服务器并不会将 OpenAI 风格的请求体直接透传给运行时。它会首先将它们转换为内部请求对象。

### 聊天请求 (Chat requests)

`ChatCompletionRequest` 包含了标准的 OpenAI 风格字段，例如：

- `model`
- `messages`
- `temperature`
- `top_p`
- `max_tokens`
- `stop`
- `seed`
- `stream`

它还包含 `sglang-omni` 的独有扩展字段，例如：

- `images`
- `audios`
- `videos`
- `modalities`
- `audio`
- `stage_sampling`
- `stage_params`
- `request_id`

### 转换为 `GenerateRequest`

`openai_api.py` 中的 `_build_chat_generate_request()` 是关键的转换点。它负责：

- 归一化 stop 序列
- 构建 `SamplingParams` (采样参数)
- 将聊天消息转换为内部的 `Message` 对象
- 映射各阶段自定义的采样参数 (per-stage sampling overrides)
- 将媒体输入和音频配置存储在请求的元数据 (metadata) 中
- 将 `modalities` 复制到 `output_modalities` 中

从这一点开始，运行时处理的就全是内部的 `GenerateRequest` 对象了，而不是原始的 OpenAI 风格数据包。

## 响应路径 (Response Paths)

### 非流式聊天 (Non-streaming chat)

对于非流式的聊天请求，路径大致为：

`聊天请求` → `Client.completion()` → OpenAI 风格的 JSON 响应

`Client.completion()` 会聚合：

- 文本片段
- 音频数据块
- 最终的用量信息 (usage)
- 最终的结束原因 (finish reason)

如果存在音频数据，它会在返回给 API 层之前被进行 base64 编码。

### 流式聊天 (Streaming chat)

对于流式聊天请求，服务器会发出 SSE (服务器发送事件)。

当前重要的语义细节包括：

- 第一个数据块可能仅包含 `role="assistant"`
- 文本和音频作为独立的增量 (deltas) 被发送
- 最后一个完成数据块包含了 `finish_reason`
- 数据流以 `data: [DONE]` 结束
- `usage`（用量信息）附加在最后一个完成数据块上

### 语音 / TTS

语音路由复用了相同的内部请求路径，而不是引入一套独立的上菜 (serving) 技术栈。

`CreateSpeechRequest` 会被转换为 `GenerateRequest`，并附带以下属性：

- `output_modalities=["audio"]`
- 元数据中设置 `task="tts"`
- TTS 特有的参数存储在 `tts_params` 中

然后，`Client.speech()` 会收集音频数据块，对其进行编码，并将原始的音频字节流返回给 HTTP 层。
