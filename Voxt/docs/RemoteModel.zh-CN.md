# 远程模型

Voxt 支持分别为 `Remote ASR` 和 `Remote LLM` 选择独立的远程服务商。下面只介绍 app 当前内置的服务商与模型范围，方便你先了解各家定位；API Key、端点填写、模型选型建议等操作说明可以在后续章节继续补充。

## ASR 模型

### OpenAI Whisper

- 默认推荐：`whisper-1`
- 内置模型：`whisper-1`、`gpt-4o-mini-transcribe`、`gpt-4o-transcribe`
- 简介：适合通用多语言音频转写。Voxt 当前按文件转写方式接入，适合追求兼容性和稳定体验的用户。
- Voxt 说明：这一档也支持填写自定义模型 ID，以及 OpenAI-compatible 的转写 endpoint。

- [官网](https://openai.com/)
- [API 文档](https://developers.openai.com/api/reference/resources/audio)
- [key 网址](https://platform.openai.com/api-keys)

端点：`https://api.openai.com/v1/audio/transcriptions`
key: `$OPENAI_API_KEY`

兼容的自定义 endpoint 示例：

- [MOSI Studio MOSS Transcribe](https://studio.mosi.cn/docs/moss-transcribe)
  - endpoint：`https://studio.mosi.cn/api/v1/audio/transcriptions`
  - 模型：`moss-transcribe`、`moss-transcribe-diarize`
- [Groq Speech-to-Text](https://console.groq.com/docs/speech-to-text)
  - endpoint：`https://api.groq.com/openai/v1/audio/transcriptions`
  - 模型：`whisper-large-v3-turbo`、`whisper-large-v3`

重要说明：

- 在 Voxt 里，只有 `OpenAI Whisper` 这一档支持自定义 ASR 模型 ID。
- endpoint 需要填写完整的转写路径，不能只填 API 根地址。
- Voxt 当前在这条链路上统一使用文件上传转写。即使兼容服务本身支持 URL 或 Base64 输入，Voxt 也仍然走文件上传。
- 如果兼容服务返回了额外的结构化字段，例如 diarization 分段，Voxt 当前会读取转写文本，但不会把这些字段作为独立 UI 能力展示。

<img width="923" height="676" alt="image" src="https://github.com/user-attachments/assets/62be17c8-78f5-418e-a4ba-873d18d58f18" />

### Doubao ASR

- 默认推荐：`volc.seedasr.sauc.duration`
- 内置模型：`volc.seedasr.sauc.duration`、`volc.bigasr.sauc.duration`
- 简介：偏实时流式识别，中文场景友好，也适合中英混说。适合想要低延迟语音输入体验的用户。

- [官网](https://www.volcengine.com/)
- [豆包语音](https://console.volcengine.com/speech/app)
- [api 文档](https://www.volcengine.com/docs/6561/1354869?lang=zh)
- [模型开通](https://console.volcengine.com/ark/region:ark+cn-beijing/openManagement?LLM=%7B%7D&advancedActiveKey=model) 

<img width="1437" height="1049" alt="image" src="https://github.com/user-attachments/assets/a62a34a2-33f4-4ddd-b3ec-5bc507852a2b" />
<img width="1442" height="1059" alt="image" src="https://github.com/user-attachments/assets/188f5abd-7bec-4592-8aa9-68b5e9ee260d" />

1. 创建应用，选择模型
2. 点击左侧菜单 “豆包流式语音识别模型2.0”，复制 `APP ID` 和 `Access Token` 填入即可

<img width="939" height="683" alt="image" src="https://github.com/user-attachments/assets/18c4032e-b3e8-4ab5-92fe-9f6f0117ecd1" />

### GLM ASR

- 默认推荐：`glm-asr-1`
- 内置模型：`glm-asr-2512`、`glm-asr-1`
- 简介：智谱 GLM 体系下的语音识别模型，适合已经在使用 GLM 服务，想统一放在同一服务商管理的用户。

- [官网](https://bigmodel.cn)
- [key 管理](https://bigmodel.cn/usercenter/proj-mgmt/apikeys)
- [api 文档](https://docs.bigmodel.cn/cn/guide/models/sound-and-video/glm-asr-2512)

端点：`https://open.bigmodel.cn/api/paas/v4/audio/transcriptions`
key: `xxx.xxx`

<img width="971" height="701" alt="image" src="https://github.com/user-attachments/assets/5670b95d-0daa-4559-980e-2f11e427a7c9" />

### Aliyun Bailian ASR

- 默认推荐：`fun-asr-realtime`
- 内置模型分组：
  - Qwen3 ASR Flash Realtime：`qwen3-asr-flash-realtime`、`qwen3-asr-flash-realtime-2026-02-10`、`qwen3-asr-flash-realtime-2025-10-27`
  - Fun ASR Realtime：`fun-asr-realtime`、`fun-asr-realtime-2026-02-28`、`fun-asr-realtime-2025-11-07`、`fun-asr-realtime-2025-09-15`、`fun-asr-flash-8k-realtime`、`fun-asr-flash-8k-realtime-2026-01-28`
  - Paraformer Realtime：`paraformer-realtime-v2`、`paraformer-realtime-v1`、`paraformer-realtime-8k-v2`、`paraformer-realtime-8k-v1`
- 简介：阿里云百炼在 Voxt 中是目前远程 ASR 里预置模型最多的一组，既有 Qwen3 ASR，也有 Fun ASR 和 Paraformer，适合需要实时识别、并希望在同一平台内比较不同 ASR 系列的用户。

- [官网](https://bailian.console.aliyun.com/cn-beijing)
- [key 管理](https://bailian.console.aliyun.com/cn-beijing/?tab=model#/api-key)
- [模型用量](https://bailian.console.aliyun.com/cn-beijing/?tab=model#/model-usage/free-quota)
- [api 文档](https://bailian.console.aliyun.com/cn-beijing/?tab=doc#/doc/?type=model&url=2989727)

<img width="1445" height="1060" alt="image" src="https://github.com/user-attachments/assets/b30e8ee1-b035-4603-8426-a8461fe66676" />

1. key 管理创建 key
2. 模型用量中开启模型

端点：`wss://dashscope.aliyuncs.com/api-ws/v1/realtime`
key：`xxxx`

<img width="915" height="681" alt="image" src="https://github.com/user-attachments/assets/1da2f293-df35-4d6a-aa3a-6bc50cab571a" />

### Google Gemini

- 默认推荐：`gemini-3.5-transcribe-live`
- 内置模型：`gemini-3.5-transcribe-live`
- 简介：Google 的实时语音转写模型，通过 Gemini Live API 的 WebSocket 接入。支持 85+ 语言自动检测，保留句中中英混说，SMART 模式会在文本落到光标前去掉口水词并补好标点。

- [官网](https://ai.google.dev/)
- [api 文档](https://ai.google.dev/gemini-api/docs/live-api/live-transcribe)
- [key 管理](https://aistudio.google.com/apikey)

端点：`wss://generativelanguage.googleapis.com/ws/google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent`
key: `xxxx`

重要说明：

- Voxt 中这一档只用于语音输入，不支持文件转写和会议模式，因为只接了流式模型。
- API Key 在建立连接时追加到 WebSocket URL 上，不要填进 endpoint；粘贴进来的 `key` 查询参数会被去掉。
- 上下文短语和词典条目会作为 `customVocabulary` 发送，上限 1000 条，Google 建议 100 条左右效果最好。
- 语言提示来自你选择的用户语言。关掉“跟随主语言”会发送空列表，那才是完全自动检测。
- 单次实时会话上限 10 分钟，Voxt 不会中途重连，超过这个长度的一次听写会被截断。
- endpoint 支持任意 `wss://` 地址，`generativelanguage.googleapis.com` 直连不通时可以填中转地址。

## LLM 模型

Voxt 的远程 LLM 配置页除了内置预置模型外，也支持手动填写自定义模型 ID；下面先列出每个服务商在 app 中已经预置好的主流模型范围。

### Anthropic

- 默认推荐：`claude-sonnet-4-6`
- 内置模型范围：Claude 4.6、Claude 4.5、Claude Haiku 4.5、Claude 3 Haiku
- 代表模型：`claude-opus-4-6`、`claude-sonnet-4-6`、`claude-opus-4-5-20251101`、`claude-sonnet-4-5-20250929`、`claude-haiku-4-5-20251001`
- 简介：Anthropic 原生接口，适合高质量文本增强、改写、总结与翻译。

*Soon*

### Google

- 默认推荐：`gemini-2.5-pro`
- 内置模型范围：Gemini 3 Preview、Gemini 2.5 Pro / Flash / Flash-Lite、Gemini 2.0 Flash、Gemini 1.5 Flash / Pro
- 代表模型：`gemini-3.1-pro-preview`、`gemini-3-pro-preview`、`gemini-2.5-pro`、`gemini-2.5-flash`、`gemini-2.5-flash-lite`
- 简介：Gemini 原生接口，覆盖高质量生成和更轻量的快速模型，适合在质量与速度之间灵活切换。

*Soon*

### OpenAI

- 默认推荐：`gpt-5.2`
- 内置模型范围：GPT-5.2 / 5.1 / 5、`o4` / `o3` / `o1` 推理系列、GPT-4.1、GPT-4o、GPT-4、GPT-3.5
- 代表模型：`gpt-5.2`、`gpt-5.2-chat-latest`、`gpt-5.2-pro`、`gpt-5.1-codex`、`o3`、`o4-mini`、`gpt-4.1`、`gpt-4o`、`gpt-4o-mini`
- 简介：OpenAI-compatible 接口里选择最丰富的一组，既可以做通用文本增强，也可以覆盖推理、编码和轻量快速响应场景。

*Soon*

### Ollama

- 默认推荐：`qwen2.5`
- 内置模型范围：Qwen、Llama、Mistral、Gemma、DeepSeek、GPT-OSS 等本地 / 自建模型
- 代表模型：`qwen2.5`、`qwen3`、`llama3.1`、`mistral`、`gemma2`、`deepseek-v3.1:671b`、`gpt-oss:120b`
- Key：可选。本地 Ollama 通常留空即可；如果你接的是自建网关，也可以继续配置 bearer token。
- 端点：同时支持 Ollama 原生 `/api/chat` 与 OpenAI-compatible `/v1/chat/completions` / `/v1/responses`。
- 高级配置：配置弹窗支持响应格式、think 模式、keep-alive、logprobs、JSON schema，以及 `Options JSON` 编辑区。
- `Options JSON` 示例：`{"num_ctx": 8192}`、`{"seed": 7}`、`{"repeat_penalty": 1.1}`、`{"stop": ["</s>"]}`
- 简介：适合本地部署或自建网关用户。使用原生 Ollama 端点时可以启用更多运行时控制；走 OpenAI-compatible 端点时也能继续复用现有网关。

*Soon*

### oMLX

- 默认推荐：`qwen3`
- 内置模型范围：由本机 oMLX 服务暴露的 Qwen、Llama、DeepSeek、Gemma、Mixtral、GPT-OSS 等 MLX 本地模型别名
- 代表模型：`qwen3`、`Qwen3-Coder-Next-8bit`、`gpt-oss-120b-MXFP4-Q8`、`Qwen3.5-122B-A10B-4bit`、`Step-3.5-Flash-8bit`
- Key：可选。默认本机 `localhost` 场景通常留空即可；如果你启动 oMLX 时配置了 `--api-key`，再填 bearer token。
- 端点：Voxt 默认使用 oMLX 的 OpenAI-compatible 根地址 `http://localhost:8000/v1`，实际请求会自动落到 `/v1/chat/completions`。
- 模型 ID：既可以直接填 `/v1/models` 返回的别名，也可以使用 oMLX 自动发现的本地模型目录名。
- 简介：适合在 Apple Silicon 上运行 MLX 原生本地推理，并希望通过一个独立本地模型服务把能力接入 Voxt 的用户。

*Soon*

### DeepSeek

- 默认推荐：`deepseek-flash`（DeepSeek V4.1 Flash）；另提供 `deepseek-v4-pro`。
- 兼容选项：保留 `deepseek-v4-flash`、`deepseek-v4-flash-vision-exp` 及历史 `deepseek-chat` / `deepseek-reasoner` 配置，不自动改写用户的自定义模型 ID。
- 官方说明：V4.1 Flash 的 API ID 是 `deepseek-flash`，不要根据显示名称自行拼接 `deepseek-v4.1-*`。旧 V4 Flash / Vision ID 目前由官方转到 V4.1 Flash；旧 ID 测试成功不代表仍在使用旧版本。
- 思考模式：官方默认开启。Voxt 默认文本调用显式关闭思考，避免短输出预算被推理占满后没有正文；保留显式开启、推理强度和历史 `deepseek-reasoner` 默认行为。DeepSeek 不支持独立 `thinking.budget_tokens`，历史预算配置按开启思考处理。
- 连接测试：轻量探测关闭思考、最多输出 1 token；HTTP 400/422 会显示失败及服务端错误，不再误报成功。测试通过不代表长文本、流式调用或所有高级参数均可用。
- 若增强看似被跳过：先核对模型 ID，关闭思考及不必要的 Extra Body JSON 后重试。日志中的 `enhancement failed, using raw text` 表示调用失败后回退原文，而非未执行增强。
- 官方依据：[V4.1 Flash 发布说明](https://api-docs.deepseek.com/news/news260910)、[思考模式](https://api-docs.deepseek.com/guides/thinking_mode)、[Chat Completions API](https://api-docs.deepseek.com/api/create-chat-completion)。

### OpenRouter

- 默认推荐：`openrouter/auto`
- 内置模型范围：自动路由，以及经由 OpenRouter 暴露的 OpenAI、Google、DeepSeek、Qwen、Anthropic 等模型
- 代表模型：`openrouter/auto`、`deepseek/deepseek-chat-v3.1`、`deepseek/deepseek-r1`、`openai/gpt-4.1`、`openai/gpt-4.1-mini`、`google/gemini-2.5-pro`
- 简介：适合希望用一个统一入口切换多家模型，或者让平台自动路由到合适模型的用户。

*Soon*

### xAI (Grok)

- 默认推荐：`grok-4`
- 内置模型范围：Grok 4、Grok 4.1 Fast、Grok 3、Grok Code Fast
- 代表模型：`grok-4`、`grok-4-1-fast-reasoning`、`grok-4-1-fast-non-reasoning`、`grok-3`、`grok-3-mini`、`grok-code-fast-1`
- 简介：适合想尝试 Grok 系列文本生成与推理能力的用户。

*Soon*

### Z.ai

- 默认推荐：`glm-5`
- 内置模型范围：GLM-5、GLM-4.7、GLM-4.6、GLM-4.5 及 Flash / Air / Vision 变体
- 代表模型：`glm-5`、`glm-4.7`、`glm-4.7-flash`、`glm-4.6`、`glm-4.6v`、`glm-4.5-air`
- 简介：智谱系 LLM 入口，适合已经使用 GLM 模型族，或偏好中文体验与国产模型生态的用户。

*Soon*

### Volcengine

- 默认推荐：`doubao-seed-2-0-pro-260215`
- 内置模型范围：Doubao Seed 2.0、Doubao Seed 1.8 / 1.6、翻译模型、代码模型、视觉模型，以及部分 GLM 兼容模型
- 代表模型：`doubao-seed-2-0-pro-260215`、`doubao-seed-2-0-lite-260215`、`doubao-seed-2-0-mini-260215`、`doubao-seed-2-0-code-preview-260215`、`doubao-seed-translation-250915`
- 简介：火山引擎 / 豆包模型入口，适合想在同一平台使用通用生成、代码、翻译等多种模型的用户。

- [api 管理](https://console.volcengine.com/ark/region:ark+cn-beijing/apiKey?apikey=%7B%7D)
- [模型开通](https://console.volcengine.com/ark/region:ark+cn-beijing/openManagement?LLM=%7B%7D&advancedActiveKey=model)

<img width="1454" height="1063" alt="image" src="https://github.com/user-attachments/assets/fa82d79f-513a-40e2-88e9-d75a8a1b9218" />

端点：`https://ark.cn-beijing.volces.com/api/v3/chat/completions`
key: `xxx`

<img width="1005" height="704" alt="image" src="https://github.com/user-attachments/assets/8f1ada47-142f-49a3-af48-86978271bd2c" />


### Kimi

- 默认推荐：`kimi-k2.5`
- 内置模型范围：Kimi K2.5、K2 Thinking，以及 Moonshot V1 8K / 32K / 128K / Auto
- 代表模型：`kimi-k2.5`、`kimi-k2-thinking`、`kimi-latest`、`moonshot-v1-8k`、`moonshot-v1-32k`、`moonshot-v1-128k`
- 简介：适合偏好多轮长文本处理与 Moonshot / Kimi 生态的用户。

*Soon*

### LM Studio

- 默认推荐：`llama3.1`
- 内置模型：`llama3.1`、`qwen2.5-14b-instruct`
- 简介：适合已经在本机使用 LM Studio 暴露 OpenAI-compatible 接口的用户，能快速把本地模型接入 Voxt。

*Soon*

### MiniMax

- 默认推荐：`MiniMax-M2.5`
- 内置模型范围：MiniMax M2.5、M2.1、M2、Lightning、Stable、Text 系列
- 代表模型：`MiniMax-M2.5`、`MiniMax-M2.5-Lightning`、`MiniMax-M2.1`、`MiniMax-M2.1-Lightning`、`MiniMax-M2-Stable`、`MiniMax-Text-01`
- 简介：MiniMax 原生接口，适合希望直接接入 MiniMax 模型族的用户。

*Soon*

### Aliyun Bailian

- 默认推荐：`qwen-plus-latest`
- 内置模型范围：Qwen Max / Plus / Turbo，以及 `qwq-plus`
- 代表模型：`qwen-max-latest`、`qwen-plus-latest`、`qwen-turbo-latest`、`qwen-max`、`qwen-plus`、`qwen-turbo`、`qwq-plus`
- 简介：适合已经在阿里云百炼中使用通义千问模型，或希望把远程 ASR 与远程 LLM 都放在阿里云体系内统一管理的用户。


- [官网](https://bailian.console.aliyun.com/cn-beijing)
- [key 管理](https://bailian.console.aliyun.com/cn-beijing/?tab=model#/api-key)
- [模型用量](https://bailian.console.aliyun.com/cn-beijing/?tab=model#/model-usage/free-quota)
- [api 文档](https://bailian.console.aliyun.com/cn-beijing/?tab=doc#/doc/?type=model&url=2989727)

<img width="1445" height="1060" alt="image" src="https://github.com/user-attachments/assets/b30e8ee1-b035-4603-8426-a8461fe66676" />

1. key 管理创建 key
2. 模型用量中开启模型

端点：`https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions`
key：`xxxx`

<img width="1001" height="719" alt="image" src="https://github.com/user-attachments/assets/6de0febe-08b0-4c7d-902a-18e6de2551ba" />
