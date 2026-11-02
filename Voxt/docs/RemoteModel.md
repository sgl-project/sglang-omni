# Remote Models

Voxt lets you choose separate remote providers for `Remote ASR` and `Remote LLM`. This document only introduces the providers and model ranges currently built into the app, so you can understand what each option is for first. Setup steps such as API key creation, endpoint configuration, and model selection guidance can be expanded in later sections.

## ASR Models

### OpenAI Whisper

- Suggested default: `whisper-1`
- Built-in models: `whisper-1`, `gpt-4o-mini-transcribe`, `gpt-4o-transcribe`
- Overview: Best for general-purpose multilingual audio transcription. Voxt currently integrates it as file-based transcription, which makes it a good fit for users who want compatibility and a stable experience.
- Voxt note: this provider also supports custom model IDs and custom OpenAI-compatible transcription endpoints.

- [Website](https://openai.com/)
- [API Docs](https://developers.openai.com/api/reference/resources/audio)
- [API Keys](https://platform.openai.com/api-keys)

Endpoint: `https://api.openai.com/v1/audio/transcriptions`  
Key: `$OPENAI_API_KEY`

Compatible custom endpoint examples:

- [MOSI Studio MOSS Transcribe](https://studio.mosi.cn/docs/moss-transcribe)
  - Endpoint: `https://studio.mosi.cn/api/v1/audio/transcriptions`
  - Models: `moss-transcribe`, `moss-transcribe-diarize`
- [Groq Speech-to-Text](https://console.groq.com/docs/speech-to-text)
  - Endpoint: `https://api.groq.com/openai/v1/audio/transcriptions`
  - Models: `whisper-large-v3-turbo`, `whisper-large-v3`

Important notes:

- In Voxt, custom ASR model IDs are supported only through the `OpenAI Whisper` provider.
- Enter the full transcription endpoint, not just the API root.
- Voxt uses file upload transcription for this provider path. If a compatible service also supports URL or Base64 inputs, Voxt still uses file upload.
- If a compatible service returns extra structured metadata such as diarization segments, Voxt currently reads the transcript text but does not surface that metadata as a dedicated UI feature.

<img width="923" height="676" alt="image" src="https://github.com/user-attachments/assets/62be17c8-78f5-418e-a4ba-873d18d58f18" />

### Doubao ASR

- Suggested default: `volc.seedasr.sauc.duration`
- Built-in models: `volc.seedasr.sauc.duration`, `volc.bigasr.sauc.duration`
- Overview: Optimized for low-latency streaming recognition, especially in Chinese, and also works well for mixed Chinese-English speech. It is a strong option for users who want a realtime voice input experience.

- [Website](https://www.volcengine.com/)
- [Doubao Speech](https://console.volcengine.com/speech/app)
- [API Docs](https://www.volcengine.com/docs/6561/1354869?lang=zh)
- [Enable Models](https://console.volcengine.com/ark/region:ark+cn-beijing/openManagement?LLM=%7B%7D&advancedActiveKey=model)

<img width="1437" height="1049" alt="image" src="https://github.com/user-attachments/assets/a62a34a2-33f4-4ddd-b3ec-5bc507852a2b" />
<img width="1442" height="1059" alt="image" src="https://github.com/user-attachments/assets/188f5abd-7bec-4592-8aa9-68b5e9ee260d" />

1. Create an application and choose the model.
2. In the left sidebar, open `Doubao Streaming Speech Recognition Model 2.0`, then copy the `APP ID` and `Access Token` into Voxt.

<img width="939" height="683" alt="image" src="https://github.com/user-attachments/assets/18c4032e-b3e8-4ab5-92fe-9f6f0117ecd1" />

### GLM ASR

- Suggested default: `glm-asr-1`
- Built-in models: `glm-asr-2512`, `glm-asr-1`
- Overview: ASR models from the Zhipu GLM family. They are a practical choice if you already use GLM services and want to keep voice capabilities under the same provider.

- [Website](https://bigmodel.cn)
- [Key Management](https://bigmodel.cn/usercenter/proj-mgmt/apikeys)
- [API Docs](https://docs.bigmodel.cn/cn/guide/models/sound-and-video/glm-asr-2512)

Endpoint: `https://open.bigmodel.cn/api/paas/v4/audio/transcriptions`  
Key: `xxx.xxx`

<img width="971" height="701" alt="image" src="https://github.com/user-attachments/assets/5670b95d-0daa-4559-980e-2f11e427a7c9" />

### Aliyun Bailian ASR

- Suggested default: `fun-asr-realtime`
- Built-in model groups:
  - Qwen3 ASR Flash Realtime: `qwen3-asr-flash-realtime`, `qwen3-asr-flash-realtime-2026-02-10`, `qwen3-asr-flash-realtime-2025-10-27`
  - Fun ASR Realtime: `fun-asr-realtime`, `fun-asr-realtime-2026-02-28`, `fun-asr-realtime-2025-11-07`, `fun-asr-realtime-2025-09-15`, `fun-asr-flash-8k-realtime`, `fun-asr-flash-8k-realtime-2026-01-28`
  - Paraformer Realtime: `paraformer-realtime-v2`, `paraformer-realtime-v1`, `paraformer-realtime-8k-v2`, `paraformer-realtime-8k-v1`
- Overview: Aliyun Bailian has the widest preset ASR selection in Voxt. It includes Qwen3 ASR, Fun ASR, and Paraformer, which makes it a strong choice for users who need realtime transcription and want to compare multiple ASR families on the same platform.

- [Website](https://bailian.console.aliyun.com/cn-beijing)
- [Key Management](https://bailian.console.aliyun.com/cn-beijing/?tab=model#/api-key)
- [Model Usage](https://bailian.console.aliyun.com/cn-beijing/?tab=model#/model-usage/free-quota)
- [API Docs](https://bailian.console.aliyun.com/cn-beijing/?tab=doc#/doc/?type=model&url=2989727)

<img width="1445" height="1060" alt="image" src="https://github.com/user-attachments/assets/b30e8ee1-b035-4603-8426-a8461fe66676" />

1. Create an API key in Key Management.
2. Enable the models you want in Model Usage.

Endpoint: `wss://dashscope.aliyuncs.com/api-ws/v1/realtime`  
Key: `xxxx`

<img width="915" height="681" alt="image" src="https://github.com/user-attachments/assets/1da2f293-df35-4d6a-aa3a-6bc50cab571a" />

### Google Gemini

- Suggested default: `gemini-3.5-transcribe-live`
- Built-in models: `gemini-3.5-transcribe-live`
- Overview: Google's live speech-to-text model, reached over the Gemini Live API WebSocket. It auto-detects language across 85+ languages, keeps mid-sentence code-mixing intact, and its SMART mode removes filler words and formats punctuation before the text reaches your cursor.

- [Website](https://ai.google.dev/)
- [API Docs](https://ai.google.dev/gemini-api/docs/live-api/live-transcribe)
- [Key Management](https://aistudio.google.com/apikey)

Endpoint: `wss://generativelanguage.googleapis.com/ws/google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent`
Key: `xxxx`

Important notes:

- Voxt uses this provider for voice input only. File transcription and Meeting mode are not supported, because only the streaming model is wired.
- The API key is appended to the WebSocket URL at connect time. Leave it out of the endpoint field; a pasted `key` query parameter is stripped.
- Contextual phrases and dictionary terms are sent as `customVocabulary`, capped at 1000 terms. Google suggests around 100 terms for best results.
- Language hints follow your selected user languages. Turning off "follow main language" sends an empty list, which is what enables full auto-detection.
- A live session is capped at 10 minutes. Voxt does not reconnect mid-session, so a single dictation longer than that will be cut short.
- The endpoint field accepts any `wss://` host, so a relay can be used where `generativelanguage.googleapis.com` is not directly reachable.

## LLM Models

In Voxt, the remote LLM configuration sheet supports both preset models and custom model IDs. The sections below list the main model ranges that are already built into the app for each provider.

### Anthropic

- Suggested default: `claude-sonnet-4-6`
- Built-in model range: Claude 4.6, Claude 4.5, Claude Haiku 4.5, Claude 3 Haiku
- Representative models: `claude-opus-4-6`, `claude-sonnet-4-6`, `claude-opus-4-5-20251101`, `claude-sonnet-4-5-20250929`, `claude-haiku-4-5-20251001`
- Overview: Uses Anthropic's native API and is well suited for high-quality rewriting, summarization, translation, and text enhancement.

*Soon*

### Google

- Suggested default: `gemini-2.5-pro`
- Built-in model range: Gemini 3 Preview, Gemini 2.5 Pro / Flash / Flash-Lite, Gemini 2.0 Flash, Gemini 1.5 Flash / Pro
- Representative models: `gemini-3.1-pro-preview`, `gemini-3-pro-preview`, `gemini-2.5-pro`, `gemini-2.5-flash`, `gemini-2.5-flash-lite`
- Overview: Uses Gemini's native API and covers both higher-quality generation and lighter, faster models, making it flexible for different speed and quality needs.

*Soon*

### OpenAI

- Suggested default: `gpt-5.2`
- Built-in model range: GPT-5.2 / 5.1 / 5, the `o4` / `o3` / `o1` reasoning series, GPT-4.1, GPT-4o, GPT-4, and GPT-3.5
- Representative models: `gpt-5.2`, `gpt-5.2-chat-latest`, `gpt-5.2-pro`, `gpt-5.1-codex`, `o3`, `o4-mini`, `gpt-4.1`, `gpt-4o`, `gpt-4o-mini`
- Overview: This is the broadest OpenAI-compatible selection in Voxt. It works well for general text enhancement, as well as reasoning, coding, and lower-latency response scenarios.

*Soon*

### Ollama

- Suggested default: `qwen2.5`
- Built-in model range: locally hosted or self-managed Qwen, Llama, Mistral, Gemma, DeepSeek, GPT-OSS, and related models
- Representative models: `qwen2.5`, `qwen3`, `llama3.1`, `mistral`, `gemma2`, `deepseek-v3.1:671b`, `gpt-oss:120b`
- API key: optional. Local Ollama setups usually leave this blank, but self-hosted gateways can still provide a bearer token if needed.
- Endpoints: supports both native Ollama `/api/chat` and OpenAI-compatible `/v1/chat/completions` or `/v1/responses`.
- Advanced config: the configuration sheet now exposes response format, think mode, keep-alive, logprobs, JSON schema, and an `Options JSON` editor.
- `Options JSON` examples: `{"num_ctx": 8192}`, `{"seed": 7}`, `{"repeat_penalty": 1.1}`, `{"stop": ["</s>"]}`
- Overview: Best for users running local deployments or self-hosted gateways. Native Ollama endpoints unlock extra runtime controls, while OpenAI-compatible endpoints keep existing gateways reusable.

*Soon*

### oMLX

- Suggested default: `qwen3`
- Built-in model range: local MLX-hosted Qwen, Llama, DeepSeek, Gemma, Mixtral, GPT-OSS, and other model aliases exposed by your oMLX server
- Representative models: `qwen3`, `Qwen3-Coder-Next-8bit`, `gpt-oss-120b-MXFP4-Q8`, `Qwen3.5-122B-A10B-4bit`, `Step-3.5-Flash-8bit`
- API key: optional. Leave it blank for the default localhost setup, or provide a bearer token if you started oMLX with `--api-key`.
- Endpoint: Voxt defaults to the oMLX OpenAI-compatible root `http://localhost:8000/v1` and resolves requests to `/v1/chat/completions`.
- Model IDs: oMLX accepts either the configured alias returned by `/v1/models` or the discovered local model directory name.
- Overview: Best for users running MLX-native local inference on Apple Silicon and want a dedicated local model server with OpenAI-compatible access from Voxt.

*Soon*

### DeepSeek

- Suggested default: `deepseek-flash` (DeepSeek V4.1 Flash); `deepseek-v4-pro` is also available.
- Compatibility options: `deepseek-v4-flash`, `deepseek-v4-flash-vision-exp`, and historical `deepseek-chat` / `deepseek-reasoner` configurations remain available. Custom model IDs are not rewritten.
- Official model ID: use `deepseek-flash`, not an inferred `deepseek-v4.1-*` name. The old V4 Flash / Vision IDs currently route to V4.1 Flash, so a successful test with an old ID does not imply the old model is still running.
- Thinking: enabled by default upstream. Voxt explicitly disables it for default text requests to avoid spending the short output budget on reasoning without visible text. Explicit thinking/effort choices and the legacy `deepseek-reasoner` default are preserved. DeepSeek does not support a separate `thinking.budget_tokens`; saved budget settings are treated as thinking enabled.
- Connection test: a lightweight, non-thinking probe with at most one output token. HTTP 400/422 now fail with the server error instead of reporting success. A passing probe does not validate long input, streaming, or every advanced setting.
- If enhancement appears skipped, check the model ID and retry with thinking and unnecessary Extra Body JSON disabled. The log message `enhancement failed, using raw text` indicates fallback after a failed call, not a skipped enhancement step.
- Official references: [V4.1 Flash release](https://api-docs.deepseek.com/news/news260910), [Thinking Mode](https://api-docs.deepseek.com/guides/thinking_mode), [Chat Completions API](https://api-docs.deepseek.com/api/create-chat-completion).

### OpenRouter

- Suggested default: `openrouter/auto`
- Built-in model range: automatic routing, plus OpenAI, Google, DeepSeek, Qwen, Anthropic, and other models exposed through OpenRouter
- Representative models: `openrouter/auto`, `deepseek/deepseek-chat-v3.1`, `deepseek/deepseek-r1`, `openai/gpt-4.1`, `openai/gpt-4.1-mini`, `google/gemini-2.5-pro`
- Overview: Good for users who want one unified entry point for multiple vendors, or who prefer automatic routing to a suitable model.

*Soon*

### xAI (Grok)

- Suggested default: `grok-4`
- Built-in model range: Grok 4, Grok 4.1 Fast, Grok 3, and Grok Code Fast
- Representative models: `grok-4`, `grok-4-1-fast-reasoning`, `grok-4-1-fast-non-reasoning`, `grok-3`, `grok-3-mini`, `grok-code-fast-1`
- Overview: A good option for users who want to try the Grok family for text generation and reasoning tasks.

*Soon*

### Z.ai

- Suggested default: `glm-5`
- Built-in model range: GLM-5, GLM-4.7, GLM-4.6, GLM-4.5, plus Flash, Air, and Vision variants
- Representative models: `glm-5`, `glm-4.7`, `glm-4.7-flash`, `glm-4.6`, `glm-4.6v`, `glm-4.5-air`
- Overview: A GLM-family LLM entry point that works well for users already using GLM models or those who prefer the Chinese domestic model ecosystem.

*Soon*

### Volcengine

- Suggested default: `doubao-seed-2-0-pro-260215`
- Built-in model range: Doubao Seed 2.0, Doubao Seed 1.8 / 1.6, translation models, code models, vision models, and some GLM-compatible models
- Representative models: `doubao-seed-2-0-pro-260215`, `doubao-seed-2-0-lite-260215`, `doubao-seed-2-0-mini-260215`, `doubao-seed-2-0-code-preview-260215`, `doubao-seed-translation-250915`
- Overview: The Volcengine and Doubao entry point is a good fit for users who want one platform for general generation, coding, translation, and other model types.

- [API Keys](https://console.volcengine.com/ark/region:ark+cn-beijing/apiKey?apikey=%7B%7D)
- [Enable Models](https://console.volcengine.com/ark/region:ark+cn-beijing/openManagement?LLM=%7B%7D&advancedActiveKey=model)

<img width="1454" height="1063" alt="image" src="https://github.com/user-attachments/assets/fa82d79f-513a-40e2-88e9-d75a8a1b9218" />

Endpoint: `https://ark.cn-beijing.volces.com/api/v3/chat/completions`  
Key: `xxx`

<img width="1005" height="704" alt="image" src="https://github.com/user-attachments/assets/8f1ada47-142f-49a3-af48-86978271bd2c" />

### Kimi

- Suggested default: `kimi-k2.5`
- Built-in model range: Kimi K2.5, K2 Thinking, and Moonshot V1 8K / 32K / 128K / Auto
- Representative models: `kimi-k2.5`, `kimi-k2-thinking`, `kimi-latest`, `moonshot-v1-8k`, `moonshot-v1-32k`, `moonshot-v1-128k`
- Overview: A good fit for users who prefer the Kimi and Moonshot ecosystem, especially for multi-turn and long-context text tasks.

*Soon*

### LM Studio

- Suggested default: `llama3.1`
- Built-in models: `llama3.1`, `qwen2.5-14b-instruct`
- Overview: Best for users already exposing a local OpenAI-compatible endpoint from LM Studio and want to connect those local models to Voxt quickly.

*Soon*

### MiniMax

- Suggested default: `MiniMax-M2.5`
- Built-in model range: MiniMax M2.5, M2.1, M2, Lightning, Stable, and Text series
- Representative models: `MiniMax-M2.5`, `MiniMax-M2.5-Lightning`, `MiniMax-M2.1`, `MiniMax-M2.1-Lightning`, `MiniMax-M2-Stable`, `MiniMax-Text-01`
- Overview: Uses MiniMax's native API and is suitable for users who want direct access to the MiniMax model family.

*Soon*

### Aliyun Bailian

- Suggested default: `qwen-plus-latest`
- Built-in model range: Qwen Max / Plus / Turbo, plus `qwq-plus`
- Representative models: `qwen-max-latest`, `qwen-plus-latest`, `qwen-turbo-latest`, `qwen-max`, `qwen-plus`, `qwen-turbo`, `qwq-plus`
- Overview: A good choice for users already using Tongyi Qianwen models in Aliyun Bailian, or for those who want both remote ASR and remote LLM under the same Aliyun stack.

- [Website](https://bailian.console.aliyun.com/cn-beijing)
- [Key Management](https://bailian.console.aliyun.com/cn-beijing/?tab=model#/api-key)
- [Model Usage](https://bailian.console.aliyun.com/cn-beijing/?tab=model#/model-usage/free-quota)
- [API Docs](https://bailian.console.aliyun.com/cn-beijing/?tab=doc#/doc/?type=model&url=2989727)

<img width="1445" height="1060" alt="image" src="https://github.com/user-attachments/assets/b30e8ee1-b035-4603-8426-a8461fe66676" />

1. Create an API key in Key Management.
2. Enable the models you want in Model Usage.

Endpoint: `https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions`  
Key: `xxxx`

<img width="1001" height="719" alt="image" src="https://github.com/user-attachments/assets/6de0febe-08b0-4c7d-902a-18e6de2551ba" />
