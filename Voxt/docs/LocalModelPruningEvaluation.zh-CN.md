# 本地 ASR 与 LLM 模型精简评估

日期：2026-06-18

> 历史评估，不再作为当前实现清单。本分支已改为删除隐藏模型运行时，不再采用下文“隐藏兼容优先”的策略。当前方案见 [模型栈现代化方案](ModelStackModernizationPlan.zh-CN.md)，实际完成项与待验证事项见 [实施记录](ModelStackModernizationImplementation.zh-CN.md)。

本评估基于当前代码中的本地模型清单：

- ASR：`Voxt/Transcription/MLXModelSupport.swift`
- LLM：`Voxt/Core/Models/CustomLLMModelSupport.swift`
- 历史参考：`docs/MLXModelSupport2025.zh-CN.md`

评分不是实测 WER / BLEU / MT-Bench，而是用于产品精简的综合分。它把模型质量预期、下载体积、阶段覆盖、系列冗余、维护成本、Voxt 任务适配度放在一起比较。

## 精简规则

1. 系列模型按阶段收敛：同一系列只保留低性能、中性能、高性能或强性能中的优势型号。
2. 普通模型全局打分：不属于主系列或阶段价值不清晰的模型，和现有核心模型一起比较，低分模型隐藏或剔除。
3. 默认可见列表优先“少而清晰”：能解释清楚使用场景的模型才进入默认 UI。
4. 隐藏兼容优先于直接删除：旧用户配置、已下载模型、历史选择先走兼容迁移。
5. Voxt 的核心任务是本地语音转写、文本清理、翻译、改写；纯推理、超长上下文、视觉能力只有明确产品入口时才加分。

## 阶段定义

| 阶段 | ASR 预期 | LLM 预期 | 产品含义 |
|---|---|---|---|
| 低性能 | 小包体、低内存、启动快 | 1B 到 2B 级或小体积量化 | 默认入门、低配 Mac、快速试用 |
| 中性能 | 质量明显更稳，体积仍可接受 | 3B 到 5B 级 | 主推平衡档，覆盖大多数用户 |
| 高性能 | 准确率或质量优先 | 7B 到 10B 级 | 高质量本地处理，适合更大内存 |
| 强性能 / 实验 | 资源占用大或场景特殊 | 10B+、MoE、推理/视觉/超长上下文 | 默认隐藏，仅高级用户或 A/B 测试 |

## 建议最终可见列表

### ASR

| 阶段 | 场景 | 推荐保留 | repo / ID | 当前体积 | 结论 |
|---|---|---|---|---:|---|
| 低性能 | 默认通用 ASR | Qwen3-ASR 0.6B 4bit | `mlx-community/Qwen3-ASR-0.6B-4bit` | 0.71 GB | 保留为默认 |
| 中性能 | 主推通用 ASR | Qwen3-ASR 1.7B 6bit | `mlx-community/Qwen3-ASR-1.7B-6bit` | 2.03 GB | 保留为主推质量档 |
| 高性能 | 高精度通用 ASR | Qwen3-ASR 1.7B 8bit | `mlx-community/Qwen3-ASR-1.7B-8bit` | 2.47 GB | 保留为高性能档 |
| 兼容 | 实时多语言 | Voxtral Realtime Mini 4B | Voxtral Realtime repos | 3.15–8.89 GB | 全系列隐藏支持，仅保留已有安装和旧配置兼容 |
| 低 / 中性能 | 英文专用 | Parakeet TDT 0.6B v3 | `mlx-community/parakeet-tdt-0.6b-v3` | 2.51 GB | 只保留 Parakeet 英文主入口 |
| 已移除 | 离线准确率 | FireRed ASR 2 | MLX FireRed ID | 0.54–4.57 GB | sherpa-onnx FireRed 路径已移除；MLX FireRed 仅保留现有安装与旧配置迁移 |
| 已移除 | 离线多语言 | FunASR Nano | `funasr-nano-int8` | 0.95 GB | sherpa-onnx 路径已移除 |
| 中性能 | 多语言 / 事件检测 | SenseVoice Small | `mlx-community/SenseVoiceSmall` | 936 MB | 保留 |
| 兼容 | MLX Whisper 历史迁移 | Whisper Tiny / Base / Small / Large v3 / Large v3 Turbo | MLX Whisper repos | 76 MB 到 3.09 GB | 保留 MLX 路径；旧短 ID 迁移 |
| 低 / 中性能 | 流式 ASR | Nemotron ASR Streaming 0.6B 8bit | `mlx-community/nemotron-3.5-asr-streaming-0.6b-8bit` | 约 0.76 GB | 保留为流式 ASR 入口 |

### LLM

| 阶段 | 场景 | 推荐保留 | repo / ID | 当前体积 | 结论 |
|---|---|---|---|---:|---|
| 低性能 | 轻量质量档 | Qwen3.5 2B 4bit | `mlx-community/Qwen3.5-2B-4bit` | 1.74 GB | 保留 |
| 中性能 | 默认本地 LLM | Qwen3.5 4B OptiQ 4bit | `mlx-community/Qwen3.5-4B-OptiQ-4bit` | 2.97 GB | 保留为主推默认 |
| 高性能 | 高质量本地 LLM | Qwen3.5 9B OptiQ 4bit | `mlx-community/Qwen3.5-9B-OptiQ-4bit` | 6.04 GB | 保留 |
| 中 / 高性能 | 非 Qwen 备用 | Gemma 4 E4B IT 4bit | `mlx-community/gemma-4-e4b-it-4bit` | 未内置 | 保留 |
| 高性能 | 中文 / 多语言备用 | GLM 4 9B | `mlx-community/GLM-4-9B-0414-4bit` | 5.31 GB | 保留 |

## ASR 系列评估

| 系列 | 当前模型 | 推荐动作 | 精简评分 | 理由 |
|---|---|---:|---:|---|
| Qwen3-ASR | 0.6B 4bit | 可见保留 | 92 | 默认模型，体积小，通用性强，低性能阶段边界清晰 |
| Qwen3-ASR | 0.6B 6bit | 隐藏兼容 | 78 | 比 4bit 稍强，但和 1.7B 6bit 的中性能定位重叠 |
| Qwen3-ASR | 0.6B 8bit | 隐藏兼容 | 74 | 0.6B 高量化收益不如直接上 1.7B 6bit |
| Qwen3-ASR | 0.6B bf16 | 隐藏兼容 | 70 | 体积接近 1.7B 4bit，阶段价值不清晰 |
| Qwen3-ASR | 1.7B 4bit | 隐藏兼容 | 80 | 夹在 0.6B 4bit 和 1.7B 6bit 中间，解释成本高 |
| Qwen3-ASR | 1.7B 6bit | 可见保留 | 95 | 主推平衡档，质量、体积、用户理解成本最均衡 |
| Qwen3-ASR | 1.7B 8bit | 可见保留 | 90 | 高性能档清晰，仍比 bf16 更可控 |
| Qwen3-ASR | 1.7B bf16 | 隐藏兼容 | 76 | 高资源占用，默认 UI 中没有足够区分度 |
| Voxtral Realtime | 4bit | 隐藏兼容 | 77 | 实时定位明确，但和 6bit 差异不如 6bit 平衡 |
| Voxtral Realtime | 6bit | 隐藏兼容 | 87 | 不再默认展示，保留已有安装和旧配置兼容 |
| Voxtral Realtime | fp16 | 隐藏兼容 | 69 | 8.89 GB 体积过重，不适合默认展示 |
| Parakeet | TDT CTC 110M | 隐藏兼容 | 72 | 极小英文模型有价值，但质量阶段不如 0.6B v3 清晰 |
| Parakeet | TDT 0.6B v2 | 隐藏兼容 | 65 | 被 v3 替代，仅保留已安装用户卸载路径 |
| Parakeet | TDT 0.6B v3 | 可见保留 | 84 | 英文专用入口最清晰，系列内保留它即可 |
| Parakeet | CTC 0.6B | 隐藏兼容 | 64 | 解码变体解释成本高，和 TDT v3 重叠 |
| Parakeet | RNNT 0.6B | 隐藏兼容 | 64 | 同尺寸同场景重叠，默认用户难以选择 |
| Parakeet | TDT 1.1B | 隐藏兼容 | 72 | 体积 4.28 GB，英文专用高配价值有限 |
| Parakeet | TDT CTC 1.1B | 隐藏兼容 | 74 | 1.1B 系列里相对可保留，但不进默认 UI |
| Parakeet | CTC 1.1B | 隐藏兼容 | 62 | 同系列冗余 |
| Parakeet | RNNT 1.1B | 隐藏兼容 | 62 | 同系列冗余 |
| MLX Whisper | Tiny | 隐藏兼容 | 58 | 质量和产品价值被 Qwen3-ASR 低配档覆盖，仅用于旧配置迁移 |
| MLX Whisper | Base | 隐藏兼容 | 63 | 历史默认迁移目标，但不建议继续作为新安装入口 |
| MLX Whisper | Small | 可见保留 | 76 | Whisper 轻量入口，保留给偏好 Whisper 路径的用户 |
| MLX Whisper | Large v3 Turbo | 可见保留 | 82 | Whisper 平衡质量和延迟的主入口 |
| MLX Whisper | Large v3 | 可见保留 | 84 | Whisper 高质量入口，保留为高精度选项 |

## ASR 普通模型评估

| 模型 | repo / ID | 推荐动作 | 精简评分 | 理由 |
|---|---|---:|---:|---|
| Cohere Transcribe 03-2026 fp16 | `beshkenadze/cohere-transcribe-03-2026-mlx-fp16` | 隐藏兼容 | 76 | 当前精简列表未保留为默认入口，仅保留已安装用户卸载路径 |
| GLM-ASR Nano 2512 4bit | `mlx-community/GLM-ASR-Nano-2512-4bit` | 隐藏兼容 | 72 | 小体积中文/多语言备用，但默认价值低于 Qwen3-ASR 0.6B 4bit |
| Granite Speech 4.0 1B 5bit | `mlx-community/granite-4.0-1b-speech-5bit` | 隐藏兼容 | 70 | 仅支持英语、法语、德语、西班牙语、葡萄牙语、日语；和 Qwen3-ASR 默认入口重叠 |
| Nemotron ASR Streaming 0.6B 8bit | `mlx-community/nemotron-3.5-asr-streaming-0.6b-8bit` | 可见保留 | 82 | 新增依赖已支持，流式定位清晰，体积适合作为默认展示入口 |
| FireRed ASR 2 | MLX FireRed ID | 已移除 sherpa 路径 | 82 | sherpa-onnx 模型与运行时已移除；MLX 路径仍单独保留 |
| FunASR Nano | `funasr-nano-int8` | 已移除 | 72 | sherpa-onnx 离线识别路径已移除 |
| SenseVoice Small | `mlx-community/SenseVoiceSmall` | 可见保留 | 80 | 多语言和事件检测有差异化，体积也适合默认展示 |

## LLM 系列评估

| 系列 | 当前模型 | 推荐动作 | 精简评分 | 理由 |
|---|---|---:|---:|---|
| Qwen2 / Qwen2.5 | Qwen2 1.5B Instruct | 剔除 / 不推荐 | 54 | 旧默认，体积 3.10 GB，能力和体积比低于 Qwen3.5 2B / 4B |
| Qwen2 / Qwen2.5 | Qwen2.5 3B Instruct | 剔除 / 不推荐 | 58 | 6.18 GB 体积不适合 3B 定位 |
| Qwen2 / Qwen2.5 | Qwen2.5 7B Instruct 4bit | 隐藏兼容 | 62 | 旧配置兼容即可，被 Qwen3.5 9B 覆盖 |
| Qwen2.5 VL | Qwen2.5 VL 3B 4bit | 隐藏兼容 | 68 | 视觉能力有差异，但 Voxt 主路径不是视觉输入 |
| Qwen3 | 0.6B 4bit | 隐藏兼容 | 67 | 低配阶段被 Qwen3.5 2B 覆盖 |
| Qwen3 | 1.7B 4bit | 隐藏兼容 | 72 | 轻量可用，但 Qwen3.5 2B 更适合作为主线 |
| Qwen3 | 4B 4bit | 隐藏兼容 | 76 | 被 Qwen3.5 4B OptiQ 替代 |
| Qwen3 | 8B 4bit | 隐藏兼容 | 78 | 被 Qwen3.5 9B OptiQ 替代 |
| Qwen3 | 30B-A3B 4bit | 隐藏兼容 | 70 | 强性能实验，但体积和解释成本过高 |
| Qwen3 VL | Qwen3 VL 4B Instruct 4bit | 隐藏兼容 | 72 | 有视觉差异化；除非产品明确展示图像输入，否则不进默认 |
| Qwen3.5 | 0.8B OptiQ 4bit | 隐藏兼容 | 75 | 低配入口改为 2B，0.8B 仅保留历史选择 / 已安装显示 |
| Qwen3.5 | 2B 4bit | 可见保留 | 88 | 低配质量档，适合改写和翻译的最低稳定入口 |
| Qwen3.5 | 4B 4bit | 隐藏兼容 | 84 | 可作为普通 4bit 兼容；默认优先 4B OptiQ |
| Qwen3.5 | 4B OptiQ 4bit | 可见保留 | 94 | 中性能主推，体积略低于普通 4bit，评分更高 |
| Qwen3.5 | 9B OptiQ 4bit | 可见保留 | 91 | 高性能质量档，阶段边界清晰 |
| GLM | GLM 4 9B | 可见保留 | 86 | 中文 / 多语言备用价值明确 |
| GLM | GLM-4 9B Chat 1M 4bit | 已移除 | 72 | 长上下文不是 Voxt 当前核心需求，和 GLM 4 9B 重叠 |
| GLM | GLM-Z1 9B 0414 4bit | 已移除 | 74 | 推理向，但改写/翻译不需要默认展示 |
| GLM | GLM-4.7 Flash 4bit | 已移除 | 68 | 16.9 GB，强性能兼容即可 |
| Llama | Llama 3.2 1B 4bit | 已移除 | 60 | 低配阶段被 Qwen3.5 2B 覆盖 |
| Llama | Llama 3.2 3B 4bit | 已移除 | 62 | 非主线轻量模型，和 Qwen3.5 2B / 4B 重叠 |
| Llama | Llama 3 8B 4bit | 已移除 | 66 | 旧 8B，默认价值低于 Qwen3.5 9B / GLM 4 9B |
| Llama | Llama 3.1 8B 4bit | 已移除 | 68 | 2024 系列，主线已收敛到 Qwen / Gemma / GLM |
| Mistral | Mistral 7B Instruct v0.3 4bit | 已移除 | 70 | 稳定但非主线，默认价值低于 Qwen3.5 / Gemma4 |
| Mistral | Mistral Nemo 2407 4bit | 已移除 | 72 | 高质量备用，但 6.91 GB 体积偏重 |
| Mistral | Mistral 3 3B | 可见保留 | 82 | 新 Mistral3 轻量入口，直接显示不分组 |
| Gemma | Gemma 2 2B 4bit | 剔除 / 不推荐 | 61 | 被 Gemma 4 E2B 和 Qwen3.5 2B 替代 |
| Gemma | Gemma 2 9B 4bit | 剔除 / 不推荐 | 66 | 被 Gemma 4 E4B 和 Qwen3.5 9B 替代 |
| Gemma | Gemma 4 E2B 4bit | 隐藏兼容 | 76 | 轻量视觉备用，但默认低配优先 Qwen3.5 2B |
| Gemma | Gemma 4 E4B 4bit | 可见保留 | 84 | 非 Qwen 主线备用，差异化明确 |

## LLM 普通模型评估

| 模型 | repo / ID | 推荐动作 | 精简评分 | 理由 |
|---|---|---:|---:|---|
| Phi 3.5 Mini Instruct 4bit | `mlx-community/Phi-3.5-mini-instruct-4bit` | 已移除 | 60 | 轻量价值被 Qwen3.5 2B 覆盖 |
| InternLM2.5 7B Chat 4bit | `mlx-community/internlm2_5-7b-chat-4bit` | 已移除 | 70 | 中文友好，但和 GLM 4 9B、Qwen3.5 高配重叠 |
| MiniCPM4 8B 4bit | `mlx-community/MiniCPM4-8B-4bit` | 已移除 | 72 | 双语能力可能有价值，但默认列表已有 GLM / Qwen |
| Granite 3.3 2B Instruct 4bit | `mlx-community/granite-3.3-2b-instruct-4bit` | 已移除 | 59 | 结构化能力不够支撑默认展示，且版本较旧 |
| MiMo 7B SFT 4bit | `mlx-community/MiMo-7B-SFT-4bit` | 已移除 | 69 | 需要实测中文改写质量，先不默认展示 |
| AceReason Nemotron 7B 4bit | `mlx-community/AceReason-Nemotron-7B-4bit` | 已移除 | 73 | 推理向模型，适合实验池，不适合作为改写/翻译默认 |

## 建议执行顺序

| 优先级 | 动作 | 范围 |
|---:|---|---|
| P0 | 将默认可见 ASR 收敛为 9 个入口 | MLX Audio 9 个：Qwen3-ASR 3 个、Parakeet 1 个、Nemotron 1 个、SenseVoice 1 个、MLX Whisper 3 个；Voxtral、FireRed 和 FunASR Nano 隐藏支持 |
| P0 | 将默认可见 LLM 收敛为精简入口 | Qwen3.5、Gemma4、GLM 4 9B、Mistral 3 3B |
| P0 | 已修正 Qwen3.5 0.8B OptiQ repo ID | 当前代码使用 `mlx-community/Qwen3.5-0.8B-OptiQ-4bit`，并为旧 `mlx-community/Qwen3.5-0.8B-4bit-OptiQ` 保留 alias |
| P1 | 将旧系列移入隐藏兼容 | Qwen2 / Qwen2.5、Qwen3、Llama、Mistral、Gemma2、MLX Whisper 旧档位 |
| P1 | UI 上按阶段展示，而不是平铺模型名 | 低性能 / 中性能 / 高性能 / 强性能或实验 |
| P2 | 增加迁移说明和旧配置解析测试 | 确保已选模型不会因为隐藏而失效 |

## 数量变化

| 类型 | 当前默认可见 | 建议默认可见 | 隐藏兼容 / 实验 | 剔除 / 不推荐 |
|---|---:|---:|---:|---:|
| MLX ASR | 26 | 8 | 18 | 0 |
| MLX Whisper ASR | 5 | 3 | 2 | 0 |
| 本地 LLM | 31 | 6 | 18 | 7 |
| LLM 隐藏兼容 | 3 | 0 | 3 | 0 |

## 默认推荐

| 类型 | 默认模型 | 理由 |
|---|---|---|
| 本地 ASR 默认 | Qwen3-ASR 0.6B 4bit | 下载小、实时友好、低配可用 |
| 本地 ASR 主推质量档 | Qwen3-ASR 1.7B 6bit | 质量和资源平衡最好 |
| 本地 LLM 默认 | Qwen3.5 4B OptiQ 4bit | Voxt 改写、翻译、清理任务的平衡点最好 |
| 本地 LLM 低配默认 | Qwen3.5 2B 4bit | 比 0.8B 更稳，仍保持轻量 |
| 本地 LLM 高质量 | Qwen3.5 9B OptiQ 4bit | 高性能本地质量档清晰 |
