# MLX 2025+ 模型支持清单与接入建议

更新时间：2026-06-06

> 历史评估说明：本文保留当时的上游调研和精简建议。当前 Voxt 实现已进一步移除旧版隐藏本地 LLM（包括 Llama 3.x、Mistral 7B/Nemo、Phi 3.5、InternLM2.5、MiniCPM4、Granite 3.3、MiMo、AceReason，以及 GLM Chat 1M / GLM-Z1 / GLM-4.7 Flash）和 sherpa-onnx ASR；当前可用清单以代码为准。

## 统计口径

- 只统计当前上游 MLX 明确支持的模型，不使用 Voxt 本地 App 的清单作为来源边界。
- ASR / STT 来源：`mlx-audio-swift` 当前 README 与模型目录，commit `417df21`，日期 `2026-06-05`。
- LLM 来源：`mlx-swift-lm` 当前 `LLMRegistry` 注册表，commit `a47894a`，日期 `2026-06-04`。
- 时间筛选：Hugging Face `lastModified >= 2025-01-01`。
- 包体积：Hugging Face `usedStorage`，按十进制 GB 四舍五入。`size unavailable` 表示 HF API 返回 `0` 或未给出有效体积。
- Voxt 接入状态：基于 `Voxt/Transcription/MLXModelSupport.swift`、`Voxt/Support/CustomLLMModelSupport.swift`。
- `Qwen3-ForcedAligner` 是强制对齐模型，不是转写模型，本表不计入 ASR / STT。

## ASR / STT 2025+ 上游清单

| 系列 | 2025+ 系列时间 | 当前 MLX 变体 / 包体积 | Voxt 当前状态 | 建议 |
|---|---:|---|---|---|
| Qwen3-ASR 0.6B | 2026-01-29 | `4bit` 0.71 GB, `6bit` 0.86 GB, `8bit` 1.01 GB, `bf16` 1.56 GB | 已接入全部 4 个变体；默认模型是 `4bit` | 保留。建议把 `4bit` 作为默认，`6bit` 作为推荐平衡档，`bf16` 仅高级用户可见 |
| Qwen3-ASR 1.7B | 2026-01-29 | `4bit` 1.60 GB, `6bit` 2.03 GB, `8bit` 2.46 GB, `bf16` 4.08 GB | 已接入全部 4 个变体 | 保留。建议突出 `6bit` / `8bit`，`bf16` 作为高质量大内存档 |
| Voxtral Mini 4B Realtime 2602 | 2026-02-06 | `4bit` 3.15 GB, `6bit` 3.62 GB, `fp16` 8.89 GB | 已接入全部 3 个变体；`6bit` 已 canonical 到实际 HF repo | 全系列隐藏兼容，不再向新用户展示或推荐下载 |
| Cohere Transcribe 03-2026 | 2026-03-26 | `fp16` 4.13 GB | 已接入 | 保留。适合高准确率多语言；体积中等偏大 |
| Parakeet 110M | 2025-05-10 | `tdt_ctc-110m` 0.46 GB | 已接入 | 保留为英文轻量极速档 |
| Parakeet 0.6B | 2025-05-10 / 2025-08-16 | `tdt-v2` 2.47 GB, `tdt-v3` 2.51 GB, `ctc` 2.44 GB, `rnnt` 2.47 GB | 已接入全部 4 个变体 | 建议收敛。保留 `tdt-v3`，其余 `v2` / `ctc` / `rnnt` 可隐藏兼容 |
| Parakeet 1.1B | 2025-05-10 | `tdt` 4.28 GB, `tdt_ctc` 4.29 GB, `ctc` 4.25 GB, `rnnt` 4.28 GB | 已接入全部 4 个变体 | 建议收敛。保留 `tdt_ctc` 或 `tdt` 一个主推荐，其余隐藏兼容 |
| Nemotron 3.5 ASR Streaming 0.6B | 2026-06-05 | base 1.28 GB, `8bit` 0.76 GB | 未接入 | 优先接入。它是上游最新 ASR streaming 系列，体积小，适合作为新实时/流式候选 |
| GLM-ASR Nano 2512 | 2025-12-19 | `4bit` 1.28 GB | 已接入 | 保留为小体积中文/多语言候选，但不要作为默认 |

## LLM 2025+ 上游清单

| 系列 / 模型 | 2025+ 系列时间 | 当前 MLX 变体 / 包体积 | Voxt 当前状态 | 建议 |
|---|---:|---|---|---|
| DeepSeek R1 Distill Qwen 7B | 2025-02-26 | `4bit` 8.58 GB | 未接入本地 Custom LLM | 可选接入。适合推理/代码，但体积偏大 |
| DeepSeek R1 | 2025-02-26 | `4bit` 419.54 GB | 未接入本地 Custom LLM | 不建议内置。体积过大，只适合外部服务 |
| Gemma 3 1B | 2025-04-18 | `qat-4bit` 0.77 GB | 未接入 | 可选轻量档，但优先级低于 Qwen3.5 2B |
| Gemma 3n E2B | 2025-06-29 | `4bit` 2.55 GB, `bf16` 8.95 GB | 未接入 | 可选。Gemma 4 已更新，优先级中低 |
| Gemma 3n E4B | 2025-06-29 | `4bit` 3.90 GB, `bf16` 13.77 GB | 未接入 | 可选。若接入 Gemma 4，可不接入 |
| Gemma 4 E2B | 2026-05-19 | `4bit` 3.61 GB, `bf16` 10.28 GB, `OptiQ-4bit` 16.95 GB, `nvfp4` size unavailable | 已接入 `4bit` | 保留。建议继续保留 `4bit`，暂不内置大体积 OptiQ |
| Gemma 4 E4B | 2026-05-19 | `4bit` 5.25 GB, `bf16` 16.02 GB, `OptiQ-4bit` 22.71 GB, `nvfp4` 6.90 GB | 已接入 `4bit` | 保留。适合作为 Gemma 主推高质量档 |
| Gemma 4 12B | 2026-06-03 | `4bit` 44.02 GB, `bf16` 95.74 GB, `OptiQ-4bit` 8.89 GB, `nvfp4` 44.02 GB | 未接入 | 谨慎接入。仅 `OptiQ-4bit` 体积可接受，但需实测质量/稳定性 |
| Gemma 4 26B-A4B | 2026-05-19 | `4bit` 30.98 GB, `bf16` 51.64 GB, `OptiQ-4bit` 58.96 GB, `nvfp4` 30.98 GB | 未接入 | 不建议默认内置。体积过大 |
| Qwen3 0.6B | 2025-04-28 | `4bit` 0.68 GB | 已接入为隐藏兼容 | 被 Qwen3.5 2B 覆盖 |
| Qwen3 1.7B | 2025-04-28 | `4bit` 0.98 GB | 已接入为隐藏兼容 | 被 Qwen3.5 2B 覆盖 |
| Qwen3 4B | 2025-04-28 | `4bit` 2.27 GB | 已接入为隐藏兼容 | 被 Qwen3.5 4B OptiQ 替代 |
| Qwen3 8B | 2025-04-28 | `4bit` 4.62 GB | 已接入为隐藏兼容 | 被 Qwen3.5 9B OptiQ 替代 |
| Qwen3 30B-A3B | 2025-04-29 | `4bit` 17.19 GB | 已接入为隐藏兼容 | 保持隐藏兼容。体积偏大，不适合默认推荐 |
| Qwen3.5 0.8B | 2026-03-02 | `4bit` 0.65 GB, `bf16` 1.73 GB, `OptiQ-4bit` 886.1 MB, `nvfp4` 0.65 GB | 已接入为隐藏兼容，旧错误 ID 保留 alias | 低配入口改为 Qwen3.5 2B |
| Qwen3.5 2B | 2026-03-02 | `4bit` 1.74 GB, `bf16` 4.45 GB, `OptiQ-4bit` 4.64 GB, `nvfp4` 1.74 GB | 已接入 `4bit` | 保留。建议作为轻量质量档 |
| Qwen3.5 4B | 2026-03-02 | `4bit` 3.05 GB, `bf16` 9.10 GB, `OptiQ-4bit` 6.32 GB | 已接入 `4bit` 和 `OptiQ-4bit` | 强烈保留。建议作为多数用户主推荐 |
| Qwen3.5 9B | 2026-03-02 | `4bit` 5.97 GB, `bf16` 18.84 GB, `OptiQ-4bit` 13.35 GB, `nvfp4` size unavailable | 已接入 `OptiQ-4bit` | 保留为高质量档，但不设默认 |
| Qwen3.5 27B | 2026-02-24 | `4bit` 16.07 GB, `bf16` 54.73 GB, `OptiQ-4bit` 35.46 GB | 未接入 | 可选隐藏接入。仅适合大内存 Mac |
| Qwen3.5 35B-A3B | 2026-02-24 | `4bit` 20.41 GB, `bf16` 70.23 GB, `OptiQ-4bit` 43.80 GB | 未接入 | 不建议默认内置 |
| Qwen3.5 122B-A10B | 2026-02-24 | `4bit` 69.61 GB, `bf16` 245.15 GB | 未接入；远程 oMLX 预设里有同名别名 | 不建议本地内置，保留 oMLX/外部服务路径 |
| Qwen3.5 397B-A17B | 2026-02-16 | `4bit` 594.62 GB, `nvfp4` 223.03 GB | 未接入 | 不建议内置 |
| Qwen3.6 27B | 2026-04-22 | `4bit` 16.07 GB, `bf16` 54.73 GB, `OptiQ-4bit` 35.55 GB, `nvfp4` 16.07 GB | 未接入 | 建议作为高端隐藏候选，优先接 `4bit` 或 `nvfp4` |
| Qwen3.6 35B-A3B | 2026-04-16 | `4bit` 20.42 GB, `bf16` 70.23 GB, `OptiQ-4bit` 44.93 GB, `nvfp4` 20.42 GB | 未接入 | 不建议默认内置 |
| Llama 3.2 1B | 2025-03-05 | `Instruct-4bit` 1.41 GB | 已接入 | 可保留为兼容，但推荐位低于 Qwen3/Qwen3.5 |
| Llama 3.2 3B | 2025-03-05 | `Instruct-4bit` 5.44 GB | 已接入 | 可保留，但不应主推 |
| Granite 3.3 2B | 2025-04-16 | `instruct-4bit` 1.43 GB | 已接入 | 隐藏兼容。结构化能力不够支撑默认展示 |
| MiMo 7B SFT | 2025-05-02 | `4bit` 4.30 GB | 已接入 | 隐藏兼容。保留旧选择，不进默认列表 |
| GLM 4 9B | 2025-04-17 | `4bit` 5.31 GB | 已接入 | 保留。中文/多语言备用价值高 |
| AceReason Nemotron 7B | 2025-05-26 | `4bit` 4.30 GB | 已接入 | 隐藏兼容。推理向备用，不进默认列表 |
| BitNet b1.58 2B-4T | 2025-06-10 | `4bit` 0.72 GB | 未接入 | 可选实验接入。极小体积，但需要重点验证输出稳定性 |
| Baichuan M1 14B | 2025-07-13 | `Instruct-4bit-ft` 8.16 GB | 未接入 | 可选。中文生态价值有，但体积偏大 |
| ERNIE 4.5 0.3B | 2025-07-13 | `PT-bf16-ft` 0.73 GB | 未接入 | 不建议主推。PT 非 Instruct，应用在改写/翻译上风险较高 |
| LFM2 1.2B | 2025-07-11 | `4bit` 0.66 GB | 未接入 | 可选接入。轻量备用 |
| EXAONE 4.0 1.2B | 2025-07-15 | `4bit` 0.72 GB | 未接入 | 可选接入。需要实测中文任务 |
| Lille 130M | 2025-09-05 | `instruct-bf16` 0.25 GB | 未接入 | 不建议主推。体积小但能力可能不足 |
| OLMoE 1B-7B 0125 | 2025-03-04 | `Instruct-4bit` 3.89 GB | 未接入 | 可选。不是优先项 |
| Ling-mini 2.0 | 2025-09-22 | `2bit-DWQ` 6.12 GB | 未接入 | 可选实验项。量化特殊，需要实测 |
| Granite 4.0 H Tiny | 2025-10-03 | `4bit-DWQ` 3.91 GB | 未接入 | 建议评估接入。比 Granite 3.3 更新，可替代 Granite 3.3 |
| LFM2 8B-A1B | 2025-10-08 | `3bit-MLX` 4.17 GB | 未接入 | 可选评估。MoE/A1B 体积合理 |
| nanochat d20 | 2025-10-15 | MLX 1.12 GB | 未接入 | 不建议主推。更适合作为实验/开发参考 |
| gpt-oss 20B | 2026-03-19 | `MXFP4-Q8` 12.10 GB | 不接入本地 | 当前本地输出格式不稳定，不进入 Voxt 本地 LLM 支持计划 |
| AI21 Jamba Reasoning 3B | 2025-10-14 | `4bit` 1.74 GB, `bf16` 6.06 GB | 未接入 | 建议接入 `4bit`。小体积推理向补位 |

## Voxt 当前本地模型覆盖统计

| 类型 | 当前 Voxt 本地可见/支持 | 2025+ 上游已覆盖 | 2025+ 上游未覆盖 | 需要纠正 |
|---|---:|---:|---:|---:|
| MLX ASR / STT | 25 个本地 MLX ASR 选项 | 22 个 2025+ 上游 STT 变体已接入 | 2 个 Nemotron ASR 变体未接入 | 0 |
| MLX Whisper ASR | 5 个 MLX Whisper 选项 | MLX Whisper family | 已接入 | 旧短 ID 迁移到 MLX repo |
| Custom LLM | 可见本地 LLM 已收敛，旧模型隐藏兼容 | 默认可见覆盖 Qwen3.5、Qwen3 VL、Gemma4 E2/E4/12B、GLM4、Mistral3 | MiniCPM、Phi、InternLM、Granite、MiMo、AceReason Nemotron 均转隐藏兼容 | Qwen3.5 0.8B OptiQ repo ID 已修正 |

## 最值得优先接入

| 优先级 | 类型 | 模型 / 系列 | 原因 | 建议动作 |
|---:|---|---|---|---|
| P0 | ASR | Nemotron 3.5 ASR Streaming 0.6B | 上游最新，体积小，streaming 方向清晰，可补足 Voxt 本地 ASR 的新实时路线 | 接入 base 与 `8bit`，先作为实验/新模型标记 |
| P0 | LLM | 修正 Qwen3.5 0.8B OptiQ repo ID | 已验证旧 ID 不可访问，新 ID 可访问 | 已将 `Qwen3.5-0.8B-4bit-OptiQ` alias 到 `Qwen3.5-0.8B-OptiQ-4bit` |
| P0 | LLM | Qwen3.5 4B / 9B OptiQ | 质量/体积比最好，适合 Voxt 的增强、改写、翻译主路径 | 继续主推 4B OptiQ；9B OptiQ 放高质量档 |
| P1 | LLM | AI21 Jamba Reasoning 3B 4bit | 1.74 GB，推理向小模型，体积适中 | 接入并实测结构化输出 |
| P1 | LLM | Granite 4.0 H Tiny | 比 Granite 3.3 更新，体积 3.91 GB | 评估替代 Granite 3.3 |
| P2 | LLM | Qwen3.6 27B 4bit / nvfp4 | 高质量高端模型，体积 16.07 GB | 隐藏接入，按大内存设备展示 |

## 已接入但建议淘汰或降级维护

| 类型 | 当前模型 | 问题 | 替代方案 | 建议 |
|---|---|---|---|---|
| ASR | Parakeet 0.6B `v2` | 已有 `v3`，保留多个同尺寸同家族模型会增加选择噪音 | Parakeet 0.6B `tdt-v3` | 隐藏兼容 |
| ASR | Parakeet 0.6B `ctc` / `rnnt` | 与 `tdt-v3` 目标重叠，用户难以理解差异 | Parakeet 0.6B `tdt-v3` | 隐藏兼容，除非实测证明更优 |
| ASR | Parakeet 1.1B 多个解码变体 | 4 个 4.25-4.29 GB 变体过多 | 只保留 `tdt_ctc` 或 `tdt` 一个推荐 | 收敛为 1 个可见，其他隐藏兼容 |
| ASR | Voxtral `fp16` | 8.89 GB，体积明显高于 4bit / 6bit | Voxtral `6bit` | 降级为高级选项 |
| ASR | MLX Whisper Tiny/Base | 质量落后于 Qwen3-ASR / GLM-ASR Nano | Qwen3-ASR 0.6B 4bit、GLM-ASR Nano | 保留旧配置迁移但降低推荐 |
| LLM | Qwen/Qwen2 1.5B | 旧系列，非 MLX 上游 2025+ 注册项 | Qwen3 1.7B、Qwen3.5 2B | 标记 deprecatedSoon，未来隐藏兼容 |
| LLM | Qwen/Qwen2.5 3B | 旧系列，本地下载体积约 6.18 GB，性价比低于 Qwen3.5 4B | Qwen3.5 4B 4bit / OptiQ | 标记 deprecatedSoon |
| LLM | Qwen2.5 7B hidden compat | 已是隐藏兼容，且 Qwen3/Qwen3.5 更合适 | Qwen3.5 4B / 9B | 继续隐藏，不恢复可见 |
| LLM | Gemma 2 2B / 9B | Gemma 4 已进入上游 2026 支持，提示行为更新 | Gemma 4 E2B / E4B | 降级维护或隐藏旧 Gemma2 |
| LLM | Llama 3 / Llama 3.1 8B | 2024 系列，体积 4.5-5.3 GB，Voxt 任务上未必优于 Qwen3.5 | Qwen3.5 4B / 9B、Gemma4 E4B | 降级为兼容选项 |
| LLM | Mistral 7B / Mistral Nemo | 2024 系列，仍可用但不是 MLX 2025+ 重点 | Mistral 3 3B | 降级为隐藏兼容 |
| LLM | Phi 3.5 Mini | 2024 系列，轻量但能力和生态更新度不如 Qwen3.5 | Qwen3.5 2B、LFM2 1.2B | 隐藏或降级维护 |

## 建议的目标组合

| 用途 | 推荐组合 |
|---|---|
| 默认本地 ASR | Qwen3-ASR 0.6B `4bit` |
| 平衡本地 ASR | Qwen3-ASR 1.7B `6bit` |
| 实时/流式 ASR | Qwen3-ASR 0.6B `4bit` 现用；新增 Nemotron 3.5 ASR Streaming 后做 A/B |
| 英文轻量 ASR | Parakeet 110M 或 Parakeet 0.6B `tdt-v3` |
| 默认本地 LLM | Qwen3.5 4B `OptiQ-4bit` |
| 低内存本地 LLM | Qwen3.5 2B、LFM2 1.2B |
| 高质量本地 LLM | Qwen3.5 9B OptiQ、Gemma4 E4B、GLM 4 9B |
| 高端/实验本地 LLM | Qwen3.6 27B、Gemma4 12B OptiQ |

## 最终目标模型列表

本节是后续代码清理和新增接入的执行口径。`最终可见` 表示模型选择器中主动展示；`隐藏兼容` 表示旧用户配置仍可解析但不再主动推荐；`剔除/不接入` 表示后续可以从可见列表或接入计划中移除。

### 最终 ASR / STT 列表

| 状态 | 模型 | repo / ID | 角色 |
|---|---|---|---|
| 最终可见 | Qwen3-ASR 0.6B 4bit | `mlx-community/Qwen3-ASR-0.6B-4bit` | 默认本地 ASR，低内存、实时友好 |
| 最终可见 | Qwen3-ASR 0.6B 6bit | `mlx-community/Qwen3-ASR-0.6B-6bit` | 轻量平衡档 |
| 最终可见 | Qwen3-ASR 1.7B 6bit | `mlx-community/Qwen3-ASR-1.7B-6bit` | 主推高质量平衡档 |
| 最终可见 | Qwen3-ASR 1.7B 8bit | `mlx-community/Qwen3-ASR-1.7B-8bit` | 高精度本地 ASR |
| 隐藏兼容 | Voxtral Realtime Mini 4B 4bit | `mlx-community/Voxtral-Mini-4B-Realtime-2602-4bit` | 保留已有安装和旧配置兼容 |
| 隐藏兼容 | Voxtral Realtime Mini 4B 6bit | `mlx-community/Voxtral-Mini-4B-Realtime-6bit` | 保留已有安装和旧配置兼容 |
| 最终可见 | Cohere Transcribe 03-2026 fp16 | `beshkenadze/cohere-transcribe-03-2026-mlx-fp16` | 高准确率多语言档 |
| 最终可见 | Parakeet TDT CTC 110M | `mlx-community/parakeet-tdt_ctc-110m` | 英文极速轻量档 |
| 最终可见 | Parakeet TDT 0.6B v3 | `mlx-community/parakeet-tdt-0.6b-v3` | 英文主推轻量档 |
| 最终可见 | Parakeet TDT CTC 1.1B | `mlx-community/parakeet-tdt_ctc-1.1b` | 英文高质量档 |
| 最终可见 | Nemotron 3.5 ASR Streaming 0.6B | `mlx-community/nemotron-3.5-asr-streaming-0.6b` | 新增流式 ASR 候选 |
| 最终可见 | Nemotron 3.5 ASR Streaming 0.6B 8bit | `mlx-community/nemotron-3.5-asr-streaming-0.6b-8bit` | 新增小体积流式 ASR 候选 |
| 最终可见 | GLM-ASR Nano 2512 4bit | `mlx-community/GLM-ASR-Nano-2512-4bit` | 小体积中文/多语言备用 |
| 隐藏兼容 | Qwen3-ASR 0.6B 8bit | `mlx-community/Qwen3-ASR-0.6B-8bit` | 与 6bit / 1.7B 档位重叠，保留旧配置 |
| 隐藏兼容 | Qwen3-ASR 0.6B bf16 | `mlx-community/Qwen3-ASR-0.6B-bf16` | 体积和资源占用偏高，保留高级兼容 |
| 隐藏兼容 | Qwen3-ASR 1.7B 4bit | `mlx-community/Qwen3-ASR-1.7B-4bit` | 与 0.6B 6bit / 1.7B 6bit 定位重叠 |
| 隐藏兼容 | Qwen3-ASR 1.7B bf16 | `mlx-community/Qwen3-ASR-1.7B-bf16` | 高资源占用高级档 |
| 隐藏兼容 | Voxtral Realtime Mini 4B fp16 | `mlx-community/Voxtral-Mini-4B-Realtime-2602-fp16` | 体积 8.89 GB，仅保留高级兼容 |
| 隐藏兼容 | Parakeet TDT 0.6B v2 | `mlx-community/parakeet-tdt-0.6b-v2` | 被 `tdt-v3` 替代 |
| 隐藏兼容 | Parakeet CTC 0.6B | `mlx-community/parakeet-ctc-0.6b` | 与 `tdt-v3` 目标重叠 |
| 隐藏兼容 | Parakeet RNNT 0.6B | `mlx-community/parakeet-rnnt-0.6b` | 与 `tdt-v3` 目标重叠 |
| 隐藏兼容 | Parakeet TDT 1.1B | `mlx-community/parakeet-tdt-1.1b` | 与 `tdt_ctc-1.1b` 重叠 |
| 隐藏兼容 | Parakeet CTC 1.1B | `mlx-community/parakeet-ctc-1.1b` | 与 `tdt_ctc-1.1b` 重叠 |
| 隐藏兼容 | Parakeet RNNT 1.1B | `mlx-community/parakeet-rnnt-1.1b` | 与 `tdt_ctc-1.1b` 重叠 |
| 隐藏兼容 | MLX Whisper Tiny / Base | `mlx-community/whisper-tiny-mlx`, `mlx-community/whisper-base-mlx` | 旧配置迁移目标，不作为主推 |
| 可见保留 | MLX Whisper Small / Large v3 Turbo / Large v3 | `mlx-community/whisper-small-mlx`, `mlx-community/whisper-large-v3-turbo`, `mlx-community/whisper-large-v3-mlx` | 保留 Whisper 系列但只走 MLX Audio |
| 剔除/不推荐 | FireRed ASR 2 | `mlx-community/FireRedASR2-AED-mlx` | 非本轮 2025+ 上游 MLX STT 主清单，体积较大 |
| 剔除/不推荐 | SenseVoice Small | `mlx-community/SenseVoiceSmall` | 非本轮 2025+ 上游 MLX STT 主清单 |
| 剔除/不推荐 | Granite Speech 4.0 1B | `mlx-community/granite-4.0-1b-speech-5bit` | 当前调研源未列入 `mlx-audio-swift` STT 主 README 清单 |

### 最终 LLM 列表

| 状态 | 模型 | repo / ID | 角色 |
|---|---|---|---|
| 最终可见 | Qwen3.5 2B 4bit | `mlx-community/Qwen3.5-2B-4bit` | 轻量质量档 |
| 最终可见 | Qwen3.5 4B OptiQ 4bit | `mlx-community/Qwen3.5-4B-OptiQ-4bit` | 主推质量/体积比最佳档 |
| 最终可见 | Qwen3.5 9B OptiQ 4bit | `mlx-community/Qwen3.5-9B-OptiQ-4bit` | 高质量本地档 |
| 最终可见 | Qwen3 VL 4B Instruct | `lmstudio-community/Qwen3-VL-4B-Instruct-MLX-4bit` | 视觉入口，不替代文本默认模型 |
| 最终可见 | Gemma 4 E2B IT 4bit | `mlx-community/gemma-4-e2b-it-4bit` | Gemma 轻量路线 |
| 最终可见 | Gemma 4 E4B IT 4bit | `mlx-community/gemma-4-e4b-it-4bit` | Gemma 主推质量档 |
| 最终可见 | Gemma 4 12B IT OptiQ 4bit | `mlx-community/gemma-4-12B-it-OptiQ-4bit` | Gemma 高端质量档，约 8.96 GB |
| 最终可见 | GLM 4 9B | `mlx-community/GLM-4-9B-0414-4bit` | 中文/多语言高质量备用 |
| 隐藏兼容 | GLM-Z1 9B 0414 4bit | `mlx-community/GLM-Z1-9B-0414-4bit` | 推理向中文备用，不进入默认入口 |
| 隐藏兼容 | AceReason Nemotron 7B 4bit | `mlx-community/AceReason-Nemotron-7B-4bit` | 推理向备用，不进入默认入口 |
| 最终可见 | LFM2 1.2B 4bit | `mlx-community/LFM2-1.2B-4bit` | 极轻量 LFM2 入口 |
| 最终可见 | LFM2 8B A1B 3bit | `mlx-community/LFM2-8B-A1B-3bit-MLX` | MoE LFM2 入口 |
| 最终可见 | Qwen3.6 27B 4bit | `mlx-community/Qwen3.6-27B-4bit` | 高端大内存 Qwen 入口 |
| 隐藏兼容 | Granite 4.0 H Tiny 4bit DWQ | `mlx-community/Granite-4.0-H-Tiny-4bit-DWQ` | 有实验价值，但不进入默认入口 |
| 隐藏兼容 | Qwen3 0.6B 4bit | `mlx-community/Qwen3-0.6B-4bit` | 能力偏弱，保留旧配置 |
| 隐藏兼容 | Qwen3 1.7B 4bit | `mlx-community/Qwen3-1.7B-4bit` | 被 Qwen3.5 2B 覆盖 |
| 隐藏兼容 | Qwen3 4B 4bit | `mlx-community/Qwen3-4B-4bit` | 被 Qwen3.5 4B 替代 |
| 隐藏兼容 | Qwen3 8B 4bit | `mlx-community/Qwen3-8B-4bit` | 被 Qwen3.5 9B 替代 |
| 隐藏兼容 | Qwen3 30B-A3B 4bit | `mlx-community/Qwen3-30B-A3B-4bit` | 体积偏大，保留旧配置 |
| 隐藏兼容 | Qwen3.5 4B 4bit | `mlx-community/Qwen3.5-4B-4bit` | 默认入口使用 Qwen3.5 4B OptiQ |
| 隐藏兼容 | MiMo 7B SFT 4bit | `mlx-community/MiMo-7B-SFT-4bit` | 保留但不主推，需实测中文质量 |
| 隐藏兼容 | EXAONE 4.0 1.2B 4bit | `mlx-community/exaone-4.0-1.2b-4bit` | 轻量实验备用 |
| 隐藏兼容 | BitNet b1.58 2B-4T 4bit | `mlx-community/bitnet-b1.58-2B-4T-4bit` | 极小体积实验档，需验证稳定性 |
| 剔除/不推荐 | Qwen2 1.5B Instruct | `Qwen/Qwen2-1.5B-Instruct` | 被 Qwen3 / Qwen3.5 替代 |
| 剔除/不推荐 | Qwen2.5 3B Instruct | `Qwen/Qwen2.5-3B-Instruct` | 体积/质量比低于 Qwen3.5 4B |
| 剔除/不推荐 | Qwen2.5 7B Instruct 4bit | `mlx-community/Qwen2.5-7B-Instruct-4bit` | 被 Qwen3.5 替代 |
| 剔除/不推荐 | Gemma 2 2B / 9B | `mlx-community/gemma-2-2b-it-4bit`, `mlx-community/gemma-2-9b-it-4bit` | 被 Gemma 4 E2B / E4B 替代 |
| 隐藏兼容 | Llama 3 / Llama 3.1 8B | `mlx-community/Meta-Llama-3-8B-Instruct-4bit`, `mlx-community/Meta-Llama-3.1-8B-Instruct-4bit` | 2024 系列，不再作为本地主推 |
| 隐藏兼容 | Llama 3.2 1B / 3B | `mlx-community/Llama-3.2-1B-Instruct-4bit`, `mlx-community/Llama-3.2-3B-Instruct-4bit` | 被 Qwen3.5 2B、LFM2 轻量入口替代 |
| 最终可见 | Mistral 3 3B | `mlx-community/Ministral-3-3B-Instruct-2512-4bit` | 新 Mistral3 轻量入口 |
| 隐藏兼容 | Mistral 7B / Mistral Nemo | `mlx-community/Mistral-7B-Instruct-v0.3-4bit`, `mlx-community/Mistral-Nemo-Instruct-2407-4bit` | 2024 系列，不再作为本地主推 |
| 隐藏兼容 | Phi 3.5 Mini | `mlx-community/Phi-3.5-mini-instruct-4bit` | 被 Qwen3.5 轻量档替代 |
| 隐藏兼容 | Granite 3.3 2B | `mlx-community/granite-3.3-2b-instruct-4bit` | 默认价值不足，保留兼容 |
| 隐藏兼容 | InternLM2.5 7B | `mlx-community/internlm2_5-7b-chat-4bit` | 与 Qwen3.5 / GLM 重叠，保留兼容 |
| 隐藏兼容 | MiniCPM4 8B | `mlx-community/MiniCPM4-8B-4bit` | 与 Qwen3.5 / GLM 重叠，保留兼容 |
| 隐藏兼容 | GLM-4 9B Chat 1M | `mlx-community/glm-4-9b-chat-1m-4bit` | 超长上下文不是 Voxt 当前核心需求，保留旧配置 / 已安装显示 |
| 剔除/不推荐 | Baichuan M1 14B | `mlx-community/Baichuan-M1-14B-Instruct-4bit-ft` | 体积偏大，优先级低 |
| 剔除/不推荐 | ERNIE 4.5 0.3B PT | `mlx-community/ERNIE-4.5-0.3B-PT-bf16-ft` | PT 非 Instruct，不适合作为 Voxt 改写/翻译主路径 |
| 剔除/不推荐 | Lille 130M | `mlx-community/lille-130m-instruct-bf16` | 能力预期不足 |
| 剔除/不推荐 | OLMoE 1B-7B 0125 | `mlx-community/OLMoE-1B-7B-0125-Instruct-4bit` | 不是优先接入项 |
| 剔除/不推荐 | Ling-mini 2.0 | `mlx-community/Ling-mini-2.0-2bit-DWQ` | 特殊量化，先不进入目标列表 |
| 剔除/不推荐 | nanochat d20 | `dnakov/nanochat-d20-mlx` | 实验/开发参考价值高于产品价值 |
| 剔除/不推荐 | AI21 Jamba Reasoning 3B 4bit | `mlx-community/AI21-Jamba-Reasoning-3B-4bit` | 推理向定位和当前 Voxt 默认任务重叠，暂不进入本批 5 个 |
| 剔除/不推荐 | Qwen3.5 27B+ / 35B+ / 122B+ / 397B+ | 见上游清单 | 本地体积过大，保留 oMLX/外部服务路径 |
| 剔除/不推荐 | Qwen3.6 35B-A3B | `mlx-community/Qwen3.6-35B-A3B-4bit` 等 | 体积过大，不进入目标列表 |

### 最终数量统计

| 类型 | 最终可见 | 隐藏兼容 / 实验 | 剔除/不推荐 |
|---|---:|---:|---:|
| ASR / STT | 13 | 14 | 5 |
| LLM | 15 | 10 | 22+ |

## 最终精简模型组与系列

本节优先级高于上一节的 13 个 ASR / 15 个 LLM 目标列表，是后续代码清理、模型剔除和新增接入的执行口径。精简原则：按“组”保留能力边界，按“系列”收敛重复型号；同能力模型优先保留安装更方便、repo 命名更稳定、用户更容易理解的普通 `4bit` / `6bit` 版本。

### 分组规则

| 类型 | 组 | 保留系列 | 处理原则 |
|---|---|---|---|
| ASR / STT | 核心通用 ASR | Qwen3-ASR | 作为默认和主推质量档，只保留 0.6B 4bit、1.7B 6bit、1.7B 8bit |
| ASR / STT | 流式 / 实时 ASR | Nemotron 3.5 ASR Streaming | Nemotron 保留小体积流式入口；Voxtral 全系列转为隐藏兼容 |
| ASR / STT | 英文专用 ASR | Parakeet | 只保留 0.6B `tdt-v3` 可见，其余同尺寸解码变体隐藏或剔除 |
| ASR / STT | 高准确率批处理 | Cohere Transcribe | 只保留一个 fp16 入口，面向高性能和高准确率场景 |
| LLM | 核心通用 LLM | Qwen3.5、Qwen3.6 | Qwen3.5 是主线；Qwen3.6 只保留一个高端入口 |
| LLM | 非 Qwen 备用 | Gemma 4、GLM-4、Mistral 3、LFM2 | 覆盖不同提示风格、中文/多语言和轻量非 Qwen 路线 |
| LLM | 推理 / 实验池 | AceReason Nemotron、AI21 Jamba、Granite 4、BitNet、LFM2、EXAONE | 默认隐藏，保留后续 A/B 和高级用户入口 |
| LLM | 历史兼容 | Qwen2 / Qwen2.5、Gemma 2、Llama 3.x、Mistral、Phi、Granite 3.3 等 | 不再主动展示；旧配置解析优先通过隐藏兼容完成 |

### 设备阶段推荐

| 设备阶段 | ASR / STT 主推 | LLM 主推 | 说明 |
|---|---|---|---|
| 低性能电脑 | Qwen3-ASR 0.6B 4bit；需要流式时用 Nemotron 3.5 ASR Streaming 8bit | Qwen3.5 2B 4bit | 低下载体积、低内存、低配置成本 |
| 中性能电脑 | Qwen3-ASR 1.7B 6bit；实时方向用 Nemotron 3.5 ASR Streaming | Qwen3.5 4B OptiQ；中文/多语言备用用 GLM 4 9B；非 Qwen 备用用 Gemma 4 E4B 4bit | 主推档，覆盖大多数用户 |
| 高性能电脑 | Qwen3-ASR 1.7B 8bit；高准确率批处理用 Cohere Transcribe fp16 | Qwen3.5 9B OptiQ；高端实验用 Qwen3.6 27B 4bit | 只保留少量高性能入口，不展开大模型全家桶 |

### ASR / STT 最终可见模型

| 组 | 阶段 | 系列 | 保留模型 | repo / ID | 包体积 | 保留原因 |
|---|---|---|---|---|---:|---|
| 核心通用 ASR | 低性能 | Qwen3-ASR 0.6B | 4bit | `mlx-community/Qwen3-ASR-0.6B-4bit` | 0.71 GB | 默认模型，安装小，功能完整，实时友好 |
| 核心通用 ASR | 中性能 | Qwen3-ASR 1.7B | 6bit | `mlx-community/Qwen3-ASR-1.7B-6bit` | 2.03 GB | 主推高质量平衡档，替代 0.6B 6bit / 1.7B 4bit |
| 核心通用 ASR | 高性能 | Qwen3-ASR 1.7B | 8bit | `mlx-community/Qwen3-ASR-1.7B-8bit` | 2.46 GB | 高精度本地 ASR，仍比 bf16 更可控 |
| 流式 / 实时 ASR | 低性能 | Nemotron 3.5 ASR Streaming 0.6B | 8bit | `mlx-community/nemotron-3.5-asr-streaming-0.6b-8bit` | 0.76 GB | 新流式路线，比 base 更适合低配尝试 |
| 隐藏兼容 ASR | 中性能 | Voxtral Mini 4B Realtime 2602 | 6bit | `mlx-community/Voxtral-Mini-4B-Realtime-6bit` | 3.62 GB | 不再默认展示，仅保留已有安装和旧配置兼容 |
| 英文专用 ASR | 低 / 中性能 | Parakeet 0.6B | `tdt-v3` | `mlx-community/parakeet-tdt-0.6b-v3` | 2.51 GB | 仅保留一个英文轻量入口，删除其他 0.6B 解码变体 |
| 高准确率批处理 | 高性能 | Cohere Transcribe 03-2026 | fp16 | `beshkenadze/cohere-transcribe-03-2026-mlx-fp16` | 4.13 GB | 高准确率多语言批处理入口 |

### ASR / STT 系列收敛结论

| 处理 | 系列 / 模型 | 原因 |
|---|---|---|
| 可见保留 | Qwen3-ASR 0.6B 4bit、Qwen3-ASR 1.7B 6bit、Qwen3-ASR 1.7B 8bit | 三个档位边界清晰：默认、主推平衡、高精度 |
| 隐藏兼容 | Qwen3-ASR 0.6B 6bit / 8bit / bf16 | 与 0.6B 4bit、1.7B 6bit、1.7B 8bit 重叠 |
| 隐藏兼容 | Qwen3-ASR 1.7B 4bit / bf16 | 4bit 被 0.6B 4bit 和 1.7B 6bit 夹住；bf16 资源占用偏高 |
| 可见保留 | Nemotron 3.5 ASR Streaming 0.6B 8bit | 作为新流式路线的唯一可见入口 |
| 隐藏兼容 | Nemotron 3.5 ASR Streaming base | 先主推 8bit，base 等实测后再决定是否恢复可见 |
| 隐藏兼容 | Voxtral Mini 4B Realtime 6bit | 不再默认展示，保留已有安装和旧配置兼容 |
| 隐藏兼容 | Voxtral Mini 4B Realtime 4bit / fp16 | 保留已有安装和旧配置兼容；fp16 体积 8.89 GB |
| 可见保留 | Parakeet 0.6B `tdt-v3` | 英文轻量只保留一个最清晰的新版本 |
| 隐藏兼容 | Parakeet 110M、Parakeet TDT CTC 1.1B、GLM-ASR Nano 2512 4bit | 有特定场景价值，但不进入默认可见列表 |
| 剔除/不推荐 | Parakeet 0.6B v2 / CTC / RNNT、Parakeet 1.1B TDT / CTC / RNNT | 同系列同尺寸变体太多，安装和解释成本高 |
| 剔除/不推荐 | MLX Whisper Tiny / Base、FireRed ASR 2、SenseVoice Small、Granite Speech 4.0 1B | 非精简主线或被 Qwen3-ASR / Nemotron / Cohere 覆盖 |

### LLM 最终可见模型

| 组 | 阶段 | 系列 | 保留模型 | repo / ID | 包体积 | 保留原因 |
|---|---|---|---|---|---:|---|
| 核心通用 LLM | 低性能 | Qwen3.5 2B | 4bit | `mlx-community/Qwen3.5-2B-4bit` | 1.74 GB | 轻量但输出更稳，作为低配质量档 |
| 核心通用 LLM | 中性能 | Qwen3.5 4B | OptiQ 4bit | `mlx-community/Qwen3.5-4B-OptiQ-4bit` | 6.32 GB | 主推默认 LLM，普通 4bit 仅隐藏兼容 |
| 非 Qwen 备用 | 中性能 | Gemma 4 E4B IT | 4bit | `mlx-community/gemma-4-e4b-it-4bit` | 5.25 GB | 非 Qwen 主线备用，能力和体积平衡 |
| 非 Qwen 备用 | 中 / 高性能 | GLM 4 9B | 4bit | `mlx-community/GLM-4-9B-0414-4bit` | 5.31 GB | 中文/多语言高质量备用，配置简单 |
| 核心通用 LLM | 高性能 | Qwen3.5 9B | OptiQ 4bit | `mlx-community/Qwen3.5-9B-OptiQ-4bit` | 13.35 GB | 高质量本地档 |
| 核心通用 LLM | 高性能实验 | Qwen3.6 27B | 4bit | `mlx-community/Qwen3.6-27B-4bit` | 16.07 GB | 只保留一个高端大内存入口 |

### LLM 系列收敛结论

| 处理 | 系列 / 模型 | 原因 |
|---|---|---|
| 可见保留 | Qwen3.5 2B / 4B OptiQ / 9B OptiQ | 形成低配质量、默认主推、高质量三个清晰档位 |
| 隐藏兼容 | Qwen3.5 0.8B OptiQ | 低配入口改为 2B，0.8B 仅保留历史选择 / 已安装显示 |
| 隐藏兼容 | Qwen3.5 4B 普通 4bit | 默认可见优先 Qwen3.5 4B OptiQ |
| 隐藏兼容 | Qwen3 1.7B / 4B / 8B / 30B-A3B | Qwen3.5 已覆盖主线能力，Qwen3 只保留旧配置 |
| 可见保留 | Qwen3.6 27B 4bit | 高端阶段只保留一个大模型入口 |
| 剔除/不推荐 | Qwen3.5 27B+ / 35B+ / 122B+ / 397B+、Qwen3.6 35B-A3B | 本地体积过大，保留 oMLX/外部服务路径 |
| 可见保留 | Gemma 4 E4B 4bit、GLM 4 9B、Mistral 3 3B、LFM2 1.2B / 8B A1B | 分别覆盖非 Qwen 主线、中文/多语言高质量、Mistral 轻量入口、轻量非 Qwen |
| 隐藏兼容 | Gemma 4 E2B、Gemma 4 12B OptiQ | E2B 与 Qwen3.5 2B / Gemma E4B 重叠；12B 仅高端实验 |
| 隐藏兼容 | AceReason Nemotron 7B、AI21 Jamba Reasoning 3B、Granite 4.0 H Tiny | 能力有价值，但与 Qwen3.5 / GLM / Gemma 主线重叠，先作为实验池 |
| 隐藏兼容 | LFM2 1.2B、EXAONE 1.2B、BitNet 2B、MiMo 7B | 保留实验价值，不进入默认可见列表 |
| 剔除/不推荐 | Qwen2 / Qwen2.5 系列 | 被 Qwen3.5 全面替代 |
| 隐藏兼容 | Llama 3 / 3.1 / 3.2、Mistral 7B / Nemo | 保留旧配置 / 已安装显示，不进入默认入口 |
| 隐藏兼容 | Gemma 2、Phi 3.5 | 旧系列或同能力下不如 Qwen3.5 / Gemma4 方便 |
| 隐藏兼容 | Granite 3.3、InternLM2.5、MiniCPM4、GLM-4 Chat 1M | 不在精简主线，或已有更清晰替代 |
| 剔除/不推荐 | Baichuan M1、ERNIE 4.5 PT、Lille、OLMoE、Ling-mini、nanochat | 体积、任务适配、量化形态或产品价值不适合作为 Voxt 主列表 |

### 分组后数量统计

| 类型 | 可见核心模型 | 隐藏兼容 / 实验池 | 剔除/不推荐 |
|---|---:|---:|---:|
| ASR / STT | 7 | 11 | 11+ |
| LLM | 8 | 17 | 22+ |

### 执行顺序

| 优先级 | 动作 | 范围 |
|---:|---|---|
| P0 | 将最终可见模型列表作为模型选择器默认展示口径 | ASR 7 个、LLM 11 个 |
| P0 | 修正 Qwen3.5 0.8B OptiQ repo ID，并把 OptiQ 系列降为隐藏兼容 | LLM |
| P1 | 接入 Nemotron 3.5 ASR Streaming 0.6B 8bit | ASR |
| P1 | 接入 Qwen3.5 9B 4bit、LFM2 系列 | LLM |
| P1 | 将同系列重复变体转为隐藏兼容 | Qwen3-ASR、Parakeet、Voxtral、Qwen3、Qwen3.5 OptiQ |
| P2 | 删除或不再展示剔除/不推荐组 | MLX Whisper Tiny/Base、旧 LLM 系列、非主线实验模型 |

## 来源

- `mlx-audio-swift` README：<https://github.com/Blaizzy/mlx-audio-swift/blob/417df212f54b8b4214a9815c1cd2eabb05fd4fdf/README.md>
- `mlx-audio-swift` Qwen3ASR：<https://github.com/Blaizzy/mlx-audio-swift/blob/417df212f54b8b4214a9815c1cd2eabb05fd4fdf/Sources/MLXAudioSTT/Models/Qwen3ASR/README.md>
- `mlx-audio-swift` Voxtral Realtime：<https://github.com/Blaizzy/mlx-audio-swift/blob/417df212f54b8b4214a9815c1cd2eabb05fd4fdf/Sources/MLXAudioSTT/Models/VoxtralRealtime/README.md>
- `mlx-audio-swift` Cohere Transcribe：<https://github.com/Blaizzy/mlx-audio-swift/blob/417df212f54b8b4214a9815c1cd2eabb05fd4fdf/Sources/MLXAudioSTT/Models/CohereTranscribe/README.md>
- `mlx-audio-swift` Parakeet：<https://github.com/Blaizzy/mlx-audio-swift/blob/417df212f54b8b4214a9815c1cd2eabb05fd4fdf/Sources/MLXAudioSTT/Models/Parakeet/README.md>
- `mlx-audio-swift` Nemotron ASR：<https://github.com/Blaizzy/mlx-audio-swift/blob/417df212f54b8b4214a9815c1cd2eabb05fd4fdf/Sources/MLXAudioSTT/Models/NemotronASR/README.md>
- `mlx-audio-swift` GLMASR：<https://github.com/Blaizzy/mlx-audio-swift/blob/417df212f54b8b4214a9815c1cd2eabb05fd4fdf/Sources/MLXAudioSTT/Models/GLMASR/README.md>
- `mlx-swift-lm` LLMRegistry：<https://raw.githubusercontent.com/ml-explore/mlx-swift-lm/a47894a1e7e963b24bd48c030f5fc1d1627e60e9/Libraries/MLXLLM/LLMModelFactory.swift>
- Hugging Face API 示例：<https://huggingface.co/api/models/mlx-community/Qwen3-ASR-0.6B-4bit>
- Hugging Face API 示例：<https://huggingface.co/api/models/mlx-community/Qwen3.5-2B-4bit>
- Hugging Face API 示例：<https://huggingface.co/api/models/mlx-community/gemma-4-e2b-it-4bit>
