# ASR 模型系列与模型名称清单

本分支以 `Voxt/Transcription/MLXModelSupport.swift` 为唯一目录来源。所有受支持的本地模型均可见，不再有“已安装后重新显示”的隐藏模型。

## 本地 ASR

系统听写 `dictation` 保留，无需下载本地 MLX 权重。MLX ASR 保留以下 11 个模型：

| 系列 | 显示名称 | Repo ID | 主要路径 |
|---|---|---|---|
| Whisper | Whisper Large v3 Turbo | `mlx-community/whisper-large-v3-turbo` | Batch preview / Final |
| Whisper | Whisper Large v3 | `mlx-community/whisper-large-v3-mlx` | Batch preview / Final |
| Whisper | Whisper Small | `mlx-community/whisper-small-mlx` | Batch preview / Final |
| Qwen3 | Qwen3 0.6B (4bit) | `mlx-community/Qwen3-ASR-0.6B-4bit` | 默认；native Qwen live |
| Qwen3 | Qwen3 1.7B (6bit) | `mlx-community/Qwen3-ASR-1.7B-6bit` | Native Qwen live |
| Qwen3 | Qwen3 1.7B (8bit) | `mlx-community/Qwen3-ASR-1.7B-8bit` | Native Qwen live |
| Cohere | Cohere 03-2026 | `beshkenadze/cohere-transcribe-03-2026-mlx-fp16` | Native streaming |
| MOSS | MOSS | `OpenMOSS-Team/MOSS-Transcribe-Diarize` | Native streaming；结构化时间戳 / 说话人 |
| Parakeet | Parakeet v3 | `mlx-community/parakeet-tdt-0.6b-v3` | Batch preview / Final；欧洲 25 语 |
| Nemotron | Nemotron | `mlx-community/nemotron-3.5-asr-streaming-0.6b-8bit` | Native Nemotron live |
| SenseVoice | SenseVoice | `mlx-community/SenseVoiceSmall` | Batch preview / Final；语言 / 情绪 / 事件 |

语言矩阵、时间戳粒度、VAD 策略由同一 capability 注册表提供，不按 repo 名称猜测。

## 说话人分离与 VAD

- 独立会议说话人分离：MLXAudioVAD 的 Sortformer v2。
- MOSS 原生说话人输出保留。
- FluidAudio、Offline VBx 及其流式回退全部移除。
- OmniVAD、MLX Silero 与能量兜底保留，不依赖 FluidAudio。
- 不把 VBx 人数提示直接映射为 Sortformer 的模型结构参数。

## 已移除模型与旧配置

Voxtral、Canary、Moonshine、Wav2Vec2、MMS、GLM-ASR Nano、Granite Speech、FireRed 及各系列隐藏量化变体不再提供本地加载、下载或配置入口。

旧 ID 只用于迁移，不代表继续支持旧权重：

| 原选择 | 迁移目标 |
|---|---|
| Whisper Tiny / Base（包括旧短 ID） | Whisper Small |
| 旧 Parakeet 变体 | Parakeet v3 |
| Qwen3-ASR 0.6B 隐藏量化 | Qwen3-ASR 0.6B 4bit |
| Qwen3-ASR 1.7B 隐藏量化 | Qwen3-ASR 1.7B 6bit |
| 已移除的其他本地 ASR | 默认 Qwen3-ASR 0.6B 4bit |
| Offline VBx | Sortformer v2 |

旧缓存不自动删除；替代模型尚未安装时使用现有下载 / 配置入口。历史会议文本、说话人标签与用户改名不重写。

## 远程 ASR

远程供应商、模型选择和配置流程保持不变，不受上述本地模型清理影响。参见 [远程模型配置](RemoteModel.zh-CN.md)，实际选项以 `Voxt/Core/Models/RemoteModelConfiguration.swift` 为准。

## 相关文档

- [现代化方案](ModelStackModernizationPlan.zh-CN.md)
- [实施与验证状态](ModelStackModernizationImplementation.zh-CN.md)
- [MLX 依赖策略](MLXAudioDependency.md)
