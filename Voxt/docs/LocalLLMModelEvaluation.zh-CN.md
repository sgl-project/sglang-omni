# 本地 LLM 模型最终状态列表

日期：2026-06-29

本次精简已移除 Hy-MT2 Q6_K，以及下方“隐藏兼容”中列出的旧版本地 LLM；保留条目才代表当前仍可选的模型。

## 默认可见

| 状态 | 模型 | repo | 备注 |
|---|---|---|---|
| 保留 | Qwen3.5 2B 4bit | `mlx-community/Qwen3.5-2B-4bit` | 轻量质量档 |
| 保留，设为默认 | Qwen3.5 4B OptiQ 4bit | `mlx-community/Qwen3.5-4B-OptiQ-4bit` | 默认本地 LLM |
| 保留 | Qwen3.5 9B OptiQ 4bit | `mlx-community/Qwen3.5-9B-OptiQ-4bit` | 高质量档 |
| 保留 | Gemma 4 E4B IT 4bit | `mlx-community/gemma-4-e4b-it-4bit` | 非 Qwen 备用 |
| 保留 | GLM 4 9B | `mlx-community/GLM-4-9B-0414-4bit` | 中文 / 多语言备用 |
| 新增保留 | LFM2 1.2B 4bit | `mlx-community/LFM2-1.2B-4bit` | 极轻量 LFM2 入口，约 663 MB |
| 新增保留 | LFM2 8B A1B 3bit | `mlx-community/LFM2-8B-A1B-3bit-MLX` | MoE LFM2 入口，约 4.18 GB |
| 新增保留 | Qwen3.6 27B 4bit | `mlx-community/Qwen3.6-27B-4bit` | 高端大内存 Qwen 入口，约 16.08 GB |

## 翻译 GGUF

| 状态 | 模型 | repo / 文件 | 备注 |
|---|---|---|---|
| 保留 | Hy-MT2 1.8B Q4_K_M | `tencent/Hy-MT2-1.8B-GGUF#Hy-MT2-1.8B-Q4_K_M.gguf` | 默认可见，低延迟翻译入口 |
| 保留 | Hy-MT2 1.8B Q8_0 | `tencent/Hy-MT2-1.8B-GGUF#Hy-MT2-1.8B-Q8_0.gguf` | 默认可见，高质量翻译入口 |

## 视觉 / 实验入口

| 状态 | 模型 | repo | 备注 |
|---|---|---|---|
| 视觉入口 | Qwen3 VL 4B Instruct | `lmstudio-community/Qwen3-VL-4B-Instruct-MLX-4bit` | 默认不替代文本模型 |
| 保留 | Gemma 4 E2B IT 4bit | `mlx-community/gemma-4-e2b-it-4bit` | Gemma 轻量档 |
| 保留 / 高端实验 | Gemma 4 12B IT OptiQ 4bit | `mlx-community/gemma-4-12B-it-OptiQ-4bit` | Gemma 高质量档，约 8.96 GB |

## 隐藏兼容

| 状态 | 系列 / 模型 | repo | 备注 |
|---|---|---|---|
| 隐藏兼容 | Qwen2 1.5B Instruct | `Qwen/Qwen2-1.5B-Instruct` | 当前默认迁移到 Qwen3.5 4B OptiQ |
| 隐藏兼容 | Qwen2.5 3B Instruct | `Qwen/Qwen2.5-3B-Instruct` | 旧选择继续可用 |
| 隐藏兼容 | Qwen2.5 VL 3B | `mlx-community/Qwen2.5-VL-3B-Instruct-4bit` | 被 Qwen3 VL 覆盖 |
| 隐藏兼容 | Qwen3 0.6B / 1.7B / 4B / 8B | `mlx-community/Qwen3-*` | 被 Qwen3.5 覆盖 |
| 隐藏兼容 | Qwen3.5 0.8B OptiQ 4bit | `mlx-community/Qwen3.5-0.8B-OptiQ-4bit` | 体积优势不足以单独占默认入口；低配入口改为 Qwen3.5 2B |
| 隐藏兼容 | Qwen3.5 4B 4bit | `mlx-community/Qwen3.5-4B-4bit` | 默认入口使用 Qwen3.5 4B OptiQ，普通 4bit 仅兼容旧选择 |
| 保留 | Mistral 3 3B | `mlx-community/Ministral-3-3B-Instruct-2512-4bit` | 轻量非 Qwen / 欧洲语系备用 |
| 隐藏兼容 | Gemma 2 / Gemma 3 / Gemma 3n | `mlx-community/gemma-2-*`, `mlx-community/gemma-3-*`, `mlx-community/gemma-3n-*` | Gemma 仅默认展示 4 E2B / 4 E4B / 4 12B，旧 Gemma 保留隐藏支持 |
| 隐藏兼容 | Qwen3 30B A3B | 现有 hiddenCompat | 继续隐藏兼容 |

## 默认隐藏 / 观察池

| 状态 | 模型 / 架构 | repo / model_type | 备注 |
|---|---|---|---|
| 观察 | EXAONE4 | `exaone4` | 多语言候选 |
| 新增 | Mistral3 | `mistral3` | 通过 `Ministral-3-3B-Instruct-2512-4bit` 接入 |
| 新增可见支持 | LFM2 | `lfm2` | 通过 `LFM2-1.2B-4bit` 和 `LFM2-8B-A1B-3bit-MLX` 接入 |
| 不进计划 | Jamba 3B | `jamba_3b` | 公开可下载 repo 需另行确认 |

## 必须调整

| 优先级 | 状态 | 项目 | 目标 |
|---:|---|---|---|
| P0 | 已完成 | Qwen3.5 0.8B repo ID | `mlx-community/Qwen3.5-0.8B-4bit-OptiQ` -> `mlx-community/Qwen3.5-0.8B-OptiQ-4bit`，保留旧 ID alias |
| P0 | 已完成 | 默认本地 LLM | `Qwen/Qwen2-1.5B-Instruct` -> `mlx-community/Qwen3.5-4B-OptiQ-4bit` |
| P0 | 精简 | 默认可见模型 | 核心文本模型控制在 6 个 |
| P1 | 调整 | 旧模型 | 转隐藏兼容，保留历史选择、已下载显示和卸载路径 |
| P1 | 新增 | 模型分组 | 轻量 / 平衡 / 高质量 / 视觉 / 实验 |
| P1 | 新增 | 模型健康检查 | repo 可访问、`model_type` 支持、chat template、关键文件完整性 |
