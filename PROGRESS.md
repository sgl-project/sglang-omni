# SGLang-Omni 项目进展追踪（PROGRESS.md）

> 本文件为本地审查用的全局进展追踪文档（中文），**不提交到 git / 远程仓库**。
> 最后更新：2026-07-31

---

## 一、项目概述

SGLang-Omni 是 SGLang 项目的多模态/语音扩展，定位为**统一的多模态语音推理 serving runtime**。
其主旨是：用一套分层、声明式、可拼接的运行时，把各类语音模型（TTS / ASR / Omni / 实时对话）
以 **OpenAI 兼容 API** 的形式对外服务，并支持单卡 / 多卡 / 流式 / 批处理等多种部署形态。

仓库地址：
- 上游：`https://github.com/sgl-project/sglang-omni`
- 本 Fork：`https://github.com/1571859588/sglang-omni`

---

## 二、核心架构（分层）

| 层 | 目录 | 职责 |
|---|---|---|
| 模型注册与能力声明 | `sglang_omni/models/` | 17 个模型家族的 `PipelineConfig` 声明、能力 `CAPABILITIES`、stage 定义 |
| 模型运行器 | `sglang_omni/model_runner/` | 各模型的实际推理执行、KV cache、显存管理 |
| 流水线 | `sglang_omni/pipeline/` | 多 stage 编排（tokenizer / encode / decode / vocoder 等） |
| 调度 | `sglang_omni/scheduling/` | 请求调度、批处理、并发 |
| 服务层 | `sglang_omni/serve/` | OpenAI 兼容接口：`/v1/audio/speech`、`/v1/audio/transcriptions`、`/v1/audio/translations`、`/v1/realtime` |
| 数据面 | `sglang_omni/relay/`、`sgang_omni/comm/` | 内部 stage 间音频/张量传输 |
| 分析与诊断 | `sglang_omni/profiler/`、`sglang_omni/diagnostics/` | 性能剖析、健康检查 |

设计理念详见 `docs/design/refactor_rfc.md`（统一 serving runtime + OpenAI 兼容语音 API 的 RFC）。

---

## 三、已完成功能（截至 2026-07-31）

### 1. 模型家族 onboarding（17 个，全部可用）
ASR：`whisper_asr`、`fun_asr`、`qwen3_asr`、`arkasr`、`moss_transcribe_diarize`
TTS：`higgs_tts`、`zonos2`、`qwen3_tts`、`moss_tts`、`moss_tts_local`、`ming_tts`、`audar_tts`、`fishaudio_s2_pro`、`voxtral_tts`
Omni：`qwen3_omni`、`ming_omni`、`llada2_uni`

每个家族均有：
- `sglang_omni/models/<family>/` 下的 `PipelineConfig` + stage 实现
- `docs/cookbook/<family>.md` 评测/架构文档

### 2. OpenAI 兼容服务接口
- 文本转语音 `/v1/audio/speech`（含 `/batch`、`/stream`）
- 语音转写 `/v1/audio/transcriptions`、翻译 `/v1/audio/translations`
- 实时对话 `/v1/realtime`
- 多模型 reference-audio / voice-cloning 能力（`supports_reference_audio=True` 已声明于 higgs_tts、zonos2、qwen3_tts、moss_tts、moss_tts_local、ming_tts、audar_tts、fishaudio_s2_pro）

### 3. 单卡端到端 CI（部分模型）
- `tests/test_model/test_asr_ci_fun_asr.py` — Fun-ASR 单卡 ASR serving
- `tests/test_model/test_zonos2_tts_ci.py` — ZonOS2 单卡 TTS serving
- `tests/test_model/test_tts_serving_ci.py` — Higgs TTS 单卡 serving

### 4. 示例配置（examples/configs）
已覆盖：audar_tts、ming_*、moss_*、qwen3_asr、qwen3_omni、qwen3_tts、fishaudio_s2_pro、voxtral_tts 等约 20 个 yaml。

---

## 四、待解决的问题 / 缺口

| 问题 | 影响 | 优先级 |
|---|---|---|
| `examples/configs/` 缺少 higgs_tts / zonos2 / fun_asr 的即开即用 yaml（代码可跑但无开箱示例） | 新用户上手成本高 | 中 |
| `llada2_uni` 是 17 个家族中**唯一零测试**的模型 | 回归风险 | 中 |
| 部分小模型 cookbook 缺少「Quick Start / Serve with example config」段落 | 文档与可执行配置脱节 | 低 |
| `serve/openai_api.py` 转录上传接口缺少 body 大小限制（见 `openai_api.py:1603` 附近） | 健壮性 | 低 |
| 上游已有 PR 集中在 Qwen3-Omni 性能优化、MOSS-TTS 采样融合、消费级 GPU 支持、TTS 修复 —— 新贡献需避开这些方向 | 避免重复 | — |

---

## 五、本轮贡献点（2026-07-31）

**目标**：以专业开源贡献者身份，提交 1 个高质量、不与上游重复、且可在 **1-2 张 A100 40G** 上验证的 onboarding PR。

**方向修正说明**：
原计划拟为 higgs_tts / zonos2 / fun_asr「补单卡 smoke 测试」，但经核对，这三个模型**已存在完整的单卡端到端 CI**
（`test_asr_ci_fun_asr.py`、`test_zonos2_tts_ci.py`、`test_tts_serving_ci.py`）。为避免与现有 CI 重复，
将 PR 范围收敛为**补齐端到端「开箱即用」资产**，而非新增测试：

1. 新增 `examples/configs/higgs_tts.yaml`、`zonos2.yaml`、`fun_asr.yaml`
   - 复用现有 `config_cls`（`HiggsTtsPipelineConfig` / `Zonos2PipelineConfig` / `FunASRPipelineConfig`）+ 官方 `model_path`
2. 在 `docs/cookbook/higgs_tts.md`、`zonos2.md`、`fun_asr.md` 末尾追加 Quick Start / Serve with example config 段落
3. 在 `examples/README.md` 索引表登记三个新 example 及单卡说明

**为何不与现有 PR 重复**：上游在做的 TTS 工作集中在 Qwen3-Omni 性能、MOSS 采样、消费级 GPU；
本 PR 聚焦"文档/示例资产补全"，与运行时代码路径无重叠，且填补的是公认的 onboarding 缺口。

**验证方式（受限硬件）**：当前集群仅 1-2 张 A100 40G 空闲，无法跑多卡大模型（Qwen3-Omni / MOSS / Ming-Omni）。
本 PR 仅改 yaml/文档，无运行时热路径变更，校验方式：yaml 加载解析 + 现有单卡 CI 不受影响。

---

## 六、未来规划 / 潜在贡献点

1. **为 `llada2_uni` 补充最小端到端测试**（零测试家族，优先级中）
2. **reference-audio 参数的 OpenAI 接口透传**：若后续确认 serving 层未完整暴露该能力，可作为功能 onboarding PR
3. **examples 索引自动化**：为 `examples/configs/*.yaml` 增加 CI 校验（config_cls 可解析、model_path 存在）
4. **转录上传 body 大小限制**：`openai_api.py` 健壮性修复
5. **更多小模型 example yaml 补全**：arkasr、whisper_asr、audar_tts 等仍缺单卡示例
6. **消费级 GPU 部署文档**（与上游 PR 差异化：侧重 cookbook 而非运行时代码）

---

## 七、环境备注

- 硬件：8×A100 40G，但仅 1-2 张长期空闲；多卡大模型实验不可行。
- Python：3.11+
- 测试运行参考 `tests/README.md`；单卡模型可用 `CUDA_VISIBLE_DEVICES=0` 限定。
