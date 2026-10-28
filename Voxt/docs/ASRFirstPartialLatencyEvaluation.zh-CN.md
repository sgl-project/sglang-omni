# Qwen / MOSS Final ASR 优化方案

日期：2026-08-02  
分支参考：`feat/final-asr-pipeline-opt`  
对照：Handy（`tmp/Handy`，transcribe.cpp GGUF Qwen3-ASR）

---

## 1. 目标与原则

**目标**：缩短「停录 → Final 文本就绪」（`stopToASRMs`），且 **Final 准确率不低于当前 offline 基线**。

| 原则 | 含义 |
|---|---|
| 合同不变 | Final = `postStopFinal` / 整段 `generateStream`；live 只做预览 |
| 优化链路 | 缓存、参数、配置、预处理、预热/驻留；不换架构 |
| 质量优先于炫技 | 禁止把 live `.ended` 当 Final（已验证会掉质量） |
| 按模型分流 | Qwen 可 trim；MOSS 保时间轴；batch 模型另论 |

Handy 印证：Qwen3-ASR 完全可以「**无 streaming Final + VAD 裁切 + 模型预热/驻留**」；可抄流程，不可抄 GGUF/Metal 参数字面量。

---

## 2. 目标链路

```text
热键按下
  → ASR 模型预热 + VAD 预热，与开麦并行
录音中
  → live 预览（可选）
  → 本地 VAD：语音帧进缓冲 / 标记（便于 Final trim）
  → 模型 keep-resident（禁止 idle unload）
停录
  → 立刻释放 live session（不等 ended）
  → 按 vadPolicy 选 Final 样本（trim 或 full）
  → 历史 WAV 归档与 Final 推理并行
  → postStopFinal：generateStream（动态 maxTokens + Final KV + 单块窗口）
  → onTranscriptionFinished
```

---

## 3. 已落地

### Wave A — Final 前无效开销

| 项 | 状态 |
|---|---|
| 去掉 ended 空等 | 已做 |
| vadPolicy 样本选择 | 已做 |
| 动态 maxTokens | 已做（现 28 tok/s + 64） |
| Final Qwen KV `finalQwen` | 已做 |
| 打点 prepare/inference/finalization | 已做 |

### Wave B — 预热与驻留（无交互变更）

| 项 | 状态 |
|---|---|
| 热键/开录最早预热 ASR | 已做 |
| Silero 预取 | 已做 |
| 录音+Final 全程 `sessionModelPinned` | 已做 |
| 选中后常驻偏好 | 不做（设置变更） |

### Wave C — Final 输入缩短（无交互变更）

| 项 | 状态 |
|---|---|
| VAD on 仅当 filtered 严格更短时采用 | 已做 |
| 丢掉句尾 trailing 静音 | 已做 |
| VAD 关闭后置 trim / 采集侧少攒静音 | 不做（会改默认行为） |

### Wave D — generateStream 参数（无交互变更）

| 项 | 状态 |
|---|---|
| maxTokens 系数 28 tok/s（密中英） | 已做 |
| Final chunk 抬到 ≥1200（避免 accuracyFirst=90 切块） | 已做 |
| Final KV 经 `postStopFinalKVCachePolicy` | 已做 |
| MOSS dictation 保持 plain + 共用动态 maxTokens | 已确认，不改设置 |
| context bias Final 保留 / live 不带 | 保持 |

### Wave E — 预处理与并行

| 项 | 状态 |
|---|---|
| 历史 WAV 与 Final 推理并行 | 已做 |
| prepare（resample）与 model load 重叠 | 已做 |
| cancel→Final 不加 clearCache | 已确认保持 |
| 录音期常驻 16k 缓冲 | 不做（改动面大、收益次于并行 prepare） |

### Wave F — 实时 provisional（仅预览）

| 项 | 状态 |
|---|---|
| 抑制过短 provisional-only 闪字 | 已做 |
| 抑制 confirmed 不变时的预览收缩/抖动 | 已做 |
| 不用 `.ended` 当 Final | 保持禁止 |

---

## 4. 明确不做

1. 用 live `.ended` 替换 `postStopFinal`
2. 为质量不满意加「再跑一遍」兜底主路径
3. 为追 Handy 而换 transcribe.cpp / 放弃 mlx-audio
4. 把会议 diarization、LLM enhance 塞进 Final 热路径
5. 新增用户设置项或改变默认交互

---

## 5. 度量与验收

- `stopToASRMs` / `prepareMs` / `inferenceMs` / `finalizationMs`
- `audioSec` / `source`（full vs voiceActivityFiltered）
- 冷启首句：load 与录音重叠；停录后无二次 model load
- 质量：长句 / 中英切换 ≥ 改前 offline Final

---

## 6. 成功长什么样

说完松手：仍是整段 Final（准），但模型早已热、音频更短、tokens/KV/chunk 贴身、归档与 prepare 不挡推理——`stopToASRMs` 可感知变快，实时预览更稳，且无「为快而糊」的回归。
