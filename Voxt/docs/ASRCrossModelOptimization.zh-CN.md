# 跨 ASR 模型 Final 优化评估方案

日期：2026-08-02  
对照基线：Qwen3-ASR / MOSS 上已落地的 Wave A–F（见 [`ASRFirstPartialLatencyEvaluation.zh-CN.md`](ASRFirstPartialLatencyEvaluation.zh-CN.md)）

> 状态说明：本文中的 Sherpa ONNX 章节是历史方案记录。Sherpa-onnx ASR 已从当前实现移除，不应再按本文方案接入或验收。

---

## 1. 结论（先看这个）

多数 **MLX 本地模型已经吃到共享 Final 热路径**（预热、驻留、VAD 分流、并行归档/prepare、Final chunk 抬窗、动态 maxTokens）。  
接下来不是「再抄一套 Qwen 架构」，而是：

1. **验证**哪些 batch 模型在真实语料上收益最大  
2. **补齐**尚未共享的引擎（Sherpa / Remote / Dictation）的等价优化  
3. **明确禁止**对 `modelManaged` / `preserveTimeline` 族乱做 trim

| 优先级 | 对象 | 动作 | 状态 |
|---|---|---|---|
| P0 | Whisper / Parakeet / Granite / GLM / FireRed | 共享路径合同锁 + Final 日志补 `family`/`externalTrim` | **已落地** |
| P1 | SenseVoice | Dictation Final 禁 PCM trim；catalog 保持 `.standard` 以保留会议校验 | **已落地** |
| P2 | Canary / Moonshine | Final 动态 maxTokens（tuning 作下限）；中间态仍用 tuning | **已落地** |
| P3 | Cohere / Nemotron / Voxtral | 禁外部 Final trim；Cohere Final 保持 tuning 预算 | **已落地** |
| P4 | Sherpa ONNX | 新做：预热 + stop 后 VAD 再 offline | **不做** |
| P5 | Remote / Dictation | 维持现状为主；streaming 不抄 skip-ended | **不做** |

---

## 2. 原则（继承 Qwen/MOSS）

| 原则 | 含义 |
|---|---|
| 合同不变 | Final = 整段 offline 识别；live 只预览 |
| 按 `vadPolicy` 分流 | `standard` 可 trim；`preserveTimeline` / `modelManaged` 保全量 |
| 优化链路不改交互 | 不新增设置、不改默认 UX |
| 先度量再改参 | 共享路径已覆盖的族，优先打点确认，再谈专项调参 |

---

## 3. 模型地图与可迁移性

### 3.1 MLX 族（共享 `MLXTranscriber.runFinalizationPipeline`）

| 族 | live | vadPolicy | 共享优化现状 | 额外空间 |
|---|---|---|---|---|
| **Qwen3-ASR** | native | `standard` | Wave A–F 基线 | 维持 |
| **MOSS** | native | `preserveTimeline` | 预热/并行/动态 tokens；**不 trim** | 维持 |
| **Whisper** | batch | `standard` | 预热、VAD trim、并行、动态 tokens | **P0 度量** |
| **Parakeet** | batch | `standard` | 同上 | **P0 度量** |
| **Granite / GLM / FireRed** | batch | `standard` + recognitionPreset | Final chunk 已抬 ≥1200 | **P0 度量**（accuracyFirst 长句） |
| **SenseVoice** | batch | catalog `.standard`（会议校验保留） | Dictation Final 按 family 禁 PCM trim；内层 Silero 保留 | P1 已修会议副作用 |
| **Canary / Moonshine** | batch | `standard` | Final 动态 maxTokens；中间态 tuning | P2 已落地 |
| **wav2vec2 / MMS** | batch | `standard` | 共享路径已覆盖 | 低优先度量 |
| **Cohere** | native | **`modelManaged`** | 不 trim；Final 保持 tuning tokens | P3 已落地 |
| **Nemotron / Voxtral** | native | **`modelManaged`** | 不 trim；共享预热/并行 | P3 已落地 |

### 3.2 非 MLX 引擎（不共享 Final 热路径）

| 引擎 | Stop → Final | 可迁移点 | 风险 |
|---|---|---|---|
| **Sherpa ONNX** | 停录 → offline 解码整文件 | 开录预热 recognizer；stop 后 VAD 裁切再 decode | 中（新实现） |
| **Remote 上传** | 本地 VAD 预处理 → HTTP | 已有上传前 VAD；可再重叠 prepare | 瓶颈多在网络 |
| **Remote 流式** | 关流后 **等待服务端 final**（最长约 20s） | **不要**抄 skip-ended；最多观测超时 | 高（截断） |
| **系统 Dictation** | `endAudio` → isFinal / 900ms 兜底 | 几乎无本地可调 | 高（系统合同） |

---

## 4. Wave A–F 对其他模型的含义

| Wave | 对多数 MLX | 对 modelManaged 族 | 对 Sherpa/Remote |
|---|---|---|---|
| A 去 ended 空等 | native 已生效；batch 本无 ended | 已生效 | Remote 流式 **不适用** |
| B 预热+驻留 | 已共享 | 已共享 | Sherpa 需新做 |
| C VAD 缩短输入 | `standard` 已生效 | **禁止**（必须 full） | Sherpa/上传可对齐 |
| D tokens/KV/chunk | tokens/chunk 多数已生效；KV **仅 Qwen** | Cohere tokens 走专用设置 | N/A 或各自参数 |
| E 并行归档/prepare | 已共享 | 已共享 | 部分可做 |
| F provisional | 仅 native live 预览 | 可受益 | 各引擎 partial 另论 |

要点：**KV `finalQwen` 不可照搬到无 KV 的族**；chunk 抬窗主要惠及 recognitionPreset 族（Granite/GLM/FireRed/Cohere）。

---

## 5. 分波实施计划

### 已落地（P0–P3）

1. **P0**：catalog/测试锁死 Whisper/Parakeet/Granite/GLM/FireRed 为 `standard` + recognitionPreset Final chunk≥1200；Final 日志增加 `family` / `externalTrim`  
2. **P1**：SenseVoice catalog 保持 `.standard`（会议外部 VAD 仍启用）；dictation Final 经 family 禁 PCM trim  

3. **P2**：Canary/Moonshine 仅 `postStopFinal` 用时长动态 maxTokens，tuning 作为下限  
4. **P3**：Cohere/Nemotron/Voxtral（及 SenseVoice）禁外部 Final trim；Cohere Final 不改写 tuning tokens  

### 手工验收建议

对 Whisper-turbo / Parakeet / Granite / SenseVoice / Canary 各跑短句与长叙述，看：

- `family=` / `externalTrim=` / `source=`  
- SenseVoice 长句应为 `source=full`（VAD on 时仍可 no-speech skip）  
- Canary 长句不再被 ~200 tokens 早截断  

---

## 6. 明确不做

1. 任何模型用 live `.ended` 当 Final  
2. 对 Cohere / Nemotron / Voxtral / MOSS / SenseVoice 做破坏时间轴的 Final trim  
3. 把 Qwen KV 参数硬套到无 KV 模型  
4. P4 Sherpa / P5 Remote·Dictation 本轮不实现  
5. 新增用户设置或改变默认交互来「开关」这些优化

---

## 7. 下一步

手工跑一遍上表验收语料即可；P4/P5 暂不排期。
