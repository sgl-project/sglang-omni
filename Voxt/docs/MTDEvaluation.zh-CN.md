# MOSS-Transcribe-Diarize（MTD）在 Mac 上的实测

日期：2026-11-16
机器：Mac17,14（Apple M5 Max，36 GB，macOS 27.2.1）
运行时：已安装 Voxt 2.0.3 自带的 Omni 运行时（`/Applications/Voxt.app/Contents/Resources/omni`，与 app 启动的是同一批服务器二进制）
测量代码：`eval_mtd_test.go`（`TestEvalMTD`、`TestEvalMTDSpeakers`、`TestEvalMTDLanguages`），走 `internal/omni` 的完整路径——同一服务器进程管理、同一组启动参数、同一条 Final 请求策略（切块、token 预算、上下文偏置）。没有写任何独立脚本。

目的：回答 MTD 能否做 Voxt 的默认语音识别模型（听写与会议两个场景），以及默认开 DFlash 投机解码是否安全。

## 方法

- **评测集**：`testdata/eval` 的 29 条短剪辑（中、英、中英混、词典）加 2 条长剪辑（95 s、104 s），合成干净语音（Qwen3-TTS），见 [eval.md](eval.md)。
- **重复与计时**：每个计时重复 3 次取中位数；每次请求前读 `sysctl -n vm.loadavg`，把 1 分钟负载区间写在时间表里；开跑前用 `pgrep` 确认没有其他 `qwen3_asr_server` / `moss_transcribe_diarize_server`（无 CONTENTION 记录）。一次只跑一个语音模型服务器（运行时会退役其他的）。
- **内存**：每 0.5 s 轮询服务器进程的 physical footprint（`macx.Memory`，含 Metal 缓冲区即权重），取整轮峰值。
- **错误率**：`internal/eval` 的标准化比较（中文按字、英文按词，双方去掉语气词和标点），每条每轮都打分。
- **配置**：
  - `MTD`：`asr.moss`（MOSS-Transcribe-Diarize，bf16），`WithInstruction` 用 app 的纯文本指令（"Transcribe the audio as plain text without timestamps or speaker labels."）；
  - `MTD+dflash`：同上，加 `--draft-model-path <dflash> --draft-bits 8`（app 投草稿的方式）；
  - `qwen3-0.6B-4bit`：`asr.fast`，app 的策略（含语言提示、词典上下文偏置）。
- **权重**：MTD 与 dflash 直接读 HF 缓存的 snapshot（只读）；Qwen3-ASR 0.6B 4bit 走模型库（`models.Installed`）。

## 单人听写（dictation 的用法）

### 错误率（越低越好；每轮都打分）

| 剪辑组（token 数） | MTD | MTD+dflash | Qwen3-ASR 0.6B 4bit |
|---|---|---|---|
| zh-filler（354） | 1.7% | 1.7% | 2.5% |
| zh-plain（69） | 0.0% | 0.0% | 0.0% |
| en（216） | 8.3% | 11.1% | **1.4%** |
| mix（210） | 10.0% | 10.0% | 11.4% |
| dict（177） | 35.6% | 33.9% | **20.3%** |
| dict2（342） | 16.7% | 18.4% | **13.2%** |
| long（1470） | 23.3% | 23.3% | **5.3%** |
| **全部（2838）** | **17.9%** | **18.2%** | **6.9%** |

MTD 的主要失分模式（逐条转录见 `build/eval/mtd.md`）：

- **标签漏进正文**。纯文本指令下 MTD 仍时不时输出 `[S01]`、时间戳：en-1 写成 `[S01] So,[S01] um,[S01] …`（15.8%），en-3 开头多个 `[2.62]`（23.1%），两条长剪辑从头到尾挂满 `[S01]`（各 23%）。这是长剪辑错误率的主要来源——这类音频（多人标注风格的连续语流）触发了它的分人习惯。app 若直接采用，这些标签会原样进入用户文本。
- **拉丁缩写拆字母**：`P R`、`C P U`、`U I`、`s g long`（mix、dict 组），Qwen 写 `PR`、`CPU`。
- **专名听错**：VoxtNew→`vox tenue`，Pixcall→`Pixcore`，Silero→`Siloro`，Omni→`army`（它不吃识别上下文，见"词典"节）。

### 时间与内存（各 3 轮取中位数；负载为请求前的 1 分钟平均区间）

| 配置 | 首个 Final（冷启动） | 热 Final 中位（29 条，平均 5.8 s 音频） | 热 Final 最大 | 长剪辑 Final（95–104 s） | 实时因子 | 峰值 footprint | 负载 |
|---|---|---|---|---|---|---|---|
| MTD | 1296 ms | 187 ms | 275 ms | 2384 ms | 0.032 | 3.13 GB | 2.69–3.22 |
| MTD+dflash | 1411 ms | 162 ms | 225 ms | 1558 ms | 0.028 | 3.31 GB | 2.50–3.63 |
| Qwen3-ASR 0.6B 4bit | 1357 ms | **61 ms** | 94 ms | **758 ms** | **0.011** | **1.62 GB** | 3.50–3.54 |

- dflash 草稿让短剪辑快 13%、长剪辑快 35%，代价是约 180 MB 内存。
- MTD 的热 Final 是 0.6B 的 3 倍，内存是 2 倍；实时因子都远小于 1，日常使用都够快。
- 冷启动三者接近（1.3–1.4 s，进程启动加加载），与录音重叠后用户无感（app 开录即预热）。

### 草稿的文本一致性（逐条核对）

93 个请求中 81 个完全一致；4 条剪辑（en-3、dict-3、dict2-7、long-all）每一轮都不同，且差异确定（非噪声）：

| 剪辑 | 无草稿 | 有草稿 |
|---|---|---|
| en-3 | `…tomorrow at 9:00 in the morning.` | 句尾多一个时间戳标签 `…morning.[6.62]` |
| dict-3 | `Siloro settings` | `Silero settings`（有草稿反而对） |
| dict2-7 | `to MyGo last week` | `to my Go last week`（无草稿对） |
| long-all | `…Vox Nu share the same…` | `…Vox New share the same…`（第 663 字） |

投机解码的逐 token 验证在并行验证的 logits 与逐步解码有浮点差异，近似持平的 token 上贪心会翻向另一边。差异双向都有，总错误率几乎不变（18.2% 对 17.9%）。结论：草稿可换速度，但**不保证逐字一致**；对"草稿绝不影响文本"的假设（#3874 对 Qwen3-ASR 成立：448/448 一致）在 MTD 上不成立。

## 多人（会议的用法）

MTD 自己的分人任务（`WithInstruction("")`，服务器内置的分人提示词），2/3/4 个说话人由评测集不同嗓音的剪辑拼接（中间 1.5 s 静音），布局（谁在什么时候说话）已知：

| 剪辑 | 布局（嗓音：秒） | 段数 | 标签正确 | 错误率 | Final（中位） | 负载 |
|---|---|---|---|---|---|---|
| 2 人（zh-filler-1 + en-1） | Vivian 0.0–10.3；Ryan 11.8–21.5 | 2 | 2/2 | 2.3% | 468 ms | 3.08–3.09 |
| 3 人（+zh-filler-2） | +Uncle_Fu 11.8–20.2 | 4 | 4/4 | 2.9% | 802 ms | 3.08–3.08 |
| 4 人（+zh-filler-3） | +Serena 21.7–29.7 | 5 | 5/5 | 1.1% | 963 ms | 3.08–3.08 |

- 11/11 段标签全对，3 轮完全确定；时间戳与实际布局吻合（段边界落在静音附近）。
- 分人任务下文本质量明显好于纯文本指令（1.1–2.9% 对 17.9%）——这是它的原生任务，也没有标签泄漏问题（标签本来就是输出格式的一部分，解析后去掉）。
- 注意：这是干净、轮流说话、无重叠的合成语音，是分人能力的上限；真实会议（噪声、重叠、远场）未测。

## 识别上下文（词典）

MTD 不吃识别上下文（`ContextTerms` 为 0，词典到不了服务器）。把整套词典（评测集 9 个词 + 用户自己的 8 个词）作为上下文偏置，统计含词典词的 14 条剪辑里各模型写对的词数（共 26 个）：

| 模型 | 有词典上下文 | 无 |
|---|---|---|
| MTD | 4 / 26（上下文到不了它，两列是同一次请求） | 4 / 26 |
| Qwen3-ASR 0.6B 4bit | **16 / 26** | 8 / 26 |

也就是说，一个依赖词典的用户换到 MTD，26 个词里要少写对约 12 个（相对 0.6B）；相对 1.7B 的差距更大（eval.md 里 1.7B 有词典时能写对 21/26）。人名（陈喆、韩煜、周芷若、欧阳明远）MTD 反而全对，损失集中在拉丁产品名（VoxtNew、Pixcall、Silero、Omni 等）。

## 其他语言（探针）

HF 语料不在本机（见"未测量"），改用 macOS `say` 现场合成各一条（Kyoko、Yuna、Sinji、Mónica；音频用完即弃，只是探针不是评分）：

| 语言 | 说的 | MTD 输出 | Qwen3-ASR 0.6B |
|---|---|---|---|
| 日语 | 今日の天気はとてもいいですね。午後は公園を散歩しましょう。 | 与原文一致 | 与原文一致 |
| 韩语 | 오늘은 날씨가 정말 좋네요. 오후에 공원을 산책해요. | **输出成日语**：今日の天気がとてもいいですね。午後の公園を散策します。 | 与原文一致 |
| 粤语 | 今日天氣好好，我哋下晝去公園散步啦。 | 与原文一致（繁体、口语词保留） | 内容对，部分字转简体 |
| 西班牙语 | Hace muy buen tiempo hoy. Vamos a pasear por el parque esta tarde. | 内容一致（句号变逗号） | 与原文一致 |

MTD 的目录语言列表只有 zh/en 是准确的概括：日、粤、西在这条探针上可用，**韩语被当日语输出了**。

## 未测量及原因

- **Qwen3-ASR 1.7B 8bit 及其 dflash 草稿**：本机模型库与 HF 缓存都没有权重（缓存里只有 refs，无数据），任务规则禁止下载模型。既有数据可参照 [eval.md](eval.md)：mac-air（M4）上 1.7B 8bit 总错误率 2.6%、热 Final 中位 648 ms；本机同型号（M5 Max）11-11 测过 1.7B 带 8bit 草稿在 448 条语料上文本 448/448 一致。
- **HF 语料 XXXXRT/qwen3-asr-test-corpus（448 条）**：缓存只有 refs，没有数据文件。
- **真实会议音频**：噪声、重叠说话、远场都没有；分人数字是上限。
- **live/流式路径**：只测了 Final（停录后的整段识别，dictation 的落字路径）；realtime 预览未测。
- **超过 2 分钟的音频**：长剪辑 95–104 s 已覆盖 60 s 切块；更长未测。
- **MTD 服务器的缓存上限**：它没有 `--cache-limit-mb`（固定 2 GiB），本表的 3.1–3.3 GB 峰值即该配置下的数字。

## 复现

```sh
HF=~/.cache/huggingface/hub
MTD=$HF/models--OpenMOSS-Team--MOSS-Transcribe-Diarize/snapshots/704aa4a9c304e8520be88901e0d1960158ef5b15
DRAFT=$HF/models--wilsonzheng0327--MOSS-Transcribe-Diarize-dflash/snapshots/2e3f00b0932112e4f92a4ef67a4cd60ee855056c
cd Voxt
# 单人听写 + 词典 + 草稿一致性（mtd.md）
VOXTNEW_EVAL=1 VOXTNEW_OMNI_RUNTIME=/Applications/Voxt.app/Contents/Resources/omni \
  VOXTNEW_MTD_DIR=$MTD VOXTNEW_MTD_DRAFT_DIR=$DRAFT VOXTNEW_EVAL_OUT=build/eval \
  go test -count=1 -timeout 90m -run 'TestEvalMTD$' -v .
# 多人分人（mtd-speakers.md）
VOXTNEW_EVAL=1 VOXTNEW_OMNI_RUNTIME=/Applications/Voxt.app/Contents/Resources/omni \
  VOXTNEW_MTD_DIR=$MTD VOXTNEW_EVAL_OUT=build/eval \
  go test -count=1 -timeout 60m -run 'TestEvalMTDSpeakers' -v .
# 其他语言探针（mtd-languages.md）
VOXTNEW_EVAL=1 VOXTNEW_OMNI_RUNTIME=/Applications/Voxt.app/Contents/Resources/omni \
  VOXTNEW_MTD_DIR=$MTD VOXTNEW_EVAL_OUT=build/eval \
  go test -count=1 -timeout 60m -run 'TestEvalMTDLanguages' -v .
```

`VOXTNEW_OMNI_RUNTIME` 指到任一包含 `bin/qwen3_asr_server` 与 `bin/moss_transcribe_diarize_server` 的运行时即可（如 `scripts/build-omni.sh --out` 的产物）；Qwen3-ASR 行需要模型库里装着对应模型，缺失时该行跳过并打日志。

## 结论

- **听写默认模型：不能。** 总错误率 17.9% 对 0.6B 的 6.9%；长剪辑会把 `[S01]` 标签写进用户文本；词典词少写对约 12 个（26 中 4 对 16）；速度慢 3 倍、内存多 1 倍。中文短句（zh-filler、zh-plain）两者相当，但这撑不起默认。
- **会议：分人能力可用且有条件。** 干净轮流语音上 2/3/4 人标签全对、文本错误率 1–3%；但 Voxt 现有路径拿不到分人结果（`internal/omni` 只回纯文本，本次靠新加的 `WithInstruction` 才测到），真实会议音质未验证，词典照样吃不到。要做会议分人，先接 segments、再在真实录音上复测。
- **dflash 草稿：可以默认开，但别说它不改文本。** 短剪辑快 13%、长剪辑快 35%、内存 +180 MB；31 条里 4 条文本会变（双向、确定性），对 Qwen3-ASR 成立的"草稿不动文本"假设在 MTD 上不成立。
