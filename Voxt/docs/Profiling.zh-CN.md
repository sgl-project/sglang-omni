# 长录音 Profiling 与配置矩阵

**本轮尚未完成真实长音频测量。** 当前交付的是可运行的测量工具、fake 回归测试和结果格式；文档不包含虚构的时长、内存或准确率数字。真实结果只能在目标 Mac 上按下面的口径取得后填写。

## 1. 安全边界与隐私

- 默认 `go test ./...` 不启动真实模型。只有 `VOXTNEW_PROFILE_CONFIG` 指向一个有效 JSON 时，根包的 `TestMeetingProfile` 才运行配置中的语音识别模型和可选语言模型；overlap 实验使用独立的 `VOXTNEW_OVERLAP_CONFIG` 和 `TestMeetingOverlap`，interactive-priority 实验使用独立的 `VOXTNEW_INTERACTIVE_CONFIG` 和 `TestMeetingInteractive`，三个环境变量互不复用。
- 先复制音频，再测量。直接单次运行时，调用者负责这一步；`profile-loop.sh` 总是把输入和 reference 复制到批次临时目录，并把副本设为只读，原文件只被读取一次用于复制。
- 循环脚本把模型目录中的文件硬链接到临时目录，不复制大权重，也不修改权限。临时目录必须与模型在同一文件系统；跨设备硬链接失败会报错，不会退化成复制或下载模型。不要对硬链接后的文件写数据。
- 每次运行在独立的 `voxt-meeting-profile-*` 临时目录创建 SQLite 和会议 WAV，结束时删除。不会打开 Voxt 的真实数据库、设置、偏好或 Keychain。
- 语言模型 key 只从 `language_model.key_env` 指定的环境变量读取。Go 和 loop 使用同一份字段白名单，在创建 results/batch/config.json 或复制音频前拒绝未知字段（包括 `language_model.key`），不会先把敏感值写入临时文件。校对和总结会把识别文本发给配置的语言模型 provider；不需要它们时把两项都关掉。
- `result.json` 的 config 删除语言模型 endpoint/model，但保留公开 ASR catalog id 以复现实验。执行 newLLM 时登记解析后的实际 endpoint/model/key，stage 诊断、文本、native.log、WriteReport 和 process.log 均清理这些值；loop 用于启动进程的 config.json 仍需保留非 key 的 endpoint/model 配置。
- 循环脚本不杀已有进程。启动前或结束后发现其他 Omni native server、未接 AC、idle 不达标，或运行中任一定期 power/idle 样本异常，结果都无效。contention 使用 `/usr/bin/pgrep -fl` 检查完整命令行，不调用 `ps`。

## 2. 单次运行

在 `Voxt/` 下准备 `profile.json`：

```json
{
  "name": "mtd-draft8",
  "mode": "import",
  "audio": "/ABS/copied-input/meeting.m4a",
  "reference": "/ABS/copied-input/reference.txt",
  "output_dir": "/ABS/results/mtd-draft8/run-1",
  "model": {"id": "asr.moss", "dir": "/ABS/model-links/moss"},
  "draft": {"enabled": true, "dir": "/ABS/model-links/moss-draft", "bits": 8},
  "segment_length_seconds": 22,
  "vad": {"enabled": true, "model_dir": "/ABS/model-links/silero"},
  "language": "zh",
  "cleanup": true,
  "summary": true,
  "language_model": {
    "provider": "openai",
    "endpoint": "https://example.invalid/v1",
    "model": "configured-model",
    "key_env": "VOXT_PROFILE_LLM_KEY"
  }
}
```

然后运行：

```sh
export VOXT_PROFILE_LLM_KEY='只在当前 shell 环境中设置'
VOXTNEW_OMNI_RUNTIME=/ABS/omni \
VOXTNEW_PROFILE_CONFIG=/ABS/profile.json \
go test -count=1 -run '^TestMeetingProfile$' -timeout 6h -v .
```

结果写入 `output_dir/result.json`、`native.log`、`cpu.pprof` 和 `heap.pprof`。进程失败或 JSON 中有任何 warning/error 都不是有效测量，不能拿去算中位数。

### 配置字段

| 字段 | 含义与限制 |
|---|---|
| `name` | 矩阵中的唯一名称；单次运行可省略。`.`、`..`、清洗后的重名及 resolve 后越出 batch/temp 的路径在复制前拒绝。 |
| `mode` | `import`：复用导入会议的 MTD 整轨分人；`subtitles`：普通分段 ASR，不加分人。`import` 必须使用 `asr.moss`。 |
| `audio` | 已复制的 `.m4a`、`.mp4` 或 `.wav`；通过现有 `decodeAudio`/`afconvert` 解码为 16 kHz 单声道 PCM16。 |
| `reference` | 可选 UTF-8 文本。存在时同时评价原始 ASR 文本和后处理最终文本；不存在时 evaluation 为 `null`。 |
| `output_dir` | 本次 artifact 目录。循环脚本逐 attempt 覆盖此字段。 |
| `model.id/dir` | `internal/omni` 已支持的模型 id 和本地目录；不会查 app 的模型设置或下载模型。 |
| `draft.enabled/dir/bits` | 草稿开关、目录和量化位数。开启时目录必填，bits 为 4、6、8，省略 bits 时为 8；只有 Qwen3-ASR 和 MTD 支持。服务拒绝草稿而回退会产生 warning，整轮失败。 |
| `decode_interval_ms` | Qwen3-ASR native server 的实时预览 decode interval；0 保留模型默认值。其他模型明确报错。此 harness 测离线 Final，不运行实时预览，所以该启动参数不应被解释为会改变离线 chunk 时间。 |
| `segment_length_seconds` | 普通分段 ASR 的 `meeting.PlanChunks` 段长，0 使用默认 22 秒。必须大于 1 秒重叠；只改变会议切块，不改变 Omni 内部 Final 切块。`import` 的 MTD 分人由 `diarize_chunk_seconds` 另行控制。 |
| `diarize_chunk_seconds` | 仅 profiling 的 MTD 分人块长和打包阈值，0 使用生产 `diarizeChunkMax`；正数必须大于 1 秒重叠。它只改变 `diarizeTrack` 的请求分块，不改变 mic 的 `PlanChunks`、识别参数或生产默认值。 |
| `cache_limit_mb` | Qwen3-ASR 的 `--cache-limit-mb`；省略保留模型默认策略，0 显式禁用缓存，正数设置 MB。其他模型明确报错。 |
| `idle_timeout_ms` | Omni 模型在无人使用后的卸载等待。省略或 0 使用生产默认 10 分钟；正数用于生命周期诊断，例如 1000 表示 1 秒；负数拒绝。profiling 在 final pass 前释放 VAD lease、final pass 后首个 correction 前释放 ASR lease，不插入人工等待，避免把计时器等待误算成模型工作。它不改变识别参数。 |
| `vad_strategy` | 仅 profiling：`whole` 强制整轨一次 `Speech` 调用，`batched` 强制十分钟窗口的 `batchedSpeechRegions`。默认沿用生产路径。VAD 在 final pass 前释放 lease，请求级 VAD 使用仍照常进行；这项只改变 VAD 窗口策略，不改变 ASR、chunk 或文本规则。 |
| `vad.enabled/model_dir` | 开启时创建并持有本地 Silero，目录必填；普通分段 ASR 用它检测语音。关闭时普通路径不调用神经或能量 VAD，而把整轨交给会议 chunk planner。`import` 有可用 MTD 时先调用 VAD 取得切点，再按 `diarize_chunk_seconds` 合段；没有 MTD 时仍以 `ErrNoDiarizer` 失败。 |
| `language` | 语音识别语言提示和总结主语言；空字符串让模型检测，总结沿用转录稿主要语言。 |
| `cleanup` | 调用 `Manager.CorrectTranscript`，按现有 30 行/3000 runes 规则分批。profiling 边界额外要求每批响应是 `{i,text}` 的 JSON array，malformed response 使 stage/run 失败；普通 Meeting 的容错不变。 |
| `summary` | 调用 `Manager.GenerateMeetingSummary`，保留现有层次总结请求。可在 cleanup 关闭时单独开启。 |
| `language_model.provider/endpoint/model` | 复用 `internal/providers` 的 provider、preset 和地址校验；endpoint/model 可按 preset 省略。endpoint 不得带 userinfo、query 或 fragment，避免凭据落盘。 |
| `language_model.key_env` | key 的环境变量名。provider 需要 key 时必填；无 key 的本地 OpenAI 兼容服务可省略。未知 JSON 字段一律报错。 |

## 3. 矩阵：独立进程与 median-of-3

矩阵是非空数组，或 `{"runs": [...]}`。例如：

```json
{
  "runs": [
    {
      "name": "mtd-draft-off",
      "mode": "import",
      "audio": "/ABS/original-read-only-once/meeting.m4a",
      "output_dir": "overwritten-by-loop",
      "model": {"id": "asr.moss", "dir": "/ABS/models/moss"},
      "draft": {"enabled": false},
      "vad": {"enabled": true, "model_dir": "/ABS/models/silero"},
      "cleanup": false,
      "summary": false,
      "language_model": {"provider": "openai"}
    },
    {
      "name": "qwen-fast",
      "mode": "subtitles",
      "audio": "/ABS/original-read-only-once/meeting.m4a",
      "output_dir": "overwritten-by-loop",
      "model": {"id": "asr.fast", "dir": "/ABS/models/qwen-fast"},
      "draft": {"enabled": true, "dir": "/ABS/models/qwen-fast-draft", "bits": 4},
      "segment_length_seconds": 22,
      "cache_limit_mb": 512,
      "vad": {"enabled": false},
      "cleanup": false,
      "summary": false,
      "language_model": {"provider": "openai"}
    }
  ]
}
```

运行：

```sh
scripts/profile-loop.sh /ABS/matrix.json /ABS/profile-results --idle-seconds 300 --environment-sample-seconds 15 --temp-root /ABS/same-filesystem-temp
```

每个配置默认执行三次，每次都是新的 `go test` 进程和新的临时 SQLite。`--attempts` 只接受 1 或 3：1 次只作诊断，config 和 batch 都写 `status="diagnostic"`，不生成 median；3 次才进入三次中位数汇总，2 次及其他次数直接拒绝。每次开始前等待 AC Power 和至少 300 秒 HID idle，并用 pgrep 检查已知 native server 是否争用；`Popen` 运行期间每 15 秒采样 power/idle，结束后再检查 power/idle/contenders。before/after 正常也不能掩盖中途违规。用户输入、掉到电池、出现争用、进程非零退出、result 带 warning/error，都会使该次无效；违规后不杀进程。无效样本不会加入 reports，因而也不会进入 median。不足三次有效结果时配置失败，脚本最终返回非零。`environment.json` 分别保存 before、during、after，process.log 持续落盘并逐行脱敏。

结果目录按 UTC 批次追加，不覆盖旧批次：

```text
profile-results/
  matrix-summary.jsonl
  20261116T120000000000Z/
    summary.json
    mtd-draft-off/
      summary.json
      run-1/{config.json,result.json,environment.json,process.log,native.log,cpu.pprof,heap.pprof}
      run-2/...
      run-3/...
```

`summary.json` 保留三次原始 result 的路径，汇总总时长、RTF、首 final、内存和每个同名同 index stage 的中位数。任一原始值不可得时，对应聚合值也是 `null`，不会取两次平均或填 0。临时音频和模型硬链接在批次结束后删除；result 中的文本和 hash 按任务要求保留。

可继续加入 `asr.accurate`、`asr.qwen3-6bit`、`asr.parakeet`、`asr.whisper-turbo` 的本地目录形成矩阵。模型能否放入目标 Air 的统一内存必须由测量回答，不预先声称某个模型适合。

## 4. 数字口径

- `stages[]` 按完成顺序记录 decode、setup、load、普通路径和 MTD 分块路径的 VAD、每个 `asr_chunk`、diarization、final pass、每个 correction batch、每个 summary request 和 teardown。`import` 的 MTD 路径先调用 VAD 取得切点，再按 cap 合段，因此有 `vad` 而没有 `asr_chunk` stage。嵌套阶段的外层 `final_pass` 最后完成；`index` 在同类阶段中从 0 开始。
- `wall_ms` 是墙钟时间。含 `audio_ms` 的阶段，`rtf = wall_ms / audio_ms`；总 RTF 是整轮墙钟除以音频时长。不要用总 RTF 代替单阶段瓶颈。
- `queue_acquire.value` 是 Omni 获取/启动/等待模型的时间；`result_latency.value` 是 `asr.Result.Latency`，从 acquire 完成后到完整文本。两者分开，不把排队当推理。provider 未报告时为 `null`。
- `first_final_text.value` 从 final pass 开始到第一份非空完整 ASR/MTD 文本。文件 ASR 没有 token 事件，所以 `asr_first_token.value` 永远为 `null`，不是 0 ms。
- MTD 一次请求返回全部分人段。每段有文本、时间范围和 hash，但 `segments[].inference_duration.value` 为 `null`，因为现有 API 没有每段独立推理时间。
- `asr_transcript` 是 final pass 后、校对前的文本；`transcript` 是校对后的最终文本；hash 是各自 UTF-8 文本的 SHA-256。`summary` 保存实际落盘的 summary JSON 和 hash。
- 有 reference 时，`evaluation.value.asr/final.mixed_cer_wer` 都直接来自 `eval.Compare`：中文按字、英文按词，混合文本采用同一 token 化与归一化。它是混合 CER/WER，不是 DER，也不证明分人正确。

  需要按真实 MTD 请求定位编辑差异时，在 `Voxt/` 运行 `go run ./scripts/meetingquality CAP_SECONDS /ABS/result.json`。脚本用同一 `eval.Tokens` 和 Levenshtein traceback，按参考 clip 在原始时间线的起点归入请求；硬切后下一段重读 1 秒，所以 cap120 的未加 padding 起点是 0/119/238/357/476/595 秒，不能简单按 0/120/240…分组。输出包含每个请求的 timeline/native 起止和 S/D/I；`go test ./scripts/meetingquality` 覆盖这个边界和请求数上限。
- `peak_rss_bytes` 和 `peak_physical_footprint_bytes` 是每 100 ms 对 harness 与 native server 求和后的最大观测值，不是内核生命周期 high-water mark；短峰值可能漏掉。`Runtime.Processes` 同时返回 ready 和已 spawned、仍在加载的 PID，因此 ready 前的权重加载也可采样；`Running()` 的 ready 语义不变。physical footprint 包含 Metal buffer，适合观察 Apple silicon 统一内存。
- CPU user/system 是这些进程累计 CPU 时间之和；native 进程退出后保留最后成功样本。GPU 时间、独立 GPU 内存和能耗没有安全可用的精确接口，写 `null` 和 `unavailable_reason`，不填 0。
- start/end 记录 memory pressure、available%、swap 用量和 swapin/out 累计计数；delta 只由同次 start/end 相减。`peak_swap_used_bytes` 另存 100 ms 采样期间看到的最大系统 swap 用量，评估内存预算时用 start-to-peak，不用结束值掩盖过程峰值。thermal 来自 `pmset -g therm` 原文，power source 来自 `pmset -g batt`。不可得字段一律是 `{"value": null, "unavailable_reason": "..."}`。
- `native.log` 独立保存 `omni.Config.Log`（含 native 自带计时）。Omni 将 native stderr 统一标为 warning，因此正常计时行不会误判失败；native error、崩溃/停滞、草稿或 cache 回退等会使配置失效的 warning 仍会使结果无效。VAD wrapper 只计时：普通路径整轨 Speech 失败后，10 分钟窗口的神经 VAD 成功不算 fallback；只有实际进入能量分析的窗口才通过 `VADFallback` 通知并使配置失效，默认 Meeting 的 nil 通知行为不变。MTD 分块路径也调用 VAD 取得切点，因此不能把 VAD 耗时或 fallback 当作普通 ASR 专属阶段。

## 5. pprof 与 Instruments

每次真实运行已经生成 Go CPU/heap profile：

```sh
go tool pprof -http=127.0.0.1:0 /ABS/results/run-1/cpu.pprof
go tool pprof -http=127.0.0.1:0 /ABS/results/run-1/heap.pprof
```

Darwin 上 harness 用现有 purego 动态调用 `os_signpost_interval_begin/end`，subsystem 为 `dev.hehehai.voxtnew.profiling`，category 为 `PointsOfInterest`。非 Darwin 或符号不可用时 JSON 的 `tracing.signposts` 明确为 unavailable，其他测量继续。先编译测试二进制，再让 Instruments/xctrace 启动它：

```sh
mkdir -p build
go test -c -o build/meeting-profile.test .
VOXTNEW_OMNI_RUNTIME=/ABS/omni VOXTNEW_PROFILE_CONFIG=/ABS/profile.json \
  xcrun xctrace record --template 'Time Profiler' --output /ABS/results/run-1.trace \
  --launch -- ./build/meeting-profile.test -test.run '^TestMeetingProfile$' -test.v
```

在 Instruments 中同时查看 Points of Interest/signpost 与 CPU 样本，并把 native server 进程纳入分析。Go pprof 只覆盖 Go 进程，native 的真实推理耗时以 result 阶段、native.log 和 Instruments/native 样本交叉核对，避免把 acquire、排队或远程语言模型等待误判为 MLX 推理。

## 6. 离线测试

```sh
TMPDIR=/tmp CGO_ENABLED=0 go test ./internal/profile ./internal/omni ./internal/meeting
TMPDIR=/tmp CGO_ENABLED=0 go test -run 'TestMeetingProfileFake' .
/usr/bin/env python3 -m unittest scripts.profile_loop_test
```

这些测试只用短 fake PCM、fake ASR/VAD/语言模型和临时文件，不下载模型、不播放音频、不访问真实用户数据。

## 7. 短听写的首字、Final 与投递延迟

`TestMacE2EFirstLatency` 通过真实 app 会话和真实 ASR 测量 cold、warm、切换模型三种场景，每类默认 3 轮。每轮 cold 都新建 app、ASR 和 VAD；warm 复用同一 app 的模型；switch 只重新加载 ASR。`VOXTNEW_LATENCY_PCM` 指定已复制的短音频，`VOXTNEW_LATENCY_ROUNDS` 改轮数，`VOXTNEW_LATENCY_OUT` 每轮追加一行 JSON：

```sh
VOXTNEW_LATENCY=1 VOXTNEW_LATENCY_ROUNDS=3 \
VOXTNEW_MAIN_SHA=<完整的源码提交 SHA> \
VOXTNEW_LATENCY_PCM=/ABS/copied-input/short.wav \
VOXTNEW_LATENCY_OUT=/ABS/results/short-open.jsonl \
VOXTNEW_OMNI_RUNTIME=/ABS/omni VOXTNEW_OMNI_MODELS='asr.moss=/ABS/moss;asr.accurate=/ABS/qwen;vad.silero=/ABS/vad' \
go test -count=1 -run '^TestMacE2EFirstLatency$' -v -timeout 30m .
```

JSON 分别记录 `app_ready_ms`（app 打开到模型/helper ready）、`first_partial_ms`、`stop_to_final_ms`（观察到 ASR 的最终文本）、`stop_to_insert_ms`（fake helper 收到实际 paste 调用）和 `stop_to_done_ms`（整个会话完成），另含 ASR/VAD cold 标记、模型加载、实时更新间隔、文本和 SHA-256。三个终点不得互相替代。测试在 paste 没有出现时失败，不会把 Done 当成插入。需要独立 cold 进程时，分别启动三个进程并各设 `VOXTNEW_LATENCY_ROUNDS=1`、`VOXTNEW_LATENCY_ATTEMPT=1/2/3`，这样 cold 和 warm 各得到三个不同样本，JSON 的 `round` 和 `attempt` 也不会重复。A/B 实验另用 `VOXTNEW_LATENCY_ARM` 写入 `arm`；`min_preview_samples` 来自 native 实际预览事件，不是 runner 回显的配置。`decode_events` 记录会话内每个 Final 请求的计时（`kind` 为 preview/piece/final、`submitted_ms`/`finished_ms` 和它前面排队的同会话请求数 `inflight`，不含文本），用来区分「请求在服务器上排队」和「请求自己解码慢」；没有探针事件的会话写 `decode_events_unavailable_reason`，不写空列表。这里的插入覆盖 app 的真实投递逻辑，但终点仍是 fake helper，不是屏幕上的真实光标；真实 GUI 插入必须另行验收。

## 8. 听写/会议 overlap 调度实验

这个入口回答的是“真实 ASR 与可控模拟语言模型怎样调度”，不是语言模型质量评测，也不改变 #4060、生产流式识别、校对、总结或插入行为。只有显式设置 `VOXTNEW_OVERLAP_CONFIG` 才会启动真实 ASR；该配置没有远程语言模型、endpoint、API key 字段，未知字段会直接报错。

先复制输入音频，再准备 `overlap.json`：

```json
{
  "name": "meeting-overlap-smoke",
  "mode": "meeting",
  "audio": "/ABS/copied-input/meeting.wav",
  "output_dir": "/ABS/overlap-results/run-1",
  "model": {"id": "asr.fast", "dir": "/ABS/model-links/qwen-fast"},
  "draft": {"enabled": true, "dir": "/ABS/model-links/qwen-fast-draft", "bits": 4},
  "vad": {"enabled": true, "model_dir": "/ABS/model-links/silero"},
  "language": "zh",
  "segment_length_seconds": 22,
  "simulated_llm": {"latency_ms": 250}
}
```

`mode` 可改为 `dictation`；听写只做 cleanup，会议做 cleanup、summary map 和层次 final reduce。`draft.enabled=false` 时不需要 draft 目录，`vad.enabled=false` 时整轨交给现有 `meeting.PlanChunks` 按固定长度切分；开启 VAD 时先在录音重放前计算整轨 speech regions，失败或没有 speech region 都使结果无效，不暗中切换能量 VAD。段长为 0 时使用现有默认值，其他非零值必须大于 1 秒硬重叠。

```sh
VOXTNEW_OMNI_RUNTIME=/ABS/omni \
VOXTNEW_OVERLAP_CONFIG=/ABS/overlap.json \
go test -count=1 -run '^TestMeetingOverlap$' -timeout 12h -v .
```

三种方式使用同一个已 Hold 的 ASR client、同一模型/draft 和同一个 ASR worker，按 `one_shot`、`serial`、`overlap` 固定顺序运行：

| 方式 | 调度 |
|---|---|
| `one_shot` | 音频时钟走完后，一次整段 ASR，再一次整段 cleanup；会议随后对 cleanup 全文做 summary map 和 final reduce。 |
| `serial` | 音频时钟走完后，按计划段逐段 ASR，每段立即串行 cleanup；会议逐段 map，最后层次 reduce。 |
| `overlap` | 按音频时钟把计划段 PCM 喂入实验调度器；段完成即进入唯一 ASR worker，ASR 完成即进入模拟 LLM worker，后续音频继续按原时钟喂入。cleanup 请求带上一段清理文本，result 记录该上下文的 hash；会议 map 随段执行，停止后排 final reduce。 |

每次测试分别写 `output_dir/{one_shot,serial,overlap}/result.json`，并共享 `native.log`。模型加载和音频 decode 在三种计时之外；每次 `total_wall_ms` 从该方式的 segment planning 前开始，到最终结果或失败结束。三种方式都按音频真实时长重放，因此一次三方式实验至少需要三倍音频时长，不能用 fake 测试的瞬时 wait 代替真实实验。

### 结果口径与归因

- `simulated_llm` 恒为 `true`。模拟 cleanup 只做确定性的首尾空白整理并返回带 `i` 的 JSON array；模拟 summary 返回可解析的固定结构，内容包含规范化输入的 SHA-256。相同输入产生相同输出，分段响应按 `i` 重新组装，调用数、每次输入 hash、响应文本和实际墙钟均可检查。它不生成事实性纪要，不打质量分。
- `timing.asr_total_ms` / `llm_total_ms` 是各自请求墙钟之和；`asr_after_stop_ms` / `llm_after_stop_ms` 只累计与停止后区间重叠的部分。真实 ASR 数字与模拟 LLM 数字因此分开，不能把后者解释成远程服务延迟。
- `stop_to_final_ms` 是停止到听写 cleanup 完成或会议 final reduce 完成；`first_final_text` 和 `first_llm_text` 分别从录音重放开始到首个非空 ASR/模拟 LLM 文本。文件 ASR 没有 token 事件，本实验不测首 token。
- `backlog_at_stop` 记录停止时仍未完成的计划 ASR 段和 LLM 段任务，包括还在 ASR 后面的段；final reduce 不单独计入。`asr_transcript`、`transcript`、`summary` 及 `structure` 保存文本或结构与 SHA-256。`comparison_to_one_shot` 给出文本相等性、共同前缀/后缀、首个不同 rune、结构 hash、cleanup/map/reduce 数量差和 summary 相等性，不把这些差异换算成 CER/WER 或质量分。
- `scheduling_effects.asr_chunking_after_stop_saved_ms = one_shot.asr_after_stop - serial.asr_after_stop`；`llm_overlap_after_stop_saved_ms = serial.llm_after_stop - overlap.llm_after_stop`。另列 ASR overlap、LLM chunking 和 stop-to-final 差分。正数表示停止后时间减少，负数表示增加。`one_shot → serial` 同时改变 ASR 和 cleanup 分段，所以这些是有明确定义的算术对照，不是严格控制变量的因果收益。
- 任一 ASR、VAD、模拟响应或结构校验失败时 `status="invalid"`，不进入 median。无法得到的字段为 `null` 并带原因，不用 0 填充；合法的零 backlog 或零停止后耗时仍保留 0。

独立运行三次后，复用现有 summarize/median 逻辑：

```sh
/usr/bin/env python3 -m scripts.overlap_summary \
  /ABS/overlap-results/run-1 /ABS/overlap-results/run-2 /ABS/overlap-results/run-3 \
  /ABS/overlap-results/median
```

输出为 `median/{one_shot,serial,overlap}/summary.json`，包含 total wall、stop-to-final、ASR/LLM 总时间与停止后时间、backlog、首文本、stage 停止后时间及 hash 列表。这个汇总命令只读 JSON；它不复制音频/模型、不检查 AC/idle/争用，也不替代 `profile-loop.sh` 的环境门禁。当前 `profile-loop.sh` 仍只接受原 `VOXTNEW_PROFILE_CONFIG` schema，不能把 overlap 配置交给它。

限制：会议实验是单混合音轨，不做 diarization、双轨竞争或生产提示词；VAD 是重放前预规划，不是实时流式 VAD；三种方式固定顺序且共享暖模型，缓存和热状态可能跨方式携带；模拟延迟是每请求固定等待，不含 token 数、排队限流、远程网络、远程质量或首 token；没有实现或评价 #4060 的流式上屏/插入。本轮只交付并运行 fake 回归，没有真实 ASR 测量数字。

离线回归：

```sh
CGO_ENABLED=0 go test -run 'TestMeetingOverlapFake|TestSimulatedLLM|TestOverlapConfig' ./internal/profile
/usr/bin/env python3 -m unittest scripts.profile_loop_test.SummaryTests
```

## 8. 长任务中插入短听写：interactive-priority 实验

这个入口只测量调度，不实现生产优先级，也不改变 #4060、会议 final pass、普通听写或模型 runtime。只有显式设置 `VOXTNEW_INTERACTIVE_CONFIG` 才启动真实 ASR 和 VAD；默认根包测试跳过它，`internal/profile` 的 fake 测试不启动模型、不访问真实音频。第 3 轮必须配置 VAD，以便同时运行 `baseline`、`priority` 和 `priority_vad`；没有远程语言模型、endpoint 或 key 字段，未知字段在模型启动前报错。

先复制长音频和短听写音频，再准备 `interactive.json`：

```json
{
  "name": "mtd-interactive-smoke",
  "operation": "diarize",
  "long_audio": "/ABS/copied-input/meeting.wav",
  "interactive_audio": "/ABS/copied-input/short-dictation.wav",
  "output_dir": "/ABS/interactive-results/run-1",
  "model": {"id": "asr.moss", "dir": "/ABS/model-links/moss"},
  "draft": {"enabled": true, "dir": "/ABS/model-links/moss-draft", "bits": 8},
  "segment_length_seconds": 25,
  "vad": {"enabled": true, "model_dir": "/ABS/model-links/silero"},
  "language": "zh"
}
```

`operation` 为 `diarize` 时对长任务调用 `Diarize`，应使用 MTD/`asr.moss`；改为 `transcribe` 时长短任务都调用 `Transcribe`，可换成 `asr.fast` 及对应模型目录。`segment_length_seconds` 省略时为 25，显式值必须在 20–30 秒之间；它同时设置普通 priority 的固定段长和 `priority_vad` 的 `meeting.DefaultChunkParams.MaxDur`，后者保留现有 1 秒 hard overlap、120 ms padding 和暂停切点。`vad.enabled=false`、缺少 `vad.model_dir`、VAD 创建或调用失败、没有 speech region 或没有 chunk plan 都会明确失败，不回退到能量 VAD 或伪整轨 region。draft 规则与其他 profiling 入口相同。

```sh
VOXTNEW_OMNI_RUNTIME=/ABS/omni \
VOXTNEW_INTERACTIVE_CONFIG=/ABS/interactive.json \
go test -count=1 -run '^TestMeetingInteractive$' -timeout 12h -v .
```

结果写入 `output_dir/result.json`，native 日志写入 `native.log`。三种方式顺序执行，共用同一个已 Hold 的 ASR client、runtime、模型和 draft，不启动第二份 ASR 进程；`priority_vad` 另持有一个配置的 Silero VAD server：

| 方式 | 调度与测量 |
|---|---|
| `baseline` | 长音轨只提交一次整轨 `Diarize`/`Transcribe`；长 client 调用开始 100 ms 后，短交互 `Transcribe` 在另一 goroutine 并发提交到同一个 client。harness 不用自己的锁伪造串行，也不声称 native 并行推理；native worker 实际串行、排队或并行行为由墙钟结果呈现。 |
| `priority` | 长音轨按无重叠的 20–30 秒 PCM cursor 连续切块。交互同样在长调用开始 100 ms 后到达；当前 chunk 返回后先执行短交互，再从下一个 cursor 续跑。交互失败或长 chunk 失败都会使结果无效。 |
| `priority_vad` | 长调用开始前对整轨调用一次配置的 VAD，再用 `meeting.DefaultChunkParams`/`meeting.PlanChunks` 规划有 overlap 的块。调度暂停/恢复点与 priority 相同；相邻块文本用 `meeting.DedupeOverlapText` 去重后拼接。VAD 预规划不计入首个到最后一个 ASR 调用之间的 `long_total_ms`。 |

三种方法各自包含 `interactive_wait_ms`、`interactive_total_ms`、`long_total_ms`、`long_calls`、`interactive_calls`、`chunks`、`resume`、`long_segments`、文本/hash 和 `errors`。调用数按 harness 的 `Diarize`/`Transcribe` client 调用计，Qwen 内部 Final 子请求无法由这个入口单独观测。`interactive_wait_ms` 从交互到达到完整响应返回，包含调度等待、native 排队和识别；`interactive_total_ms` 只测交互 client 调用的墙钟，也包含 native 内部排队，现有 API 不能再把其中的排队和纯推理分开。`long_total_ms` 从首个长调用到最后一个长调用返回；两种 priority 因此包含中间短交互造成的暂停。`chunks[]` 记录每个实际调用的 PCM sample 起止 cursor、相对起止时间、文本 hash 和错误；`priority_vad` 的 chunk hash 对应该块去重后的文本贡献。`resume` 记录停止 chunk、下一 chunk、停止点、交互开始点和恢复点。整轨与续跑文本做首尾空白整理后按 `\n` 拼接，`long_text_hash` 是拼接文本的 SHA-256。

`long_segments[]` 保留 MTD 每个返回段的文本/hash、原始 speaker 和所在 chunk index，并将局部 `Start`/`End` 加上 chunk 的 sample 起点，转换成整轨绝对毫秒。overlap 内的返回段只做时间平移，没有另造 segment ownership 去重规则。`priority` 和 `priority_vad` 的 `comparison_to_baseline` 分别记录 `long_text_equal`、双方长文本 hash、segment 数量、同序号 `segment_timestamps_equal` 和 `speaker_labels_equal`；transcribe 没有 diarization segment 时后两项为 `null`。为兼容第 2 轮报告，顶层 `long_text_equal` 仍专指 baseline 与普通的 priority 比较。

顶层 `speaker_stitching` 固定为 `value:null` 并带 `unavailable_reason`：当前 MTD API 只给出 chunk-local `S01`/`S02` 和时间戳，没有 voice embedding 或 global cluster，不能安全跨块缝合身份。本实验不按标签出现顺序映射说话人，也不会把原始局部标签改写后声称已经 stitched。

为防止把没有发生的竞争写成有效实验，交互必须在长任务未完成时到达，两种 priority 还必须真的在剩余 chunk 前停止并恢复；否则该方法 `status="invalid"`。长音频应明显长于一个 chunk，且首个 chunk 在目标机器上应运行超过 100 ms。任一请求失败、音频为空或不是 PCM16、调度前提不满足，都不会产生成功结果；三种方法都成功时顶层状态才是 `success`。文本或其他值不可用时为 `null`，不填 0；本入口不新增 RSS、footprint、swap、thermal 等资源采样，所以 `resources.value=null` 并带原因，而不是虚构资源零值。

限制：decode 和模型加载不在方法计时内；baseline、priority、priority_vad 固定顺序执行，共享暖 ASR 的缓存和热状态可能携带，VAD 只在第三种方法前做整轨预规划；固定 100 ms 到达点不是生产热键事件。普通 priority 仍不做 VAD、overlap 或去重；`priority_vad` 虽补齐 VAD plan、文本 overlap 去重和绝对时间平移，但精确词边界去重不等于生产 segment ownership 重建，也不证明识别或分人质量。`long_text_equal`、时间戳和标签相等性都只是精确对照，不是 DER/CER/WER、质量分或身份对齐。分块 MTD 丢失整请求内的 speaker 稳定性；生产方案后续必须暴露并保留 native speaker embedding，执行跨块 global clustering，并用 DER、漏检和说话人混淆验收，不能用本地标签顺序替代。本入口没有 AC/idle/contention 门禁和 median-of-3 汇总，不能直接把单次结果当性能结论；不修改生产调度、不实现 UI 或全局身份映射。

离线回归：

```sh
CGO_ENABLED=0 go test -run 'TestMeetingInteractiveFake|TestInteractiveConfig' ./internal/profile
go test -run 'TestMeetingInteractive' .
```

第二条默认只验证真实入口的 opt-in skip；未设置 `VOXTNEW_INTERACTIVE_CONFIG` 时不会启动真实模型。
