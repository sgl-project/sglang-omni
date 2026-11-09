# 任务侦察：SGLang-Omni Apple Silicon 路线图（#2888）

- 机器：MacBook Pro（Mac16,7），Apple M4 Pro，48 GB
- 生成时间：2026-11-01 13:50 PST（21:50 UTC）出初稿，13:53 PST 核对后定稿。14:18–14:27 PST 用登录账户补查了初稿第 6 节没做完的检查，结论没变
- 路线图：[#2888 [Roadmap] Apple Silicon (native MLX) 2026 Q4](https://github.com/sgl-project/sglang-omni/issues/2888)
  - `updated_at` = **2026-11-01T21:32:40Z**。这是 13:50 PST 用 API 读到的值。登录后 14:18:46 和 14:25:49 PST 又读了两次，没有变，仍是 16 条评论，最新一条就是下面 luojiaxuan 21:32:40Z 那条。
  - 13:28 PST 读到的还是 20:16:32Z，也就是 minyi-tang 结果评论（[comment](https://github.com/sgl-project/sglang-omni/issues/2888#issuecomment-6561269216)）最后一次编辑的时间。
  - 变化来自 luojiaxuan 21:32:40Z 新发的一条 bug 报告（[comment](https://github.com/sgl-project/sglang-omni/issues/2888#issuecomment-6564187167)）。这条不是认领，见第 6 节。
  - 状态 open，16 条评论，没有 assignee，没有 label。正文的行号和 13:28 时一致。
- **15:35 PST 复查**：`updated_at` 变为 22:45:07Z，多了两条评论，正文没改。结论和更正见第 7 节。
- 数据：13:16–13:52 PST 用 GitHub REST API（未登录）实时抓取。14:18–14:27 PST 用户登录 cosekii 账户后补查，只发 REST GET 和 GraphQL 查询，没有任何写操作。没有用本地 clone、记忆或搜索引擎摘要判断认领状态。

## 0. 结论

| 漏斗 | 数量 |
|---|---|
| 未勾选 | 72 条复选框，另有 A/B/C 三张表 21 行（没有复选框，也逐行列出）。共 93 条 |
| 没人认领 | 37 条（复选框 23 条，表格 14 行），另有 1 个子范围没人认领（236 行）。15:35 PST 前是 38 条，270 行改判为已完成 |
| 没人认领且这台机器能跑（FIT） | 11 条，加上那个子范围。15:35 PST 前是 12 条，341 行改为 PARTIAL |
| 其中适合他（≥4 分） | 1 个，就是 236 行的那个子范围。完整条目里没有 ≥4 分的 |

14:18–14:27 PST 登录后补查了全部 PR、issue、评论、行内 review 和 Discussion #2158（第 6 节），上面的计数和认领状态都没有变化。

15:35 PST 复查（第 7 节）：

- 270 行改判为已完成，没人认领的从 38 条变为 37 条。341 行改为 PARTIAL，能跑的从 12 条变为 11 条。
- Top 1 仍没人认领。但这一块的认领人刚说整条已经完成，所以要先问维护者还要不要。
- 另外出现两条新的验证项：在 app 里实测 #3419 和 #3420。

Top 5 只有 1 个：**在 32 GB 以上的 Mac 上，重测 10 GB 以上模型（Drama 3 4-bit、Music 3）的空闲内存**。

这一块的认领人在结果里写明“Not measured on Macs with 32 GB or more”。详见第 4 节。

其余没人认领的条目分三类：

- 依赖还没合并的 #3043（重构版 Voxt）。
- 需要 iPhone、旧版 macOS 或发布签名。
- 维护者自己的决定。

另外有 5 条 3 分的“接近”项，第 4 节单独列出。

## 1. 机器画像（13:41 PST 实时采集）

| 项 | 值 |
|---|---|
| 主机名 | 按你的要求不写（主机名里包含序列号） |
| 型号 | MacBook Pro，`Mac16,7` |
| 芯片 | Apple M4 Pro。CPU 14 核（10 性能核 + 4 能效核），GPU 20 核，Metal 4 |
| 内存 | 48 GB（`hw.memsize` = 51,539,607,552 字节） |
| 内存预算 | 75% = **36 GB** |
| 系统 | macOS 27.2（`BuildVersion 26B123`），arm64 |
| Xcode | 27.2（`27B52`），协议已接受 |
| 磁盘 | 系统盘 926 GiB，剩余 364 GiB |
| 电源 | 接着电源，电池 100% |
| Python / MLX | `~/voxt-dev-setup/venv-apple`：Python 3.12.13，`mlx` 0.32.4 |
| 当前内存状况 | 13:35 PST `top`：PhysMem 46G used，其中 19G 在 compressor，unused 736M。大头是 Cursor、WindowServer、Chrome、Outlook、Slack |

**今天在这台机器上实际跑通过的东西**（来自同一台机器的 Voxt 复测，证据在 `~/voxt-dev-setup/REPORT.md`）：

- Voxt 的 `xcodebuild` 编译和完整测试：2052 个测试，0 失败。
- native runtime 用 CLT 和 Xcode 都编得出来。
- golden 在 exact 模式下和 Apple M4 Pro 的 golden 392/392 一致，native API 测试 83 个全过。
- 本地打包能完成，`install.sh` 能装好。
- SeedTTS EN 基准：
  - MLX 4-bit：11.346 samples/s，WER 1.922%。
  - Torch/MPS 全量：3.350 samples/s，WER 1.581%，stage 进程峰值 4411 MB（13:57 PST 跑完）。

**按内存预算算的适配**（每个参数 4-bit ≈ 0.5 B、8-bit 1 B、bf16 2 B，再加 20%；有实测就用实测）：

| 模型 | 估算或实测 | 预算 36 GB 内？ |
|---|---|---|
| Fish Drama 3 4-bit | 实测峰值 28.7 GB（[lyttttt3333 10-30T03:38:31Z](https://github.com/sgl-project/sglang-omni/issues/2888#issuecomment-6514362238)） | 能 |
| Fish Drama 3 8-bit | 实测 47.1 GB（同上）和 46.0 GB（[larryliu0820 11-01T04:41:48Z](https://github.com/sgl-project/sglang-omni/pull/3338#issuecomment-6553517165)） | 不能 |
| MiniMax Music 3 mxfp8 | 权重 10.412 + 2.493 GiB（[#2008](https://github.com/sgl-project/sglang-omni/pull/2008)）。native 实现峰值 15.97–17.92 GB（[#3173](https://github.com/sgl-project/sglang-omni/pull/3173)，24 GiB 机器上有 swap 时测的；#1990 实现是 16.62–18.54 GB） | 能 |
| Ming-omni-tts 16.8B-A3B 4-bit | checkpoint 9.4 GB，空闲 9.75 GB（[#3411](https://github.com/sgl-project/sglang-omni/pull/3411)） | 能 |
| Qwen3-Omni 30B-A3B 4-bit | 估算 30 × 0.5 × 1.2 = 18 GB | 能 |
| Higgs Audio v3 4B | 按 bf16 估算 4 × 2 × 1.2 = 9.6 GB | 能 |

## 2. 方法和判断规则

**抓了什么：**

- [#2888](https://github.com/sgl-project/sglang-omni/issues/2888)：正文 355 行，16 条评论。
- [#3351](https://github.com/sgl-project/sglang-omni/issues/3351)（offload 路线图）的 6 条评论。正文写着 “Claim tasks in comments using their IDs”。
- [#2537](https://github.com/sgl-project/sglang-omni/issues/2537)（Voxt 路线图）的 2 条评论。
- [#2941](https://github.com/sgl-project/sglang-omni/issues/2941)（推测解码路线图）的 3 条评论。
- [#3338](https://github.com/sgl-project/sglang-omni/pull/3338) 的 5 条评论。
- PR 状态：
  - search API 扫了 open PR 按更新时间的前 300 个（共 724 个）、9 月以来合并的前 300 个（共 784 个）、9 月以来关闭没合并的全部 111 个。
  - 扫不到的 13 个 PR 用 core API 逐个查。
- 正文：读了约 25 个关键 PR 的正文，以及 `main...release/voxt-1.16` 的 compare。
- 关键词搜索：对没人认领的条目搜了 PR 和 issue 标题。
- 14:18–14:27 PST 登录后补查：
  - 全部 open PR，以及 9 月以来更新过的关闭 PR 和 issue，标题加正文全文匹配。
  - #2888 创建（10-17）以来的全部对话评论和行内 review 评论。
  - Discussion #2158。
  - 详见第 6 节。

**认领分类：**

| 类别 | 含义 |
|---|---|
| DONE | 对应 PR 已合并。路线图没勾的写“路线图未更新” |
| IN PROGRESS | 有 open PR |
| CLAIMED | 有 assignee，或评论里明确认领、并且 30 天内有活动 |
| STALE CLAIM | 认领后超过 30 天没有活动 |
| UNCLAIMED | 评论和 PR 里都没找到认领 |

“需维护者确认”表示证据指向已完成，但路线图本身没写。

**机器结论：** FIT / PARTIAL / NOT-FIT（芯片、设备、系统）/ DEFERRED（依赖没就绪）。

**个人适合度（1–5）：** 按六个方面综合打分：技能匹配、上手成本、规模、验收是否客观、白做的风险、学习和曝光。已有人做或已完成的条目一律 1 分。

## 3. 完整表格

行号是 #2888 正文的行号。链接里 `#N` 都指向 sglang-omni 仓库。

### Layer 1 · A 表（Bugs）

| 板块 | 条目 | 链接 | 认领状态和证据 | 机器结论 | 适合度 |
|---|---|---|---|---|---|
| L1·A | A1（52 行）热键后有波形没文字 | [#3394](https://github.com/sgl-project/sglang-omni/pull/3394) | IN PROGRESS。minyi-tang [10-31T06:01:26Z 认领](https://github.com/sgl-project/sglang-omni/issues/2888#issuecomment-6539847651)：“Claiming Layer 1 bugs **A1**, **A5** and **A6**”。#3394 open，11-01T21:24 更新 | FIT：Voxt 今天在这台机器上编译测试都过了 | 1：已有人在做 |
| L1·A | A2（53 行）播放和录音互相抢 | [#2888](https://github.com/sgl-project/sglang-omni/issues/2888) | UNCLAIMED。16 条评论没有提到。标题搜 `playback` 只有 cascade 和 benchmark 的 PR；搜 `("video call" OR ducking OR "AVAudioSession") Voxt`，前几条都是不相关的 Voxt PR | PARTIAL：要有另一方配合开视频通话才能复现 | 2：CoreAudio 和 Swift 不是他的主栈；原文写着 “Not yet decided whether this is intended”，没有验收标准 |
| L1·A | A3（54 行）长时间 Qwen3-ASR 会话内存增长 | [#3393](https://github.com/sgl-project/sglang-omni/pull/3393) | IN PROGRESS。XXXXRT666 的 #3393（10-31 创建，open）正文写 “Part of #2888 (A3).”、“Closes #3122.”。相关：renxinx [10-29T18:13:58Z](https://github.com/sgl-project/sglang-omni/issues/2888#issuecomment-6504728118) 报 70 分钟涨到 48.19 GB；draft [#3172](https://github.com/sgl-project/sglang-omni/pull/3172)、[#3360](https://github.com/sgl-project/sglang-omni/pull/3360) | FIT | 1：已有人在做 |
| L1·A | A4（55 行）#2676 的 review 需要 M4 输出 | [#3064](https://github.com/sgl-project/sglang-omni/pull/3064) | DONE（需维护者确认）。原文写 “Tracked with the M4 Pro goldens item under CI”，而 yxs 的 #3064 “Record M4 Pro goldens for Nemotron and Parakeet” 10-22T19:45 已合并 | FIT：这台就是 M4 Pro | 1：已完成 |
| L1·A | A5（56 行）下载的模型消失 | [#3416](https://github.com/sgl-project/sglang-omni/pull/3416) | IN PROGRESS。同上 minyi-tang 的认领，#3416 open，11-01T21:26 更新 | FIT | 1：已有人在做 |
| L1·A | A6（57 行）开头一两个音节被吞 | [#3417](https://github.com/sgl-project/sglang-omni/pull/3417) | IN PROGRESS。同上，#3417 open | FIT | 1：已有人在做 |

### Layer 1 · B 表（UX，原文要求在重构版里做）

B 表和 C 表都没找到单独的认领：评论里没有，标题搜 `preview`、`export`、`system audio` 也没有对应的 PR。284 行要求在重构 PR 上把 B/C 里缺的继续做完，所以这些条目都依赖 [#3043](https://github.com/sgl-project/sglang-omni/pull/3043)。

| 板块 | 条目 | 链接 | 认领状态和证据 | 机器结论 | 适合度 |
|---|---|---|---|---|---|
| L1·B | B1（63 行）预览窗口太小 | [#2537 评论](https://github.com/sgl-project/sglang-omni/issues/2537#issuecomment-6374246838) | UNCLAIMED。zhaochenyang20 10-23T20:37:13Z 在 #2537 提过同样的需求，没人接 | DEFERRED：依赖 #3043 | 2：要改别人 35 万行的 PR，UI 验收主观 |
| L1·B | B2（64 行）prompt 编辑框太小 | #3043 | UNCLAIMED | DEFERRED：依赖 #3043 | 2：同上 |
| L1·B | B3（65 行）设置分散 | [#3293](https://github.com/sgl-project/sglang-omni/pull/3293) | UNCLAIMED。longbowzz 的 #3293（open，现有 app 上的 “Apply to Scenes”）相关，但不是重构版 | DEFERRED：依赖 #3043 | 2：同上 |
| L1·B | B4（66 行）设置一键导入导出 | #3043 | UNCLAIMED。标题搜 `export` + Voxt 为 0 | DEFERRED：依赖 #3043 | 2：同上 |
| L1·B | B5（67 行）功能太多 | #3043 | IN PROGRESS。#3043 本身就是 “a rewrite in Go with a reduced feature set” | DEFERRED：依赖 #3043 | 1：属于重构本身 |
| L1·B | B6（68 行）界面上看不出用的哪个后端 | #3043 | UNCLAIMED | DEFERRED：依赖 #3043 | 2：同上 |
| L1·B | B7（69 行）系统音频开关难找 | #3043 | UNCLAIMED | DEFERRED：依赖 #3043 | 2：同上 |
| L1·B | B8（70 行）长录音最终转写慢 | [#3062](https://github.com/sgl-project/sglang-omni/pull/3062) | IN PROGRESS。原文 “Acceptable for now; see also Meeting mode live preview”，对应 fweilun 的 #3062（open，10-29 更新） | FIT | 1：已有人在做 |

### Layer 1 · C 表（讨论过、没实现的功能）

| 板块 | 条目 | 链接 | 认领状态和证据 | 机器结论 | 适合度 |
|---|---|---|---|---|---|
| L1·C | C1（76 行）翻译和润色分开 | #2888 | UNCLAIMED | DEFERRED：重构版、产品决定 | 2：需求还在讨论，没有验收标准 |
| L1·C | C2（77 行）按 app 切换热词 | #2888 | UNCLAIMED。标题搜 `hotword` 只有一个无关的 fun_asr PR #1874 | DEFERRED：同上 | 2：同上 |
| L1·C | C3（78 行）MOSS 容易被热词带偏 | #2888 | UNCLAIMED | DEFERRED：同上 | 2：同上 |
| L1·C | C4（79 行）重构版保留远程 API | #3043 | UNCLAIMED（见 286 行） | DEFERRED：依赖 #3043 | 2：验证本身见 286 行 |
| L1·C | C5（80 行）默认本地 LLM 用 SGLang MLX | #2888 | UNCLAIMED。原文 “Niche … Low priority” | DEFERRED | 1：优先级低 |
| L1·C | C6（81 行）模型下载兼容、显示进度 | #2888 | UNCLAIMED | DEFERRED：重构版 | 2：需求不具体 |
| L1·C | C7（82 行）按模型调参数 | #2888 | UNCLAIMED。原文 “To be listed as validation proceeds” | DEFERRED | 1：范围还没定 |

### Layer 1 · D 清单（重构版的回归检查，复选框）

| 板块 | 条目 | 链接 | 认领状态和证据 | 机器结论 | 适合度 |
|---|---|---|---|---|---|
| L1·D | 86 行：#2996 热键权限和 OpenAI API 格式的专门测试 | [#2996](https://github.com/sgl-project/sglang-omni/pull/2996) | UNCLAIMED（指重构版）。#2996 本身 10-20 已合并 | DEFERRED：依赖 #3043 | 2：要在别人的大 PR 里写 Go 测试，PR 10 天没动 |
| L1·D | 87 行：#2833 VAD 会话泄漏的专门测试 | [#2833](https://github.com/sgl-project/sglang-omni/pull/2833) | UNCLAIMED（指重构版）。#2833 10-16 已合并 | DEFERRED：依赖 #3043 | 2：同上 |
| L1·D | 88 行：A1 的修复和测试 | #3394 | IN PROGRESS（现有 app，minyi-tang，测试在 `voxt-unit`）。重构版里没人认领 | DEFERRED：重构版部分 | 1：已有人在做 |
| L1·D | 89 行：A6 不吞开头，按键到首帧延迟测试 | #3417 | IN PROGRESS（现有 app） | DEFERRED：重构版部分 | 1：已有人在做 |
| L1·D | 90 行：C4 远程 API 路径还能用 | #3043 | UNCLAIMED | DEFERRED：依赖 #3043 | 2：同 286 行 |
| L1·D | 91 行：长听写 5/10/30 分钟不退化 | #3043；[#3360](https://github.com/sgl-project/sglang-omni/pull/3360) | UNCLAIMED（指重构版）。native runtime 上的长音频由 #3360 在测 | DEFERRED：依赖 #3043 | 2：依赖 #3043 |
| L1·D | 92 行：系统音频采集还能用 | #3043 | UNCLAIMED | DEFERRED：依赖 #3043 | 2：同上 |
| L1·D | 93 行：A5 模型目录不在 Downloads | #3416 | IN PROGRESS（现有 app） | DEFERRED：重构版部分 | 1：已有人在做 |

### Layer 2 · Cleanup

| 板块 | 条目 | 链接 | 认领状态和证据 | 机器结论 | 适合度 |
|---|---|---|---|---|---|
| L2·Cleanup | 109 行：native 源码按模型分目录 | [#2931](https://github.com/sgl-project/sglang-omni/pull/2931) | DONE。10-19T20:25 合并，路线图未更新 | FIT | 1：已完成 |
| L2·Cleanup | 110 行：迁移或关闭仍然指向 `sglang_omni` 的 Apple PR | [#2306](https://github.com/sgl-project/sglang-omni/pull/2306)、[#1960](https://github.com/sgl-project/sglang-omni/pull/1960) | IN PROGRESS。#2306 已于 11-01T14:11 关闭（没合并），minyi-tang 结果评论写 “#2306 fold-in (Cleanup): done.”。还开着的旧 PR：TTS 的 #1960（替代的 #3139 已合并）、#2085、#2017、#2243、#2055；Qwen3-Omni 的 #2130、#2783（164 行） | FIT：不需要硬件 | 1：关别人的 PR 需要维护者或作者权限 |
| L2·Cleanup | 113 行：共享 native 组件 | [#3108](https://github.com/sgl-project/sglang-omni/pull/3108)、[#3179](https://github.com/sgl-project/sglang-omni/pull/3179) | IN PROGRESS。#3108 10-25 已合并；altale 的 #3179 open（10-25 更新） | FIT | 1：已有人在做 |

### Layer 2 · Models

| 板块 | 条目 | 链接 | 认领状态和证据 | 机器结论 | 适合度 |
|---|---|---|---|---|---|
| L2·Models | 134 行：ARK-ASR-3B | [#2956](https://github.com/sgl-project/sglang-omni/pull/2956) | IN PROGRESS。AzizMuminov，open，10-22 更新 | FIT：3B，按 bf16 估 7.2 GB | 1：已有人在做 |
| L2·Models | 135 行：Ming3 ASR | [#3219](https://github.com/sgl-project/sglang-omni/pull/3219) | IN PROGRESS。guozhihao-224，open，10-26 更新；#2977 10-31 已合并 | 未估算：PR 里没写体积 | 1：已有人在做 |
| L2·Models | 136 行：Ming-3.0-audio-tiny | [#3262](https://github.com/sgl-project/sglang-omni/pull/3262) | IN PROGRESS。altale，open，10-29 更新 | 未估算 | 1：已有人在做 |
| L2·Models | 144 行：Ming-omni-tts 16.8B-A3B 4-bit | [#2870](https://github.com/sgl-project/sglang-omni/pull/2870) | IN PROGRESS。altale，open，11-01 更新 | FIT：9.4 GB | 1：已有人在做 |
| L2·Models | 145 行：AuK | [#2444](https://github.com/sgl-project/sglang-omni/pull/2444) | IN PROGRESS。xuchen-w，open，11-01T20:25 更新 | 未估算 | 1：已有人在做 |
| L2·Models | 146 行：AuK-Flash | #2444 | IN PROGRESS。同上 | 未估算 | 1：已有人在做 |
| L2·Models | 148 行：dots.tts | [#3105](https://github.com/sgl-project/sglang-omni/pull/3105) | IN PROGRESS。guozhihao-224，open，10-23 更新 | 未估算 | 1：已有人在做 |
| L2·Models | 149 行：Fish Drama-3 | [#3338](https://github.com/sgl-project/sglang-omni/pull/3338) | DONE。11-01T18:02 合并，路线图未更新 | PARTIAL：4-bit 能跑（28.7 GB），8-bit 不能（47.1 GB） | 1：已完成 |
| L2·Models | 153 行：Higgs Audio v3 4B | [#3137](https://github.com/sgl-project/sglang-omni/pull/3137) | IN PROGRESS。JiaxinD，open，11-01 更新 | FIT：估 9.6 GB | 1：已有人在做 |
| L2·Models | 154 行：MOSS-TTS-Local | [#3138](https://github.com/sgl-project/sglang-omni/pull/3138) | IN PROGRESS。JiaxinD，open | 未估算 | 1：已有人在做 |
| L2·Models | 155 行：Qwen3-TTS CustomVoice | [#3139](https://github.com/sgl-project/sglang-omni/pull/3139) | DONE。10-31T17:34 合并 | FIT | 1：已完成 |
| L2·Models | 156 行：Fun-CosyVoice3 | [#3140](https://github.com/sgl-project/sglang-omni/pull/3140) | DONE。11-01T20:01 合并 | FIT：空闲 2.82 GB | 1：已完成 |
| L2·Models | 157 行：Chatterbox-Turbo | [#3141](https://github.com/sgl-project/sglang-omni/pull/3141) | IN PROGRESS。JiaxinD，open | 未估算 | 1：已有人在做 |
| L2·Models | 158 行：Fish S2-Pro | [#3142](https://github.com/sgl-project/sglang-omni/pull/3142) | IN PROGRESS。JiaxinD，open | 未估算 | 1：已有人在做 |
| L2·Models | 164 行：Qwen3-Omni 30B-A3B 4-bit | [#3154](https://github.com/sgl-project/sglang-omni/pull/3154) | IN PROGRESS。adityavaid，open，10-25 更新。是否迁移由维护者决定 | FIT：估 18 GB | 1：已有人在做 |

### Layer 2 · 每个模型 PR 的验收清单（189–196 行）

这 8 条是每个模型 PR 自带的验收要求，不是独立任务。认领状态、机器结论和适合度都不适用。

| 板块 | 条目 | 链接 | 认领状态和证据 | 机器结论 | 适合度 |
|---|---|---|---|---|---|
| L2·验收 | 189 行：按 TTS/ASR bench 和 MLX 指南的方法评估 | #2888 | 不适用 | — | — |
| L2·验收 | 190 行：正确性和性能在同一样本集上测 | #2888 | 不适用 | — | — |
| L2·验收 | 191 行：SeedTTS 式 TTS 正确性检查 | #2888 | 不适用 | — | — |
| L2·验收 | 192 行：并发 1 和 16 | #2888 | 不适用 | — | — |
| L2·验收 | 193 行：和其它开源 MLX 框架比速度，结果贴到 Discussion #2158 | #2888 | 不适用 | — | — |
| L2·验收 | 194 行：打包测试、外部协议的单测 | #2888 | 不适用 | — | — |
| L2·验收 | 195 行：Lint 和 Mac CI 通过 | #2888 | 不适用 | — | — |
| L2·验收 | 196 行：在 PR 里贴指标，标记可以合并 | #2888 | 不适用 | — | — |

### Layer 2 · MiniMax Music 3 和 Drama 3

| 板块 | 条目 | 链接 | 认领状态和证据 | 机器结论 | 适合度 |
|---|---|---|---|---|---|
| L2·Music3 | 212 行：用 `hf` CLI 下载权重 | [#2927](https://github.com/sgl-project/sglang-omni/pull/2927) | UNCLAIMED。没有单独认领。#2927 和 #2008 已经用固定 revision 的 checkpoint，这条可能已经不需要单独做 | FIT：约 13 GiB，磁盘够 | 1：没有交付物 |
| L2·Music3 | 213 行：建立基线 A 并保存输入输出 | #2927 | DONE。#2927（10-24 合并）以冻结的 #1990 为数值基线，波形 parity 通过（peak ≤ 0.001，RMS ≤ 0.0001），并记录了 golden。原文：“only Apple M5 is recorded here” | FIT | 1：已完成 |
| L2·Music3 | 214 行：实现 stage offload（#2008） | [#2008](https://github.com/sgl-project/sglang-omni/pull/2008) | DONE。10-26T02:20 合并 | FIT | 1：已完成 |
| L2·Music3 | 215 行：offload 和不 offload 对比 | #2008 | DONE。#2008 正文有 “Resident versus offload measurements” 表，测试机是 Apple M5 24 GiB | FIT：不 offload 峰值 16–18 GB | 1：已完成。附带发现见第 4 节 |
| L2·Drama3 | 221 行：带 offload 跑通 Drama 3 | [#3351 评论](https://github.com/sgl-project/sglang-omni/issues/3351#issuecomment-6544114123) | CLAIMED。#3351 的 C1 是 “Drama3 disk-backed expert cache — After #3338 lands, prototype bounded expert residency”。lgyStoic 10-31T13:40:30Z 认领：“I'd like to contribute to C1, building on #3338.”。B1（“Drama3 pilot … through existing SGLang MoE execution paths”）由 zesiar0 [10-31T09:14:07Z](https://github.com/sgl-project/sglang-omni/issues/3351#issuecomment-6541566279) 认领 | FIT：4-bit 不 offload 也能放下，可以做对照 | 1：已认领，而且是 C++ native 代码 |
| L2·Drama3 | 222 行：邀请有 MLX 实现的外部贡献者提 PR | #3338 | DONE（需维护者确认）。外部贡献者 lyttttt3333（CONTRIBUTOR）的 #3338 已合并 | — | 1：维护者的事 |
| L2·Drama3 | 223 行：公平对比那份实现和我们的 | #3338 | DONE（需维护者确认）。lyttttt3333 在 #2888 写：和 mlx-audio 0.6.2 相比，“seeded codes and PCM16 are bit-identical … real-time factor is within 3%” | FIT | 1：已完成 |

### Layer 2 · Performance

| 板块 | 条目 | 链接 | 认领状态和证据 | 机器结论 | 适合度 |
|---|---|---|---|---|---|
| L2·Perf | 227 行：低精度和批处理 kernel（父项） | #2708、#2713 | IN PROGRESS。两个子项都有 open PR | FIT | 1：已有人在做 |
| L2·Perf | 228 行：#2708 批处理 Silero VAD 和 Sortformer | [#2708](https://github.com/sgl-project/sglang-omni/pull/2708) | IN PROGRESS。JiaxinD，open，最后更新 10-12T05:54Z（20 天） | FIT | 1：已有人在做 |
| L2·Perf | 229 行：#2713 Qwen3-ASR 音频编码器用 bf16 | [#2713](https://github.com/sgl-project/sglang-omni/pull/2713) | IN PROGRESS。同上 | FIT | 1：已有人在做 |
| L2·Perf | **236 行：TTS 和 10 GB 以上模型的空闲内存重测** | [#3411](https://github.com/sgl-project/sglang-omni/pull/3411) | **整条 IN PROGRESS，但有一个子范围 UNCLAIMED**。详见第 4 节。15:35 PST：认领人 [22:45:07Z](https://github.com/sgl-project/sglang-omni/issues/2888#issuecomment-6565011674) 认为整条已经 “done”，请 JiaxinD 勾选，见第 7 节 | **FIT** | 整条 1；**子范围 4**，要先问维护者 |
| L2·Perf | 237 行：大模型的 stage offload 另说 | [#3351](https://github.com/sgl-project/sglang-omni/issues/3351) | IN PROGRESS。#3351 的 assignee 是 AkazaAkane；Music3 已有 #2008，Drama3 有 C1 的认领 | FIT | 1：已有人在做 |
| L2·Perf | 238 行：长音频 ASR 压测，说明 30 秒窗口的分块 | [#3360](https://github.com/sgl-project/sglang-omni/pull/3360) | IN PROGRESS。OwenMagzzz 的 draft #3360（10-30 创建）：“Apple M4 Pro, 64 GiB … All eight 5/10/30/60-minute realtime cases completed”。另有 #3393 限制长请求的内存 | FIT | 1：已有人在做。“说明分块”的文档部分没看到单独的 PR，但属于同一条 |

### Layer 2 · 推测解码（原文写着不在本里程碑）

| 板块 | 条目 | 链接 | 认领状态和证据 | 机器结论 | 适合度 |
|---|---|---|---|---|---|
| L2·Spec | 256 行：调研 MLX 生态的推测解码实现 | #2888 | UNCLAIMED。#2888 评论没有；#2941 的 P4 也没有调研项 | FIT | 2：调研没有客观验收，原文写 “out of scope until the basics are done” |
| L2·Spec | 257 行：研究 SpeechSpec | [#2941](https://github.com/sgl-project/sglang-omni/issues/2941) | IN PROGRESS。WilsonZheng0327 的 #2941 已勾 “Serve SpeechSpec's DSpark draft…” 等多项 | FIT | 1：已有人在做 |
| L2·Spec | 258 行：在 SGLang Omni MLX 上为一个模型做原型 | [#3285](https://github.com/sgl-project/sglang-omni/pull/3285) | IN PROGRESS。WilsonZheng0327 的 #3285（Qwen3-ASR + DFlash draft，native，open，11-01T20:44 更新），以及 #3352 | FIT | 1：已有人在做 |

### Layer 3 · Releases

| 板块 | 条目 | 链接 | 认领状态和证据 | 机器结论 | 适合度 |
|---|---|---|---|---|---|
| L3·Rel | 268 行：把 `release/voxt-1.16` 的改动带回 main | [#2683](https://github.com/sgl-project/sglang-omni/pull/2683) | DONE（需维护者确认）。#2683 10-22T17:47 合入 `main`。main 上的 `Voxt/tools/package_local_app.sh` 已有 `--provisioning-profile`（第 52 行）和 `xcrun notarytool submit`（第 218 行）。compare `main...release/voxt-1.16` 显示 `diverged`、ahead_by 17、behind_by 380，ahead 的提交里有 cherry-pick，单看 SHA 不能判断内容是否缺 | — | 1：发布分支归维护者管 |
| L3·Rel | 269 行：让 1.15 能通过应用内更新升到 1.16 | #2888 | UNCLAIMED。评论里没有。标题搜 `(Sparkle OR appcast OR updater OR "in-app update")` 为 0（13:51 PST） | PARTIAL：要装 1.15，还要签名的发布包和 appcast 密钥 | 1：要用维护者的发布凭据 |
| L3·Rel | 270 行：native runtime 没有缺口后去掉 MLX Audio | [#2683](https://github.com/sgl-project/sglang-omni/pull/2683)、[#3412](https://github.com/sgl-project/sglang-omni/pull/3412) | DONE（需维护者确认，15:35 PST 更正，原先写的 UNCLAIMED 不对）。minyi-tang [11-01T22:37:49Z](https://github.com/sgl-project/sglang-omni/issues/2888#issuecomment-6564931064)：“on main, Voxt no longer links mlx-audio-swift. #2683 removed it on 10-22”。实时核对 main `e5b87a8` 的 `Voxt/Voxt.xcodeproj/project.pbxproj`，`MLXAudio` 0 处 | — | 1：已完成 |
| L3·Rel | 271 行：OmniTyper 功能映射 | [#2614](https://github.com/sgl-project/sglang-omni/pull/2614) | IN PROGRESS。kalebguo 的 draft `[WIP]`，10-08 创建，最后更新 10-10T00:59Z（22 天，还没到 30 天） | FIT | 1：已有人在做 |
| L3·Rel | 272 行：会议模式实时预览 | #3062 | IN PROGRESS。fweilun 10-15T16:09:09Z 在 [#2537 认领](https://github.com/sgl-project/sglang-omni/issues/2537#issuecomment-6196625162)，#3062 open，10-29 更新 | FIT | 1：已有人在做 |
| L3·Rel | 273 行：录演示 | #2888 | UNCLAIMED。标题搜 `demo` + Voxt 为 0 | FIT | 1：不是代码，维护者的事 |
| L3·Rel | 274 行：确定 native runtime 最低支持的 macOS | #2888 | UNCLAIMED。标题搜 `("minimum macOS" OR "macOS 14" OR "macOS 13" OR "deployment target")` 为 0。登录后全文补查也没有认领。相关的只有 [#2683](https://github.com/sgl-project/sglang-omni/pull/2683) 正文里的一句：“The app's deployment target changes from macOS **15.0 to 26.2** … remains a separate product-scope confirmation item”。这说的是 app，不是 native runtime | PARTIAL（系统）：原文暂定 15，“untested on older systems”。这台是 macOS 27.2，要在更老的系统上测，得用另一台 Mac 或虚拟机；虚拟机里能不能跑 MLX 没验证 | 2：要额外的环境 |
| L3·Rel | 275 行：把 Voxt 移到二级目录 | #2888 | UNCLAIMED。搜 `("MLX Software" OR "second-level directory")` 只命中 #2888 本身 | FIT | 1：目录结构由维护者决定，而且会和所有 open 的 Voxt PR 冲突 |

### Layer 3 · 重构版 Voxt（#3043）

#3043 的概况（实时数据）：

- hehehai 的 open PR，10-22T02:57Z 创建，之后没有更新过。
- +359,064/−2 行，1661 个文件，151 个提交。
- 0 条评论，0 条 review comment，`mergeable_state` = `unstable`。
- 原文：“The refactor is a rewrite in Go … regression tests can run on Linux; a Mac is only needed for UI checks and real device input.”

| 板块 | 条目 | 链接 | 认领状态和证据 | 机器结论 | 适合度 |
|---|---|---|---|---|---|
| L3·Refactor | 283 行：本地装重构版，和 1.16.3 对比 | [#3043](https://github.com/sgl-project/sglang-omni/pull/3043) | UNCLAIMED。#3043 上没有任何评论和 review，#2888 评论里也没人认领 | FIT | **3**：验收是他的强项，也能读 Go。但 PR 体量大，10 天没动，验证结果可能白做，交付物只是一条 review |
| L3·Refactor | 284 行：逐项核对 B、C 表 | #3043 | UNCLAIMED | FIT | **3**：清单很客观，其它同上 |
| L3·Refactor | 285 行：逐项核对 D 清单 | #3043 | UNCLAIMED | FIT | **3**：同上 |
| L3·Refactor | 286 行：确认保留了远程 API（C4） | #3043 | UNCLAIMED | FIT：要一个 OpenAI 兼容的端点，可以用本机起的 server | **3**：同上 |
| L3·Refactor | 287 行：规划合并后怎么下线旧 app | #3043 | UNCLAIMED | FIT | 1：维护者的决定 |
| L3·Refactor | 288 行：验证后合并并发大版本 | #3043 | UNCLAIMED | FIT | 1：要维护者权限 |

### Layer 3 · 架构（iPhone）

| 板块 | 条目 | 链接 | 认领状态和证据 | 机器结论 | 适合度 |
|---|---|---|---|---|---|
| L3·Arch | 326 行：在近期 iPhone 上测 Qwen3-ASR 0.6B 的速度、内存、发热、耗电 | #2888 | UNCLAIMED | NOT-FIT（设备）：要 iPhone | 2：要设备，iOS 不是他的主栈 |
| L3·Arch | 327 行：对比 Qwen3-ASR、iOS 26 系统识别和离线键盘的准确率 | #2888 | UNCLAIMED | NOT-FIT（设备）：键盘离线识别要 iPhone | 2：同上 |
| L3·Arch | 328 行：评估 iOS 打包方式和 app 体积 | #2888 | UNCLAIMED | PARTIAL：不需要真机，但要装 iOS SDK | 2：Swift 和 iOS 不是他的主栈 |
| L3·Arch | 329 行：在 Voxt Remote iOS app 里加端上 ASR | [#3054](https://github.com/sgl-project/sglang-omni/pull/3054) | UNCLAIMED。路线图 17 行：“Voxt Remote, an iOS keyboard and companion app (#3054)”。jt-wang 的 #3054 open，10-23 更新 | NOT-FIT（设备）；而且 DEFERRED：#3054 还没合并 | 1 |

### CI

| 板块 | 条目 | 链接 | 认领状态和证据 | 机器结论 | 适合度 |
|---|---|---|---|---|---|
| CI | 339 行：Nemotron 和 Parakeet 的 M4 Pro golden | #3064 | DONE。10-22T19:45 合并，路线图未更新 | FIT | 1：已完成 |
| CI | 340 行：动到 `Voxt/` 或 `sglang_omni_mlx/` 的 PR 都跑 Mac CI | [#3375](https://github.com/sgl-project/sglang-omni/pull/3375)、[#3396](https://github.com/sgl-project/sglang-omni/pull/3396) | IN PROGRESS。JiaxinD 的 #3375（open，11-01T20:07 更新）；XXXXRT666 的 #3396 叠在它上面 | FIT | 1：已有人在做 |
| CI | 341 行：在 CI 里对比原 Voxt 后端的准确率、速度、内存 | #2888 | UNCLAIMED。评论里没有，标题搜 `backend` + Voxt 为 0 | PARTIAL（15:35 PST 更正，原先写的“两条路径这台都能跑”不对）：原 Voxt 后端（Swift + MLX Audio）在 main `e5b87a8` 和 `a91bcac` 上都已经不链接，`project.pbxproj` 里 `MLXAudio` 0 处。`VOXT_ASR_BACKEND=swift` 现在只是让 native runtime 不启动（`OmniASRBackend.swift` 116–118 行）。只能和冻结的 golden 或旧版本比 | 2：对比对象要维护者先定，CI 正在被 #3375 重构 |
| CI | 342 行：native server 的预编译二进制 | #2888 | UNCLAIMED。标题搜 prebuilt、pre-built、binaries 为 0 | FIT：`build_runtime.sh` 今天在这台机器上 65 秒编完 | 2：签名和发布位置要维护者决定，验收不清楚 |

## 4. Top 5（UNCLAIMED + FIT + ≥4 分）

只有 1 个符合，没有凑数。

### 1. 在 32 GB 以上的 Mac 上，重测 10 GB 以上模型的空闲内存（236 行的子范围）

**认领状态的证据链：**

1. 路线图 236 行：“Re-measure idle memory for TTS (#2870) and for any model of 10 GB or more before applying the same default there.”
2. minyi-tang [10-31T04:36:29Z 认领](https://github.com/sgl-project/sglang-omni/issues/2888#issuecomment-6539082116)，范围写得很明确：“on the native TTS PRs that **fit a 16 GB Apple M4**, starting with #3140 and #2870”。
3. 她的[结果](https://github.com/sgl-project/sglang-omni/issues/2888#issuecomment-6561269216)（11-01T17:19:08Z，20:16:32Z 编辑）原文：
   - “Everything ran on an Apple M4 (10-core GPU), 16 GB”
   - “On 16 GB, "warm" is only 83-100% cached … **Not measured on Macs with 32 GB or more.**”
   - 她在 #2870 的测量评论（[11-01T20:15:24Z](https://github.com/sgl-project/sglang-omni/pull/2870#issuecomment-6563304913)）写：“This is the 10 GB+ case, on the smallest Apple Silicon memory size.”。也就是说，“10 GB 以上”这一项她用 Ming 在 16 GB 上测过了。没测的是 32 GB 以上的机器，以及放不进 16 GB 的模型。
4. Drama 3 4-bit（峰值 28.7 GB）和 Music 3（峰值 16–18 GB）都放不进 16 GB，所以不在她的认领范围里。两个模型都已经合并到 main：#3338 在 11-01，#2927 在 10-24。
5. 16 条评论里没有别人认领这一块。#3411 是唯一的相关 PR，也是她的。

**为什么是这台机器：**

- 48 GB，按 75% 预算有 36 GB，两个模型都能常驻。
- 别人的测试机放不下或要 swap：
  - minyi-tang 的 16 GB 机器上，连 Ming 都只能部分缓存。她还写到：“Two of 15 spawns made right after the machine had been swapping hit first-request stalls of about 190 s and more than 600 s … at the 11.45 GB Metal working-set limit”。
  - #3173 在 24 GiB 的 M5 上测 Music 3，原文：“these numbers are bound by swap, not compute”。
- 今天在这台机器上已经编过 native runtime（Xcode 版 184 秒），也跑过 golden 和 server。

**为什么适合他：**

- 全是测量，用 Python。#3411 的 `sglang_omni_mlx/native/ci/tts_spawn_bench.py` 已经写好。
- 方法已经定了：用第一条评论的方法，5 次冷启动加 5 次热启动取中位数，冷启动用 `mincore` 核实。
- 结果是一张数字表，验收客观。
- 不用写 C++ 和 Swift。
- 结果会出现在 #2888 和 Discussion #2158，作为第一次贡献的曝光合适。

**规模：** S。下载 13.87 GB（`mlx-community/MiniMax-Music3-mxfp8`）加 23.78 GB（`mlx-community/fish-audio-drama-3-4bit`），两个数是 16:38 PST 从 HF tree API 查的。然后测一两个晚上。

**验证深度：**

- 每个模型 5 冷 5 热，列和 minyi-tang 的表一致：spawn-to-first-audio、steady request、idle footprint。
- Drama 3 可以加 SeedTTS 式的 ASR 回测，并发 1 和 16，和她的认领一致。
- Music 3 没有 WER，只测时间和内存。

**第一步：**

1. 先在 #2888 回复认领这个子范围。引用她那句 “Not measured on Macs with 32 GB or more”，@minyi-tang 确认不会撞车。16:43 PST：你让我用 cosekii 代发，已发出 [issuecomment-6566366176](https://github.com/sgl-project/sglang-omni/issues/2888#issuecomment-6566366176)，同时问了 @JiaxinD 和 @minyi-tang。15:35 PST 补充：她已经说整条完成，所以这条回复还要问 @JiaxinD，把 90 秒空闲卸载的默认用到 Drama 3 和 Music 3 之前，还要不要在大内存 Mac 上测。等回复再动手。
2. 拉 #3411 的分支。用 main 上的 `build_runtime.sh` 编出 Drama 3 和 Music 3 的 server，按 docstring 跑：`python sglang_omni_mlx/native/ci/tts_spawn_bench.py --server <BIN> --model-path <DIR> --request <BODY.json> --runs 5`。

**风险：**

- #3411 还没合并，脚本可能还会改。
- 手上有大内存 Mac 的人不少：
  - larryliu0820：M4 Max 64 GB
  - OwenMagzzz：M4 Pro 64 GiB（#3360）
  - lyttttt3333：48 GB 和 64 GB 两台
- 几台机器在并行跑同一份侦察，别的报告也可能选中这一条。我没有、也不能和它们协调，所以第一步一定要先认领。
- 这台机器平时有 18–19 GB 压缩内存。测热启动前要关掉大应用并记录，否则会出现和 16 GB 机器一样的“部分缓存”。

### 3 分的“接近”项（不进 Top 5）

| 条目 | 为什么只有 3 分 |
|---|---|
| 283–286 行：验证 #3043 重构版 | 清单客观，Go 他能读，验收正是他的强项。但 #3043 有 35.9 万行、1661 个文件，10 天没有任何活动，验证结果可能白做 |
| 341 行：CI 里对比原 Voxt 后端 | 15:35 PST 更正：MLX Audio 已经不在 app 里，对比对象要维护者先定，降为 2 分，不再算“接近”项。见第 7 节 |

### 认领人公开求助（已认领，不算 UNCLAIMED）

#3351 C1（对应 221 行，Drama 3 带 offload）的认领人 lgyStoic 在 [10-31T13:40:30Z](https://github.com/sgl-project/sglang-omni/issues/3351#issuecomment-6544114123) 写：

> I'm testing on a 16 GiB Mac. Assistance with a fully resident baseline on a larger Mac would be helpful.

之后 #3351 的评论里没人回应这一句。Drama 3 4-bit 全常驻在这台机器上放得下（28.7 GB ≤ 36 GB），可以和 Top 1 用同一批数据。要先和 lgyStoic 协调。

### 附带发现（不是路线图条目）

- #3173 的 Music 3 性能数字是在 24 GiB 机器上测的，受 swap 影响。这台 48 GB 机器可以给出不受 swap 影响的数字，和验收标准 “faster than all competing open-source MLX frameworks” 有关。
- #2927 写着 Music 3 的 golden “only Apple M5 is recorded here”。M4 Pro 的 golden 要维护者从 Mac CI 的产物里导入。

## 5. STALE CLAIM

**没有。** 截至 2026-11-01，所有认领和对应 PR 在 30 天内都有活动。

下面这些快到 30 天了，按最后更新时间列出：

| 对象 | 最后活动 | 满 30 天的日期 |
|---|---|---|
| kalebguo 的 draft [#2613](https://github.com/sgl-project/sglang-omni/pull/2613)、[#2614](https://github.com/sgl-project/sglang-omni/pull/2614)、[#2615](https://github.com/sgl-project/sglang-omni/pull/2615)（271 行 OmniTyper） | 2026-10-10T00:59Z | 2026-11-09 |
| JiaxinD 的 [#2708](https://github.com/sgl-project/sglang-omni/pull/2708)、[#2713](https://github.com/sgl-project/sglang-omni/pull/2713)（228、229 行） | 2026-10-12T05:54Z | 2026-11-11 |
| hehehai 的 [#3043](https://github.com/sgl-project/sglang-omni/pull/3043)（283–288 行，创建后没有更新） | 2026-10-22T02:57Z | 2026-11-21 |
| AzizMuminov 的 [#2956](https://github.com/sgl-project/sglang-omni/pull/2956)（134 行） | 2026-10-22T16:21Z | 2026-11-21 |
| guozhihao-224 的 [#3105](https://github.com/sgl-project/sglang-omni/pull/3105)（148 行） | 2026-10-23T15:22Z | 2026-11-22 |
| altale 的 [#3179](https://github.com/sgl-project/sglang-omni/pull/3179)（113 行） | 2026-10-25T13:05Z | 2026-11-24 |
| adityavaid 的 [#3154](https://github.com/sgl-project/sglang-omni/pull/3154)（164 行） | 2026-10-25T17:33Z | 2026-11-24 |

14:25:49 PST 登录后复查：这 10 个 PR 的 `updated_at` 和上表一致，都没有新活动。

## 6. 补查结果和仍然做不到的检查

**初稿时的限制**

- 13:11 PST 未登录的 core 配额是 0，13:27:51 PST 才重置。重置后用了 48 次请求，都成功了。
- 初稿因此留了四个缺口：
  - 分页没扫全。
  - 部分条目只按标题搜。
  - 没读 Discussion #2158 和 PR 的行内 review 评论。
  - 生成时间之后的新认领没查。

**14:18–14:27 PST 登录后补查**

账户是 cosekii，只发 REST GET 和 GraphQL 查询，大约用了 190 次 core 请求，没有再被限流。

| 初稿的缺口 | 这次怎么查的 | 结果 |
|---|---|---|
| 分页：open PR 只扫了前 300 个，合并的也只扫了前 300 个 | 用 REST 分页全部拉下来：<br>open PR 724 个（最早一个更新于 08-27）；<br>9 月以来更新过的关闭 PR 1060 个（合并 810，没合并 250）；<br>9 月以来更新过的 issue 188 个 | 没有发现新的对应 PR 或 issue |
| 只按标题搜（269、273、274、275、341、342 行，A2，B/C 表） | 给每条没人认领的条目和 236 行子范围定一组关键词。在上面所有 PR 和 issue 的标题加正文，以及下面所有评论里做全文匹配 | 命中的都是条目本身、已知的相关 PR，或者无关内容。274 行多了一条相关信息，已写进第 3 节 |
| Discussion #2158 | 用 GraphQL 读全部 11 条评论，都没有回复。9 月以来只有这一个 Discussion 有更新 | 全是基准结果，没有认领。最新一条是 minyi-tang [11-01T20:16:27Z](https://github.com/sgl-project/sglang-omni/discussions/2158#discussioncomment-19337745) 在 16 GB M4 上的 TTS 结果 |
| PR 的行内 review 评论 | 全仓库 10-17（#2888 创建）以来更新过的 991 条。<br>另外读了 9 个关键 PR 的 review 总评：#3411、#3338、#2927、#2008、#3173、#3360、#3043、#3412、#3375 | 没有认领。#3411 的 6 个 review 都来自 Copilot 机器人。#3043 仍然没有任何评论和 review |
| 生成时间之后的新认领 | 全仓库 10-17 以来更新过的 691 条对话评论；21:30Z 以后新建或编辑的全部评论有 17 条，逐条看过 | #2888、#3351、#2537、#2941 都没有新评论。21:30Z 以后只有 luojiaxuan 那条 bug 报告，其余是 #3416、#3412、#3189、#3414、#3203 上和这些条目无关的讨论 |

带认领措辞的命中逐条核对过，都不改变结论：

- Zhuofeng-Li 在 #2880、#2882、#2883 写的 “No independent clean-source verification is claimed”（[例](https://github.com/sgl-project/sglang-omni/pull/2880#issuecomment-6236449521)）：不是认领。
- minyi-tang 的 “Results for the two items I claimed above”，以及她 [11-01T06:17:19Z 在 #3140](https://github.com/sgl-project/sglang-omni/pull/3140#issuecomment-6554343586) 写的 “I'm also measuring this PR … on an Apple M4 with 16 GB”：都是已知的 TTS 空闲内存认领，范围仍是 16 GB。
- xiaoyu-xyz [11-01T07:28:37Z](https://github.com/sgl-project/sglang-omni/issues/3351#issuecomment-6554969349) 写 “Claiming **D1**”：这是 #3351 的 D1（offload 前后的可复现结果），不是 236 行。
- wenchen76 [10-25T10:41:49Z](https://github.com/sgl-project/sglang-omni/issues/2941#issuecomment-6396629346) 想接 P2-2、P2-3：这是 #2941 自己的条目，不是 256 行的调研。
- charliechenye 在 [#2938 的 review](https://github.com/sgl-project/sglang-omni/pull/2938#discussion_r4384136959) 里写 “Happy to help investigate”：针对的是那个 PR 里的一个操作，不是认领。
- kzhou003 [10-29T00:54:00Z 在 #3008](https://github.com/sgl-project/sglang-omni/issues/3008#issuecomment-6484817809) 写 “I'd like to take this if it is open”：#3008 是另一个模型的支持，不是 #2888 的条目。

236 行子范围又单独核对了一遍。相关的 7 条评论分别来自：

- JiaxinD：方法说明。
- minyi-tang：16 GB 上的认领、测量和结果。
- xiaoyu-xyz：#3351 的 D1。

32 GB 以上没人认领，Top 1 不变。

**仍然做不到，或者没做的**

- 我读完全部评论的地方只有 #2888、#3351、#2537、#2941、#3338 和 Discussion #2158，其余评论都是关键词匹配。如果有人在别的地方认领，又没用条目里的词，就可能漏掉。
- 10-17 之前的评论，以及 9 月之前最后更新的关闭 PR，都没扫。那时 #2888 还不存在，不会有针对它的认领。#2537 里 10-15 的认领初稿已经读过。
- GitHub 之外的渠道（Slack、微信群等）看不到。
- 14:27 PST 之后新出现的认领。

**评论里提了、但还没进正文的新 bug**（不在表里）：

- cosekii 的 #3397、#3398（minyi-tang 的 #3420、#3419 已经在修）
- luojiaxuan 11-01T18:53:34Z 报的 Sortformer 安装失败（#3161）
- luojiaxuan 11-01T21:32:40Z 新报的两个会议模式实时气泡 bug：停顿后半句话被重复；复读循环，评论里提到 #3413 在 server 端加了 `repetition_penalty`。这条评论就是 `updated_at` 变化的原因
- zoeyliu-ai 的 Raycast 插入问题（#3180、#2878）
- XXXXRT666 贴的 #3378

## 7. 15:35 PST 复查

**数据：** 15:35–15:37 PST 用 cosekii 账户实时读取，只发 REST GET 和 GraphQL 查询，大约 20 次请求，没有被限流。

**变化**

- #2888：
  - `updated_at` 从 21:32:40Z 变为 **22:45:07Z**，评论从 16 条变为 18 条。
  - 正文没改。GraphQL `lastEditedAt` 仍是 2026-10-29T16:19:12Z。正文仍是 355 行，72 条未勾、21 条已勾。
  - 状态 open，没有 assignee，没有 label。
- 两条新评论都来自 minyi-tang，都不是新的认领：
  - [22:37:49Z](https://github.com/sgl-project/sglang-omni/issues/2888#issuecomment-6564931064)：四个旧 issue 的进展。#3094 已合并；#3415、#3413、#3191 等 review；#3412 进行中。她还指出 main 上的 Voxt 已经不链接 mlx-audio-swift：“#2683 removed it on 10-22: `project.pbxproj` has 0 `MLXAudio` references”。Status 那句 `VOXT_ASR_BACKEND=swift`，以及 270 行，她认为可能过时。
  - [22:45:07Z](https://github.com/sgl-project/sglang-omni/issues/2888#issuecomment-6565011674)：给 JiaxinD 的 PR 链接清单。
    - A1、A5、A6 对应 #3394、#3416、#3417。
    - #3397、#3398 对应 #3420、#3419，原文：“could be added to table A as A7 and A8, as @cosekii asked above”。这句里 @ 的是 cosekii [10-31T19:31:37Z 的请求](https://github.com/sgl-project/sglang-omni/issues/2888#issuecomment-6548130306)，没有要他做别的。
    - 236 行她认为 “done”。
    - #2931、#3108、#3064 三行已完成但没勾，和第 3 节的判断一致。
- 时间线：21:29–22:32Z 有 4 次 minyi-tang 的提交引用 #2888；22:56:06Z #3412 交叉引用了它。
- 21:30Z 以后全仓库的新内容逐条看过，没有新的认领：
  - 对话评论 5 条；
  - 行内 review 评论 26 条，分布在 #3416、#3412、#3189；
  - 更新过的 PR 9 个，都不是新建的；
  - 新 issue 0 个。
- #3351、#2537、#2941 没有新评论。Discussion #2158 仍是 11 条，最后更新在 20:16:27Z。lgyStoic 在 #3351 请人在大内存 Mac 上做 Drama 3 全常驻基线，到现在还没人回应。
- 16:43 PST 再看一次：236 行仍未勾，正文没改。新增 [AkazaAkane 00:11:58Z](https://github.com/sgl-project/sglang-omni/issues/2888#issuecomment-6565976906)，提了三条 Voxt 切换模型的建议：把下载、配置、选默认模型放进一个流程；配置警告直接链到要改的设置；每个功能显示当前模型和一个 “Switch model” 入口，不用先删旧模型。这条不是认领，正文里也没有对应的行。之后 cosekii 发了 [00:43:42Z](https://github.com/sgl-project/sglang-omni/issues/2888#issuecomment-6566366176)（236 行的问题）。#2158 在 00:41:27Z 多了这台机器的跑分。

**新出现的可做项**

| 项 | 证据 | 机器结论 | 适合度 |
|---|---|---|---|
| 在 app 里实测 [#3419](https://github.com/sgl-project/sglang-omni/pull/3419)（修 #3398） | 正文：“Not tested: live dictation in the app with automatic language, because the app does not build on the author's local Xcode.”。0 条评论，0 个 review，`mergeable_state` = blocked，最后更新 20:23:16Z。改动：`.languageName` 不再强制主语言，配置多种语言时 Qwen3-ASR 自动识别。正文也写明只有一种语言的用户“still gets the main-language hint” | FIT：今天这台编过 Voxt Dev，修复前的数据也有了。15:28 PST，`["en","zh-hans"]`、跟随主语言开着，“今天天气很好。”仍出成 “Today, the weather is very good.”（`~/voxt-dev-setup/logs/phase5-notes.md`） | 5：他是 #3398 的报告人；改动只有 +33/−4；验收客观，看三句是不是都是中文；补的正是作者写明没测的部分 |
| 在 Apple Silicon 上带真实下载测 [#3420](https://github.com/sgl-project/sglang-omni/pull/3420)（修 #3397） | 正文：“Not covered: a check with a real model download in progress on an Apple Silicon Mac.”。作者只在 GitHub 托管的 macos-26 runner 上让 Model 页空闲了 2 分钟。0 条评论，0 个 review | FIT | 4：他是 #3397 的报告人。要测几十分钟，和阶段 6（R25）重合。要下载一个还没下过的模型，超过 2 GB 要先问 |

两项都要在仓库外另建一个 checkout 编 PR 分支，固定在 `a91bcac` 的复测仓库不动。

**对前几节的更正**

- 341 行：第 3 节原先写“native 和 `VOXT_ASR_BACKEND=swift`（MLX Audio）两条路径这台都能跑”，这不对，已改为 PARTIAL、2 分。
- 270 行：原先写 UNCLAIMED / DEFERRED，已改为 DONE（需维护者确认）。
- 236 行：子范围（32 GB 以上，Drama 3 4-bit 和 Music 3）仍然没人认领。但认领人认为整条已经完成，如果维护者勾掉，这块可能就不需要了。第 4 节的第一步已补上“先问 JiaxinD”。

**这次没做的**

- 只看了 21:30Z 以后的变化，以及上面提到的 PR 和评论全文。更早的内容沿用 14:27 PST 的结论。
- GitHub 之外的渠道看不到。

## 8. 11/2 13:25 PST 复查（cosekii，只读）

- #2888 正文没变（最后编辑仍是 10-29T16:19:12Z）。新评论只有 XXXXRT666 贴的 #3429；我们 00:43Z 在 236 行的提问，JiaxinD 和 minyi-tang 都没回。#3411 open、0 评论。
- 11-02T05:00Z 之后新开的 open issue 只有我们的 #3430、#3431。
- #3351 的 A1–A3、B1–B2、C1、D1–D2 全部有人认领。
- 新的可做项：lgyStoic 的 #3425（Drama 3 磁盘专家缓存，11-02 开）正文写 “Whole-model comparison against a fully resident baseline and streaming performance remain unmeasured”，数据来自 16 GiB Mac。这台机器能跑全常驻（峰值约 28.7 GB）和 `--expert-offload 2gib` 两种，正好补上。和 236 行子范围用同一个 Drama 3 4-bit（23.78 GB，要先问）。要先在 #3425 和 lgyStoic 协调。
- #3429 已在 07:47Z 合并。R1/R2 仍在：`Voxt/CONTRIBUTING.md` 的 build/test 命令没有 `-skipPackagePluginValidation`，也没写装 Metal Toolchain，CI（`voxt-mac-ci.yaml` 137–166 行）两样都做。R5 不再成立：CI 现在用 `DEVELOPER_DIR` 固定 Xcode 26.5。#3412 改了 `AGENTS.md` 和 `VoxtTests/README.md`，但没碰 build 命令和 `CONTRIBUTING.md`。
- 283–286 行（#3043 验证）：#3043 仍停在 10-22，没有评论。342 行（预编译二进制）仍要维护者决定。
