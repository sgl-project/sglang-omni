# Qwen3-TTS r20 vocoder 瓶颈拆解与 Nari 对标(2026-09-05/06)

跟踪 issue:sgl-project/sglang-omni#1754。主机 eval-h100(H200 ×1,GPU 0),
容器 `sglang-omni-jaxan-1`,run 根 `/data/jaxan/runs/20260902-mainline-nari-ab`。
harness:tts-bench 开环 Poisson,`--rps 20 --seed 0 --warmup 30s --duration 60s`,
CustomVoice `Ryan`,seed-tts-eval 英文集,1173 请求/臂。所有时间 PT。

## 结论(截至 2026-09-06 01:50 PT)

1. **r20 的 underrun 不是 CPU 开销,是 vocoder GPU 产能。** 每个 follow-up decode
   group 平均 15.9ms,其中 85% 是等 CUDA 完成事件,CPU 侧合计 2.3ms。之前
   "约 30ms CPU"的归因是错的。
2. **产能被 chunk ramp 吃掉。** ramp `(1,2,4)`→8 让每条流经过窗口 1/3/7/15/23 再到
   稳态 24;64% 的 decode 调用是这些 transient,吃掉 57% 的 vocoder 时间,且
   3 帧窗口(11.6ms)与 24 帧窗口(14.4ms)在 batch=1 下几乎同价(launch-bound)。
3. **关掉 ramp 就 0% underrun(`s-noramp`),代价是 TTFA p50 155→329ms。** 23% 的
   解码负载削减消灭了全部 underrun,说明系统只是刚越过容量阈值。
4. **config 旋钮全部无效**:收集窗口 1→8ms 让 underrun 从 15.5% 恶化到 33.8%
   (攒到的行被形状切碎,batch 不变宽);worker 2→4 更差、6+ OOM。
5. **与 Nari 的真实差距是两半**:低负载固定成本(r1 p95 76.7ms 对 26.4ms,2.9×)
   和随负载退化(r20 p95 225ms 对 38ms;我们涨 2.9×,它涨 1.4×)。underrun
   那一半已证明可解;TTFA 固定成本那一半尚未碰。
6. **结构性解法已有在途 PR**:#1846(arena 状态槽的批量增量解码,T-PR8)+
   #1855(COLD/WARM 增量 CUDA graph,T-PR9)。已 rebase 到当前 main,266 单测
   全过,r20 实测进行中(见下)。

## 参照:Nari r20(同机同 harness,3 seed)

| seed | 成功 | underrun | TTFA p50 | TTFA p95 |
|---|---|---|---|---|
| 0 | 1173/1173 | 0.60% | 26.3ms | 38.1ms |
| 1 | 1162/1162 | 0.77% | 26.1ms | 38.0ms |
| 2 | 1201/1201 | 0.17% | 26.2ms | 38.0ms |

Nari 1 RPS 的 p95 为 26.4ms(#1754 记录),r20 只涨到 38ms:几乎不随负载退化。

## 拆解:follow-up decode group 四段计时(r20,main 91e9c309)

探针把 `_run_followup_batch` 拆为 plan(持锁建计划)/ launch(staging+图重放)/
resolve(等完成事件)/ commit(切波形+消息+IPC),另记 collect 到的行数与形状切分。

| 阶段 | mean | p95 | 占比 |
|---|---|---|---|
| plan | 0.69ms | 1.95ms | 4.3% |
| launch | 1.43ms | 3.16ms | 9.0% |
| resolve(GPU 等待) | 13.59ms | 27.88ms | 85.4% |
| commit | 0.21ms | 0.74ms | 1.3% |

- 90s 窗口内 decode 累计需求 136.3s,2 个 worker 容量 180s → 占用 75.7%,p95 组墙钟 30.9ms。
- 收集窗口平均攒 3.38 行,被 `_group_decode_plans` 按精确形状切成 1.83 组;14 种形状。
- 每行成本随宽度:1 行 13.0ms、2 行 8.2ms、4 行 5.2ms、8 行 4.0ms。
- 按形状:窗口 3 帧 11.6ms、24 帧 14.4ms(batch=1);64% 的组是 ramp transient。

## 负面结果:config 旋钮

| wait / workers | underrun | 攒到行数 | 切成组数 | 行/组 |
|---|---|---|---|---|
| 1ms / 2(默认) | 15.5% | 3.41 | 1.82 | 1.86 |
| 4ms / 2 | 28.6% | 4.89 | 2.47 | 1.96 |
| 8ms / 2 | 33.8% | 5.08 | 2.51 | 2.02 |
| 4ms / 1 | 87.0% | 7.86 | 3.48 | 2.26 |
| 8ms / 1 | 86.7% | 7.88 | 3.45 | 2.28 |

| workers(wait 1ms) | underrun | resolve mean | decode 需求 |
|---|---|---|---|
| 2 | 17.8% | 13.6ms | 136s |
| 3 | 17.7% | 15.0ms | 153s |
| 4 | 19.7% | 15.5ms | 159s |
| 6 / 8 | OOM(图捕获) | | |

## 拓扑与 ramp 前沿(r20)

| 臂 | 改动 | underrun | TTFA p50 | TTFA p95 | decode 需求 |
|---|---|---|---|---|---|
| s-base | 现状 ramp (1,2,4) | 17.31% | 154.7ms | 225.0ms | 137.2s |
| s-stateful | 开 #1757 的串行增量解码 | 26% 成功,TTFA 17.6s | | | 崩 |
| **s-noramp** | 首块 8 帧,无 ramp | **0.00%** | 328.7ms | 458.9ms | 106.2s |
| r-24 | ramp (2,4) | **5.37%** | 176.4ms | 246.8ms | 126.8s |
| r-14 | ramp (1,4) | 87.30% | 170.1ms | 279.1ms | 125.9s |
| r-4 | ramp (4) | 5.29% | 230.0ms | 386.0ms | 115.1s |
| r-2 | ramp (2) | 未测(被我提前杀掉,见决策日志) | | | |

读法:r-14 的 87% 是构造性的(1 帧首块后要等 4 个 AR 步,客户端必然断流),
不是产能问题。r-24 支配 r-4(同等 underrun 下 TTFA 低 54ms)。**在不改结构的
前提下,r-24 是当前默认 (1,2,4) 的更优替代:underrun 18.3%→5.4%,TTFA +20ms。**

## 在途 PR #1846/#1855 的 rebase 与实测

- 两 PR 均 CONFLICTING(基线早于 #1900/#1901/#1912/#1928/#1930)。在 worktree
  `qwen3-tts-pr1855-rebase` 上把 origin/main 合入 #1855(它包含 #1846 内容):
  9 处冲突,7 处两侧并存;`warmup_now` 改为 initial → 各 worker 的 legacy holder
  → WARM → COLD;`_launch_async` 的 legacy 分支改用 main 的每 worker graph holder,
  增量分支用 `stream in self._followup_decode_streams`。
- **线程安全隐患**:#1855 的增量图 runner(`incremental_codec_cuda_graph.py`)
  `decode()` 无锁,写静态缓冲后返回其视图;main 现有 2 个 follow-up worker 并发
  会互踩。首轮实测统一 `followup_worker_count: 1`,以 `l-w1`(legacy 1 worker)
  作对照;若增量路径胜出,再按 `_followup_graph_holders` 的样式做每 worker runner。
- 单测:`test_incremental_codec*.py` + `test_pipeline.py` **266 passed**(修了两处
  仅涉及测试夹具的 main 新属性:`_followup_graph_holders`、`_followup_decode_streams`)。
- 首轮实测(r20,slots 128,**1 worker**):

  | 臂 | 成功 | underrun | TTFA p50 | 备注 |
  |---|---|---|---|---|
  | l-w1(legacy,1 worker) | 100% | 50.3% | 168ms | 1 worker 本身已不够 |
  | i-off(增量 eager) | 1.0% | 62.0% | 481ms | 崩 |
  | i-on(WARM 图) | 17.9% | 79.2% | 292ms | 图全捕获、0 回退、arena 93/128 在用 |
  | i-on-cold(+COLD 1,2 帧图) | 61.7% | 85.8% | 214ms | |
  | i-on-r24(WARM 图 + ramp (2,4)) | 82.6% | 87.6% | 245ms | |

  定位到两处并修掉:`_decode_incremental_group` 硬走 `_followup_decode_stream` 不用
  worker 自己的 stream;WARM 图只捕获 ramp∪稳态 = {2,4,8},到达抖动/尾块给出 1/3/5/6/7
  帧,60s 内 **423 次 miss 回落 eager**(legacy 为此捕获 17..24 全段)。改为捕获
  1..stride 全段(12 张图实测 0.4GB,32 张 ×2 worker 约 2GB,余量 9.5GB)。
- 第二轮(**2 worker,每 worker 一份 WARM runner,全段 WARM 图**):

  | 臂 | 成功 | underrun | First playable p50 | TTFA p50 | WARM replays/miss |
  |---|---|---|---|---|---|
  | w2-i-off(增量 eager) | 1.5% | 61.3% | 200ms | 560ms | 0 / 909+912 eager |
  | w2-i-on | 98.9% | 53.1% | **1611ms** | 1686ms | 3123+3129 / **0 miss** |
  | w2-i-on-r24 | 100% | 45.1% | 1477ms | 1559ms | 2801+2813 / 0 miss |

  WARM 路径已无回退,但 **首帧可播时间 p50 从 legacy 的 82ms 涨到 1.6s**:瓶颈从
  follow-up 转移到了 bootstrap。原因在 `_run_initial_batch`:#1846 把每个 bootstrap 都
  按 B=1 eager 增量解码(注释理由是 Base 的 ref 前缀长度不一、天然 ragged),COLD 图
  又是 opt-in 且只捕 B=1;CustomVoice 没有 ref 前缀,所有 bootstrap 同宽(1 帧,
  抑制时 2 帧),本可以像 WARM 一样成 cohort。60s 内 1526 次 COLD 全走 eager。
- 第三轮(2 worker,只改 config 开 COLD 图):

  | 臂 | 成功 | underrun | First playable p50 / p95 | TTFA p50 | COLD/WARM replays |
  |---|---|---|---|---|---|
  | w2-cold(COLD 图 [1,2]) | 100% | 40.9% | 114 / 580ms | 193ms | 1535 / 2944+2973,0 miss |
  | w2-cold-r24(+ramp (2,4),COLD [2,3]) | 100% | 19.4% | 152 / 492ms | 225ms | 1554 / 2684+2715,0 miss |
  | w2-i-on-r1(r1 低负载) | 100% | 0% | 54 / 90ms | 54ms | COLD 87 eager |

  读法:COLD 图一开,首帧从 1.6s 回到 114ms,证实 1.6s 全是 eager bootstrap 排队。
  但 underrun 仍 41%(legacy 2-worker 同 ramp 17.3%),(2,4) ramp 下 19.4%(legacy
  5.4%)。低负载固定成本:增量路径 First playable p50 54ms 对 legacy 44ms,差 10ms
  (COLD 在 r1 臂未开图)。**增量路径每 cohort 的 GPU 成本按 PR 自测应为 legacy 的
  1/3,但 underrun 没有相应改善——每 cohort 必有别的固定开销(arena gather/scatter、
  每 cohort 一次 `resolve_partial()` 同步、状态行拷贝),需要探针归因。**
- 第四轮(2 worker,bootstrap 按 fresh_frames 成 cohort,COLD runner 用 1/2/4/8 桶):

  | 臂 | underrun | First playable p50 / p95 | TTFA p50 | COLD replays |
  |---|---|---|---|---|
  | c-cold(ramp (1,2,4)) | 46.1% | 104 / 477ms | 181ms | 1155(此前 1535,cohort 生效) |
  | c-cold-r24(ramp (2,4)) | **12.5%** | 122 / 200ms | 193ms | 875 |
  | c-cold-r1 | 0% | 54 / 82ms | 59ms | 87 |

  对照 legacy 2-worker:同 ramp 17.3% / 5.4%。增量路径 0 回退、图全命中,但仍不如 legacy。
- **增量路径四段探针(p-cold,r20,6804 个 cohort,平均 2.32 行)**:

  | 阶段 | mean | p95 | 占比 |
  |---|---|---|---|
  | plan | 0.72ms | 1.95ms | 3.5% |
  | **launch(gather + 发射)** | **13.97ms** | 21.9ms | **67.7%** |
  | resolve(GPU 等待) | 3.23ms | 7.9ms | 15.6% |
  | 其它(slot 记账、日志) | 2.18ms | 4.4ms | 10.5% |
  | commit | 0.54ms | 1.7ms | 2.6% |

  cohort 墙钟 20.6ms,**比 legacy 的 15.9ms 还长**;GPU 侧确实只剩 3.2ms(PR 自测的
  WARM 图 4ms 一致),但省下的时间被 14ms 的 CPU launch 吃光。1 行 cohort 18.5ms,
  8 行 26.6ms(3.3ms/行)——固定开销 14ms 与行数无关。
- **假设"launch 是 arena 的 52 个小张量搬运"被实测否定**:单独计时(真模型,B=1..8)
  gather 0.24 + copy_in 0.5 + slice 0.06 + scatter 0.26 ≈ **1.0ms**。
- **细探针定位到真凶:`arena.scatter` 生产中 8.0ms、单测 0.26ms。** 它用
  `torch.as_tensor(list, device=cuda)` 建索引,这是 pageable 内存的同步 H2D,会阻塞 host
  直到该 stream 上已排队的 kernel(刚发射的整个 graph)跑完——等于每次 gather/scatter
  都隐式 `resolve()` 一次,两个 worker 的 CPU 全被堵住。stage 段 3.8ms 里的
  `frame_positions = torch.tensor(list, device)` 同理。
- **修法**:arena 按线程预分配 pinned host + device 索引/positions 缓冲,`copy_(...,
  non_blocking=True)`;worker 只在 `resolve()` 同步过上一 cohort 后才复用 staging。
  commit 7ef0e8cf。
- **修后(P3/P4,2 worker,COLD cohort,r20)**:underrun 42.8% / 45.4%,First playable
  p50 106 / 98ms——**没有改善**。探针:launch 14.0→7.1ms(scatter 8.0→1.5),时间转移
  到 resolve 3.2→8.9ms,cohort 墙钟 20.6→18.0ms。即 **每个 cohort 真实要等 GPU 约 9-11ms**,
  而测量窗口内 GPU SM 利用率只有 60-69%。PR 自测 WARM 图 B1 4.0ms / B8 9.0ms 是单流空闲
  GPU 下的数字;生产里 2 个 follow-up stream + 1 个 initial stream 交错,每个 cohort 的
  graph 含 8 层 transformer + ~30 个因果卷积 + 52 个状态拷贝,在 B≈2、T=8 下全是微型
  kernel,launch-latency 受限(~10µs/kernel),三条流交错并不能重叠。**增量路径每次调用
  的 kernel 比 legacy 多、每个更小,"少 3 倍算力"没有换成"少 3 倍时间"。**
- 第五轮(wait × ramp,2 worker,COLD cohort,staging 修复,带探针):

  | 臂 | wait | ramp | underrun | First playable p50 | TTFA p50 | rows/cohort | launch | resolve |
  |---|---|---|---|---|---|---|---|---|
  | P4 | 1ms | (1,2,4) | 45.4% | 98ms | 177ms | 2.30 | 7.1ms | 8.9ms |
  | c-w4 | 4ms | (1,2,4) | 23.5% | 88ms | 157ms | 2.39 | 5.0ms | 9.5ms |
  | c-w8 | 8ms | (1,2,4) | 24.0% | 86ms | 158ms | 2.44 | 4.6ms | 9.2ms |
  | c-w4-r24 | 4ms | (2,4) | **5.1%** | 116ms | 177ms | 2.43 | 5.1ms | 10.0ms |
  | c-w8-r24 | 8ms | (2,4) | 5.8% | 113ms | 170ms | 2.46 | 4.7ms | 9.4ms |

  读法:加大收集窗口把 underrun 减半,但 **cohort 宽度几乎没变(2.30→2.46),resolve
  也没变**——收益来自 launch 变便宜(7.1→4.6ms):worker 多睡一会儿,与 initial 流 / Talker
  的争抢就少。这与外审"干扰而非节点数"的判断一致,也否定了我"做宽 cohort"的假设。
  **c-w4-r24(5.1% / 177ms)与 legacy 的 r-24(5.4% / 176ms)统计上打平**:增量路径修到
  现在,最好配置只追平 legacy 的最好配置,没有超越。
- **证伪实验(外审 (b) 的等价物)结果反转了它的"干扰"猜测**:

  | 臂 | 负载 | worker | rows/cohort | launch | **resolve** |
  |---|---|---|---|---|---|
  | p-r1 | r1(GPU 近乎空闲) | 2 | 1.00 | 1.5ms | **12.8ms** |
  | p-w1 | r20 | 1 | 2.90 | 4.6ms | **8.6ms** |
  | P4 | r20 | 2 | 2.30 | 7.1ms | **8.9ms** |

  单 worker 与双 worker 的 resolve 几乎相同 → 两个 follow-up worker 之间**没有互相干扰**;
  低负载 B=1 反而最慢(12.8ms;空闲 GPU 降频对 launch-bound 的 graph 最不利)。慢在
  graph 本身。
- **孤立测量(容器内,CUDA event,背靠背 replay)**:

  | B | WARM 图 replay | 增量 eager | legacy 24 帧图 |
  |---|---|---|---|
  | 1 | 3.4ms | 10.9ms | 9.3ms |
  | 2 | 4.0ms | 11.3ms | 10.7ms |
  | 4 | 6.6ms | 11.3ms | 14.5ms |
  | 8 | 8.6ms | 11.5ms | 24.5ms |

  与 PR 自测一致:孤立时增量图比 legacy 快 2.3-2.9×。**但生产里 resolve 8.6-12.8ms,是
  孤立值的 2-4 倍。** 原因在 kernel 数:

  | 路径(B2) | CUDA kernel 数 | 设备时间合计 |
  |---|---|---|
  | legacy 24 帧 eager | 955 | 6.5ms |
  | legacy + fused SnakeBeta | 694 | 5.3ms |
  | **增量 WARM 图** | **1147** | 3.7ms |
  | 增量 eager + fused SnakeBeta | 887 | 3.2ms |

  增量路径每 cohort **1147 个 kernel 只做 3.7ms 的活,平均每个 3.2µs**——完全是节点过渡/
  依赖延迟受限,不是算力。其中 elementwise 约 550 个(transformer 的 norm/rope/softmax/
  残差/layer-scale + SnakeBeta 的 exp/sin/pow 链)、cudnn nchw↔nhwc 布局转换 62 个
  (0.55ms,纯浪费)、微型 GEMM 57 个、状态 cat/index_select 66 个。孤立背靠背时节点
  流水化得好(3.4ms);生产里 Talker 与 initial 流的 kernel 插进这 1147 个节点之间,每个
  节点都可能等一下,墙钟翻倍。legacy 节点少三成、每个 kernel 大得多,对插队不敏感。
  **"少 3 倍算力"没换成时间,是因为换来了"多 1.6 倍节点"。**
- 第六轮:config 开 `fused_snake_activation`(#1794 的融合 kernel,增量路径自动用上,
  1147→887 节点),wait 4:ramp (1,2,4) underrun 23.5%→**20.1%**、resolve 9.5→8.3ms;
  ramp (2,4) 5.1%→5.4%(持平)。有效但小。
- **节点来源拆分(B2,eager,fused snake,887 个 kernel)**:transformer 8 层单独占
  **545 个**(elementwise 365、GEMM 74、cat/copy 71),只做 1.26ms 的活;卷积栈约 340 个,
  其中 cudnn nchw↔nhwc 布局转换 62 个(0.56ms);关掉 cudnn 走原生卷积会炸到 6647 个,
  不可行;`cudnn.benchmark` 无影响。
- **`torch.compile` 原型(容器内,B2,fresh_frames=8)**:

  | 对象 | kernel 数 | 墙钟 / replay |
  |---|---|---|
  | transformer 单步 eager | 543 | 5.4ms |
  | transformer 单步 compiled(fullgraph,0 break) | 203 | 1.7ms |
  | transformer 单步 compiled 捕进 CUDA graph | 163 | **0.44ms** replay |
  | 整个增量 decode eager | 1148 | 12.5ms |
  | 整个增量 decode compiled | 356-397 | 3.1-3.5ms |
  | **整个增量 decode compiled 捕进 CUDA graph** | **356** | **1.83ms** replay |

  对照:PR 现有 WARM 图 B2 孤立 replay 4.0ms(1147 节点),legacy 24 帧图 10.7ms。
  编译约 15s/静态形状(dynamic=True 首次 41s)。踩到的坑:(a) #1794 的手写 SnakeBeta
  Triton kernel 用了 `torch.cuda.device_of`,Dynamo 不能 trace,编译时须不套它(Inductor
  自己会融合);(b) `state.advance()` 与 `transformer_context_length` 是 Python int,Dynamo
  按值特化、每步重编译,必须移出编译区;(c) 首次调用的 arena 视图与后续 clone 的 stride
  不同,触发一次重编译,runner 用固定静态缓冲即可避免;(d) 原型里 replay 读到被释放的
  状态张量(0.96 rel-L2 的垃圾)是因为没走 runner 的静态 input/output 协议,不是算法问题。
  **数值**:compiled vs eager 逐帧 rel-L2 中位 0.03-0.05、最大 0.04-0.17(随机码),高于
  batch 噪声底(0.01-0.02),要用真实请求的码做接缝/STFT 对照后才算过。
- 下一步(实现中):把增量 decode 拆成"张量-only 内核 + 外层记账",内核可选 `torch.compile`,
  runner 捕获时用编译版;只编译 WARM 稳态形状(fresh_frames=8 × B 1/2/4/8,约 1 分钟
  启动开销),其余形状保持 eager-in-graph;开编译时禁用手写 SnakeBeta 融合。

## 决策日志

| 问题 | 默认答案 | 理由 | 回滚 | 外审 |
|---|---|---|---|---|
| 窗口补齐(alignment)还要不要继续 | 封存为 WIP 分支 `qwen3-tts-align-followup-windows`(commit beacbc47) | 增量解码让每个 WARM chunk 恒为 8 帧,补齐问题整体消失 | 分支仍在,可重开 | 上轮外审 (a)3 已指出补齐可能比现状更慢,与本决定一致 |
| r-2 臂未测 | 不补测 | 我误判其卡死(实际仍在 warmup)提前杀掉;r-24 与 r-4 已夹住它 | 用 `arm-r-2.yaml` 单独重跑 3 分钟 | 不涉及 |
| #1855 首轮实测用 1 worker | 是 | runner 无锁,2 worker 不安全 | 每 worker runner 后重测 2 worker | 待发 |
| 修 #1855 测试夹具而非改实现 | 是 | 失败仅因夹具用 `__new__` 绕过 `__init__` 缺 main 新属性 | 无需 | 不涉及 |

## 工具(未入库,仅 run 树)

`patch_followup_probe3.py`(四段计时探针)、`analyze_followup3.py`、
`analyze_shapes.py`、`bench_decode.py`(窗口×batch 成本面)、
`test_causal3.py`(补齐数值等价对照)、各 `arm-*.yaml` / `main-arm-*.py` /
`run_sweep_*.sh`。
