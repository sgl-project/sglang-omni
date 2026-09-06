# Qwen3-TTS r20 vocoder 瓶颈拆解与 Nari 对标(2026-09-05/06)

跟踪 issue:sgl-project/sglang-omni#1754。主机 eval-h100(85.234.79.62,NVIDIA H100 80GB HBM3 ×1,GPU 0;此前误写为 H200,2026-09-06 10:30 PT 按 nvidia-smi 更正),
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
- **第七轮:编译内核接进 runner**(worktree commits 40e5d153 / 4cdadb61 / 48f032b5):
  `decode` 拆成张量-only `_decode_tensors` + 外层记账;`precompile(B, T)` 只对预编译过的
  形状走编译内核,未见形状走 eager,避免 15s 编译落在请求关键路径;runner 捕获 WARM
  稳态形状(fresh_frames=steady × B 1/2/4/8)前预编译,捕获的即编译版;config
  `incremental_codec_compile`(默认关)。266 + 1 单测过。r20,2 worker,wait 4:

  | 臂 | ramp | underrun | First playable p50 / p95 | TTFA p50 |
  |---|---|---|---|---|
  | k-w4-r24(编译) | (2,4) | **2.73%** | 110 / 155ms | 161ms |
  | c-w4-r24(不编译,同配置) | (2,4) | 5.12% | 116 / 173ms | 177ms |
  | legacy r-24(config 最优) | (2,4) | 5.37% | 106 / 168ms | 176ms |
  | k-w4(编译) | (1,2,4) | 22.6% | 87 / 148ms | 157ms |
  | legacy s-base | (1,2,4) | 17.3% | 82 / 136ms | 155ms |

  **第一次在同等 TTFA 下 underrun 低于 legacy**((2,4) ramp:2.7% 对 5.4%)。(1,2,4)
  ramp 下无收益:那里 bootstrap/COLD 路径主导,COLD 形状未编译。图显存 1.0GB/进程。
- 编译臂的探针(pk-w4-r24,带探针时 underrun 4.1%,探针本身有开销):cohort 14.1ms
  (此前 18.0),resolve 9.4-10.0 → **7.7ms**,launch 5.0ms。孤立 replay 只要 1.8ms,生产
  仍等 7.7ms:vocoder 自己的节点已经便宜,剩下的等待来自与 Talker(默认优先级流、
  batch 48 的大 kernel)交错——vocoder 流优先级已是 -4(最高 -5),但优先级不能抢占
  正在跑的 kernel。下一步验证:r10(Talker 负载减半)下 resolve 是否随之下降。
- **负面:COLD(首块)形状也编译(commit 342f36e5)反而变差**:ramp (2,4) underrun
  2.73%→4.26%、First playable p50 110→136ms;ramp (1,2,4) underrun 22.6%→18.8% 但
  First playable p95 148→433ms。首块在 initial worker 上 B 多为 1-2,编译版对它没有节点
  优势,却多了每形状 15s 的启动编译与运行时 guard 开销。已 revert(5f25e155),COLD 保持
  eager-in-graph。
- **编译内核的 357 个 kernel 从哪来(B2,1.76ms)**:Inductor Triton 融合核 166(0.38ms)、
  GEMM/conv 117(0.80ms;其中 transformer 的 64×8 微型 GEMM 57 个)、**cudnn nchw↔nhwc
  布局转换 62 个(0.56ms,占设备时间 32%、节点 17%)**。下一刀:把因果卷积改成
  channels-last 的 conv2d 在编译区内执行,让 cudnn 直接走 NHWC 不再来回转置;
  再往后是 transformer 的 q/k/v 三 GEMM 合一(每层省 2 节点)。
- channels-last conv2d 原型(编译区内替换因果卷积):布局 kernel 62→19,总 397→367,
  设备时间 1.90→1.66ms,eager 墙钟不变(CPU 受限);数值 max-abs 0.85×rms(bf16 量级)。
  收益有但小,排在后面。
- **r10 探针(编译内核,Talker 负载减半):resolve 8.7ms(p50 7.7),与 r20 的 7.7ms 一样。**
  "剩余 6ms 是 Talker 交错"被否定。候选:GPU 在 cohort 之间空转(launch 2-5ms 的 CPU 段)
  导致 SM 降频、每次重放前再爬升(r1 空闲时 12.8ms、孤立紧循环 1.8ms 与此一致)。下一轮
  探针以 100ms 采样 SM 时钟验证。
- **潜在 bug(真实码对照脚本撞出)**:`_incremental_transformer` 内更新 Python int
  `transformer_context_length`,Dynamo 按值特化,eager 路径每条流每步重编译,8 次后
  `fullgraph=True` 直接硬报错。生产未触发只因编译内核只在 graph 捕获时执行一次。已把该
  记账移到 `decode()` 的非追踪区(c1b4b3e6),测试同步调整。移出后对照脚本仍撞
  recompile 上限(还有别的值守卫),于是改为**编译内核按调用显式 opt-in**(822ed089):
  只有 graph runner 在 warmup/capture 时传 `compiled=True`,所有 eager 解码永不进
  Dynamo;未预编译形状传 `compiled=True` 直接抛错。剩余守卫用 `TORCH_LOGS=recompiles`
  在对照脚本里定位(排队中)。
- **SM 时钟假设被否**:r20 编译臂测量窗内以 100ms 采样 400 次,`clocks.sm` 恒为 1980MHz
  (min=median=max)。至此 Talker 交错、时钟降频都不成立;剩下的 5-6ms 只能靠时间线
  (进程内 torch.profiler 抓 CUDA 流时间线)看 replay 前后到底排了什么。
- **"GPU 保持忙碌"诊断臂(进程内一条最低优先级微型 kernel 循环)**:resolve 7.6 → **4.7ms
  (p50 3.1)**,但该循环抢 GIL/CPU,launch 5 → 10ms、首帧 p50 2s、underrun 79%,不能当
  产品方案。它说明:**GPU 空闲后再被唤醒有毫秒级代价**(时钟采样 100ms 粒度看不到),
  每个 cohort 前 2-5ms 的 CPU 段让流空转,replay 就付这个代价。产品化的对应设计是
  **worker 内双缓冲/提前发射**:先发下一 cohort 再 resolve 上一个,让解码流不空转
  (runner 每 key 需两套静态输入/输出缓冲)。等 profiler 时间线确认空隙位置后实施。
- 真实码对照脚本的重编译来源已定位:裸 `Qwen3TTSIncrementalCodecState()` 在前 9 步里
  conv 历史从无到有、K/V 从 8 宽长到 71 宽,每步一个新形状,加上 inference_mode 内外
  的 dispatch key 差异;runner 用的是 arena 全宽状态,这些守卫在生产里恒定。全量单测 267 过。
- **真实码数值对照(6 段生成音频经编码器取码,48-80 帧,逐帧相对 L2 中位 / p95 / 最大)**:

  | 对照 | 中位 | p95 | 最大 |
  |---|---|---|---|
  | eager 增量 vs legacy(PR 自带路径) | 0.0075-0.0103 | 0.020-0.030 | 0.03-0.87 |
  | compiled 增量 vs legacy | 0.015-0.022 | 0.037-0.056 | 0.07-0.89 |
  | compiled vs eager 增量 | 0.015-0.022 | 0.034-0.050 | 0.07-0.87 |

  最大值 0.7-0.9 出现在 eager-vs-legacy 也有的同一帧(尾部近静音帧,相对量纲失效),与
  编译无关。接缝一阶差分跳变 legacy/compiled 逐段一致(19.6/19.3、3.9/3.8、35.2/35.4…),
  **无接缝伪影**。编译引入的噪声约为 batch-size 噪声(中位 0.009-0.010)的 1.5-2 倍,
  仍在 bf16 量级。log-mel 距离未做。
- **进程内 profiler 时间线(401ms 窗,48742 个 GPU 事件)**:4 条流——流 7 忙 28%
  (30665 事件,含 flash attention / `_seeded_top_k_top_p_sample` / flashinfer RMSNorm:
  **是 Talker 的 kernel,Talker 与 vocoder 在同一进程同一 GPU 上下文**),流 23 / 19
  (两个 follow-up worker)各忙 10% / 7%,流 15(initial)2%。vocoder 流每 cohort 只执行
  约 2ms,却等 7.5ms:它的 356 个串行微节点每个都可能排在 Talker 占满 SM 的 kernel 后面
  (优先级只管排队不管抢占),r10 也一样是因为 Talker 单个 kernel 的时长与批大小无关。
  **所以这是同进程内的 Talker 交错,r10 实验不能证伪它。**
- 两条产品化方向同时实验:(a) **提前发射**(worker 内保持一个 cohort 在飞,先发下一个再
  resolve 上一个;每线程两个 pinned 槽;commit ed5af36f)——填掉 CPU 段的流空转;
  (b) **vocoder 独立进程 + CUDA MPS**(仓库已有 `--mps` 运行时与 `stages.vocoder.process`)
  ——让 vocoder 的微 kernel 与 Talker 的 kernel 在不同上下文里并发,而不是排队。MPS 首跑
  因控制 socket 路径超过 AF_UNIX 107 字节失败(state root 取自 TMPDIR),改短后重排。
- **第八轮结果(r20,2 worker,wait 4,ramp (2,4),编译内核;单 seed 噪声约 ±1%)**:

  | 臂 | underrun | First playable p50 / p95 / p99 | TTFA p50 |
  |---|---|---|---|
  | k(基线:同进程,无提前发射) | 2.73% | 110 / 155 / 220ms | 161ms |
  | la(+提前发射) | 3.67% | 108 / 149 / 185ms | 148ms |
  | n(vocoder 独立进程,无 MPS,+提前发射) | 4.77% | 99 / 145 / 204ms | 165ms |
  | m(独立进程 + 原生 MPS,+提前发射) | **2.39%** | **93** / 130 / **542**ms | 148ms |
  | la-w4(提前发射,ramp (1,2,4)) | **17.1%**(此前 22.6%) | 85 / 116 / 187ms | 142ms |

  提前发射在 (2,4) 下中性、在默认 ramp (1,2,4) 下明显有效(22.6→17.1%);独立进程把
  首帧 p50 提前 10-17ms;原生 MPS 的 underrun 最低但 p99 有 542ms 的尾巴,且服务收尾时
  运行时报 "MPS health check failed … daemon identity query failed"——不作默认。
- **第二轮外审**(`docs/reviews/2026-09-06-qwen3-tts-incremental-codec-ship-decision.md`):
  根因是 GPU 调度干扰,优先级不能抢占;提前发射打空洞、分区 MPS 打干扰、原生 MPS 多为
  仪式、融合是最后的清理;建议**现在就把增量路径发成默认(legacy 留回滚开关)→ 提前发射
  → 用 replay 前后的 CUDA 事件证明 Talker 因果 → SM 静态分区 → 最后融合**。与实测一致,
  采纳。

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

## 发布状态(2026-09-06 04:45 PT)

- **默认翻转已在分支上落地(61611615 / c36f6acb)**:`create_vocoder_executor` 的默认改为
  `enable_stateful_codec_decoder=True`、`followup_batch_wait_ms=4`;图与编译开关默认跟随
  stateful(显式指定才独立生效),COLD 形状默认按首块推导(首块帧数,抑制时 +1);
  scheduler 类的默认不变(单测用假解码器)。回滚只需 `enable_stateful_codec_decoder: false`。
  cookbook 增加"Codec decoding defaults"。ramp 默认仍为 (1,2,4),(2,4) 作为高并发剖面写在
  文档里。零覆盖默认配置的 r20 验收与 (2,4) 剖面排队中。
- CUDA 事件判别实验第一次跑坏了:`IncrementalCodecGraphResult` 是冻结 dataclass,给它挂
  属性抛错,每个 cohort 都回落 legacy(891 次),underrun 74%——探针自身的 bug,与产品代码
  无关;改为把事件挂在 handle 的 keepalives 上重跑(排队)。
- 分支 `qwen3-tts-pr1855-rebase`(origin)共 20 个 commit:#1855 与 main 合并、6 个修复、
  张量-only 内核 + 编译、每 worker runner、提前发射、默认翻转、测试。下一步:把这些整理成
  面向 #1846/#1855 的评审说明与(或)独立 PR。

## 默认配置验收与噪声带(2026-09-06 05:00 PT)

- 零覆盖默认(stateful + 图 + 编译 + wait 4,ramp (1,2,4),COLD 自动推导为 (1,2)):
  underrun **20.1%**,First playable p50 89 / p95 123ms,TTFA p50 140ms。legacy 默认单次
  17.3% / 82 / 155ms。**默认 ramp 下新旧路径打平**(TTFA 稍好,underrun 稍差,均在噪声内)。
- 仅覆盖 ramp (2,4)(COLD 自动推导为 (2,3)):underrun **5.12%**,p50 109 / p95 151ms。
- 同一配置((2,4) ramp,2 worker,wait 4,编译)迄今 6 次单 seed 结果:2.39、2.73、3.67、
  3.92、4.01/4.09(带探针)、5.12 → **均值约 3.7%,极差 2.7 个百分点**。之前汇报的
  "2.7% 对 5.4%"取的是这条带的下沿,不诚实;legacy (2,4) 只有一次 5.37%。正在跑
  3 seed × {新, 旧} × {默认 ramp, (2,4)} 的正式对比,以均值 ± 极差下结论。
- 提前发射版的 CUDA 事件探针(host 侧部分):resolve 里的 host 等待均值 2.9ms(p50 2.0),
  但从发射到完成 20.5ms(p50 17、p95 46)——两个 cohort 在飞时串行排队,每 cohort 的 GPU
  服务时间仍约 8-10ms。replay 事件本身的 GPU 时长(判定 Talker 是插在节点之间还是挡在
  第一个节点之前)因探针读 keepalives 的时机错了没记到,已修,排在多 seed 之后重跑。

## 三 seed 正式对比(2026-09-06 05:40 PT;同一服务进程跑 seed 0/1/2,r20)

| 臂 | underrun(s0 / s1 / s2 → 均值) | First playable p50 | First playable p95 | TTFA p50 | TTFA p99 |
|---|---|---|---|---|---|
| **新默认 + ramp (2,4)** | 4.01 / 2.75 / 4.16 → **3.6%** | 105-109ms | **145-151ms** | 164-167ms | 240-285ms |
| legacy + ramp (2,4) | 6.22 / 6.71 / 9.91 → **7.6%** | 109-114ms | 247-509ms | 180-188ms | 545-939ms |
| **新默认(ramp 1,2,4)** | 22.8 / 19.7 / 20.3 → **20.9%** | 86-87ms | **117-134ms** | 147-148ms | 210-305ms |
| legacy 默认 | 19.0 / 17.6 / 21.2 → **19.3%** | 83-99ms | 159-767ms | 157-175ms | 590-980ms |

读法:(2,4) ramp 下新路径 underrun 减半(3.6% 对 7.6%),且尾部大幅收紧(首帧 p95 150ms
对 250-510ms,TTFA p99 260ms 对 550-940ms);默认 ramp (1,2,4) 下 underrun 打平
(20.9% 对 19.3%,噪声内),但尾部同样大幅收紧(首帧 p95 117-134ms 对 159-767ms)。
**新路径在两种 ramp 下都不差于 legacy,并在尾延迟上显著更好**——这是翻默认的依据。
Nari 参照仍是 0.6% / TTFA p50 26ms,差距在首帧固定成本与 (1,2,4) ramp 下的产能。

- **CUDA 事件探针(v3,提前发射版)**:replay 本身 start→end 均值 3.76ms(p50 3.78,
  p95 6.2;孤立 1.8ms),从发射到完成 21ms(p50 17;两 cohort 在飞,≈ 2 个服务时间),
  resolve 里 host 等待 2.9ms。按外审的判定表:replay 内部被 Talker 拉长约 2×(节点间
  交错),但每 cohort 约 8.5ms 的服务时间里 replay 只占 3.8ms,**另外 ~4.7ms 是 graph 之外
  的 GPU 工作**(52 次状态拷入、52 次 scatter、gather、切片、D2H)在交错下的耗时。
  下一刀:让 graph 直接消费 arena 状态(静态 slot 索引张量,index_select/index_copy 进
  graph),把这些散 kernel 收进一次 replay。

## 第九轮:graph 直接消费 arena(2026-09-06 09:40 PT)

- 改动(5bd83547 / 2aa92f9b):arena 多一行 scratch;`gather_by_index` / `scatter_by_index`
  接受设备端索引张量;runner 绑定 arena 后,捕获区内是 `gather(static_index) → decode →
  scatter(static_index)`,每 key 不再分配静态输入/输出状态;`decode_slots(codes, slots)`
  每 cohort 只写 slot 索引(pinned 非阻塞)与码、replay,未用的桶行指向 scratch。调度器的
  launch 不再预先 gather,positions 以 arena 行为准(host 镜像只用于规划);graph miss 才
  `gathered()` + eager + scatter。267 单测过。三 seed 验收(默认 ramp 与 (2,4))进行中,
  对照第八轮:3.6% / 20.9%。
- **第九轮三 seed 验收(2026-09-06 10:37 PT,eval-h100 H100 80GB,r20,零覆盖默认配置)**:

  | 臂 | underrun s0 / s1 / s2 → 均值 | First playable p50 / p95 | TTFA p50 / p99 | 回退 | 图显存 |
  |---|---|---|---|---|---|
  | **新默认 + ramp (2,4)** | 0.94 / 0.52 / 0.17 → **0.54%** | 95-96 / 128-133ms | 103-111 / 202-280ms | 0 | 0.50GB |
  | 第八轮同配置(arena 在 graph 外) | 4.0 / 2.8 / 4.2 → 3.6% | 105-109 / 145-151ms | 164-167 / 240-285ms | 0 | 1.07GB |
  | **新默认 ramp (1,2,4)** | 4.35 / 3.96 / 4.16 → **4.2%** | 76-78 / 105-106ms | 86-89 / 168-201ms | 0 | 0.50GB |
  | 第八轮同配置 | 22.8 / 19.7 / 20.3 → 20.9% | 86-87 / 117-134ms | 147-148 / 210-305ms | 0 | 1.07GB |
  | legacy 默认(第八轮) | 19.0 / 17.6 / 21.2 → 19.3% | 83-99 / 159-767ms | 157-175 / 590-980ms | — | — |
  | Nari 参照 | 0.60 / 0.77 / 0.17 → 0.5% | 26 / 34ms | 26 / 56ms | — | — |

  读法:把 gather/scatter 收进 graph 后,(2,4) ramp 的 underrun 到了 Nari 的水平
  (0.54% 对 0.5%),TTFA p50 从 165ms 降到 105ms;默认 ramp 从 20.9% 降到 4.2%,首帧
  p50 77ms、p95 105ms(legacy 83-99 / 159-767ms)。graph 外的 ~4.7ms 状态搬运确实是
  第八轮剩余等待的主体,与事件探针的归因一致。剩余差距只在首帧固定成本:TTFA p50
  86-105ms 对 Nari 26ms。
- **第九轮首批数字需重验(2026-09-06 10:50 PT)**:r1 验收只有 64.8% 请求完成(19 条流挂到
  客户端超时),r20 三 seed 的完成率也是 99.8-99.9% 而非此前各轮的 100%——上表 underrun 只按
  完成的流统计,略偏乐观。根因:提前发射让一个线程同时有两个 cohort 在飞,而 arena 的
  pinned 索引 staging 每线程只有一份,后一个 cohort 的写入会覆盖前一个尚未执行的 H2D 拷贝,
  前一个 cohort 于是解码/推进了错误的 slot,受害流永远等不到自己的下一段。修法:staging
  改为每线程 4 份的环(da0fe4ed),268 单测过。r1、三 seed r20、首帧路径事件剖面按序重跑。
- **主机与收尾更正(2026-09-06 11:05 PT)**:eval-h100 实为 8×H100 80GB,本任务容器只暴露 GPU 0
  (`device=0`);GPU 6/7 是 omni-ci runner 在用。停 server 时 `pkill -f main-arm-` 只杀父进程,
  multiprocessing spawn 出来的 stage worker(`spawn_main`,持有 ~72GB)会成为孤儿继续占卡,后续
  server 全部 OOM。各 launcher 的收尾改为连 `spawn_main|compile_worker|resource_tracker` 一起杀。
- **r1 真根因:collect 锁 + 提前发射的互相等待(2026-09-06 11:30 PT)**。环形 staging 修复后 r1 仍
  35% 失败,于是做 r1 二分(每臂 47 请求,带请求事件记录器):关掉增量解码 47/47 全成功、0 underrun;
  增量路径不论 CUDA graph 开关都失败 28-49%。客户端逐请求记录显示失败几乎都不是挂死,而是
  **每条请求恰好一次 0.5-3.9 s 的停顿,起点正好在首块音频播完之时**——即第二块(bootstrap 段)
  迟迟不到。进程内栈采样器(20ms 采样,只记经过 streaming_vocoder 的线程)给出答案:follow-up
  worker 一共 22 s 停在 `with self._followup_collect_lock:` 这一行。机制:两个 follow-up worker
  共用一把 collect 锁保证批收集原子;空闲的 worker 甲在锁内无限期 `queue.get`;worker 乙刚把新流的
  第二段提前发射出去、要回来 resolve+commit,却先得拿锁——而让队列再有东西的唯一来源恰恰是乙的
  这次 commit(它会把该流重新入队)。于是第二块要等到无关流量(下一批 talker 帧或下一条请求)
  唤醒甲才能发出。r20 流量密集所以只剩 0.1-0.2% 挂流,r1 则每条请求都中招。提前发射之前 commit
  是同步做完再回到 collect,所以从未暴露。修法(74914eb0):持有 in-flight 段的 worker 对锁只等
  `followup_batch_wait_ms`(4ms),等不到就先 drain(keep=0) 再回来;回归单测
  `test_qwen3_tts_vocoder_in_flight_worker_commits_while_sibling_holds_collect_lock`。
  教训:**低负载(r1)是并发 bug 最灵敏的探针**,任何改动执行顺序的优化都要跑 r1 完整率验收。
- **修复后 r1 验收(2026-09-06 11:37 PT,默认树 = 增量 + WARM/COLD 图 + 编译内核 + 提前发射 + 锁等待有界)**:
  47/47 完成、underrun 0、first playable p50 45.3ms / p95 73ms(legacy 同批 44.5 / 73.5ms)、E2E p99 733ms。
  低负载下增量路径与 legacy 打平且无停顿。三 seed r20 与首帧事件剖面在同一棵树上重跑中。
- 同一棵树关掉 CUDA graph 与编译内核(eager 增量)的 r1 对照:47/47、underrun 0、first playable
  p50 54.0ms / p95 94ms、E2E p99 907ms——同样无停顿,graph 在首帧上省约 9ms(p50)、尾部省 20-30ms。
