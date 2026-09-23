# Qwen3-TTS r20 vocoder 瓶颈拆解与 参照引擎 对标(2026-09-05/06)

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
5. **与 参照引擎 的真实差距是两半**:低负载固定成本(r1 p95 76.7ms 对 26.4ms,2.9×)
   和随负载退化(r20 p95 225ms 对 38ms;我们涨 2.9×,它涨 1.4×)。underrun
   那一半已证明可解;TTFA 固定成本那一半尚未碰。
6. **结构性解法已有在途 PR**:#1846(arena 状态槽的批量增量解码,T-PR8)+
   #1855(COLD/WARM 增量 CUDA graph,T-PR9)。已 rebase 到当前 main,266 单测
   全过,r20 实测进行中(见下)。

## 参照:参照引擎 r20(同机同 harness,3 seed)

| seed | 成功 | underrun | TTFA p50 | TTFA p95 |
|---|---|---|---|---|
| 0 | 1173/1173 | 0.60% | 26.3ms | 38.1ms |
| 1 | 1162/1162 | 0.77% | 26.1ms | 38.0ms |
| 2 | 1201/1201 | 0.17% | 26.2ms | 38.0ms |

参照引擎 1 RPS 的 p95 为 26.4ms(#1754 记录),r20 只涨到 38ms:几乎不随负载退化。

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
参照引擎 参照仍是 0.6% / TTFA p50 26ms,差距在首帧固定成本与 (1,2,4) ramp 下的产能。

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
  | 参照引擎 | 0.60 / 0.77 / 0.17 → 0.5% | 26 / 34ms | 26 / 56ms | — | — |

  读法:把 gather/scatter 收进 graph 后,(2,4) ramp 的 underrun 到了 参照引擎 的水平
  (0.54% 对 0.5%),TTFA p50 从 165ms 降到 105ms;默认 ramp 从 20.9% 降到 4.2%,首帧
  p50 77ms、p95 105ms(legacy 83-99 / 159-767ms)。graph 外的 ~4.7ms 状态搬运确实是
  第八轮剩余等待的主体,与事件探针的归因一致。剩余差距只在首帧固定成本:TTFA p50
  86-105ms 对 参照引擎 26ms。
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

### 第九轮重验(锁等待有界 + 环形 staging,2026-09-06 11:55 PT,全部 100% 完成)

| arm | seed 0 | seed 1 | seed 2 | 均值 | first playable p50 / p95(三 seed 范围) |
|---|---|---|---|---|---|
| 默认 ramp (1,2,4) | 2.98% | 2.41% | 3.08% | **2.8%** | 76-79 / 106-114 ms |
| ramp (2,4) | 0.68% | 0.09% | 0.00% | **0.26%** | 95-98 / 127-137 ms |

对照此前(有挂流、完成率 99.8-99.9%)的 4.2% / 0.54%:修掉停顿后 underrun 进一步下降,
默认 ramp 从 legacy 的 20.9% 降到 2.8%(7.5×),ramp (2,4) 0.26% 已低于 参照引擎 的 0.5%。
每 seed n≈1160-1200 请求;单 seed 噪声约 ±1 个百分点(默认 ramp)/ ±0.4(ramp 2,4)。

### 首帧路径分解(请求事件记录器,同一棵树)

r1(47 请求,first playable p50 47.5ms)与 r20(795 请求,p50 79ms)的服务端阶段均值:

| 阶段 | r1 avg | r1 p95 | r20 avg | r20 p95 |
|---|---|---|---|---|
| preprocessing(input→complete) | 7.0 | 16.6 | 9.0 | 16.8 |
| tts_engine request_build | 0.2 | 0.3 | 0.3 | 0.5 |
| build_end→queue_enter(准入等当前 decode step) | 3.4 | 8.1 | **13.8** | 19.6 |
| queue_enter→prefill_start | 1.0 | 1.1 | 1.1 | 2.0 |
| prefill_start→first_emit(prefill + 首帧 code predictor) | **16.5** | 14.9 | 14.3 | 17.7 |
| first_emit→first chunk sent | 0.2 | — | 0.2 | — |
| vocoder 整个请求生命周期(非首块) | 10.6 | 18.1 | 19.8 | 35.2 |
| hop vocoder→coordinator(每块) | 0.39 | 0.47 | 1.03 | 0.34 |

读法:r1 的 47.5ms 里服务端可归因约 28ms(7.0 + 3.4 + 1.0 + 16.5 + 首块 vocoder ≈ 2-3 + hop),
其余 ~15-20ms 在 HTTP/coordinator/客户端一侧,需要单独量。r20 比 r1 多出的 ~30ms 主要是
准入等待(+10ms,新请求要等正在跑的 batch-48 decode step 结束)、preprocessing(+2ms)和
vocoder 首块排队。**prefill→首帧 16ms 是最大单项**:1.7B talker 短 prompt prefill 不该要 16ms,
里面含首帧的 code predictor 自回归(16 个量化器)——下一步分别计时。

### 第十轮:vocoder 与 Talker 的默认流耦合(2026-09-06 12:55 PT)

首块探针把 r1 下 vocoder 首块的 25.8ms 拆开后,两项异常:`arena.acquire()` 的 52 个清零 kernel
在 CPU 侧要 5.4ms(孤立 0.19ms),`resolve` 等 GPU 8.2ms(COLD 图孤立 replay 只要 3.2ms)。
先证伪了 GIL 切换间隔(改 0.2ms 无变化);把清零挪到解码流后 plan 降到 0.4ms 但 resolve 升到
13.4ms,总量不变——说明整段都在等同一件事。读代码找到根因:**Talker 的 omni model runner 不建流,
整个 AR 步跑在 legacy 默认流上;Talker 把 codes 以 CUDA tensor 直接交给同进程的 vocoder,vocoder
在自己线程的默认流上 `torch.cat` 这些 codes,再让解码流 `wait_stream(默认流)`**。于是每次解码都
排在 Talker 当时已入队的整步(含 launch-ahead 的下一步)之后:r1 约 5-10ms,r20 batch-48 一步
12ms+——这正是第六轮外审归因为"SM 干扰"的那 5.5ms(2ms 计算等 7.5ms),实为流顺序,不是算力。
跳过 `wait_stream` 直接崩(device assert:codes 未就绪就被编译内核 embedding),证明依赖真实存在。

修法(5ad19d18,按"消费者等生产者的事件,不等生产者的整条流"设计):
- Talker 每步在 `post_process_outputs` 的 codes 快照后记录一个 CUDA event,随 stream chunk 的
  metadata(`codes_ready_event`)交给 vocoder;跨进程发送时由 transport 剥掉(那条路径本就自行保证就绪);
- vocoder 的每个解码 worker 线程 `torch.cuda.set_stream(自己的解码流)`,规划、槽位清零、cat、发射
  全部落在自己的流上;`_build_*_plan` 在 cat 之前 `wait_event(state.codes_ready)`(同一 talker 流
  按序生产,最新 chunk 的 event 覆盖更早的);`_launch_async` 不再 `wait_stream(默认流)`;
  scheduler 线程的同步 flush 路径包在解码流上下文里。

r1 验收(47 请求,100% 完成,0 underrun):**first playable p50 35.8ms(此前 45.5)、min 32.7ms**;
初始解码 plan 0.38 / launch 0.28 / resolve 4.3 / commit 0.11 ms,合计 4.6ms(此前 17-18ms)。
单测 507 过(3 个 test_compat 文档测试在该 rsync 树上失败,与本改动无关,server-30ms 树上通过)。
三 seed r20 与事件剖面在跑。

r20 三 seed(事件解耦树 5ad19d18 + record_stream 5456cc54,全部 100% 完成,2026-09-06 13:10 PT):

| arm | seed 0 | seed 1 | seed 2 | 均值 | 第九轮均值 | first playable p50 / p95 |
|---|---|---|---|---|---|---|
| 默认 ramp (1,2,4) | 1.11% | 0.60% | 1.83% | **1.18%** | 2.8% | 64-66 / 88-91 ms(此前 76-79 / 106-114) |
| ramp (2,4) | 0.26% | 0.00% | 0.00% | **0.09%** | 0.26% | ~81 / ~109 ms(此前 95-98 / 127-137) |

默认 ramp 的 underrun 再减半以上、首帧 p50 少 12ms、p95 少 20ms;legacy 起点是 20.9% / 159-767ms。
外审(`docs/reviews/2026-09-06-qwen3-tts-stream-decoupling.md`)指出的生命周期漏洞已用
record_stream 堵上;它另指出 PyTorch 以 `cudaGraphInstantiateFlagUseNodePriority` 实例化图,
kernel 优先级取自**捕获流**而非 replay 流——我们的图一直在默认优先级流上捕获,所谓高优先级
vocoder 流对图内 kernel 从未生效。第十一轮:在解码流同优先级的流上捕获,r20 A/B。

事件解耦树的请求事件剖面(2026-09-06 13:15 PT):r1 first playable p50 **36.2ms**(min 32.3),
r20 p50 **68.7ms**(第九轮 79.1)。r20 各阶段均值:preprocessing 8.9ms、准入等待(build_end→queue_enter)
13.9ms、prefill→首帧 13.9ms、vocoder 整个请求 14.3ms(第九轮 19.8);r1 的 vocoder 生命周期 8.6ms
(第九轮 10.6)。剩下的首帧预算里最大的三项都在 Talker 侧:准入等一整个 decode step、prefill 本身、
以及 preprocessing——下一步分别拆。

### 第十一轮(进行中,2026-09-06 13:30 PT):同病相怜的 preprocessing,以及图捕获优先级

- **preprocessing 也在默认流上**:CustomVoice 的 preprocessing = 文本 tokenize(CPU)+ 说话人/特殊
  token 的 embedding 查表(GPU 小 kernel + `torch.tensor(..., device)` 同步 H2D),8 个线程都在
  各自线程的默认流上做——同一条 legacy 默认流,于是也排在 Talker 当时那一步之后:r1 均值 2.8ms
  但 p95 16.6ms,r20 均值 8.9ms、p95 17ms,正好是一步的长度。改法(0578cb81)与 vocoder 同构:
  同进程的 preprocessing context 持有自己的流,`preprocess_qwen3_tts_payload` 在该流上做准备并
  记 ready event 挂到 `Qwen3TTSPreparedRequest`;AR 侧 request builder 拿到后在当前流(默认流)
  `wait_event` + `record_stream`。跨进程(standalone)路径不变。预期 r20 preprocessing 8.9 → ~1-2ms。
- **图捕获优先级**(cf6ef922):按外审提示在解码流同优先级的流上捕获。第一次 A/B 只跑完 ramp(2,4)
  臂就被我排错的链撞死(0.51/0.17/0.58%,对事件树的 0.26/0/0%,偏差在噪声内但方向不利);
  两臂三 seed 重跑中,不利就撤。
- **准入等待 13.9ms(r20)的机制**:scheduler 主循环每轮 `process_input_requests` 一次,而每轮都在
  `_resolve_and_process` 里等上一步的 GPU 事件,所以新请求平均要等一整个 decode step 才进等待
  队列;prefill 又要先 flush 在飞的 decode 步。`prefill_coalesce_requests` 默认 0(未开合并等待)。
  要压这 14ms 得让 prefill 混进 decode 步(SGLang `enable_mixed_chunk` + chunked prefill)或
  在 resolve 等待期间做准入——属于 scheduler 结构改动,先用 prefill 探针把 prefill 自身的
  13.9ms 拆开再定。

第十一轮结果(2026-09-06 13:50 PT,树 = 事件解耦 + record_stream + 图捕获优先级 + preprocessing 独立流;
全部 100% 完成):

| arm | underrun 三 seed | 均值 | 上一轮 | first playable p50 / p95 | 上一轮 |
|---|---|---|---|---|---|
| 默认 ramp (1,2,4) | 1.11 / 0.69 / 1.08% | 0.96% | 1.18% | 55-57 / 78-81 ms | 64-66 / 88-91 |
| ramp (2,4) | 0.51 / 0.09 / 0.00% | 0.20% | 0.09% | 70-73 / 98-103 ms | ~81 / ~109 |

- **preprocessing 独立流(0578cb81)**:事件剖面 r1 preprocessing 2.8 → 2.1ms(p95 16.6 → 2.5),
  r20 8.9 → 3.2ms(p95 17 → 6.3);r20 首帧 p50 68.7 → 60.2ms(剖面口径),三 seed 口径 65 → 56ms。
  underrun 持平(噪声内)。
- **图捕获优先级(cf6ef922)**:单独跑的 ramp(2,4) 臂 0.51/0.17/0.58%,合并跑 0.51/0.09/0.00%,对
  事件树 0.26/0/0%——无可测收益,方向略不利但在噪声内。保留(语义正确:图 kernel 优先级随捕获流),
  不宣称收益。
- **prefill 探针(r1,55 次 prefill)**:`execute()` 12.0ms = build 0.2 + prepare_forward 7.0
  (embeds 0.2 + **forward 6.2** + sample 0.5)+ code predictor 发射 0.1 + **finalize 4.6**(等 GPU:
  16 个量化器的 code predictor 图 replay + D2H)。prefill 只有 22 个 token 却要 6.2ms:当前
  `cuda_graph_backend_prefill=breakable`(注意力在图外、逐层分段)——对小 token 数是发射开销主导;
  decode 步的 forward 约 2ms。候选:`full` 后端(SGLang 另有 tc_piecewise),冒烟已排队。
  code predictor 4.4ms/步是 16 个串行子步,已在一个图里;要再降要动子解码器的 kernel。
- **r1 首帧账**(p50 35.6-35.9ms):preprocessing 2.1 + 准入 3.3 + prefill 12 + 首帧 code 到 vocoder
  发出 ~5 + 第二帧等待(bootstrap 静音抑制要 2 帧,一步 ≈ forward 2 + predictor 4.4 ≈ 7)+ hop/HTTP。

### 分支现状(2026-09-06 14:05 PT)

`qwen3-tts-pr1855-rebase`(origin,HEAD 742ca60f)自 21a16bc4 起 26 个 commit。第十/十一轮新增:
da0fe4ed 环形 staging;74914eb0 + a6402532 collect 锁有界等待与回归测试;5ad19d18 解码 worker 自有流 +
Talker chunk 事件;5456cc54 record_stream;cf6ef922 图在解码流优先级上捕获;0578cb81 preprocessing
独立流 + ready event;742ca60f black。#1855 本身是 leihehehe 的 draft(fork 分支),这些 SHA 只在
评论里被引用、不在 PR diff 内——是否以本分支另开 PR 由 luojiaxuan 定(此前 JiaxinD 直接推到对方
分支后道歉,不宜重演)。
- **`initial_batch_wait_ms` 0 对 2(默认)A/B**(同树,三 seed,100% 完成):0 → 1.71 / 1.46 / 1.83%
  (均值 1.67%),2 → 0.94 / 0.60 / 0.58%(0.71%);首帧 p50 两边都 55-57ms。r1 下那 2ms 是纯延迟,
  但 r20 下把同时到达的 bootstrap 段合成一个 cohort 值回来;**保留默认 2ms**。默认臂在本树上六个
  seed 的 underrun:1.11 / 0.69 / 1.08 / 0.94 / 0.60 / 0.58%(均值 0.83%)。
- **Talker 混合 chunked prefill(`enable_mixed_chunk: true`,`chunked_prefill_size: 2048`)r1 冒烟**:
  能跑,47/47、0 underrun、首帧 p50 36.1ms(与默认持平);但 E2E p95 1.34s / p99 3.75s(默认 0.73 /
  0.92s)——低负载下尾部明显变差,说明混批让部分请求的 decode 步被拉长。它瞄准的是 r20 那 ~14ms
  准入等待,三 seed r20 已排队;若 r20 首帧无收益或尾部照样变差,此路作废。
- **prefill 图 `full` 后端**:server 启动即被 `generation_batch_policy` 拒绝("invalid generation batch
  policy: prefill CUDA graph ...")——Qwen3-TTS 只声明支持 breakable。要压 22-token prefill 的 6.2ms
  得改 model runner 的 prefill 图路径,不是配置能开的。
- **decode 步分解(r1,batch 1,2408 步,同步 execute 路径)**:wall 7.2ms = build 0.2 + 发射 0.7 +
  predictor 发射 0.1 + **finalize 等 GPU 6.1**;即 GPU 每步 ≈ decode forward ~2 + code predictor
  ~4.4ms。低 batch 下 code predictor(16 个量化器串行子步,已在一张图里,~0.27ms/子步)是 Talker
  每帧成本的主项:它同时决定 prefill 的 finalize 4.6ms、第二帧的等待、以及每帧节奏。下一刀候选:
  用与 vocoder 同样的手法(tensor-only 子步 + torch.compile 降 kernel 数)压子步内的 kernel 数。
- **混合 chunked prefill r20 三 seed:作废**。首帧 p50 3.7s(!)、完成率 63%、E2E p50 5.7s——
  `enable_mixed_chunk` 下本 scheduler 的 prefill/decode 混批把请求整体拖垮,r1 冒烟里 E2E 尾部
  变差就是前兆。r20 那 ~14ms 准入等待另找办法(例如在 resolve 等待期间做准入,或缩短 decode 步
  本身——code predictor 那 4.4ms 正是 decode 步的大头)。
- **code predictor 一步的 kernel 计数(bucket 1,capture 时 profiler)**:**1371 个 CUDA kernel,
  4.04ms 设备时间**,平均 3µs/kernel——纯延迟主导。分布:aten::mm+addmm 2.1ms(52%,255 次
  64x8 的 nvjet 小 GEMM,每次 ~6µs)、copy_ 0.32、seeded top-k/top-p 采样 0.31、elementwise 0.3、
  rmsnorm 0.28、cudnn attention 0.22、qknorm+rope 0.23、act_and_mul 0.11、index_select 0.11。
  与 vocoder 第六轮同构(1147 kernel → 编译后 356,3.2 → 1.8ms):把 `_predictor_forward_one_token`
  与子步胶水(dtype 转换、copy_/add_、unsqueeze)做成 tensor-only 并 torch.compile,可望把 elementwise/
  norm/copy 那 ~1.2ms 压掉大半;255 个小 GEMM 是层结构决定的(qkv/gate-up 若未融合可再省)。预期
  每帧 4.4 → ~2.5-3ms:首帧 −3ms(prefill finalize + 第二帧),每步 −1.5ms(低 batch 下 Talker 提速
  约 20%,r20 的准入等待与 decode 步同步缩短)。这是第十二轮的候选,需要先过外审(改 Talker 侧的
  采样路径,涉及数值/种子一致性)。
- **B(full prefill 图)的门槛**:`generation_batch_policy._validate_prefill_graph_policy` 对所有
  omni 模型只放行 breakable/disabled——不是 Qwen3-TTS 特有的能力声明,而是 omni 的 prefill 输入
  注入(`attach_omni_prefill_inputs`)只接了 breakable 路径。做 B 要把 input_embeds 注入接进
  SGLang 的 full prefill 图捕获(固定 token 桶、注意力在图内),属第十三轮 model runner 工作。

- **外审前账目的一处更正(事件树 r1 时间线,66 请求 p50)**:admission 0 → prefill start 5.1 →
  prefill end 17.7 → 首块 codes 到 vocoder 18.1 → vocoder 首块发出 **33.9** → coordinator 34.3;
  客户端 first playable 35.9。所以模型之外的 hop/HTTP 总共只有 ~2ms,不是我送审时估的 ~7ms;
  vocoder 段 15.8ms 才是大头:等第二帧 ~7.2(bootstrap 静音抑制要两帧)+ initial 批等待 2 +
  解码 4.6 + 胶水 ~2。
- **full 后端 prefill 图(临时放开 `_validate_prefill_graph_policy`,r1 探针)**:能捕获、能跑,
  47/47、0 underrun;prefill forward **6.2 → 0.79ms**,`execute()` 12.0 → 8.2ms,首帧 p50
  **35.6 → 32.1ms**。attestation 只验证图被捕获、不验证数值,合入前还差:固定采样种子下 full 对
  breakable 的输出码/音频一致性;以能力开关(而非放开全局策略)只对 Qwen3-TTS CustomVoice 打开;
  r20 三 seed。这一项适合作为独立的小 PR 跟进。

### 第十二轮:单帧 bootstrap(2026-09-06 14:50 PT)

`suppress_bootstrap_silence: false`(首块只等 1 帧)r1:47/47、0 underrun、**first playable p50
27.7ms、min 24.8、p95 40**(默认 35.6 / 32.3 / 50)。即首帧比默认少 8ms,落到 参照引擎 26ms 的量级。
可闻 TTFA 与 r20 underrun 见下。
**但可闻 TTFA 反而从 44ms 变成 107ms**:模型的第一帧本来就是 ~80ms 的静音("bootstrap silence"),
关掉抑制后首块立刻发出的就是这 80ms 静音,可闻音频要等第二帧——first playable −8ms 换来可闻延迟
+63ms,对听者是净损失。抑制的设计(解码两帧、扣掉静音帧)正是为此。**结论:保留默认抑制。**
对标口径要写清楚:我们的 first playable(=TTFB)27.7ms 与 参照引擎 的 26ms 同量级,但那是"首字节",
对听者有意义的是可闻 TTFA(默认路径 r1 p50 44ms)。可闻 TTFA 的下限 = 拿到第二帧的时刻 =
prefill 12 + 一步 7 + vocoder 4.6 + 胶水,所以外审说的 B(full prefill 图,−3.7ms)与 A(predictor
编译,−3~4ms)对可闻 TTFA 仍然成立,目标从 44 → ~36ms;r20 三 seed 跑完后本轮收口。

r20 三 seed(单帧 bootstrap 对默认抑制,同树,100% 完成):

| arm | underrun | first playable p50 | 可闻 TTFA p50 |
|---|---|---|---|
| 单帧 bootstrap | 1.19 / 1.03 / 1.17% | 46-47 ms | **124 ms** |
| 默认抑制 | 1.53 / 1.72 / 2.33% | 57-58 ms | **66-72 ms** |

underrun 在噪声内(这一轮默认臂偏高:同一棵树九个 seed 的默认臂 underrun 为 0.58-2.33%,
均值 ~1.2%——单 seed 噪声带比早先估的 ±0.4 更宽,约 ±1 个百分点)。**决定:保留默认抑制**;
参照引擎 对标时同时报 TTFB 与可闻 TTFA。
- **PR 已开(2026-09-06 15:40 PT)**:sgl-project/sglang-omni#1997,从 `qwen3-tts-pr1855-rebase`
  (HEAD 54e2d4f1,含 pinned black 格式化)开到 main,正文含上表与验证方式;已打 `run-ci`、请
  Hayden727 / yxs / zhaochenyang20 / BruceLoveDecimal / leihehehe 评审,并在 #1846、#1855 留了
  交叉引用。合并顺序:#1907(prefill 图去 QK-norm/RoPE 断点,CI 全绿)→ #1997 → full prefill 图
  跟进 PR(待数值验证)。

### 第十三轮:full prefill 图(PR 3,2026-09-06 21:00 PT)

分支 `qwen3-tts-full-prefill-graph`(基于 origin/main,9da81408):`ModelCapabilities` 新增显式字段
`supports_full_prefill_cuda_graph`(13 个模型全部显式声明,只有 Qwen3-TTS 为 True);引擎工厂对
`full` 后端做与 breakable 相同的能力门控并打开 input-embeds 传输;批策略放行 full(tc_piecewise
仍拒);CustomVoice 默认 `full`。单测 608 过(含新增的门控测试)。验证在跑:固定 client seed +
请求 seed=1234 下,prefill 图关闭 / breakable / full 三臂各跑贪心(top_k=1)与采样两组,按 prompt
比较 PCM(sha 是否相同、首秒相关系数、最大差);随后 r20 三 seed full 对 breakable。

### #1997 自审(2026-09-06 21:40 PT,十个角度并行子代理 + 逐条核实)

修在 4f47debf(净删 81 行)与 4ccf87ec(删非 arena 图路径,净删 277 行):
- **真 bug(多角度同时命中)**:(1) follow-up 批次里若同时有增量 cohort 与 legacy 回落流,legacy 解码
  会拿到还被在飞 cohort 占着的 pinned slot 而被误杀;且 legacy-only 批次永远不 drain 在飞 cohort。
  修:跑 legacy 组前先 drain(keep=0)。(2) 带参考音频的首块 codes 由 builder 里的 `torch.cat`
  产生,而 ready event 在 cat 之前记录——同进程 Base/voice-clone 请求可能读到未写完的内存。修:
  cat 之后另记一个事件。(3) 槽释放时上一任的清零/scatter 可能还排在别的流上。修:arena 释放时
  记事件,acquire 先等。(4) 同步 cohort 路径与提前发射路径重复且已分叉(finish 里有不可达的
  except;同步路径下坏行会整 cohort 回落)。修:合成一条 launch/finish。(5) runner `stats()` 在
  另一 worker 改 dict 时遍历。修:加锁。(6) CPU-only 构建下 `pin_memory()` 抛错。修:CPU 走普通张量。
- **精简**:`record_stream` 逐 chunk 调用(每 cohort ~24 次、持 `_state_lock`)改为 plan 持有 chunk
  引用到 commit;去掉重复的 grouper、runner 选择表达式、两处 teardown、arena 的 `_index`/
  `positions()`、无调用的 `state_bytes_per_stream`、`precompile` 里重复的 frame_positions 赋值、
  不可达的 None 检查;注释改为现状陈述并补前缀。
- **记录但未做(后续 PR)**:同步 `decode_delta` 仍有一套 per-request 状态的增量实现(~100 行);
  `codec_frame_position` 是可推导的镜像;staging 环深度手工推导自 keep=1;worker 上下文用
  threading.local + getattr;ready event 走 metadata 键而非消息字段;factory 与 scheduler 默认值
  不一致;WARM 图 1..8 全桶捕获中 {1,3,5,6,7}×B4/B8 基本不命中;capture 后逐图 gc/empty_cache。

**第十三轮结果与 PR(2026-09-06 21:50 PT)**:PR sgl-project/sglang-omni#1998 已开(分支
`qwen3-tts-full-prefill-graph`,基于 main,20 文件 +78/−16),已打 `run-ci`、请默认评审。
验证(#1997 树,同一 harness):
- 与"prefill 图关闭"(eager)的输出一致性,61 条 prompt、client seed 7、请求 seed 1234:**首秒音频
  两种后端对每条 prompt 都 bit 级一致**;整段 PCM 完全相同的比例:贪心 breakable 84% / full 84%,
  采样 breakable 92% / full 74%——差异都出现在首秒之后的采样平局翻转,说明 full 图的 logits 与
  eager 的距离比 breakable 略大;质量由 CI 的 TTS stage 2/3(WER、一致性)把关。
- r20 三 seed,100% 完成:full underrun 0.68 / 0.43 / 0.00%(均值 0.37%)对 breakable 3.24 / 1.64 /
  2.08%(2.32%);first playable p50 62-63 对 66-67ms,p95 80-84 对 90-96;可闻 TTFA p50 67-70
  对 78-89ms。r1 首帧 35.6 → 32.1ms。
- 注意前一轮 "d-default 对 d-pfull" 的对照无效:主机树上 PR 3 的默认已是 full,两臂其实都是 full;
  本轮改为显式 `breakable` 臂后才是真对照。
- **主机收尾(2026-09-06 22:00 PT)**:eval-h100 上任务容器 `sglang-omni-jaxan-1` 已停止并删除,
  map 记录已清(map 24 行、jaxan 容器 0 个);全部日志与 254 个输出目录留在
  `/data/jaxan/runs/20260902-mainline-nari-ab/`。后续若 CI 需要 GPU 复现,按规范新建容器。
- **#1998 CI 全绿(2026-09-06 23:30 PT)**:32 个 check 通过(TTS CI 五个 stage 含 stage 3 一致性、
  ASR、Qwen3-Omni 全部 stage、XPU 重跑通过;首轮 XPU 失败是镜像构建时 pip 下载超时,与代码无关)。
- **#1997 CI 全绿(2026-09-06 23:55 PT,head 89d5c1ed)**:33 个 check 通过,含 TTS CI 五个 stage、
  ASR 两段、Qwen3-Omni 十一段、XPU、单测。三个 PR 状态:#1907(CI 绿)→ #1997(CI 绿)→
  #1998(CI 绿,能干净合到 main+#1997 之上),均待 review。

## 第十四轮:三个 PR 的 review 回合与 #1754 roadmap 同步(2026-09-07 19:00-20:00 PT)

三个 PR 都收到 review。#1907 与 #1997 的意见上一轮已答复并修好,但**两个 fix commit 推错了
地方,reviewer 看到的仍是旧代码**:

- `67f7c7a7`(#1907 的 EagerRunner 作用域修复)推到了 `origin`(sgl-project)的同名分支,而
  #1907 的 head 读的是 fork(luojiaxuan/sglang-omni),PR head 一直停在 `20e81fc0`;
- `4ecf10a9`(#1998 的能力门修复)只在本地,未推。

两者已推正,PR head 现为 `67f7c7a7` 与 `5256e070`。**教训:PR 的 head repo 未必是 origin,
推之前先看 `gh api .../pulls/N --jq .head.repo.full_name`。**

### #1998(charliechenye CHANGES_REQUESTED + JiaxinD COMMENTED)

- 阻塞项已修(`4ecf10a9`):`validate_generation_batch_policy` 增加 `allowed_prefill_backends`,
  默认 `("breakable",)`;`qwen3_omni/stages.py:1063` 与 `:1188` 两个直调点行为不变,仍拒 `full`。
  能力由 `SGLangGenerationEngineBuilder.allowed_prefill_cuda_graph_backends()` 加宽。
- 集成契约已跟上(`4aa47993`):`test_qwen3_tts_batch_invariance.py` 的 CustomVoice 臂断言改 `full`,
  cookbook 的 prefill 一节改写(默认 `full`、三个值都列、顺带修掉"Non-Base 默认 breakable"与
  "Only CustomVoice takes this default"的自相矛盾),`tests/README.md` 的 policy 描述去掉 breakable 限定。
- 格式修复(`5256e070`):`4ecf10a9` 有一行 89 字符超 black 88 上限,lint 会挂。已用
  `uvx black@24.10.0` + `uvx isort@5.13.2` 核过全部改动文件。
- 上游 experimental 一节写进 PR 正文,依据是实读 sglang 源码:`arg_groups/cuda_graph_hook.py:490`
  对 `prefill.backend == FULL` 打 experimental 警告,`disable_full_prefill_cudagraph_if_incompatible`
  (:344)的 `rules = []` 确为空。含义:上游不会为任何 feature 自动关掉 full,我们 policy 里那份
  不兼容清单是这条路径上唯一的守卫。
- CI 全绿:lint / build-docs / omni-ci-gate / test-layout / CodeQL 均 pass。

### #1997

无需改动,逐项复核确认上一轮的修复都在:机器人 6 条(`except BaseException`、两处
`entry = None`、两个无用 import、变长 tuple lambda)全部落实;PR 正文已无 `record_stream`;
cookbook:407 已是 H100 数字并去掉了不在树里的 benchmark doc 引用。

### #1754 roadmap 同步

上次状态评论停在 2026-09-03 12:29 PT,已落后四天。本轮同步了:

- **更正**:09-03 评论里"follow-up 批每批约 30ms CPU 开销"的归因是错的,本文档
  「拆解」一节的四段计时(resolve 占 85.4%、CPU 合计 2.3ms)才是实际情况。该更正已写进 issue。
- 09-03 以来合入的五个 PR(#1852 / #1930 / #1900 / #1928 / #1901)列表与 commit;
- 三个在途 PR 的合并顺序与 #1998 依赖 #1907 的理由;
- issue 正文按事实修正:T-PR5 移入 Landed 并改指 #1852(原写 #1847,已关闭)、T-PR8/T-PR9 改指
  #1997 并注明 #1846/#1855 已并入、T-PR15 补上 #1900/#1907/#1998(原写"no PR yet")、
  T-PR18 去掉"review is blocked"(#1794 已于 09-02 合入)、T-PR6 补 #1928、T-PR19 的
  observability 半边注明由 #1997 承担。

### 本轮决策日志

| 问题 | 默认答案 | 理由 | 回滚 | 外审 |
|---|---|---|---|---|
| #1998 的 `eager_on_graph` 失效怎么处理 | 声明 #1998 依赖 #1907,不先于它合并;不把 #1907 的 mrope 修复拷进 #1998 | full backend 下 `eager_on_graph` 是 pass-through,QK-norm/RoPE 被捕获后读到绑定的陈旧 `mrope_positions` 槽,正是 seeded 一致率 92%→74% 的来源;#1907 是根治,拷贝会与 #1907 冲突且让它变冗余 | 若 #1907 被拒,改为在 #1998 内单独实现槽刷新 | JiaxinD 在 review 中明确给出"land #1907 first or drop that comment"两选项,本决定取其一,不另发 ChatGPT 外审 |
| 只发状态评论还是同时改 issue 正文 | 两者都做 | 正文的 Active/Landed 若不改,下一个读者仍会看到"T-PR9 no PR yet";luojiaxuan 此前已多次编辑该正文(参照引擎 外部验证条目即是) | GitHub 保留编辑历史,可回退 | 不涉及 |
| #1998 的 parity 数字(92%/74%)是否立刻重测 | 不重测,正文如实标注"measured on main without #1907, needs a rerun on the rebased tree before merge" | 重测需数小时 H100;#1907 仍在 review,定稿前重测有白烧风险 | 本地 `sglang-omni-p1907` 的 `f747d1bb` 就是 merge 后的树,随时可跑 | 不涉及 |

### 新规则:合并时给 reviewer 加 Co-authored-by(2026-09-07 20:10 PT per luojiaxuan)

已写入全局 CLAUDE.md 的「Git commits」节:合并 PR 时,凡帮忙 review 的人一律加
`Co-authored-by` 致谢;人类协作者署名,Claude 一律不署;机器人(`*[bot]`)排除;
邮箱用 GitHub noreply 形式而非对方 commit 里的私人邮箱;squash 必须显式传正文,
否则 trailer 未必落下。

**#1907 与 #1997 合并早于本规则十几分钟,未带 reviewer 致谢**,且已合并的 commit
不追补(改它要 force-push 共享分支)。#1997 带了被 squash 抹掉的原作者
(BruceLoveDecimal、leihehehe),但 reviewer @JiaxinD 未列入。

**#1998 的 trailer 已经写进分支本身**(2026-09-07 20:20 PT,tip amend 为 `6bca62be`),
而不是只记在这里等合并时再补——这样即使由别人从网页 UI 合并、走仓库默认的
`COMMIT_MESSAGES` 拼接,GitHub 也会把 co-author 认出来:

```
Co-authored-by: charliechenye <32603442+charliechenye@users.noreply.github.com>
Co-authored-by: JiaxinD <49501057+JiaxinD@users.noreply.github.com>
```

### 另一条新规则:tracking issue 用编辑不用追加评论(2026-09-07 20:25 PT per luojiaxuan)

起因是我在 #1754 上一条接一条摞状态评论,用户的原话是"显得太像 AI"。已写入
全局 CLAUDE.md 的「Development Rules」:现状写进 issue **正文**,同一段进展有
后续时**编辑自己上一条评论**而不是再发一条。本轮已回溯执行:把今天两条评论
合并成一条(`5578169739`,另一条已删),并把正文里 T-PR8/T-PR9 移进 Landed、
T-PR15 更新为 #1900+#1907 已合入、T-PR19 的 observability 半边标为已落地。


### 第十五轮:#1998 的重测与两条方法教训(2026-09-10 16:30 PT)

**背景**:#1907、#1997 已于 09-07 19:48 PT 合入 main;#1997 引入的跨进程 ready-event 漏洞由
Ratish1 在 #2046(09-08 18:15 PT)修好,#2042 修了图池吃掉 KV slack、#2057 让 rope kernel 直接写
predictor cache。#1998(full prefill 图)仍未合。

**重测口径**:eval-h100(一张 H100 80GB)、CustomVoice 1.7B、seed-tts-eval、开环泊松到达;
每臂都打印它实际捕获的 backend(eager 臂无捕获行,另两臂 `backend=breakable` / `backend=full`),
确认测的是被改的那条路径。

r20 三 seed(100% 完成,每 seed 786-795 请求):

| backend | underrun | first playable p50 | p95 | 可闻 TTFA p50 |
|---|---|---|---|---|
| breakable | 0.13 / 0.13 / 0.00% | 52.5 / 53.0 / 52.9 ms | 72.4 / 72.7 / 69.1 | 55.5 / 58.6 / 55.8 |
| full | 0.00 / 0.38 / 0.13% | 49.6 / 49.7 / 50.0 ms | 64.4 / 67.4 / 66.1 | 52.2 / 52.2 / 52.2 |

**underrun 两边都在地板上**(0-0.4%),所以 #1998 在 r20 的理由只剩首帧:p50 −3ms、p95 −5ms、
可闻 TTFA −4ms,臂内三 seed 离散度 <1ms。PR 正文早先写的"2.32% → 0.37%"作废:那是 #2046/#2042/
#2057 之前测的,现在 breakable 自己就 0.09%。

rps 2 一致性(56 请求/臂,client seed 7 + 请求 seed 1234,五个 server):

| 对比 | 贪心相同 | 采样相同 | 首秒 corr<0.9 |
|---|---|---|---|
| eager 重启(控制) | 82% | 93% | 0 |
| breakable 重启(控制) | 90% | 79% | 0 |
| full 对 breakable | 84% | 82% | 0 |
| breakable 对 eager | 72% | 80% | 0 |
| full 对 eager | 80% | 75% | 0 |

**教训一:没有同配置重启控制的"一致率"是废数**。同一配置重启一次就有 7-21% 的整段音频不同
(cohort 批次组成随时序变化 → 浮点细节变化 → 早期 argmax 翻转),所有跨 backend 的对比都落在
这个带里。我 09-08 报给 PR 的"92% → 74%"就是把 run-to-run 噪声读成了 backend 效应,已在 PR 正文
更正。JiaxinD 当时的判断(那是 #1907 之前被捕进图的 QK-norm/RoPE)方向对,但真正的问题是我的
度量缺控制。

**教训二:hyper01 上那次重测的 eager 臂是坏样本**。它的 first playable p95 达 1180ms(其余臂
50-80ms),并且它与包括自己 pre-#1907 版本在内的所有四个臂都在同样 22 条 prompt 上首秒不同;
H100 上换成健康的 eager 臂后,首秒差异归零。**臂内尾部异常就是该臂不可用作参照的信号**。

**CI 覆盖缺口**:`tests/test_model/tts_ci_config.py` 的 `qwen3-tts` preset 只有
`Qwen3-TTS-12Hz-1.7B-Base`,而 full 默认只对 CustomVoice 生效,所以 TTS CI **结构上测不到这条
路径**(#1900 的 breakable 默认同样没被覆盖过)。已在 PR 正文点名,建议单开一条 PR 给 CI 轮换
加 CustomVoice 臂。

**主机纪律**:eval-h100 现有四个 runner 安装、`omni-autoscaler.yaml` 的 `max_runners: 3`,三条
lane(2,3 / 4,5 / 6,7)在跑作业时各有 reaper 会 SIGKILL 外来 GPU 进程。测量期间我把 max_runners
临时降到 2(备份 `omni-autoscaler.yaml.bak-luojiaxuan-pfull-20260910T2249Z`),**但降 cap 不绑定
lane**——runner 仍可占任一 lane,我在 GPU 4 上还是被杀过一次,换到当时无 reaper 的 GPU 3 才跑完。
测完已把 max_runners 恢复为 3、容器删除、map 清理。

### 第十六轮:给 TTS CI 加 CustomVoice 臂(PR #2094,2026-09-10 17:20 PT)

**为什么需要**:排查 #1998 的覆盖时发现缺口比"CustomVoice 默认没被测"更宽——
`tests/test_model/tts_ci_config.py` 的 `qwen3-tts` preset 服务的是
`Qwen3-TTS-12Hz-1.7B-Base`,而 Base 在 `server_args_builder.py` 里把
`cuda_graph_backend_prefill` 解析为 `disabled`;prefill 图的默认只对 CustomVoice
checkpoint 生效(#1900 的 breakable、#1998 的 full)。stage 4(serving)又硬编码
`TTS_CI_PRESETS["higgs"]`。所以**Qwen3-TTS 的 prefill 图路径从来没有任何 CI 作业跑过**,
breakable 和 full 都没有。#1997 的跨进程回退能带着全绿的 check 进 main,就是这个缺口。

**PR #2094 做了什么**:`TtsCiModelPreset` 增 `voice` / `voice_clone` 两个字段(benchmark
侧 `benchmark_tts_seedtts.py` 早就支持 `--no-ref-audio --voice`,所以只是接线);新 preset
`qwen3-tts-custom-voice` 用 CustomVoice checkpoint + 具名音色;相似度 stage 对具名音色
跳过(它是拿生成音频与请求里的参考片段比,没有参考就无意义);**本臂暂不 gate**
(thresholds 标 `calibrated=False`、`gate_thresholds=False`),stage 1-3 照常跑并打印,
留给下一条 PR 在 CI 主机上做重复观测标定;新增三条契约测试(只有标定过的 preset 才能
gate、具名音色必须带 voice 而克隆臂必须不带、workflow 的 rotation 与 preset 注册表必须
同名)。标签 `run-qwen3-tts-custom-voice`(新建)+ dispatch 值 + 互斥检查一并接好。

**代价与备选**:第四个 rotation 成员把每个模型从 1/3 摊到 1/4。备选是让 `qwen3-tts` 这个
槽位内部再二选(Base / CustomVoice),higgs 与 moss 保持 1/3、CustomVoice 拿 1/6。我选了
显式第四成员,因为选择逻辑本身就是最容易被误读的地方(这次的教训),标签与 dispatch 也
能直接点名;PR 正文写明了这条取舍,评审若偏好 sub-pick 我就改。权重已在 runner 缓存里
(`/data/cache/huggingface` 下 4.3GB),不额外下载。

**收尾**:测试用的 CPU 容器已删、map 已清;`max_runners` 保持 3。

## 第十七轮:#1998 的 MMSU 红灯不是 #1998 的问题,是判据推导本身有 bug(PR #2095)

**背景**:#1998(full prefill graph)的 board 上唯一的红灯是
`Qwen3-Omni CI / stage 5 - MMSU accuracy + speed`,与 Qwen3-TTS 无关。它连着两次
在同一个 commit 上失败,失败行是:

```
1. latency_mean_s 0.203 > 0.2 at concurrency 16
1. latency_mean_s 0.211 > 0.2 at concurrency 16
```

**根因**:`tests/utils/apply_slack` 把 lower-is-better 判据算成
`round(P95 * 1.125, 1)`。MMSU 文本臂的 P95 参照是 **0.201 s**,于是
`round(0.201 * 1.125, 1) = round(0.226, 1) = 0.2`——**判据落到了它自己的参照以下**。
也就是说,一个跑出与标定完全一致的延迟的 run 会失败。这是全树里唯一一个低于自身参照的
判据,但同一个取整对每一个上界判据都有最多 0.05 的扰动;对亚秒级参照来说,0.05 与
政策本身想给的 12.5% slack 同量级(0.755 s 的参照实际只拿到 6% 余量)。

**修法(PR #2095)**:`latency_mean_s_max` 与 `rtf_mean_max` 走
`readable_upper_gate(reference, slack, digits)`,只在取整**放松**时保留好读的整数值,
否则直接用 `reference * slack`。关键性质:**没有任何判据变严**,所以这个改动不可能引入
新的 CI 失败;13 个判据放松不到 0.05(回到政策本来要给的 slack),MMSU 那一个不再低于
自身参照(0.2 → 0.226125)。

**为什么不干脆全部不取整**:那会让 6 个现存判据变严——最差的是 moss stream 的
`latency_mean_s_max` 从 1.1 收到 1.0676(严 2.9%)。higher-is-better 一侧同理,
`output_tok_per_req_s` 6.1 的判据会从 5.3 收到 5.3375。这是 bugfix 不是重标定,
所以只动"取整会收紧"的那一侧,min 判据一行不改。

**教训**:判据的 readability 取整是个没人要求的启发式,而它的误差在小参照上与 slack
同量级。判据推导里任何"为了好看"的变换,都要能回答"它最坏能把判据挪多远、挪的方向
是松还是紧"。这条与 `gate-calibration-lessons` 里那批教训同源:判据本身也是代码,
也需要契约测试(`tests/unit_test/test_ci_threshold_slack.py`)。

**收尾**:eval-h100 上跑单元测试的 CPU 容器 `sglang-omni-jaxan-1` 已删、map 行已清
(容器数 0 == map 条目数 0);`actions-runner-h100` 的 `omni-autoscaler.yaml`
`max_runners` 维持 3,我的备份文件比对一致后已删,四个 runner 安装回到我动手前的状态
(h100=3,-2/-3/-4=4)。

## 第十八轮:torch profiler + nsys 剖面,回答"还有没有余量"(2026-09-14 PT)

完整报告在 [qwen3_tts_profiler_headroom_20260914.md](qwen3_tts_profiler_headroom_20260914.md)。这里只记结论与对本文档旧结论的更正。

**判决**:两个速率下 GPU 都不是"又忙又发射满"。r1 请求内部忙时 SM Issue 5.4%(93% 的忙时低于 10%),中位 kernel 1.95 us,91% 的 kernel 时间落在填不满 132 个 SM 的 grid 上;r20 kernel 忙 61.6%,空闲全是 100 us 到 10 ms 的洞,talker 流每 12.9 ms 步空 4.55 ms 等 CPU。Tensor Active 任何阶段不超过 4.2%,DRAM read 最高 40.8%(bs=1 talker 图)。余量有两种:调度/CPU 空洞(大头由在途 #2123 覆盖,剩下的是准入白等一整个 decode 步)与低发射 kernel(predictor 982 kernel/帧、vocoder 60% 的重放是 eager-in-graph、B4/B8 图的 CUDA-core conv)。

**对旧结论的更正**:第 7 轮"cohort 间降频、重放前爬频"的猜想不成立,GPC 时钟稳定在 1.98 GHz(p1 1,878 MHz)。第 12 轮外审说"predictor GEMM 2.1 ms 已是地板"没有量 grid:27 个 GEMM 每子步只用 16-96 个 CTA,权重 2.65 GB/帧按 3.35 TB/s 只需 0.79 ms,实测 2.10 ms 即 38% 峰值。第 7 轮 COLD 编译臂作废是单 seed、在 arena 入图与流解耦之前,3 seed 复测后再定。

**排好序的候选**(收益都是算术推演):C1 准入同轮进队(r1 -3.4 ms 均值,r20 -8 到 -14 ms);C2 vocoder 编译 COLD 与 ramp 宽度(r1 -1.5 到 -1.9);C3 自适应 initial_batch_wait(r1 -2.0);C4 predictor 小 M GEMM 全 SM 化(r1 -1.0 到 -1.7,r20 约 -3);C5 predictor glue 融合 + 短上下文 attention;C6 seeded 采样 kernel;C7 双 token prologue 合并;C8 arena 扁平化;C9 修 B4/B8 图的 cudnn 算法(只动产能,r20 vocoder GPU -10 到 -12%)。被否的 8 条(含异步 decode 循环、talker GEMV 带宽、batched emit)与理由见报告第 4 节。下一步 E1(C1 A/B)、E2(C2+C9+C3)、E3(C4 离线微基准)见报告第 5 节。

## 第十九轮(上):E3,predictor 小 M GEMV 微基准,C4 按预注册阈值判 no-go(2026-09-14 PT)

**做了什么**:第十八轮报告的 E3。六个 predictor GEMM 形状(qkv 1024→4096、o_proj 2048→1024 加残差、o_proj_k1024、gate_up 1024→6144、down 3072→1024、project_input / lm_head 1024→2048),M ∈ {1,2,4,8,48},bf16 权重、fp32 累加。Triton 三种内核(寄存器 GEMV、其三维变体、tensor-core split-K dot),每个 (shape, M) 单独自动调优(76-182 个配置),对照 predictor 自己的 cuBLAS 路径(`F.linear` bf16)。计时协议:20 次背靠背调用捕成 CUDA graph,5 次预热后 30 次复放取中位数,40 份权重副本轮转保证每次从 HBM 读。skeptic 换 seed、反转形状顺序、无缓存重调优复跑,105 个格子全部在 ±1.5% 内。脚本与全表在 [qwen3_tts_e3_predictor_gemv/](qwen3_tts_e3_predictor_gemv/)(`gemv_bench.py`、`results.md`),原始产物在 eval-h100 `…/analysis/e3/`。跑在 GPU 1 上,用一次性容器 `sglang-omni-jaxan-2`(已删,map 已清)。

**结果**(验证者复跑版,每调用 us):

| 形状 | M=1 cuBLAS → Triton | M=48 cuBLAS → Triton | 纯流式读下限 | 阈值 |
|---|---|---|---|---|
| o_proj 4.19 MB | 5.98 → 4.00(1.49x) | 6.20 → 7.36(慢 19%) | 3.19-3.44 | ≤ 3.0 未达 |
| lm_head 4.19 MB | 5.04 → 3.71(1.36x) | 5.23 → 5.15 | 3.19-3.43 | ≤ 3.0 未达 |
| project_input 4.19 MB | 5.18 → 3.86(1.34x) | 5.38 → 5.30 | 3.19-3.42 | ≤ 3.0 未达 |
| down 6.29 MB | 7.81 → 4.75(1.64x) | 8.21 → 8.40 | 4.07 | 无 |
| qkv 8.39 MB | 5.79 → 5.15(1.12x) | 6.06 → 6.05 | 4.69 | 无 |
| gate_up 12.58 MB | 7.16 → 6.48(1.11x) | 7.73 → 7.80 | 5.77-6.01 | ≤ 5.5 未达 |

权重 load 加 `eviction_policy="evict_first"` 再省 4-8%(o_proj M=1 3.69 us),仍高于 3.0。数值通过:Triton 最大绝对误差与 cuBLAS 同量级(bf16 输出舍入主导),fp32/fp64 参考一致,无 NaN。

**判决**:四项阈值三项未达,**C4 按预注册规则划掉**。但要如实记两点:(1) 报告写的 3.0 / 5.5 us 阈值低于同 harness 里任何已测 kernel,包括不做算术的纯流式读(3.19 / 5.77 us),即阈值本身定错了,不是 Triton 做不到;(2) 真实收益是 bucket 1 每帧 GEMM 2.30 → 1.75 ms(省 0.55 ms,加 evict_first 0.61),bucket 48 慢 0.11 ms/帧,除非按形状混合派发(M ≥ 8 回 cuBLAS)才持平。换算到首帧:r1 约 -1.1 ms(两次曝光),r20 约 0。为这个收益引入自定义内核加混合派发,不值;C4 停在这里,除非以后只剩 r1 首帧一个目标再重开(重开要按实测下限重新预注册阈值)。

**skeptic 修正的三句话**:bench 报告说"fused atomic split-K 在 BLOCK_M=64 出确定性错误",实际错的是 BLOCK_M=16(M∈{2,4,8})且 REDUCE=0 的 74 个配置,M=48 全部通过,没有胜出配置用到这条路径,但混合派发不能碰它;down 行的误差对比把带残差的 Triton 和不带残差的 cuBLAS 放在一起,换成同 epilogue 的 cuBLAS 对照后 Triton 更低;"阈值任何 kernel 都不可达"改成"已测的都没达到"。

**教训**:阈值要在同一 harness 里先量流式下限再定,报告第 3 节 C4 的"按数据时间加 1.3 us 固定开销定价"低估了固定开销(实测空 kernel 1.17-1.47 us,流式读 4.19 MB 需 3.19 us,即约 1.9 us 固定 + bytes/3.25 TB/s)。

## 第十九轮(下):E2、#2123、C1 在 H100 上的实测,两个 PR(2026-09-16 至 09-17 PT)

**环境**:eval-h100 在 09-16 宕机,改到 Tilde(worker-29,H100 HBM3,驱动 580)。用 uv venv 从零配环境,坑记在 memory `tilde-sglang-omni-venv-recipe`。
Tilde 的绝对延迟比 eval-h100 的 Docker 环境差约 20%,所以只做同机配对 A/B:单卡、各臂顺序跑、每臂重启服务,3 个客户端 seed,rps 1 和 rps 20,预热 30 s、计量 60 s。
run 根目录:`tilde:/home/guests/zhen/jaxan/runs/20260916-qtts-e2/`(`outputs/`、`prof/`、`logs/`)。

**E2(eval-h100,09-14 到 15,5 臂 x 3 seed,基于 442e559b)**:首块 COLD 宽度编译后,rps 1 首帧 p50 -2.5 ms,rps 20 -2.9 ms。自适应初始等待在 rps 20 再 -2.5 ms,rps 1 在噪声内。
WARM ramp 宽度编译和 cudnn benchmark 让 vocoder GPU 时间下降(B8 replay 6.4 → 3.6 ms、B4 5.3 → 2.4 ms,`implicit_convolve_sgemm` 清零),但用户侧延迟和 underrun 都没有变化,所以不上。
Tilde 上在当前 main 复核 PR 分支,rps 20 首帧 p50 56.9/56.5/56.6 → 51.2/51.7/50.7,rps 1 32.0-32.1 → 29.8-31.0。
**输出一致性**:codes 不变(56 条长度全部一致,首秒相关 1.0000)。首块改走编译内核后,和整段解码相比 eager 增量是 39.6 dB,编译增量是 34.2 dB,首块从前者降到后者。main 的稳态块本来就在后者这一档。编译内核比 eager 低 5 dB 的原因还没查,值得单列。
测量上踩的坑:4 路并发会让一个 main 样本在冷启动时卡 8 秒、打乱批次,22 条的长度都变了,这个样本作废;杀服务必须杀整棵进程树。→ **PR #2217**。

**#2123(Ratish1,09-16 合入)**:同节点对比合入之前的 main,rps 1 首帧 p50 38.9 → 31.7 ms,rps 20 64.7 → 56.0 ms,rps 20 可闻 p50 78.9 → 58.0 ms,underrun 2.0-2.5% → 0-0.5%。
rps 20 trace 里每个 decode 步 p50 7.9 ms = talker 图 2.3 + predictor 图 4.1 + 空洞 0.1 ms(占 2%);剖面报告里是空洞 4.55 ms、步长 12.9 ms。**第十八轮的 CPU 空洞瓶颈已经消除**,p95 仍有 12 ms,集中在 prefill 和准入那几轮。现在步长里最大的一块是 predictor。

**C1(准入同轮进队)**:`process_input_requests` 在取批次之前,对本轮提交的 build 最多等 2 ms。3 个 seed 配对,rps 20 首帧 p50 51.9/56.2/56.0 → 43.9/47.2/47.6 ms(-8.0/-9.0/-8.4),可闻 p50 -8.4/-9.8/-7.9,rps 1 与 underrun 不变。main 在 C1 臂之后又跑了两次,差异不到 1 ms,说明没有漂移。→ **PR #2216**。
坑:只开请求事件记录(不开 torch)也会在 rps 20 把服务拖垮(40 秒写了 7.1 万条事件),那一轮 seed 作废后重测。事件数据只能在不计量的 run 里采。

**准入时间线(main 已含 #2123,rps 20,p50;来自被干扰的那一轮,只作量级参考)**:请求到达 → build 开始 6.3 ms(要等下一轮循环) | build 2.2 ms(p95 9.9) | build 完 → 进队列 7.5 ms(C1 针对这段) | 进队列 → prefill 7.8 ms(疑似在飞请求顶到 48 的上限)。

**下一步**:predictor 在每个 decode 步里占 4.1 ms,是 C5(glue 融合 + 短上下文 attention)、C6(采样内核)、C7(双 token prologue)的目标;准入的另外两段(到达后等循环、进队列后等位置)还没有候选,需要先拆开看。

## 第二十轮:#2216 + #2217 + #1998 叠加实测(2026-09-17 PT)

Tilde worker-10,单张 H100,各臂顺序跑,每臂重启服务,3 个客户端 seed,rps 1 和 rps 20。main 在首尾各跑一次(M 与 M2),rps 20 首帧 p50 分别是 55.1-55.9 和 55.7-56.3,没有漂移。
CE = main + #2216 + #2217;CEF = CE + #1998(CustomVoice 走 full prefill graph)。三个分支都能干净地合到 main `7b49bc4b`。

| 指标(p50,ms) | M | CE | CEF | CEF 对 M |
|---|---|---|---|---|
| rps 1 首帧 | 31.6 / 32.1 / 32.7 | 29.9 / 30.1 / 30.0 | 27.0 / 27.2 / 26.8 | -5.1 |
| rps 1 可闻 | 31.8 / 32.3 / 32.8 | 30.1 / 31.3 / 30.7 | 27.0 / 27.4 / 27.1 | -5.1 |
| rps 20 首帧 | 55.5 / 55.1 / 55.9 | 42.8 / 42.0 / 43.1 | 39.0 / 39.4 / 39.5 | -16.2(-29%) |
| rps 20 首帧 p95 | 81.4 / 82.8 / 75.5 | 73.5 / 70.6 / 66.6 | 60.5 / 59.4 / 55.8 | -21 |
| rps 20 可闻 | 57.6 / 57.0 / 57.7 | 45.0 / 44.1 / 44.8 | 40.3 / 40.4 / 40.8 | -17 |
| rps 20 可闻 p95 | 122.5 / 102.1 / 115.9 | 102.0 / 84.7 / 95.1 | 83.9 / 68.6 / 63.1 | -41 |
| rps 20 underrun | 0.17-0.26% | 0.08-1.4% | 0-0.17% | 不变 |

**读法**:CE 在 rps 20 上 -13 ms,约等于 C1(-8.5)加 E2(-5.4),说明两者基本独立、可以相加。#1998 在此之上 rps 1 再 -3、rps 20 再 -3.5,与之前在 eval-h100 上测的一致。
四个 PR 合起来,rps 1 首帧从 32 降到 27 ms(参照实现约 26),rps 20 从 56 降到 39 ms(参照约 26,p95 约 38)。**rps 1 已追平,剩下的差距集中在负载下**。
注意 Tilde 的绝对值比 eval-h100 的 Docker 环境高约 20%,所以这里的差值比绝对值更可信。

**下一步**:rps 20 还差约 13 ms。剩下的已知项是准入的两段等待(到达后等下一轮循环约 6 ms、进队列后等空位约 8 ms,p50)和 predictor(每步 4.1 ms)。

## 第二十一轮:在 2026-09-19 的 main 上重测 #2216 / #2217(2026-09-19 16:14-17:39 PT)

**为什么要重测**:第二十轮的基准是 09-17 的 main。两天里 main 进了 65 个提交,自己就快了一截,旧的差值不能再直接引用。
Tilde worker-11,单张 H100,五个臂顺序跑(M3 = main `d055353a`,C = main + #2216,E = main + #2217,CE2 = 两个都加,M4 = 再测一次 main),每臂重启服务,3 个客户端 seed,rps 1 和 rps 20,预热 30 s、计量 60 s。

| 首帧 p50(ms) | M3(main) | C(#2216) | E(#2217) | CE2(两个) | M4(main 复测) |
|---|---|---|---|---|---|
| rps 1 | 29.5 / 30.1 / 29.6 | 29.3 / 29.4 / 29.8 | 29.1 / 28.8 / 29.5 | 28.4 / 28.7 / 28.5 | 31.6 / 30.8 / 30.7 |
| rps 20 | 48.6 / 47.5 / 47.6 | 39.7 / 39.7 / 39.8 | 44.3 / 44.6 / 44.7 | 36.9 / 37.2 / 37.2 | 50.0 / 50.4 / 50.6 |
| rps 20 可闻 p50 | 48.8 / 48.8 / 49.7 | 40.5 / 41.2 / 41.1 | 45.6 / 45.7 / 45.8 | 38.1 / 38.7 / 38.2 | 51.0 / 51.7 / 51.2 |
| rps 20 首帧 p95 | 67.1 / 67.8 / 70.1 | 58.7 / 60.2 / 62.2 | 62.9 / 65.0 / 65.4 | 56.4 / 56.8 / 58.6 | 70.2 / 72.2 / 75.2 |
| underrun | 0 | 0-0.17% | 0-0.08% | 0-0.09% | 0-0.43% |

**今天的 main 比 09-17 的快**:rps 20 首帧 p50 从 55.5 降到 47.6,rps 1 从 32.1 降到 29.6。同一台机器、同一套 harness,所以这是 main 自己的改进,不是环境差异。

**M3 与 M4 差了 2.8 ms,原因查到了**:worker-11 上在 17:27 PT 起了另一个作业(同账号另一条实验线,64 核 2 卡),M4 正好在 17:22-17:39 PT 这一段跑,整段与它重叠。前四个臂跑在干净的节点上,所以基准取 M3,M4 只用来说明这次有外部干扰。

**结论(对 M3)**:#2216 在 rps 20 上 -8 ms,#2217 -3 ms,两个一起 -10.5 ms,可闻 p50 -10.6 ms;rps 1 三个臂都在 -0.5 到 -1.2 ms,落在噪声里。两者仍然可加,和第二十轮在旧 main 上看到的关系一致。
绝对值上,两个 PR 合起来把 rps 20 首帧带到 37 ms、可闻 38 ms。Tilde 比 CI 环境慢约 20%,换算过去大约是 31 和 32 ms,参照实现是 26 ms。

**CI 侧**:CustomVoice 臂的 PR #2094 已提,lint / docs / test-layout / CodeQL 全绿,但模型 CI 从 09-19 14:47 PT 起一直排队,GPU runner 所在主机连不上(`Permission denied (publickey)`),要等主机恢复才能验证这条臂真的跑起来。

**下一步**:同一节点上接着跑两个 PR 合并树的剖面(rps 1 与 rps 20),用来重排 predictor 的三个候选 C5 / C6 / C7;这轮剖面与上面那个外部作业同节点,CPU 侧的空洞读数会偏大,只拿它看内核构成。

## 第二十二轮:两个 PR 合并树的剖面,重排剩下的候选(2026-09-19 18:00 PT)

Tilde worker-10,单卡,`main d055353a + #2216 + #2217`,torch profiler 带 stack,rps 1 采 8 秒、rps 20 采 2 秒。
第一次采集(作业 358488)两份 trace 都是截断的:`stop_profile` 的导出还在写,脚本就开始了下一轮采集、最后又把服务杀了,rps 20 那份原始 JSON 有 841 MB。改成"导出完成再往下走"(轮询文件大小稳定 + 用 gzip 解一遍验证)后重采,两份都完整(r1 968 MB、r20 450 MB 原始)。
采样期间客户端延迟本身是坏的(profiler 带 stack 的开销把服务压垮,rps 20 那一轮只有 16% 成功),所以这轮只读内核构成与图重放时间,不读延迟。

**decode 步的构成**(stream 7 是主 talker 流):

| | rps 1 | rps 20 |
|---|---|---|
| 步长 p50 | 6.05 ms | 7.62 ms |
| talker 图 | 1.99 | 2.30 |
| predictor 图 | 3.80 | 4.09 |
| CPU 空洞 | 0.09(2%) | 0.11(2%) |

**读法(外审指出的口径问题,已改)**:表里每一格都是该分量自己的 p50,**不能相加当成关键路径**。
rps 20 下三项相加 6.50 ms,步长 p50 是 7.62 ms,差的 1.12 ms 没有归属(rps 1 差 0.17 ms);
那部分落在 talker 结束到 predictor 开始之间、以及各分量分布不同带来的中位数偏差里,这轮没有拆开。

**#2123 之后 CPU 空洞确实不再是瓶颈**,两个速率下都只占步长的 2%。按各自 p50 算,predictor 约占步长的 63%(rps 1)和 54%(rps 20),是现在最大的一块。

**predictor 内部的构成**(rps 1,每帧 16 次单 token 前向 + 15 次 lm_head 与采样)。
分母取 rps 1 下 predictor 图重放的均值 3.34 ms(506 次重放的平均,与上表 3.80 ms 的口径不同,
后者是步长分解里的 p50):

| 项 | 每次 | 每帧次数 | 每帧合计 | 占 predictor |
|---|---|---|---|---|
| 小 GEMM(`nvjet_..._64x8_64x16`) | 7.4 us | 268 | 约 2.0 ms | 60% |
| 种子采样内核 | 21.3 us | 15 | 0.32 ms | 9% |
| attention(cudnn sdpa) | 3.1 us | 64 | 0.20 ms | 6% |
| rmsnorm / rope / act 等 | 1.3-1.7 us | 约 300 | 0.45 ms | 13% |

这些百分比只对 rps 1 成立;rps 20 下 predictor 图重放是 3.72 ms,同一批 GEMM 的占比要重新分摊,这轮没有按形状拆 rps 20 的 GEMM。

小 GEMM 是 batch 1 下的单 token 矩阵乘,按权重体积算 HBM 下限约 2.2 us,实测 7.4 us,和第十八轮"GEMM 只跑到 HBM 峰值 38%"的结论一致。**但这是低负载才有的低效**:rps 20 下同族内核变成 64x16 / 64x24 的分片、6-10 us 跑的是几十倍的行数,已经接近合理。这也解释了第十九轮 C4(全 SM GEMV)的微基准结论——batch 1 能省 0.55 ms/帧,batch 48 省不到。

**rps 20 下新出现的一项**:`cudnn::engines_precompiled::nchwToNhwcKernel` 12.4 us x 4919 次 = 60.9 ms,占主流 GPU 时间的 5.6%(rps 1 只占 1.7%)。这是 vocoder 卷积前的布局转换,属于新候选。

**按这轮数据重排候选**:

| id | 目标 | 每帧可省 | 占步长 | 风险 |
|---|---|---|---|---|
| C10 | 调度循环在等 GPU 的那段时间里收信箱 | 不在 GPU 上,目标是准入的 6.3 ms 等待 | 与 C1 同族 | 中,动调度热路径 |
| C6 | 种子采样内核 | 0.32 ms | 5% | 高,要保持与参照实现逐位一致的 top-k 并列语义 |
| C7 | 两个 prologue token 合成一次前向 | 约 0.21 ms | 3.5% | 低 |
| C11 | vocoder 卷积的布局转换 | rps 20 主流时间的 5.6% | 未估 | 未知 |
| C5 | 短上下文 attention | 0.20 ms | 3% | 中 |

**下一步取 C10**:它针对的是准入链路上"到达后等下一轮循环"的 6.3 ms,和已经拿到 -8 ms 的 C1(#2216)是同一族;GPU 侧的候选加起来也就百分之几的步长,而 rps 20 的剩余差距主要还在负载下的排队结构里。

**调度循环是主机侧受限的,不是在等 GPU**(同一份 trace,按 CPU 侧的 CUDA runtime 调用统计):
调度线程每步只在 `cudaEventSynchronize` 上阻塞 0.77 ms(rps 20,184 次调用)或 1.31 ms(rps 1,506 次),
占步长的 10% 与 22%;`cudaGraphLaunch` 自己就吃掉 210 us(rps 20)到 556 us(rps 1)的 CPU,每步要调 8 次以上。
rps 20 下全部 runtime 调用合计只占窗口的 27%,其余是 Python。
**这说明"在等 GPU 的时候顺手收信"这个说法本身不成立**——循环里根本没有几毫秒的 GPU 等待可以借用,
迭代长是因为主机侧工作多。C10 仍然有意义(它把收信点从每迭代一次变成两次),但真正的上限来自迭代本身的长度,
所以下一个候选应该是把收信与 build 提交搬离调度线程(记作 C12),而不是在循环里找更好的插入点。
注意这份 trace 是开着 profiler 采的,主机侧时间被放大,所以上面的比例是"主机占比偏高"的一侧,
真实的 GPU 等待占比会更大一些;结论的方向不受影响。

**外审(GPT-6 Pro)对这个选择的两条修正**,全文见 `docs/reviews/2026-09-19-qwen3-tts-c10-mid-iteration-receive.md`:
其一,两个收信点的平均收益有上限——迭代周期 T、到达均匀分布时,平均收信等待最多从 T/2 降到 T/4,
即 3.8 ms 降到 1.9 ms,我原来"预期 -3 ms"是高估。
其二,**代码里的中间不等于时间上的中间**:`run_batch_launch` 是异步返回的,插在它之后的收信点在墙钟上
可能离循环顶部只有一两毫秒,迭代的大部分时间还在后面的等 GPU 与 resolve 上。这一条直接决定 C10 有没有效,
下一轮实测先看 A/B 差值,若为零则把收信点移到真正的等待里。
它还建议做 2x2 消融(#2216 的 2 ms 等待 开/关 x C10 开/关),避免两者其实在消除同一个"错过批次截止"。

## 第二十三轮:C10 第一次实测为零,原因是代码放错了循环(2026-09-19 18:23-19:15 PT)

Tilde worker-3,单卡,三个臂顺序跑,每臂 3 个 seed。A = main `2d0b48ff` + #2216 + #2217,B = A 再加 C10,A2 = A 重跑(当 A/A 噪声对照)。

| 首帧 p50(ms) | A | B(加 C10) | A2 |
|---|---|---|---|
| rps 1 | 28.8 / 29.2 / 29.3 | 28.7 / 28.8 / 29.9 | 28.5 / 28.9 / 29.0 |
| rps 20 | 37.9 / 38.1 / 38.9 | 38.0 / 38.1 / 38.2 | 37.9 / 38.2 / 38.2 |

**A 与 A2 只差 0.1 ms,B 落在这个区间里,所以 C10 的效应是零。**

**原因查清了:C10 加在 `event_loop_async_decode` 里,而 Qwen3-TTS 跑的是 `event_loop_normal`。**
`enable_async_decode` 默认 False,只有 zonos2、fun_asr、moss_transcribe_diarize、whisper 的 engine builder 会打开它,
Qwen3-TTS 的没有。也就是说那九行代码在被测的臂里从头到尾没有执行过。这是我的实现错误,不是"改动无效"的结论。
**教训**:改调度循环之前先确认这个模型走的是哪个循环,判据是 engine builder 里有没有传 `enable_async_decode`。
A/A 对照在这里起了作用:如果没有 A2,0.1 ms 的差可能会被读成"有一点点效果"。

改正后的版本把这次读取放进 `event_loop_normal`,位置在 `run_batch` 返回之后、`process_batch_result` 之前
——按第二十二轮的主机侧统计,那个位置大约在一次迭代的 60% 到 70% 处,后面还有 `process_batch_result`、
`get_next_batch_to_run` 和下一次发射的 Python 工作。按外审给的公式,收信点相隔 a 时平均等待降幅是 a(T-a)/T,
a ≈ 0.65T 时约为 0.23T,即 1.7 ms 左右,比插在发射之后(a ≈ 0.2T,约 1.5 ms)略好,但都低于理论上限 T/4。
重测已排队。

**这也把 C12 的动机变强了**:既然循环是主机侧受限、迭代长度就是 7.6 ms,那么在循环里挪收信点最多只能拿到 T/4;
把收信与 build 提交搬到独立线程才能把这段等待整个去掉。

## 第二十四轮:C10 改正后反而变慢,这条路判死(2026-09-20 02:20-21:16 PT)

两个作业接着跑,都在 Tilde、单卡、每臂 3 个 seed。

**2x2 的下半(作业 358946,worker-8)**。C = C10(此时还是死代码)+ 去掉 #2216 的 2 ms 等待,D = 只有 #2217,A3 = 两个 PR 都在。

| 首帧 p50(ms) | C | D | A3 |
|---|---|---|---|
| rps 20 | 45.3 / 45.3 / 45.4 | 45.1 / 45.8 / 46.0 | 39.1 / 39.8 / 43.6 |
| rps 1 | 29.2 / 29.6 / 29.9 | 29.0 / 29.1 / 29.1 | 29.3 / 29.4 / 30.6 |

C 与 D 在 0.5 ms 内相等(符合预期,C10 在这两个臂里都没执行),两者一起构成一对"不同 build、语义相同"的对照。
**#2216 在今天的 main 上单独值约 6 ms**(D 45.8 对 A3 39.8),与 09-19 那轮的 -8 到 -10 同向,幅度偏小,因为这次 A3 的第三个 seed 抖到 43.6。

**改正后的 C10(作业 359150,收信点移进 `event_loop_normal`)**。A4/A5 是两个 PR 的臂,首尾各跑一次;B2 是再加 C10。

| 首帧 p50(ms) | A4 | B2(加 C10) | A5 |
|---|---|---|---|
| rps 20 | 38.6 / 38.9 / 39.7 | 42.4 / 43.4 / 43.7 | 39.8 / 40.4 / 43.2 |
| rps 20 可闻 | 40.3 / 40.5 / 40.8 | 44.4 / 45.3 / 45.4 | 41.1 / 42.0 / 44.9 |
| rps 1 | 28.7 / 28.8 / 29.1 | 30.9 / 30.9 / 32.0 | 30.7 / 31.1 / 31.1 |

**B2 比首尾两个 A 臂的均值(39.65)慢 3.75 ms,A/A 之间的漂移是 1.5 ms,所以这是一个真实的回归。**

**为什么加一次收信反而更慢**:循环是主机侧受限的(第二十二轮),迭代长度 T 就是首帧等待的来源。
多一次 `process_input_requests` 每轮都要付出排空信箱、暂存、两次 drain 的 Python 开销,**哪怕一个请求都没到**;
而它换来的相位收益上限只有 T/4。在这台机器上,前者明显大于后者:T 变长,所有请求的等待一起变长。
收信点还排在 `process_batch_result` 之前,把刚算完那一步的流式输出往后推,也直接落在音频的关键路径上。

**结论:C10 两种放法都判死,不开 PR。** 记录在案的可复用结论是:
**在一个主机侧受限的调度循环里,"多做一次检查"这类改动的代价按迭代计,收益只按相位计,默认是亏的。**
反过来,这也给出了一个可观测的杠杆——每轮省下 0.2 ms 的主机工作,量级上就抵得上这次 3.75 ms 的回归。
下一步不是在循环里挪位置,而是**减少每轮的主机工作**,或者把收信与 build 提交整个搬离调度线程(C12)。

**下一步要先补一个测量**:这轮的 trace 只记了 kernel 与 cuda_runtime,没有 cpu_op,所以那段每轮约 4 ms、
没有任何 CUDA 调用的纯 Python 空档(graph launch 之间的 p90 间隔 4.27 ms)还没有归属。
先用 Python 级采样剖面把它拆开,再决定 C12 与"减少每轮主机工作"哪个先做。

## 第二十五轮:直接给调度循环插桩,拆开每轮的主机时间(2026-09-20 21:35-21:46 PT)

**为什么不用剖面**:Tilde 上 py-spy attach 不了(`Permission Denied`,ptrace 被拒);而 `/start_profile` 出来的 trace
只有 kernel / cuda_runtime / gpu_memcpy / cuda_driver 几类事件,**没有 cpu_op 也没有 python_function**,
所以第二十二轮那段"每轮约 4 ms、没有任何 CUDA 调用"的纯 Python 空档没法归属。
改用最直接的办法:单独开一棵实验代码树,在 `event_loop_normal` 的四个阶段插 `time.perf_counter()` 累加,每 200 轮打一行均值。

**rps 20(均值,单位 ms,n=8600)**:

| 阶段 | 内容 | 时间 | 占比 |
|---|---|---|---|
| ingress | `process_admin_requests` + `recv_requests` + `process_input_requests` | 0.14 | 1.5% |
| select | `get_next_batch_to_run` | 1.03 | 11% |
| run | `run_batch`(建 sched output、模型执行含等 GPU、发流) | 7.40 | 81% |
| result | `process_batch_result` | 0.59 | 6% |
| 合计 | | 9.16 | |

rps 1 的同一组是 0.11 / 0.87 / 6.80 / 0.46,合计 8.20 ms。
(这里的合计比第二十二轮的步长 p50 大,因为它是所有迭代的均值、含 prefill 轮,且插桩本身有开销。)

**两个读法**:

其一,**收信本身只要 0.14 ms**,所以 C10 那 3.75 ms 的回归不可能是"多做一次收信的开销"。剩下的解释是它**改变了批次结构**:
提前准入让等待队列在 `get_next_batch_to_run` 的时刻更经常非空,于是 prefill 批次切得更碎、更频繁地打断 decode。
下一轮用插桩树对比开/关第二次收信时的 prefill 批次数与平均批大小来验证。
**这条如果成立,对 C12 是坏消息**:把收信搬到独立线程同样是"更早准入",会撞上同一个机制。

其二,**`get_next_batch_to_run` 每轮 1.03 ms 是纯主机开销**,占了一轮的 11%,而且 rps 1 下也要 0.87 ms。
它不依赖 GPU,是现成的、可度量的减负目标。

## 第二十六轮:配对插桩对照推翻了"C10 有害"的读法,也推翻了"循环是主机侧受限"的读法(2026-09-20 21:48-22:07 PT)

同一棵插桩树,只差"有没有第二次收信",同一节点上背靠背跑,同一个 client seed,rps 20:

| | 无第二次收信(t) | 有第二次收信(u) |
|---|---|---|
| 首帧 p50 | 44.44 ms | 44.43 ms |
| 每轮合计 | 9.09 ms | 9.12 ms |
| ingress | 0.14 | 0.03 |
| select | 1.02 | 1.02 |
| run | 7.31 | 7.44 |
| result | 0.62 | 0.63 |
| decode 批次数(平均批大小) | 7323(12.1) | 7323(11.6) |
| prefill 批次数(平均批大小) | 1277(1.2) | 1277(1.1) |

**两条修正**:

其一,**第二十四轮那 3.75 ms 的回归没有被复现**。这一轮里两臂首帧只差 0.01 ms,批次数完全相同,每轮总时间只差 0.03 ms,
第二十五轮猜的"提前准入把 prefill 切碎"没有发生。第二十四轮的 A/A 漂移本身就有 1.5 ms,B2 的 +3.75 ms 更可能是节点状态而不是改动。
**正确的结论不是"C10 有害",而是"C10 的收益没有被测出来"**:两次独立测量分别给出 0.0 和 +3.75,没有一次是负的,
按预注册的判据(要看到可复现的负向差值)这条改动不该上。外审那句"单独一次 37→34 不足以说服人",反过来同样成立。
ingress 从 0.14 降到 0.03 只是说明工作被挪到了循环的另一处,总量没变。

其二,**"循环是主机侧受限"这个说法要收回**。把 `run_batch` 拆开之后:
`build_sched_output` 0.02 ms、**`model_runner.execute` 6.69 ms**、发流与收尾 0.56 ms。
execute 这 6.69 ms 与 GPU 图的时间(talker 2.30 + predictor 4.09 = 6.4 ms)几乎相等,也就是说这一段基本就是 GPU 本身,
主机在里面主要是发射与收集。第二十二轮从 trace 里读出的"27% 在 CUDA 调用里、其余是 Python"是 profiler 放大后的假象。

**一轮 9.09 ms 的真实构成:GPU 图 6.4 ms + 循环自己的主机开销约 2.3 ms**
(select 1.02 + 发流收尾 0.56 + result 0.62 + ingress 0.14)。

**所以候选排序回到 GPU 侧**,外审当初的判断在这一点上是对的,我之前用"主机侧受限"驳回它是基于被放大的剖面数字。
predictor 的 4.09 ms 是单项最大的一块,其次是 talker 的 2.30 ms;循环主机开销那 2.3 ms 里最大的一项
`get_next_batch_to_run`(1.02 ms)在上游 sglang 包里,不属于本仓库。

## 第二十七轮:把剩余空间算清楚,以及为什么现在测不动了(2026-09-20 22:20 PT)

拿到 predictor 的真实尺寸(`code_predictor_config`:hidden 1024、5 层、16 头 / KV 8 头 / head_dim 128、
intermediate 3072、vocab 2048、`num_code_groups` 16),把每帧的权重流量算出来:

每层 qkv(1024→4096)8.4 MB + o_proj(2048→1024)4.2 MB + gate_up(1024→6144)12.6 MB + down(3072→1024)6.3 MB = 31.5 MB;
5 层 = 157 MB;每帧 16 次单 token 前向 = **2.52 GB**,再加 15 个 lm_head 63 MB。
157 MB 远大于 H100 的 52 MB L2,所以这 16 次前向每次都要重新从 HBM 拉一遍权重,而且 16 步之间有采样依赖、不能重排。

**但这不等于有 4 倍空间**。第十九轮那份 E3 微基准里有一列"stream floor":同尺寸纯拷贝内核的下限。
8.4 MB 的读取在这台机器上的实测下限是 4.71 us(1.8 TB/s,占标称峰值 53%),而 cuBLAS 是 5.78-5.91 us。
也就是说**小矩阵的带宽上限本来就只有标称的一半左右,而 cuBLAS 已经跑到那个上限的 80-90%**。
按 M=8 那一列把一帧的 GEMM 加起来约 2.3 ms,纯流式下限约 1.55 ms,**整帧 GEMM 最多再省 0.8 ms**,
而 Triton 版本只在 M=1-2 时赢、M≥8 反而输——这正是第十九轮 C4 判 no-go 的原因,那个判断是对的。

**于是 rps 20 下一轮 9.09 ms 的账可以完整写出来**:

| 项 | 时间 | 还能拿走多少 | 代价 |
|---|---|---|---|
| predictor 图 | 4.09 ms | GEMM 至多 0.8(实际拿不到)、采样内核 0.32(C6)、attention 0.20(C5)、少一次前向 0.25(C7) | C6 要保持逐位一致的并列语义,风险高 |
| talker 图 | 2.30 ms | 未拆 | |
| `get_next_batch_to_run` | 1.02 ms | 在上游 sglang 包里,不属于本仓库 | 要开上游 PR |
| 发流与收尾 | 0.56 ms | | |
| `process_batch_result` | 0.62 ms | | |
| 收信与准入 | 0.14 ms | C10 已判无效 | |

**结论:便宜的结构性改动已经做完了**(#2123、#2216、#2217,以及仍是 draft 的 #1998),
剩下的候选单项都在 0.2 到 0.8 ms 这一档,而且每一个都要真刀真枪的内核工作。

**更要紧的是现在测不动**:Tilde 上同一 build 首尾两次的 A/A 漂移在 0.1 到 1.5 ms 之间,
而剩下的候选单项效应就是 0.2-0.3 ms 量级,**信噪比不够**。要继续往下做,要么拿回 eval-h100
(更安静,而且是参照实现所在的同一环境),要么改用外审建议的配对区块协议、把重复次数提上去。
在此之前再发射单臂 A/B 只是烧卡。

**盘面收尾(2026-09-20 22:30 PT)**:`tilde:/home/guests/zhen/jaxan/runs/20260916-qtts-e2/` 当时 112 GB,
其中 `outputs/*/audio/` 占 79 GB——219 个臂的逐请求音频,属于可再生的中间产物(延迟数字早已归约进各目录的
`summary.json`,音色一致性检查在第十九轮做完)。共享盘当时已用到 99%,所以把这 79 GB 删掉,
保留 `summary.json` 与各 jsonl 清单(219 份,删前删后都点过数),trace 留在 `prof/`(975 MB)。

## 第二十八轮:混合 chunked prefill(`enable_mixed_chunk`)判死(2026-09-21 00:37-01:55 PT)

动机来自第二十五轮的批次统计:rps 20 下 prefill 批次 1277 个、平均每批 1.1 个请求,每六七轮就有一轮专门给一个新请求做 prefill,
那一轮所有在跑的流都不推进。SGLang 的 `enable_mixed_chunk` 让 prefill 拼进 decode 批次一起算,是服务配置项、不改代码,
所以两臂是同一个 binary,只差 yaml 里一个字段。Tilde worker-28,单卡,A6 / MX / A7 各 3 个 seed,后接两个插桩单跑看批次结构。

| 首帧 p50(ms) | A6(默认) | MX(`enable_mixed_chunk: true`) | A7(默认) |
|---|---|---|---|
| rps 1 | 29.8 / 30.2 / 30.5 | 30.1 / 30.6 / 31.6 | 30.0 / 30.2 / 30.3 |
| rps 20 | 39.7 / 40.4 / 40.5 | **2718 / 2748 / 2799** | 40.1 / 40.1 / 40.7 |

**rps 20 下首帧从 40 ms 变成 2.7 秒**,rps 1 不变。插桩臂给出了机制:混合之后 decode 批平均 47.8 个请求、prefill 批 47.6 个
(默认是 11.5 与 1.1),也就是每一轮都把等待队列里的请求全部拼进来、批次顶到 `max_running_requests` 的 48;
`model_runner.execute` 从 6.58 涨到 11.37 ms,发流从 0.47 涨到 3.01 ms,一轮 17.85 ms,比默认的 8.76 慢一倍。
批次变大并没有换来吞吐:同样 60 秒窗口只完成 2200 轮对 9000 轮。在这个负载上,"每轮省一次 prefill 专用迭代"的收益
被"每轮都背着一个满批 prefill"的代价完全压过。

**结论:这个开关对流式 TTS 是负优化,不上;也不必再调 `chunked_prefill_size`**——问题不在 chunk 大小,在于混合模式把
"新请求进批"的节流全部去掉了。A6/A7 的首尾对照相差 0.3 ms,这轮的读数可信。

## 第二十九轮:#2216 的 ASR 对照(2026-09-21 02:13-02:27 PT)

JiaxinD 在 #2216 上指出:同一轮准入的等待对所有带 build 线程池的模型都生效,ASR 的几个模型也在内,不能只测 TTS。
Tilde worker-28,单卡,Qwen3-ASR-1.7B,SeedTTS EN 全量 1088 条,并发 32(CI 用的并发),每个服务跑三遍,
main 首尾各起一次。每个服务的第 0 遍含预热与编译,不计。

| | main | #2216 | main 复测 |
|---|---|---|---|
| 吞吐(samples/s) | 228.8 / 233.3 | 234.3 / 221.8 | 241.7 / 219.6 |
| 延迟 p90(ms) | 184 / 171 | 174 / 179 | 166 / 185 |
| 延迟 p95(ms) | 203 / 194 | 185 / 204 | 179 / 203 |
| corpus WER | 1.21% / 1.20% | 1.21% / 1.22% | 1.20% / 1.23% |

#2216 的每一列都落在两次 main 的区间里:ASR 池的吞吐和尾延迟没有变化,WER 不变。已回复到 PR 并写进正文。

## 第三十轮:#2217 的自适应等待单独消融(2026-09-21 02:42-03:36 PT)

JiaxinD 在 #2217 上要求把"首块编译"和"自适应初始等待"两个改动的贡献拆开。同一份 build,
`stages.vocoder.factory.adaptive_initial_batch_wait` 开(A8、A9,首尾各一次)与关(NA),各 3 个 seed,Tilde worker-27 单卡。
前两次尝试都因为同节点上我自己的 CI 作业的收尾清理误杀而作废(见第二十四轮之后的教训),这次排在独占节点上。

| 首帧 p50(ms) | 开(A8) | 关(NA) | 开(A9) |
|---|---|---|---|
| rps 20 | 38.2 / 38.2 / 38.6 | 38.6 / 38.7 / 38.8 | 37.2 / 37.3 / 37.6 |
| rps 20 可闻 | 39.8 / 39.8 / 40.0 | 40.3 / 40.3 / 40.3 | 38.5 / 39.0 / 39.1 |
| rps 1 | 28.6 / 28.9 / 29.0 | 28.6 / 28.6 / 28.7 | 28.3 / 28.6 / 28.8 |

**自适应等待在这台机器上值 0.5 到 1 ms(rps 20)**,落在两个"开"臂之间 0.9 ms 的漂移里;rps 1 完全一样。
这比第十九轮在 eval-h100 上估的 -2.5 ms 小,与外审当初的判断一致:它的上限就是那 2 ms 的等待。
#2217 单独那 3 到 6 ms 的大头来自首块走编译内核。已回复到 PR 并写进正文。

## 第三十一轮:把 TTS CI 的 qwen3-tts 线搬到 Tilde 上跑(2026-09-21 01:51-03:33 PT)

GPU runner 所在主机连不上,CI 的模型阶段一直排队。按 `.github/workflows/test-tts-ci.yaml` 里原样的命令在 Tilde 上复现:
两张 H100(harness 固定起两个 worker 挂在 Rust router 后面),`pytest tests/test_model/test_tts_ci.py -v -s -x --concurrency 16 --tts-stage <stage>`,
`TTS_CI_MODEL` 选臂。为此在 Tilde 上从源码编了 Rust router(用户级 rustup),提前离线放好 Base / CustomVoice / Qwen3-ASR 三个 checkpoint、
seedtts 与 seedtts-50 数据集、WavLM 相似度权重、UTMOS 权重。三处环境适配,都不进 PR:CustomVoice 缓存补 `refs/main`(手工放的快照离线解析不到仓库名);
preset 的 `startup_timeout` 从 300 改成 900 s(Tilde 冷编译,CI runner 有热缓存);FFmpeg 8.1 共享库(torchaudio 走 torchcodec 读 wav,主机上没有 libavcodec)。

| 臂 | 第 1 阶段(非流式) | 第 2 阶段(流式) |
|---|---|---|
| #2217 qwen3-tts | 速度 ✓、WER ✓ 1.01%、相似度 ✓、UTMOS 待重跑(FFmpeg) | 速度 ✓、流式 WER ✓ 1.01% |
| #2216 qwen3-tts | 速度 ✓、WER ✓ 1.00%、相似度 ✓、UTMOS 待重跑 | 速度 ✓、流式 WER ✓ 1.07% |
| #2094 CustomVoice | 速度 ✓、WER ✓ 1.74%、相似度按设计跳过、UTMOS 待重跑 | 速度 ✓、流式 WER ✓ 1.88% |

途中两个自己造成的坑:CI 作业每阶段收尾按 `nvidia-smi` 全杀 GPU 进程,Tilde 的 Slurm 不隔离设备,把同节点上自己另一个作业的服务杀了两次
(现在只杀带自己 `SLURM_JOB_ID` 的进程);harness 的 8200 固定端口在同节点多作业时冲突(现在每次起服务取空闲端口)。

**UTMOS 重跑(作业 366189,03:33-04:00 PT)**:装上 FFmpeg 8.1 共享库后,三个臂的第 1 阶段全部通过:
#2217 臂 4 passed,#2216 臂 4 passed,#2094 的 CustomVoice 臂 3 passed + 1 skipped(相似度按设计跳过)。
连同前一轮各臂第 2 阶段的 2 passed,**三个 PR 的 qwen3-tts 线全绿**。

## 第三十二轮:在 main + #2216 + #2217 上重做 torch profiler 与 nsys,开下一轮的 tracking issue(2026-09-21 16:43-17:31 PT)

Tilde worker-15,单卡(账号有"每人最多两个节点"的限制,把作业钉到已占用的节点上才排进去),树是 main `9f1260e6` + 两个 PR。
torch profiler 先不带栈采一轮(rps 1 8 秒、rps 20 2 秒),再带 `SGLANG_TORCH_PROFILER_WITH_STACK=1 RECORD_SHAPES=1` 采一轮
(带栈的 trace 有 70 万条 `python_function` 事件,kernel 能映射回 Python 帧,留给候选实现时用),最后 nsys 带 GPU metrics(20 kHz)。
导出用第二十二轮那套"等文件稳定再验 gzip"的办法,四份 torch trace 全部完整。

**decode 步(talker 流,p50)**:rps 1 步长 6.06 = talker 1.99 + predictor 3.81 + 空洞 0.09;rps 20 步长 9.75 = 2.36 + 4.17 + 1.23(p95 12 ms)。
rps 20 的步长比不带 profiler 的 7.6 长,是 `with_stack` 把主机侧放大了;图重放时间是 GPU 侧的,与前几轮一致。

**nsys GPU metrics(整窗均值;GR Active 在 20 kHz 下没有一个采样是 0,所以没有"忙碌子集"可分)**:

| | SM Issue | SMs Active | DRAM 读 | Tensor Active | 在飞 warp |
|---|---|---|---|---|---|
| rps 1 | 1.2% | 7.5% | 7.3% | 0.3% | 2.3% |
| rps 20 | 10.8%(p90 36%) | 35.0%(p90 81%) | 13.4%(p90 33%) | 2.3% | 10.1% |

和 09-14 那份剖面同一个结论:没有任何一项饱和,GPU 发射效率低是因为工作由成千上万个小 kernel 组成;GPC 时钟稳在 1.98 GHz。

**GPU 时间去向**:rps 1 下 predictor 的小 GEMM 占全部 kernel 时间的 35%(11.9 万次、7.4 us),种子采样内核 5.7%。
rps 20 下换了面孔:predictor 更宽的 GEMM 分片 6.7%,然后是两项 rps 1 下看不见的 vocoder 项——
`cudnn nchwToNhwcKernel` 6.3%(2 万次、12.7 us)和 `implicit_convolve_sgemm` 6.1%(396 次、每次 634 us),采样内核 3.1%。
这两项是没编译的 chunk 宽度走的 eager 卷积路径,在 vocoder 的流上;合计 12.4%,比第二十二轮估的 5.6% 大——JiaxinD 在 #2217 上问的
"WARM/WINDOW 宽度仍是 eager"指的正是这条路,它不影响首帧(首块已编译),影响的是连续性与尾部。C11 的定位据此改写。

**产出**:tracking issue **#2294**(挂在 #1754 下面),把 5 个 PR 的现状、这轮剖面、剩余候选(C6/C7/C5/C11)与判死清单写成一页;
CI 阶段 **#2293**(叠在 #2094 上)加了 `tts-stage-latency`,设计经外审修订(见 `docs/reviews/2026-09-21-qwen3-tts-latency-ci-stage.md`)。

## 第三十三轮:按 review 意见改完后的最终对照(2026-09-22 02:50-04:01 PT)

JiaxinD 在 #2216 上要求在"只等队头 build"的新 head 上重测,在 #2217 上建议去掉自适应等待只留首块编译(0.5 到 1 ms 在噪声里,
且每次收批都要拿 `state_lock` 扫一遍)。两条都采纳。Tilde worker-15 单卡,main `2d0b48ff`,每臂 3 个 seed,main 首尾各一次。

| 首帧 p50(ms) | main(M5) | #2216 新 head(C5) | + #2217 只留编译(CE5) | main 复测(M6) |
|---|---|---|---|---|
| rps 1 | 30.8 / 31.2 / 31.2 | 30.6 / 30.8 / 31.0 | 29.4 / 29.4 / 30.0 | 30.8 / 31.5 / 31.9 |
| rps 20 | 50.2 / 50.5 / 50.8 | 42.4 / 42.8 / 43.0 | 41.2 / 41.6 / 41.8 | 50.3 / 50.5 / 50.6 |
| rps 20 可闻 | 51.6 / 51.7 / 51.9 | 43.9 / 44.1 / 44.6 | 42.8 / 43.1 / 43.9 | 51.8 / 52.0 / 52.1 |

**两次 main 只差 0.1 ms,这轮是所有轮里最干净的一次。** #2216 改成只等队头后仍是 -7.7 ms;#2217 去掉自适应等待后单独值
rps 1 -1.5、rps 20 -1.2 ms。两个一起 -8.9 ms(50.5 → 41.6),可闻 -8.6 ms。
与第二十轮"两个 PR 合计 -13"相比小了,差额就是被去掉的自适应等待和那一轮 main 自身更慢的基线,方向一致。
数字已进两个 PR 正文并回复到评论;#2216、#2094 已 approve。

**合并(2026-09-22 04:20 PT)**:#2094 → main `e7168592`,#2216 → main `e57fd94d`,均为 squash、自写正文、JiaxinD 记 Co-authored-by。
#2094 合并前 rebase 过一次:main 在同一批文件里新进了 cosyvoice3 臂,两臂都保留,契约测试从"轮换 == 已校准集合"改成
"轮换只抽注册过的 preset 且不含 CustomVoice"。#2217(只留首块编译)等 review;#2293(延迟阶段)两臂在 Tilde 上各跑一遍全过,
CustomVoice 臂 1 rps 中位 31.8 ms、20 rps 53.5 ms(CI 部署形态:经 router、vocoder 独立进程),Base 臂 79.6 / 160.1 ms。

**#2217 合并(2026-09-22 22:52 PT)**:只留首块宽度编译的版本 → main `a0087317`,JiaxinD 记 Co-authored-by。合并前 GitHub 上的
GPU CI 恢复了(hyper/eval-h100 改由 Radix 平台分配),`pick TTS model` 按 `run-qwen3-tts` 标签选中 qwen3-tts,stage 1 日志里
`TTS_CI_MODEL: qwen3-tts`、加载 `Qwen3-TTS-12Hz-1.7B-Base`,TTS CI 五个 stage 全部 success,确认测的是改动路径。
luojiaxuan 同时授权:之后凡已 approve 的我方 PR 直接合。

## 第三十四轮:C7 已由上游完成;下一步改查"CI 部署形态为什么多花 12 到 15 ms"(2026-09-23 PT,决策记录)

**C7 不用做了**:准备动手时发现 Ratish 的 #2286(09-21 23:50 PT 合入)已经把 predictor 的两个 prologue token 合成一次因果前向,
做法与我计划的一致(对齐参照实现的两 token prefill)。#2294 里标为已由 #2286 完成。

**决策四件套**
- 问题:predictor 侧只剩 C6(采样内核,约 0.3 ms/帧,要保持逐位一致的并列语义)和 C5(约 0.2 ms/帧),下一步做什么。
- 默认:先不做 C6,改做一个不改代码的对照——同一个 main build,在三种部署形态下量首帧:出厂默认(D0)、
  CI 形态即 vocoder 独立进程加显存份额(D1)、历轮 harness 形态 `max_running 48`(D2),出厂默认首尾各一次。
- 理由:#2293 在 CI 形态下量到 CustomVoice 20 rps 首帧 53.5 ms,而同卡同树在 harness 形态下是 38 到 42 ms,
  差 12 到 15 ms,比剩下任何一个 kernel 候选大一个数量级;而 `--vocoder.process vocoder` 正是 CI 和 cookbook 的推荐形态。
  如果差距主要来自 vocoder 独立进程,那才是用户实际能拿到的最大一块。这个对照零代码、单卡约 80 分钟。
- 回滚:这只是测量,不改任何默认;若结果显示差距来自 router 或别处,再回到 C6。
- 外审:这是低成本侦察而非方向性投入,未单独发审;若结果要求改默认部署形态,落地前发审。

## 第三十五轮:vocoder 放独立进程,rps 20 首帧多 16 ms、可闻多 29 ms、卡顿 25 倍(2026-09-23 15:00-16:10 PT)

Tilde 单卡,main `a0087317`(已含 #2216、#2217、#2286、#2311;venv 升到 SGLang 0.5.20),同一 build 四种部署形态,各 3 个 seed。

| rps 20 | D0 出厂默认 | D1 vocoder 独立进程 | D2 历轮 harness | D0b 出厂默认复测 |
|---|---|---|---|---|
| 首帧 p50(ms) | 40.8 / 41.0 / 41.9 | **56.3 / 56.9 / 59.0** | 40.7 / 41.3 / 41.5 | 40.4 / 40.7 / 41.0 |
| 首帧 p95 | 68.6 | **96.6** | 69.2 | 67.7 |
| 可闻 p50 | 42.3 | **71.4** | 42.7 | 42.0 |
| 可闻 p95 | 78.2 | **146.4** | 78.1 | 84.5 |
| underrun | 0 到 0.3% | **4.7 到 6.4%** | 0 到 0.3% | 0 到 0.9% |

rps 1:D0 29.6、D1 32.4、D2 29.8、D0b 29.6 ms。D1 = `--vocoder.process vocoder --tts_engine.gpu_memory_fraction 0.85
--vocoder.gpu_memory_fraction 0.10`,即 CI preset 与 cookbook 基准表用的形态;D0 是 `examples/configs/qwen3_tts_1_7b_customvoice.yaml`
那种什么都不设的出厂默认(三个 stage 同进程,`max_running_requests` 取 #2311 之后的 64)。

**读法**:D0 与 D0b 只差 0.3 ms,读数可信。D2(`max_running 48`)与 D0 一样,说明 48 与 64 这个上限不影响首帧。
**vocoder 独立进程是这轮找到的最大一项**:首帧 +16 ms、可闻 +29 ms、可闻 p95 翻倍、卡顿从 0.2% 到 5% 以上,而 rps 1 只多 2.8 ms。
这解释了 #2293 在 CI 形态下量到的 53.5 ms。

**机制的第一嫌疑是 GPU 时间片**:`mps` 默认 `off`,两个 CUDA context 在同一张卡上不能并发、只能轮流占用,负载越高互相等待越久,
正好是"rps 1 小、rps 20 大"的形状。另一个观察:D1 在预热阶段 codec 状态池打满(64 个槽)、168 个请求退到慢得多的 left-context 解码,
这发生在服务刚起、计量开始之前,不是计量期间卡顿的原因,但说明独立进程的 vocoder 在冷启动时跟不上。

**下一步**:同一 build 加一组 `mps: on` 的独立进程臂,看 MPS 能否把这 16 ms 拿回来(作业已排)。
若能,建议是"独立进程必须配 MPS";若不能,建议是 CustomVoice 的推荐部署保持同进程。
