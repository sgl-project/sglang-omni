# 外审:Qwen3-TTS 增量 codec 路径每 cohort 延迟归因

- 日期:2026-09-06 02:45 PT
- 模型档位:最新旗舰 + 极高推理档(思考 2m 24s,含联网查证 NVIDIA 文档),临时会话
- 送审对象:#1846/#1855(arena 状态槽 + WARM CUDA graph 增量解码)rebase 到 main 后,
  r20 实测 underrun 43-46%(legacy 17%)。探针:每 cohort 18ms = plan 0.7 + gather/
  positions 3.3 + copy-in/replay CPU 2.1 + GPU 等待 8.9 + scatter 1.5 + commit 0.6;
  SM 利用率 60-69%;2 个 follow-up stream + 1 个 initial stream。

## 我送审前的判断

每个 cohort 的 graph 含 8 层 transformer + ~30 个因果卷积 + 52 个状态拷贝,B≈2、T=8
下全是微型 kernel,按 ~10µs/kernel 是 launch-latency 受限;三条流交错不能重叠。补救:
把 cohort 做宽(加大收集窗口;fresh_frames 均匀不会碎片化),可能把 initial 流并入
follow-up worker。

## 它的意见(要点,按截图整理)

开场:**不要现在就加收集窗口。** 新 codec 的算术明显更快;生产退化更像调度/串行化/
编排问题,不是"kernel 太多"。

### (a) 三个最弱点

1. **"~10µs/kernel"这个先验对 replay 的 CUDA graph 不成立。** graph 已去掉绝大部分
   host launch 开销;NVIDIA 文档给的 GPU-resident graph 节点过渡延迟约 0.5-2µs。节点数
   仍可能通过设备侧调度、依赖链、微 kernel 低效伤到性能,但机制不同。
2. **"三条流交错无法重叠"是断言,且有更糟的可能:** 若两个 worker 发射的是**同一个
   `cudaGraphExec_t`**,CUDA 明确串行化——同一个 exec 不能与自己并发,无论从哪条流发。
   若每种形状只有一份 warm exec 被两个 worker 共用,"流并发"从构造上就是假的。另外
   60-69% 的聚合 SM 利用率几乎说明不了 graph 是重叠、串行、卡依赖还是被拉长。
3. **"所以要更宽的 cohort"不从延迟数据里推出来。** 孤立 replay B1 4ms、B8 9ms;生产 B≈2.3
   就已经等 8.9ms——这在喊"生产干扰/排序/排队",不是"B2 本身因为节点多就要 9ms"。更宽
   的 cohort 提高 rows/s,但 B8 不降低 replay 延迟,反而抬高;再叠一层收集延迟。只有当
   降低 replay 频率去掉的排队多于新增的批延迟时才有用。

另一个难看的数字:非 GPU 部分合计约 8.2ms/cohort,其中 gather/positions + replay 侧
CPU + scatter 就 6.9ms。我的方案瞄准的是 18ms 里的另一半。

### (b) 最便宜的证伪

不必先上 Nsight Compute,做一个小 A/B:**在生产进程里用 CUDA event 测 B2/B3 graph
的 replay 耗时**,(1) 只留一条 follow-up 流且压掉 initial 解码,(2) 正常三流负载;同时
记录每个 worker 用的 `cudaGraphExec_t` 身份。若孤立 B2/B3 落在 microbench 曲线附近
(约 4-6ms)、只在其它流活跃时跳到 8.9ms,则"节点数/graph 固有开销"作为退化解释被证伪,
剩下的是干扰、串行化或排队。若两 worker 共用同一 exec handle,甚至不需要 A/B:CUDA
保证这些执行是有序的。若孤立 B2/B3 本身就 8-9ms,微节点 graph 假设存活,该去看 graph
本身。下一层:一次短的 Nsight Systems 抓取(带 graph 节点追踪)看毫秒是真 kernel 还是
空隙、流是否重叠;NVIDIA 警告节点级追踪本身开销大,只作诊断不信其绝对墙钟。

### (c) 它会怎么做

把 GPU 当成**集中调度的资源**,不让两个 follow-up 线程加一个 initial 线程各自往上扔 graph。

- 第一,follow-up 解码走**一个 deadline-aware 的 dispatcher、一条流**。跨 slot 把就绪的
  全部批起来(fresh_frames 均匀)。按播放余量派发:最老的请求接近 underrun 就立刻发;
  有余量就短暂累积到 B4/B8。这样得到批处理而不硬加固定收集税。initial 解码应当做
  admission control 排在流式工作之后,而不是简单"并进 worker";保护已在播放的流比
  制造名义上的流并发更有价值。
- 第二,先判断并发是否真的有用。有用就给每条并发流**各自实例化的 graph exec**;若两份
  graph 提高吞吐但拉长每 cohort 延迟到增加 underrun,退回一份。优化的指标是 deadline
  miss,不是 SM%。
- 第三,攻击 **8.2ms 的 host/编排路径**。arena 实现里 3.3ms gather/positions + 1.5ms
  scatter 太贵:把 slot id/positions/状态寻址推成常驻设备的元数据,让 graph 尽可能直接
  消费 arena 状态,不要为了满足静态 graph 缓冲而把状态来回搬。
- 然后才碰节点数。**52 个状态拷贝**比抽象的"30 个卷积"更可疑:若 profile 显示微串行
  节点主导,把状态更新融进因果卷积的生产者、依赖安全处用环形/原地状态、合并相邻记账
  kernel——这是真正减少 graph 关键路径节点;"加大 batch 直到开销消失"只是摊薄问题。

结语顺序:deadline-aware batching → 砍掉 6-8ms CPU/staging 路径 → 然后才融合 graph 节点。

## 我的处置

**采纳 (a)2 的检查,已经做过:** 第二轮起每个 follow-up worker 一份独立 WARM runner
(commit 21a16bc4),initial 用独立 COLD runner;不存在两 worker 共用一个 exec 的情况。
但它提醒的另一半——"并发是否真的有用"——没有验证过:P4 的 resolve 8.9ms 在 2 worker
下测得,需要 1 worker 对照。

**采纳 (b) 的证伪实验,改用我已有的探针做等价物:** 同一探针 (1) r1 低负载(Talker 与
其它流几乎空闲)、(2) r20 单 follow-up worker、(3) r20 双 worker(已有 P4)。看 resolve
是 4-6ms 还是 8-9ms。已排队。

**采纳 (a)3,撤回"先加收集窗口"的判断;但第五轮 wait sweep 已在跑,数据照收**——它恰好
能回答它说的"降低 replay 频率去掉的排队是否多于批延迟"。

**采纳 (c) 第三条作为下一步代码改动方向:** stage 段 3.3ms 里 arena 的 52 个 index_select
只占 0.24ms(已单测),其余在 `_reserve_slot`/`_stage_decoder_input`/`_screen_out_of_range`
——先把这段再拆开,再决定是"graph 直接消费 arena"还是别的。

**保留意见:** (c) 第一条"单 dispatcher 单流 + 按播放余量派发"就是 #1754 的 T-PR10,
方向一致;但它假设 initial 解码可以排在流式工作之后,这与 TTFA 目标冲突(Nari 26ms 的
差距主要就在首帧),需要用 deadline 而不是固定优先级来裁决,不能简单后置。
