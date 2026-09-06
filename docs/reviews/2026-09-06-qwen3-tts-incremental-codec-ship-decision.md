# 外审:Qwen3-TTS 增量 codec 路径的发布决策与剩余 5ms 归因

- 日期:2026-09-06 04:20 PT
- 模型档位:最新旗舰 + 极高推理档(思考 1m 16s,联网查 NVIDIA MPS / 流优先级文档),临时会话
- 送审对象:增量路径(arena + WARM 图 + 编译内核)r20 underrun 2.7% 对 legacy 5.4%;
  探针显示 vocoder 流每 cohort 只执行 ~2ms 却等 7.5ms(r10 相同);进程内时间线显示
  Talker 与 vocoder 同进程同上下文;真实码数值对照通过;正在跑的两条方案:提前发射、
  vocoder 独立进程 + MPS;之后是融合。问:哪条真打到那 5ms、最便宜的判别实验、默认发什么。

## 我送审前的判断

剩余 5ms 是 vocoder 的 356 个串行微节点排在 Talker 占满 SM 的 kernel 后面;提前发射
和 MPS 两条都值得跑;编译内核 + 增量路径应翻成默认。

## 它的意见(要点)

### (a) 哪条真的攻击那 ~5ms

根因是 **GPU 调度干扰**:~2ms 真实 vocoder 工作、~7.5ms 墙钟等待、r10 不变——缺的 5.5ms
不是 vocoder 算术;CUDA 流优先级只是调度提示,**不能抢占正在跑的 kernel**(NVIDIA 文档明示)。

- **提前发射**:有用,但**不是那 5ms 的直接修复**。它消掉 2-5ms 的 CPU 供给空洞,让 vocoder
  工作更早排队,间接更容易赢下一次调度机会;便宜合理,该做;但若有人说它解释/修复了 7.5ms
  的等待,那是 cargo cult。r10 结果尤其说明 CPU 规划不是主因。
- **独立进程 + 原生 MPS**:也**不自动是修复**。同上下文里独立流本来就能并发执行;把 vocoder
  搬过进程边界既不产生抢占,也不能阻止一个占满 SM 的 Talker kernel 挡住它。
- **MPS + 真正的资源划分**:**这才攻击问题**。限制 Talker 可达的 SM,或给 vocoder 一块 Talker
  用不到的空间容量;NVIDIA 把 MPS 资源划分明确描述为 QoS/干扰控制机制,当前 MPS 在 Hopper
  上支持静态 SM 分区。
- **更多融合**:次要但真实。356 个串行 kernel 减少到更少更胖的 kernel,减少 Talker 能插进
  vocoder 相邻节点之间的依赖边界数;但孤立 replay 已经 1.8ms,别为了解释 7.5ms 的延迟去
  猛优化 1.8ms 的计算。

一句话:提前发射打空洞;分区 MPS 打干扰;原生 MPS 大多是仪式;融合是调度修好之后的清理。

### (b) 最便宜的判别实验

别先建 MPS。做一次短 A/B:**把 Talker 的发射门住、让完全相同的 vocoder 流量继续跑**,最好
交替 Talker-on/off 窗口以排除热与无关负载。同时在 vocoder graph 前后紧贴放 CUDA start/end
事件:Talker on 时 host 完成 ~7.5ms;Talker off 若掉到 ~2ms,即证明是 Talker 干扰,不必碰
vocoder;若 CUDA 事件 start→end 在 Talker on 时也 ~7ms,Talker 是**插在 356 个串行节点之间**;
若 start→end 保持 ~2ms 而 host 入队→完成 ~7.5ms,则大部分延迟在**第一个 graph 节点之前**。
这一个测量比 MPS 实现实验告诉你的更多。更狠的确认:临时用一个可控时长、占满 SM 的
kernel 替代 Talker,若 vocoder 延迟≈该 kernel 的剩余寿命,机制被人工复现。

### (c) 会发什么

**现在就把新增量路径发成默认,legacy 放回滚开关后面。** 2.7% 对 5.4% 是 2× 改善,代价是
首帧 p50 差 4ms——除非 110ms 撞上硬 SLO,这是轻松的取舍;数值差异在真实码结果与无接缝
伪影的前提下不构成阻塞;保留音频回归套件,但别追 bit 等价。
接着发**有界的单 cohort 提前发射**——只要实验在首帧与内存上中性或更好;低复杂度,让流水线
结构更健康。然后若 Talker-off 测试让 7.5ms 崩塌,走**显式 GPU 资源隔离**而不是"打开 MPS":
在 H200(原文;实际主机是 H100 80GB)上给延迟敏感的 vocoder 一个小静态 SM 分区,其余给 Talker,对分区大小扫 Talker 吞吐
与 vocoder p95/p99;静态分区才是真正的空间隔离,普通的 active-thread 限制是软 QoS,NVIDIA
明确说它本身不预留资源。
顺序:**新路径 → 提前发射 → 证明 Talker 因果 → SM 隔离 → 最后才更多融合。**
"356 个 kernel 难看,但 1.8ms 的孤立 replay 说明火不在你这边,在 Talker。"

## 我的处置

**采纳 (c) 的发布顺序,并已有实测支撑**:提前发射臂 la-w4-r24 underrun 3.67%、首帧 p50 108ms
(基线 k-w4-r24 2.73% / 110ms,单 seed 噪声约 ±1%),(1,2,4) ramp 下 22.6% → 17.1%——中性
或更好,发。独立进程无 MPS 4.77% / 99ms,独立进程 + 原生 MPS 2.39% / 93ms(但 p99 542ms
有尾巴,且收尾时 MPS 健康检查报错)——与它"原生 MPS 大多是仪式"的判断一致,不作为默认。

**采纳 (b),用现成探针做**:在 runner 的 replay 前后放 CUDA 事件,把 start→end 与 host 等待
一起记;Talker-off 窗口用"只跑 vocoder 流量"的方式近似(把 Talker 压掉在生产进程里做不到,
改用 r1 + 人工微 kernel 负载的对照;或直接看 start→end 与 host 等待的差)。

**保留意见**:它假定"分区 MPS"在我们的运行时可直接配置;仓库的 `--mps` 只有 off/on/auto,
SM 分区需要额外的 `CUDA_MPS_ACTIVE_THREAD_PERCENTAGE` 或 Hopper 静态分区接口,属于新工程量,
排在发布之后。
