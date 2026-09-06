# 外审:第十二轮方向(code predictor 编译 / prefill 完整图 / 准入时机)

- 日期:2026-09-06 14:40 PT
- 模型档位:最新旗舰 + 极高推理档,临时会话
- 送审对象:r1 首帧 p50 36ms 的账(preprocessing 2 + 准入 3 + prefill 12 + 第二帧 7 + vocoder 4.6 +
  hop/HTTP ~7),目标 26ms;r20 56ms 中多出的 14ms 是准入等 decode 步。三个候选:
  (A) torch.compile code predictor 子步(1371 kernel / 4.04ms,GEMM 255 个占 2.1ms,其余 ~1100 个
  非 GEMM kernel);(B) 小 token 桶用完整 prefill 图替代 breakable(6.2 → ~2.5ms,只影响首帧);
  (C) 在 scheduler 等在飞步的期间做准入(只在负载下有用)。

## 我送审前的判断

顺序 A → B → C。A 对每一步都有效(首帧里 predictor 出现两次,共 ~9ms;每 decode 步 4.4ms 里
非 GEMM 部分约 1.9ms 可压),且手法在 vocoder 上验证过;B 只救首帧 3-4ms 且 Qwen3-TTS 的批策略
只放行 breakable,要动 model runner;C 的收益只在 r20,且混合 chunked prefill 刚证明 scheduler
改动风险高。A 的风险是采样路径(seeded top-k/top-p)与 dynamo 的相容、以及数值/种子一致性。

## 它的意见(要点;思考 3m 44s,联网查 NVIDIA / PyTorch / FlashInfer 文档)

**总判断:按我的账,A+B 加起来也到不了 26ms。** r1 关键路径 35.6ms:B 省 3.7;A 在首帧前出现
两次(prefill 后的 predictor、第二帧的 predictor),4.6/4.4 → 2.5-3.0 共省 3-4ms;A+B 落在
27.9-28.9ms。

### (1) 顺序
- 低负载目标 26ms:**B → A → C**。B 是已测出的 6.2 → ~2.5ms、直接砍目标指标、几个固定 prompt 桶
  比"含 RNG 的编译捕获"干净得多;A 对每帧都有效(predictor 占普通 7ms 步的 ~63%),是更好的
  **容量**优化,若更看重并发就 A、B 对调;C 是单独的负载修复,但要先确认那 14ms 是"host scheduler
  在 cudaStreamSynchronize 返回前没注意到到达"(C 有效)还是"H100 真被上一步占着"(提前准入不会
  让 GPU 提前空出来)——**分别打四个时间戳:到达 → scheduler 注意到 → GPU 工作入队 → GPU 工作
  真正开始**。C 的延伸:首帧关键工作立刻入队到更高优先级流(predictor 全是小 kernel,调度边界多),
  但优先级只是提示、不能抢占正在跑的大 GEMM。20 rps 下顺序变为 C → A → B(前提是 trace 证实
  C 是真的调度气泡)。

### (2) A 的坑
- 自回归本身不可怕(固定 17 步、采样 token 留在 GPU 上喂下一步可以捕获);怕的是 RNG 归属、
  数据依赖的 Python、动态采样形状、嵌套图机制。
- **不要在 CUDA capture 里第一次触发 Inductor 编译**(PyTorch 有过 capture 期间首次编译使捕获
  失效的问题):先编译并 warm 所有静态路径,**关掉 Inductor 自己的 cudagraph 层**
  (`max-autotune-no-cudagraphs` 一类),再把已编译 kernel 捕进外层图。结构:
  编译的固定形状 predictor cell → 采样 kernel → 静态 token/KV buffer,17 步静态展开进外层图。
- 具体禁忌:任何 `.item()`/CPU 侧 top-p 决策;top-p 不能生成变长张量(sort→cumsum→mask→x[mask]
  →multinomial 的存活词表大小是数据依赖的),用 −inf 掩码保持定宽或用专用采样器;Python 值的
  top_k/temperature/mode 会触发 guard/recompile,要么分桶要么做成设备输入;静态存储要真静态
  (不要每个量化器 cat KV,预分配 17 个位置);**不要指望 seed 相同就 bit 级等价**——fusion 改变
  reduction/softmax 细节,概率的微小差异会改变随机结果,测编译实现的自一致性和与旧实现的
  分布/质量对比。
- RNG:CUDA graph 里允许 RNG 操作,PyTorch 有 graph-safe 的 generator 状态 API;但共享全局
  generator 对请求级可复现是错的(A 的输出会依赖 B/C 是否先消耗了 RNG)——用每请求 RNG 状态,
  或更好:以 (request_seed, frame_index, quantizer_index) 计数的显式方案,seed/offset 作为静态
  张量输入。FlashInfer 有直接从 logits 融合的 top-k/top-p 采样,接受 generator 或显式 seed/offset,
  先试融合采样器再让 Inductor 理解手写采样栈。
- **别高估 A 的天花板**:4.04ms 里 GEMM 2.1ms 已是地板,到 2.5ms 意味着几乎消灭剩下的 1.94ms;
  按 ~3ms 规划。开发期用 `fullgraph=True`,让 graph break 大声失败。

### (3) 我漏掉的更便宜的杠杆
- **最大的一项:第二帧的启动税(~7ms)**——vocoder 拒绝从一个 codec 帧起播。若"需要两帧"是实现/
  分块要求而非模型硬约束:左/右补一帧、复制首帧、合成更短的首块 PCM、或做一条单帧 warm-start
  路径再切到常规两帧状态。first playable 只需要最初几毫秒的 PCM,不必与稳态同块长。若可行:
  35.6 − 7.0 = 28.6,再加 B ≈ **24.9ms**,比 predictor 的编译苦工划算得多。
- 第二项:~7ms 的 hop/HTTP。A+B 合计只值 6.7-7.7ms,而模型之外有同样多的时间;A+B 之后只再需
  2-3ms:少一跳代理、去掉请求/响应缓冲、连接保温、流式端点同机、首批 PCM 字节立即 flush。
- 检查自己账里没解释的胶水:prefill 6.2+4.6=10.8 却报 12(差 1.2);普通帧 2.0+4.4=6.4 却报 7
  (差 0.6)——首帧前约 1.8ms 的图间发射/同步/拷贝/scheduler 胶水;若 talker 采样、predictor 调用、
  静态 buffer 拷贝能缝进一张复合图,不碰 transformer kernel 也能收回一部分。
- 它的路线图:**单帧 vocoder bootstrap / 网络与胶水 → B → A,C 作为负载/SLO 修复并行做**;
  不要做的事:花一周把 predictor 从 4.4 压到 2.5,然后发现端点仍然量出 28ms。

## 我的处置

- **采纳,立即做:单帧 bootstrap**。"需要两帧"确实是我们自己的 `suppress_bootstrap_silence`
  (解码 2 帧、扣掉首帧静音以保持播放余量)。它牺牲 first playable 换"可闻 TTFA 不变 + 余量不变"。
  以 Nari 对标的口径是 first playable,先做配置 A/B(`suppress_bootstrap_silence: false`)量
  r1 首帧、r20 underrun 与可闻 TTFA,再决定是"关掉"还是做它建议的单帧 warm-start 路径。
- **采纳:B 先于 A**。B 需要改 model runner(Qwen3-TTS 批策略只放行 breakable),排第十三轮。
- **采纳:C 先打四个时间戳再动**;A 按 ~3ms 规划、先编译后捕获、关 Inductor cudagraph、
  FlashInfer 融合采样、每请求 RNG——记入 #1754 的设计前提。
- **采纳:量 hop/HTTP 的 7ms**,用事件记录器对齐客户端时间戳;胶水 1.8ms 记为 A 的一部分。
- 无保留意见:它的算术与我的账一致,结论比我送审前的判断更硬(我把 A 排第一是错的)。
