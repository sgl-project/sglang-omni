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
- 实测臂(r20,slots 128,1 worker):`l-w1` / `i-off`(增量 eager)/ `i-on`(WARM 图)/
  `i-on-cold`(另捕获 COLD 1、2 帧)/ `i-on-r24`(WARM 图 + ramp (2,4))。
  **结果待填。**

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
