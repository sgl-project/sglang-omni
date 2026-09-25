# 外审:#2293 正式校准后,Base 20 rps p95 与 c50 要不要 gate(2026-09-24 19:40 PT)

- 通道:网页 ChatGPT 临时会话,模型"最新"(6 Pro,推理档拉满),思考 6m 39s。
- 附件:`judgment.md`(发审前写下的判断与数据,见下)、`calibration_rounds_so_far.json`(正式校准 Base r1-r4)、
  `test_tts_latency_ci.py`、`tts_ci_config.py`(#2293 在 8521cf68,参照值仍是校准前的暂定值)、
  `runner_arrivals_diff.txt`(固定种子的开环到达与发送延迟记录)、`pr2293_body_before_calibration.md`。
- 背景:正式校准用原生 `tune.py` 跑在 CI 主机 host-85-234-79-221 的 lane 2-15,66-79(GPU 0,1),
  但此刻 CI runner 都在 eval-h100(8 卡全被 CI 占着,4 张在跑),221 上除本任务外空闲。

## 发审前我的判断

对所有指标照工具的 worst-of-5 × 1.25 落 gate,包括 Base 20 rps p95(约 185 × 1.25 = 231 ms)。理由:旧的
runner 数据测的是另一个量(到达时间没固定种子,首帧从发送时刻算),不能直接证明新量也是双峰;手册规定
从校准 lane 取最差值;临时放宽是启发式。然后在真实 CI runner 上每个 arm 至少跑 3 次验证,有一次越线就把
那一项改成只打印,不另造余量。备选:一开始就让 Base 20 rps p95 只打印。

## 回复全文

### 1. Is the plan sound, or should the calibration method change?

I would change the method before enabling the Base 20 rps p95 gate. The idle-host measurements are useful
candidate references, but they do not establish that the gate works on the environment that will enforce
it. I agree with keeping the seeded arrivals, planned-arrival timing, and the prohibition on widening a
threshold merely to make validation pass.

The four available Base rounds are tight: p95 is 176.0–185.2 ms, a 5.2% range relative to the minimum,
while the loaded median is 102.9–103.7 ms. Using those four rounds, the prospective p95 threshold is
exactly 185.2 × 1.25 = 231.5 ms. But round five is not supplied, and these measurements are from idle
host-221, while the active CI runners are on another host with four GPUs busy. This demonstrates
repeatability in the calibration environment, not comparability with active CI.

The strongest evidence against assuming transfer is in your earlier PR description: eight runs per arm on
a separate H100 produced Base p95 values of 174–194 ms, despite matching image and dependency identifiers;
the first actual CI run then reached 256 ms and failed. That is a previous instance of the exact
calibration-transfer problem under review. It does not prove that the revised runner still has the
problem, but it means another tight off-runner series is not sufficient reassurance.

You are right that the older data do not establish bimodality of the new measurement. Unseeded versus
seeded arrivals, send-time versus planned-arrival timing, and idle versus busy execution changed together.
The supplied observations cannot identify which change explains the difference. However, "different
measurement" is a reason not to pool the series, not a reason to dismiss the environment warning. Also,
holding a request's execution timestamps fixed, measuring from an earlier planned arrival adds dispatch
delay; it does not make that request's latency smaller. The clock change alone cannot explain away the
older large tails.

What I would do instead: collect the five calibration rounds per arm using the current code and seed on
an active CI runner lane while the other lanes run their normal jobs. Either restrict subsequent latency
jobs to that calibrated lane/resource configuration, or include the other eligible lanes in explicitly
labeled qualification runs. Keep the idle-host results and old-code results as separate series. The old
runs already span different lane cpusets, so "an H100 CI lane" is not a sufficiently specific execution
contract.

I would retain worst-of-five × 1.25 as the existing engineering rule for that matched environment, not
replace it with a fitted tail statistic. But I would not discard an otherwise valid, ordinary-co-load run
merely because its latency triggers the "destructive" filter. As described, rejection uses MAD and a
percentage gap; that can remove precisely the slow operating conditions the CI gate needs to tolerate. If
such conditions are supposed to be excluded, exclude them through the runner's resource contract, not
retrospectively because their p95 is inconvenient. The calibration tool implementation is not attached,
so I cannot verify whether it already makes that distinction.

Until that matched calibration exists, I would keep Base 20 rps p95 print-only, with 231.5 ms displayed as
a provisional comparator. That agrees with your alternative operationally, but not with its justification:
the reason is unresolved environment transfer, not established bimodality.

### 2. Is "print-only if a validation run fails" principled, or should there be a tail model?

Temporary quarantine is principled; automatic retirement of the metric after one exceedance is not. A tail
model fitted to these rounds would not fix this.

I disagree with the inference in "a gate that fails on an unchanged build is noise, not a regression
signal." One exceedance establishes a baseline exceedance in that setup. It does not establish that p95 is
inherently unusable: the reference environment may be wrong, the execution may violate the intended
resource contract, or the threshold may have an unacceptable, but measurable, baseline failure rate. Your
proposed rule skips that distinction and removes coverage immediately.

My replacement rule would be: a validation exceedance blocks promotion and triggers classification of
that run, while the metric remains visible. If the run violated independently specified execution checks,
fix that problem and repeat qualification, preserving the failed artifact. If it was a valid run under
supported CI co-load, retain it as evidence about the target environment and revisit the representative
calibration. Do not drop it as an outlier, expand the margin on the spot, or conclude that the metric must
permanently disappear.

Three passing runs per arm are a smoke test, not a convincing flakiness qualification. Under an idealized
independent-run model, observing zero failures in three held-out runs gives a one-sided 95% upper bound on
the failure probability of p_upper = 1 − 0.05^(1/3) = 63.2%. For scale, 59 independent zero-failure runs
would bring that bound just below 5%. These are mathematical bounds under the stated assumptions, not
estimates of your actual failure rate. They show how little the proposed minimum of three runs
establishes; results from the other arm cannot simply be added to the same sample.

I would not fit a run-to-run tail model to the four current Base rounds, or combine them with the five
old-code rounds to make the sample appear larger. You have four observations of the current run-level p95
in one environment. The 1,088 requests within each run help calculate that run's p95; they are not 1,088
independent observations of CI run-to-run variation. Worst-of-five plus 25% slack likewise does not
identify the false-failure probability achieved by that margin.

There is also a concrete cost to your fallback: the median gate would not preserve the tail coverage you
removed. The available rounds imply a loaded-median gate of 103.7 × 1.25, rounded by the code, = 129.6 ms.
Every older loaded median, including 122.1 ms and 112.9 ms in the two high-tail runs, would pass it. "Their
medians were also elevated" therefore does not mean those episodes would still be caught after disabling
p95.

My decision would be: qualify a representative worst-of-five threshold first, then assess baseline
exceedances over a fixed, disclosed validation campaign. If supported co-load makes an absolute p95 gate
too variable to retain useful sensitivity, change the execution isolation or explicitly ship without that
coverage. Do not manufacture confidence by fitting a tail distribution to this small, mismatched sample.

### 3. What in the test or config makes p95 or c50 fragile?

**The c50 slack makes the gate largely insensitive.** This is the most consequential numerical issue.
`_continuity_gate` multiplies the success percentage by 0.75. The worst current observed c50 is 99.36%, so
applying your plan produces a c50 minimum of 74.52%: an allowed bad-stream share of 25.48% versus the
observed 0.64%, about 39.8 times the observed bad-stream share. Conditional on all 1,088 requests being
continuity-eligible, 277 bad streams would still pass, versus approximately seven at the observed worst
reference. For comparison, allowing 25% more bad streams would produce 100 − 1.25 × (100 − 99.36) =
99.20%, not 74.52%; its run-to-run stability has not been established. My change: set `c50_min_pct=None`
until eligible-stream and bad-stream counts are recorded and the permissible increase in bad streams is
explicitly chosen and qualified.

**c50's denominator is printed, but not protected.** The test checks completion, success, non-empty audio,
and zero client-slot waits. A successful one-chunk stream satisfies its audio check. It prints
`playback_continuity_requests` and `playback_continuity_na_requests`, but does not require a minimum
eligible denominator before checking c50. (It ran the validation helper on a synthetic result with 1,087
single-chunk streams and one multi-chunk stream: no failure, and a c50 of 100% passes.)

**Seeded planned arrivals do not guarantee identical actual dispatch.** Overdue requests are dispatched as
the loop catches up, so the comment that all remaining spread "comes from the server" is too strong.
`client_slot_waits == 0` does not exclude a late event loop. Retain planned-arrival latency as the gate
metric and inspect request-level dispatch delay on validation exceedances; do not subtract lateness to
rescue a failure or automatically reject high-lateness runs.

**The supplied config enforces different thresholds from the proposed ones.** The attached Base p95
reference is still 0.3082 s (gate 385.2 ms), and both c50 references are 95.0% (gates 71.25%). The older PR
description says Base p95 is print-only and continuity is not gated; reconcile it before review.

**"Disable that single metric" is not implemented for every metric.** p95 and c50 fields accept `None`;
the median comparison is unconditional when `latency.calibrated` is true. For a Base-tail quarantine, set
only the 20 rps `ttfp_p95_max_s` to `None`.

**The reducers are not attached.** Required check: a request planned at 0 ms, sent at 30 ms, first audio at
130 ms must contribute 130 ms; aggregate request-wise arrival latencies before taking p95.

**Bottom line:** keep the new measurement design, but do the deciding calibration on active CI. Keep Base
p95 provisional rather than declaring it either safe or inherently noisy. Separately, do not ship c50 with
the current success-percentage slack and unguarded denominator.

## 采纳情况(2026-09-24 20:20 PT)

**它同意的**:保留固定种子的开环到达、从计划到达时刻计时;不为让验证通过而临时放宽阈值;在匹配的环境里继续用
worst-of-5 × 1.25,不对 4 到 5 个点拟合尾部模型。

**它反对、我改判的**:
- Base 20 rps p95 不设 gate,只打印。我原计划先上 gate 再用 CI 验证。它指出 eval-h100 那次正是同一个指标
  "空闲机器上标得很紧、第一次 CI 就 256 ms 误报",所以空闲 lane 的五轮只证明可重复、不证明能迁移;并且 3 次 CI
  验证全过也只能把误报率上界压到 63%,算不上验证。我接受:这一项等在 CI 共载条件下重新标定再上 gate。
- c50 不设 gate。`_continuity_gate` 用 × 0.75 乘成功率,worst 99.36% 得到 74.52%,等于允许坏流从约 7 条涨到约 277 条,
  门槛形同虚设;这是它指出的最重要的数值问题,我之前没看出来。改成"坏流占比 × 1.25"又会被每轮 3 到 7 条的计数噪声打爆
  (按 Poisson(5) 算,一次干净运行超过 8 条的概率约 7%)。所以两种写法都不对,c50 只打印,计数感知的 gate 记为后续。
  工具里对应的 `c50_pct` 指标一并删除。
- "验证越线就永久改成只打印"这条规则它不同意:一次越线应当阻止上线并归类原因,而不是自动撤掉覆盖。现在没有
  依赖这条规则的 gate 了(Base p95 一开始就不上),这条留作以后 CustomVoice p95 若在 CI 上越线时的处置:
  先归类(执行条件是否合规、是否 CI 共载),再决定重标还是隔离,不当场放宽。

**它反对、但我保留原判的**:
- CustomVoice 20 rps p95 照样 gate(48.7 → 60.9 ms)。它的迁移担忧同样适用,但这一项有反证:此前 5 次 CI runner
  重跑(旧代码)是 48.9 到 59.0 ms,全在 60.9 之内;CustomVoice 也不在 TTS CI 的随机轮换里,只有打了
  `run-qwen3-tts-custom-voice` 标签的 PR 才跑,误报的波及面小。若 CI 上越线,按上一条先归类。
- 四个中位数照样 gate:runner 历史全在新 gate 之内(Base 20 rps 历史最高 122.1 对 gate 130.1)。

**它提出、已核实无问题的**:到达时刻口径是逐请求先加发送延迟再取分位数(`benchmarks/metrics/performance.py`
里 `o.audio_ttfp_s + o.dispatch_lateness_s` 组成列表后才 `np.percentile`),不是两个分位数相加。c50 的分母每轮都是
1088(`playback_continuity_na_requests` 为 0),它构造的"1087 条单块流"情形在实测里没有出现;c50 不 gate 后也不再
影响判定。
