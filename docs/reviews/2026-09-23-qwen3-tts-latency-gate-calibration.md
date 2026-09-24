# 外审:#2293 延迟阶段的 gate 校准(2026-09-23 19:10 PT)

- 通道:网页 ChatGPT 临时会话,模型"最新"(6 Pro),思考 11m 5s。
- 附件:`my-judgment.md`(发审前写下的判断,见文末)、`eval-h100-rounds.txt`(当时 r0-r4)、
  `latency_stage_code.py.txt`(#2293 在 2a6557a7 的延迟相关代码)、`ci-host-customvoice-run.txt`
  (CI 主机那一次 CustomVoice 的输出)、`prior-review-2026-09-21.md`(上一轮外审)。
- 提问:参照主机取哪台;×1.25 还是 worst + M;20 rps p95 要不要 gate;有没有让 gate 失效或可被刷分的地方。

## 回复全文

### Verdict

Use the CI-host reruns as the primary reference. Treat eval-h100 thresholds as a deliberately loose
fallback, not as CI-host calibration. The faster CI observation does not threaten false alarms under an
eval-based ceiling; it gives regressions extra room to pass.

×1.25 is defensible as an explicit "roughly 25% regression" policy. "That constant already exists" is not
a statistical justification. For your stated objective of catching additive +13–16 ms regressions, I would
use explicit, per-arm, per-point margins instead. Your proposed multiplier does not provide that
sensitivity consistently across the two arms.

Yes, add a separately calibrated 20 rps p95 gate. But first close the measurement-contract holes: the
supplied stage does not establish that zero client-slot waits means faithful open-loop dispatch, and it
does not enforce the continuity metrics it prints. Those were already explicit requirements in the earlier
review.

Everything below uses r1–r4 only. You currently have four counted calibration runs, not five; r0 is
explicitly excluded. The eventual fifth run can only maintain or increase the proposed worst-based
thresholds. You have one CI-host CustomVoice observation and no supplied CI-host Base observation.

### 1. Which host should supply the reference?

The host mismatch currently hides regressions; it does not consume your slack. CustomVoice, ms:

| Point | Eval mean of run medians | Eval worst | Observed CI median | Eval worst ×1.25 | Increase allowed from observed CI |
|---|---|---|---|---|---|
| 1 rps | 24.525 | 25.2 | 20.2 | 31.50 | 11.30 ms / 55.9% |
| 20 rps | 37.300 | 37.8 | 34.8 | 47.25 | 12.45 ms / 35.8% |

The difference between the eval average and the observed CI run is 4.325 ms at 1 rps but only 2.5 ms at
20 rps. Therefore: there is no evidence for a universal "subtract 4 ms" correction; there is no basis for
transferring CustomVoice's host difference to Base; one CI observation cannot establish the CI host's
run-to-run distribution.

Your judgment asks whether the cross-host difference could "eat" the roughly 5 ms slack at low load. The
observed difference points the other way. With the slower host supplying the ceiling, it expands the
regression budget. That concern becomes a false-alarm issue only if future CI conditions are slower than
the conditions represented by the calibration.

For scale, using the single CI observation as a reference would give multiplier thresholds of 25.25 ms and
43.50 ms, rather than 31.50 and 47.25 ms (6.25 ms and 3.75 ms tighter). Those are counterfactual
calculations, not a recommendation to calibrate from one run.

What I would do: use a prespecified set of CI-job reruns of the same known-good build to establish both
arms' references, preserving every attempt. Match the eventual gate's server lifecycle, point order, CPU
lane, companion workload, benchmark configuration, and retry semantics. You cannot lease the CI machine,
but the CI job itself is a calibration instrument. Keep eval-h100 as independent evidence about
sensitivity and a second environment. Do not pool the two hosts' observations as if they were
exchangeable. If CI genuinely runs on multiple hardware configurations, either maintain
configuration-specific references or explicitly choose a worst-supported-environment ceiling and accept
the resulting blind spot on faster machines.

Your own config says the latency threshold remains unset until calibrated on the CI host, and the
surrounding threshold documentation warns that references from another topology can either fail builds
unnecessarily or wave regressions through. Using eval-h100 is an intentional exception to that policy, not
compliance with it. An eval-based ceiling can still be useful as a temporary catastrophic-regression
tripwire. Just do not call it a measured upper bound on CI performance.

### 2. ×1.25 versus worst + M

With W the rounded calibration maximum, T× = 1.25W = W + 0.25W and T+ = W + M. ×1.25 is an additive margin
that happens to scale with the baseline. The decision is whether the allowed deterioration should scale
with baseline latency, be expressed in milliseconds, or reflect measured noise.

| Arm / point | Mean run median | Worst W | 1.25W | Headroom | W+5 ms | Headroom |
|---|---|---|---|---|---|---|
| Base, 1 rps | 57.750 | 60.0 | 75.00 | 17.250 | 65.0 | 7.250 |
| Base, 20 rps | 109.425 | 111.4 | 139.25 | 29.825 | 116.4 | 6.975 |
| CustomVoice, 1 rps | 24.525 | 25.2 | 31.50 | 6.975 | 30.2 | 5.675 |
| CustomVoice, 20 rps | 37.300 | 37.8 | 47.25 | 9.950 | 42.8 | 5.500 |

Additive shift needed to fail even from the fastest observed eval run:

| Arm / point | ×1.25 | worst + 5 ms |
|---|---|---|
| Base, 1 rps | 19.30 ms | 9.30 ms |
| Base, 20 rps | 32.15 ms | 9.30 ms |
| CustomVoice, 1 rps | 7.50 ms | 6.20 ms |
| CustomVoice, 20 rps | 10.75 ms | 6.30 ms |

The major error in your reasoning is treating "about 35 ms" as the relevant baseline for both arms. Base's
loaded median is roughly 109 ms, so its multiplier margin above the worst is 27.85 ms. An additive +13–16
ms penalty at Base's loaded point passes in every supplied counted run (111.4 + 16 = 127.4 < 139.25). For
CustomVoice on the observed CI run, +13 ms gives 47.8 ms against 47.25 ms: only 0.55 ms of clearance. The
current data do not justify claiming reliable +13 ms detection on CI.

False alarms: under iid continuous run statistics, the next run exceeds the maximum of five calibration
runs with probability 1/6 = 16.7% (1/5 with four). That is for an unpadded maximum; a margin reduces it by
a distribution-dependent amount. Two independent unpadded gates: 30.6% that at least one exceeds; four:
51.8%. Two passing holdouts are a smoke test: zero failures in m jobs gives a one-sided 95% upper bound of
1 - 0.05^(1/m), 77.6% for m = 2; 299 jobs to get below 1%.

Run-median sample SDs (r1–r4): Base 2.266 / 1.767 ms, CustomVoice 0.512 / 0.560 ms. Normal-model one-sided
99% prediction bounds for one future run: 69.26, 118.40, 27.13, 40.14 ms. Worst + 5 ms lies below that
bound for both Base points but above it for both CustomVoice points (a model-dependent diagnostic only).

My choice: explicit per-arm, per-point additive margins, chosen against the actual CI run distribution and
the required regression size. Do not replace an unjustified universal 25% with an unjustified universal
5 ms. False-alarm probability = 1 - F0(T); power against additive Δ = 1 - F0(T - Δ). When no threshold
satisfies both, improve measurement stability or change the job statistic. Test with injected first-audio
delays, not just passing A/A runs.

### 3. Should 20 rps p95 be gated?

Yes, with its own reference and margin. At 1,088 requests about 54 observations lie above p95; at 60,
about three. Loaded p95 r1–r4: Base 178.9–189.4 ms (SD 4.708, worst ×1.25 = 236.75); CustomVoice 49.8–54.1
ms (SD 1.828, worst ×1.25 = 67.625). The reason to add p95 is coverage: a regression affecting the slowest
10% of requests can leave the median unchanged. Keep 1 rps p95 report-only initially. The current schema
has only `ttfp_median_max_s`; adding a p95 gate requires an actual field and comparison.

### 4. What makes the gate invalid or gameable?

A. "Calibrated" can still mean "silently not gated": `if latency.calibrated and point.ttfp_median_max_s is
not None` skips a calibrated point with a missing threshold. Add a contract test requiring finite positive
thresholds for every point of a calibrated preset; a missing threshold must fail configuration validation.
Apply slack exactly once.

B. Zero slot waits does not prove faithful open-loop arrivals: no scheduled arrival, dispatch lateness,
connection-pool wait or scheduler drift is checked. Retain a seeded arrival schedule and per-request
scheduled/send/first-audio/completion timestamps; impose a dispatch-fidelity contract.

C. A tiny early audio chunk followed by a stall can pass: first-chunk duration and continuity are printed,
not checked. Define "playable" in audio-duration terms and add a coarse continuity guard (calibrated, not
"exactly perfect": the CI 1 rps run already has C50 = C100 = 98.33%).

D. Complete requests do not establish complete timing coverage; negative medians would pass
`median <= threshold`.

E. The admission cap is reconstructed from `_PRESET.worker_extra_args` only, while the worker launches with
`TTS_WORKER_EXTRA_ARGS` plus the preset args. Take it from the same resolved configuration used to launch.

F. No explicit arrival seed, dataset revision or request-order contract in the excerpt. The two points run
sequentially on the same server; calibration and CI must preserve that. 61 dispatches for 60 requests is
consistent with one warm-up request.

G. Five calibration runs plus two holdouts is reasonable if the split is fixed beforehand; raising a
threshold because of a holdout turns it into tuning data. Preserve retry attempts; do not accept "any
passing attempt".

What I would approve: CI-host references for both arms, single-run median gates at both points, and a
separately calibrated loaded-p95 gate, with explicit per-point margins and fail-closed checks; validated
with unchanged-build runs and injected failures (+13/+16 ms uniform, tail-only delay, delayed dispatch,
tiny-first-chunk stall). I would not approve the claim that eval-worst ×1.25 reliably catches +13–16 ms.
The data support a coarse tripwire.

## 我的判断(发审前写下,原文)

见同目录下 run doc 第三十六轮"发审前的判断"一节;要点:沿用 ×1.25、只 gate 中位数、eval-h100 作校准主机、
CI 主机做迁移验证;最不确定的是 ×1.25 与加性余量之争以及单次中位数的噪声。

## 采纳情况

见 `docs/benchmarks/qwen3_tts_r20_vocoder_study.md` 第三十六轮的"外审处理"。

后续(2026-09-24 00:10 PT):第一条(参照取自 CI runner 重跑)起初没采纳,Base 臂在 CI 上的第一次运行就因 20 rps p95 256.4 ms
超过按 eval-h100 定的 236.8 ms 而误报,随后照这条改了:每臂在 runner 上重跑 5 次作参照,Base 的 20 rps p95 因双峰只打印。
它说的"不能把 CustomVoice 的主机差迁移到 Base"被实测证实。
