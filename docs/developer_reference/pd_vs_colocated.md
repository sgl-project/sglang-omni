# PD versus colocated at equal GPU count

At equal GPU count on Qwen3-Omni thinker, two colocated replicas deliver higher
throughput than one prefill/decode pair on both text and image workloads. PD's
measurable win is that it removes prefill/decode interference almost entirely.
It pays for that by leaving the prefill GPU about 90% idle, because prefill is a
small share of this workload's GPU work.

Read every number against the A/A band in the first section. Differences smaller
than the band are noise.

## Setup

Two H200s, the same two cards for both arms, one pinned tree for both.

| | Arm C | Arm PD |
| --- | --- | --- |
| Shape | 2 colocated replicas, one per GPU, load split evenly | 1 prefill half + 1 decode half |
| CUDA graph | on | on |
| `mem_fraction_static` | 0.87 | 0.72 |
| KV pool (tokens) | 677,613 / 677,542 | 472,030 / 468,068 |

Text prompts measure 42-44 tokens. Image prompts measure about 6127 tokens
(2496x2496 gives 6084 vision tokens). Open-loop Poisson arrivals, 90 s per point,
3 repeats, identical offered rates in both arms.

The arms differ in `mem_fraction_static`, so this is not a flag-matched
comparison. PD's prefill card additionally holds the `mm_aggregate` CUDA-IPC
relay pool and encoder activations, and it ran out of memory at 0.87 and at 0.80.
PD text throughput is identical across 0.87, 0.80 and 0.72 (8.14/8.14/8.14 at
offered 8; 10.51/10.59/10.61 at 16; 10.65/10.55/10.60 at 24), which controls the
text arm. It does not control the image arm, where the memory pressure occurred.

At equal GPU count PD runs one encoder pipeline against colocated's two, because
a colocated replica is a whole pipeline including its encoder. Moving encoders
between the prefill and decode cards relocates that load without changing the
count. This is an uncontrolled confound on the image arm.

### How the numbers are computed

`achieved_rate_rps` is completed requests divided by wall time, where wall spans
the first request sent to the last completion. Arms drain after the offering
window closes, so completions divided by a nominal duration overstates an arm
that queues: PD at offered 44 completes 1672 requests over 157 s, which is 10.7
rps, not the 18.6 that dividing by 90 would give.

SLO attainment counts every offered request. A refused request meets no SLO. Raw
TTFT percentiles are conditioned on admitted requests and are printed with the
admission rate, because computing attainment over admitted requests rewards an
arm for shedding load.

## A/A band

Arm C pass a against pass b, identical configuration, four matched points.

| Metric | Median | Max |
| --- | --- | --- |
| achieved rps | 0.59% | 1.16% |
| prefill-conditioned ITL ratio | 2.04% | 3.76% |
| ITL p50 | 3.40% | 4.08% |
| p95 TTFT | 7.49% | 67.59% |
| goodput | 11.86% | 23.02% |

Throughput and the conditioned ratio are stable. p95 TTFT is not: it varied
67.6% between identical arms. Goodput inherits that instability near the knee, so
goodput differences under about 23% are not readable at this sample size.

## Text

| Offered | Arm | Admitted | Wall (s) | Achieved rps |
| --- | --- | --- | --- | --- |
| 8 | C | 100% | 91.3 | 7.69 |
| 8 | PD | 100% | 90.9 | 8.14 |
| 16 | C | 100% | 91.5 | 16.15 |
| 16 | PD | 100% | 135.0 | 10.54 |
| 24 | C | 100% | 91.7 | 23.43 |
| 24 | PD | 79.8% | 157.6 | 10.60 |
| 32 | C | 100% | 92.6 | 30.58 |
| 32 | PD | 59.0% | 157.3 | 10.58 |
| 44 | C | 100% | 102.6 | 37.76 |
| 44 | PD | 43.3% | 157.1 | 10.70 |

PD saturates by offered 16 and holds about 10.6 rps, shedding load above that.
Colocated still tracked the offered rate at 44 with full admission, so its text
ceiling was not established and the 3.5x gap at offered 44 is a lower bound.

Failures are connection refusals, not client errors. Colocated refused none at
any rate.

## Image

| Offered | Arm | Admitted | Wall (s) | Achieved rps | TTFT p95 (s) |
| --- | --- | --- | --- | --- | --- |
| 1 | C | 100% | 86.1 | 0.79 | 1.610 |
| 1 | PD | 100% | 88.8 | 0.86 | 1.300 |
| 2 | C | 100% | 90.2 | 1.69 | 2.285 |
| 2 | PD | 100% | 90.8 | 1.82 | 3.893 |
| 3 | C | 100% | 94.9 | 2.63 | 4.771 |
| 3 | PD | 98.5% | 120.4 | 2.13 | 30.281 |
| 4 | C | 100% | 96.9 | 3.41 | 9.770 |
| 4 | PD | 99.6% | 161.9 | 2.15 | 69.810 |
| 6 | C | 90.8% | 135.6 | 3.44 | 42.127 |
| 6 | PD | 99.5% | 259.9 | 2.13 | 162.250 |

Three regions. At offered 1 PD leads on both throughput (+8.9%, about 15x the
A/A band) and TTFT. At offered 2 PD leads on throughput and trails on TTFT, so an
SLO on TTFT can already favour colocated. From offered 3 upward colocated leads
on both, reaching 1.62x at saturation.

Counts and rate disagree at offered 6: PD completes 554 against colocated's 472
but takes 259.9 s against 135.6 s. PD queues where colocated refuses.

## Prefill/decode interference

Ratio of the median inter-token gap that overlaps a prefill to the median gap
that does not.

| Offered | Text: C | Text: PD | Image: C | Image: PD |
| --- | --- | --- | --- | --- |
| lowest | 8.50 | 1.001 | 20.49 | 1.016 |
| highest | 5.94 | 1.535 | 8.41 | 0.969 |

PD removes the interference. On the image arm a prefill in flight on the other
card is statistically indistinguishable from no prefill at all. Colocated pays
5.8x to 8.5x on text and 8.4x to 20.5x on images, worst at low image rates where
an unchunked 6127-token prefill lands on an otherwise quiet decode stream.
`_needs_full_prefill` returns `req._input_embeds_are_projected`, and when any
queued request needs it `rem_chunk_tokens` is None, so mixed-chunk folds no
decodes into an image prefill.

PD's clean gap degrades with load, from 7.80 ms to 31.15 ms on text, while
colocated's stays between 7.6 ms and 12.9 ms. PD replaces interference with
queueing on the decode card.

## Where PD's time goes

Segment medians at text offered 16, where both arms admit every request.

| Segment | Arm C replica | Arm PD |
| --- | --- | --- |
| build | 0.24 ms | 0.18 ms |
| build to queued | 0.05 ms | 0.05 ms |
| queued to scheduled | 3.90 ms | 1.99 ms |
| first forward | 59.99 ms | 58.33 ms |
| decode tail | 2206 ms | 41560 ms |

Admission is not the constraint and prefill did not get slower. The whole
difference is the decode tail. Utilization at the same offered rate is 10.1%
mean on PD's prefill card against 67.9% on a colocated card.

A 58 ms first forward against a 2.2 s decode tail puts prefill at **at most**
2.6% of a request's GPU work, and below that whenever a prefill step carries more
than one request, which it does under load. A 1P:1D pair splits hardware 50/50
against that split, which predicts roughly 10% utilization on the prefill card,
which is what was measured.

Prefill steps batch, and the batch factor grows with load: at 42 prompt tokens
and 128 output tokens a step carries 1.19 requests at the lowest rate swept and
5.81 at the highest, while the step itself stays between 59 ms and 66 ms across a
14x range of batch size. The step cost model is therefore a cost per step, not
per request, and `1 / 0.059` bounds prefill **steps** per second rather than
requests per second.

That distinction decides which resource binds. Prefill duty cycle, meaning steps
times step cost over wall time, on one colocated replica:

| Offered | 8 output tokens | 128 output tokens |
| --- | --- | --- |
| 8 | 24.8% | 19.2% |
| 20 | 53.3% | 34.9% |
| 36 | 72.3% | 32.9% |
| 56 | 81.6% | 24.0% |

At 8 output tokens the prefill path climbs toward saturation and is the binding
constraint. At 128 it peaks near 35% and falls back as the system saturates,
because prefill admission is throttled behind decode. The colocated ceiling
reported above is decode-bound, not prefill-bound.

By Little's law PD holds about 437 requests in the decode tail against
`max_running_requests=64`, so most of that tail is queueing rather than decoding.
Colocated holds about 18.

Prefill share is governed by prompt length divided by output length. At 128
output tokens it reaches only about 9% even at 6944 prompt tokens, because the
output dominates. Shortening the output raises it faster: at 8 output tokens a
42-token prompt gives about 42%.

The handoff is not serialized: 41.5% of PD's decode gaps overlap a prefill
window, and first forward is unchanged.

This data does not explain why PD's single decode card sustains 10.54 rps while a
colocated card doing both prefill and decode sustains 18.9. Candidates that these
events cannot separate are the mandatory `page_size=1`, the mandatory
`disable_radix_cache`, the absence of mixed-chunk batching on the decode side,
and the cost of receiving KV over CUDA IPC. Separating them needs an experiment
that varies each independently.

## Admission rate lags

| Arm | Rate per server | total_s p50 | In flight |
| --- | --- | --- | --- |
| C, offered 32 | 15.41 | 4.24 s | 65 |
| C, offered 44 | 19.66 | 7.95 s | 156 |
| PD, offered 16 | 10.59 | 40.96 s | 434 |
| PD, offered 24 | 10.55 | 68.81 s | 726 |
| PD, offered 44 | 10.54 | 74.12 s | 781 |

In flight is achieved rate times median total time. At offered 16 PD admits
every request while each takes 41 s against colocated's 2.3 s. Admission rate
reports a healthy system there. In-flight count and total time report the real
state, and they should be read together.

The connection failures this sweep recorded above offered 16 were **client-side
descriptor exhaustion, not the server shedding load**. The client holds one
socket per in-flight request and ran with an unlimited connector under a 1024
soft limit, so a saturated arm exhausts descriptors. Every such record reads
`ClientConnectorError ... [Too many open files]`, which is EMFILE from the
client's own `socket()` call; `[Connect call failed]`, which is what a refusing
server produces, appears nowhere in the results. The colocated arm ran two
client processes with a budget each and never approached the limit, so the two
arms were not comparable on that axis at all. Treat admission rate above
offered 16 in this sweep as a property of the harness.

The harness now raises the descriptor limit and caps the connector, so a client
at its ceiling queues rather than failing. Throughput and latency figures are
unaffected: `achieved_rate_rps` counts completions over wall time, and the
41-second `total_s` at offered 16 was measured where there were no failures at
all.

## Reproducing

```
python -m benchmarks.eval.benchmark_omni_thinker_attribution \
    --port PORT --mode open --rates 8,16,24,32,44 --duration-s 90 \
    --repeats 3 --prompt-tokens 42 --max-tokens 128 --nonce TAG \
    --profile-events --profile-event-dir events/TAG \
    --sample-util --util-gpu-ids GPU --output results/TAG.json
python -m benchmarks.eval.analyze_thinker_events events/TAG/RUN_ID
python -m benchmarks.eval.aggregate_pd_vs_colocated --root .
```

The client offers Poisson arrivals for the full duration without waiting for
completions, so an arm that cannot keep up shows it in the achieved rate rather
than in a reduced offered rate. Each prompt carries a unique marker at token 0
and a per-repetition counter, so prefix caching does not absorb the prompt.
Multimodal runs need a large image pool with disjoint per-invocation offsets:
replaying one image set dropped TTFT p95 from 882 ms to 482 ms at the same prompt
length.

`--prompt-tokens` is a target rather than a count, and it overshoots by a factor
that grows with the value: 42 gives 42, 1150 gives about 1966, and 4400 gives
about 7645. Quote the `prompt_tokens` recorded in the result JSON, which is the
measured value. Check the target against the model's context limit before
choosing arms; Qwen3-Omni thinker allows 8192, which 4400 prompt tokens reaches
once 128 output tokens are added. Let the client generate the nonce rather than
passing `--nonce`, because the nonce sits inside every repetition counter and a
different width shifts the prompt length.

`analyze_thinker_events` accepts events from `thinker`, `thinker_prefill` and
`thinker_decode`, so one code path covers both arms.
