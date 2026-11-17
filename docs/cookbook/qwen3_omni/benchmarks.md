# Qwen3-Omni Benchmarks

[Back to the Qwen3-Omni quick start](../qwen3_omni.md).

## Speech Stage Placement

A concurrency-8 experiment held code2wav fixed on the thinker's GPU and varied only the talker's placement, from sharing the thinker's GPU to having its own. Every concurrent metric improved:

| metric (concurrency 8, two measured pairs) | talker shares the thinker's GPU | talker alone | change |
|---|---|---|---|
| wall-clock time | 22.8–24.1 s | 20.2–20.7 s | 9–16% lower |
| time to first audio (TTFA), p50 | 1.43–1.71 s | 0.99–1.20 s | 16–42% lower |
| TTFA, p90 | 2.66–2.70 s | 1.42–1.46 s | 46–47% lower |
| end-to-end latency, p50 | 10.04–10.05 s | 8.94–9.44 s | 6–11% lower |

A follow-up single-variable pair tested moving code2wav to its own GPU instead, with the talker already isolated in both arms. Wall-clock time and end-to-end latency were flat (−0.7% and −0.9%, well within noise for a single interleaved pair) — giving the vocoder a dedicated GPU bought nothing, which is why code2wav shares the thinker's GPU instead of getting isolated like the talker.

Single-stream traffic (one request at a time) sees a different trade-off. In the talker-placement experiment above, isolating the talker — moving it off the thinker's GPU — made single-stream end-to-end latency 14–17% worse: with one request in flight there is no contention to escape, while that experiment's thinker-to-talker handoff started crossing a device boundary. Single-stream TTFA still improved by about 30% (0.48–0.49 s to 0.34–0.35 s), because the talker no longer waits its turn on a shared GPU. Both the old and the new default layout already keep the thinker and talker on separate GPUs, so these numbers describe that experiment's topology change, not the default code2wav placement. Streaming TTS workloads should prefer this layout regardless, since TTFA is the latency users notice first.

See [Speech Stage Placement](tuning.md#speech-stage-placement) for the default topology and memory headroom. For the fused MMSU text-path launch, see [Text-Only Deployment](platforms.md#text-only-deployment).
