# Qwen3-Omni Speech Tuning

Advanced settings and measurements for Qwen3-Omni speech deployment. Start with the launch defaults in the [Qwen3-Omni cookbook](../qwen3_omni.md#speech-deployment-and-tuning) and measure with your expected traffic before changing them.

## Speech Chunking and First-Audio Latency

First-audio latency is the wait before the first playable chunk arrives. See [Speech Streaming Latency](../../basic_usage/omni.md#speech-streaming-latency) for the general trade-off.

The speech pipeline sets `codec_coalesce_frames=10`, `codec_coalesce_early_frames=10`, and `codec_coalesce_first_frames=0` under `stages.talker_ar.factory`. The first 10 codec frames are sent individually; later frames are coalesced into groups of 10. Omitting a YAML override keeps these pipeline defaults; set `codec_coalesce_early_frames=0` explicitly to disable the early prefix. This aligns with the default Code2Wav 10-frame threshold: the first three windows contain 10, 20, and 30 frames, and subsequent full windows contain 35 frames including left context. These shapes can use the captured windows when CUDA Graph is enabled. Each window decodes one 10-frame chunk past its context, so an early prefix of 12 keeps these window lengths and its two extra frames wait for the next window.

With `initial_codec_chunk_frames=2` and `stream_chunk_size=10`, explicitly set `codec_coalesce_early_frames=12` to make the first two windows eligible at generated frames 2 and 12. Uniform groups of 10 (`early_frames=0`, `first_frames=0`) instead publish the first group at step 11: the sender retains the newest row until the next step can exclude EOS, or the request finishes. For a request that continues past step 10, first-window input readiness therefore moves from step 2 to step 11, adding nine Talker decode intervals. If the interval is approximately `d` ms, the added input wait is approximately `9d` ms. With the default 10-frame first window, readiness instead moves from step 10 to step 11. The default 10-frame early prefix preserves readiness at step 10; the next two windows become ready at steps 21 and 31. The extra step after the prefix retains the newest row for EOS detection, compared with steps 20 and 30 without coalescing.

This is an input-readiness estimate, not a measured end-to-end TTFA delta or nine frames of audio playback time. Actual TTFA also depends on transport, queueing, and vocoder execution; the overall coalescing benchmark does not isolate the early-prefix setting.

## Code2Wav Execution

Exact-shape CUDA Graph replay is enabled by default for Qwen3-Omni Code2Wav. The default stage config supplies a 2% typed GPU memory budget; colocated example configs override it with their hardware-specific budget.

To disable replay, set it on the stage in the YAML config:

```yaml
stages:
  code2wav:
    factory:
      enable_cuda_graph: false
```

When replay is enabled, a custom Code2Wav stage must define `gpu_memory_fraction`; startup rejects a missing typed budget before loading the model.

The feature derives the exact `B=1` threshold windows from `stream_chunk_size` and `left_context_size`; the defaults capture `T{10,20,30,35}`. It also captures, best effort, the final window of every stream length: the 1 to `stream_chunk_size - 1` frames left after a stream's last threshold window plus the context it holds. With the threshold windows this covers every length from `T1` to `T35` at the defaults. A final window that capture skips runs eagerly, as does any other shape. Capture-time incompatibilities also fall back to eager execution.

The SnakeBeta activation of the Code2Wav decoder runs as one fused Triton kernel by default on CUDA devices, shared with the Qwen3-TTS vocoder. The kernel keeps the checkpoint parameters and every BF16 rounding step of the eager activation, so its output is bitwise identical, and it is compiled before graph capture, with or without CUDA Graph replay. Inputs outside its contract (a dtype other than BF16, a non-contiguous layout, or rows longer than 65535 x 1024 samples) run the eager arithmetic, and if the kernel cannot be compiled the eager modules stay installed. To keep the eager activation:

```yaml
stages:
  code2wav:
    factory:
      fused_snake_activation: false
```

Code2Wav takes every queued codec chunk before it decodes, then decodes the ready windows of one length in one replay, up to `max_replay_rows` (default 8) under `stages.code2wav.factory`, each row count a captured graph. A request alone still gets one window per replay; under load, the windows that several requests complete on the same talker step share one. A row of a shared replay can differ from the same window decoded alone in its last bits, since the convolution kernels depend on the batch.

Output overlap is also enabled by default on CUDA devices: each threshold window's waveform readback runs as an asynchronous device-to-host copy into a pinned staging buffer and is materialized while the GPU computes the next window, and the codec EOS check runs once per window instead of once per frame. The first window of every request stays synchronous, so time-to-first-audio is unchanged. For successfully completed requests, audio bytes and message boundaries are identical to the synchronous path. A request that fails or is aborted can drop an un-emitted pending window while its in-flight staging buffer is retired. To disable it:

```yaml
stages:
  code2wav:
    factory:
      enable_output_overlap: false
```

## Speech Stage Placement

At concurrency 8, the talker is the heaviest speech stage: it holds its GPU at about 86% median utilization. Give it a GPU of its own. The code2wav vocoder is light by comparison (9–13% median utilization) and shares the thinker's GPU by default. That default holds only when the thinker stays on its own default GPU — a `--thinker.gpu` override moves the thinker alone, not code2wav, so pass `--code2wav.gpu` explicitly too if you relocate the thinker.

This is the default topology for `sgl-omni serve` without GPU overrides: thinker alone, talker alone, code2wav on the thinker's GPU. When code2wav shares the thinker's GPU, the thinker's auto-sized KV pool shrinks to make room for it — about 4.3 GiB smaller, measured on H200, since the vocoder itself needs roughly 1.4–1.6 GiB. That adjustment only happens for the auto-sized budget: an explicitly pinned `--thinker.engine.mem_fraction_static` gets no automatic carve-out, so a tightly pinned fraction should leave headroom for code2wav.

A concurrency-8 experiment held code2wav fixed on the thinker's GPU and varied only the talker's placement, from sharing the thinker's GPU to having its own. Every concurrent metric improved:

| metric (concurrency 8, two measured pairs) | talker shares the thinker's GPU | talker alone | change |
|---|---|---|---|
| wall-clock time | 22.8–24.1 s | 20.2–20.7 s | 9–16% lower |
| time to first audio (TTFA), p50 | 1.43–1.71 s | 0.99–1.20 s | 16–42% lower |
| TTFA, p90 | 2.66–2.70 s | 1.42–1.46 s | 46–47% lower |
| end-to-end latency, p50 | 10.04–10.05 s | 8.94–9.44 s | 6–11% lower |

A follow-up single-variable pair tested moving code2wav to its own GPU instead, with the talker already isolated in both arms. Wall-clock time and end-to-end latency were flat (−0.7% and −0.9%, well within noise for a single interleaved pair) — giving the vocoder a dedicated GPU bought nothing, which is why code2wav shares the thinker's GPU instead of getting isolated like the talker.

Single-stream traffic (one request at a time) sees a different trade-off. In the talker-placement experiment above, isolating the talker — moving it off the thinker's GPU — made single-stream end-to-end latency 14–17% worse: with one request in flight there is no contention to escape, while that experiment's thinker-to-talker handoff started crossing a device boundary. Single-stream TTFA still improved by about 30% (0.48–0.49 s to 0.34–0.35 s), because the talker no longer waits its turn on a shared GPU. Both the old and the new default layout already keep the thinker and talker on separate GPUs, so these numbers describe that experiment's topology change, not the default code2wav placement. Streaming TTS workloads should prefer this layout regardless, since TTFA is the latency users notice first.
