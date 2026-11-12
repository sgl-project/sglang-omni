# Qwen3-Omni

[Qwen3-Omni](https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Instruct) accepts text, images, audio, and video, and can return text alone or with audio. Use the generator for a hardware-specific launch command, then check the tables for supported configurations.

For shared request examples, response formats, SSE clients, and error handling, see [Omni Model Usage](../basic_usage/omni.md). This page documents Qwen3-Omni deployment, model-specific parameters, and realtime profiles.

## Prerequisites

Install `sglang-omni` using the [installation guide](../get_started/installation.md).

## Server Configuration

Use the selector below to generate the exact launch command for your configuration.

```{raw} html
<div id="sgl-server-gen-mount"></div>
```

## Compatibility Matrix

Colocated topology requires `--config examples/configs/qwen3_omni_colocated_h20.yaml` (or `qwen3_omni_colocated_h200.yaml` on H200) to set per-stage GPU memory budgets.

All configurations listed below are supported.

| Mode | Topology | Thinker TP | Precision |
|---|---|---|---|
| Thinker-only | — | — | BF16 |
| Thinker-only | — | — | FP8 |
| Thinker-only | — | — | AutoRound INT4 |
| Thinker-Talker | Disaggregated | TP=1 | BF16 |
| Thinker-Talker | Disaggregated | TP=1 | FP8 |
| Thinker-Talker | Disaggregated | TP=1 | AutoRound INT4 thinker + BF16 talker/code2wav |
| Thinker-Talker | Disaggregated | TP=2 | BF16 |
| Thinker-Talker | Disaggregated | TP=2 | FP8 |
| Thinker-Talker | Disaggregated | TP=2 | AutoRound INT4 thinker + BF16 talker/code2wav |
| Thinker-Talker | Colocated | TP=1 | BF16 |
| Thinker-Talker | Colocated | TP=1 | FP8 |
| Thinker-Talker | Colocated | TP=1 | AutoRound INT4 thinker + BF16 talker/code2wav |

## Text-Only Deployment

Text-only mode runs the thinker pipeline on a single GPU. It accepts multi-modal input (text, image, audio) and produces text output only.

### Launch a Text-Only Server

```bash
sgl-omni serve \
  --model-path Qwen/Qwen3-Omni-30B-A3B-Instruct \
  --text-only \
  --port 8008
```

Multimodal preprocessing (tokenization, image/video/audio feature extraction) runs serially by default. Add `--preprocessing.factory.max_concurrency 4` to run it on a thread pool when CPU preprocessing is the bottleneck under concurrent load. Threaded preprocessing changes how requests arrive at the thinker, so greedy outputs of the bf16 MoE can differ from the serial default; keep the default when comparing accuracy across runs.

Each preprocessing worker decodes a request's videos on parallel decoders that share min(16, the stage's CPUs divided by its workers) FFmpeg threads, split evenly among the request's videos. Set `--preprocessing.factory.video_decode_threads N` to choose that count. A value at or below the video reader's own threads (`TORCHCODEC_NUM_THREADS`, 8 by default) decodes each video on the reader alone.

For MMSU-style audio-input / text-output benchmarks with short requests, use the fused text-path config so the full text path stays inside one worker process:

```bash
sgl-omni serve \
  --model-path Qwen/Qwen3-Omni-30B-A3B-Instruct \
  --config examples/configs/qwen3_omni_mmsu.yaml \
  --text-only \
  --port 8008
```

## Speech Deployment and Tuning

Speech mode runs the full eight-stage pipeline on one or more GPUs. It produces both text (from the thinker) and audio (from the talker) output.

### Launch a Speech Server

Speech mode can run as a colocated one-GPU worker using the colocated config:

```bash
sgl-omni serve \
  --model-path Qwen/Qwen3-Omni-30B-A3B-Instruct \
  --config examples/configs/qwen3_omni_colocated_h20.yaml \
  --colocate \
  --port 8008
```

Use `examples/configs/qwen3_omni_colocated_h200.yaml` on single-H200 workers.

For manual multi-GPU placement, use the example script:

```bash
python examples/run_omni.py qwen3-speech-server \
  --model-path Qwen/Qwen3-Omni-30B-A3B-Instruct \
  --gpu-thinker 0 \
  --gpu-talker 1 \
  --gpu-code-predictor 1 \
  --gpu-code2wav 0 \
  --port 8008
```

Or use the CLI without `--text-only` for the standard speech pipeline:

```bash
sgl-omni serve \
  --model-path Qwen/Qwen3-Omni-30B-A3B-Instruct \
  --port 8008
```

### Memory and Concurrency

By default, leave `mem_fraction_static` unset and let SGLang-Omni auto-size the SGLang AR memory budget. If a specific machine needs manual tuning, you can pin the value globally or per AR stage:

```bash
sgl-omni serve \
  --model-path Qwen/Qwen3-Omni-30B-A3B-Instruct \
  --port 8008 \
  --mem-fraction-static 0.88
```

Use per-stage flags when the thinker and talker need different budgets:

```bash
sgl-omni serve \
  --model-path Qwen/Qwen3-Omni-30B-A3B-Instruct \
  --port 8008 \
  --thinker.engine.mem_fraction_static 0.88 \
  --talker_ar.engine.mem_fraction_static 0.88
```

The speech server launcher exposes the same per-stage controls:

```bash
python examples/run_omni.py qwen3-speech-server \
  --model-path Qwen/Qwen3-Omni-30B-A3B-Instruct \
  --gpu-thinker 0 \
  --gpu-talker 1 \
  --gpu-code-predictor 1 \
  --gpu-code2wav 0 \
  --port 8008 \
  --thinker-mem-fraction-static 0.88 \
  --talker-mem-fraction-static 0.88
```

`--mem-fraction-static` applies to every SGLang engine stage. A dotted per-stage path overrides the global value for that stage. Values must be greater than `0` and less than `1`.

The thinker admits up to 64 running requests by default. Use the thinker-specific flag to lower or raise that limit in either text-only or speech mode:

```bash
sgl-omni serve \
  --model-path Qwen/Qwen3-Omni-30B-A3B-Instruct \
  --thinker.engine.max_running_requests 16
```

To configure the thinker through a pipeline YAML file instead, set the same path under the stage entry:

```yaml
stages:
  thinker:
    engine:
      max_running_requests: 16
```

### Codec Coalescing and First-Audio Latency

The speech pipeline sets `codec_coalesce_frames=10`, `codec_coalesce_early_frames=10`, and `codec_coalesce_first_frames=0` under `stages.talker_ar.factory`. The first 10 codec frames are sent individually; later frames are coalesced into groups of 10. Omitting a YAML override keeps these pipeline defaults; set `codec_coalesce_early_frames=0` explicitly to disable the early prefix. This aligns with the default Code2Wav 10-frame threshold: the first three windows contain 10, 20, and 30 frames, and subsequent full windows contain 35 frames including left context. These shapes can use the captured windows when CUDA Graph is enabled. Each window decodes one 10-frame chunk past its context, so an early prefix of 12 keeps these window lengths and its two extra frames wait for the next window.

With `initial_codec_chunk_frames=2` and `stream_chunk_size=10`, explicitly set `codec_coalesce_early_frames=12` to make the first two windows eligible at generated frames 2 and 12. Uniform groups of 10 (`early_frames=0`, `first_frames=0`) instead publish the first group at step 11: the sender retains the newest row until the next step can exclude EOS, or the request finishes. For a request that continues past step 10, first-window input readiness therefore moves from step 2 to step 11, adding nine Talker decode intervals. If the interval is approximately `d` ms, the added input wait is approximately `9d` ms. With the default 10-frame first window, readiness instead moves from step 10 to step 11. The default 10-frame early prefix preserves readiness at step 10; the next two windows become ready at steps 21 and 31. The extra step after the prefix retains the newest row for EOS detection, compared with steps 20 and 30 without coalescing.

This is an input-readiness estimate, not a measured end-to-end TTFA delta or nine frames of audio playback time. Actual TTFA also depends on transport, queueing, and vocoder execution; the overall coalescing benchmark does not isolate the early-prefix setting.

### Code2Wav Execution

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

### Speech Stage Placement

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

## Single-GPU FP8 on H100/H20

SGLang-Omni can also serve native FP8 Qwen3-Omni checkpoints. Native FP8 uses the checkpoint quantization config when loading the thinker and talker AR stages, while keeping the same Qwen3-Omni request format shown below.

For one-GPU H100/H20 colocated launch, use the FP8 colocated config:

```bash
sgl-omni serve \
  --config examples/configs/qwen3_omni_fp8_colocated.yaml \
  --colocate \
  --model-name qwen3-omni \
  --port 8008
```

The config file contains the FP8 checkpoint path: `marksverdhei/Qwen3-Omni-30B-A3B-FP8`. You can still pass `--model-path` to override the config value.

The FP8 path keeps dense FP8 GEMM on SGLang `auto` and defaults native FP8 MoE to CUTLASS when supported. For Qwen3-Omni pipeline launches, `SGLANG_JIT_DEEPGEMM_PRECOMPILE=0` is set as a default unless the operator has already set that environment variable. This disables SGLang's all-M DeepGEMM precompile session while keeping DeepGEMM available for dense FP8 GEMMs.

To opt back into SGLang's all-M DeepGEMM precompile behavior:

```bash
SGLANG_JIT_DEEPGEMM_PRECOMPILE=1 sgl-omni serve \
  --config examples/configs/qwen3_omni_fp8_colocated.yaml \
  --colocate \
  --model-name qwen3-omni \
  --port 8008
```

## Single-GPU AutoRound INT4 Thinker on H100/H20

SGLang-Omni also supports AutoRound INT4 quantized Qwen3-Omni checkpoints. AutoRound uses a 4-bit quantization scheme with group size 128, significantly reducing memory footprint compared to BF16 or FP8.

The public AutoRound checkpoint quantizes the thinker transformer layers. In speech mode, the talker and code2wav stages load as BF16 from the same checkpoint. For one-GPU H100/H20 colocated launch, use the colocated config with the AutoRound checkpoint:

```bash
sgl-omni serve \
  --config examples/configs/qwen3_omni_colocated_h20.yaml \
  --colocate \
  --model-name qwen3-omni \
  --model-path Intel/Qwen3-Omni-30B-A3B-Instruct-int4-AutoRound \
  --port 8008
```

AutoRound quantization provides:

- **~50% memory reduction** compared to BF16 (from ~60GB to ~30GB)
- **~25% memory reduction** compared to FP8 (from ~40GB to ~30GB)
- **Accuracy at ultra-low bit widths**: maintains high accuracy even at 2–4 bits, requiring minimal tuning effort thanks to its sign-gradient descent optimization.

## Server Readiness

Before it reports ready, the server sends itself a chat request with a small image, one second of audio and a short instruction (text and audio output in speech mode, text output in text-only mode, at most 8 new tokens from each stage that generates tokens), once per replica of its most replicated process, one after another, so that every replica serves one. `/health` returns 503 with status `starting` until these requests have passed through every stage, so the one-time kernel builds of that request path happen before the first user request. Inputs of other shapes, such as video or a large concurrent batch, can still build a kernel variant on first use. A failed warmup stops the server. `SGLANG_OMNI_WARMUP_TIMEOUT` (seconds, default 600) bounds each request, and `--skip-server-warmup` reports ready without them.

## Input / Output Modalities

Qwen3-Omni accepts text, image, audio, and video input in both text-only and speech deployments. The [shared request examples](../basic_usage/omni.md#general-usage) cover image questions, spoken questions with images or video, and text-and-speech replies. To return audio, launch the speech pipeline; requesting audio on a text-only chat server returns text without an error. Realtime audio negotiation has a separate contract described below.

### Sampling Parameters

Standard sampling parameters apply to the thinker. When `modalities` includes `"audio"`, talker-specific parameters control speech generation.

| Parameter | Type | Default | Applies to |
|---|---|---|---|
| `temperature` | float | `0.0` (greedy) | Thinker |
| `top_p` | float | `1.0` | Thinker |
| `top_k` | int | `-1` | Thinker |
| `min_p` | float | `0.0` | Thinker |
| `repetition_penalty` | float | `1.0` | Thinker |
| `max_tokens` | int | `2048` | Thinker |
| `max_completion_tokens` | int | `null` | Thinker; OpenAI-compatible alias for `max_tokens` |
| `stop` | str or list | `null` | Thinker |
| `seed` | int | `null` | Thinker |
| `stream` | bool | `false` | Both |
| `audio` | dict | `null` | Speech response format config, e.g. `{"format": "wav"}` |
| `talker_temperature` | float | `0.9` | Talker (audio output only) |
| `talker_top_p` | float | `1.0` | Talker (audio output only) |
| `talker_top_k` | int | `50` | Talker (audio output only) |
| `talker_repetition_penalty` | float | `1.05` | Talker (audio output only) |
| `talker_max_new_tokens` | int | `4096` | Talker (audio output only) |
| `stage_sampling` | dict | `null` | Per-stage sampling override |
| `stage_params` | dict | `null` | Per-stage non-sampling params |
| `video_fps` | float | `null` | Frame sampling rate for video input (uses server default if unset) |
| `video_max_frames` | int | `null` | Maximum number of frames sampled from a video |
| `video_min_pixels` | int | `null` | Minimum pixels per video frame |
| `video_max_pixels` | int | `null` | Maximum pixels per video frame |
| `video_total_pixels` | int | `null` | Total pixel budget across all video frames |
| `use_audio_in_video` | bool | `null` | Set to `true` to process embedded audio. Requests where all videos lack audio use video-only processing; mixing videos with and without audio is rejected. Audio decoding errors are reported. |

For HTTP and SSE errors, see [Known Limitations](../basic_usage/omni.md#known-limitations).

## Realtime Speech with Server-Side Turn Detection

The speech pipeline also streams spoken responses over `/v1/realtime` with `--enable-realtime`, with server-side turn detection and barge-in. Its protocol is in {ref}`Realtime on the speech pipeline <realtime-on-the-speech-pipeline>`; the Session graph that most realtime clients use is in [Turn-Based: Qwen3-Omni](../developer_reference/full_duplex_guide.md#turn-based-qwen3-omni).

## Qwen3-Omni persistent session stages

The session profile serves Qwen3-Omni as a turn-based realtime voice assistant on `/v1/realtime`: a VAD node ends each user turn and the model answers it with the conversation so far. Launching on H100 or H200, settings, the protocol and benchmarking are in [Full-Duplex Guide](../developer_reference/full_duplex_guide.md#turn-based-qwen3-omni).

## Speculative Decoding

The thinker can verify tokens proposed by SpeechSpec's DFlash draft [`yuekai/qwen3_omni_30b_a3b_instruct_dflash_block8`](https://huggingface.co/yuekai/qwen3_omni_30b_a3b_instruct_dflash_block8) or EAGLE3 draft [`yuekai/qwen3_omni_30b_a3b_instruct_eagle3_audio`](https://huggingface.co/yuekai/qwen3_omni_30b_a3b_instruct_eagle3_audio) on a text-only server.

```bash
# DFlash: one 8-token block per step
sgl-omni serve \
  --model-path Qwen/Qwen3-Omni-30B-A3B-Instruct \
  --text-only \
  --thinker.engine.speculative_algorithm DFLASH \
  --thinker.engine.speculative_draft_model_path yuekai/qwen3_omni_30b_a3b_instruct_dflash_block8 \
  --thinker.engine.speculative_num_draft_tokens 8 \
  --port 8000

# EAGLE3: a 3-step chain
sgl-omni serve \
  --model-path Qwen/Qwen3-Omni-30B-A3B-Instruct \
  --text-only \
  --thinker.engine.speculative_algorithm EAGLE3 \
  --thinker.engine.speculative_draft_model_path yuekai/qwen3_omni_30b_a3b_instruct_eagle3_audio \
  --thinker.engine.speculative_num_steps 3 \
  --thinker.engine.speculative_eagle_topk 1 \
  --thinker.engine.speculative_num_draft_tokens 4 \
  --port 8000
```

- **Acceptance length.** Set `return_spec_tokens_details: true` to receive `sglext.spec_tokens_details`: `spec_accept_length` (output tokens per verify, including the prefill-sampled token, so it can exceed scheduler `accept len`), `spec_accept_rate`, draft counts, and a per-verify histogram. Streaming sends it in a chunk with empty `choices` before `[DONE]`. `spec_accept_rate` and `spec_num_proposed_drafts` are omitted when no drafts were proposed or draft width is adaptive. Only thinker drafts are reported; per-stage reporting can follow when Talker supports drafts.
- **Context.** These drafts have an 8,192-token context; the prompt, output budget, and speculative reservations must fit. Set `--preprocessing.factory.max_seq_len 8192` to reject oversized prompts early.
- **Sampling.** Use `repetition_penalty=1`, `frequency_penalty=0`, `presence_penalty=0`, and `min_p=0`; disable logprobs and `return_hidden_states`. For `/generate`, set `return_logprob: false`.
- **Scheduling.** Speculative decoding is synchronous by default. Add `--thinker.factory.enable_async_decode true` to overlap work, keeping `async_decode_min_batch_size=1`. Mixed prefill/decode batches (`enable_mixed_chunk=true`), sessions, and weight updates are unsupported with a draft.
- **Overlap trade-offs.** Overlap can improve decode throughput, but prefill-heavy workloads or short outputs may have higher TTFT and lower throughput.
- **Prefill boundary.** Set `SGLANG_DISABLE_CONSECUTIVE_PREFILL_OVERLAP=1` before launch to process the previous prefill batch's result before launching the next consecutive prefill batch. This can lower TTFT at a throughput cost; decode rounds still overlap.
- **Execution failure.** A failure during speculative overlap stops the pipeline and fails pending requests.
- **Speech output.** Drafts do not support speech generation; use `--text-only`.

Compare modes with matching workloads and prefix-cache conditions; measure TTFT, throughput, and accuracy for your workload.

## Known Limitations

- **Multiple videos must use the same sampled frame rate.** The processor uses one rate for video timestamps; different rates return an invalid-input error.

- **`content` must be `""` when the query is entirely in `audios`, `videos`, or `images`.** Otherwise, any text in `content` is processed too.
- **Colocated topology does not support `--thinker.tp_size 2`.** Startup raises a `ValueError`; use disaggregated topology for TP=2.
- **Context limits apply to prompt and output.** The preprocessor rejects requests when prompt tokens meet or exceed `max_seq_len`, or when prompt tokens plus `max_new_tokens` meet or exceed it. Without both `--preprocessing.factory.max_seq_len` and `--thinker.factory.max_seq_len`, the limit is the thinker's checkpoint context (65,536 tokens for Instruct); setting both lower caps it. The talker uses its checkpoint limit unless `--talker_ar.factory.max_seq_len` sets it. A long spoken prompt may leave too little room for `talker_max_new_tokens`, so audio can end before the text.
