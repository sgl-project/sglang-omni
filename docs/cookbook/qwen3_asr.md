# Qwen3-ASR

[Qwen3-ASR](https://huggingface.co/Qwen/Qwen3-ASR-1.7B) is an audio transcription model served through the OpenAI-compatible `/v1/audio/transcriptions` endpoint. It accepts one uploaded audio file per request and returns text.

Qwen3-ASR does not serve `/v1/audio/translations`; that route returns HTTP 400. The endpoint semantics, response formats, SSE lifecycle, and request parameters that Qwen3-ASR shares with the other ASR models are documented in [ASR Model Usage](../basic_usage/asr.md); this page keeps the behavior specific to this checkpoint.

![Qwen3-ASR model architecture](https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-ASR-Repo/overview.jpg)

## Prerequisites

Install `sglang-omni` by following the [CUDA installation](../get_started/installation.md) or [Apple Silicon installation](../other_platforms/installation_apple_silicon.md), then download the model:

```bash
MODEL_REVISION=7278e1e70fe206f11671096ffdd38061171dd6e5
MODEL_PATH="$(
  hf download Qwen/Qwen3-ASR-1.7B \
    --revision "${MODEL_REVISION}" \
    --quiet
)"
```

## Server Configuration

Start the default CUDA server with the downloaded checkpoint:

```bash
sgl-omni serve \
  --model-path "${MODEL_PATH}" \
  --model-name Qwen/Qwen3-ASR-1.7B \
  --port 8000
```

Qwen3-ASR runs a single ASR stage on one GPU. Its default `auto` dtype follows the checkpoint configuration (BF16 for Qwen3-ASR-1.7B); pass `--asr.factory.dtype float16` to force FP16. For decode scheduling and admission controls, see [Concurrency tuning](#concurrency-tuning).

For a single 24 GB RTX 4090 (SM89), use the checked-in consumer profile:

```bash
sgl-omni serve \
  --config examples/configs/qwen3_asr_rtx4090.yaml \
  --port 8000
```

This qualified profile keeps the model in BF16, limits the stage to 16 running requests, and sets `mem_fraction_static` to `0.65`. Its bounds are specific to the validated RTX 4090 layout; use the default configuration or a separately qualified profile on other GPU architectures.

For a single 32 GB RTX 5090, use:

```bash
sgl-omni serve \
  --config examples/configs/qwen3_asr_rtx5090.yaml \
  --port 8000
```

This profile uses BF16, allows up to 16 running requests, and sets `mem_fraction_static=0.65`. See the [RTX 5090 benchmark report](https://github.com/sgl-project/sglang-omni/issues/1212) for results measured on an earlier release.

Apple Silicon and Ascend NPU launch different configurations; see [Platform Configuration](#platform-configuration).

## Choose a Transcription Method

| Goal | Method | Audio input |
|---|---|---|
| Transcribe a file | [HTTP transcription](#transcribe-audio) | Upload a complete file |
| Receive text incrementally | [SSE transcription](#stream-transcription) | Upload a complete file before decoding starts |
| Transcribe live audio | [Realtime WebSocket](#realtime-transcription-websocket) | Send PCM audio packets as they arrive |

## Transcribe Audio

Send a local audio file to the running server:

```bash
curl -X POST http://localhost:8000/v1/audio/transcriptions \
  -F model=Qwen/Qwen3-ASR-1.7B \
  -F file=@path/to/audio.wav \
  -F response_format=json
```

Replace `path/to/audio.wav` with your audio file. The JSON response contains the transcript in `text`.

Send one uploaded file to `POST /v1/audio/transcriptions`. The request shape and the `response_format` semantics shared with the other ASR models are documented in [ASR Model Usage](../basic_usage/asr.md#use-curl). Qwen3-ASR adds one detail: `verbose_json` reports one segment per transcribed chunk with that chunk's real start and end timestamps, not word-level timestamps. When duration probing succeeds, it also includes duration-based usage (rounded-up audio seconds).

## Stream Transcription

Set `stream=true` to receive incremental transcript text over SSE. The event lifecycle, batching behavior, and the response formats that streaming accepts are shared and documented in [ASR Model Usage](../basic_usage/asr.md#stream-transcription). Qwen3-ASR batches deltas for up to 50 ms by default, and EOS or another terminal condition flushes any buffered text before the final transcript event.

<a id="realtime-audio"></a>
<a id="live-pcm-transcription"></a>

## Realtime Transcription (WebSocket)

SSE streaming starts decoding after a complete multipart upload. For live audio ingestion, mount the realtime WebSocket endpoint:

```bash
sgl-omni serve \
  --model-path "${MODEL_PATH}" \
  --model-name Qwen/Qwen3-ASR-1.7B \
  --enable-realtime \
  --port 8000
```

Connect to `/v1/realtime?intent=transcription`, configure the session, and append base64-encoded mono 16 kHz PCM16 packets. `input_audio_buffer.commit` finalizes the active segment manually; server VAD also finalizes after the configured silence interval. Send `transcription.done` after the final packet to receive `transcription.completed`. `input_audio_buffer.clear` discards the current segment and any partial hypothesis derived from it while keeping the WebSocket session open for new audio.

```json
{
  "type": "session.update",
  "session": {
    "language": "English",
    "decode_interval_ms": 1000,
    "turn_detection": {
      "type": "server_vad",
      "threshold": 0.5,
      "prefix_padding_ms": 300,
      "silence_duration_ms": 500
    }
  }
}
```

`decode_interval_ms` is how much new audio arrives between partial decodes. It defaults to 2000 ms, must be at least 200 ms, and `null` restores the default. Like `language` and `turn_detection`, it can change only while no uncommitted audio is buffered; with server VAD that means before the first audio packet.

Each periodic decode is an ordinary stateless Qwen3-ASR request over all audio in the active segment. After the first two refreshes, the server rolls five tokens back from the prior hypothesis and uses the retained text as the next prompt prefix. No decoder KV cache or worker affinity is retained. Segments are also finalized at the configured `audio_chunking.max_audio_clip_s` boundary (30 seconds by default).

Partial results are full replacements, not append-only deltas:

```json
{
  "type": "transcription.segment",
  "event_index": 7,
  "segment_id": 0,
  "audio_start_ms": 0,
  "audio_end_ms": 2000,
  "text": "hello wor",
  "is_final": false,
  "language": "English"
}
```

A later event for the same `segment_id` replaces this text. `audio_start_ms` and `audio_end_ms` are the span of input audio the text was decoded from. A partial can trail the audio already sent, since audio that arrives while it decodes is left to the next partial. `audio_start_ms` stays fixed for the segment, including segments opened by the 30 second boundary, which have no `input_audio_buffer.speech_started` event. An event with `is_final=true` is immutable. `transcription.completed` contains the joined text from all final segments.

`language` is the language the segment was transcribed in, as Qwen3-ASR names it (for example `English`): the detected language, or the `language` hint when the session set one. `transcription.completed` carries the language of its last non-empty final segment. Both fields are `null` when the language is unknown.

Append events are not idempotent. After a transport failure, reconnect and restart the transcription rather than retrying packets on the old session. Reconnecting resets uncommitted audio, partial hypotheses, VAD state, and Qwen rollback state.

First convert the repository sample, which is 48 kHz, to the format required by the WebSocket endpoint:

```bash
ffmpeg -i tests/data/query_to_cars.wav -ac 1 -ar 16000 -c:a pcm_s16le query_to_cars_16k.wav
```

A minimal client appends packets containing 100 ms of audio, commits the segment, and waits for the completed transcript:

```python
import asyncio
import base64
import json
import wave

import websockets


async def main() -> None:
    async with websockets.connect(
        "ws://localhost:8000/v1/realtime?intent=transcription"
    ) as ws:
        await ws.send(json.dumps({
            "type": "session.update",
            "session": {"language": "English"},
        }))
        event = json.loads(await ws.recv())
        if event["type"] == "error":
            raise RuntimeError(event["error"]["message"])
        else:
            print(event)

        with wave.open("query_to_cars_16k.wav", "rb") as audio_file:
            if (
                audio_file.getnchannels() != 1
                or audio_file.getframerate() != 16000
                or audio_file.getsampwidth() != 2
                or audio_file.getcomptype() != "NONE"
            ):
                raise ValueError("Expected an uncompressed mono 16 kHz PCM16 WAV file")
            else:
                pcm16 = audio_file.readframes(audio_file.getnframes())
        packet_bytes = 1600 * 2
        for start in range(0, len(pcm16), packet_bytes):
            await ws.send(json.dumps({
                "type": "input_audio_buffer.append",
                "audio": base64.b64encode(pcm16[start:start + packet_bytes]).decode(),
            }))

        await ws.send(json.dumps({"type": "input_audio_buffer.commit"}))
        await ws.send(json.dumps({"type": "transcription.done"}))

        while True:
            event = json.loads(await ws.recv())
            if event["type"] == "error":
                raise RuntimeError(event["error"]["message"])
            elif event["type"] == "transcription.completed":
                print(event["text"])
                break
            else:
                pass


asyncio.run(main())
```

Server `error` events raise `RuntimeError`, so a failed segment or rejected session update cannot silently produce a successful-looking partial transcript.

The example extracts PCM frames from an uncompressed mono 16 kHz PCM16 WAV file and sends packets containing 100 ms of audio. It sends the file as fast as possible; add pacing when simulating live capture. The WebSocket endpoint does not resample audio, so convert other input formats before sending them.

## Request Parameters

The endpoint accepts the shared ASR request fields (`file`, `model`, `language`, `response_format`, `temperature`, `stream`) described in [ASR Model Usage](../basic_usage/asr.md#request-parameters). Qwen3-ASR also accepts:

| Field | Default | Usage |
|---|---|---|
| `prompt` | None | Vocabulary biasing; see below |
| `repetition_penalty` | `1.0` | Range `(0, 2]`; values above `1` discourage repeats but can reduce accuracy on clean audio. Keep the default unless repetition occurs. See [DFlash](#dflash-speculative-decoding) for platform restrictions. |
| `max_new_tokens` | Stage limit | Per-request generation cap |

### Prompt and Vocabulary Biasing

`prompt` raises the model's preference for supplied terms without guaranteeing them. An irrelevant list can introduce words that were not spoken. Use a short, relevant list; testing found little accuracy improvement beyond roughly 20 terms, while latency continued to grow because the prompt is prefilled with every request.

### Language Hints

When `language` is omitted, Qwen3-ASR detects the spoken language before transcribing. Set an explicit hint when the language is known or automatic detection is unreliable for short or ambiguous audio.

Qwen3-ASR accepts the following 30 explicit language codes and their canonical names:

| Codes | Canonical names |
|---|---|
| `ar`, `yue`, `zh`, `cs`, `da`, `nl`, `en`, `fil`, `fi`, `fr` | Arabic, Cantonese, Chinese, Czech, Danish, Dutch, English, Filipino, Finnish, French |
| `de`, `el`, `hi`, `hu`, `id`, `it`, `ja`, `ko`, `mk`, `ms` | German, Greek, Hindi, Hungarian, Indonesian, Italian, Japanese, Korean, Macedonian, Malay |
| `fa`, `pl`, `pt`, `ro`, `ru`, `es`, `sv`, `th`, `tr`, `vi` | Persian, Polish, Portuguese, Romanian, Russian, Spanish, Swedish, Thai, Turkish, Vietnamese |

For example, `language=es` and `language=Spanish` both force the prompt suffix `language Spanish<asr_text>`. The legacy `cn` and regional `zh-*` spellings are also accepted as Chinese. Unsupported language hints return HTTP 400 instead of silently falling back to English.

The model also has ASR coverage for 22 Chinese dialects, but those dialect names are not supported as forced `language` hints; use `Chinese`/`zh` for them.

## Long Audio

For non-streaming uploads, audio longer than `audio_chunking.max_audio_clip_s` is split into chunks, which are transcribed separately and joined in order. The default chunk length is 30 seconds, below the model's native 1,200-second single-request limit.

| Setting | Default | Purpose |
|---|---|---|
| `audio_chunking.max_audio_clip_s` | `30` | Chunk length in seconds; capped at the native clip limit |
| `audio_chunking.max_concurrent_chunks` | `8` | Concurrent chunks per upload |
| `audio_chunking.max_total_audio_s` | `3600` | Upload limit in seconds; longer uploads return HTTP 400 |
| `audio_chunking.max_concurrent_long_audio_requests` | Derived from engine capacity; `4` with stock defaults | Admitted long uploads; excess requests return HTTP 503 |

Set these values with dotted CLI flags or matching YAML keys. Torch MPS caps the whole upload at 60 seconds. `stream=true` cannot use chunking; see [Known Limitations](#known-limitations). If duration probing fails for an unusual audio format, the server uses the non-chunked path.

For the shared behavior, see [ASR Model Usage](../basic_usage/asr.md#long-audio). For capacity planning, model properties, and encoder-graph memory costs, see [Long-audio capacity](../developer_reference/qwen3_asr_concurrency_profile.md#long-audio-capacity).

## Known Limitations

The limits shared with the other ASR models are listed in [ASR Model Usage](../basic_usage/asr.md#known-limitations). Qwen3-ASR adds:

- The Torch MPS path caps both native and whole-upload requests at 60 seconds,
  rejects longer audio, and provides no token logprobs or sampling controls
  beyond `repetition_penalty`.
- Streaming requests cannot chunk and run as one engine request, bounded by
  `max_native_clip_s` (1,200 s) on CUDA and 60 seconds on Torch MPS. Use
  `stream=false` for longer uploads.

## Platform Configuration

<a id="ascend-npu-graph-execution"></a>

### Ascend NPU

See [Qwen3-ASR graph execution on Ascend NPU](../other_platforms/installation_npu.md#qwen3-asr-graph-execution) for queue settings and runtime restrictions.

After following the [Ascend NPU installation guide](../other_platforms/installation_npu.md), start the server with the shared example.

```bash
sgl-omni serve --config examples/configs/qwen3_asr_ascend910.yaml --port 8000
```

The example defaults to Qwen3-ASR-1.7B. To use 0.6B, add `--model-path Qwen/Qwen3-ASR-0.6B`. For a local checkpoint, pass its path with `--model-path` and the model identifier used in transcription requests with `--model-name`.

This example uses eager LM decode and prefill to avoid truncated outputs observed in [targeted transcription checks on Ascend910_9362](https://github.com/sgl-project/sglang-omni/pull/3199#discussion_r4510634009), while retaining the default encoder graphs.

DFlash has its own validated NPU examples on the same queue setup. NPU DFlash supports only greedy decoding (`temperature=0`): requests with a nonzero temperature are rejected because NPU verification uses argmax. NPU DFlash does not apply `repetition_penalty`: a value other than `1.0` returns HTTP 400. Serve without a draft to use the penalty; per-request plain fallback currently requires CUDA. For one Ascend device, use the BF16 eager example, which uses the same pinned draft that CUDA serves by default and sets `enable_async_decode: false` to retain the validated greedy, synchronous configuration:

```bash
sgl-omni serve --config examples/configs/qwen3_asr_1_7b_dflash_npu.yaml --port 8000
```

To enable draft and target verification decode graphs, select `examples/configs/qwen3_asr_1_7b_dflash_npu_graph.yaml` instead. Its request buckets are one and two; prefill and encoder remain eager, and Torch Compile stays disabled. Keep `SGLANG_DFLASH_EAGER_DRAFT_SAMPLER` unset or `0` to capture the reduced-vocabulary draft head. These buckets describe the validated example, not a platform concurrency limit or an optimal performance setting.

Both NPU examples explicitly select `ascend` for target and draft attention. Direct factory callers must supply the same engine overrides and `enable_encoder_cuda_graph=False`. Keep effective `context_length`, including the audio/text prefix and verification reserve, within the pinned causal draft's 32768-token attention window; a larger output budget or context override can exceed it. For offline serving, pass local snapshots with `--model-path` and `--asr.engine.speculative_draft_model_path`. Overlap, realtime, TP>1, quantization and longer contexts remain outside this validation.

<a id="apple-silicon-torch-mps"></a>

### Apple Silicon

Qwen3-ASR runs on Apple Silicon through a Torch/MPS compatibility path. Install with the [Apple Silicon installer](../other_platforms/installation_apple_silicon.md) and follow its environment-activation and `DYLD_LIBRARY_PATH` steps; TorchCodec needs Homebrew's keg-only FFmpeg libraries on the path before the server starts.

Pass an official PyTorch checkpoint such as `Qwen/Qwen3-ASR-0.6B`. The path uses one device (`tp_size=1`), greedy decoding, and the eager `torch_native`/`sdpa` profile:

```bash
export DYLD_LIBRARY_PATH="$(brew --prefix ffmpeg@7)/lib${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}"

sgl-omni serve \
  --model-path Qwen/Qwen3-ASR-0.6B \
  --model-name Qwen/Qwen3-ASR-0.6B \
  --asr.engine.max_running_requests 1 \
  --port 8000
```

Radix caching, chunked prefill, and CUDA graphs are not used by this path. The HTTP and SSE transcription interfaces above are the same as on CUDA; `stream=true` provides pseudo-streaming transcript deltas as tokens are decoded. The Apple path applies `repetition_penalty` but does not provide other sampling penalties or token logprobs yet.

The initial Torch MPS profile is bounded to a 2,048-token KV budget, uses `audio_chunking.max_audio_clip_s` for non-streaming chunks (30 seconds by default), and caps native/whole-upload requests at 60 seconds. Values above that qualified limit are rejected.

## DFlash speculative decoding

On one CUDA GPU, Qwen3-ASR-1.7B serves with its DFlash draft, [`wilsonzheng0327/Qwen3-ASR-1.7B-dflash`](https://huggingface.co/wilsonzheng0327/Qwen3-ASR-1.7B-dflash), by default, at the revision `sglang_omni/speculative/draft_registry.py` pins. This holds when the model path that launches, from `--model-path` or a config file, is `Qwen/Qwen3-ASR-1.7B`, with or without an `@<revision>` pin, or a Hugging Face cache snapshot of it, such as `${MODEL_PATH}` above. The 1.7B default does not apply on other platforms, to a stage with `tp_size` above 1, to another checkpoint, or to a copy outside the Hugging Face cache. Qwen3-ASR-0.6B remains off by default and has the opt-in profile below. The RTX 4090 and RTX 5090 profiles above turn the 1.7B draft off, since they were validated without it.

Qwen3-ASR-0.6B can opt in to its pinned Apache-2.0 DFlash draft with the memory-conservative RTX 3090 profile:

```bash
sgl-omni serve \
  --config examples/configs/qwen3_asr_0_6b_dflash_rtx3090.yaml \
  --port 8000
```

The registry pins [`wilsonzheng0327/Qwen3-ASR-0.6B-dflash`](https://huggingface.co/wilsonzheng0327/Qwen3-ASR-0.6B-dflash) at `fbf0ce97e3b76d8cd9f3da1791a8872c3d4c3a45`; an ordinary `--model-path Qwen/Qwen3-ASR-0.6B` launch stays on plain decoding. The profile uses BF16, caps running requests at 16 and gives the stage a 0.65 static memory fraction. Add `--asr.engine.disable_speculative_decoding true` to the command above for the same profile without its draft.

On CUDA, DFlash supports sampling temperatures and applies `repetition_penalty` during target verification. The draft does not see the penalty, so stronger penalties can lower acceptance. CUDA realtime sessions use the draft too. Ascend NPU uses separate opt-in examples, supports greedy decoding only, and rejects non-default repetition penalties; see [Ascend NPU](#ascend-npu).

On CUDA with the draft, async decode defaults to batches of at least two requests; a single request decodes synchronously. Set `--asr.factory.enable_async_decode false` for synchronous scheduling, or `--asr.factory.async_decode_min_batch_size 1` to overlap every batch. Draft weights and KV cache share the stage memory budget. Benchmark your workload: speedups vary with GPU, concurrency, and prefix-cache use, and high concurrency can reduce throughput. See the [DFlash benchmark report](../developer_reference/qwen3_asr_dflash_benchmarks.md) for measured results and test conditions.

To serve without the draft:

```bash
sgl-omni serve \
  --model-path Qwen/Qwen3-ASR-1.7B \
  --asr.engine.disable_speculative_decoding true \
  --port 8000
```

Send the form field `return_spec_tokens_details=true` to get `sglext.spec_tokens_details` (`spec_accept_length`, `spec_accept_rate`, the draft counts and the per-verify histogram) on `json` and `verbose_json` responses and on the streamed `transcript.text.done` event. Audio long enough to be transcribed in chunks returns none. `spec_accept_rate` and `spec_num_proposed_drafts` are left out when no drafts were proposed or the draft width is adaptive.

### Per-request plain fallback

Set `asr.factory.enable_plain_fallback` to route requests that DFlash cannot serve through ordinary decoding: requests that fit the target context but leave too little room for draft KV. CUDA already applies `repetition_penalty` during DFlash verification; this fallback does not enable that parameter on NPU.

```bash
sgl-omni serve \
  --config examples/configs/qwen3_asr_1_7b_plain_fallback.yaml \
  --port 8000
```

This option requires synchronous DFlash on one CUDA GPU, with prefill CUDA graphs, mixed chunks, and DP attention disabled. Decode CUDA graphs and radix caching can stay enabled. Both paths share the target model and KV pool, and each path caches its prefixes separately.

When both paths have work, the scheduler gives speculative batches one to two turns per plain turn, according to their live request counts. This reduces the cost of sparse fallback traffic on speculative requests, but increases latency for plain requests. The two-turn cap prevents plain requests from starving; it does not bound wall-clock waiting time because forward durations vary. Fallback requires synchronous decoding, while Qwen3-ASR DFlash on its own overlaps draft execution by default from 2 decoding requests, so even 0% fallback can cost throughput relative to the default DFlash server. Benchmark both request cohorts before enabling this option.

## Benchmarking

The benchmark entry point, its metric definitions, and the flags shared by all ASR checkpoints are documented in [ASR Model Usage](../basic_usage/asr.md#benchmarking). The command below pins a Qwen3-ASR-1.7B checkpoint and dataset revision for this example. Individual benchmark reports record their own model, draft, and runtime revisions.

```bash
sgl-omni serve \
  --model-path "${MODEL_PATH}" \
  --model-name Qwen/Qwen3-ASR-1.7B \
  --port 8000

# Sweep the full SeedTTS EN set (1088 clips), 3 repeats per concurrency:
# Set SERVER_GPU_PID to the server process PID reported by nvidia-smi.
python -m benchmarks.eval.benchmark_asr_seedtts \
  --port 8000 \
  --gpu-process-pid "${SERVER_GPU_PID}" \
  --dataset-revision 27f4c1adee83b5b29b7c4b375f6b976324bda308 \
  --model-revision 7278e1e70fe206f11671096ffdd38061171dd6e5 \
  --concurrencies 1,2,4,8,16,32,64 \
  --repeats 3 --warmup
```

Result fields, GPU sampling, and the provenance flags are documented in [ASR Model Usage](../basic_usage/asr.md#benchmarking) and the [ASR benchmark runbook](../developer_reference/asr_bench.md).

For the current-main concurrency baseline, the fixed-baseline comparison, and the per-stage bottleneck decomposition (issue #1324), see [Qwen3-ASR concurrency profile](../developer_reference/qwen3_asr_concurrency_profile.md). The benchmark's `--profile-events`, `--sample-util`, `--save-raw-dir`, and `--fingerprint` flags capture the telemetry that report uses.

## Concurrency tuning

The stock request-build settings are 8 workers and 32 pending requests; the default engine running cap is 64. GPU profiles can lower this cap to fit their memory budget. Start with a validated profile, then measure throughput, latency, and rejected requests at your expected concurrency before raising capacity.

`--asr.engine.max_running_requests` controls engine capacity and affects CUDA-graph and KV memory. The long-upload admission limit controls how much of that capacity long audio may occupy. For memory-constrained GPUs without a dedicated profile, the earlier sweep used this override:

```bash
sgl-omni serve --model-path Qwen/Qwen3-ASR-1.7B \
  --asr.engine.max_running_requests 32
```

Async decode is enabled by default. Set `--asr.factory.enable_async_decode false` to compare synchronous execution. See [Scheduling defaults](../developer_reference/qwen3_asr_concurrency_profile.md#scheduling-defaults) for prefill coalescing and the [historical capacity sweep](../developer_reference/qwen3_asr_concurrency_profile.md#capacity-sweep) for the evidence behind the defaults. These results are specific to their recorded hardware and workload.
