# Ming-Omni-TTS

[Ming-Omni-TTS-16.8B-A3B](https://huggingface.co/inclusionAI/Ming-omni-tts-16.8B-A3B)
is a mixture-of-experts audio generation model from inclusionAI. The current SGLang-Omni
serving path supports **text-to-speech** and **zero-shot voice cloning** through the
OpenAI-compatible `/v1/audio/speech` endpoint and produces **44.1 kHz** audio.

![Ming-Omni-TTS model architecture](https://github.com/inclusionAI/Ming-omni-tts/raw/main/figures/ming_omni_tts.png)

The serving pipeline keeps the SGLang autoregressive backbone and the Ming acoustic feedback
loop in one generation stage:

```text
preprocessing -> reference_encode -> tts_engine -> audio_decode
                                      |       ^
                                      +-------+
                                       latent feedback
```

`reference_encode` is a no-op for text-only requests. For voice cloning it extracts the speaker
embedding and prompt latents before the autoregressive loop starts. `tts_engine` runs the
SGLang backbone, FlowLoss/CFM acoustic tail, stop head, and feedback projection. `audio_decode`
converts the generated latent sequence into the final waveform with the Ming AudioVAE.

## Prerequisites

Install `sglang-omni` by following [Installation](../get_started/installation.md), then download
the checkpoint:

```bash
hf download inclusionAI/Ming-omni-tts-16.8B-A3B
```

The provided configuration uses TP1 on GPU 0.

## Server Configuration

```bash
sgl-omni serve \
  --model-path inclusionAI/Ming-omni-tts-16.8B-A3B \
  --config examples/configs/ming_omni_tts.yaml \
  --port 8000
```

The provided configuration enables the AR and acoustic-tail CUDA graphs and a fixed-width CUDA
graph for streaming AudioVAE transitions. Non-streaming full-sequence AudioVAE decode remains
compact eager, and requests are non-streaming unless `stream` is set.

For non-streaming requests, `audio_decode` sends the complete generated latent sequence through
one full-sequence AudioVAE decode. Streaming requests use the separate incremental AudioVAE path
with request-local cache and overlap state. Older configs need three changes, all enforced
during config loading: remove `decode_mode` from the audio_decode stage's
`factory` group, because non-streaming chunked decode is no longer supported;
set `tts_engine.stream_to` to `[audio_decode]` to declare the latent stream edge; and set
`audio_decode.can_accept_stream_before_payload` to `true` so the consumer accepts latents that
arrive while generation is still running. The provided YAML already carries all three.

Cross-request non-streaming AudioVAE batching is not implemented yet. The only supported non-streaming batch configuration is `max_batch_size: 1` with `max_batch_wait_ms: 0`, as shown in the provided YAML; other values are rejected before the server starts.

`stream_slots` is the maximum number of streaming requests that the AudioVAE decoder can keep active at the same time. Each active stream uses one slot to preserve its decoding progress between audio chunks. If all slots are occupied, additional streams wait until a slot is released. The provided configuration uses `stream_slots: 8` to match its concurrency-8 workload. Increasing it supports more simultaneous streams but uses more GPU memory and fixed-graph work; reducing it lowers those costs but also lowers streaming concurrency. It does not change non-streaming batching.

### Apple Silicon (MLX)

The native C++ server supports text-only WAV/PCM responses and streaming PCM with the official BF16 checkpoint or the pre-quantized `mlx-community/Ming-omni-tts-16.8B-A3B-4bit` checkpoint:

```bash
bash sglang_omni_mlx/native/scripts/build_runtime.sh build/native/runtime build/native/work

build/native/runtime/bin/ming_tts_server \
  --model-path "${MING_MODEL_DIR:?Set MING_MODEL_DIR to the local checkpoint directory}" \
  --model-name ming-omni-tts --host 127.0.0.1 --port 8000
```

For native supervisor integration, launch with `--supervised --model-kind ming_tts --model-directory "$MING_MODEL_DIR"`. The shared native server defaults to a free port and a generated model name; pass `--port` and `--model-name` to fix them. Supervised mode reports them in a JSON `ready` event on stdout. Send `{"command":"shutdown"}` on stdin to stop. Closing stdin also stops the process.

Native voice cloning accepts one inline `data:audio/wav;base64,...` reference in `ref_audio`, together with its transcript in `ref_text`. The WAV must be mono 16 kHz PCM16 or float32 and contain 0.1–30 seconds of audio. Request bodies are limited to 4 MiB. Both non-streaming WAV/PCM and streaming PCM are supported. CampPlus and AudioVAE reference encoding run in C++ MLX and are loaded on the first reference request. Later requests reuse the encoder weights.

Voice cloning requires `campplus.safetensors` in the model directory. The Q4 checkpoint includes it. The official BF16 checkpoint instead includes ONNX weights, which the native server does not read. For BF16, place a copy of the Q4 checkpoint's `campplus.safetensors` alongside the model files. Its FP32 tensors are identical to the official CampPlus ONNX initializers.

With the native server running, send a reference clip and its transcript:

```bash
export MING_REFERENCE_WAV=/path/to/reference-16k-mono.wav
export MING_REFERENCE_TEXT='The exact words spoken in the reference clip.'
python - <<'PY' | curl --fail-with-body http://127.0.0.1:8000/v1/audio/speech \
  -H 'Content-Type: application/json' --data-binary @- --output cloned.wav
import base64
import json
import os
from pathlib import Path

print(json.dumps({
    "model": "ming-omni-tts",
    "input": "Hello, this is a test of reference-conditioned speech.",
    "ref_text": os.environ["MING_REFERENCE_TEXT"],
    "ref_audio": "data:audio/wav;base64," + base64.b64encode(
        Path(os.environ["MING_REFERENCE_WAV"]).read_bytes()
    ).decode(),
    "response_format": "wav",
}))
PY
```

For 4-bit inference, download the MLX community checkpoint and use its local directory as `--model-path`. Quantization is read from the checkpoint; there is no native online-quantization flag. The checkpoint also quantizes acoustic and AudioVAE linear layers. The server accepts one active request; additional concurrent requests return HTTP 429. Serving does not require Python or SGLang workers.

## Synthesizing Speech

### Text Only

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "ming-omni-tts",
    "input": "SGLang-Omni is a great project!",
    "response_format": "wav"
  }' \
  --output output.wav
```

### Voice Cloning

Ming-Omni-TTS currently accepts one local reference clip and requires its transcript. Start the
server with access to the directory containing the clip:

```bash
sgl-omni serve \
  --model-path inclusionAI/Ming-omni-tts-16.8B-A3B \
  --config examples/configs/ming_omni_tts.yaml \
  --allowed-local-media-path /path/to/references \
  --port 8000
```

Then submit the reference as a `file://` URL:

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "ming-omni-tts",
    "input": "Get the trust fund to the bank early.",
    "references": [{
      "audio_path": "file:///path/to/references/prompt.wav",
      "text": "We asked over twenty different people, and they all said it was his."
    }],
    "response_format": "wav"
  }' \
  --output cloned.wav
```

`ref_audio` and `ref_text` are accepted as shorthand for the single `references` item.

### Streaming

Streaming returns headerless mono signed 16-bit little-endian PCM (`s16le`) at 44.1 kHz with
`Content-Type: audio/pcm`. The `X-Sample-Rate`, `X-Channels`, and `X-Bit-Depth` headers report the
sample rate, channel count, and bit depth; HTTP EOF ends the stream.

Ming AudioVAE uses separate initial and steady cadence settings. Its first non-terminal call
buffers `stages.audio_decode.factory.initial_chunk_patches` latent patches and emits no PCM; a
later call supplies the right-hand context needed to emit that initial group. Subsequent calls
consume `steady_chunk_patches` at a time. The provided configuration uses two initial patches and
four steady patches, and the terminal step flushes any remainder. Pipe the response to `ffplay`
to play it during generation:

```bash
curl -sS --fail --no-buffer -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "ming-omni-tts",
    "input": "SGLang-Omni supports streaming speech generation.",
    "stream": true,
    "response_format": "pcm"
  }' \
  | ffplay -nodisp -autoexit -f s16le -ar 44100 -ac 1 -
```

To save the raw stream instead, use `--output output.pcm`. The file has no WAV header; convert it
with `ffmpeg -f s16le -ar 44100 -ac 1 -i output.pcm output.wav`.

## Generation Parameters

| Parameter | Default | Notes |
|---|---|---|
| `input` | (required) | Non-empty text to synthesize |
| `references` | `null` | At most one local reference clip with non-empty `text` |
| `ref_audio` / `ref_text` | `null` | Shorthand for the reference clip and transcript |
| `max_new_tokens` | `200` when omitted | Per-request upper bound on acoustic generation steps. The provided configuration accepts values from `1` to `256`; generation may stop earlier |
| `temperature` | `0.0` when omitted | Non-negative SDE temperature used by the FlowLoss sampler |
| `response_format` | `wav` | Use `pcm` when `stream` is enabled; `wav` is used by the reference benchmark |
| `stream` | `false` | Streams raw PCM audio when enabled |
| `voice` | `default` | Only the default voice selector is accepted |
| `speed` | `1.0` | Other speed values are not supported |

Advanced FlowLoss controls can be passed through `stage_params.tts_engine`:

```json
{
  "stage_params": {
    "tts_engine": {
      "cfg": 2.0,
      "sigma": 0.25,
      "temperature": 0.0
    }
  }
}
```

`cfg` must be at least `1e-5` and cannot equal `1.0`; `sigma` and `temperature` must be
non-negative.

## Benchmarking

The benchmark uses Seed-TTS-Eval with concurrency 8. Run each row below for
both `en` and `zh`, replacing `{lang}` in the output directory:

| Response mode | Input mode | Scenario flags | Output directory |
|---|---|---|---|
| Non-streaming | Reference | _(none)_ | `results/ming_tts/nonstream/reference/{lang}` |
| Non-streaming | Text-only | `--no-ref-audio` | `results/ming_tts/nonstream/text_only/{lang}` |
| Streaming | Reference | `--stream` | `results/ming_tts/stream/reference/{lang}` |
| Streaming | Text-only | `--stream --no-ref-audio` | `results/ming_tts/stream/text_only/{lang}` |

With the Ming-TTS server running on port 8000, generate each scenario by
substituting its language, flags, and output directory in this command:

```bash
python -m benchmarks.eval.benchmark_tts_seedtts \
  --generate-only --use-existing-server \
  --base-url http://127.0.0.1:8000 \
  --model ming-omni-tts \
  --meta zhaochenyang20/seed-tts-eval-arrow \
  --output-dir <output-directory> \
  --lang <lang> --ref-format references \
  --max-new-tokens 256 --max-concurrency 8 --warmup 8 \
  <scenario-flags>
```

After generation finishes, stop the TTS server and start the ASR server in
another terminal:

```bash
sgl-omni serve \
  --model-path Qwen/Qwen3-ASR-1.7B \
  --port 8100
```

Then transcribe each output directory with the same language and scenario
flags used for generation:

```bash
python -m benchmarks.eval.benchmark_tts_seedtts \
  --transcribe-only --use-existing-server \
  --host 127.0.0.1 --port 8100 \
  --model ming-omni-tts \
  --meta zhaochenyang20/seed-tts-eval-arrow \
  --output-dir <output-directory> \
  --lang <lang> --ref-format references \
  --max-new-tokens 256 --max-concurrency 8 --asr-concurrency 1 \
  <scenario-flags>
```

### Native Apple Silicon evaluation

The native entry point manages servers and reuses the shared SeedTTS evaluator. Run it from the repository root using a Python environment with the repository's benchmark dependencies, not the minimal native build environment. Model arguments accept HuggingFace IDs or local directories. IDs are resolved through the HuggingFace cache, downloading missing files before starting each server.

```bash
MING_EVAL_DIR="$(mktemp -d /tmp/ming-native-seedtts.XXXXXX)"

python -m sglang_omni_mlx.native.ci.evaluate_tts \
  --runtime-bin build/native/work/cmake \
  --model inclusionAI/Ming-omni-tts-16.8B-A3B \
  --asr-model-path mlx-community/Qwen3-ASR-1.7B-8bit \
  --asr-model-revision a8379a2e2f9e313c9292cdf1af4055ab56d50d55 \
  --meta zhaochenyang20/seed-tts-eval-arrow --lang zh \
  --max-samples 50 \
  --output-dir "$MING_EVAL_DIR/nonstream/zh"
```

This command starts Ming, generates audio, stops Ming, starts Qwen3-ASR, scores the saved audio, and stops ASR. Native mode defaults to text-only, concurrency 1, one warmup request, a 256-token generation limit, and no speech-outcome queries. It does not run CUDA cleanup. Stop manually launched servers first, or choose a free `--port`. Logs are saved under `server_logs/` in the output directory.

The model IDs above are the native defaults. Use `--model-revision` and `--asr-model-revision` to pin Hub checkpoints; these flags do not apply to local directories. `HF_HUB_OFFLINE=1` requires the requested checkpoints to be cached. Record the ASR checkpoint and revision when reporting scores: this MLX 8-bit evaluator is not the same precision as CUDA CI's Qwen3-ASR-1.7B evaluator.

For pre-quantized Ming, pass `--model mlx-community/Ming-omni-tts-16.8B-A3B-4bit` and `--model-revision 97752f43c4d7a6b3f045f484c798bf6bb1f950df`, using a separate output directory from BF16 evaluation.

Repeat for `--lang en`, saving to `nonstream/en`. For streaming, add `--stream` and save to `stream/zh` or `stream/en`. This is a 50-sample smoke evaluation; omit `--max-samples` for the full splits. `--generate-only` and `--transcribe-only` still support running either phase separately.

For voice cloning, add `--voice-clone` and use a separate output directory. The evaluator converts each SeedTTS reference to a 16 kHz mono PCM16 data URL before timing generation, preserving the reference transcript and original audio files. The model directory must contain `campplus.safetensors`. Both streaming and non-streaming evaluation are supported. Keep `--voice-clone` when running `--transcribe-only` so the WER result records the correct input mode.

The evaluator prepares 16 kHz mono copies for native ASR under `asr/`, preserving the original WAVs and `generated.json`. Scoring results are saved to `asr/wer_results.json` and `asr/asr_speed_results.json`. Report EN WER / ZH CER together with generation and ASR failure counts; failed samples must not be hidden by the error-rate average. These are measured quality results, not PCM golden comparisons or calibrated Apple-specific CI thresholds. Speaker similarity is not evaluated.

## Benchmark Results

### Recommended Single-H200 TP1

The recommended TP1 configuration was evaluated on **1× H200 141 GB** with concurrency 8, eight warmup requests, and the full Seed-TTS-Eval EN and ZH splits. Streaming used two initial patches followed by four-patch steady groups, with the AR, acoustic-tail, and streaming AudioVAE CUDA graphs enabled. Non-streaming requests continued to use compact full-sequence AudioVAE eager decode.

Streaming:

| Slice | Lang | Samples | Failed | Corpus WER/CER | RTF Mean | Latency Mean (s) | First Audio Mean (s) | Throughput (qps) | Audio s/s |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| text-only | EN | 1088 | 0 | 0.95% | 0.1620 | 0.764 | 0.3380 | 10.456 | 49.428 |
| text-only | ZH | 2020 | 0 | 0.64% | 0.1577 | 0.789 | 0.3283 | 10.124 | 50.770 |
| reference | EN | 1088 | 0 | 1.25% | 0.1942 | 0.863 | 0.4501 | 9.247 | 41.769 |
| reference | ZH | 2020 | 0 | 0.66% | 0.1659 | 0.953 | 0.4081 | 8.381 | 48.285 |

Non-streaming:

| Slice | Lang | Samples | Failed | Corpus WER/CER | RTF Mean | Latency Mean (s) | Throughput (qps) | Audio s/s |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| text-only | EN | 1088 | 0 | 0.95% | 0.1413 | 0.667 | 11.965 | 56.721 |
| text-only | ZH | 2020 | 0 | 0.64% | 0.1335 | 0.668 | 11.959 | 60.040 |
| reference | EN | 1088 | 0 | 1.26% | 0.1573 | 0.696 | 11.469 | 51.688 |
| reference | ZH | 2020 | 0 | 0.67% | 0.1254 | 0.718 | 11.127 | 63.931 |

All 12,432 requests completed successfully. Streaming returned its first audio payload in 0.33-0.45 seconds, while non-streaming retained higher complete-response throughput. The worst corpus WER was 1.26%, and the worst corpus CER was 0.67%.

Streaming playback continuity:

| Slice | Lang | Scored | N/A | Underrun P95 (s) | Underrun P99 (s) | C50 | C100 | C200 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| text-only | EN | 1088 | 0 | 0.0000 | 0.0000 | 100.00% | 100.00% | 100.00% |
| text-only | ZH | 2020 | 0 | 0.0000 | 0.0000 | 100.00% | 100.00% | 100.00% |
| reference | EN | 1072 | 16 | 0.0000 | 0.0000 | 100.00% | 100.00% | 100.00% |
| reference | ZH | 2020 | 0 | 0.0000 | 0.0000 | 100.00% | 100.00% | 100.00% |

`N/A` means that a request returned one PCM payload, so it had no inter-payload seam to score. Every scored slice has zero P95 and P99 measured playback underrun.

## Known Limitations

- **Serving optimizations.** Prefix/radix cache and `torch.compile` are not supported and remain
  disabled in the provided configuration.
- **Reference inputs.** The current request adapter accepts one local reference audio file with a
  non-empty transcript; remote URLs, data URLs, precomputed prompt latents, and speaker embeddings
  are not supported.
- **Generation controls.** Request-local `seed`, logits sampling fields (`top_p`, `top_k`,
  `repetition_penalty`), named voices, explicit language selection, instructions, and duration
  control are not supported. `initial_codec_chunk_frames` is rejected because AudioVAE cadence
  is a pipeline-level setting.
- **Checkpoint coverage.** The current serving implementation supports only the 16.8B-A3B MoE
  checkpoint; the dense 0.5B checkpoint is not supported.
