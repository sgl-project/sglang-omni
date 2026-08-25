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

The provided configuration enables the AR and acoustic-tail CUDA graphs. AudioVAE decode remains
eager, and requests are non-streaming unless `stream` is set.

For non-streaming requests, `audio_decode` sends the complete generated latent sequence through
one full-sequence AudioVAE decode. Streaming requests use the separate incremental AudioVAE path
with request-local cache and overlap state. Older configs need three changes, all enforced
during config loading: remove `decode_mode` from `audio_decode.factory_args` or
`runtime_overrides.audio_decode`, because non-streaming chunked decode is no longer supported;
set `tts_engine.stream_to` to `[audio_decode]` to declare the latent stream edge; and set
`audio_decode.can_accept_stream_before_payload` to `true` so the consumer accepts latents that
arrive while generation is still running. The provided YAML already carries all three.

Cross-request AudioVAE batching is not implemented yet. The only supported audio-decode batch
configuration is `max_batch_size: 1` with `max_batch_wait_ms: 0`, as shown in the provided YAML;
other values are rejected before the server starts. A future batching change can expand this
configuration only after it implements and validates a real multi-request AudioVAE decode.

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
buffers `audio_decode.factory_args.initial_chunk_patches` latent patches and emits no PCM; a
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

## Benchmark Results

### Recommended Single-H200 TP1

The recommended TP1 configuration was evaluated on **1× H200 141 GB** with concurrency 8,
eight warmup requests, and the full Seed-TTS-Eval EN and ZH splits. Streaming used two initial
patches and four steady patches. AR and acoustic-tail CUDA graphs were enabled, while AudioVAE
decode remained eager.

Streaming:

| Slice | Lang | Samples | Failed | Corpus WER | RTF Mean | Latency Mean (s) | First Audio Mean (s) | Throughput (qps) | Audio s/s |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| text-only | EN | 1088 | 0 | 0.90% | 0.2686 | 1.236 | 0.6208 | 6.462 | 30.617 |
| text-only | ZH | 2020 | 0 | 0.71% | 0.2592 | 1.278 | 0.6131 | 6.253 | 31.417 |
| reference | EN | 1088 | 0 | 1.09% | 0.3107 | 1.358 | 0.7471 | 5.883 | 26.622 |
| reference | ZH | 2020 | 0 | 0.75% | 0.2670 | 1.507 | 0.7326 | 5.303 | 30.377 |

Non-streaming:

| Slice | Lang | Samples | Failed | Corpus WER | RTF Mean | Latency Mean (s) | Throughput (qps) | Audio s/s |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| text-only | EN | 1088 | 0 | 0.91% | 0.2035 | 0.959 | 8.328 | 39.369 |
| text-only | ZH | 2020 | 0 | 0.69% | 0.1965 | 0.983 | 8.134 | 40.790 |
| reference | EN | 1088 | 0 | 1.07% | 0.2308 | 1.031 | 7.742 | 35.021 |
| reference | ZH | 2020 | 0 | 0.71% | 0.1966 | 1.125 | 7.103 | 40.737 |

All 12,432 requests completed successfully. Streaming returned its first audio payload in
0.61-0.75 seconds, while non-streaming retained higher complete-response throughput.
The worst corpus WER was 1.09%.

Streaming playback continuity:

| Slice | Lang | Scored | N/A | Underrun P95 (s) | Underrun P99 (s) | C50 | C100 | C200 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| text-only | EN | 1088 | 0 | 0.0000 | 0.0000 | 100.00% | 100.00% | 100.00% |
| text-only | ZH | 2020 | 0 | 0.0000 | 0.0000 | 100.00% | 100.00% | 100.00% |
| reference | EN | 1071 | 17 | 0.0000 | 0.0000 | 100.00% | 100.00% | 100.00% |
| reference | ZH | 2020 | 0 | 0.0000 | 0.0000 | 100.00% | 100.00% | 100.00% |

`N/A` means that a request returned one PCM payload, so it had no inter-payload seam to score.
All 11,641 later seams had zero measured playback underrun.

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
