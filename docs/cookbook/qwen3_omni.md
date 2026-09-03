# Qwen3-Omni

[Qwen3-Omni](https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Instruct)
accepts text, image, audio, and video and returns text or text plus 24 kHz audio.

## Overview

| Item | Value |
|---|---|
| Task | Omni |
| Checkpoint(s) | `Qwen/Qwen3-Omni-30B-A3B-Instruct` |
| Endpoint(s) | `/v1/chat/completions`, `/v1/realtime` |
| Text pipeline | preprocessing/encoders → multimodal aggregate → thinker → decode |
| Speech pipeline | preprocessing/encoders → thinker and talker AR → decode/code2wav |
| Input / output | Text, image, audio, or video → text and optional audio |
| Streaming | Chat SSE and realtime WebSocket; text and optional audio output |
| Validated hardware | H100 |

## Prerequisites

Follow [Installation](../get_started/installation.md). No additional
model-specific package is required.

## Deploy

Start one worker with the H100 BF16 profile:

```bash
sgl-omni serve \
  --config examples/configs/qwen3_omni_colocated_h100_bf16.yaml \
  --colocate \
  --port 8008
```

## Send a request

This example combines image and text input and requests text plus audio. Use a
speech-mode server for audio output.

```bash
curl -X POST http://localhost:8008/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-omni",
    "messages": [{"role": "user", "content": "How many cars are there?"}],
    "images": ["tests/data/cars.jpg"],
    "modalities": ["text", "audio"],
    "max_tokens": 16
  }'
```

## Capabilities

- Text-only mode runs a six-stage pipeline with a multimodal aggregation stage.
  Speech mode uses a seven-stage topology: its encoders join directly into the
  thinker and talker, so it does not include the separate aggregation stage.
- Any supported input modality can produce text. Speech mode can additionally
  return audio with `modalities: ["text", "audio"]`.
- Native BF16, native FP8, and an AutoRound INT4 thinker with BF16
  talker/code2wav are supported in the documented topologies.
- The speech pipeline supports streaming chat and server-VAD realtime input;
  the shared transport contracts are documented in [Streaming](../user_guide/advanced_features/streaming.md).
- Disaggregated thinker TP=1 or TP=2 is supported. Colocated speech requires
  thinker TP=1 and explicit per-stage memory budgets.

See [Omni model usage](../basic_usage/qwen3_omni.md) for complete modality
examples, model-specific placement measurements, precision details, and
sampling fields. Shared placement behavior is documented in
[Stage placement](../user_guide/deployment/stage_placement.md).

## Configuration

Use `examples/configs/qwen3_omni_colocated_h100_fp8.yaml` with the H100 FP8
checkpoint. H20 and H200 profiles are also checked in as
`qwen3_omni_colocated_h20.yaml` and `qwen3_omni_colocated_h200.yaml`; their
presence records an available topology, not runtime validation on those
accelerators.

See [Omni model usage](../basic_usage/qwen3_omni.md) for the command selector,
text-only and speech topology choices, tensor parallelism, precision options,
and sampling fields. Config-file composition and command-line precedence follow
the shared [configuration contract](../developer_reference/config.md).

## Limitations

- A text-only server accepts `modalities: ["text", "audio"]` but returns no
  audio; use a speech-mode server when audio output is required.
- Use an empty message `content` when the request's semantic input is entirely
  in `images`, `audios`, or `videos`. Non-empty content is processed as an
  additional text input.
- Colocated speech does not support thinker TP=2. Use disaggregated placement.
- Requests are rejected when prompt tokens or prompt plus requested output meet
  or exceed the model context length.

## Benchmark

Use MMMU as the canonical image-plus-text benchmark:

```bash
python benchmarks/eval/benchmark_omni_mmmu.py \
  --model qwen3-omni \
  --host localhost \
  --port 8008
```

Follow the [benchmark methodology](../benchmarks/methodology.md) when
publishing results.

## Related documentation

- [Omni model usage](../basic_usage/qwen3_omni.md)
- [Omni router](../basic_usage/omni_router.md)
- [Streaming](../user_guide/advanced_features/streaming.md)
- [Stage placement](../user_guide/deployment/stage_placement.md)
- [Benchmark methodology](../benchmarks/methodology.md)
- [Pipeline architecture](../developer_reference/pipeline.md)
- [Supported models](../supported_models.md)
