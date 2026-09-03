# Audar-TTS-V1 Turbo

[Audar-TTS-V1 Turbo](https://huggingface.co/audarai/Audar-TTS-V1-Turbo) is an
Arabic text-to-speech model with reference-clip voice cloning, served through
the OpenAI-compatible speech API.

## Overview

| Item | Value |
|---|---|
| Task | TTS |
| Checkpoint(s) | `audarai/Audar-TTS-V1-Turbo` |
| Endpoint(s) | `/v1/audio/speech` |
| Pipeline | preprocessing → reference encoder → TTS engine → vocoder |
| Input / output | Text plus one reference clip → encoded audio |
| Streaming | No |
| Validated hardware | Not recorded |

## Prerequisites

Follow [Installation](../get_started/installation.md). The Turbo checkpoint
ships GGUF weights, so the engine runs through `llama-cpp-python` and needs the
optional extra:

```bash
pip install -e '.[audar-tts]'
```

For a CUDA build of llama.cpp, install `llama-cpp-python` with the build flags
your target CUDA image requires before installing SGLang-Omni.

## Deploy

Pass the checked-in configuration explicitly. The Hugging Face repository
contains GGUF weights and no Transformers `config.json`, so the architecture
cannot be resolved from `--model-path` alone:

```bash
sgl-omni serve \
  --config examples/configs/audar_tts_turbo.yaml \
  --allowed-local-media-path /path/to/references \
  --port 8000
```

## Send a request

Supply one 5-15 second reference clip and its transcript:

```bash
curl http://localhost:8000/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "audarai/Audar-TTS-V1-Turbo",
    "input": "مرحبا، أهلا وسهلا بكم.",
    "ref_audio": "file:///path/to/references/voice.wav",
    "ref_text": "النص المطابق للمقطع المرجعي.",
    "response_format": "wav"
  }' \
  --output audar.wav
```

## Capabilities

### Voice cloning

Clone a voice with `ref_audio` and `ref_text`, or the equivalent
`references[0]` form. Local reference paths require
`--allowed-local-media-path`.

### Language

The model infers output language from `input`. The optional `language` field is
accepted as metadata and is not consumed by this model.

## Configuration

Engine limits use the `tts_engine` stage, for example
`--tts_engine.engine.max_running_requests`. See
[Admission control](../user_guide/advanced_features/admission_control.md).

## Known limitations

- Streaming is not supported; the pipeline has no streaming vocoder.
- Batch vocoder, CUDA Graph, and torch.compile are not supported.
- `--model-path` alone cannot start the server; the checked-in configuration is
  required.

## Benchmark

Run the Seed-TTS benchmark against the deployed server:

```bash
python -m benchmarks.eval.benchmark_tts_seedtts \
  --use-existing-server \
  --model audarai/Audar-TTS-V1-Turbo \
  --port 8000
```

Follow the [benchmark methodology](../benchmarks/methodology.md) when
publishing results.

## Related documentation

- [Speech API](../user_guide/serving/speech_api.md)
- [Admission control](../user_guide/advanced_features/admission_control.md)
- [Benchmark methodology](../benchmarks/methodology.md)
- [Supported models](../supported_models.md)
