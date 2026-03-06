# FishAudio OpenAudio-S1

Text-to-speech via the DualAR (slow+fast transformer) architecture with DAC codec vocoding.

## Quick Start

`torch.compile` and radix cache are **on by default** — no extra flags needed.

```bash
# Basic TTS
python examples/run_fishaudio_e2e.py \
    --text "Hello, how are you?" \
    --output output.wav

# Voice cloning
python examples/run_fishaudio_e2e.py \
    --text "Hello, how are you?" \
    --reference-audio ref.wav --reference-text "Transcript of ref audio." \
    --output output.wav

# Disable compile / radix cache if needed
python examples/run_fishaudio_e2e.py \
    --text "Hello" --no-compile --no-radix-cache --output output.wav
```

## Server

Launch an OpenAI-compatible HTTP server:

```bash
python examples/run_fishaudio_server.py \
    --model-id fishaudio/openaudio-s1-mini \
    --port 8000
```

### Speech API

```bash
# Basic TTS
curl http://localhost:8000/v1/audio/speech \
    -H "Content-Type: application/json" \
    -d '{"input": "Hello, how are you?"}' \
    --output output.wav

# Voice cloning
curl http://localhost:8000/v1/audio/speech \
    -H "Content-Type: application/json" \
    -d '{"input": "Hello", "ref_audio": "ref.wav", "ref_text": "transcript"}' \
    --output cloned.wav
```

### Other endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /health` | Health check |
| `GET /v1/models` | List available models |
| `POST /v1/audio/speech` | Text-to-speech |

## Pipeline

3-stage linear pipeline: `preprocessing` (CPU) &rarr; `tts_engine` (GPU) &rarr; `vocoder` (GPU).

| Stage | Executor | What it does |
|-------|----------|--------------|
| `preprocessing` | `PreprocessingExecutor` | Tokenize text, encode reference audio via DAC codec, build DualAR prompt |
| `tts_engine` | `EngineExecutor` wrapping `OmniEngine` | DualAR decode: slow transformer samples semantic token, fast transformer samples 4 codebook tokens per step |
| `vocoder` | `PreprocessingExecutor` | DAC codec decode: VQ codes &rarr; 44.1kHz audio waveform |

## CLI Reference

### `run_fishaudio_e2e.py`

| Flag | Default | Description |
|------|---------|-------------|
| `--checkpoint` | `fishaudio/openaudio-s1-mini` | HF model ID or local path |
| `--text` | `"Hello, how are you today?"` | Text to synthesize |
| `--device` | `cuda:0` | GPU device |
| `--output` / `-o` | None | Save output as WAV |
| `--reference-audio` | None | Reference WAV for voice cloning |
| `--reference-text` | `""` | Transcript of reference audio |
| `--no-compile` | off | Disable `torch.compile` (on by default) |
| `--no-radix-cache` | off | Disable radix-tree prefix cache (on by default) |
| `--max-new-tokens` | 1024 | Max decode steps |
| `--temperature` | 0.8 | Sampling temperature |
| `--top-p` | 0.8 | Top-p sampling |
| `--test-cache` | off | Run cache correctness test with 2 requests |

### `run_fishaudio_server.py`

| Flag | Default | Description |
|------|---------|-------------|
| `--model-id` | `fishaudio/openaudio-s1-mini` | HF model ID or local path |
| `--tts-device` | `cuda:0` | GPU device for TTS engine |
| `--vocoder-device` | `cuda:0` | GPU device for vocoder |
| `--max-new-tokens` | 2048 | Max decode steps |
| `--max-seq-len` | 4096 | Max sequence length for KV cache |
| `--no-compile` | off | Disable `torch.compile` (on by default) |
| `--no-radix-cache` | off | Disable radix-tree prefix cache (on by default) |
| `--relay-type` | `shm` | Relay backend (`shm`, `nccl`, `nixl`) |
| `--host` | `0.0.0.0` | Server bind address |
| `--port` | 8000 | Server port |
| `--model-name` | pipeline name | Model name for `/v1/models` |
