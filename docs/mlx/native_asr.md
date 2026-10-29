# Native MLX ASR Quick Start

Run Qwen3-ASR-0.6B or Whisper small on Apple Silicon through the native C++ MLX runtime. These servers have their own build and checkpoint formats; the Python Torch/MPS installation is a separate path.

## Build the Runtime

Use an Apple Silicon Mac with Xcode command-line tools and [uv](https://docs.astral.sh/uv/). Run the setup commands from the repository root. The build script prepares Python 3.12 and the pinned build dependencies in its work directory, then installs the binaries and MLX libraries together.

```bash
RUNTIME_ROOT="$HOME/omni-runtime"
DATA_ROOT="$HOME/.cache/sglang-omni-native-asr"

bash sglang_omni_mlx/native/scripts/build_runtime.sh \
  "$RUNTIME_ROOT" "$RUNTIME_ROOT/.build-arm64"
```

Choose one model below and keep its server running while sending requests from another terminal. Both examples use the same port; stop one server before starting the other.

## Qwen3-ASR

Download the pinned MLX 4-bit checkpoint. `uvx` supplies the Hugging Face CLI independently of the runtime's build environment:

```bash
MODEL_PATH="$(
  uvx --from huggingface_hub hf download mlx-community/Qwen3-ASR-0.6B-4bit \
    --revision 313d850181767edf09f00a9c289becca70e58cd0 \
    --quiet
)"

"$RUNTIME_ROOT/bin/qwen3_asr_server" \
  --model-path "$MODEL_PATH" \
  --model-name Qwen/Qwen3-ASR-0.6B \
  --host 127.0.0.1 --port 8000
```

## Whisper Small

Provision the pinned `mlx-community/whisper-small-mlx` checkpoint and its separately pinned `openai/whisper-small` tokenizer files. The provisioning script checks the SHA-256 of every file and places the tokenizer beside `weights.npz`; downloading only the MLX weights is insufficient. `--model` limits the download to this checkpoint.

```bash
"$RUNTIME_ROOT/.build-arm64/venv/bin/python" \
  sglang_omni_mlx/native/ci/provision_models.py \
  "$DATA_ROOT" --model mlx-community/whisper-small-mlx

"$RUNTIME_ROOT/bin/whisper_server" \
  --model-path "$DATA_ROOT/models/mlx-community_whisper-small-mlx" \
  --model-name openai/whisper-small \
  --host 127.0.0.1 --port 8000
```

## Transcribe Audio

From another terminal at the repository root, check readiness and the served model:

```bash
curl --fail-with-body --silent --show-error http://127.0.0.1:8000/health
curl --fail-with-body --silent --show-error http://127.0.0.1:8000/v1/models
```

Both native servers require **16 kHz mono WAV** input, encoded as PCM16 or float32. The repository's `query_to_cars.wav` is 48 kHz; convert it before uploading. macOS provides `afconvert`:

```bash
afconvert -f WAVE -d LEI16@16000 -c 1 \
  tests/data/query_to_cars.wav /tmp/sglang-omni-native-asr.wav

curl --fail-with-body --silent --show-error --max-time 120 \
  http://127.0.0.1:8000/v1/audio/transcriptions \
  -F file=@/tmp/sglang-omni-native-asr.wav \
  -F language=en
```

A successful request returns HTTP 200 with a nonempty `text` field. Uploading the original 48 kHz file returns HTTP 400 with `audio must be 16 kHz mono`. A missing file or invalid audio also returns HTTP 400.

For an SSE response with generation metadata:

```bash
curl --fail-with-body --silent --show-error -N --max-time 120 \
  http://127.0.0.1:8000/v1/audio/transcriptions \
  -F file=@/tmp/sglang-omni-native-asr.wav \
  -F language=en \
  -F stream=true \
  -F include_generation_metadata=true
```

The native transcription endpoint sends comment heartbeats while decoding, followed by `transcript.text.done` and `data: [DONE]`. It does not send token-by-token transcript deltas. `include_generation_metadata=true` requires `stream=true`; the non-streaming combination returns HTTP 400.

Whisper's native server has no `/v1/realtime` WebSocket. See the [Voxt backend guide](https://github.com/sgl-project/sglang-omni/blob/main/Voxt/backend/README.md) for the native model support matrix and Qwen3-ASR's realtime path, and the [MLX path guide](index.md) for contribution requirements.
