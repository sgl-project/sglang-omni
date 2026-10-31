# Native MLX Runtime

SGLang-Omni's native MLX runtime serves selected models on Apple Silicon. It builds C++ server binaries that use MLX and expose model-specific HTTP APIs. This is a separate runtime from the CUDA pipeline and from the Python Torch/MPS path described in the [Apple Silicon installation guide](../get_started/installation_apple_silicon.md).

## Supported models and tasks

The following models are wired into the native runtime. The listed checkpoints and serving behavior reflect the current native integration; they do not imply that every model supports every API or streaming mode.

| Model or checkpoint | Task | Native serving path |
| --- | --- | --- |
| `mlx-community/Qwen3-ASR-0.6B-4bit`, `mlx-community/Qwen3-ASR-1.7B-6bit`, `mlx-community/Qwen3-ASR-1.7B-8bit` | Speech recognition | Transcription and realtime WebSocket sessions. |
| `mlx-community/whisper-small-mlx`, `mlx-community/whisper-large-v3-mlx`, `mlx-community/whisper-large-v3-turbo` | Speech recognition | Transcription with batch preview; no realtime socket. |
| `beshkenadze/cohere-transcribe-03-2026-mlx-fp16` | Speech recognition | Transcription with batch preview; long audio can use Silero VAD. |
| `OpenMOSS-Team/MOSS-Transcribe-Diarize` | Speech recognition and speaker diarization | Timestamped speaker segments and live updates. |
| `mlx-community/nemotron-3.5-asr-streaming-0.6b-8bit` | Speech recognition | Transcription and realtime streaming. |
| `mlx-community/parakeet-tdt-0.6b-v3` | Speech recognition | Transcription with batch preview and sentence timestamps. |
| `mlx-community/SenseVoiceSmall` | Speech recognition | Transcription with language, emotion, and event metadata. Long audio can use Silero VAD. |
| `mlx-community/silero-vad-v6` | Voice activity detection | Streaming speech probabilities and offline speech ranges. |
| `mlx-community/diar_streaming_sortformer_4spk-v2.1-fp16` | Speaker diarization | Meeting speaker analysis. |
| Voxtral 4B TTS | Text-to-speech | Native speech generation server. |
| MiniMax Music3 | Music generation | Text-to-music server and synthesis CLI. |

For model-specific behavior in Voxt, see the [Voxt native runtime guide](https://github.com/sgl-project/sglang-omni/blob/main/Voxt/backend/README.md). For MiniMax Music3 checkpoint setup and requests, see the [Music3 cookbook](../cookbook/minimax_music3.md#apple-silicon-native-mlx).

## Requirements

- An Apple Silicon Mac running macOS 15 or newer. The native build defaults its deployment target to macOS 15.0.
- `uv` and the Xcode Command Line Tools, including a C++ compiler.
- An internet connection for the pinned MLX build dependencies, native C++ dependencies, and model checkpoint.
- Disk space for the checkpoint and build artifacts. The repository does not currently publish a minimum disk or memory requirement.

The build script creates a Python 3.12 build environment and installs the pinned MLX 0.32.3, CMake, and Ninja versions. Python is used to build the runtime; the resulting server is a native executable with its MLX runtime libraries beside it.

## Quickstart: Qwen3-ASR 0.6B

This quickstart builds the native Qwen3-ASR server and sends one WAV file for transcription. It uses the 4-bit checkpoint from the native runtime's pinned model list.

From the repository root, download the pinned checkpoint:

```bash
MODEL_DIR="$PWD/.build/models/qwen3-asr-0.6b-4bit"
mkdir -p "$MODEL_DIR"
uvx --from huggingface-hub hf download \
  mlx-community/Qwen3-ASR-0.6B-4bit \
  --revision 313d850181767edf09f00a9c289becca70e58cd0 \
  --local-dir "$MODEL_DIR"
```

Build the native runtime:

```bash
RUNTIME_DIR="$PWD/.build/native-mlx"
sglang_omni_mlx/native/scripts/build_runtime.sh "$RUNTIME_DIR"
```

Start the server in one terminal:

```bash
"$RUNTIME_DIR/bin/qwen3_asr_server" \
  --model-path "$MODEL_DIR" \
  --host 127.0.0.1 \
  --port 8000
```

In another terminal, check that the server is healthy:

```bash
curl --fail http://127.0.0.1:8000/health
```

Send a local WAV file for transcription:

```bash
curl --fail http://127.0.0.1:8000/v1/audio/transcriptions \
  -F file=@/path/to/audio.wav
```

The response is JSON with a `text` field. Stop the server with Ctrl-C.

## Further reading

- [Native MLX runtime source layout](https://github.com/sgl-project/sglang-omni/tree/main/sglang_omni_mlx/native).
- [Apple Silicon Torch/MPS installation](../get_started/installation_apple_silicon.md).
- [Voxt backend guide](https://github.com/sgl-project/sglang-omni/blob/main/Voxt/backend/README.md).

## Contribution guidance

### Keep the MLX path independent

Place native MLX models, kernels, and servers under `sglang_omni_mlx/`. Do not add MLX-specific workers or platform branches to `sglang_omni/`, or create a shared backend interface between the MLX and CUDA paths. Share implementation within the MLX path where appropriate.

### Reuse shared native components

Keep model-specific code under `sglang_omni_mlx/native/src/<model>/`. Put reusable audio, checkpoint, tokenizer, layer, and HTTP primitives in the shared native libraries rather than in another model's directory. Reuse existing front ends and shared components when they fit.

### Preserve and measure reference behavior

Name the reference implementation and pin its source revision. Compare the same weights and MLX version, and keep behavior changes in a separate PR. For ASR changes, report English WER, Chinese CER, and mixed MER on the 392-clip corpus, the number of identical clips, and where differing outputs diverge.

### Pin models and test runtime behavior

Pin checkpoint revisions and file SHA-256 values in `sglang_omni_mlx/native/ci/model_pins.json`. Add model goldens and server API coverage for success and invalid requests, streaming or session behavior where applicable, and shutdown or cancellation paths. Validate checkpoint configuration during load and bound memory for long-running sessions.

### Report performance and checks

Report total time, realtime factor, peak memory, and model load time against the reference, with the device and software versions stated. The [CI Workflows and Labels guide](../developer_reference/ci_labels.md) explains CI workflows, labels, and path filters. Also run applicable local checks and follow the repository's [coding style](https://github.com/sgl-project/sglang-omni/blob/main/.claude/skills/code-review/coding-style.md).
