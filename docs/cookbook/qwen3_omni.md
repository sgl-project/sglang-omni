# Qwen3-Omni

[Qwen3-Omni](https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Instruct) is a multi-modal model
that accepts text, image, audio, and video input and can produce text-only or text + audio output.
This page covers every supported server configuration — use the generator to get the exact launch
command for your hardware, then check the tables to confirm your combination is supported.

## Prerequisites

```bash
docker pull hongccc/sglang-omni:dev
docker run -it --shm-size 32g --gpus all hongccc/sglang-omni:dev /bin/zsh
```

```bash
pip install --upgrade pip
pip install uv

uv venv .venv -p 3.12 && source .venv/bin/activate
uv pip install --prerelease=allow "sglang-omni==0.1.4"
```

See [Installation](../get_started/installation.md) for Docker digests and source installs.

<a id="apple-silicon-mlx-and-torch-mps"></a>
### Apple Silicon (MLX and Torch MPS)

Qwen3-Omni also runs on macOS Apple Silicon (`arm64`), reusing the same Apple
platform policy, MPS device selection, and `SGLANG_USE_MLX` backend switch
introduced for [Qwen3-ASR](qwen3_asr.md#apple-silicon-mlx). Install with
[`install.sh`](../../install.sh) (see
[Installation](../get_started/installation.md#macos-apple-silicon)) or build
the environment manually as described in the Qwen3-ASR Apple guide — the same
`.venv-apple` environment and SGLang `all_mps` extra serve both models.

Two Apple backends are supported, each with a tested checkpoint layout:

| Backend | Env var | Tested checkpoint layout |
|---|---|---|
| Torch MPS (default) | `SGLANG_USE_MLX` unset | Dense, officially supported split Hugging Face checkpoint (thinker/talker/code2wav weights plus the official processor/tokenizer assets). |
| Torch MPS weight-only quantization | `SGLANG_USE_MLX` unset; `SGLANG_QWEN3_OMNI_MPS_QUANTIZATION=int4` or `int8` | Dense Hugging Face weights or root-namespaced MLX affine packed weights, converted at load time for native Torch MPS operators. |
| MLX | `SGLANG_USE_MLX=1` | Direct launch of the downloaded pinned `mlx-community/Qwen3-Omni-30B-A3B-Instruct-4bit` directory described below. No extra artifact generation or copy step is required. |

Launch commands and the full Apple runtime profile (one Metal device, greedy
generation, eager execution, SHM transport, no CUDA-only features) are in the
[Qwen3-Omni usage guide](../basic_usage/qwen3_omni.md#apple-silicon-mlx-and-torch-mps).

Before attempting a production-size MLX checkpoint, confirm it is at most
30 GiB on disk and that its documented or measured peak working set is at most
40 GiB. The default backend tests in
`tests/test_ci/test_qwen3_omni_apple.py` use deterministic test-sized weights.
Its opt-in community-checkpoint core and semantic matrices instead load the
pinned trained 30B 4-bit checkpoint described below (see `tests/README.md`).
Passing the tiny tests alone does not establish production-size checkpoint
memory safety or semantic correctness.

#### Public 4-bit checkpoint

The public
[`mlx-community/Qwen3-Omni-30B-A3B-Instruct-4bit`](https://huggingface.co/mlx-community/Qwen3-Omni-30B-A3B-Instruct-4bit)
checkpoint can be used for Apple MLX serving. Pin revision
`93b3cbddd65ed4babff8f22fba491cdba7a21778` so the commands and tensor layout
remain reproducible.

Other MLX-compatible 4-bit layouts may also load when they satisfy the Apple
checkpoint validator, but the pinned mlx-community checkpoint above is the
tested and recommended deployment. Validator-accepted examples include
component-local thinker/talker shards and the root-namespaced MLX-VLM layout.

Current ownership for the Apple MLX path is:

| Runtime | Components |
|---|---|
| Native MLX | vision, audio, thinker, talker, code predictor, code2wav |
| CPU | preprocessing, token decoding |

Native MLX means those model components stay in MLX for the Apple launch, but
it does **not** imply radix cache, multi-request batching, or CUDA-oriented
optimizations.

The native implementation requires `mlx>=0.32.2` and `mlx-lm>=0.31.2`,
without an `mlx-vlm` dependency. It reuses MLX's fused SDPA (including vision
head dimension 72), normalization and standard RoPE kernels, plus MLX-LM's
KV caches and routed expert layers. Three-axis M-RoPE uses `mx.compile` to
reuse its graph and fuse elementwise operations while retaining external
multimodal positions. Vision attention still bounds query chunks to guard
against quadratic score buffers when a shape takes the unfused path.

Set the repository and environment paths:

```bash
export REPO="/path/to/sglang-omni"
export PY="$REPO/.venv-apple/bin/python"
export MODEL_DIR="$HOME/models/Qwen3-Omni-30B-A3B-Instruct-4bit-93b3cbdd"
export MODEL_REVISION="93b3cbddd65ed4babff8f22fba491cdba7a21778"
cd "$REPO"
```

Download the exact checkpoint:

```bash
"$PY" - <<'PY'
import os
from huggingface_hub import snapshot_download

path = snapshot_download(
    repo_id="mlx-community/Qwen3-Omni-30B-A3B-Instruct-4bit",
    revision=os.environ["MODEL_REVISION"],
    local_dir=os.path.expanduser(os.environ["MODEL_DIR"]),
)
print(path)
PY
```

The canonical production-size Apple MLX launch points `sgl-omni serve`
directly at that downloaded directory with `SGLANG_USE_MLX=1`. Keep the CLI in
the foreground for normal operation so logs and Ctrl-C remain attached to the
supervising terminal:

```bash
SGLANG_USE_MLX=1 "$PY" -m sglang_omni.cli serve \
  --model-path "$MODEL_DIR" \
  --host 127.0.0.1 \
  --port 8008
```

For MLX text-only serving, use the same CLI with `--text-only`:

```bash
SGLANG_USE_MLX=1 "$PY" -m sglang_omni.cli serve \
  --model-path "$MODEL_DIR" \
  --text-only \
  --host 127.0.0.1 \
  --port 8008
```

Poll readiness from a second terminal:

```bash
until curl -fsS http://127.0.0.1:8008/v1/models >/dev/null; do
  sleep 2
done
```

Ctrl-C in the foreground terminal, or the external supervisor managing that
foreground process, terminates the canonical launcher.

For automated Bash smoke-test scripts only, start the same command in the
background and capture its exact process id. Use a ten-minute readiness deadline
(each HTTP probe is bounded to five seconds), fail early if the child exits, and
show the server log tail on readiness failure. The traps stop only that child,
including when readiness or a smoke request fails:

```bash
set -e

SGLANG_USE_MLX=1 "$PY" -m sglang_omni.cli serve \
  --model-path "$MODEL_DIR" \
  --host 127.0.0.1 \
  --port 8008 >qwen3-omni-apple.log 2>&1 &
SERVER_PID=$!
trap 'kill -TERM "$SERVER_PID" 2>/dev/null || true; wait "$SERVER_PID" || true' EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

DEADLINE=$((SECONDS + 600))
while true; do
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "Qwen3-Omni exited before readiness; server log:" >&2
    tail -n 100 qwen3-omni-apple.log >&2
    exit 1
  fi
  if (( SECONDS >= DEADLINE )); then
    echo "Timed out waiting for /v1/models after 600 seconds; server log:" >&2
    tail -n 100 qwen3-omni-apple.log >&2
    exit 1
  fi
  if curl -fsS --max-time 5 http://127.0.0.1:8008/v1/models >/dev/null; then
    break
  fi
  sleep 2
done

# Run smoke requests here.

kill -TERM "$SERVER_PID"
wait "$SERVER_PID" || true
trap - EXIT INT TERM
```

Apple scheduler restrictions remain explicit on the native MLX path:

- `tp_size=1`
- one resident request (`max_running_requests=1`)
- greedy generation only
- radix disabled
- overlap disabled
- mixed/chunked prefill disabled
- CUDA graphs disabled
- logprobs unsupported
- no partial talker start

Dense Torch MPS is a separate Apple mode selected with `SGLANG_USE_MLX` unset. It
uses the official split Hugging Face checkpoint, requires substantially more
unified memory, and is not production-qualified. For dense MPS text-only
testing:

```bash
env -u SGLANG_USE_MLX "$PY" -m sglang_omni.cli serve \
  --model-path /absolute/path/to/Qwen3-Omni-30B-A3B-Instruct \
  --text-only \
  --host 127.0.0.1 \
  --port 8008
```

For dense MPS speech-mode testing, omit `--text-only` explicitly:

```bash
env -u SGLANG_USE_MLX "$PY" -m sglang_omni.cli serve \
  --model-path /absolute/path/to/Qwen3-Omni-30B-A3B-Instruct \
  --host 127.0.0.1 \
  --port 8008
```

Expect higher memory use and treat the generated WAV as structurally valid
only, not semantically production-qualified.

For native Torch MPS INT4, reuse the downloaded community checkpoint:

```bash
env -u SGLANG_USE_MLX SGLANG_QWEN3_OMNI_MPS_QUANTIZATION=int4 \
  "$PY" -m sglang_omni.cli serve \
  --model-path "$MODEL_DIR" \
  --host 127.0.0.1 \
  --port 8008
```

Replace `int4` with `int8` for per-output-channel INT8. Leave the variable
unset for the original dense path. Quantized linears and routed experts
execute through PyTorch's native MPS kernels; no MLX inference or TorchAO is
used in this mode. Floating-point embeddings, convolutions, router weights,
prompt projections, activations, and KV caches still consume memory.
An INT8 conversion of a 4-bit source uses more storage without recovering
the source's lost precision. AWQ, compressed-tensors, and GPTQ formats are
not supported. This option does not change the conservative serving profile
or establish semantic production qualification.

Send a text request:

```bash
curl -fsS http://127.0.0.1:8008/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3-omni",
    "messages": [{"role": "user", "content": "Say hello in one short sentence."}],
    "modalities": ["text"],
    "max_tokens": 32,
    "temperature": 0,
    "top_p": 1,
    "top_k": -1
  }' | tee text-response.json
```

Keep the foreground native MLX speech server running for the following text and
audio examples.

Send a non-streamed text-and-audio request:

```bash
curl -fsS http://127.0.0.1:8008/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3-omni",
    "messages": [{"role": "user", "content": "Say hello in one short sentence."}],
    "modalities": ["text", "audio"],
    "audio": {"voice": "chelsie", "format": "wav"},
    "max_tokens": 32,
    "temperature": 0,
    "top_p": 1,
    "top_k": -1,
    "talker_temperature": 0,
    "talker_top_p": 1,
    "talker_top_k": -1,
    "talker_max_new_tokens": 128
  }' | tee speech-response.json
```

Decode and inspect the returned WAV:

```bash
"$PY" - <<'PY'
import base64
import json
import soundfile as sf

response = json.load(open("speech-response.json", encoding="utf-8"))
message = response["choices"][0]["message"]
open("speech-response.wav", "wb").write(base64.b64decode(message["audio"]["data"]))
audio, rate = sf.read("speech-response.wav")
print({"text": message["content"], "samples": len(audio), "sample_rate": rate})
PY
file speech-response.wav
```

Send the same request over SSE:

```bash
curl -NfsS http://127.0.0.1:8008/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3-omni",
    "messages": [{"role": "user", "content": "Say hello in one short sentence."}],
    "modalities": ["text", "audio"],
    "audio": {"voice": "chelsie", "format": "wav"},
    "max_tokens": 32,
    "temperature": 0,
    "top_p": 1,
    "top_k": -1,
    "talker_temperature": 0,
    "talker_top_p": 1,
    "talker_top_k": -1,
    "talker_max_new_tokens": 128,
    "stream": true
  }' | tee speech-response.sse
grep -F 'data: [DONE]' speech-response.sse
```

Run the opt-in real-checkpoint qualification, which repeats text and speech
requests in one server lifetime managed by the test harness:

```bash
export QWEN3_OMNI_APPLE_REAL_MODEL="$MODEL_DIR"
export QWEN3_OMNI_APPLE_REAL_REVISION="$MODEL_REVISION"
export QWEN3_OMNI_APPLE_ARTIFACTS="$PWD/.qwen3-omni-apple-artifacts"
"$PY" -m pytest \
  tests/test_ci/test_qwen3_omni_apple.py::test_qwen3_omni_mlx_community_core_matrix \
  -v -s
```

#### Real semantic qualification

The core matrix above verifies serving structure and transport. The separate
real semantic qualification verifies the meaning of every text response and
of every synthesized speech response; it is not the deterministic tiny
checkpoint matrix.

The fixture generator requires the macOS `say` and `sw_vers` commands, FFmpeg
on `PATH`, and `/System/Library/Fonts/Supplemental/Arial Bold.ttf`. The real
qualification fails if these prerequisites are missing; its font is not
replaced by the portable font used in unit tests.
Speech verification uses the following pinned Whisper checkpoint and downloads
approximately 967 MB on first use:

| Field | Pinned value |
|---|---|
| Model | `openai/whisper-small.en` |
| Revision | `e8727524f962ee844a7319d92be39ac1bd25655a` |
| `model.safetensors` size | `966992008` bytes |
| `model.safetensors` SHA-256 | `6014ac49b506df900f66f4aca6b0801eed7245594ace97bcaf73e0ae5b863066` |

Run the complete Apple serving file, including the opt-in semantic matrix,
against the pinned downloaded checkpoint, using the `REPO`, `PY`, `MODEL_DIR`,
and `MODEL_REVISION` variables set above:

```bash
cd "$REPO"
export QWEN3_OMNI_APPLE_REAL_MODEL="$MODEL_DIR"
export QWEN3_OMNI_APPLE_REAL_REVISION="$MODEL_REVISION"
export QWEN3_OMNI_APPLE_ARTIFACTS="$PWD/.qwen3-omni-apple-artifacts"
export SGLANG_USE_MLX=1
"$PY" -m pytest \
  tests/test_ci/test_qwen3_omni_apple.py -v -s
```

Artifacts are written beneath
`$QWEN3_OMNI_APPLE_ARTIFACTS/mlx-community-semantic-real/`. They include
`semantic-run.json`, generated fixtures and provenance, and the server log.
The manifests record backend/library versions, fixture tool versions, and
measured input-audio durations. Because `say` has no standalone version, its
provenance records the macOS version/build and executable SHA-256.

Under `cases/`, `*.request.json` is saved before sending, and `*.sse` records
wire bytes before parsing (including raw HTTP error bodies and interrupted
streams). Parsed JSON retains partial text and failure diagnostics. Each
`*.audio-NNN.wav` preserves an original audio chunk, including malformed
containers; a combined `*.wav` is written only when decoding succeeds.

A semantic pass means every case returned all explicit required facts, every
text-only response omitted audio, and every requested speech response passed
the waveform, transcription, and content-recall checks. A failure can be an
honest model or backend quality limitation. Preserve the artifacts and report
the missing facts; never hide a qualification failure by weakening the
assertions or thresholds.

Current real semantic qualification is 6/13: text, audio, and video controlled
facts pass; image OCR still misses `42`; requested speech produces a structurally
valid WAV, but the generated speech is not semantically qualified. Treat the
pinned community-checkpoint native MLX path as the tested and recommended
production-size serving route, while keeping
dense MPS and generated speech out of production qualification until these
failures are resolved.

## Server Configuration

Use the selector below to generate the exact launch command for your configuration.

```{raw} html
<div id="sgl-server-gen-mount"></div>
```

## Compatibility Matrix

Colocated topology requires `--config examples/configs/qwen3_omni_colocated_h20.yaml`
(or `qwen3_omni_colocated_h200.yaml` on H200) to set per-stage GPU memory budgets.

| Mode | Topology | Thinker TP | Precision | Status |
|---|---|---|---|---|
| Thinker-only | — | — | BF16 | ✅ |
| Thinker-only | — | — | FP8 | ✅ |
| Thinker-only | — | — | AutoRound INT4 | ✅ |
| Thinker-Talker | Disaggregated | TP=1 | BF16 | ✅ |
| Thinker-Talker | Disaggregated | TP=1 | FP8 | ✅ |
| Thinker-Talker | Disaggregated | TP=1 | AutoRound INT4 thinker + BF16 talker/code2wav | ✅ |
| Thinker-Talker | Disaggregated | TP=2 | BF16 | ✅ |
| Thinker-Talker | Disaggregated | TP=2 | FP8 | ✅ |
| Thinker-Talker | Disaggregated | TP=2 | AutoRound INT4 thinker + BF16 talker/code2wav | ✅ |
| Thinker-Talker | Colocated | TP=1 | BF16 | ✅ |
| Thinker-Talker | Colocated | TP=1 | FP8 | ✅ |
| Thinker-Talker | Colocated | TP=1 | AutoRound INT4 thinker + BF16 talker/code2wav | ✅ |

## Input / Output Modalities

All input modality combinations work with both text-only and speech servers.
`modalities: ["text", "audio"]` requires a **speech-mode server** (omit `--text-only`).

| Input | Output | Speech server | Minimal request body | Notes |
|---|---|---|---|---|
| Text | Text | No | `{"messages": [{"role": "user", "content": "..."}], "modalities": ["text"]}` | — |
| Image + text | Text | No | `{"messages": [{"role": "user", "content": "..."}], "images": ["path/or/url"], "modalities": ["text"]}` | — |
| Audio | Text | No | `{"messages": [{"role": "user", "content": ""}], "audios": ["path/or/url"], "modalities": ["text"]}` | content must be "" when the query is spoken |
| Image + audio | Text | No | `{"messages": [{"role": "user", "content": ""}], "images": ["path/or/url"], "audios": ["path/or/url"], "modalities": ["text"]}` | content must be "" when the query is spoken |
| Image | Text | No | `{"messages": [{"role": "user", "content": ""}], "images": ["path/or/url"], "modalities": ["text"]}` | content must be "" when query comes from image |
| Video + text | Text | No | `{"messages": [{"role": "user", "content": "..."}], "videos": ["path/or/url"], "modalities": ["text"]}` | — |
| Video + audio | Text | No | `{"messages": [{"role": "user", "content": ""}], "videos": ["path/or/url"], "audios": ["path/or/url"], "modalities": ["text"]}` | content must be "" when the query is spoken |
| Video | Text | No | `{"messages": [{"role": "user", "content": ""}], "videos": ["path/or/url"], "modalities": ["text"]}` | content must be "" when query comes from video |
| Text | Text + Audio | **Yes** | `{"messages": [{"role": "user", "content": "..."}], "modalities": ["text", "audio"]}` | — |
| Image + text | Text + Audio | **Yes** | `{"messages": [{"role": "user", "content": "..."}], "images": ["path/or/url"], "modalities": ["text", "audio"]}` | — |
| Audio | Text + Audio | **Yes** | `{"messages": [{"role": "user", "content": ""}], "audios": ["path/or/url"], "modalities": ["text", "audio"]}` | content must be "" when the query is spoken |
| Image + audio | Text + Audio | **Yes** | `{"messages": [{"role": "user", "content": ""}], "images": ["path/or/url"], "audios": ["path/or/url"], "modalities": ["text", "audio"]}` | content must be "" when the query is spoken |
| Image | Text + Audio | **Yes** | `{"messages": [{"role": "user", "content": ""}], "images": ["path/or/url"], "modalities": ["text", "audio"]}` | content must be "" when query comes from image |
| Video + text | Text + Audio | **Yes** | `{"messages": [{"role": "user", "content": "..."}], "videos": ["path/or/url"], "modalities": ["text", "audio"]}` | — |
| Video + audio | Text + Audio | **Yes** | `{"messages": [{"role": "user", "content": ""}], "videos": ["path/or/url"], "audios": ["path/or/url"], "modalities": ["text", "audio"]}` | content must be "" when the query is spoken |
| Video | Text + Audio | **Yes** | `{"messages": [{"role": "user", "content": ""}], "videos": ["path/or/url"], "modalities": ["text", "audio"]}` | content must be "" when query comes from video |

### Sampling Parameters

Standard sampling parameters apply to the thinker stage. When `modalities` includes `"audio"`, the additional talker-specific parameters below control the speech generation independently.

| Parameter | Type | Default | Applies to |
|---|---|---|---|
| `temperature` | float | `1.0` | Thinker |
| `top_p` | float | `1.0` | Thinker |
| `top_k` | int | `-1` | Thinker |
| `min_p` | float | `0.0` | Thinker |
| `repetition_penalty` | float | `1.0` | Thinker |
| `max_tokens` | int | `2048` | Thinker |
| `max_completion_tokens` | int | `null` | Thinker; OpenAI-compatible alias for `max_tokens` |
| `stop` | str \| list | `null` | Thinker |
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

### Known Limitations

- **`modalities: ["text", "audio"]` has no effect on a text-only server.** No error is raised — the response simply contains no audio. Use a speech-mode server (without `--text-only`) to get audio output.
- **`content` must be `""` when the query is entirely in `audios`, `videos`, or `images`.** Leaving a text query in `content` alongside audio causes the model to process both, which is usually not what you want.
- **Colocated topology does not support `--thinker.tp_size 2`.** The server raises a `ValueError` at startup ("Qwen Phase 1 colocation does not support thinker TP"). Use disaggregated topology for TP=2.
- **Requests that exceed the model's context length are rejected with an error.** The preprocessor raises a `ValueError` when the prompt token count alone meets or exceeds `max_seq_len`, or when `prompt tokens + max_new_tokens ≥ max_seq_len`. Reduce input length or lower `max_tokens` to stay within the limit.
- **Apple Silicon (MLX and Torch MPS) keeps a restricted scheduler profile.** `tp_size=1`, `max_running_requests=1`, greedy generation only, radix disabled, overlap disabled, mixed/chunked prefill disabled, CUDA graphs disabled, logprobs unsupported, and no partial talker start. Native MLX does not imply radix cache, multi-request batching, or CUDA-oriented optimizations, and backend selection never falls back between MLX and Torch MPS. See [Apple Silicon (MLX and Torch MPS)](#apple-silicon-mlx-and-torch-mps) above.
