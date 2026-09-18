# Nemotron 3.5 ASR

[Nemotron 3.5 ASR Streaming 0.6B](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b)
is a multilingual speech-recognition model with a FastConformer encoder and
an RNN-T decoder. SGLang-Omni supports complete-file transcription through
`/v1/audio/transcriptions` and native cache-aware PCM streaming in the internal
pipeline runtime.

## Prerequisites

Follow [Installation](../get_started/installation.md), then run the examples
from the repository root. Use the repository's pinned Transformers version;
the Nemotron compatibility implementation is included.

## Server Configuration

The default pipeline runs one ASR stage on one GPU in `float32`:

```bash
sgl-omni serve \
  --model-path nvidia/nemotron-3.5-asr-streaming-0.6b \
  --port 8000
```

Tune the ASR stage with `--asr.factory.*` flags:

| Option | Default | Description |
|---|---|---|
| `dtype` | `float32` | Model dtype |
| `num_lookahead_tokens` | `3` | Encoder right context; the checkpoint supports `0`, `3`, `6`, and `13` |
| `max_batch_size` | `8` | Maximum number of requests in a scheduler batch |
| `max_batch_wait_ms` | `2.0` | Maximum wait to form a batch, in milliseconds |
| `max_pending_stream_messages` | `256` | Pending input-message limit for the internal streaming runtime |

For example:

```bash
sgl-omni serve \
  --model-path nvidia/nemotron-3.5-asr-streaming-0.6b \
  --asr.factory.num_lookahead_tokens 3 \
  --asr.factory.max_batch_size 8 \
  --asr.factory.max_batch_wait_ms 2 \
  --port 8000
```

Within each scheduler batch, complete-file requests with the same
`max_new_tokens` value share one model `generate()` call. Different token limits
are processed in separate batches to preserve each request's output limit.

## Transcribe Audio

Upload a complete audio file; the server converts it to mono 16 kHz audio:

```bash
curl http://localhost:8000/v1/audio/transcriptions \
  -F model=nvidia/nemotron-3.5-asr-streaming-0.6b \
  -F file=@tests/data/query_to_cars.wav \
  -F language=auto \
  -F response_format=verbose_json
```

Use `response_format=text` for the transcript alone, or `json` for a JSON
response. `verbose_json` also includes duration and language information.
Locale tags are removed from the transcript. With `language=auto`, the
response reports a language when the model emits one unambiguous locale tag.

## Request Parameters

| Parameter | Default | Description |
|---|---|---|
| `file` | required | Audio file uploaded as multipart form data |
| `model` | server default | Model identifier |
| `language` | `auto` | Checkpoint-defined locale or language code, matched case-insensitively; `auto` enables language detection |
| `response_format` | `json` | `json`, `verbose_json`, or `text`; streaming responses accept only `json` or `text` |
| `temperature` | `0` | Greedy RNN-T decoding only; non-zero values are rejected |
| `max_new_tokens` | model default | Optional positive output-token limit |
| `prompt` | unset | Text prompts are unsupported; non-empty values are rejected |
| `stream` | `false` | Stream the HTTP response after a complete file upload; see below |

Supported language values come from the checkpoint's prompt dictionary.
Unsupported values fail before model inference. Nemotron supports transcription
only; `/v1/audio/translations` returns HTTP 400. Segment timestamps, SRT/VTT
output, and speaker diarization are not supported.

## Native Streaming

The internal streaming runtime accepts mono PCM16 at 16 kHz. It forms
lookahead-dependent audio windows and retains each request's attention,
convolution-padding, and RNN-T decoder caches across chunks. Compatible chunks
from different requests can share a model batch. End-of-input flushes the final
partial window; completion, cancellation, or failure releases the request state.

This PCM input path is an internal runtime capability. A public Nemotron
WebSocket/session API, VAD controls, and stable input-event schema are outside
the current integration. Setting `stream=true` on `/v1/audio/transcriptions`
streams the response to a complete uploaded file; it does not open a persistent
PCM input session or select the native cache-aware input path.

## Apple Silicon (MLX)

Run `./install.sh` and activate `.venv-apple`. Set `SGLANG_USE_MLX=1` to
select the native MLX FastConformer, language projector, LSTM predictor and
RNN-T joint network. The shared processor prepares log-mel features on CPU;
`mlx-audio` and NeMo are not runtime dependencies.

The MLX profile uses FP32, greedy decoding, one active request, and at most
60 seconds per uploaded file. The scheduler clamps `max_batch_size` to one
while preserving queued requests and their independent decoder state.
`json`, `text`, `verbose_json`, and SSE completion responses are supported.
SSE emits the final transcript after offline inference; this path does not
provide incremental audio ingestion or cache-aware live transcription.

Official Hugging Face safetensors load directly, without a separate MLX
conversion or quantization step:

```bash
source .venv-apple/bin/activate
export DYLD_LIBRARY_PATH="/opt/homebrew/opt/ffmpeg@7/lib${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}"
SGLANG_USE_MLX=1 sgl-omni serve \
  --model-path nvidia/nemotron-3.5-asr-streaming-0.6b \
  --asr.factory.dtype float32 \
  --port 8000
```

### Download weights from ModelScope

The ModelScope mirror supplies the original `.nemo` archive. Convert its
learned parameters to the official safetensors layout once. The converter
uses Torch's weights-only loader, verifies every parameter name and shape,
and writes a source SHA-256 manifest. It obtains only tokenizer and processor
metadata from the pinned official Hugging Face revision, not model weights.
Use `--metadata-path` to supply that metadata locally.

```bash
mkdir -p "$HOME/models/nemotron-nemo"
curl -fL --retry 5 -C - \
  'https://modelscope.cn/models/nv-community/nemotron-3.5-asr-streaming-0.6b/resolve/master/nemotron-3.5-asr-streaming-0.6b.nemo' \
  -o "$HOME/models/nemotron-nemo/nemotron-3.5-asr-streaming-0.6b.nemo"

python -m sglang_omni.models.nemotron3_5_asr.convert_nemo \
  --nemo-path "$HOME/models/nemotron-nemo/nemotron-3.5-asr-streaming-0.6b.nemo" \
  --output "$HOME/models/nemotron-3.5-asr-0.6b"

SGLANG_USE_MLX=1 sgl-omni serve \
  --model-path "$HOME/models/nemotron-3.5-asr-0.6b" --port 8000
```

The same converted checkpoint can be loaded by the Torch runner for numerical
comparison. Setting `SGLANG_USE_MLX=0` preserves the existing Torch path;
Torch/MPS is not qualified by the MLX validation below.

### Reproduce validation

```bash
SGLANG_USE_MLX=1 python -m pytest tests/unit_test/nemotron3_5_asr -q
python -m benchmarks.eval.verify_nemotron_mlx \
  --model-path "$HOME/models/nemotron-3.5-asr-0.6b" \
  --output /tmp/nemotron-validation.json
```

The e2e verifier checks complete token sequences against the Torch CPU
reference for both bundled speech clips, explicit English and automatic
language detection, and lookahead `0`, `3`, `6`, and `13`. It then launches
an actual MLX server, verifies HTTP response formats and SSE completion,
queued/repeated requests, invalid inputs and recovery, and stops the server.
The JSON report records package versions, revision, transcripts, timings,
and MLX peak memory. These short-clip checks do not establish corpus-level
WER or sustained-load performance.

#### Local validation snapshot (2026-09-07)

Apple M1 Pro, 32 GiB unified memory; FP32; MLX 0.32.2, Torch 2.11.0,
Transformers 5.12.1, SGLang source `71de97b264b04dcd514cf904003028aefe9775c8`
(installer's v0.5.18 source). The ModelScope archive SHA-256 was
`210214ed94039bf6bfbb9a047c7fa289628db75b103e2bf6381fa78285436a74`;
all 655 learned parameter tensors passed conversion name/shape checks.

| Check | Result |
|---|---|
| Complete Torch CPU / MLX token sequences | 16/16 exact matches |
| HTTP/SSE requests | 25/25 expected responses: 19 HTTP 200, 6 HTTP 400 |
| Locale cleanup and `verbose_json.language` | Explicit English and automatic detection passed |
| Concurrent/repeated requests and recovery after invalid input | Passed |
| Warm serial HTTP observations, 4.620 s / 5.064 s speech clips | Approximately 0.120–0.164 s total request time |
| MLX peak allocation during component validation | 2.62 GB (excludes Torch CPU and HTTP processes) |

Timings are smoke-test observations across response formats, not a throughput
benchmark. The reproducible JSON report includes source-file and audio hashes
so results can be tied to the implementation even when run from a dirty tree.


## Apple Silicon (PyTorch MPS)

The shared Torch runner also supports Apple GPU inference through MPS. It reuses
CPU audio preprocessing and the existing Torch encoder, RNN-T decoder, batching,
and native cache-aware streaming scheduler. No separate model implementation or
weight conversion is required for an existing safetensors checkpoint.

After `./install.sh`, explicitly disable MLX to select Torch MPS on Apple Silicon:

```bash
source .venv-apple/bin/activate
SGLANG_USE_MLX=0 sgl-omni serve --model-path /path/to/nemotron-checkpoint
```

The platform selects `mps:0` automatically. The default precision is FP32.
`SGLANG_USE_MLX=1` selects the separate native MLX backend instead. MPS uses the
Torch scheduler's batching settings rather than the MLX single-request limit.
The native PCM streaming path remains an internal runtime capability; HTTP
`stream=true` returns SSE for a complete uploaded file.

Validate MPS against the CPU reference and exercise the real HTTP server:

```bash
SGLANG_USE_MLX=0 PYTORCH_ENABLE_MPS_FALLBACK=0 \
  python -m benchmarks.eval.verify_nemotron_mlx \
  --backend mps --model-path /path/to/nemotron-checkpoint \
  --output /tmp/nemotron-mps-validation.json
```

The check compares full offline token sequences and batched streaming tokens
and frame advances across two fixtures, both language modes, and all four
lookahead settings. It also checks HTTP formats, SSE, concurrent requests, and
recovery after invalid input. MPS operator fallback is disabled during validation;
audio preprocessing intentionally remains on CPU. These fixture checks are not
a dataset-level WER evaluation or a sustained throughput benchmark.


### MPS validation snapshot (2026-09-18)

Apple M1 Pro, 32 GiB unified memory; FP32; Torch 2.13.0, Transformers 5.12.1,
SGLang 0.5.19. MPS operator fallback was disabled.

| Check | Result |
|---|---|
| Complete Torch CPU / MPS offline token sequences | 16/16 exact matches |
| Batched cache-aware streaming | 16 streams, 370 chunks; exact tokens and frame advances |
| HTTP/SSE requests | 24/24 expected responses, including invalid-input recovery |
| Unit and shared serving/scheduling regressions | 281 passed |
| Existing MLX regression | 16/16 exact token matches and 25/25 HTTP/SSE checks |

The MPS check omits the MLX-specific 60-second rejection case. Broad WER,
sustained throughput, and latency-distribution evaluation remain pending.
