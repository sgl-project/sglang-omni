# Apple Silicon Support Matrix

Which models have been exercised on Apple Silicon (macOS arm64), and on which
backend. Tracked in RFC #1967.

Statuses are evidence-aware: a cell is validated only when the model was run on
that backend on Apple hardware and the result recorded below. `Related work`
links implementation and validation discussions; a link alone does not validate a cell.

- `✅` validated end-to-end on Apple Silicon, evidence below
- `❌` not validated on this backend

| Model | MLX | Torch/MPS | Related work |
| --- | --- | --- | --- |
| Qwen3-ASR | ✅ | ✅ | landed in #1730 |
| Fun-ASR-Nano | ❌ | ❌ | #1982, #1983; shared runners landed in #1981 |
| Fun-CosyVoice3 | ❌ | ❌ | #1964 |
| Qwen3 TTS | ❌ | ❌ | #1960 |
| Whisper ASR | ❌ | ❌ | #1977 |
| MOSS-Transcribe-Diarize | ❌ | ❌ | #1989 |
| Qwen3-Omni | ❌ | ❌ | tracking PR TBD |
| ARK-ASR-3B | ❌ | ❌ | — |
| Audar-TTS V1 Turbo | ❌ | ❌ | — |
| AuK | ❌ | ❌ | — |
| dots.tts | ❌ | ❌ | — |
| Fish Audio S2-Pro | ❌ | ❌ | — |
| Higgs TTS | ❌ | ❌ | — |
| LLaDA2.0-Uni | ❌ | ❌ | — |
| Ming-Omni | ❌ | ❌ | — |
| Ming-Omni-TTS | ❌ | ❌ | — |
| MiniMax Music 3 | ❌ | ❌ | #1990 |
| MOSS-TTS | ❌ | ❌ | — |
| MOSS-TTS-Local | ❌ | ❌ | — |
| Nemotron VoiceChat | ❌ | ❌ | — |
| Voxtral TTS | ❌ | ❌ | — |
| ZONOS2 | ❌ | ❌ | — |

Cookbook pages under `docs/cookbook/` remain the canonical launch and request
examples; this page only records which of them have been exercised on Apple.

## Evidence: Qwen3-ASR

Re-tested on the `main` snapshot `6ff46426469a1af2746cef71a9fdcfc09613966d`,
after the shared Apple runner changes in
[#1981](https://github.com/sgl-project/sglang-omni/pull/1981).
The runs used an isolated installation on an **Apple M2, 8 GiB unified memory,
macOS 26.5.2 arm64**: Python 3.12.13, torch 2.13.0, TorchCodec 0.15.0,
MLX 0.32.2, mlx-lm 0.31.3, Transformers 5.12.1 and FFmpeg 7.1.2.
SGLang was built from tag `v0.5.19`, commit
`0bcd822377da7b5718e674eaf9c870d349424dd1`.


### Findings from the recorded runs

- **The official checkpoint runs on both Apple backends.** MLX and Torch/MPS
  served the same pinned official checkpoint, and returned identical transcripts
  for the timed input in both runs. A converted artifact is not required for this
  tested Qwen3-ASR path.
- **MLX recorded lower warm-request medians on this host.** The two runs gave
  0.366 / 0.371 s on MLX and 0.618 / 0.640 s on Torch/MPS, using the same official
  checkpoint and input. These are observed service timings under the conditions
  below, not a general backend speed guarantee.
- **The optional 4-bit configuration reduced MLX model storage.** The loader
  probe found 0.66 GiB of parameter storage versus 1.46 GiB for the official
  checkpoint, and the configuration served the timed input successfully.

### Primary comparison: official checkpoint on both backends

Both columns use `Qwen/Qwen3-ASR-0.6B`, revision
`5eb144179a02acc5e5ba31e748d22b0cf3e303b0`, from a local snapshot.
The [cookbook's Apple Silicon examples](../cookbook/qwen3_asr.md#apple-silicon-mlx)
pair a converted 4-bit checkpoint on MLX with the official checkpoint on Torch/MPS
for convenient deployment. That pairing changes both backend and quantisation;
this primary table holds the checkpoint constant.

Values separated by `/` are the first and second completed run of each
configuration, each with a freshly started service.

| | MLX + official | Torch/MPS + official |
| --- | --- | --- |
| Process start → healthy service, local snapshot | 19.901 / 15.220 s | 29.226 / 30.445 s |
| Warm HTTP request median, 20 requests per run | 0.366 / 0.371 s | 0.618 / 0.640 s |
| Warm HTTP request min–max, all 40 requests | 0.346–0.506 s | 0.590–1.348 s |
| Successful timed requests | 40/40 | 40/40 |
| Allocated KV pool, tokens | 4206 / 9057 | 2048 / 2048 |

### Supplementary configuration: 4-bit checkpoint on MLX

The converted checkpoint is `mlx-community/Qwen3-ASR-0.6B-4bit`, revision
`313d850181767edf09f00a9c289becca70e58cd0` (4-bit affine, group size 64).
This comparison stays within MLX; it is not evidence of a backend-only effect.

| | MLX + official | MLX + converted 4-bit |
| --- | --- | --- |
| Process start → healthy service, local snapshot | 19.901 / 15.220 s | 15.768 / 13.961 s |
| Warm HTTP request median, 20 requests per run | 0.366 / 0.371 s | 0.227 / 0.228 s |
| Warm HTTP request min–max, all 40 requests | 0.346–0.506 s | 0.221–0.284 s |
| Successful timed requests | 40/40 | 40/40 |
| Loaded parameter storage, separate loader probe | 1.46 GiB | 0.66 GiB |
| Allocated KV pool, tokens | 4206 / 9057 | 12426 / 14645 |

### Protocol and output checks

- One local service at a time, with `max_running_requests=1` and scheduler
  asynchronous decode disabled. Completed-run order was MLX official, MPS
  official, MLX 4-bit, then the reverse order. No model downloads were included.
- Each service received three warm-up requests, then 20 measured requests using
  identical bytes from `tests/data/query_to_cars.wav` (4.620 s, 48 kHz mono PCM16).
  The client loaded the file once and timed HTTP open through complete response
  read with `time.perf_counter()`, excluding disk reads and multipart construction.
- Requests used `temperature=0`, `stream=false`, `response_format=json`, and no
  language hint. The pipeline's default duration-based output budget was 128
  tokens for these short inputs. Radix caching was disabled on both Apple paths;
  audio features and audio-encoder outputs were recomputed for each request.
  Normal within-request KV caching and warmed runtime resources remained in use.
- Startup was timed from process creation to the first `/health` response with
  HTTP 200, `running=true` and `status=healthy`, polled every 0.1 s. This measures
  service readiness, before request warm-up; it is not a cold-disk startup test.

All three configurations returned `How many cars are there in the picture?` for
the timed input, and each repeated its own output across both runs. This
establishes execution and output agreement on that input; it does not measure
word error rate on a labelled dataset.

### Scope of the measurements

The official snapshot's stored tensors are BF16. A separate probe using the same
MLX loader and model classes found 611 loaded BF16 parameter tensors; the MPS
service logged its installed Torch language model as `torch.bfloat16`.
This is the same official BF16 checkpoint with backend-native execution precision,
not a claim that every operation uses identical arithmetic. The 4-bit loader
probe found packed `uint32` weights alongside BF16 tensors.

Startup and complete-request times have common external boundaries. The internal
weight-loading logs do not: Torch/MPS's `Load weight end` precedes a further
Hugging Face language-model replacement. Consequently, those diagnostic load
values are retained in the downloadable record but are not compared as one
weight-loading metric in the primary table.

This was a working 8 GiB Mac with background applications and swap in use, not
an otherwise idle machine. MLX auto-sized its KV pool using the available memory
at startup; its pool varied even between restarts of the same checkpoint.
Torch/MPS used its 2048-token profile budget. Parameter storage is not whole-process
memory, and allocated KV pools are not measured backend capacity limits.
The runs support the recorded configurations and inputs; they cannot isolate a
source-code performance change from the dependency upgrades or host state.

### Historical observations at `ef7ae01`

The original record is retained separately: macOS 26.5.2 arm64, 8 GiB memory,
Python 3.12.13, torch 2.11.0 and SGLang v0.5.18 from source. All three
configurations transcribed `query_to_cars.wav` identically. These older timings
had only two warm samples per configuration, without a fully recorded warm-up
or timing protocol; they are not pooled with the re-test above.

| Configuration | Checkpoint files | Startup, warm cache | Logged weight load | Reported model memory | KV pool | Two warm requests |
| --- | --- | --- | --- | --- | --- | --- |
| MLX, official | 1.88 GB | 13.1 s | 1.28 s | 1.46 GB | 6164 tokens | 0.33 / 0.41 s |
| Torch/MPS, official | 1.88 GB | 18.4 s | 4.51 s | not reported the same way | 2048 tokens | 0.58 / 0.61 s |
| MLX, converted 4-bit | 0.71 GB | 12.7 s | 0.90 s | 0.66 GB | 7527 tokens | 0.24 / 0.36 s |

The historical memory values retain their original `GB` labels. MLX used a
5.3 GB wired limit and auto-sized its pool; Torch/MPS used a 2048-token budget.

For related validation on other models and hosts, see the
[Apple Silicon roadmap (#1967)](https://github.com/sgl-project/sglang-omni/issues/1967)
and the Fun-ASR-Nano reports for
[Torch/MPS (#1982)](https://github.com/sgl-project/sglang-omni/pull/1982) and
[MLX (#1983)](https://github.com/sgl-project/sglang-omni/pull/1983).

## Import surface

Recorded at `6ff46426` on the same host with the Torch/MPS selection
(`SGLANG_USE_MLX` unset). Model packages were discovered as direct child
directories of `sglang_omni/models` containing `__init__.py`. For each package the
sweep attempted six module names — `config`, `engine_builder`, `stages`,
`model_runner`, `sglang_model` and `request_builders` — importing every present
module in its own sequential child process.

| | Count |
| --- | --- |
| Model packages discovered | 22 |
| Target slots, 22 packages x 6 names | 132 |
| Modules present and attempted | 107 |
| Imported successfully | 107 |
| Names absent from their package | 25 |

Not every package implements all six names. An absent name is recorded as
`not_present` and never counts as a success. No package reported an import
failure in this sweep.

The successes include `higgs_tts.model_runner`, which failed at `ef7ae01` with
`ModuleNotFoundError: No module named 'sgl_kernel'`; that dependency is CUDA-only
and the Apple installer skips it. Upstream
[#1721](https://github.com/sgl-project/sglang-omni/pull/1721) has since added
platform-based kernel selection with pure-PyTorch top-k/top-p fallbacks for
non-CUDA platforms.

An import check establishes that a module loads. The support matrix requires
recorded end-to-end execution on Apple hardware, so a `❌` above means "not
validated", not "known broken".

## Limitations

- Evidence was collected on a machine with **8 GiB of unified memory**. Some models
  cannot be evaluated on it at all — MiniMax Music 3 alone loads 16 GB of weights.
  A `❌` on a large model may reflect that, not a defect.
- The matrix is model-level and does not enumerate conditional fallbacks inside a
  model path.
- Model owners are welcome to update their row with a commit, environment,
  inputs and recorded end-to-end results.

## Reproducing

Use a separate checkout of `6ff46426469a1af2746cef71a9fdcfc09613966d` for the
recorded re-test. For current setup guidance, see the
[Qwen3-ASR cookbook](../cookbook/qwen3_asr.md). The installer does not lock every
transitive dependency; compare installed versions with the recorded environment
before attributing a changed result to source code.

```bash
SGLANG_VERSION=v0.5.19 ./install.sh --non-interactive
source .venv-apple/bin/activate
export DYLD_LIBRARY_PATH="$(brew --prefix ffmpeg@7)/lib${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}"
OFFICIAL=$(hf download Qwen/Qwen3-ASR-0.6B --revision 5eb144179a02acc5e5ba31e748d22b0cf3e303b0)
QUANTIZED=$(hf download mlx-community/Qwen3-ASR-0.6B-4bit --revision 313d850181767edf09f00a9c289becca70e58cd0)
```

Run one server at a time, stopping it before starting the next. Separate ports
below also avoid the launcher's port fallback immediately after shutdown.

Torch/MPS, official checkpoint:

```bash
unset SGLANG_USE_MLX
sgl-omni serve --model-path "$OFFICIAL" --model-name Qwen/Qwen3-ASR-0.6B \
  --asr.factory.dtype bfloat16 --asr.engine.max_running_requests 1 \
  --asr.factory.enable_async_decode false --host 127.0.0.1 --port 8000
```

MLX, official checkpoint:

```bash
SGLANG_USE_MLX=1 sgl-omni serve --model-path "$OFFICIAL" \
  --model-name Qwen/Qwen3-ASR-0.6B --asr.factory.dtype bfloat16 \
  --asr.engine.max_running_requests 1 --asr.factory.enable_async_decode false \
  --host 127.0.0.1 --port 8001
```

MLX, optional 4-bit checkpoint:

```bash
SGLANG_USE_MLX=1 sgl-omni serve --model-path "$QUANTIZED" \
  --model-name Qwen/Qwen3-ASR-0.6B --asr.factory.dtype bfloat16 \
  --asr.engine.max_running_requests 1 --asr.factory.enable_async_decode false \
  --host 127.0.0.1 --port 8002
```

On MLX, the loader preserves checkpoint dtypes; the `dtype` flag does not convert
quantized weights back to BF16.

From a second terminal at the repository root, activate the same environment and
use the [request client](../_static/apple_support_matrix/benchmark_requests.py)
from this documentation checkout, or download it as `benchmark_requests.py`.
After `/health` reports ready, run it with the **actual port printed by the
service** and a unique output filename for each configuration and restart:

```bash
python docs/_static/apple_support_matrix/benchmark_requests.py \
  --base-url http://127.0.0.1:8000 --model Qwen/Qwen3-ASR-0.6B \
  --audio tests/data/query_to_cars.wav --warmups 3 --requests 20 \
  --output mps-official-run1.json
```

The client records individual responses and timings; HTTP success alone does
not validate the transcript. It measures requests only. To reproduce startup
timing, use a separate monotonic timer from server process creation through the
first healthy response, following the polling protocol above. Keep complete
server logs for the runtime profile, actual backend and KV allocation.
