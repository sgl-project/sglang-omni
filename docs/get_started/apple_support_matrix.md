# Apple Silicon Support Matrix

Which models have been exercised on Apple Silicon (macOS arm64), and on which
backend. Tracked in RFC #1967.

Statuses are evidence-aware: a cell is validated only when the model was run on
that backend on Apple hardware and the result recorded below. `In flight` links
open work and makes no claim about it.

- `✅` validated end-to-end on Apple Silicon, evidence below
- `❌` not validated on this backend

| Model | MLX | Torch/MPS | In flight |
| --- | --- | --- | --- |
| Qwen3-ASR | ✅ | ✅ | landed in #1730 |
| Fun-ASR-Nano | ❌ | ❌ | #1981 → #1982 → #1983 |
| Fun-CosyVoice3 | ❌ | ❌ | #1964 (draft) |
| Qwen3 TTS | ❌ | ❌ | #1960 (draft, no Apple Metal E2E yet) |
| Whisper ASR | ❌ | ❌ | #1977 (draft, MLX encoder only) |
| MOSS-Transcribe-Diarize | ❌ | ❌ | #1989 (draft) |
| Qwen3-Omni | ❌ | ❌ | tracking PR TBD |
| ARK-ASR-3B | ❌ | ❌ | — |
| Audar-TTS V1 Turbo | ❌ | ❌ | — |
| dots.tts | ❌ | ❌ | — |
| Fish Audio S2-Pro | ❌ | ❌ | — |
| Higgs TTS | ❌ | ❌ | — |
| LLaDA2.0-Uni | ❌ | ❌ | — |
| Ming-Omni | ❌ | ❌ | — |
| Ming-Omni-TTS | ❌ | ❌ | — |
| MiniMax Music 3 | ❌ | ❌ | #1990 (draft) |
| MOSS-TTS | ❌ | ❌ | — |
| MOSS-TTS-Local | ❌ | ❌ | — |
| Voxtral TTS | ❌ | ❌ | — |
| ZONOS2 | ❌ | ❌ | — |

Cookbook pages under `docs/cookbook/` remain the canonical launch and request
examples; this page only records which of them have been exercised on Apple.

## Evidence: Qwen3-ASR

Apple Silicon, macOS 26.5.2 arm64, 8 GiB unified memory. `./install.sh`,
Python 3.12.13, torch 2.11.0, sglang v0.5.18 from source, sglang-omni at `ef7ae01`.
Audio `tests/data/query_to_cars.wav`; all three configurations returned the same
transcript.

| | MLX + converted | MLX + official | Torch/MPS |
| --- | --- | --- | --- |
| Checkpoint | `mlx-community/Qwen3-ASR-0.6B-4bit` (0.71 GB) | `Qwen/Qwen3-ASR-0.6B` (1.88 GB) | `Qwen/Qwen3-ASR-0.6B` (1.88 GB) |
| Server startup, warm cache | 12.7 s | 13.1 s | 18.4 s |
| Weight load | 0.90 s | 1.28 s | 4.51 s |
| Resident model memory | 0.66 GB | 1.46 GB | not reported the same way |
| KV pool | 7527 tokens | 6164 tokens | 2048 tokens |
| `/v1/audio/transcriptions`, warm | 0.24 s / 0.36 s | 0.33 s / 0.41 s | 0.58 s / 0.61 s |

Three things follow from this, each answering an open roadmap item:

- **The converted MLX artifact is an optimisation, not a requirement.** The MLX path
  loads the official PyTorch checkpoint and serves from it correctly. What the 4-bit
  conversion buys is resident memory, 0.66 GB against 1.46 GB, and MLX turns that
  headroom into a larger KV pool: 7527 tokens against 6164.
- **The KV pool differs by 3x between backends on the same checkpoint.** MLX
  auto-sizes against a 5.3 GB wired limit and reached 6164 tokens; Torch/MPS took the
  documented 2048-token budget. This is the concrete reason
  `docs/cookbook/qwen3_asr.md` recommends MLX for long audio.
- **Backend cost, checkpoint held constant.** Comparing the two official-checkpoint
  columns removes quantisation as a variable: MLX loads weights 3.5x faster and serves
  warm requests at roughly 0.6x the latency.

Warm latency on MLX was about 1.8x lower than Torch/MPS on the same checkpoint. Same
direction and roughly the same size as the 1.9x reported for Fun-ASR-Nano on an M5 Pro
in #1967, measured here independently on a different model and machine.

## Import surface

Importing each model package's `config`, `engine_builder`, `stages`,
`model_runner`, `sglang_model` and `request_builders` on Apple Silicon:
**19 of 20 import cleanly.**

The exception is `higgs_tts`, whose `model_runner` fails with
`ModuleNotFoundError: No module named 'sgl_kernel'`.
`sglang_omni/models/higgs_tts/sampler.py` imports `sgl_kernel` at module scope; it
is CUDA-only and the Apple installer skips it by design. These are the only
module-scope `sgl_kernel` imports in the tree — the four other call sites are
already function-local.

So a `❌` above means "not validated", not "known broken". Most model packages load
on Apple Silicon; what has not been established is whether they serve correctly.

## Limitations

- Evidence was collected on a machine with **8 GiB of unified memory**. Some models
  cannot be evaluated on it at all — MiniMax Music 3 alone loads 16 GB of weights.
  A `❌` on a large model may reflect that, not a defect.
- The matrix is model-level and does not enumerate conditional fallbacks inside a
  model path.
- `In flight` entries are open at the time of writing and should be re-checked when
  the linked PR merges. Model owners are welcome to update their own row.

## Reproducing

`./install.sh`, then `source .venv-apple/bin/activate` and export
`DYLD_LIBRARY_PATH="$(brew --prefix ffmpeg@7)/lib"`.

```bash
# Torch/MPS: official checkpoint, SGLANG_USE_MLX unset
sgl-omni serve --model-path Qwen/Qwen3-ASR-0.6B --model-name Qwen/Qwen3-ASR-0.6B \
  --asr.engine.max_running_requests 1 --port 8000

# MLX, SGLANG_USE_MLX=1 — either checkpoint works
sgl-omni serve --model-path mlx-community/Qwen3-ASR-0.6B-4bit \
  --model-name Qwen/Qwen3-ASR-0.6B --asr.engine.max_running_requests 1 --port 8000
sgl-omni serve --model-path Qwen/Qwen3-ASR-0.6B \
  --model-name Qwen/Qwen3-ASR-0.6B --asr.engine.max_running_requests 1 --port 8000

curl http://localhost:8000/v1/audio/transcriptions \
  -F file=@tests/data/query_to_cars.wav -F model=Qwen/Qwen3-ASR-0.6B
```
