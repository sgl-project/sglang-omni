# SGLang Omni Benchmarks

Benchmark suite for SGLang Omni, covering performance (latency, throughput, RTF)
and accuracy (WER, MMSU, MMMU, Video-MME, Video-AMME) across supported modality
combinations.

## Directory Structure

```
benchmarks/
├── tasks/          # Per-task logic (tts, audio_understanding, visual_understand, video_understanding)
├── metrics/        # Metric computation (performance, accuracy)
├── dataset/        # Dataset loaders + download helpers
├── benchmarker/    # Framework: runner, data structures, utilities
├── eval/           # Entry-point scripts (one per task × model)
├── tts_serving/    # TTS serving harness and Docker contract
├── cache/          # (gitignored) dataset caches
└── results/        # (gitignored) evaluation outputs
```

## Quick Start

```bash
# 0. Prepare dataset (once)
python -m benchmarks.dataset.prepare --dataset seedtts

# 1. Start a server on port 8000 (pick one matching the benchmark below)

# S2-Pro — for sections 2a/2b/2c
python -m sglang_omni.cli serve \
    --model-path fishaudio/s2-pro \
    --config examples/configs/s2pro_tts.yaml --port 8000

# Voxtral-4B-TTS — for section 2d (plain TTS, no voice cloning)
python -m sglang_omni.cli serve \
    --model-path mistralai/Voxtral-4B-TTS-2603 --port 8000

# Higgs TTS — for section 2e (voice cloning via references[])
python -m sglang_omni.cli serve \
    --model-path boson-sglang/higgs-audio-v3-tts-4b-base \
    --port 8000

# MOSS-TTS — for section 2f (voice cloning via references[], duration via token_count)
python -m sglang_omni.cli serve \
    --model-path OpenMOSS-Team/MOSS-TTS-v1.5 \
    --config examples/configs/moss_tts.yaml --port 8000

# Qwen3-Omni, speech mode — for section 3 (SeedTTS; multi-GPU)
python -m sglang_omni.cli serve \
    --model-path Qwen/Qwen3-Omni-30B-A3B-Instruct --port 8000

# Qwen3-Omni, text-only mode — for sections 4 (MMSU) and 5 (MMMU)
python -m sglang_omni.cli serve \
    --model-path Qwen/Qwen3-Omni-30B-A3B-Instruct --text-only --port 8000

# 2a. S2-Pro — full pipeline: generate + WER (server needed for phase 1 only)
python -m benchmarks.eval.benchmark_tts_seedtts \
    --meta zhaochenyang20/seed-tts-eval-arrow \
    --model fishaudio/s2-pro --port 8000 \
    --output-dir results/s2pro_en --lang en --max-samples 50 --concurrency 8

# 2b. S2-Pro — generate only (speed metrics, no transcription)
python -m benchmarks.eval.benchmark_tts_seedtts \
    --generate-only --stream \
    --meta zhaochenyang20/seed-tts-eval-arrow \
    --model fishaudio/s2-pro --port 8000 --max-samples 50 --concurrency 8

# 2c. S2-Pro — transcribe only (reuses audio from a prior generate run; no server)
python -m benchmarks.eval.benchmark_tts_seedtts \
    --transcribe-only \
    --meta zhaochenyang20/seed-tts-eval-arrow \
    --model fishaudio/s2-pro \
    --output-dir results/s2pro_en --lang en --device cuda:0

# 2d. Voxtral — full pipeline without voice cloning
python -m benchmarks.eval.benchmark_tts_seedtts \
    --meta zhaochenyang20/seed-tts-eval-arrow \
    --model mistralai/Voxtral-4B-TTS-2603 --port 8000 \
    --max-concurrency 16 \
    --output-dir results/voxtral_en --lang en --max-samples 50 \
    --no-ref-audio --voice cheerful_female

# 2e. Higgs TTS — full pipeline with SeedTTS voice-cloning references
python -m benchmarks.eval.benchmark_tts_seedtts \
    --meta zhaochenyang20/seed-tts-eval-arrow \
    --model boson-sglang/higgs-audio-v3-tts-4b-base --port 8000 \
    --ref-format references \
    --max-concurrency 16 \
    --output-dir results/higgs_tts_en --lang en --max-samples 50

# 2f. MOSS-TTS — full pipeline with SeedTTS voice-cloning references
python -m benchmarks.eval.benchmark_tts_seedtts \
    --meta zhaochenyang20/seed-tts-eval-arrow \
    --model OpenMOSS-Team/MOSS-TTS-v1.5 --port 8000 \
    --ref-format references --token-count auto \
    --max-concurrency 8 \
    --output-dir results/moss_tts_en --lang en --max-samples 50

# 3a. Qwen3-Omni — full pipeline (generate + transcribe)
python -m benchmarks.eval.benchmark_omni_seedtts \
    --meta zhaochenyang20/seed-tts-eval-arrow \
    --output-dir results/qwen3_omni_en \
    --max-concurrency 16 \
    --model qwen3-omni --port 8000 --max-samples 50

# 3b. Qwen3-Omni — generate only (server required; use in CI to split phases)
python -m benchmarks.eval.benchmark_omni_seedtts \
    --generate-only \
    --meta zhaochenyang20/seed-tts-eval-arrow \
    --output-dir results/qwen3_omni_en \
    --max-concurrency 16 \
    --model qwen3-omni --port 8000 --max-samples 50

# 3c. Qwen3-Omni — transcribe only (reuses audio; ASR server on --port)
python -m benchmarks.eval.benchmark_omni_seedtts \
    --transcribe-only \
    --meta zhaochenyang20/seed-tts-eval-arrow \
    --output-dir results/qwen3_omni_en \
    --model qwen3-omni --lang en --port 8000

# 4. Qwen3-Omni — MMSU (audio comprehension)
python -m benchmarks.eval.benchmark_omni_mmsu \
    --model qwen3-omni --port 8000 \
    --modalities text+audio --max-samples 50

# 5. Qwen3-Omni — MMMU (VLM accuracy, image input)
python -m benchmarks.eval.benchmark_omni_mmmu \
    --model qwen3-omni --port 8000 --max-samples 50 --max-concurrency 16

# 6. Qwen3-Omni — Video-MME (video understanding)
python -m benchmarks.eval.benchmark_omni_videomme \
    --model qwen3-omni --port 8000 --max-samples 50

# 7a. Qwen3-Omni — Video-AMME (video + audio question understanding)
python -m benchmarks.eval.benchmark_omni_videoamme \
    --model qwen3-omni --port 8000 \
    --repo-id zhaochenyang20/Video_AMME_ci \
    --max-samples 50 --max-concurrency 16 \
    --video-fps 2 --video-max-frames 128 --video-max-pixels 401408

# 7b. Qwen3-Omni — Video-AMME Talker (text + audio output)
python -m benchmarks.eval.benchmark_omni_videoamme \
    --model qwen3-omni --port 8000 \
    --repo-id zhaochenyang20/Video_AMME_ci \
    --max-samples 50 --max-concurrency 16 \
    --video-fps 2 --video-max-frames 128 --video-max-pixels 401408 \
    --enable-audio --asr-device cuda:0 --asr-concurrency 32

# 8a. Offline UTMOS (naturalness MOS prediction) scoring on existing output
# For custom TTS models (e.g. S2-Pro, Voxtral, Higgs TTS):
python -m benchmarks.eval.benchmark_tts_seedtts \
    --utmos-only --output-dir results/s2pro_en --device cuda:0

# For Qwen3-Omni:
python -m benchmarks.eval.benchmark_omni_seedtts \
    --utmos-only --output-dir results/qwen3_omni_en --device cuda:0

# 8b. Offline Speaker Similarity (voice resemblance) scoring on existing output
# For custom TTS models (e.g. S2-Pro, Voxtral, Higgs TTS):
python -m benchmarks.eval.benchmark_tts_seedtts \
    --similarity-only --output-dir results/s2pro_en --device cuda:0

# For Qwen3-Omni:
python -m benchmarks.eval.benchmark_omni_seedtts \
    --similarity-only --output-dir results/qwen3_omni_en --device cuda:0
```

## Eval Scripts

| Script | Task | Model | API |
|--------|------|-------|-----|
| `eval/benchmark_tts_seedtts.py` | TTS speed + WER (unified) | e.g. S2-Pro, Voxtral, Higgs TTS | `/v1/audio/speech` |
| `eval/benchmark_tts_serving.py` | TTS serving contract | OpenAI-compatible TTS models | `/v1/audio/speech`, raw PCM streaming, WebSocket, voice and batch contracts |
| `eval/benchmark_omni_seedtts.py` | TTS speed + WER (unified) | Qwen3-Omni | `/v1/chat/completions` |
| `eval/benchmark_omni_mmsu.py` | MMSU (audio comprehension) | Qwen3-Omni | `/v1/chat/completions` |
| `eval/benchmark_omni_mmmu.py` | MMMU (VLM accuracy + speed) | Qwen3-Omni | `/v1/chat/completions` |
| `eval/benchmark_omni_videomme.py` | Video-MME (video understanding) | Qwen3-Omni | `/v1/chat/completions` |
| `eval/benchmark_omni_videoamme.py` | Video-AMME (video + audio question understanding) | Qwen3-Omni | `/v1/chat/completions` |
| `eval/benchmark_asr_seedtts.py` | ASR concurrency scaling on SeedTTS EN | Qwen3-ASR | `/v1/audio/transcriptions` |

See [tts_serving/README.md](tts_serving/README.md) for the TTS serving
benchmark design, harness contract, scenario matrix, and Docker usage.

The two `*_seedtts.py` scripts merge the previous `benchmark_*_tts_speed.py`
and `voice_clone_*_wer.py` pairs into a single two-phase pipeline: phase 1
generates + persists WAVs while the TTS server runs, phase 2 transcribes through
an ASR server to avoid GPU contention with the TTS server. Use `--generate-only` or
`--transcribe-only` to run a single phase. For TTS, `--concurrency` and
`--max-concurrency` are equivalent (see `benchmark_tts_seedtts.py`).
`benchmark_tts_seedtts.py` also handles model-specific voice-cloning reference
payloads: the default `--ref-format flat` sends `ref_audio`/`ref_text`, while
`--ref-format references` sends `references=[{audio_path, text}]` for Higgs TTS
and MOSS-TTS. MOSS-TTS additionally supports duration control through
`--token-count`.

`benchmark_omni_seedtts.py` documents local vs CI GPU usage in its module
docstring (sequential phases on CI to reduce OOM risk).

`benchmark_asr_seedtts.py` is a standalone ASR fan-out sweep (issue #646): it
transcribes the SeedTTS *reference* clips directly against a running Qwen3-ASR
router and reports WER + speed + per-worker routing balance per concurrency
level. Use it to measure how ASR concurrency affects throughput, latency, and
WER for a given workload.

Both `*_seedtts.py` scripts also support speech quality and similarity evaluation via UTMOS and WavLM speaker verification metrics. Running with `--utmos-only` or `--similarity-only` loads the respective pre-trained predictor and computes scores on the previously generated audio in the output directory without requiring the TTS/ASR servers to be running.

## TTS Quality Evaluation

To evaluate the overall quality and vocal resemblance of synthesized speech, the benchmark suite supports offline evaluation using UTMOS (naturalness MOS prediction) and Speaker Similarity (vocal fidelity).

### UTMOS (Naturalness)

UTMOS (UTokyo-Saru Lab MOS Prediction) is a Mean Opinion Score (MOS) predictor model used to evaluate the naturalness and overall quality of synthesized speech. SGLang Omni provides an offline UTMOS evaluator backed by the `balacoon/utmos` JIT model on Hugging Face.

- **Model Weights Cache**: By default, weights (`utmos.jit`) are downloaded from Hugging Face on the first run and cached at `~/.cache/sglang-omni/utmos` (override via `UTMOS_CACHE_DIR` environment variable).
- **Warm the Cache (Optional)**:
  ```bash
  python -m benchmarks.metrics.utmos --warm-cache
  ```
- **Outputs (`utmos_results.json`)**:
  - `summary`: section for summary metrics
    - `utmos_mean`: Mean predicted MOS score in `[1, 5]` (higher is better).
    - `utmos_median`: Median predicted MOS score.
    - `utmos_p5` / `utmos_p95`: 5th and 95th percentile scores to identify worst-case outliers or top-performing samples.
    - `total_samples`: total number of samples in the dataset.
    - `evaluated`: number of samples that were evaluated.
    - `skipped`: number of samples that were skipped.
  - `config`: evaluation configuration parameters.
  - `per_sample`: list of individual score results for each evaluated sample.

### Speaker Similarity (Vocal Fidelity)

Speaker Similarity evaluates how closely the voice of synthesized speech matches the reference prompt audio.

- **Model Weights Cache**: By default, model weights (`wavlm_large.pt` and `wavlm_large_finetune.pth`) are downloaded from Hugging Face and cached at `~/.cache/sglang-omni/speaker_sim` (override via `SEEDTTS_SIM_CACHE_DIR` environment variable).
- **Warm the Cache (Optional)**:
  ```bash
  python -m benchmarks.metrics.speaker_similarity_assets --warm-cache
  ```
- **Outputs (`similarity_results.json`)**:
  - `summary`: section for summary metrics
    - `speaker_similarity_mean`: Mean cosine similarity score scaled by 100.0 (higher is better).
    - `total_samples`: total number of samples in the dataset.
    - `evaluated`: number of samples that were evaluated.
    - `skipped`: number of samples that were skipped.
  - `config`: evaluation configuration parameters.
  - `per_sample`: list of individual score results for each evaluated sample.

## Adding a New Model or Task

- **New model, same task/API type** (e.g. another OAI-compatible TTS model):
  add an eval script under `eval/` that reuses the existing task helpers
  in `tasks/tts.py` (`make_tts_send_fn`, `run_seedtts_transcribe`, …).
- **New task or API type**: add a task class in the relevant `tasks/*.py`
  file (mirroring `VoiceCloneOmni` in `tasks/tts.py`), expose metric
  helpers, and wire it into a new eval script.

## Datasets

Download helpers live in `benchmarks/dataset/prepare.py`:

```bash
python -m benchmarks.dataset.prepare --dataset seedtts       # full SeedTTS
python -m benchmarks.dataset.prepare --dataset seedtts-mini  # smoke-test subset
python -m benchmarks.dataset.prepare --dataset seedtts-50    # 50-sample subset
python -m benchmarks.dataset.prepare --dataset mmmu          # full MMMU (30 subjects)
python -m benchmarks.dataset.prepare --dataset mmmu-ci-50    # MMMU CI subset
python -m benchmarks.dataset.prepare --dataset mmsu          # full MMSU (ddwang2000/MMSU)
python -m benchmarks.dataset.prepare --dataset videomme-ci-50  # Video-MME CI subset
python -m benchmarks.dataset.prepare --dataset videomme      # full Video-MME
python -m benchmarks.dataset.prepare --dataset videoamme-ci-50  # Video-AMME CI subset
```

All datasets are pre-warmed into the default HuggingFace cache via
`datasets.load_dataset(repo_id)`.  SeedTTS Arrow repos stage audio to
process-local tempfiles at load time; no manual `--local-dir` step is needed.

Video-AMME is generated from the Video-MME CI subset by moving the
question/options/instruction into per-sample WAV files. The benchmark request
text only contains routing/format instructions; the actual question content
stays in the dataset WAV files.
