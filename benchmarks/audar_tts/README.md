# Audar Arabic intelligibility smoke benchmark

This workflow provides a reproducible Arabic smoke benchmark for
Audar-TTS-V1 Turbo. It loads a fixed 50-sentence Arabic target-text dataset
from Hugging Face, synthesizes each target with the same official Audar
reference, transcribes the generated WAVs with Qwen3-ASR-1.7B, and directly
compares the normalized Arabic target and hypothesis.

This is not a standard Arabic TTS benchmark or a substitute for evaluation from
the Audar authors. It checks intelligibility and is useful for regressions. It
does not measure naturalness, prosody, or speaker similarity.

## Generate

Install the Audar dependencies, then run the generator from the repository
root on a CUDA machine:

```bash
pip install -e '.[audar-tts]'

python -m benchmarks.audar_tts.run_quality_benchmark \
  --output-dir results/audar-arabic \
  --samples 50
```

The runner reads
[`zhaochenyang20/sglang-omni-arabic-tts-smoke`](https://huggingface.co/datasets/zhaochenyang20/sglang-omni-arabic-tts-smoke)
at immutable revision
`65835c3a1047037f9e0cd4947652722c0a58c304`. The dataset contains only target
text and FLEURS provenance, so the same set can be reused by other Arabic TTS
models.

By default the script downloads the pinned official
`samples/demo_male_1_ar.wav` reference and uses its matching transcript. Pass
`--reference-path` to use a local copy of the same file. Audar requires a
5-15 second reference.

The generator writes:

- `generation_results.json`: dataset, model, revision, hash, and latency
  metadata;
- `generated.json`: the existing shared ASR pipeline input;
- `audio/*.wav`: generated speech.

## Transcribe

Run the repository SeedTTS ASR phase against the generated WAVs:

```bash
python -m benchmarks.eval.benchmark_tts_seedtts \
  --transcribe-only \
  --model audarai/Audar-TTS-V1-Turbo \
  --output-dir results/audar-arabic \
  --meta google/fleurs \
  --lang ar \
  --max-new-tokens 1024 \
  --asr-model-path Qwen/Qwen3-ASR-1.7B \
  --asr-concurrency 16 \
  --skip-gpu-cleanup
```

In `--transcribe-only` mode, `benchmark_tts_seedtts` reads the existing
`generated.json`. The `--meta` value is provenance only; it does not load
SeedTTS rows or issue TTS requests.

## Summarize

```bash
python -m benchmarks.audar_tts.summarize_quality \
  --generation results/audar-arabic/generation_results.json \
  --wer results/audar-arabic/wer_results.json \
  --output results/audar-arabic/quality_summary.json
```

The summary contains corpus WER, CER, BLEU, and chrF++. No translation is used.
BLEU, CER, and chrF++ are alternate views of the same ASR output, not
independent quality measurements.

## Reference result

The original 50-sentence run produced 5.43% WER, 1.46% CER, 88.75 BLEU, and
95.57 chrF++. Treat these values as a smoke-test reference, not a vendor
comparison or quality threshold.

## Source of truth

- Dataset:
  [`zhaochenyang20/sglang-omni-arabic-tts-smoke`](https://huggingface.co/datasets/zhaochenyang20/sglang-omni-arabic-tts-smoke)
- Dataset revision: `65835c3a1047037f9e0cd4947652722c0a58c304`
- Dataset manifest:
  [`manifest.json`](https://huggingface.co/datasets/zhaochenyang20/sglang-omni-arabic-tts-smoke/blob/65835c3a1047037f9e0cd4947652722c0a58c304/manifest.json)

The dataset is the canonical benchmark artifact, so this repository does not
carry its one-time materialization script. The pinned manifest records the
FLEURS source revision, selection rules, artifact path, and Parquet SHA-256.
