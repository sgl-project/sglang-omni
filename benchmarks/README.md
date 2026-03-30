# SGLang Omni Benchmarks

Comprehensive benchmark suite for SGLang Omni, covering both performance (latency, throughput, RTF etc.) and accuracy (quality metrics) across all supported modality combinations. Omni models operate on a `{video, audio, text} x {video, audio, text}` input-output matrix. The table below tracks benchmark coverage.

## Performance Benchmarks

### TTS Voice Cloning

[`performance/tts/benchmark_tts_speed.py`](performance/tts/benchmark_tts_speed.py): Benchmarks online serving latency and throughput for TTS models via the `/v1/audio/speech` HTTP API.

## Accuracy Benchmarks

### TTS Word Error Rate (WER)

[`accuracy/tts/benchmark_tts_wer.py`](accuracy/tts/benchmark_tts_wer.py): Measures intelligibility of synthesized speech by computing corpus-level Word Error Rate against the [seed-tts-eval](https://github.com/BytedanceSpeech/seed-tts-eval) test set.

**Pipeline:** For each sample the benchmark (1) generates speech with the S2-Pro TTS model, (2) transcribes the audio back to text with an ASR model, and (3) computes WER between the original target text and the ASR transcript.

**ASR models by language:**

| Language | ASR Model | Text Normalizer |
|----------|-----------|-----------------|
| English (`en`) | Whisper-large-v3 | Whisper `EnglishTextNormalizer` |
| Chinese (`zh`) | FunASR paraformer-zh | Character-level split + zhconv |

**Primary metric — corpus WER (micro-average):** Errors and reference word counts are accumulated across all samples, then divided once, consistent with the [HuggingFace `evaluate` WER definition](https://github.com/huggingface/evaluate/blob/main/metrics/wer/wer.py):

```
WER_corpus = Σ(S + D + I) / Σ(S + D + C)
```

Per-sample WER statistics (mean, median, std, p95) are also reported as secondary diagnostics.

#### Usage

```bash
# English
python -m benchmarks.accuracy.tts.benchmark_tts_wer \
    --meta seedtts_testset/en/meta.lst \
    --model-path fishaudio/s2-pro \
    --output-dir results/s2pro_en \
    --lang en

# Chinese
python -m benchmarks.accuracy.tts.benchmark_tts_wer \
    --meta seedtts_testset/zh/hardcase.lst \
    --model-path fishaudio/s2-pro \
    --output-dir results/s2pro_zh \
    --lang zh
```

#### Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--meta` | `seedtts_testset/en/meta.lst` | Path to seed-tts-eval meta file (`sample_id\|ref_text\|ref_audio\|target_text`) |
| `--output-dir` | *(required)* | Directory for generated audio, JSON, and CSV results |
| `--model-path` | `fishaudio/s2-pro` | Path or HuggingFace ID of the S2-Pro model |
| `--lang` | `en` | Language (`en` or `zh`); determines ASR model and normalizer |
| `--device` | `cuda:0` | CUDA device for ASR and vocoder |
| `--max-samples` | `None` | Cap the number of samples evaluated (useful for quick smoke tests) |
| `--max-new-tokens` | `2048` | Maximum semantic tokens per generation |
| `--temperature` | `0.8` | Sampling temperature for TTS generation |

#### Outputs

All outputs are written to `--output-dir`:

- `audio/` — Generated `.wav` files, one per sample.
- `wer_results.json` — Full results including summary metrics, run config, and per-sample details (WER, substitutions, deletions, insertions, hits, latency, audio duration).
- `wer_results.csv` — Same per-sample data in tabular format.
