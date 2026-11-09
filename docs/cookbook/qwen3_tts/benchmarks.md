---
orphan: true
---

# Qwen3-TTS Benchmarks

[Back to the Qwen3-TTS quick start](../qwen3_tts.md).

## Benchmarking

Prepare the SeedTTS dataset, then run the English split against 0.6B Base. The benchmark launches the configured TTS server on port 8000 and starts the ASR server after generation.

```bash
python -m benchmarks.dataset.prepare --dataset seedtts

python -m benchmarks.eval.benchmark_tts_seedtts \
  --meta zhaochenyang20/seed-tts-eval-arrow \
  --model Qwen/Qwen3-TTS-12Hz-0.6B-Base \
  --server-config examples/configs/qwen3_tts_0_6b.yaml \
  --port 8000 \
  --output-dir results/qwen3_tts_0_6b_en \
  --lang en \
  --max-concurrency 16
```

For the Chinese split, set `--lang zh` and use a separate output directory. For 1.7B Base, set `--model Qwen/Qwen3-TTS-12Hz-1.7B-Base` and `--server-config examples/configs/qwen3_tts_1_7b.yaml`. See the [TTS benchmark reference](../../developer_reference/tts_bench.md) for methodology and additional workloads.

## Benchmark Results

### 0.6B Base

Qwen3-TTS-12Hz-0.6B-Base on Seed-TTS EN (1088 utterances, reference voice cloning from each prompt), concurrency 16, WER scored with HF Whisper-large-v3. Hardware: 1× H200 SXM.

| Metric | Value |
|---|---|
| WER (corpus, excl. runaway outliers) | 1.07% |
| WER (per-sample median / p95) | 0.00% / 9.09% |
| WER (corpus micro-avg, raw) | 18.29% |
| Runaway samples (>50% WER) | 2 / 1088 (0.2%) |
| Latency mean / median (s) | 6.61 / 6.24 |
| RTF mean / median | 1.51 / 1.48 |
| Output throughput (tok/s) | 115.4 |
| Completed / failed requests | 1088 / 0 |

Typical output is clean (0.00% median WER, 9.09% p95). Two utterances (0.2%) ran away into a repetition loop and generated ~164 s of looping audio up to `max_new_tokens`, which alone lifts the raw micro-average to 18.29%; excluding those, corpus WER is 1.07%. RTF > 1 reflects the 0.6B codec pipeline at concurrency 16, not single-stream latency. The 1.7B checkpoint trades latency for quality.

### 1.7B CustomVoice

Qwen3-TTS-12Hz-1.7B-CustomVoice on the full Seed-TTS-Eval EN and ZH splits, concurrency 16, with 16 warmup requests per language/mode and `max_new_tokens=2048`. EN used Ryan/English and ZH used Vivian/Chinese, without reference audio or instructions. WER/CER was scored with Qwen3-ASR-1.7B at concurrency 32. Hardware: 1× H200 141 GB, BF16, TP1. Sampling overrides and seed were unset.

The server used `--tts_engine.engine.max_running_requests 64`, `--tts_engine.engine.cuda_graph_max_bs 64`, `--tts_engine.engine.torch_compile_max_bs 64`, `--vocoder.process vocoder`, `--tts_engine.gpu_memory_fraction 0.85`, and `--vocoder.gpu_memory_fraction 0.10`; `torch.compile` remained disabled. Streaming used the default `1 -> 2 -> 4` chunk ramp without a request-level override. Each language/mode was measured once, in non-streaming EN/ZH then streaming EN/ZH order on the same warmed server. The target GPU had no external GPU process during timed windows; host CPU, memory, and I/O were shared with another profiling task.

| Metric | Non-streaming EN | Non-streaming ZH | Streaming EN | Streaming ZH |
|---|---:|---:|---:|---:|
| Samples | 1088 | 2020 | 1088 | 2020 |
| Corpus WER/CER | 1.608% | 0.984% | 2.085% | 0.927% |
| Corpus WER/CER (excl. >50% outliers) | 1.359% | 0.984% | 1.454% | 0.927% |
| Samples above 50% WER/CER | 3 | 0 | 4 | 0 |
| UTMOS | 4.1723 | 3.1824 | 4.1500 | 3.1789 |
| QPS | 14.788 | 13.307 | 10.098 | 8.333 |
| Latency mean (s) | 1.075 | 1.198 | 1.573 | 1.915 |
| RTF mean | 0.2335 | 0.2079 | 0.3380 | 0.3332 |
| TTFA mean (s) | N/A | N/A | 0.1213 | 0.1047 |

Corpus WER (EN) / CER (ZH) is total edit distance divided by total reference words / characters and includes every sample. The filtered row excludes samples whose own WER/CER exceeds 50% and recomputes the corpus rate. UTMOS is the mean predicted audio-quality score, not a listening-test score. These independently sampled runs are not a paired comparison of streaming and non-streaming quality; the Base result also uses different conditioning and a different ASR evaluator.

TTFA measures arrival of the first PCM payload, not the first audible speech; its mean payload duration was 80 ms in both languages. All 3,108 streaming requests were continuity-scored: 98.99% EN and 93.76% ZH had no playback underrun longer than 50 ms. Maximum underrun was 396.1 ms EN and 2072.0 ms ZH. The default ramp therefore does not guarantee uninterrupted playback at concurrency 16; see [First-audio chunk ramp](tuning.md#first-audio-chunk-ramp) for the buffering tradeoff.

### Runaway Regression

`benchmarks.eval.benchmark_tts_generation_budget` sends each case twice with the same text, voice, and seed: once with the legacy 2048-frame limit and once with the server default. It saves the audio, finish reasons, codec frame counts, and chunk timings. The text request fails only when it stops on `length` before its paired baseline stopped naturally.

Create a request template, for example `request.json`:

```json
{
  "model": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
  "input": "replaced by each case",
  "voice": "Ryan"
}
```

For Base, use a fixed `references` entry instead of a built-in voice. The probe sends four built-in short lines in English and Chinese; `--cases` takes a JSON list of your own, and `--seedtts-samples` probes the SeedTTS prompts that missed codec EOS on 0.6B Base in #1179.

```bash
python -m benchmarks.eval.benchmark_tts_generation_budget \
  --base-url http://127.0.0.1:8000/v1 --request request.json \
  --server-revision <server-git-sha> --model-revision <checkpoint-sha> \
  --repeats 8 --seed 0 --output-dir results/budget-nonstream

python -m benchmarks.eval.benchmark_tts_generation_budget \
  --base-url http://127.0.0.1:8000/v1 --request base-request.json --seedtts-samples \
  --server-revision <server-git-sha> --model-revision <checkpoint-sha> \
  --repeats 8 --seed 123 --output-dir results/budget-runaway
```

Add `--stream` to check raw PCM streams. Output directories must be new. A fixed seed does not make waveforms bitwise identical because prefix-cache state and batch composition can change between requests.

The TTS CI non-streaming and streaming stages run this probe for Qwen3-TTS on four short cases with eight seeds, 64 requests per stage, and upload `results.jsonl` with the audio of the pairs that failed.
