# Qwen3-Omni TTS WER Evaluation via SGLang-Omni Server

## Overview

This document reports Word Error Rate (WER) benchmarks for **Qwen3-Omni-30B-A3B-Instruct** served via the **sglang-omni server** (speech pipeline), evaluated on the [seed-tts-eval](https://huggingface.co/datasets/zhaochenyang20/seed-tts-eval) English test set.

### Goals

1. **Verify Jingwen's fix** (`JingwenGu0829:fix/minimal-speech-pipeline-patches`) — confirm the Qwen3 Omni speech pipeline is functional on sglang-omni server.
2. **Benchmark WER** on the full EN set (1088 samples) and compare to the official reported 1.39%.
3. **Note**: Qwen3 Omni does **not** support voice cloning. It uses fixed speaker IDs (Ethan, Chelsie, Aiden). The official 1.39% WER is also reported without voice cloning. This is unlike S2-Pro which supports reference-audio-based voice cloning.

## Server Configuration

- **Model**: `Qwen/Qwen3-Omni-30B-A3B-Instruct`
- **Pipeline**: 9-stage speech pipeline (thinker + talker + code_predictor + code2wav + encoders + preprocessing + decode)
- **GPU allocation**: 2x H100 per server instance (thinker on GPU 0, talker/code_predictor/code2wav on GPU 1)
- **API endpoint**: `/v1/chat/completions` with `modalities: ["text", "audio"]`
- **Speaker**: Ethan (default)
- **Branch**: `qwen3-omni-wer-server-eval` (based on `qwen3-omni-wer-validation` + `JingwenGu0829:fix/minimal-speech-pipeline-patches`)

## Fix Verification

The fix (`fix/minimal-speech-pipeline-patches`) addresses several critical issues:

| Fix | Description |
|-----|-------------|
| `EntryClass` default | Changed from text-only to speech pipeline (`Qwen3OmniSpeechPipelineConfig`) |
| `audio_target_sr` | Added missing variable in preprocessor's plain-string input path |
| Tensor clone | Fixed CUDA illegal memory access by cloning aux hidden states before async SHM copy |
| Radix cache | Disabled for talker (projected embeddings break prefix matching) |
| Spawn context | Use `multiprocessing.get_context("spawn")` to avoid CUDA re-init in forked children |
| Multi-process launcher | Added `needs_mp` path in `launcher.py` for multi-GPU pipelines |

**Result**: Server starts successfully, health check passes, and TTS requests return valid audio.

## WER Results (English, seed-tts-eval)

### Full EN Set (1088 samples)

| Metric | Value |
|--------|-------|
| **WER (corpus, micro-avg)** | **2.19%** |
| WER per-sample mean | 2.30% |
| WER per-sample median | 0.00% |
| WER per-sample std | 7.95% |
| WER per-sample p95 | 13.95% |
| WER (excl >50% samples) | 1.93% |
| >50% WER samples | 4 (0.4%) |
| Latency mean | 5.69s |
| Audio duration mean | 3.61s |
| Total samples | 1088 |
| Evaluated | 1088 (100%) |

### Subset Results

| Samples | WER (corpus, micro-avg) | WER per-sample mean | WER (excl >50%) | >50% WER samples | Latency (mean) |
|---------|------------------------|---------------------|------------------|-------------------|----------------|
| 50      | **0.89%**              | 0.62%               | 0.89%            | 0 (0.0%)          | 5.28s          |
| 100     | **1.73%**              | 1.97%               | 0.92%            | 1 (1.0%)          | 5.03s          |
| 200     | **1.95%**              | 2.04%               | 1.95%            | 0 (0.0%)          | 5.13s          |
| **1088 (full)** | **2.19%**     | 2.30%               | 1.93%            | 4 (0.4%)          | 5.69s          |

- **Official reported WER**: 1.39% (full EN set, no voice cloning)
- **ASR model**: Whisper-large-v3
- **WER metric**: Micro-average (corpus-level), consistent with HuggingFace evaluate standard

### Comparison with Previous Results

| Configuration | 50 samples | Full EN set (1088) |
|---------------|-----------|---------------------|
| Qwen3 Omni (transformers pipeline, no voice clone) | 0.89% | 2.53% |
| **Qwen3 Omni (sglang-omni server, no voice clone)** | **0.89%** | **2.19%** |
| Qwen3 Omni (official, no voice clone) | N/A | 1.39% |
| S2-Pro (sglang-omni server, with voice clone) | 0.89% | 1.95% |

### Bad Cases (>50% WER, full EN set)

| Sample ID | WER | Reference | Hypothesis | Issue |
|-----------|-----|-----------|------------|-------|
| `common_voice_en_17324784-...` | 162.5% | "despite years of research the problem remained intractable" | "duoxi does a poor ziting with cha chen..." | Completely garbled output (Chinese-like phonemes) |
| `common_voice_en_19845853-...` | 100.0% | "the ninth member is the state superintendent of education..." | "phobos abukhanul all adelian illustrile..." | Completely garbled output |
| `common_voice_en_19717736-...` | 66.7% | "vernon signal engineer" | "then in signal engineer" | Short sentence, first word garbled |
| `common_voice_en_19284142-...` | 57.1% | "you aren't positive you're negative" | "you are on the positive you are negative" | Contractions expanded incorrectly |

### Notable High-WER Cases (20-50%)

| Sample ID | WER | Issue |
|-----------|-----|-------|
| `common_voice_en_19729972-...` | 37.5% | "carpet" → "carp at the" |
| `common_voice_en_37457199-...` | 33.3% | "why" → "while" |
| `common_voice_en_18735317-...` | 30.0% | "tao" → "taiyou", "song as weapon" → "son as a weapon" |
| `common_voice_en_153872-...` | 25.0% | "they" → "this" |
| `common_voice_en_17256676-...` | 25.0% | "to her amazement" → "kieran maismont" |

Most errors are on the first 1-2 words of the sentence, or on rare/proper nouns.

## Detailed Results

Full per-sample results (JSON):
- `results/qwen3_omni_server_en_full/wer_results_merged.json` — merged full set (1088 samples)
- `results/qwen3_omni_server_en_full/part_{0,1,2}/wer_results.json` — per-shard results
- `results/qwen3_omni_server_en_{50,100,200}/wer_results.json` — subset results

## How to Reproduce

### Single-server evaluation

```bash
# 1. Download dataset
huggingface-cli download zhaochenyang20/seed-tts-eval \
    --repo-type dataset --local-dir seedtts_testset

# 2. Start Qwen3 Omni speech server (requires 2 GPUs)
CUDA_VISIBLE_DEVICES=2,3 python examples/run_qwen3_omni_speech_server.py \
    --model-path Qwen/Qwen3-Omni-30B-A3B-Instruct \
    --gpu-thinker 0 --gpu-talker 1 \
    --gpu-code-predictor 1 --gpu-code2wav 0 \
    --port 8000

# 3. Run WER evaluation
CUDA_VISIBLE_DEVICES=4 python -m benchmarks.performance.tts.benchmark_tts_wer_qwen3_omni_server \
    --meta seedtts_testset/en/meta.lst \
    --output-dir results/qwen3_omni_server_en_full \
    --lang en \
    --asr-device cuda:0 \
    --port 8000
```

### Parallel evaluation (3 servers, ~3x faster)

```bash
# 1. Split the dataset
split -l 363 -d -a 1 seedtts_testset/en/meta.lst shards/shard_

# 2. Start 3 servers with isolated IPC paths
for i in 0 1 2; do
    port=$((8000 + i))
    gpus="$((i*2)),$((i*2+1))"
    CUDA_VISIBLE_DEVICES=$gpus python benchmarks/performance/tts/launch_qwen3_omni_server.py \
        --port $port --instance-id $i &
done

# 3. Run 3 clients in parallel
for i in 0 1 2; do
    port=$((8000 + i))
    CUDA_VISIBLE_DEVICES=7 python -m benchmarks.performance.tts.benchmark_tts_wer_qwen3_omni_server \
        --meta shards/shard_$i --output-dir results/part_$i \
        --lang en --asr-device cuda:0 --port $port &
done
wait

# 4. Merge results
python benchmarks/performance/tts/merge_wer_results.py \
    --parts results/part_0/wer_results.json \
            results/part_1/wer_results.json \
            results/part_2/wer_results.json \
    --output results/wer_results_merged.json
```

Or use the all-in-one script:
```bash
bash benchmarks/performance/tts/run_parallel_wer_eval.sh
```

## Conclusions

1. **Jingwen's fix works**: The Qwen3 Omni speech pipeline runs correctly on sglang-omni server after the minimal patches.
2. **Full EN set WER = 2.19%**: Compared to official 1.39%. The gap likely comes from differences in generation hyperparameters, prompt format, or non-determinism. Excluding 4 severe bad cases, WER is 1.93%.
3. **sglang-omni server is better than transformers pipeline**: Full set WER 2.19% vs 2.53% (transformers), a 13% relative improvement.
4. **sglang-omni server matches transformers on small sets**: The 50-sample WER (0.89%) is identical between sglang-omni server and transformers pipeline, confirming correctness.
5. **No voice cloning**: Qwen3 Omni only supports fixed speaker IDs, not reference-audio-based voice cloning. This is an architecture limitation, not a bug.

## Date

2026-03-27
