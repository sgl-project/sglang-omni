# SGLang Omni Benchmarks

Comprehensive benchmark suite for SGLang Omni, covering both performance (latency, throughput, RTF etc.) and accuracy (quality metrics) across all supported modality combinations. Omni models operate on a `{video, audio, text} x {video, audio, text}` input-output matrix. The table below tracks benchmark coverage.

## Data Preparation

```bash
# use the commands below to download the formatted SeedTTS dataset
# 5_samples means there are only 5 samples in the dataset, used for simple testing only
python prepare_dataset.py --dataset seedtts_tts
python prepare_dataset.py --dataset seedtts_tts_5_samples
python prepare_dataset.py --dataset seedtts_vc
python prepare_dataset.py --dataset seedtts_vc_5_samples
```

## Relay Benchmark

You can run the script below to benchmark the performance of the Relay module with different backend types.

```bash
python benchmark_relay.py
```

### TTS Voice Cloning

You can invoke the following command to benchmark the TTS voice cloning performance of Fish Audio S2 Pro:

```bash
bash ./models/benchmark_fish_audio_s2.sh
```
