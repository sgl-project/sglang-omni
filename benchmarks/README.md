# 📈 SGLang-Omni Benchmarks

## 📍 Table of Contents

- [📍 Table of Contents](#-table-of-contents)
- [📚 Overview](#-overview)
- [💻 Benchmark Workflow](#-benchmark-workflow)
  - [1. Data Preparation](#1-data-preparation)
  - [2. Run Benchmarks](#2-run-benchmarks)
    - [TTS Benchmark](#tts-benchmark)
    - [Omni Benchmark](#omni-benchmark)
    - [Relay Benchmark](#relay-benchmark)
- [📑 Developer Reference](#-developer-reference)

## 📚 Overview

This folder contains the benchmark suite for SGLang Omni, including the benchmark scripts and the dataset preparation scripts. You can follow the steps below to conduct benchmarks for both system performance and accuracy evaluation.


## 💻 Benchmark Workflow

### 1. Data Preparation

```bash
# use the commands below to download the formatted SeedTTS dataset
# 5_samples means there are only 5 samples in the dataset, used for simple testing only
python prepare_dataset.py --dataset seedtts_tts
python prepare_dataset.py --dataset seedtts_tts_5_samples
python prepare_dataset.py --dataset seedtts_vc
python prepare_dataset.py --dataset seedtts_vc_5_samples
```

### 2. Run Benchmarks


#### Benchmark TTS Model

You can invoke the following command to benchmark the TTS voice cloning performance of Fish Audio S2 Pro:

```bash
bash ./models/benchmark_fish_audio_s2.sh
```

#### Benchmark Omni Models

To be added.

#### Relay Benchmark

You can run the script below to benchmark the performance of the Relay module with different backend types.

```bash
python benchmark_relay.py
```


## 📑 Developer FAQ

**Q1: How is the `benchmark_kit` module organized?**

The core of the `benchmark_kit` module is the `Benchmarker` class, which is a base class for all benchmarkers. It provides the basic functionality for loading datasets, building requests, sending queries to the server, measuring latency and throughput, and compute the metrics.

**Q2: How to add a new benchmark metrics?**

You can implement a metric class by inheriting from the `Metrics` class in the `benchmark_kit.benchmarker.metrics` module. Then you can pass the metric objects to the `Benchmarker` class when initializing it. The metrics will be computed annd saved in the result json file.

**Q3: How to add a new benchmark dataset?**

You can add a new dataset processing function to the `prepare_dataset.py` script. Make sure that the dataset has a `data.jsonl` file in its directory and the each line is a json object which follows the schema specified by the `Data` class in the `benchmark_kit.benchmarker.data` module.

**Q4: How to add a benchmark for a new model?**

If you want to benchmark for a new TTS or Omni model, you can just follow the examples provided in the `models` folder and add a new script in the `models` folder by changing the `model_path` and other arguments for this model.
