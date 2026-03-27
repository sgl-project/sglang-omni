# SGLang Omni Benchmarks

Model-agnostic benchmark framework for SGLang Omni. V1 focuses on performance
benchmarking for the models and HTTP APIs that already exist in this repo.
The public entrypoint is model-first via explicit benchmark profiles.

## Performance Benchmarks

### Generic Runner

Run the new framework via:

```bash
python3 -m benchmarks.run \
  --base-url http://localhost:8000 \
  --model fishaudio/s2-pro \
  --model-profile s2_tts \
  --case voice-cloning \
  --dataset seedtts_testset/en/meta.lst
```

Built-in profiles:

- `s2_tts`

The legacy [`performance/tts/benchmark_tts_speed.py`](performance/tts/benchmark_tts_speed.py)
script is kept temporarily, but the supported entrypoint for the new framework
is [`benchmarks.run`](run.py).
