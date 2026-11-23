# MOSS-TTS sampling acceptance harness

Scripts used to compare a MOSS-TTS server arm with and without the compacted
Delay text draw. Every script takes its paths on the command line or from the
environment; nothing assumes a particular home directory.

| File | Purpose |
| --- | --- |
| `prepare_refs.py` | Extract the SeedTTS English split into reference WAVs plus a fixed-order manifest. |
| `ab_arm.sh` | Start one server arm, wait for `/v1/models`, run `run_eval.py`, stop that arm by PID/process group. |
| `run_eval.py` | Drive `/v1/audio/speech` in fixed order; report success-only and attempted throughput, latency, HBM, failures. Exits nonzero if any request failed. |
| `score_wer.py` | Whisper-large-v3 FP32 greedy transcription and corpus WER with exact S/D/I/H counts. |
| `microbench_sampling.py` | Replay-only timing of the compacted vs full sampling graph (synthetic batch). |
| `whisper_smoke.py` | Two-clip scorer smoke test. |

## Prerequisites

Python packages: `aiohttp numpy soundfile pyarrow jiwer>=4 torch transformers
scipy whisper-normalizer`. On Ascend the interpreter and `npu-smi` may need a
launcher wrapper (for example, the pod here uses `/tmp/e31-stack/native`).

Model/data assets are not bundled:

- MOSS-TTS-v1.5 weights served by the launcher.
- Whisper-large-v3 weights for `score_wer.py`.
- SeedTTS English Arrow parquet for `prepare_refs.py`.
- Two server launchers (`LAUNCHER_BASE`, `LAUNCHER_OPT`) that serve the same
  model/config on `BASE_URL` and are selected only by the arm.

## Prepare the fixed-order manifest (once)

```bash
python prepare_refs.py \
  --parquet /path/to/en-00000-of-00001.parquet \
  --out-dir /path/to/refs \
  --manifest /path/to/en_manifest.json
```

The manifest is a JSON list in evaluation order; `ref_wav` paths must be
readable by the server process.

## Run one arm

```bash
BENCH_DIR=$PWD \
LAUNCHER_BASE=/path/to/launch_base.sh \
LAUNCHER_OPT=/path/to/launch_opt.sh \
PYTHON=python3 \
BASE_URL=http://127.0.0.1:8000 \
MANIFEST=/path/to/en_manifest.json \
RESULTS_DIR=/path/to/results \
LOG_DIR=/path/to/logs \
NPU_SMI=/usr/local/bin/npu-smi \
NPU_SMI_WRAPPER=/tmp/e31-stack/native \
bash ab_arm.sh opt c8 8 0 c8_opt
```

`ab_arm.sh <arm> <group> <concurrency> <limit> <run_id>` accepts `0` as the
limit for the whole manifest. It refuses to start if `BASE_URL` already serves,
starts the launcher in its own session, and only signals the process group it
created. Optional variables: `BENCH_WRAPPER` (interpreter wrapper),
`START_TIMEOUT` (default 2400 s), `DEVICE_INDEX` (default 0), `STOP_TIMEOUT`
(default 60 s).

`RESULTS_DIR/<run_id>/summary.json` records `success`, `num_ok`, `num_failed`,
`successful_throughput_rps`, `attempted_throughput_rps`, latency percentiles
for successful requests, `hbm_peak_percent`, `hbm_peak_mb`, and every failure
with its index, HTTP status, reason and body prefix. A run with any invalid or
missing response is unsuccessful (`success=false`, exit code 1).

HBM MB is derived from the capacity reported by `npu-smi info -t usages -i N`;
if the device does not report capacity, only the percentage is recorded.

## Score WER

```bash
python score_wer.py \
  --manifest /path/to/en_manifest.json \
  --wav-dir /path/to/results/<run_id> \
  --model-path /path/to/whisper-large-v3 \
  --device npu:0 \
  --out /path/to/results/wer_<run_id>.json
```

The report exposes corpus `substitutions/deletions/insertions/hits`,
`error_words = S + D + I`, `ref_words = H + S + D`, per-sample counts, and a
`failures` list for missing/unreadable audio and decode errors. `success` is
false when any sample could not be scored.

## Replay microbench

```bash
PYTHONPATH=/path/to/sglang-omni python microbench_sampling.py \
  --batch 8 --warmup 20 --iters 100 --out /path/to/microbench_b8.json
```

This times `graph.replay` only for a synthetic 1-text + 7-countdown batch; it
is not an end-to-end measurement.

## A/B rules

Use the same hardware, dependencies, model/config, manifest order, seed,
warmup count and limit for both arms, alternate `base`/`opt` arms, and restart
the server between arms. Do not compare against numbers produced by different
scripts or trees.
