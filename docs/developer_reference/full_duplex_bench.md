# Full-Duplex-Bench v1.5 runbook

This page runs [Full-Duplex-Bench](https://github.com/DanielLin94144/Full-Duplex-Bench)
v1.5 against a native full-duplex model served by SGLang-Omni (MiniCPM-o 4.5 by default).
Every step is a shell command. Run them in order and copy-paste them as written.
The scripts live in `benchmarks/duplex/fdb_v15/`.

For measurement definitions and the underlying CLI, see
[benchmarks/duplex/REFERENCE.md](../../benchmarks/duplex/REFERENCE.md).

## What is measured

v1.5 has 498 sample pairs in four categories: `user_interruption` (200),
`user_backchannel` (98), `talking_to_other` (100) and `background_speech` (100).
Each pair has two user recordings: `input.wav` contains an overlapping event
(an interruption, a backchannel, a side conversation or background speech), and
`clean_input.wav` is the same audio without it. Each recording is streamed to the model
in real time in its own session, so one pair is two sessions.

The pipeline has three steps:

| Step | Script | What it does | Runs on |
|---|---|---|---|
| 1. Generate | `generate.sh` | Streams every input to `/v1/realtime`, records the model's audio, then cuts a fixed observation window per session | Model server, GPU 0 |
| 2. ASR | `asr.sh` | Transcribes the input and output audio with word timestamps (Parakeet), then computes VAD speech intervals with the official timing code | Scoring venv, GPU 1 |
| 3. Judge | `judge.sh` | Sends the four transcripts of each pair to an LLM with the official prompt, then writes `summary.json` and `report.txt` | Judge server, GPU 1 |

`aggregate.sh` then combines repeats into one table.

The results are:

- **Stop latency**: how long the model keeps talking while the user talks over it
  (overlap of merged user and model speech spans). Lower means the model yields sooner.
- **Response latency**: time from a user speech span ending to the next model speech start.
- **Behavior labels**: the judge assigns each pair one label for how the model handled the overlap.
  `C_RESPOND` means it addressed the overlap's content. `C_RESUME` means it ignored the overlap
  and continued. `C_UNCERTAIN_HANDLING` means it asked for a repeat or showed it did not catch
  the overlap. `C_UNKNOWN` means its reply was off-target or it said nothing.

## Hardware and judge

The default layout needs two GPUs with at least 80 GB each:
GPU 0 serves the model under test, and GPU 1 runs both ASR and the judge.
The generation server keeps running while ASR runs, because they use different GPUs.

Two judges are supported, selected by `JUDGE`:

| `JUDGE` | Model | Notes |
|---|---|---|
| `qwen` (default) | [Qwen3.8-27B](https://docs.sglang.io/cookbook/autoregressive/Qwen/Qwen3.8-27B), served locally by SGLang | Reproducible and free. Non-thinking mode, greedy, seeds 1-3. Results are labeled non-official |
| `gpt` | `gpt-4o-2024-08-06` | The paper's judge. Needs `OPENAI_API_KEY`. The response must report exactly this model name |

## Step 0: one-time setup

Run these commands from the repository root, with the sglang-omni virtualenv active
(`which python` should print that venv's interpreter):

```bash
cd /path/to/sglang-omni
bash benchmarks/duplex/fdb_v15/setup.sh
```

`setup.sh` is safe to rerun and skips finished steps. Everything goes under
`FDB_WORK` (default `$HOME/fdb`):

| Path | Content |
|---|---|
| `Full-Duplex-Bench/` | Official scoring code at the pinned revision `3e799c4`; file hashes are checked |
| `dataset/v1.5/` | The four v1.5 subsets, downloaded from the official Google Drive; `dataset/v1.5.revision` holds the archive hash |
| `scoring-venv/` | Python 3.12 with NeMo 3.0.0, torch 2.11 (cu130) and Silero VAD 6.2.1 for ASR and timing |
| `models/parakeet-tdt-0.6b-v2/` | ASR checkpoint, SHA-256 checked |
| `models/MiniCPM-o-4_5/` | Model under test at revision `503e754`. To reuse an existing copy, symlink it here or set `MODEL_PATH` |
| `models/Qwen3.8-27B/` | Judge checkpoint; its commit is saved in `models/Qwen3.8-27B.revision` |

To change a default, export the variable before running a script, or write it into
`~/.config/fdb_v15.env`, which every script reads. The variables are listed in
[Settings](#settings).

## Step 1-3: run the benchmark

Open three terminals in the repository root. In each one, activate the sglang-omni venv
first, and export the same overrides if you use any.

**Terminal A: model server.** Leave it running. It is ready when it prints `Uvicorn running`
(about one minute).

```bash
bash benchmarks/duplex/fdb_v15/launch_server.sh
```

**Terminal B: judge server.** Skip this terminal when `JUDGE=gpt`. It is ready when it prints
`The server is fired up and ready to roll!` (about two minutes).

```bash
bash benchmarks/duplex/fdb_v15/launch_judge.sh
```

**Terminal C: the pipeline.** Run a preflight first. It takes one pair per category
(4 pairs, about 2 minutes end to end) and catches environment problems before a long run:

```bash
export RUN_NAME=preflight MAX_PER_SUBSET=1
bash benchmarks/duplex/fdb_v15/generate.sh 1
bash benchmarks/duplex/fdb_v15/asr.sh 1
bash benchmarks/duplex/fdb_v15/judge.sh 1
```

The preflight passes when `generate.sh` prints `{"pass": 8}` and
`{"eligible_pairs": 4, ...}`, `asr.sh` prints `{"ok": 16}` and `{"ok": 8}`, and `judge.sh`
prints `{"valid": 4}`. If anything differs, see [Troubleshooting](#troubleshooting).

Then run the measured benchmark: 48 pairs (12 per category), three repeats.

```bash
export RUN_NAME=minicpmo-48 MAX_PER_SUBSET=12
bash benchmarks/duplex/fdb_v15/generate.sh 1
bash benchmarks/duplex/fdb_v15/asr.sh 1
bash benchmarks/duplex/fdb_v15/judge.sh 1
bash benchmarks/duplex/fdb_v15/generate.sh 2
bash benchmarks/duplex/fdb_v15/asr.sh 2
bash benchmarks/duplex/fdb_v15/judge.sh 2
bash benchmarks/duplex/fdb_v15/generate.sh 3
bash benchmarks/duplex/fdb_v15/asr.sh 3
bash benchmarks/duplex/fdb_v15/judge.sh 3
bash benchmarks/duplex/fdb_v15/aggregate.sh
```

For all 498 pairs, use `export RUN_NAME=minicpmo-full MAX_PER_SUBSET=` (an empty value
selects everything) and run the same lines.

Approximate wall time per repeat on H200, with one session at a time:

| Pairs | `generate.sh` | `asr.sh` | `judge.sh` |
|---|---|---|---|
| 48 | ~25 min | ~1 min | < 1 min |
| 498 | ~4 h | ~10 min | ~5 min |

Sessions run in real time (about 15 s each), so generation dominates.
`NUM_SHARDS=2` roughly halves it; read the note on `NUM_SHARDS` before using it.

When everything is done, stop terminals A and B with Ctrl-C.

## Results

`aggregate.sh` prints the table and writes it to `$FDB_WORK/runs/$RUN_NAME/RESULTS.md`.
Each latency and label cell is the mean ± sample standard deviation over the finished repeats;
count cells show one value, or a range when repeats differ. This is the validation run
(MiniCPM-o 4.5, `MAX_PER_SUBSET=3`, two repeats, Qwen judge), so the samples are small:

| Category | Pairs | Timed overlap sessions | Stop latency (s) | Response latency (s) | C_RESPOND | C_RESUME | C_UNCERTAIN_HANDLING | C_UNKNOWN | Judged pairs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all | 12 | 12 | 1.483 ± 0.070 | 2.698 ± 0.075 | 25.0 ± 11.8% | 66.7 ± 11.8% | 8.3 ± 0.0% | 0.0 ± 0.0% | 12 |
| user_interruption | 3 | 3 | 2.161 ± 0.015 | 2.479 ± 0.128 | 50.0 ± 23.6% | 50.0 ± 23.6% | 0.0 ± 0.0% | 0.0 ± 0.0% | 3 |
| user_backchannel | 3 | 3 | 0.625 ± 0.000 | 2.308 ± 0.121 | 0.0 ± 0.0% | 100.0 ± 0.0% | 0.0 ± 0.0% | 0.0 ± 0.0% | 3 |
| talking_to_other | 3 | 3 | 1.751 ± 0.083 | 2.742 ± 0.083 | 33.3 ± 0.0% | 66.7 ± 0.0% | 0.0 ± 0.0% | 0.0 ± 0.0% | 3 |
| background_speech | 3 | 3 | 1.393 ± 0.181 | 3.204 ± 0.034 | 16.7 ± 23.6% | 50.0 ± 23.6% | 33.3 ± 0.0% | 0.0 ± 0.0% | 3 |

- `Pairs` is the selected population. Failed or ineligible sessions stay in it.
- `Timed overlap sessions` is how many overlap sessions produced timing intervals.
- `Judged pairs` is how many pairs got a valid label. The label shares use only these pairs.

Per-repeat outputs are under `$FDB_WORK/runs/$RUN_NAME/repeat-N/`:

| Path | Content |
|---|---|
| `sample-ids.txt` | The selected pairs |
| `recording/shard-*/` | Raw session traces, input and output audio, protocol reports |
| `reference-audio/` | Fixed-window WAVs used for scoring, plus `reference-manifest.json` with eligibility |
| `scores/summary.json` | Timing, ASR coverage, and (for `JUDGE=gpt`) behavior labels |
| `judge-qwen/summary.json` | Qwen behavior labels (`JUDGE=qwen` only) |
| `report.txt` | Human-readable coverage and timing report |
| `logs/` | Recorder logs; `scores/logs/` holds ASR and timing logs |

With `JUDGE=qwen`, the "Official behavior label distribution" section of `report.txt`
shows `0 / 0` and `not_prepared`. That is expected: that section only counts GPT-4o
labels. The Qwen labels are in `judge-qwen/summary.json` and in `RESULTS.md`.

## Settings

All variables are defined in `benchmarks/duplex/fdb_v15/env.sh`.

| Variable | Default | Meaning |
|---|---|---|
| `FDB_WORK` | `$HOME/fdb` | Root for downloads, environments and results |
| `RUN_NAME` | `minicpmo-48` | Results go to `$FDB_WORK/runs/$RUN_NAME` |
| `MAX_PER_SUBSET` | `12` | Pairs per category, taken in sample-ID order; empty means all 498 |
| `NUM_SHARDS` | `1` | Parallel sessions against the one server |
| `JUDGE` | `qwen` | `qwen` or `gpt` |
| `SERVER_CONFIG` | `examples/full_duplex/minicpmo.yaml` | Server config. Sampling is on, so repeats measure generation variance |
| `SERVER_GPU` / `SCORING_GPU` | `0` / `1` | GPU for the model server, and for ASR plus the judge |
| `SERVER_PORT` / `JUDGE_PORT` | `8097` / `30000` | Local ports |
| `MODEL_PATH` | `$FDB_WORK/models/MiniCPM-o-4_5` | Checkpoint directory of the model under test |
| `MODEL_REVISION` | `503e754…` | Recorded in the run manifest; must match `MODEL_PATH` |
| `SESSION_TIMEOUT_S` | `90` | Per-session client deadline; the longest v1.5 input is about 18 s |
| `OMNI_VENV` | the active venv | Venv that runs the servers and the recorder |

## Notes

- **Repeat directories are write-once.** `generate.sh N` refuses to overwrite `repeat-N`.
  To redo a repeat, delete `$FDB_WORK/runs/$RUN_NAME/repeat-N` and rerun all three steps.
  `asr.sh` and `judge.sh` resume finished work when rerun.
- **Keep `RUN_NAME` and the settings fixed across the repeats of one run.**
  Never mix repeats with different `MAX_PER_SUBSET`, `NUM_SHARDS`, `SERVER_CONFIG` or judge.
- **`NUM_SHARDS` changes what you measure.** With `NUM_SHARDS=2`, two sessions share the GPU,
  so latencies are measured under load and are not comparable to `NUM_SHARDS=1`. It must not exceed
  `max_sessions` in `SERVER_CONFIG` (2 by default); extra connections are rejected with HTTP 503.
- **Repeats need sampling.** `minicpmo-parity.yaml` decodes greedily, so its repeats
  are nearly identical. Use it for regression checks against a fixed recording, not for variance.
- **One GPU only.** Set `SCORING_GPU=0`. Stop terminal A before `asr.sh`, then start terminal B
  for `judge.sh`. Restart terminal A before the next `generate.sh`.
- **The first 48 pairs are not a random sample.** `MAX_PER_SUBSET=12` takes the first 12 samples
  of each category. This gives fast, comparable numbers between runs, but they are not
  full-dataset estimates.
- **Non-passing sessions are never dropped.** They count as ineligible in the denominators,
  and empty interval sets show as `n/a`, not zero.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `generate.sh` waits on `Waiting for .../capabilities` and stops after 30 minutes | Terminal A failed or is still loading; read its log |
| `ERROR: .../recording exists` | That repeat was already generated; use the next repeat number or delete the directory |
| A shard log reports `fail` or `error` sessions | They stay in the denominator. Read `repeat-N/logs/record-shard-*.log`. If most sessions fail, fix the server and redo the repeat |
| HTTP 503 in record logs | `NUM_SHARDS` is larger than `max_sessions` in `SERVER_CONFIG` |
| `--device cuda needs exactly one visible GPU` | `SCORING_GPU` must be a single index |
| `asr.sh` or `judge.sh` prints `WARNING: ... reported failures` | Rerun the same command with `--retry-failed`, for example `bash benchmarks/duplex/fdb_v15/asr.sh 1 --retry-failed` |
| `custom judge identity changed; use a new --out` | The judge config changed since this repeat was judged (for example, a different SGLang version). Delete `repeat-N/judge-qwen` and rerun `judge.sh N` |
| `model_mismatch` with `JUDGE=gpt` | The endpoint returned a model name other than `gpt-4o-2024-08-06`; use an endpoint that serves exactly that model |
| `ModuleNotFoundError` in `asr.sh` | Rerun `setup.sh`; it reinstalls the scoring venv packages |
