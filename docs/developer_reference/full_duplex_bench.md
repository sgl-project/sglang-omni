# Full-Duplex-Bench runbook

This page runs [Full-Duplex-Bench](https://github.com/DanielLin94144/Full-Duplex-Bench) (FDB) v1.0 and v1.5 against a full-duplex model served by SGLang-Omni (MiniCPM-o 4.5 by default). Every step is a subcommand of `python -m benchmarks.duplex.fdb`; the code is in `benchmarks/duplex/fdb/`.

## What is measured

FDB has two datasets, and they test different behavior:

- **v1.0** tests turn-taking. The model should stay quiet through a pause in the user's turn, come in once the turn ends, stop and answer when the user interrupts it, and, while the user talks at length, give short backchannels such as "mm-hmm" without taking the turn.
- **v1.5** tests overlap handling. The user asks something, and while the model answers, more speech arrives: an interruption, a backchannel, speech to someone else, or background speech. The question is what the model does with it.

The two measure different things, so a run takes samples from both by default. Each dataset gets its own result table, and their numbers are never pooled.

### v1.0

v1.0 has 727 samples in five subsets. A sample is one input audio file plus an annotation (the pause positions, the end of the user's turn, the interruption, or the human backchannel timing), streamed in one session.

| Subset | Samples | Task | Source | Reported numbers |
|---|---:|---|---|---|
| `synthetic_pause_handling` | 137 | Pause handling | Synthetic | Takeover rate (lower is better) |
| `candor_pause_handling` | 216 | Pause handling | Candor conversations | Takeover rate (lower is better) |
| `candor_turn_taking` | 119 | Turn-taking | Candor conversations | Takeover rate (higher), latency (lower) |
| `synthetic_user_interruption` | 200 | User interruption | Synthetic | Takeover rate (higher), latency (lower), relevance (higher) |
| `icc_backchannel` | 55 | Backchannel | ICC conversations | Takeover rate (lower), backchannel frequency, backchannel JSD (lower) |

The following official reference numbers come from the evaluator at the pinned revision, run unchanged on the transcript of the model's output. The runbook also reports [unified v1.0 metrics](#results) for pause handling, turn taking and interruption:

- **Takeover rate**: the share of samples in which the model took the turn, meaning its transcript spans at least 1 s or has more than 3 words.
- **Latency**: the time from the annotated end of the user's turn to the model's first word, averaged over the samples in which the model took the turn. A start before the end of the turn counts as 0.
- **Relevance**: a 0-5 rating of the model's reply to the interruption, given by the judge (see [Judges](#judges)).
- **Backchannel frequency** and **JSD**: how often the model backchannels, and how far its backchannel timing is from the human one.

The capture, VAD and judge can differ from the paper's, so compare these numbers between runs of this runbook, not with the published tables.

### v1.5

v1.5 has 498 pairs in four categories: `user_interruption` (200), `user_backchannel` (98), `talking_to_other` (100) and `background_speech` (100). A pair is two input files with the same user query: `input.wav` contains the later overlapping speech and `clean_input.wav` does not. Each file is streamed in its own session.

- **Stop latency**: how long the model keeps talking while the other speech overlaps it. Lower is better on `user_interruption`, where the model should give way. In the other three categories the model should keep its turn, so the overlap mostly tracks how long the other speech lasts.
- **Response latency**: the time from the end of a user speech span to the start of the next model speech. Lower is better when a reply is due. Both latencies are whole-file intervals from the official timing code, not latencies around the overlap event.
- **Behavior labels**: the judge gives each pair one label for how the model handled the overlap. `C_RESPOND`: it addressed the overlap's content. `C_RESUME`: it ignored the overlap and continued. `C_UNCERTAIN_HANDLING`: it asked for a repeat or showed that it did not catch the overlap. `C_UNKNOWN`: its reply was off-target, or it said nothing. The share that should be high is `C_RESPOND` on `user_interruption` and `C_RESUME` in the other categories; `C_UNKNOWN` should be low.
- **Semantic quality** (`JUDGE=qwen` only, not part of the paper): a label says what the model did, not whether that was right, so a `C_RESUME` that ignored a real interruption looks the same as one that correctly ignored a backchannel. The semantic judge grades each pair on three axes, each `accept`, `fail` or `unresolved`: `interaction_handling` (the overlap was handled correctly for its category), `relevance` (the reply addressed the request that mattered) and `grounding` (no unsupported facts, claimed actions or false memories). `joint` accepts only when all three accept. A verdict must quote the transcript it relies on, and a verdict with a missing or wrong quote becomes `unresolved`. Each cell reports `quality = A/(A+F)` and `coverage = (A+F)/N` over all N selected pairs; read the two together, since a high quality over a low coverage judges few pairs.

## Setup

You need one GPU with at least 80 GB (H100 or H200), and `git`, `uv` and the FFmpeg shared libraries (version 4 to 8) on the host. Run every command in this page from the repository root, with the sglang-omni virtualenv active (`which python` should print its interpreter).

```bash
python -m benchmarks.duplex.fdb setup
```

`setup` is safe to rerun and skips finished steps. It installs the `minicpm-o` extra of sglang-omni (`onnx` and `einops`) into the active venv and puts everything else under `FDB_WORK` (default `$HOME/fdb`):

| Path | Content |
|---|---|
| `Full-Duplex-Bench/` | Official scoring code at the pinned revision `3e799c4`; file hashes are checked |
| `dataset/v1.0/`, `dataset/v1.5/` | The two datasets from the official Google Drive; `dataset/v1.0.revision` and `dataset/v1.5.revision` hold the archive hashes |
| `scoring-venv/` | Python 3.12 with NeMo, torch, Silero VAD and torchcodec for ASR, timing and the official evaluators. torchcodec loads FFmpeg, and `setup` checks that it can |
| `models/parakeet-tdt-0.6b-v2/` | ASR checkpoint, SHA-256 checked |
| `models/MiniCPM-o-4_5/` | Model under test at revision `503e754`. To reuse an existing copy, symlink it here or set `MODEL_PATH` |
| `models/Qwen3.8-27B/` | Judge checkpoint; its commit is saved in `models/Qwen3.8-27B.revision` |

### Judges

`JUDGE` selects the judge:

| `JUDGE` | v1.5 behavior labels | v1.5 semantic quality | v1.0 interruption relevance |
|---|---|---|---|
| `qwen` (default) | [Qwen3.8-27B](https://docs.sglang.io/cookbook/autoregressive/Qwen/Qwen3.8-27B), served locally: non-thinking, greedy, seeds 1-3 | Qwen3.8-27B: thinking, temperature 0, seed 1 | Qwen3.8-27B with the official prompt: non-thinking, seed 0 |
| `gpt` | `gpt-4o-2024-08-06`, the paper's judge | not run | `gpt-4-turbo`, which the official evaluator requests |

`JUDGE=gpt` needs `OPENAI_API_KEY`. With `JUDGE=qwen` the labels and ratings are not the paper's, but the judge is local and pinned, so runs with the same settings stay comparable with each other. The semantic judge sends six control cases (`benchmarks/duplex/semantic/`) together with its first grading batches and keeps the grades only if all 18 control axis statuses match `control-expected.json`; otherwise it discards them and stops. The controls check the rubric, not the judge's accuracy.

A judge starts only when the step has a request for it: the v1.5 pairs and the v1.0 interruption subset need one, while the other v1.0 subsets are scored without an LLM. A run with only those subsets starts no judge server and needs no API key.

## Run the benchmark

A run has three steps, then an aggregation. Each step starts the server it needs and stops it when it exits.

| Step | Subcommand | What it does |
|---|---|---|
| 1 | `generate` | Starts the model server, streams the selected inputs to `/v1/realtime` (two sessions per v1.5 pair, one per v1.0 sample), records the model's audio, stops the server, then cuts a fixed observation window from each session |
| 2 | `asr` | Transcribes the windows with word timestamps (Parakeet), computes v1.5 official speech intervals, then writes v1.0 unified metrics from the saved transcript and annotations on CPU |
| 3 | `judge` | Runs the official v1.0 evaluator on the v1.0 subsets that need no LLM, then judges the v1.5 pairs and the v1.0 interruption subset (with `JUDGE=qwen` it starts the judge server for them and stops it after), and writes the per-repeat summaries |
| | `aggregate` | Combines the repeats of a run into `RESULTS.md` |

Run a small preflight first. It takes one pair per v1.5 category and three samples per v1.0 subset (about 15 minutes) and catches environment problems before a long run. v1.0 needs three samples rather than one because the official turn-taking and interruption evaluators divide by the number of takeovers and fail when no selected sample takes the turn.

```bash
python -m benchmarks.duplex.fdb generate --run-name preflight --v15-per-subset 1 --v10-per-subset 3
python -m benchmarks.duplex.fdb asr --run-name preflight
python -m benchmarks.duplex.fdb judge --run-name preflight
python -m benchmarks.duplex.fdb aggregate --run-name preflight
```

The preflight passes when `generate` prints `{"pass": 8}` and `{"eligible_pairs": 4, ...}` for v1.5 and `{"pass": 15}` with 3 eligible samples per v1.0 subset; `asr` prints `{"ok": 16}`, `{"ok": 8}` and 3 transcripts per v1.0 subset; and `judge` prints `{"valid": 4}`, `controls: 18/18 axis statuses match` and the parsed v1.0 results of each subset as JSON (the evaluator's own `[Result]` block is in `repeat-N/v10/reference/logs/evaluate-<subset>.log`). If anything differs, see [Troubleshooting](#troubleshooting).

A run is one fixed selection of samples under `$FDB_WORK/runs/RUN_NAME`. Decoding uses sampling, so two generations of the same inputs differ; to measure that variance, run the three steps several times under one run name with `--repeat 1`, `--repeat 2`, and so on. `--repeat N` only names the output directory `repeat-N/`; one command always runs once. `aggregate` presents unified v1.0 scores per repeat and means ± standard deviations in its v1.5 and official v1.0 reference tables.

```bash
for repeat in 1 2 3; do
    python -m benchmarks.duplex.fdb generate --run-name minicpmo-12 --repeat "$repeat" --v15-per-subset 12 --v10-per-subset 12
    python -m benchmarks.duplex.fdb asr --run-name minicpmo-12 --repeat "$repeat"
    python -m benchmarks.duplex.fdb judge --run-name minicpmo-12 --repeat "$repeat"
done
python -m benchmarks.duplex.fdb aggregate --run-name minicpmo-12
```

`generate` refuses to write into a repeat that already holds a recording; to redo one, delete its `repeat-N/` directory and run its three steps again. `asr` and `judge` resume finished work when rerun. For v1.0, `asr` runs unified scoring after a successful or already completed `reference-asr`; failed ASR does not trigger scoring. The unified scorer does not need `judge` or its `summary.json`.

Every subcommand takes `--log-level` (`info` by default). With `--log-level debug`, a step also prints the server and recorder logs it writes, each line prefixed with the log's name, such as `[judge-server]` or `[record-shard-0]`. The logs are written under `repeat-N/` at either level (see the files table in [Results](#results)).

### Choosing samples

Only `generate` selects samples; `asr` and `judge` score whatever the repeat recorded. Selection works the same way for both datasets and is never random: each subset (v1.5 calls them categories) takes its first N samples in numeric ID order (1, 2, …, 10, 11, …). So every repeat with the same options gets the same samples, and a smaller N is always a prefix of a larger one.

The options come in two matching sets, `--v15-*` and `--v10-*`. Both default to 12 per subset. A count of 0 leaves that dataset out of the run; leaving both out is an error.

| `generate` option | Effect |
|---|---|
| `--v15-per-subset N`, `--v10-per-subset N` | N samples from every subset of that dataset (default 12); `all` takes the whole dataset; `0` leaves it out |
| `--v15-subset-count NAME=N`, `--v10-subset-count NAME=N` | Overrides the count for one subset; `0` skips it, `all` takes all of it. Repeatable |
| `--v15-sample-id NAME/ID`, `--v10-sample-id NAME/ID` | Exactly these samples. Repeatable |
| `--v15-sample-ids-file PATH`, `--v10-sample-ids-file PATH` | One `NAME/ID` per line, such as another run's `repeat-1/sample-ids.txt` or `repeat-1/v10/sample-ids.txt` |

```bash
# Both full datasets.
python -m benchmarks.duplex.fdb generate --run-name minicpmo-full --v15-per-subset all --v10-per-subset all
# v1.0 only; v1.5 only.
python -m benchmarks.duplex.fdb generate --run-name v10-only --v15-per-subset 0
python -m benchmarks.duplex.fdb generate --run-name v15-only --v10-per-subset 0
# 20 v1.5 interruptions, 10 v1.5 backchannels and every v1.0 backchannel sample.
python -m benchmarks.duplex.fdb generate --run-name small \
    --v15-per-subset 0 --v15-subset-count user_interruption=20 --v15-subset-count user_backchannel=10 \
    --v10-per-subset 0 --v10-subset-count icc_backchannel=all
# Three specific samples.
python -m benchmarks.duplex.fdb generate --run-name debug \
    --v15-sample-id user_interruption/1 --v15-sample-id background_speech/7 \
    --v10-sample-id candor_turn_taking/37
# Another run's selection.
python -m benchmarks.duplex.fdb generate --run-name replay \
    --v15-sample-ids-file "$FDB_WORK/runs/minicpmo-12/repeat-1/sample-ids.txt" \
    --v10-sample-ids-file "$FDB_WORK/runs/minicpmo-12/repeat-1/v10/sample-ids.txt"
```

`generate` prints how many samples it selected per subset and writes their IDs to `repeat-N/sample-ids.txt` (v1.5) and `repeat-N/v10/sample-ids.txt` (v1.0). All repeats of a run must select the same samples, so pass the same options every time; a repeat with a different selection stops with an error.

Sessions run in real time, so `generate` dominates the wall time. Approximate times per repeat on one H200, one session at a time, without the 1 minute of model server startup per `generate` and 2 minutes of judge server startup per `judge`:

| Selection | `generate` | `asr` | `judge` |
|---|---|---|---|
| 48 v1.5 pairs (12 per category) | ~25 min | ~1 min | ~6 min |
| 498 v1.5 pairs | ~4 h | ~10 min | ~45 min |
| 60 v1.0 samples (12 per subset) | ~20 min | ~1 min | ~1 min |
| 727 v1.0 samples | ~3.6 h | ~10 min | ~5 min |

A v1.5 session lasts about 15 s and a v1.0 sample about 18 s on average. Most of a v1.5 `judge` is the semantic judge, which runs 8 thinking-mode batches of up to 8 pairs concurrently at about 5 minutes per batch; the Qwen behavior judge sends up to 16 requests at once. `generate --num-shards 2` roughly halves the generation time, but read the note on it in [Notes](#notes) first.

## Results

`aggregate` prints its tables and writes them to `$FDB_WORK/runs/RUN_NAME/RESULTS.md`, with one section per dataset the run selected. The v1.0 unified section shows each repeat separately, without averaging its rates, quantiles or confidence intervals. In the v1.5 and official v1.0 reference tables, a numeric cell is the mean ± sample standard deviation over repeats with available values; a count cell shows one value or a range when repeats differ.

The v1.5 section has one row per category and one for all of them. This example is a validation run (MiniCPM-o 4.5, 3 pairs per category, two repeats, Qwen judge), so the samples are small:

| Category | Pairs | Timed overlap sessions | Stop latency (s) | Response latency (s) | C_RESPOND | C_RESUME | C_UNCERTAIN_HANDLING | C_UNKNOWN | Judged pairs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all | 12 | 12 | 1.483 ± 0.070 | 2.698 ± 0.075 | 25.0 ± 11.8% | 66.7 ± 11.8% | 8.3 ± 0.0% | 0.0 ± 0.0% | 12 |
| user_interruption | 3 | 3 | 2.161 ± 0.015 | 2.479 ± 0.128 | 50.0 ± 23.6% | 50.0 ± 23.6% | 0.0 ± 0.0% | 0.0 ± 0.0% | 3 |
| user_backchannel | 3 | 3 | 0.625 ± 0.000 | 2.308 ± 0.121 | 0.0 ± 0.0% | 100.0 ± 0.0% | 0.0 ± 0.0% | 0.0 ± 0.0% | 3 |
| talking_to_other | 3 | 3 | 1.751 ± 0.083 | 2.742 ± 0.083 | 33.3 ± 0.0% | 66.7 ± 0.0% | 0.0 ± 0.0% | 0.0 ± 0.0% | 3 |
| background_speech | 3 | 3 | 1.393 ± 0.181 | 3.204 ± 0.034 | 16.7 ± 23.6% | 50.0 ± 23.6% | 33.3 ± 0.0% | 0.0 ± 0.0% | 3 |

`Pairs` counts every selected pair, failed or not. `Timed overlap sessions` counts the overlap sessions that produced timing intervals. `Judged pairs` counts the pairs with a valid label, and the label shares are over these pairs only.

With `JUDGE=qwen` the v1.5 section adds the semantic table. From the same validation run:

| Category | interaction_handling quality | interaction_handling coverage | relevance quality | relevance coverage | grounding quality | grounding coverage | joint quality | joint coverage |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| all | 73.9 ± 1.6% | 95.8 ± 5.9% | 85.9 ± 5.8% | 87.5 ± 5.9% | 52.2 ± 11.0% | 79.2 ± 5.9% | 32.2 ± 17.3% | 79.2 ± 5.9% |
| user_interruption | 66.7 ± 47.1% | 83.3 ± 23.6% | 66.7 ± 47.1% | 83.3 ± 23.6% | 58.3 ± 11.8% | 83.3 ± 23.6% | 25.0 ± 35.4% | 83.3 ± 23.6% |
| user_backchannel | 100.0 ± 0.0% | 100.0 ± 0.0% | 100.0 ± 0.0% | 100.0 ± 0.0% | 100.0 ± 0.0% | 50.0 ± 23.6% | 100.0 ± 0.0% | 50.0 ± 23.6% |
| talking_to_other | 50.0 ± 23.6% | 100.0 ± 0.0% | 75.0 ± 35.4% | 66.7 ± 0.0% | 58.3 ± 11.8% | 83.3 ± 23.6% | 16.7 ± 23.6% | 83.3 ± 23.6% |
| background_speech | 83.3 ± 23.6% | 100.0 ± 0.0% | 100.0 ± 0.0% | 100.0 ± 0.0% | 16.7 ± 23.6% | 100.0 ± 0.0% | 16.7 ± 23.6% | 100.0 ± 0.0% |

The v1.0 unified section first shows per-repeat coverage and metric tables for pause handling, turn taking and user interruption. Coverage lists declared, selected, eligible and ineligible samples, exclusion reasons and the minimum observable horizon after the reference point. A sample rate uses eligible samples as its denominator; its Wilson 95% interval describes sampling uncertainty, while the all-selected bounds allow each ineligible outcome to be either false or true. Per-pause and seconds-per-second ratios pool their respective denominators over eligible samples. Other intervals bootstrap whole samples. `n/a` with zero eligible samples means no data. `not reached` with a positive eligible count means the requested response-latency quantile was not observed, including nonresponses; a window shorter than a response threshold limits the corresponding rate. The report labels any missing, stale, unsupported or incomplete unified artifacts unavailable and gives a CPU-only backfill command. It does not display stale numeric scores. Backchannel remains official-reference-only.

The following v1.0 official reference table has one row per selected subset. `Selected` counts the recorded samples, `Eligible` the ones the official evaluator can score, and `Evaluated` the ones it scored; then come `Takeover rate`, `Latency (s)`, `Relevance (0-5)`, `Judge retries`, `Backchannel frequency` and `Backchannel JSD`, with `n/a` where the subset's task does not report a number. A subset whose sessions all failed still has a row, with `Eligible 0`. A subset that a repeat has not evaluated is listed under the table as not evaluated yet (`judge` has not reached it) or failed (with its evaluator log); its cells average the repeats that did evaluate it, and show `n/a` when none did. A nonzero `Judge retries` means the interruption takeover rate is inflated (see [v1.0 scoring](#v10-scoring)).

Each repeat writes these files under `$FDB_WORK/runs/RUN_NAME/repeat-N/`:

| Path | Content |
|---|---|
| `sample-ids.txt`, `v10/sample-ids.txt` | The selected v1.5 pairs (empty when v1.5 was left out) and v1.0 samples (`v10/` exists only when v1.0 was selected) |
| `recording/shard-*/`, `v10/recording/shard-*/` | Raw session traces, input and output audio, protocol reports |
| `logs/`, `v10/logs/` | `logs/` has the server logs and the v1.5 recorder logs, `v10/logs/` the v1.0 recorder logs; the v1.5 ASR and timing logs are in `scores/logs/` |
| `reference-audio/` | v1.5 fixed-window WAVs and `reference-manifest.json` with eligibility |
| `scores/summary.json` | v1.5 timing, ASR coverage and, with `JUDGE=gpt`, behavior labels |
| `judge-qwen/summary.json` | v1.5 behavior labels with `JUDGE=qwen` |
| `semantic-qwen/` | Semantic judge: control comparison, the exact packets sent, raw responses, per-pair outcomes and `summary.json` |
| `report.txt` | Readable v1.5 report: coverage, timing and semantic quality |
| `v10/reference/` | v1.0 windows (`<subset>/<id>/output.wav`), `manifest.json`, the ASR receipt `asr.json`, per-repeat `unified.json` and `unified-samples.jsonl`, official `summary.json`, the judge ledger in `judge/` and evaluator logs in `logs/` |

With `JUDGE=qwen`, the "Official behavior label distribution" section of `report.txt` shows `0 / 0` and `not_prepared`, because that section counts GPT-4o labels only; the Qwen labels are in `judge-qwen/summary.json` and `RESULTS.md`.

## Settings

Per-run choices (`--run-name`, `--repeat`, sample selection, `--num-shards`, `--log-level`) are command options; `--help` on any subcommand lists them. Machine settings are environment variables, read in `benchmarks/duplex/fdb/common.py`:

| Variable | Default | Meaning |
|---|---|---|
| `FDB_WORK` | `$HOME/fdb` | Root for downloads, environments and results |
| `JUDGE` | `qwen` | `qwen` or `gpt` |
| `CUDA_VISIBLE_DEVICES` | unset | The one GPU index this job uses. See [Running jobs in parallel](#running-jobs-in-parallel) |
| `GPU` | `0` | The GPU index when `CUDA_VISIBLE_DEVICES` is unset |
| `SERVER_PORT`, `JUDGE_PORT` | `8097 + 10×GPU`, `30000 + 10×GPU` | Local ports. The judge also uses the next port for NCCL. Avoid 29500-29899, which the model server uses for NCCL |
| `MODEL_PATH` | `$FDB_WORK/models/MiniCPM-o-4_5` | Checkpoint of the model under test |
| `MODEL_REVISION` | `503e754…` | Recorded in the run manifest; must match `MODEL_PATH` |
| `JUDGE_MODEL_PATH` | `$FDB_WORK/models/Qwen3.8-27B` | Checkpoint of the Qwen judge |
| `SESSION_TIMEOUT_S`, `V10_SESSION_TIMEOUT_S` | `90`, `230` | Per-session client deadline for v1.5 and v1.0. v1.5 inputs run up to about 18 s; v1.0 inputs average about 18 s and some run past 90 s. Neither deadline may exceed the recorder's 230 s |
| `SCORING_VENV` | `$FDB_WORK/scoring-venv` | Venv that runs ASR, timing, the evaluators and the judge clients |

## Running jobs in parallel

One job uses one GPU. To run several at once, open one terminal per GPU and give each its own GPU index and run name:

```bash
export FDB_WORK="${FDB_WORK:-$HOME/fdb}"
unset GPU SERVER_PORT JUDGE_PORT
export CUDA_VISIBLE_DEVICES=1    # 0 in the first terminal, 1 in the second, and so on
RUN=minicpmo-30-gpu1
python -m benchmarks.duplex.fdb generate --run-name "$RUN" --v15-per-subset 30 --v10-per-subset 30
python -m benchmarks.duplex.fdb asr --run-name "$RUN"
python -m benchmarks.duplex.fdb judge --run-name "$RUN"
python -m benchmarks.duplex.fdb aggregate --run-name "$RUN"
```

The ports, the judge config and the compile caches follow the GPU index: the model port is `8097 + 10×GPU`, the judge port `30000 + 10×GPU`, the judge config is in `$FDB_WORK/judge/port-<judge port>/` and the caches in `$FDB_WORK/cache/gpu-<index>/`. The first line `generate` prints confirms them, for example `Job GPU 1: model port 8107, judge port 30010 (nccl 30011), ...`. The `unset` line matters when a shell profile sets `GPU` or a port. Never point two terminals at the same run name.

## Notes

- **Keep the settings fixed across the repeats of a run.** Mixing repeats with different selections, `--num-shards` or judges makes their mean meaningless; use a new run name instead.
- **`--num-shards` changes what you measure.** With `--num-shards 2`, two sessions share the GPU, so the latencies are measured under load and do not compare with one shard. The model server keeps `max_sessions` at 2, so `generate` refuses a larger value up front.
- **The first N samples are not a random sample.** They give fast numbers that compare between runs, not estimates for the full dataset.
- **Failed sessions are never dropped.** They stay in the denominators as ineligible, and an empty set of intervals shows as `n/a`, not as zero.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `ERROR: the model server did not become ready` (or the judge server) | The message ends with the last lines of the server log; the full log is `repeat-N/logs/model-server.log` or `judge-server.log` |
| `ERROR: the model server exited with code N during the recordings` | The model server died while recording; the message ends with the last lines of `repeat-N/logs/model-server.log`. Fix the cause, delete the repeat and generate it again |
| `ERROR: the judge server exited with code N before ...` or `during ...` | The judge server died; the message names the phase and ends with the last lines of `repeat-N/logs/judge-server.log`. Fix the cause and rerun `judge` with `--retry-failed`. If the phase is the v1.5 semantic judge, delete `repeat-N/semantic-qwen` first, because the batches it could not grade are saved as unresolved and a rerun reuses them |
| `ERROR: something already serves ...` | Another server holds the port, often left from an interrupted run or a second job on the same GPU. Stop it, or give this job another `CUDA_VISIBLE_DEVICES` |
| `ERROR: .../repeat-N already holds a recording` | That repeat was generated before; use the next `--repeat`, or delete the directory to redo it |
| `ERROR: ... selects different pairs` or `... selects different v1.0 samples` | An earlier repeat of this run used another selection; pass the same options, or use a new run name |
| `ERROR: .../repeat-N is a v1.5-only repeat` | The earlier repeats of this run left v1.0 out; continue with the option change the message names, or use a new run name |
| `ERROR: both datasets are left out` | Give at least one dataset a nonzero count |
| `ERROR: .../dataset/v1.0 lacks the v1.0 subset(s) ...` | The workspace was set up without v1.0; rerun `setup`, which adds it |
| `ERROR: .../reference-manifest.json is missing` or `.../v10/reference/manifest.json is missing` | `generate` recorded that dataset but did not finish its export; delete the repeat and generate it again |
| A shard log reports `fail` or `error` sessions | They stay in the denominators. Read `repeat-N/logs/record-shard-*.log` or `repeat-N/v10/logs/record-shard-*.log`; if most sessions fail, fix the server and redo the repeat |
| `ERROR: --num-shards N exceeds the model server's max_sessions` | Lower `--num-shards`; `generate` refuses values above the server capacity before recording |
| HTTP 503 in a record log | Another client may hold the server's sessions; check for competing recordings |
| `ERROR: set CUDA_VISIBLE_DEVICES to exactly one GPU index`, or `--device cuda needs exactly one visible GPU` | Export a single index, a different one in each terminal |
| `ERROR: GPU and CUDA_VISIBLE_DEVICES disagree` | Unset `GPU`, or set it to the same index |
| `WARNING: a v1.5 phase reported failures` | Rerun the command the warning prints; it adds `--retry-failed` |
| `WARNING: the v1.0 ASR failed` | `repeat-N/v10/reference/logs/asr.log` has the official script's output and the terminal has the traceback. A rerun of `asr` transcribes every v1.0 sample again |
| `WARNING: v1.0 unified scoring failed` | ASR completed, but the saved export, transcript or annotation did not score. Fix the stated cause and run the CPU-only backfill command printed in the warning; it does not repeat ASR or v1.5 work. Rerun `aggregate` to refresh `RESULTS.md` |
| `Unified metrics unavailable` in `RESULTS.md` | The report names a missing export/ASR file, missing output pair, stale hashes, unsupported version or incomplete subset. Repair the export or ASR if named, then use the per-repeat backfill command in the report and rerun `aggregate` |
| `WARNING: the v1.0 evaluation failed for ...` | The traceback is at the end of `repeat-N/v10/reference/logs/evaluate-<subset>.log`. A `ZeroDivisionError` in `candor_turn_taking` or `synthetic_user_interruption` means no selected sample took the turn, so select more samples in a new run. After fixing another cause, rerun `judge`; it evaluates the missing subsets again |
| `judge request ... got no parsable rating in 3 attempts` | The v1.0 relevance judge answered without a rating; the ledger in `v10/reference/judge/` has each reply. The official parser needs `Analysis: <text>`, a line break, then `I would rate the AI's response as <digit>`. The attempts share a seed, so a local judge repeats the same reply and a rerun fails again |
| `custom judge identity changed; use a new --out` | The judge config changed since this repeat was judged, for example after an SGLang upgrade. Delete `repeat-N/judge-qwen` and rerun `judge` |
| `Control check failed: this judge configuration is not accepted.` | The judge missed a control case, listed in the output; nothing graded alongside the controls is kept. Check that the judge server runs the pinned Qwen3.8-27B revision; do not grade with a configuration that fails the controls |
| `... changed since this directory was created; use a new --out` | The semantic judge settings, prompt or inputs changed since this repeat was graded. Delete `repeat-N/semantic-qwen` and rerun `judge` |
| `model_mismatch` with `JUDGE=gpt` | The endpoint answered with a model other than `gpt-4o-2024-08-06`; use an endpoint that serves exactly that model |
| `No module named 'onnx'` (or `einops`) in `model-server.log`, or `ModuleNotFoundError` in `asr` | Rerun `setup` |
| `setup` stops with `torchcodec cannot load its FFmpeg libraries` | Install the FFmpeg shared libraries (version 4 to 8) and rerun `setup` |

## Scoring CLI reference

The runbook wraps three CLIs, which also score a recording by hand: `benchmarks.eval.benchmark_duplex_v15` records v1.5 sessions, `benchmarks.eval.benchmark_duplex_reference` scores them with the official v1.5 ASR, timing and behavior code, and `benchmarks.eval.benchmark_duplex_v10` records and scores v1.0 sessions. Run `eval "$(python -m benchmarks.duplex.fdb env)"` first to set the variables the commands below use. Recording and export run in the sglang-omni venv (`python`); ASR, timing, judging and evaluation need the packages `setup` installs in the scoring venv, so those commands run with `"$SCORING_PYTHON"`. `record` needs the model server and `custom-judge` the judge server; `python -m benchmarks.duplex.fdb serve-model` and `serve-judge` start them in the foreground until Ctrl-C.

### Dependencies and source

The official scoring scripts and the behavior prompt are not vendored. They are loaded from a [Full-Duplex-Bench checkout](https://github.com/DanielLin94144/Full-Duplex-Bench/tree/3e799c45a045256f47d5f1c9cda90157e2d2ec9e) at revision `3e799c45a045256f47d5f1c9cda90157e2d2ec9e`, and their SHA-256 is checked before every run. Their [CC BY-NC 4.0 license](https://github.com/DanielLin94144/Full-Duplex-Bench/blob/3e799c45a045256f47d5f1c9cda90157e2d2ec9e/LICENSE) applies.

The scoring venv uses Python 3.12, torch and torchaudio 2.11.0+cu130, NeMo 3.0.0, Silero VAD 6.2.1, numpy 2.5.3, scipy 1.18.1, soundfile 0.14.0, openai 3.28.0, pydantic, python-dotenv and torchcodec 0.11. Each phase records the package versions it ran with. No command downloads a model on its own. ASR uses a local `nvidia/parakeet-tdt-0.6b-v2` checkpoint at revision `ae9ad07059c7c739ffaf932226a8fe64ae2620b0`, `.nemo` SHA-256 `d99e39955c9d3d0350d8fb7c75e40c64a2b2eaeb003883d7c941fd2e8747b28c`.

### v1.5 scoring

Record and export:

```bash
OUT=results/fdb-manual
python -m benchmarks.eval.benchmark_duplex_v15 record \
    --profile minicpmo-native-pr2377 \
    --dataset-root "$FDB_DATASET" --dataset-revision "$(cat "$FDB_WORK/dataset/v1.5.revision")" \
    --url "$REALTIME_URL" --model "$MODEL_ID" --model-revision "$MODEL_REVISION" \
    --server-revision "$(git rev-parse HEAD)" --timeout "$SESSION_TIMEOUT_S" \
    --output "$OUT/recording"

python -m benchmarks.eval.benchmark_duplex_reference export \
    --engine model --trace-format realtime-pcm16-v1 --run "$OUT/recording" \
    --dataset-root "$FDB_DATASET" --out "$OUT/reference-audio"
```

`record` takes the full dataset unless `--max-per-subset N` or repeated `--sample-id category/id` narrows it, and exits 0 only when every selected session passes. `export` cuts the observation windows into a new directory without touching the recording. With `--dataset-root`, every dataset pair that was not recorded counts as missing, so a partial run also passes `--only category/id` for each recorded pair. Repeated `--run` takes disjoint shards; `--engine` is a free label for the scored cohort.

A turn-based (half-duplex) endpoint speaks the conversation protocol instead of the native session protocol: it takes a `turn_detection` object in `session.update`, sends none of the `sglang.input_audio.*` receipts or `sglang.unit.done`, and streams `response.audio.delta` at 24 kHz. The profile `qwen3-omni-half-duplex` switches the recorder, the protocol oracle and the export to that protocol:

```bash
python -m benchmarks.eval.benchmark_duplex_v15 record \
    --profile qwen3-omni-half-duplex \
    --turn-detection '{"type": "server_vad", "silence_duration_ms": 500}' \
    --dataset-root "$FDB_DATASET" --dataset-revision "$(cat "$FDB_WORK/dataset/v1.5.revision")" \
    --url "$REALTIME_URL" --model "$MODEL_ID" --model-revision "$MODEL_REVISION" \
    --server-revision "$(git rev-parse HEAD)" --timeout "$SESSION_TIMEOUT_S" \
    --output "$OUT/recording"

python -m benchmarks.eval.benchmark_duplex_reference export \
    --engine model --trace-format realtime-legacy-pcm16-v1 --run "$OUT/recording" \
    --dataset-root "$FDB_DATASET" --out "$OUT/reference-audio"
```

- `--turn-detection` is required for this profile and rejected for native ones. It is sent once per session, so a detector sweep is one run per setting against the same server: `server_vad` takes `threshold`, `prefix_padding_ms` and `silence_duration_ms`, `semantic_vad` takes `eagerness`, and both take `interrupt_response`. The manifest records it under `config.legacy`. `benchmark_duplex_v10 record` takes the same options.
- A session fails unless `session.updated` echoes the requested `turn_detection.type`, so a detector that silently falls back is caught. There are no input or drain receipts: the session ends `--legacy-tail` seconds (default 1.0) after the input with a no-op `session.update`, whose `session.updated` proves the server is alive.
- Any `error` event fails the session. A response cancelled with reason `turn_detected` is accepted only after an `input_audio_buffer.speech_started` during it.
- Simulated playout follows the server's barge-in: it is cut at `output_audio_buffer.cleared`, or at `input_audio_buffer.speech_started` unless the session set `interrupt_response: false`. The export applies the same cut, records `barge_in_cuts`, and marks a session eligible only when its protocol verdict is `pass`.

Score:

```bash
TREE=(--reference-source "$FDB_SOURCE" --tree "model=$OUT/reference-audio")
"$SCORING_PYTHON" -m benchmarks.eval.benchmark_duplex_reference asr "${TREE[@]}" \
    --out "$OUT/scores" --nemo "$PARAKEET_NEMO" --nemo-sha256 "$PARAKEET_SHA256" --device cuda
"$SCORING_PYTHON" -m benchmarks.eval.benchmark_duplex_reference timing "${TREE[@]}" \
    --out "$OUT/scores" --audio-loader soundfile
"$SCORING_PYTHON" -m benchmarks.eval.benchmark_duplex_reference prepare-judge "${TREE[@]}" --out "$OUT/scores"
"$SCORING_PYTHON" -m benchmarks.eval.benchmark_duplex_reference judge "${TREE[@]}" --out "$OUT/scores" \
    --judge gpt-4o-2024-08-06 --api-key-env OPENAI_API_KEY
"$SCORING_PYTHON" -m benchmarks.eval.benchmark_duplex_reference summarize "${TREE[@]}" --out "$OUT/scores"
"$SCORING_PYTHON" -m benchmarks.eval.benchmark_duplex_reference report --scores "$OUT/scores" --engine model
```

Run the phases in order on one output directory with the same `--tree`; repeat `--tree` to score several cohorts together. CUDA ASR needs exactly one visible GPU. A rerun resumes from the receipts; `--retry-failed` retries failed work units, and `--limit N` caps the units of one call. After changing the recording, the scoring configuration or the reference code, use a new output directory. A zero exit means the phase finished, not that every sample qualified.

`report` reads only `summary.json` and the manifest, so it needs no GPU or key. It prints the populations, protocol verdicts, ASR and timing coverage, interval statistics with confidence intervals, and the behavior label distribution. `--semantic-summary` adds the semantic judge's `summary.json`, and `--replay replay.json` adds offline replay agreement in the form `{"selected": N, "replayed": M, "samples": [{"sample": "category/id", "variant": "overlap", "recorded_status": "pass", "status": "match"}]}`.

The GPT judge reads its key from `OPENAI_API_KEY`, never from an argument or a manifest; `--base-url` selects a compatible endpoint. It sends the exact reference prompt with seed 1 and makes at most three attempts. A reply from a model other than `gpt-4o-2024-08-06`, a malformed label or a failed request leaves the label invalid, with no fallback judge. Without a key the labels stay pending and `summarize` still runs.

`custom-judge` sends the same prompt to a self-hosted model. Its labels never replace GPT-4o labels. It reads an existing scoring directory and writes to a separate new one:

```bash
"$SCORING_PYTHON" -m benchmarks.eval.benchmark_duplex_reference custom-judge "${TREE[@]}" \
    --source-scores "$OUT/scores" --out "$OUT/judge-qwen" \
    --judge-config "$JUDGE_CONFIG" --base-url "$JUDGE_URL"
"$SCORING_PYTHON" -m benchmarks.eval.benchmark_duplex_reference custom-summarize "${TREE[@]}" \
    --source-scores "$OUT/scores" --out "$OUT/judge-qwen" \
    --judge-config "$JUDGE_CONFIG"
```

`judge` and `serve-judge` write the judge config to `$JUDGE_CONFIG`, which is `$FDB_WORK/judge/port-$JUDGE_PORT/judge-config.json`:

```json
{
  "model_id": "Qwen/Qwen3.8-27B",
  "model_revision": "<40-hex commit>",
  "tokenizer_id": "Qwen/Qwen3.8-27B",
  "tokenizer_revision": "<40-hex commit>",
  "served_model": "qwen3.8-27b",
  "precision": "bf16",
  "enable_thinking": false,
  "decoding": {
    "temperature": 0.0, "top_p": 1.0, "top_k": -1,
    "min_p": 0.0, "repetition_penalty": 1.0, "max_tokens": 512
  },
  "seeds": [1, 2, 3],
  "server_launch_receipt": "launch-receipt.json",
  "server_launch_receipt_sha256": "<sha256 of the receipt>"
}
```

Both revisions must be full commit hashes. `temperature`, `top_p` and `max_tokens` are required; the other decoding fields and `enable_thinking` are sent only when present, so leave out any the endpoint does not support. The launch receipt records the server command, runtime version, checkpoint and tokenizer revisions and precision, and the client checks its hash, because the model name an endpoint returns does not prove which checkpoint it loaded. The endpoint key goes in `CUSTOM_JUDGE_API_KEY`, or a placeholder such as `EMPTY` without authentication. A rerun with the same config skips finished samples; a changed config or selection needs a new `--out`. `--concurrency` (default 16) caps the requests in flight and is recorded with the run. `--retry-failed` retries transport and parsing failures only; wrong models, invalid labels and non-`stop` finishes stay invalid. The per-category Wilson 95% intervals describe dataset sampling, not judge agreement or generation variance.

### v1.0 scoring

v1.0 runs the official `get_transcript/asr.py` and `evaluation/evaluate.py` unchanged:

```bash
OUT=results/fdb10
python -m benchmarks.eval.benchmark_duplex_v10 record \
    --profile minicpmo-native-pr2377 \
    --dataset-root "$FDB10_DATASET" --dataset-revision "$(cat "$FDB_WORK/dataset/v1.0.revision")" \
    --url "$REALTIME_URL" --model "$MODEL_ID" --model-revision "$MODEL_REVISION" \
    --server-revision "$(git rev-parse HEAD)" --timeout "$V10_SESSION_TIMEOUT_S" \
    --output "$OUT/recording"
python -m benchmarks.eval.benchmark_duplex_v10 reference-export \
    --run "$OUT/recording" --dataset-root "$FDB10_DATASET" --out "$OUT/reference"
"$SCORING_PYTHON" -m benchmarks.eval.benchmark_duplex_v10 reference-asr \
    --tree "$OUT/reference" --reference-source "$FDB_SOURCE" \
    --nemo "$PARAKEET_NEMO" --nemo-sha256 "$PARAKEET_SHA256" --device cuda
CUDA_VISIBLE_DEVICES="" "$SCORING_PYTHON" -m benchmarks.eval.benchmark_duplex_v10 unified-score \
    --tree "$OUT/reference"
"$SCORING_PYTHON" -m benchmarks.eval.benchmark_duplex_v10 reference-evaluate \
    --tree "$OUT/reference" --reference-source "$FDB_SOURCE"
```

- `reference-export` writes each sample's window as `<subset>/<id>/output.wav` with its annotation into a new directory, plus `manifest.json`. Repeated `--run` takes shards. An ineligible session gets no folder and keeps its reasons in the manifest.
- `reference-asr` transcribes every sample with the local Parakeet checkpoint. For user interruption it cuts the audio at the end of the interruption and shifts the word times back, so the transcript holds only the reply. It writes `output.json` per sample and the receipt `asr.json`, and runs once per export.
- `unified-score` reads the saved manifest, ASR transcript and annotations on CPU, then writes `unified.json` and `unified-samples.jsonl`. It is independent of the official evaluator and can be rerun for an existing export without repeating ASR. To backfill a runbook repeat, use `eval "$(python -m benchmarks.duplex.fdb env)"`, then `CUDA_VISIBLE_DEVICES="" "$SCORING_PYTHON" -m benchmarks.eval.benchmark_duplex_v10 unified-score --tree "$FDB_WORK/runs/RUN_NAME/repeat-N/v10/reference"`, replacing `RUN_NAME` and `N` with the recorded run and repeat; rerun `aggregate` afterward.
- `reference-evaluate` runs the evaluator per subset and stores the printed `[Result]` numbers, the counts and the log path in `summary.json`. `--subset` evaluates only the named subsets; the rest can follow later.

The interruption relevance is rated by `gpt-4-turbo` with seed 0, as the official evaluator requests, for every interruption the model took over. That needs `OPENAI_API_KEY` (or the variable named by `--api-key-env`); without a key, evaluate the other subsets with `--subset`. `--base-url URL --served-model NAME` sends the same prompts to a self-hosted judge, with thinking off and the reply capped at 512 tokens, since the evaluator sets neither and SGLang would stop at 128 tokens. `summary.json` records the endpoint actually used and marks the rating official only for OpenAI's endpoint without `--served-model`. Every exchange is appended to `judge/user-interruption-<UTC time>.jsonl`.

The official evaluator has a retry bug: when a rating does not parse, it resends the same request without limit, and each retry counts that sample's takeover again. The client stops after three identical requests, which fails the subset, and `summary.json` reports `retries`. A nonzero value means the interruption takeover rate is inflated.

`transcribe` and `score` (`fdb-v10-synthetic-v4`) remain as a diagnostic beside the official numbers, not in their place. They add gates the official evaluator does not have: turn-taking samples where the model already speaks when the user's turn ends, and interruptions where the model was silent when the user cut in or talked straight through. Report their gate counts next to the official numbers.

### Measurement definitions

- **Input pacing.** Audio is sent in 80 ms packets. A packet that starts more than 80 ms off its source cadence makes the session ineligible. This bounds transport jitter only.
- **Observation window.** Both datasets score the output over `[0, T]`, where T is the input length, counted from the first input packet. Received audio plays first-in first-out, so initial delay, gaps and queueing stay in; audio still queued at T is cut, and the rest is silence. The window is mono 16 kHz PCM16 with exactly the input's sample count. An error inside the window makes the session ineligible; silence is valid output.
- **v1.5 intervals.** The official timing code merges VAD speech spans across gaps of 0.6 s (user) and 0.5 s (model). Stop intervals are overlaps of user and model spans; response intervals run from the end of a user span to the start of the next model span. `summary.json` reports their pooled means and medians with sample-cluster bootstrap intervals.
- **v1.0 takeover and latency.** The evaluator reads the Parakeet transcript of the window. A sample took the turn when the words span at least 1 s or number more than 3. The latency is the first word's start minus the annotated end of the user's turn, clamped at 0. Pause handling looks at the whole window, not only at the annotated pauses.
- **Missing values.** Empty interval sets and missing labels are null, never a zero latency or 0%. The counts keep every selected, eligible, invalid and unscored sample.
- **Limits.** Simulated playout is not acoustic latency, and the window can cut long replies. One generation per sample does not estimate generation variance. Prosody and MOS are not measured.

### Verification

The unit tests run the official formulas on synthetic fixtures without a GPU or API key, and cover export, missing samples, valid silence, timing, ASR deduplication, bounded and concurrent judging, the semantic judge's control gate, resumption, and the runbook's selection, phase gating, server checks and aggregation for both datasets:

```bash
FDB_REFERENCE_SOURCE="$FDB_SOURCE" python -m pytest -q \
    tests/unit_test/benchmarks/test_duplex*.py tests/unit_test/benchmarks/test_fdb*.py
```

Export is deterministic: exporting a recording again reproduces every eligible WAV byte for byte, and summarizing saved transcripts and intervals again reproduces the same summaries.
