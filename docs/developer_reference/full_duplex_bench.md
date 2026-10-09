# Full-Duplex-Bench v1.5 runbook

This page runs [Full-Duplex-Bench](https://github.com/DanielLin94144/Full-Duplex-Bench)
v1.5 against a native full-duplex model served by SGLang-Omni (MiniCPM-o 4.5 by default).

The scripts live in `benchmarks/duplex/fdb_v15/`.

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
| 3. Judge | `judge.sh` | Sends the four transcripts of each pair to an LLM with the official prompt, runs the semantic A/F/U judge (Qwen only), then writes `summary.json` and `report.txt` | Judge server, GPU 1 |

`aggregate.sh` then combines repeats into one table.

The results are:

- **Stop latency**: how long the model keeps talking while the user talks over it
  (overlap of merged user and model speech spans). Lower means the model yields sooner.
- **Response latency**: time from a user speech span ending to the next model speech start.
- **Behavior labels**: the judge assigns each pair one label for how the model handled the overlap.
  `C_RESPOND` means it addressed the overlap's content. `C_RESUME` means it ignored the overlap
  and continued. `C_UNCERTAIN_HANDLING` means it asked for a repeat or showed it did not catch
  the overlap. `C_UNKNOWN` means its reply was off-target or it said nothing.
- **Semantic quality** (non-official, `JUDGE=qwen`): a label says what the model did, not
  whether it was right. A `C_RESUME` that ignored a real interruption and a `C_RESUME` that
  correctly ignored a backchannel look the same. The semantic judge grades each pair on three
  axes, each `accept`, `fail` or `unresolved`:
  `interaction_handling` (did it handle the overlap correctly for its category),
  `relevance` (did it address the operative request) and
  `grounding` (no unsupported facts, claimed actions or false memories).
  `joint` fails if any axis fails and accepts only if all three accept.
  Every verdict must quote an exact substring of the transcript it relies on; a verdict
  with a missing or invalid quote becomes `unresolved`, never a pass.
  Each cell reports quality = A/(A+F) and coverage = (A+F)/N, where N is every selected pair.

## Hardware and judge

The default layout needs two GPUs with at least 80 GB each:
GPU 0 serves the model under test, and GPU 1 runs both ASR and the judge.
The generation server keeps running while ASR runs, because they use different GPUs.

Two judges are supported, selected by `JUDGE`:

| `JUDGE` | Model | Notes |
|---|---|---|
| `qwen` (default) | [Qwen3.8-27B](https://docs.sglang.io/cookbook/autoregressive/Qwen/Qwen3.8-27B), served locally by SGLang | Reproducible and free. Behavior labels: non-thinking, greedy, seeds 1-3. Semantic judge: thinking, temperature 0, seed 1. Results are labeled non-official |
| `gpt` | `gpt-4o-2024-08-06` | The paper's judge. Needs `OPENAI_API_KEY`. The response must report exactly this model name. Behavior labels only |

The semantic judge uses the frozen prompt, response schema and six control cases in
`benchmarks/duplex/semantic/`. Before grading, every run sends the six controls and stops if
any of the 18 axis statuses differs from `control-expected.json`. Passing the controls is a
rubric check, not a measure of judge accuracy.

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
prints `{"valid": 4}` and `controls: 18/18 axis statuses match`.
If anything differs, see [Troubleshooting](#troubleshooting).

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
| 48 | ~25 min | ~1 min | ~6 min |
| 498 | ~4 h | ~10 min | ~45 min |

Sessions run in real time (about 15 s each), so generation dominates. Most of `judge.sh`
is the semantic judge: thinking-mode batches of up to 8 pairs take about 5 minutes each,
and 8 batches run concurrently.
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

With `JUDGE=qwen`, `RESULTS.md` adds a semantic table. From the same validation run:

| Category | interaction_handling quality | interaction_handling coverage | relevance quality | relevance coverage | grounding quality | grounding coverage | joint quality | joint coverage |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| all | 73.9 ± 1.6% | 95.8 ± 5.9% | 85.9 ± 5.8% | 87.5 ± 5.9% | 52.2 ± 11.0% | 79.2 ± 5.9% | 32.2 ± 17.3% | 79.2 ± 5.9% |
| user_interruption | 66.7 ± 47.1% | 83.3 ± 23.6% | 66.7 ± 47.1% | 83.3 ± 23.6% | 58.3 ± 11.8% | 83.3 ± 23.6% | 25.0 ± 35.4% | 83.3 ± 23.6% |
| user_backchannel | 100.0 ± 0.0% | 100.0 ± 0.0% | 100.0 ± 0.0% | 100.0 ± 0.0% | 100.0 ± 0.0% | 50.0 ± 23.6% | 100.0 ± 0.0% | 50.0 ± 23.6% |
| talking_to_other | 50.0 ± 23.6% | 100.0 ± 0.0% | 75.0 ± 35.4% | 66.7 ± 0.0% | 58.3 ± 11.8% | 83.3 ± 23.6% | 16.7 ± 23.6% | 83.3 ± 23.6% |
| background_speech | 83.3 ± 23.6% | 100.0 ± 0.0% | 100.0 ± 0.0% | 100.0 ± 0.0% | 16.7 ± 23.6% | 100.0 ± 0.0% | 16.7 ± 23.6% | 100.0 ± 0.0% |

Always read quality together with coverage: a high quality over a low coverage is a
judgment on few pairs.

Per-repeat outputs are under `$FDB_WORK/runs/$RUN_NAME/repeat-N/`:

| Path | Content |
|---|---|
| `sample-ids.txt` | The selected pairs |
| `recording/shard-*/` | Raw session traces, input and output audio, protocol reports |
| `reference-audio/` | Fixed-window WAVs used for scoring, plus `reference-manifest.json` with eligibility |
| `scores/summary.json` | Timing, ASR coverage, and (for `JUDGE=gpt`) behavior labels |
| `judge-qwen/summary.json` | Qwen behavior labels (`JUDGE=qwen` only) |
| `semantic-qwen/` | Semantic judge (`JUDGE=qwen` only): `controls/comparison.json`, `inputs.json` (the exact packets sent), `batches/*/` (request, raw response, validated assessments), `quality-outcomes.json` (per-pair axes, joint and any conversion reasons) and `summary.json` |
| `report.txt` | Human-readable coverage, timing and semantic quality report |
| `logs/` | Recorder logs; `scores/logs/` holds ASR and timing logs |

With `JUDGE=qwen`, the "Official behavior label distribution" section of `report.txt`
shows `0 / 0` and `not_prepared`. That is expected: that section only counts GPT-4o
labels. The Qwen labels are in `judge-qwen/summary.json` and in `RESULTS.md`.
The "Custom semantic quality" section of `report.txt` shows the semantic judge for that repeat.

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

## Scoring CLI reference

The scripts wrap two CLIs: `benchmarks.eval.benchmark_duplex_v15` records sessions, and
`benchmarks.eval.benchmark_duplex_reference` scores them with the official v1.5 ASR, timing
and behavior code. Use them directly to score a recording outside the scripts.
Run `source benchmarks/duplex/fdb_v15/env.sh` first so the variables below are set.

### Dependencies and source

The official scoring scripts and behavior prompt are not vendored. They are loaded from a
[Full-Duplex-Bench checkout](https://github.com/DanielLin94144/Full-Duplex-Bench/tree/3e799c45a045256f47d5f1c9cda90157e2d2ec9e)
at revision `3e799c45a045256f47d5f1c9cda90157e2d2ec9e`, and their SHA-256 is checked before
every run. Their
[CC BY-NC 4.0 license](https://github.com/DanielLin94144/Full-Duplex-Bench/blob/3e799c45a045256f47d5f1c9cda90157e2d2ec9e/LICENSE)
applies.

The scoring venv uses Python 3.12, torch and torchaudio 2.11.0+cu130, NeMo 3.0.0,
Silero VAD 6.2.1, numpy 2.5.3, scipy 1.18.1, soundfile 0.14.0, openai 3.28.0 and pydantic.
Each phase records its actual package versions. No command downloads a model implicitly.
ASR uses a local `nvidia/parakeet-tdt-0.6b-v2` checkpoint at revision
`ae9ad07059c7c739ffaf932226a8fe64ae2620b0`; its `.nemo` SHA-256 is
`d99e39955c9d3d0350d8fb7c75e40c64a2b2eaeb003883d7c941fd2e8747b28c`.

### Record and export

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

`record` selects the full dataset by default; `--max-per-subset N` or repeated
`--sample-id category/id` selects a subset. It exits 0 only when every selected session passes.

`export` cuts the fixed observation windows without changing the recording and needs a new
output directory. `--dataset-root` counts every dataset sample that was not recorded as missing,
so a subset run must also pass `--only category/id` for each selected pair. Repeated `--run`
accepts disjoint shards; duplicate complete captures are rejected. `--engine` is a free label
for the scored cohort.

### Score and resume

```bash
TREE=(--reference-source "$FDB_SOURCE" --tree "model=$OUT/reference-audio")
python -m benchmarks.eval.benchmark_duplex_reference asr "${TREE[@]}" \
    --out "$OUT/scores" --nemo "$PARAKEET_NEMO" --nemo-sha256 "$PARAKEET_SHA256" --device cuda
python -m benchmarks.eval.benchmark_duplex_reference timing "${TREE[@]}" \
    --out "$OUT/scores" --audio-loader soundfile
python -m benchmarks.eval.benchmark_duplex_reference prepare-judge "${TREE[@]}" --out "$OUT/scores"
python -m benchmarks.eval.benchmark_duplex_reference judge "${TREE[@]}" --out "$OUT/scores" \
    --judge gpt-4o-2024-08-06 --api-key-env OPENAI_API_KEY
python -m benchmarks.eval.benchmark_duplex_reference summarize "${TREE[@]}" --out "$OUT/scores"
```

Run the phases in order against one output directory, with the same `--tree` mapping
for each; repeat `--tree` to score several cohorts. CUDA ASR needs exactly one visible GPU.
A rerun resumes from matching receipts. `--retry-failed` retries failed work units and keeps
the earlier attempts; `--limit N` caps the work units of one invocation. Use a new output
directory after changing the recording, the scoring configuration or the reference code.
A zero exit means the phase finished, not that every sample qualified.

### Report

```bash
python -m benchmarks.eval.benchmark_duplex_reference report --scores "$OUT/scores" --engine model
```

The report prints the selected and eligible populations, protocol verdicts, ASR and timing
coverage, interval means, medians and confidence intervals, and the official behavior label
distribution. It only reads `summary.json` and the manifest; it needs no GPU, checkpoint or
API key, and does not rescore.

- `--replay replay.json` adds offline replay agreement. Format:
  `{"selected": N, "replayed": M, "samples": [{"sample": "category/id", "variant": "overlap",
  "recorded_status": "pass", "status": "match"}]}`. Each sample and variant must be unique
  and belong to the manifest.
- `--semantic-summary summary.json` adds the semantic quality section. It reads `scope`,
  `inputs_sha256`, `uncertainty_note`, `overall_axes` and `overall_joint`; each axis has
  `selected`, `accepted`, `failed` and `unresolved` counts. `benchmarks.duplex.semantic_judge`
  writes this file.

### Judges

Qwen3.8-27B served by `sglang serve` is the recommended judge, for both the behavior labels
(`custom-judge`) and the semantic judge. It is local, pinned and reproducible.
`launch_judge.sh` starts the server and writes its config and launch receipt.

GPT-4o (`judge`) is the paper's judge. Set the API key in `OPENAI_API_KEY`, never in a
command argument or manifest; `--base-url` selects a compatible endpoint. The judge sends
the exact reference prompt with seed 1 and keeps at most three attempts. A response from a
model other than `gpt-4o-2024-08-06`, a malformed label or a failed request is not a valid
label, and no fallback judge is used. Without a key the labels stay pending, and `summarize`
still runs.

### Custom behavior judge

`custom-judge` sends the exact reference prompt to a self-hosted model. Its labels are
non-official and never replace GPT-4o labels. `--source-scores` reads an existing scoring
directory; `--out` must be a separate new directory, and source files are never modified.

```bash
python -m benchmarks.eval.benchmark_duplex_reference custom-judge "${TREE[@]}" \
    --source-scores "$OUT/scores" --out "$OUT/judge-qwen" \
    --judge-config "$FDB_WORK/judge/judge-config.json" --base-url "$JUDGE_URL"
python -m benchmarks.eval.benchmark_duplex_reference custom-summarize "${TREE[@]}" \
    --source-scores "$OUT/scores" --out "$OUT/judge-qwen" \
    --judge-config "$FDB_WORK/judge/judge-config.json"
```

The config written by `launch_judge.sh`:

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

Both revisions must be 40-character commit hashes. `temperature`, `top_p` and `max_tokens`
are required; `top_k`, `min_p`, `repetition_penalty` and `enable_thinking` are sent only
when set, so omit any the endpoint does not support. The launch receipt, a path relative to
the config, records the server command, runtime version, checkpoint and tokenizer revisions
and precision. The client checks its hash, because the returned model name alone does not
prove which checkpoint was loaded. Put the endpoint key in `CUSTOM_JUDGE_API_KEY`, or a
placeholder such as `EMPTY` for an endpoint without authentication.

Matching reruns skip finished samples. A changed config or selection needs a new `--out`,
and changed transcripts, audio or ASR receipts prevent reuse. `--retry-failed` retries only
transport and parsing failures. Wrong models, invalid labels and non-`stop` finishes stay
invalid. Label shares use valid labels only, and the per-category Wilson 95% intervals
describe dataset sampling, not human agreement or generation variance.

### Measurement definitions

- **Input pacing.** Audio is sent in 80 ms packets. Every packet start must stay within
  80 ms of its source cadence, otherwise the session is ineligible. This bounds transport
  jitter only.
- **Observation window.** The output is scored over `[0, T]`, where T is the input length,
  anchored at the first input send. Received audio plays first-in first-out with no buffer,
  so initial delay, gaps and queueing remain. Audio still queued at T is cut, and the rest
  is silence; the result is mono 16 kHz PCM16 with exactly the input sample count. Errors
  inside the window make the session ineligible; silence is a valid output.
- **Intervals.** The official timing code merges VAD speech spans with 0.6 s (user) and
  0.5 s (model) gaps. Stop intervals are overlaps of user and model spans. Response intervals
  run from a user span end to the next strictly later model span start. These are whole-file
  intervals, not event-local latencies. `summary.json` reports their pooled means and medians
  with sample-cluster bootstrap CIs.
- **Missing values.** Empty interval sets and missing labels are null, never zero latency
  or 0%. Counts keep every selected, eligible, invalid and unscored sample.
- **Limits.** Simulated playout is not acoustic latency, and the fixed window can cut long
  replies. One generation per sample does not estimate generation variance. Prosody and MOS
  are not covered.

### Verification

The unit tests cover export, missing samples, valid silence, timing, ASR deduplication,
bounded judging and resumption. With the pinned checkout they also run the official formulas
on synthetic fixtures, without a GPU or API. Run them from the sglang-omni venv:

```bash
FDB_REFERENCE_SOURCE="$FDB_SOURCE" python -m pytest -q \
    tests/unit_test/benchmarks/test_duplex_reference*.py
```

Export is deterministic: re-exporting a recording reproduces every eligible WAV byte for byte,
and re-summarizing saved transcripts and intervals reproduces the same summaries, including
coverage and bootstrap intervals.

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
| `Control check failed: this judge configuration is not accepted.` | The judge missed a control case; the printed lines show which. Do not grade with this configuration. Check that the judge server runs the pinned Qwen3.8-27B revision |
| `... changed since this directory was created; use a new --out` | The semantic judge settings, prompt or inputs changed since this repeat was graded. Delete `repeat-N/semantic-qwen` and rerun `judge.sh N` |
| `model_mismatch` with `JUDGE=gpt` | The endpoint returned a model name other than `gpt-4o-2024-08-06`; use an endpoint that serves exactly that model |
| `ModuleNotFoundError` in `asr.sh` | Rerun `setup.sh`; it reinstalls the scoring venv packages |
