# ASR benchmark runbook

This page runs the ASR benchmarks of SGLang-Omni on one GPU and explains how to measure them accurately. Audio goes in and text comes out. Correctness is WER, or CER and cpCER for MOSS-Transcribe-Diarize. Performance is throughput, latency and RTF.

The workloads are the ones in the ASR CI (`.github/workflows/test-asr-ci.yaml`). Run every command in `bash`, in order. The only values you set yourself are in [Variables](#variables).

## What is Measured

| CI stage | Model | Data | Concurrency |
|---|---|---|---|
| 1 | MOSS-Transcribe-Diarize, multiple speakers | `movies800times` (800 samples, run once without and once with streaming), `aishell4_long` (20), `googletime` (25) | 16 |
| 2 | Qwen3-ASR-1.7B, Whisper-large-v3 or Fun-ASR-Nano | SeedTTS EN (1088) and ZH (2020) | 32 |

CI stage 1 also has `aishell4_long90`, one 90 minute sample stitched together from three `aishell4_long` clips. It only runs through pytest (see [Run the CI tests](#run-the-ci-tests)). CI stage 2 runs one of the three models per CI run, picked at random unless a `run-fun-asr-nano-2512`, `run-qwen3-asr-1.7b` or `run-whisper-large-v3` label is set with `run-asr-ci` or `run-all-ci`.

This page uses one server on one GPU, with the same datasets, concurrency and request settings as CI.

## Variables

Run the host block on the host before Step 1, and again in every new host shell. Change only `GPU` and `HF_TOKEN`:

```bash
export GPU=0
export HF_TOKEN=hf_your_token_here
export IMAGE=hongccc/sglang-omni@sha256:ebe4239e29a764ee3a2806385c061c5fd438a26f01458e503d3822dcba5790df
export CPUSET=$(cat /sys/bus/pci/devices/$(nvidia-smi -i $GPU --query-gpu=pci.bus_id --format=csv,noheader | tail -c 13 | tr A-F a-f)/local_cpulist)
echo $CPUSET
```

`GPU` is the GPU under test. No other job may use it. `HF_TOKEN` needs access to the three private MOSS-Transcribe-Diarize datasets. `IMAGE` is the CI image. `CPUSET` is the CPUs on the NUMA node of `GPU`, such as `0-55,112-167`.

Run the container block inside the container after Step 1, and again after every `docker exec`. Change only `COMMIT` (the commit to measure) and `NAME` (a label for its results):

```bash
export COMMIT=origin/main
export NAME=main
export SEEDTTS_REV=27f4c1adee83b5b29b7c4b375f6b976324bda308
export HF_HUB=/root/.cache/huggingface/hub
export QWEN3_ASR=$HF_HUB/models--Qwen--Qwen3-ASR-1.7B/snapshots/7278e1e70fe206f11671096ffdd38061171dd6e5
export WHISPER=$HF_HUB/models--openai--whisper-large-v3/snapshots/06f233fe06e710322aca913c1bc4249a0d71fce1
export FUN_ASR=$HF_HUB/models--FunAudioLLM--Fun-ASR-Nano-2512-hf/snapshots/7ef6cafdc445b5beff0b730cbeb51952c793ca6e
export MOSS_TD=$HF_HUB/models--OpenMOSS-Team--MOSS-Transcribe-Diarize/snapshots/704aa4a9c304e8520be88901e0d1960158ef5b15
```

CI loads these models by repo id without a revision. The SHAs above are what the 2026-10-09 CI runners had cached. Each server here loads its model from these snapshot directories, so the revision cannot drift.

## Step 1: Start the container (on the host)

```bash
mkdir -p $HOME/asr-bench
docker run -d --name asr-bench \
    --gpus device=$GPU --cpuset-cpus $CPUSET \
    -v /dev/shm:/dev/shm \
    -v $HOME/.cache/huggingface:/root/.cache/huggingface \
    -v $HOME/asr-bench:/work \
    -e HF_TOKEN \
    $IMAGE sleep infinity
docker exec -it asr-bench bash
```

Every command after this runs inside the container. Run the container block of [Variables](#variables). Then `nvidia-smi -L` must print exactly one GPU.

If the `docker exec` shell closes, the variables are lost and a running client stops. Run `docker exec -it asr-bench bash` on the host, the container block and the `RUN` line of the cut off run. Then run that run again from Step 4.

## Step 2: Build the venv

These lines follow `.github/scripts/prepare_omni_venv.sh`, without the extras and the CosyVoice checkout that other models need. So `freeze.txt` will not match a CI venv exactly.

```bash
[ -d /work/sglang-omni ] || git clone https://github.com/sgl-project/sglang-omni.git /work/sglang-omni
cd /work/sglang-omni
git fetch origin
git checkout --detach $COMMIT
rm -rf .venv
uv venv --system-site-packages .venv -p /usr/bin/python3.12
echo 'import site; site.addsitedir("/opt/sglang/lib/python3.12/site-packages")' > .venv/lib/python3.12/site-packages/sglang-image.pth
. .venv/bin/activate
python .github/scripts/omni_missing_dependencies.py pyproject.toml | xargs -r -d '\n' -n 1 python -m pip install
python .github/scripts/omni_missing_dependencies.py --overrides pyproject.toml | xargs -r -d '\n' python -m pip install --no-deps
uv pip install --no-deps -e .
python .github/scripts/verify_omni_installed_pins.py .venv/bin/python .
```

The last line must print `Verified <N> exact dependency pins`. Run this block again whenever `COMMIT` changes. For a pull request, run `git fetch origin pull/<N>/head` first and set `COMMIT` to its SHA.

## Step 3: Download data and models

Download the SeedTTS dataset and the four models at their pinned revisions:

```bash
cd /work/sglang-omni && . .venv/bin/activate
python -m benchmarks.dataset.prepare --dataset seedtts --revision $SEEDTTS_REV
python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3-ASR-1.7B', revision='7278e1e70fe206f11671096ffdd38061171dd6e5')"
python -c "from huggingface_hub import snapshot_download; snapshot_download('openai/whisper-large-v3', revision='06f233fe06e710322aca913c1bc4249a0d71fce1')"
python -c "from huggingface_hub import snapshot_download; snapshot_download('FunAudioLLM/Fun-ASR-Nano-2512-hf', revision='7ef6cafdc445b5beff0b730cbeb51952c793ca6e')"
python -c "from huggingface_hub import snapshot_download; snapshot_download('OpenMOSS-Team/MOSS-Transcribe-Diarize', revision='704aa4a9c304e8520be88901e0d1960158ef5b15')"
ls $QWEN3_ASR/config.json $WHISPER/config.json $FUN_ASR/config.json $MOSS_TD/config.json
```

The `ls` line must print four paths and no error. If it fails, run `echo $HF_HOME`. If that prints a path, set `HF_HUB` to `$HF_HOME/hub` and rerun the container block with that value.

The three MOSS-Transcribe-Diarize datasets are private. Download them once with `HF_TOKEN` set. An access error means the token cannot read them:

```bash
python -c "import datasets; datasets.load_dataset('zhaochenyang20/movies800time', split='validation')"
python -c "import datasets; datasets.load_dataset('zhaochenyang20/AISHELL4', split='validation')"
python -c "import datasets; datasets.load_dataset('zhaochenyang20/googletime', split='validation')"
```

The client cannot pin these datasets. Record the loaded dataset fingerprints and the remote repository heads right after the download, while still online. The remote head alone does not identify the cached rows. Compare both values between machines or dates before comparing numbers:

```bash
mkdir -p /work/results
python - > /work/results/moss_dataset_revisions.txt <<'EOF'
from huggingface_hub import HfApi
from datasets import load_dataset
for r in ("zhaochenyang20/movies800time", "zhaochenyang20/AISHELL4", "zhaochenyang20/googletime"):
    dataset = load_dataset(r, split="validation")
    print(r, "remote_head=" + HfApi().dataset_info(r).sha, "loaded_fingerprint=" + dataset._fingerprint)
EOF
cat /work/results/moss_dataset_revisions.txt
```

Every run sets `HF_HUB_OFFLINE=1` (Step 4), so the datasets stay at the cached revisions. Check that every dataset loads offline. The last line must print `1088 2020 800 20 25`:

```bash
HF_HUB_OFFLINE=1 python -c "
from benchmarks.dataset.seedtts import load_seedtts_samples as s, SEEDTTS_DATASET_ID as d
from benchmarks.tasks.transcribe_diarize import load_movies800_samples as m
a = dict(split='validation', audio_column='audio', expected_column='transcription', expected_sample_count=None)
print(len(s(d, split='en', revision='$SEEDTTS_REV')), len(s(d, split='zh', revision='$SEEDTTS_REV')), len(m('zhaochenyang20/movies800time', **a)), len(m('zhaochenyang20/AISHELL4', **a)), len(m('zhaochenyang20/googletime', **a)))"
```

## Step 4: Start a run

A run measures `COMMIT` once, with a fresh server for each model. Pick the run name first. `warm` is the unscored first run (see [Measuring accurately](#measuring-accurately)), then `1`, `2` and `3`:

```bash
export RUN=warm
```

Then:

```bash
cd /work/sglang-omni && . .venv/bin/activate
export HF_HUB_OFFLINE=1 SGLANG_OMNI_STRICT_PORT=1 SGLANG_OMNI_TORCH_COMPILE_DEFAULT=0 SGLANG_OMNI_STARTUP_TIMEOUT=1800
export TORCHINDUCTOR_CACHE_DIR=/work/torchinductor
find /work/results -name server.pid -exec bash -c 'kill -9 -- -$(cat "$1") 2>/dev/null' _ {} \;
timeout 600 bash -c 'until [ $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits) -lt 1024 ]; do sleep 5; done' && echo GPU_FREE
export OUT=/work/results/$NAME/run$RUN
rm -rf $OUT
mkdir -p $OUT
git rev-parse HEAD > $OUT/commit.txt
python -m pip freeze --exclude-editable > $OUT/freeze.txt
```

CI sets `SGLANG_OMNI_TORCH_COMPILE_DEFAULT=0`. Without it, every model that does not set compile itself (Qwen3-ASR, Whisper, Fun-ASR) runs `torch.compile`. MOSS-Transcribe-Diarize turns compile on itself. `TORCHINDUCTOR_CACHE_DIR` keeps its compile cache on the `/work` mount, so it survives a new container. Delete that directory to measure a cold start. `SGLANG_OMNI_STARTUP_TIMEOUT=1800` raises the server's own startup limit from 600 s. On a host with few CPU cores, a cold Whisper start can spend more than 600 s capturing CUDA graphs.

The `find` line stops every server left over from an earlier run. The block must print `GPU_FREE`. If it does not, run `docker restart asr-bench` on the host, reattach as in Step 1, and run this step again. The block deletes earlier results of the same run, so a run is always redone whole.

## Step 5: Run SeedTTS ASR

Each block starts a server for one model, runs EN and ZH, then stops the server. Run the blocks one at a time.

In every block, the `curl` line must print `PORT_FREE`, the `timeout 1800` line `SERVER_READY` and the last line `GPU_FREE`. The `timeout 1800` line stops waiting as soon as the server process exits, and then prints the end of `server.log` instead of `SERVER_READY`. If `GPU_FREE` does not print, some server processes survived the `kill`. Restart the container as in Step 4 and redo the run. The `setsid` line starts the server in its own process group, so `kill` stops all of it. Each client run prints a line ending with `evaluated=1088/1088` (EN) or `evaluated=2020/2020` (ZH). The client exits with code 0 even when it skipped samples, so check [coverage](#coverage) after the run.

### Qwen3-ASR-1.7B

```bash
mkdir -p $OUT/qwen3
curl -sf localhost:8000/health > /dev/null || echo PORT_FREE
setsid bash -c 'echo $$ > "$1"; shift; exec "$@"' _ $OUT/qwen3/server.pid \
    .venv/bin/sgl-omni serve --model-path $QWEN3_ASR --model-name Qwen/Qwen3-ASR-1.7B --port 8000 \
    > $OUT/qwen3/server.log 2>&1 &
timeout 1800 bash -c 'until curl -sf localhost:8000/health > /dev/null; do sleep 5; grep -q "^State:[[:space:]]*[A-Y]" /proc/$(cat "$1")/status 2>/dev/null || exit 1; done' _ $OUT/qwen3/server.pid && echo SERVER_READY || tail -n 20 $OUT/qwen3/server.log
python -m benchmarks.eval.benchmark_asr_seedtts --port 8000 --model-path Qwen/Qwen3-ASR-1.7B --lang en \
    --concurrencies 32 --repeats 1 --warmup \
    --dataset-revision $SEEDTTS_REV --output $OUT/qwen3/asr_en.json
python -m benchmarks.eval.benchmark_asr_seedtts --port 8000 --model-path Qwen/Qwen3-ASR-1.7B --lang zh \
    --concurrencies 32 --repeats 1 --warmup \
    --dataset-revision $SEEDTTS_REV --output $OUT/qwen3/asr_zh.json
kill -- -$(cat $OUT/qwen3/server.pid)
timeout 600 bash -c 'until [ $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits) -lt 1024 ]; do sleep 5; done' && echo GPU_FREE
```

### Whisper-large-v3

```bash
mkdir -p $OUT/whisper
curl -sf localhost:8000/health > /dev/null || echo PORT_FREE
setsid bash -c 'echo $$ > "$1"; shift; exec "$@"' _ $OUT/whisper/server.pid \
    .venv/bin/sgl-omni serve --model-path $WHISPER --model-name openai/whisper-large-v3 --port 8000 \
    > $OUT/whisper/server.log 2>&1 &
timeout 1800 bash -c 'until curl -sf localhost:8000/health > /dev/null; do sleep 5; grep -q "^State:[[:space:]]*[A-Y]" /proc/$(cat "$1")/status 2>/dev/null || exit 1; done' _ $OUT/whisper/server.pid && echo SERVER_READY || tail -n 20 $OUT/whisper/server.log
python -m benchmarks.eval.benchmark_asr_seedtts --port 8000 --model-path openai/whisper-large-v3 --lang en \
    --concurrencies 32 --repeats 1 --warmup \
    --dataset-revision $SEEDTTS_REV --output $OUT/whisper/asr_en.json
python -m benchmarks.eval.benchmark_asr_seedtts --port 8000 --model-path openai/whisper-large-v3 --lang zh \
    --concurrencies 32 --repeats 1 --warmup \
    --dataset-revision $SEEDTTS_REV --output $OUT/whisper/asr_zh.json
kill -- -$(cat $OUT/whisper/server.pid)
timeout 600 bash -c 'until [ $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits) -lt 1024 ]; do sleep 5; done' && echo GPU_FREE
```

### Fun-ASR-Nano

```bash
mkdir -p $OUT/fun
curl -sf localhost:8000/health > /dev/null || echo PORT_FREE
setsid bash -c 'echo $$ > "$1"; shift; exec "$@"' _ $OUT/fun/server.pid \
    .venv/bin/sgl-omni serve --model-path $FUN_ASR --model-name FunAudioLLM/Fun-ASR-Nano-2512-hf --port 8000 \
    > $OUT/fun/server.log 2>&1 &
timeout 1800 bash -c 'until curl -sf localhost:8000/health > /dev/null; do sleep 5; grep -q "^State:[[:space:]]*[A-Y]" /proc/$(cat "$1")/status 2>/dev/null || exit 1; done' _ $OUT/fun/server.pid && echo SERVER_READY || tail -n 20 $OUT/fun/server.log
python -m benchmarks.eval.benchmark_asr_seedtts --port 8000 --model-path FunAudioLLM/Fun-ASR-Nano-2512-hf --lang en \
    --concurrencies 32 --repeats 1 --warmup \
    --dataset-revision $SEEDTTS_REV --output $OUT/fun/asr_en.json
python -m benchmarks.eval.benchmark_asr_seedtts --port 8000 --model-path FunAudioLLM/Fun-ASR-Nano-2512-hf --lang zh \
    --concurrencies 32 --repeats 1 --warmup \
    --dataset-revision $SEEDTTS_REV --output $OUT/fun/asr_zh.json
kill -- -$(cat $OUT/fun/server.pid)
timeout 600 bash -c 'until [ $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits) -lt 1024 ]; do sleep 5; done' && echo GPU_FREE
```

## Step 6: Run MOSS-Transcribe-Diarize

The server arguments and request settings match the CI test (`tests/test_model/test_asr_ci_multi_speaker.py`). Like CI, one server serves all four datasets. The checks are the same as in Step 5. Each client line must also print `CLIENT_OK`. The client exits with code 0, and so prints `CLIENT_OK`, only when every request succeeded. Otherwise it prints an error or an `Evaluation failed` line.

For `movies800times` CI sends no `max_new_tokens`, so the server picks a budget scaled to the audio length, and the request timeout is 300 s. The client CLI always sends `--max-new-tokens` (default 65536) and has no flag to send none. So the two `movies800times` lines below call the client with that default set to `None`. The `aishell4_long` and `googletime` lines use the CLI defaults (65536 tokens, 1800 s), as CI does.

```bash
mkdir -p $OUT/moss_td
curl -sf localhost:8000/health > /dev/null || echo PORT_FREE
setsid bash -c 'echo $$ > "$1"; shift; exec "$@"' _ $OUT/moss_td/server.pid \
    .venv/bin/sgl-omni serve --model-path $MOSS_TD --model-name OpenMOSS-Team/MOSS-Transcribe-Diarize --port 8000 \
    --asr.engine.max_running_requests 16 --asr.engine.cuda_graph_max_bs 16 --mem-fraction-static 0.8 \
    > $OUT/moss_td/server.log 2>&1 &
timeout 1800 bash -c 'until curl -sf localhost:8000/health > /dev/null; do sleep 5; grep -q "^State:[[:space:]]*[A-Y]" /proc/$(cat "$1")/status 2>/dev/null || exit 1; done' _ $OUT/moss_td/server.pid && echo SERVER_READY || tail -n 20 $OUT/moss_td/server.log
python -c 'import sys, benchmarks.eval.benchmark_asr_transcribe_diarize as b; b.DEFAULT_MAX_NEW_TOKENS = None; sys.exit(b.main(sys.argv[1:]))' \
    --use-existing-server --port 8000 --dataset movies800times --max-concurrency 16 --request-timeout-s 300 \
    --output-dir $OUT/moss_td/movies800times && echo CLIENT_OK
python -c 'import sys, benchmarks.eval.benchmark_asr_transcribe_diarize as b; b.DEFAULT_MAX_NEW_TOKENS = None; sys.exit(b.main(sys.argv[1:]))' \
    --use-existing-server --port 8000 --dataset movies800times --stream --max-concurrency 16 --request-timeout-s 300 \
    --output-dir $OUT/moss_td/movies800times_stream && echo CLIENT_OK
python -m benchmarks.eval.benchmark_asr_transcribe_diarize --use-existing-server --port 8000 \
    --dataset aishell4_long --max-concurrency 16 --output-dir $OUT/moss_td/aishell4_long && echo CLIENT_OK
python -m benchmarks.eval.benchmark_asr_transcribe_diarize --use-existing-server --port 8000 \
    --dataset googletime --max-concurrency 16 --output-dir $OUT/moss_td/googletime && echo CLIENT_OK
kill -- -$(cat $OUT/moss_td/server.pid)
timeout 600 bash -c 'until [ $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits) -lt 1024 ]; do sleep 5; done' && echo GPU_FREE
```

The `config.max_new_tokens` stored in each `transcribe_diarize_speed_results.json` shows what was sent. It must be `null` for both `movies800times` directories.

Then go back to Step 4 with the next `RUN` (`warm`, `1`, `2`, `3`). To measure only one model, run Step 4 and only that model's block, the same way in every run.

## Step 7: Summarize

Save this script once. For one result directory it reads the three scored runs (`run1`, `run2` and `run3`, never `runwarm`), checks coverage, and prints the mean, min and max of each metric:

```bash
cat > /work/results/summarize.py <<'EOF'
import glob, json, os, statistics, sys
SEED = ("corpus_wer", "throughput_samples_per_s", "latency_mean_s", "latency_p95_s", "rtf_mean")
MOSS = ("cer", "cp_cer", "cer_valid_samples", "cp_cer_valid_samples", "speaker_timestamp_der")
SPEED = ("throughput_qps", "latency_mean_s", "latency_p95_s", "rtf_mean")
EXPECTED = {"asr_en.json": 1088, "asr_zh.json": 2020, "movies800times": 800,
            "movies800times_stream": 800, "aishell4_long": 20, "googletime": 25}
runs = [f"{sys.argv[1]}/run{i}" for i in (1, 2, 3)]
expected_keys = {f"{model}/{name}" for model in ("qwen3", "whisper", "fun") for name in ("asr_en.json", "asr_zh.json")}
expected_keys |= {f"moss_td/{name}" for name in EXPECTED if not name.endswith(".json")}
rows = {}
for run in runs:
    for path in glob.glob(f"{run}/*/asr_*.json"):
        r = json.load(open(path))["results"][0]
        ok = r["evaluated"] == r["total"] == EXPECTED[os.path.basename(path)] and r["skipped"] == 0
        rows.setdefault(os.path.relpath(path, run), []).append((ok, {k: r[k]["mean"] for k in SEED}))
    for path in glob.glob(f"{run}/moss_td/*/transcribe_diarize_results.json"):
        p = json.load(open(path))
        s = json.load(open(path.replace("_results.json", "_speed_results.json")))["speed"]
        name = os.path.basename(os.path.dirname(path))
        ok = p["summary"]["evaluated"] == p["summary"]["total_samples"] == EXPECTED[name] and s.get("failed_requests") == 0
        d = p["diarization_metrics"]
        rows.setdefault("moss_td/" + name, []).append((ok, {k: d.get(k) for k in MOSS} | {k: s.get(k) for k in SPEED}))
print(f"scored runs: {[os.path.basename(r) for r in runs]}")
print(f"missing workloads: {sorted(expected_keys - rows.keys())}")
print(f"coverage complete: {rows.keys() == expected_keys and all(len(rs) == 3 and all(ok for ok, _ in rs) for rs in rows.values())}")
for key, rs in sorted(rows.items()):
    print(f"{key}  runs={len(rs)} complete={len(rs) == len(runs) and all(ok for ok, _ in rs)}")
    for k in rs[0][1]:
        v = [m[k] for _, m in rs]
        if None in v:
            print(f"  {k:26} missing in a run: {v}")
        else:
            print(f"  {k:26} mean {statistics.mean(v):.6f}  min {min(v):.6f}  max {max(v):.6f}")
EOF
python /work/results/summarize.py /work/results/$NAME
```

Report the mean with the min and max of the three scored runs. `coverage complete: True` requires all ten workloads in all three runs with the expected sample counts. If you intentionally measured one model, its workload lines should each show `runs=3 complete=True`, while coverage of the full suite remains incomplete. An incomplete workload can still print partial means, so report its missing coverage instead of presenting those means as three full runs.

## Where each metric lives

SeedTTS, in `$OUT/<model>/asr_en.json` and `$OUT/<model>/asr_zh.json`, where `<model>` is `qwen3`, `whisper` or `fun`. With `--repeats 1` each `mean` is the single measured value.

| Metric | Field | Unit |
|---|---|---|
| WER | `results[0].corpus_wer.mean` | fraction (0.0120 is 1.20%) |
| Throughput | `results[0].throughput_samples_per_s.mean` | samples per second |
| Latency mean, p95 | `results[0].latency_mean_s.mean`, `results[0].latency_p95_s.mean` | seconds |
| RTF mean | `results[0].rtf_mean.mean` | ratio |
| Coverage | `results[0].evaluated`, `results[0].total`, `results[0].skipped` | sample counts |

MOSS-Transcribe-Diarize, in `$OUT/moss_td/<dataset>/`. The first five rows are in `transcribe_diarize_results.json`, the rest in `transcribe_diarize_speed_results.json`:

| Metric | Field | Unit |
|---|---|---|
| Coverage | `summary.evaluated`, `summary.total_samples` | sample counts |
| CER, cpCER | `diarization_metrics.cer`, `diarization_metrics.cp_cer` | fraction |
| Valid sample counts | `diarization_metrics.cer_valid_samples`, `diarization_metrics.cp_cer_valid_samples` | sample counts |
| Valid samples | `per_sample[].id` with `per_sample[].cer_valid` and `per_sample[].cp_cer_valid` | IDs |
| Speaker timestamp DER | `diarization_metrics.speaker_timestamp_der`, `diarization_metrics.speaker_timestamp_der_skipped_parse_error` | fraction, count |
| Failed requests | `speed.failed_requests` | request count |
| Throughput | `speed.throughput_qps` | requests per second |
| Latency mean, p95 | `speed.latency_mean_s`, `speed.latency_p95_s` | seconds |
| RTF mean | `speed.rtf_mean` | ratio |

The printed client output uses `diarization_metrics_percent`, the same values times 100. The `speed` values are stored rounded: throughput and latency to 3 decimals, RTF to 4. When every request of a run fails, `speed` has no throughput, latency or RTF fields.

## Coverage

A run is complete only when every workload you ran has all its samples and no failures:

- SeedTTS: `evaluated` equals `total` (1088 for EN, 2020 for ZH) and `skipped` is 0.
- MOSS-Transcribe-Diarize: `summary.evaluated` equals `summary.total_samples` (800, 800, 20 and 25) and `speed.failed_requests` is 0.

`summarize.py` prints `complete=True` only when this holds for every scored run. If a check line in Step 5 or Step 6 prints nothing, or a key shows `complete=False`, read that `server.log` and redo the run from Step 4.

## Measuring accurately

- **Fixed GPU and CPU set.** Step 1 gives the container one GPU and the CPUs of its NUMA node. Keep the same GPU and `CPUSET` for every run you compare. No other job may use that GPU.
- **Fresh server per run.** Every model block starts and stops its own server, so no state carries over from an earlier run.
- **Warmup.** The SeedTTS client sends the full dataset once, unscored, before the timed pass (`--warmup`). This fills CUDA graphs, compile caches and allocator pools. The MOSS-Transcribe-Diarize client has no warmup pass (`--warmup` defaults to 0, and CI uses 0 too).
- **Compile and cold start.** Step 4 turns compile off for the models that do not set it, as CI does. The first run after a new container, a new commit or an empty `/work/torchinductor` can be much slower (see [Caveats](#caveats)). So the `warm` run is never scored.
- **Three runs.** Score `run1`, `run2` and `run3`, and report the mean with the min and max. One run is not enough to tell a change from noise.
- **Noise floor.** To learn the spread of your machine, measure the same commit twice under two names and compare the two summaries.
- **Comparing two commits.** Use the same container, GPU, `CPUSET`, models and datasets. Alternate the commits run by run, for example A, B, B, A, A, B, so slow drift hits both. Set `COMMIT` and `NAME` and rebuild the venv (Step 2) at each switch. Check that `freeze.txt` is the same for both, or the comparison includes a dependency change.
- **Not CI numbers.** CI runs two H100s behind the Rust router (DP 2), pins CPUs through `OMNI_CI_CPUSET`, and warms SeedTTS with only 64 requests. Numbers from this page are comparable only with other runs of this page, never with the CI gates.

## Reference results

These are for orientation only.

- One H100, concurrency 32 (docstring of `benchmarks/eval/benchmark_asr_seedtts.py`): Qwen3-ASR EN reaches 97.9 samples/s with WER 0.0122. Fun-ASR EN reaches about 127.5 samples/s with WER 0.0171. Fun-ASR ZH reaches 167.4 samples/s with WER 0.0135. These numbers come from an older commit, with unknown compile settings, and may not match current code.
- Two H100s in CI (the second pytest attempt inside CI job 113922224972): MOSS-Transcribe-Diarize on `movies800times` gave CER 5.82%, cpCER 13.13%, 784 CER valid samples and 58.167 req/s.
- Run to run variation in CI: the first and second pytest attempts inside CI job 113922224972 gave `movies800times` CER of 5.87% and 5.82%.

## Run the CI tests

The CI tests need two GPUs. Pick two GPUs other than `$GPU` (here 2 and 3), or run `docker rm -f asr-bench` first. On the host, start a second container with the same checkout:

```bash
docker run -d --name asr-ci \
    --gpus '"device=2,3"' \
    -v /dev/shm:/dev/shm \
    -v $HOME/.cache/huggingface:/root/.cache/huggingface \
    -v $HOME/asr-bench:/work \
    -e HF_TOKEN \
    $IMAGE sleep infinity
docker exec -it asr-ci bash
```

Inside it:

```bash
nvidia-smi -L
cd /work/sglang-omni && . .venv/bin/activate
export PYTHONPATH=$PWD
export OMNI_CI_HOME=$HOME/omni_ci
export SGLANG_OMNI_TORCH_COMPILE_DEFAULT=0
export TORCHINDUCTOR_CACHE_DIR=$OMNI_CI_HOME/.torchinductor
export SGLANG_OMNI_ROUTER_BIN=$(bash .github/scripts/prepare_rust_router.sh build)
ASR_CI_MODEL=qwen3 python -m pytest tests/test_model/test_asr_ci_seedtts.py -v -s -x
ASR_CI_MODEL=fun python -m pytest tests/test_model/test_asr_ci_seedtts.py -v -s -x
ASR_CI_MODEL=whisper python -m pytest tests/test_model/test_asr_ci_seedtts.py -v -s -x
python -m pytest tests/test_model/test_asr_ci_multi_speaker.py -v -s -x
```

`nvidia-smi -L` must print exactly two GPUs. `prepare_rust_router.sh` needs network access, `curl` and a C linker to install rustup and build the router under `$HOME/rust-router`. The CI tests load models by repo id at their latest revision, so they do not use the pins in [Variables](#variables). Do not wrap local runs in `run_flaky_pytest.sh`. It reruns a failing test up to three times, so a green result can hide a failure. Remove the container with `docker rm -f asr-ci` when done.

## Caveats

- **Private datasets.** `movies800time`, `AISHELL4` and `googletime` are private and cannot be pinned by the client. Compare the loaded fingerprints and remote heads in `moss_dataset_revisions.txt` before comparing MOSS-Transcribe-Diarize numbers from different machines or dates.
- **Known flaky stage.** On 2026-10-09, 5 sampled CI stage 1 jobs reached only 43.2 to 44.7 req/s on `movies800times` in their first pytest attempt. After the server restarted for the second attempt, they reached 57.0 to 59.1 req/s. The cause is not known. A compile or file cache on disk is the likely suspect. CI sets `TORCHINDUCTOR_CACHE_DIR` to `$OMNI_CI_HOME/.torchinductor`. Outside CI, SGLang-Omni uses `~/.cache/sglang-omni/torchinductor` unless `TORCHINDUCTOR_CACHE_DIR` is set, as Step 4 does.
- **Token budget on movies800times.** Sending a fixed 65536 token budget instead of none lets a clip that loops run far past the duration scaled budget. That changes throughput and latency a lot. Check `config.max_new_tokens` in `transcribe_diarize_speed_results.json` before comparing with CI.
- **Warmup sees the scored audio.** `--warmup` sends the same audio as the timed pass. Any cache keyed on audio content, file path or sample ID would hit on every timed request and make throughput meaningless.
- **Stage 2 is one model per CI run.** A CI run measures only the model it picked. Run all three blocks of Step 5 here to see all of them.

For a profile of where the time goes, see the model profiling skill in `.claude/skills/model-profiling/`.
