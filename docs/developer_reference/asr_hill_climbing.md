# ASR hill climbing runbook

This page is for hill climbing ASR serving in SGLang-Omni. The goal is to make the measured metrics better without regressions. Audio goes in and text comes out. Correctness is WER (CER and cpCER for MOSS-Transcribe-Diarize). Performance is throughput, latency and RTF.

The work must keep both stages of the ASR CI (`.github/workflows/test-asr-ci.yaml`) passing.

Copy every command on this page and run it as written, in order, in `bash`. The only values you set yourself are in [Variables](#variables).

## What is measured

| Stage | When it runs | Model | Data | Concurrency |
|---|---|---|---|---|
| 1 | Every PR | MOSS-Transcribe-Diarize, multiple speakers | `movies800times` (800 samples, run once without and once with streaming), `aishell4_long` (20), `googletime` (25) | 16 |
| 2 | Every PR, one model picked at random | Qwen3-ASR-1.7B, Whisper-large-v3 or Fun-ASR-Nano | SeedTTS EN (1088) and ZH (2020) | 32 |

Stage 1 also has `aishell4_long90`, one 90 minute sample stitched together from `aishell4_long` clips. It only runs through pytest (see [Before opening a PR](#before-opening-a-pr)).

CI runs on two H100s (Rust router with DP 2) and warms up with only 64 requests. This runbook uses one server on one GPU, with the same datasets and concurrency. The SeedTTS client sends the full dataset once as unscored warmup (`--warmup`). The MOSS-Transcribe-Diarize client has no warmup pass. So numbers from this runbook are compared only with your own baseline and last accepted version, never with the CI gates.

There are ten workloads. The script in Step 7 prints results by these keys: `qwen3/asr_en.json`, `qwen3/asr_zh.json`, `whisper/asr_en.json`, `whisper/asr_zh.json`, `fun/asr_en.json`, `fun/asr_zh.json`, `moss_td/movies800times`, `moss_td/movies800times_stream`, `moss_td/aishell4_long` and `moss_td/googletime`.

## Prior work

Read this first. The existing profiles: [#1887](https://github.com/sgl-project/sglang-omni/issues/1887) (Qwen3-ASR), [#1888](https://github.com/sgl-project/sglang-omni/issues/1888) (Whisper), [#1941](https://github.com/sgl-project/sglang-omni/issues/1941) (Fun-ASR), [#1886](https://github.com/sgl-project/sglang-omni/issues/1886) (MOSS-Transcribe-Diarize), [#1324](https://github.com/sgl-project/sglang-omni/issues/1324) (Qwen3-ASR high concurrency roadmap) and the [Qwen3-ASR concurrency profile](qwen3_asr_concurrency_profile.md). The open PRs: [#2609](https://github.com/sgl-project/sglang-omni/pull/2609) (Qwen3-ASR GC tail latency), [#2333](https://github.com/sgl-project/sglang-omni/pull/2333) (MOSS-Transcribe-Diarize decoder compile off by default), [#2446](https://github.com/sgl-project/sglang-omni/pull/2446) (MOSS-Transcribe-Diarize prefill and decode split) and [#2296](https://github.com/sgl-project/sglang-omni/pull/2296) (Fun-ASR context length).

## Variables

There are two blocks. Run the host block on the host before Step 1, and again in every new host shell. Change only the values of `GPU`, `CI_GPUS`, `HF_TOKEN`, `GIT_NAME`, `GIT_EMAIL`, `GITHUB_USER` and `TASK_ISSUE`:

```bash
export GPU=0
export CI_GPUS=0,1
export HF_TOKEN=hf_your_token_here
export GIT_NAME="Your Name"
export GIT_EMAIL=you@example.com
export GITHUB_USER=your-github-user
export TASK_ISSUE=https://github.com/sgl-project/sglang-omni/issues/NUMBER
export IMAGE=hongccc/sglang-omni@sha256:ebe4239e29a764ee3a2806385c061c5fd438a26f01458e503d3822dcba5790df
export CPUSET=$(cat /sys/bus/pci/devices/$(nvidia-smi -i $GPU --query-gpu=pci.bus_id --format=csv,noheader | tail -c 13 | tr A-F a-f)/local_cpulist)
echo $CPUSET
```

`GPU` is the task GPU, and no other job may use it. `CI_GPUS` are two GPUs for the CI tests, one of them `GPU`. `HF_TOKEN` needs access to the three private MOSS-Transcribe-Diarize datasets. `GITHUB_USER` owns your fork of sglang-omni. The maintainer gives you the `TASK_ISSUE` link. To report means to comment there with the output. `IMAGE` is the CI image. `CPUSET` is the CPUs on the NUMA node of `GPU`, such as `0-55,112-167`.

Run the container block inside the container after Step 1, and again after every `docker exec`. Do not change it:

```bash
export BASE_SHA=921ea2c83acbfd7e9247ff38d63d8963b7572b7d
export SEEDTTS_REV=27f4c1adee83b5b29b7c4b375f6b976324bda308
export HF_HUB=/root/.cache/huggingface/hub
export QWEN3_ASR=$HF_HUB/models--Qwen--Qwen3-ASR-1.7B/snapshots/7278e1e70fe206f11671096ffdd38061171dd6e5
export WHISPER=$HF_HUB/models--openai--whisper-large-v3/snapshots/06f233fe06e710322aca913c1bc4249a0d71fce1
export FUN_ASR=$HF_HUB/models--FunAudioLLM--Fun-ASR-Nano-2512-hf/snapshots/7ef6cafdc445b5beff0b730cbeb51952c793ca6e
export MOSS_TD=$HF_HUB/models--OpenMOSS-Team--MOSS-Transcribe-Diarize/snapshots/704aa4a9c304e8520be88901e0d1960158ef5b15
```

`BASE_SHA` is the baseline, main at `921ea2c8`. It stays fixed and does not follow main. The model revisions are the ones in the 2026-10-09 CI logs. Each server loads its model from these snapshot directories, so the revisions cannot change.

Step 8 sets three more variables per pass: `LOOP` (the loop name), `V` (`base`, `ref` or `cand`) and `RUN`.

## Step 1: Start the container (on the host)

The three checkouts live on the host, so the CI container before a PR can use them too:

```bash
mkdir -p $HOME/asr-hc/base $HOME/asr-hc/ref $HOME/asr-hc/cand
docker run -d --name asr-hc \
    --gpus device=$GPU --cpuset-cpus $CPUSET \
    -v /dev/shm:/dev/shm \
    -v $HOME/.cache/huggingface:/root/.cache/huggingface \
    -v $HOME/asr-hc/base:/base -v $HOME/asr-hc/ref:/ref -v $HOME/asr-hc/cand:/cand \
    -e HF_TOKEN -e GIT_NAME -e GIT_EMAIL \
    $IMAGE sleep infinity
docker exec -it asr-hc bash
```

Every command after this runs inside the container. Run the container block of [Variables](#variables). Then `nvidia-smi -L` must print exactly one GPU.

If the `docker exec` shell closes, the variables are lost and a running client stops. Run `docker exec -it asr-hc bash` on the host, the container block, the `LOOP` line and the line of the cut off pass. Then run that pass again from Step 4, which stops any server still running.

## Step 2: Set up the three checkouts

`/base` is the baseline, `/ref` the last accepted version and `/cand` the candidate. The client always runs from the `/base` venv.

```bash
git config --global user.name "$GIT_NAME"
git config --global user.email "$GIT_EMAIL"
git clone https://github.com/sgl-project/sglang-omni.git /base
git -C /base checkout --detach $BASE_SHA
git -C /base worktree add --detach /ref $BASE_SHA
git -C /base worktree add --detach /cand $BASE_SHA
```

Build the venv of one checkout. These lines follow `.github/scripts/prepare_omni_venv.sh`, without the extras and the CosyVoice checkout that other models need:

```bash
cd /base
rm -rf .venv
uv venv --system-site-packages .venv -p /usr/bin/python3.12
echo 'import site; site.addsitedir("/opt/sglang/lib/python3.12/site-packages")' > .venv/lib/python3.12/site-packages/sglang-image.pth
. .venv/bin/activate
python .github/scripts/omni_missing_dependencies.py pyproject.toml | xargs -r -d '\n' -n 1 python -m pip install
python .github/scripts/omni_missing_dependencies.py --overrides pyproject.toml | xargs -r -d '\n' python -m pip install --no-deps
uv pip install --no-deps -e .
python .github/scripts/verify_omni_installed_pins.py .venv/bin/python .
deactivate
```

The `verify_omni_installed_pins.py` line must print `Verified <N> exact dependency pins`. Run the same block two more times. Change only the first line: `cd /ref` the second time and `cd /cand` the third time.

Check that the candidate has the same dependencies as the baseline:

```bash
diff <(/base/.venv/bin/python -m pip freeze --exclude-editable) <(/cand/.venv/bin/python -m pip freeze --exclude-editable) && echo SAME_DEPS
```

It must print `SAME_DEPS` and nothing else. A candidate may not change dependencies. If the check fails for a candidate, reject it. Whenever `/ref` or `/cand` moves to another commit (Step 8), run the venv block for that checkout and the `SAME_DEPS` check again.

## Step 3: Download data and models

Download the SeedTTS dataset and the four models at their pinned revisions:

```bash
cd /base && . .venv/bin/activate
python -m benchmarks.dataset.prepare --dataset seedtts --revision $SEEDTTS_REV
python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3-ASR-1.7B', revision='7278e1e70fe206f11671096ffdd38061171dd6e5')"
python -c "from huggingface_hub import snapshot_download; snapshot_download('openai/whisper-large-v3', revision='06f233fe06e710322aca913c1bc4249a0d71fce1')"
python -c "from huggingface_hub import snapshot_download; snapshot_download('FunAudioLLM/Fun-ASR-Nano-2512-hf', revision='7ef6cafdc445b5beff0b730cbeb51952c793ca6e')"
python -c "from huggingface_hub import snapshot_download; snapshot_download('OpenMOSS-Team/MOSS-Transcribe-Diarize', revision='704aa4a9c304e8520be88901e0d1960158ef5b15')"
ls $QWEN3_ASR/config.json $WHISPER/config.json $FUN_ASR/config.json $MOSS_TD/config.json
```

The `ls` line must print four paths and no error. If it prints `No such file or directory`, stop and report it.

The three MOSS-Transcribe-Diarize datasets are private. Download them once with `HF_TOKEN` set (Step 1 passes it into the container). If a line fails with an access error, stop and report it:

```bash
python -c "import datasets; datasets.load_dataset('zhaochenyang20/movies800time', split='validation')"
python -c "import datasets; datasets.load_dataset('zhaochenyang20/AISHELL4', split='validation')"
python -c "import datasets; datasets.load_dataset('zhaochenyang20/googletime', split='validation')"
```

The client cannot pin these datasets. Record the cached revisions:

```bash
mkdir -p /base/results
echo "movies800time $(cat $HF_HUB/datasets--zhaochenyang20--movies800time/refs/main)" > /base/results/moss_dataset_revisions.txt
echo "AISHELL4 $(cat $HF_HUB/datasets--zhaochenyang20--AISHELL4/refs/main)" >> /base/results/moss_dataset_revisions.txt
echo "googletime $(cat $HF_HUB/datasets--zhaochenyang20--googletime/refs/main)" >> /base/results/moss_dataset_revisions.txt
cat /base/results/moss_dataset_revisions.txt
```

Every pass runs with `HF_HUB_OFFLINE=1` (Step 4), so the datasets stay at these revisions.

## Step 4: Start a pass

A pass runs one version (`V`) once (`RUN`), with a fresh server for each model. Before this block, run the `LOOP` line and the pass line from Step 8, for example `export LOOP=baseline` and `export V=base RUN=warm`. Then run:

```bash
cd /base && . .venv/bin/activate
export HF_HUB_OFFLINE=1 SGLANG_OMNI_STRICT_PORT=1
find /base/results -name server.pid -exec sh -c 'kill -9 -- -$(cat "$1") 2>/dev/null' _ {} \;
timeout 600 bash -c 'until [ $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits) -lt 1024 ]; do sleep 5; done' && echo GPU_FREE
export OUT=/base/results/$LOOP/$V/run$RUN
rm -rf $OUT
mkdir -p $OUT
git -C /$V rev-parse HEAD > $OUT/commit.txt
```

The `find` line stops every server this container started, and the block must print `GPU_FREE`. The block deletes earlier results of the same pass, so a pass is always run whole.

If a check line in Step 5 or Step 6 prints nothing, or `means.py` shows `complete=False` or too few runs, finish the block and run the pass again from this step. If it fails again in a `cand` pass, reject the candidate. In a `base` or `ref` pass, stop and report it with that `server.log`.

## Step 5: Run SeedTTS ASR

Each block starts a server for one model, runs EN and ZH, then stops the server. Run the blocks one at a time, top to bottom.

Which blocks to run:

- In the baseline loop and the noise check loop, run all four blocks (three here, one in Step 6).
- In a candidate loop, list the changed files with `git -C /cand diff --name-only $(git -C /ref rev-parse HEAD) HEAD`. If every path is under one of `sglang_omni/models/qwen3_asr/`, `sglang_omni/models/whisper_asr/`, `sglang_omni/models/fun_asr/` or `sglang_omni/models/moss_transcribe_diarize/`, run only the block of that model. Otherwise run all four blocks. Run the same blocks in every pass of the loop.

In every block, the `curl` line must print `PORT_FREE`, the `timeout 1800` line `SERVER_READY` and the last line `GPU_FREE`. The `setsid` line starts the server in its own process group, so `kill` stops all of it. Each client run prints a line ending with `evaluated=1088/1088` (EN) or `evaluated=2020/2020` (ZH). The client exits with code 0 even when it skipped samples, so Step 7 checks coverage.

### Qwen3-ASR-1.7B

```bash
mkdir -p $OUT/qwen3
curl -sf localhost:8000/health > /dev/null || echo PORT_FREE
setsid bash -c 'echo $$ > "$1"; shift; exec "$@"' _ $OUT/qwen3/server.pid \
    /$V/.venv/bin/sgl-omni serve --model-path $QWEN3_ASR --model-name Qwen/Qwen3-ASR-1.7B --port 8000 \
    > $OUT/qwen3/server.log 2>&1 &
timeout 1800 bash -c 'until curl -sf localhost:8000/health > /dev/null; do sleep 5; done' && echo SERVER_READY
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
    /$V/.venv/bin/sgl-omni serve --model-path $WHISPER --model-name openai/whisper-large-v3 --port 8000 \
    > $OUT/whisper/server.log 2>&1 &
timeout 1800 bash -c 'until curl -sf localhost:8000/health > /dev/null; do sleep 5; done' && echo SERVER_READY
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
    /$V/.venv/bin/sgl-omni serve --model-path $FUN_ASR --model-name FunAudioLLM/Fun-ASR-Nano-2512-hf --port 8000 \
    > $OUT/fun/server.log 2>&1 &
timeout 1800 bash -c 'until curl -sf localhost:8000/health > /dev/null; do sleep 5; done' && echo SERVER_READY
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

The server arguments match the CI test (`tests/test_model/test_asr_ci_multi_speaker.py`). Like CI, one server serves all four datasets. The checks are the same as in Step 5. Each client line must also print `CLIENT_OK`. The client exits with code 0, and so prints `CLIENT_OK`, only when every request succeeded. Otherwise it prints an error or an `Evaluation failed` line.

```bash
mkdir -p $OUT/moss_td
curl -sf localhost:8000/health > /dev/null || echo PORT_FREE
setsid bash -c 'echo $$ > "$1"; shift; exec "$@"' _ $OUT/moss_td/server.pid \
    /$V/.venv/bin/sgl-omni serve --model-path $MOSS_TD --model-name OpenMOSS-Team/MOSS-Transcribe-Diarize --port 8000 \
    --asr.engine.max_running_requests 16 --asr.engine.cuda_graph_max_bs 16 --mem-fraction-static 0.8 \
    > $OUT/moss_td/server.log 2>&1 &
timeout 1800 bash -c 'until curl -sf localhost:8000/health > /dev/null; do sleep 5; done' && echo SERVER_READY
python -m benchmarks.eval.benchmark_asr_transcribe_diarize --use-existing-server --port 8000 \
    --dataset movies800times --max-concurrency 16 --output-dir $OUT/moss_td/movies800times && echo CLIENT_OK
python -m benchmarks.eval.benchmark_asr_transcribe_diarize --use-existing-server --port 8000 \
    --dataset movies800times --stream --max-concurrency 16 --output-dir $OUT/moss_td/movies800times_stream && echo CLIENT_OK
python -m benchmarks.eval.benchmark_asr_transcribe_diarize --use-existing-server --port 8000 \
    --dataset aishell4_long --max-concurrency 16 --output-dir $OUT/moss_td/aishell4_long && echo CLIENT_OK
python -m benchmarks.eval.benchmark_asr_transcribe_diarize --use-existing-server --port 8000 \
    --dataset googletime --max-concurrency 16 --output-dir $OUT/moss_td/googletime && echo CLIENT_OK
kill -- -$(cat $OUT/moss_td/server.pid)
timeout 600 bash -c 'until [ $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits) -lt 1024 ]; do sleep 5; done' && echo GPU_FREE
```

## Step 7: Read the results

### Where each metric is

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
| Valid sample IDs | `per_sample[].id` where `cer_valid` and `cp_cer_valid` are both true | IDs |
| Speaker timestamp DER | `diarization_metrics.speaker_timestamp_der`, `diarization_metrics.speaker_timestamp_der_skipped_parse_error` | fraction, count |
| Failed requests | `speed.failed_requests` | request count |
| Throughput | `speed.throughput_qps` | requests per second |
| Latency mean, p95 | `speed.latency_mean_s`, `speed.latency_p95_s` | seconds |
| RTF mean | `speed.rtf_mean` | ratio |

The printed client output uses `diarization_metrics_percent`, the same values times 100. The `speed` values are stored rounded: throughput and latency to 3 decimals, RTF to 4. When every request of a run fails, `speed` has no throughput, latency or RTF fields.

### Save the means script

Save this script once. For one version directory it reads every scored run (`run<number>`, never `runwarm`), checks coverage and prints each mean. Given a second directory, it also prints its mean (`old`) and the `ratio` new/old.

```bash
cat > /base/results/means.py <<'EOF'
import glob, json, os, statistics, sys
SEED = ("corpus_wer", "throughput_samples_per_s", "latency_mean_s", "latency_p95_s", "rtf_mean")
MOSS = ("cer", "cp_cer", "cer_valid_samples", "cp_cer_valid_samples")
SPEED = ("throughput_qps", "latency_mean_s", "latency_p95_s", "rtf_mean")
def load(vdir):
    out = {}
    for run in sorted(glob.glob(f"{vdir}/run[0-9]*")):
        for path in glob.glob(f"{run}/*/asr_*.json"):
            r = json.load(open(path))["results"][0]
            ok = r["evaluated"] == r["total"] and r["skipped"] == 0
            out.setdefault(os.path.relpath(path, run), []).append((ok, {k: r[k]["mean"] for k in SEED}, set(), None))
        for path in glob.glob(f"{run}/moss_td/*/transcribe_diarize_results.json"):
            p = json.load(open(path))
            s = json.load(open(path.replace("_results.json", "_speed_results.json")))["speed"]
            d = p["diarization_metrics"]
            ok = p["summary"]["evaluated"] == p["summary"]["total_samples"] and s.get("failed_requests") == 0
            m = {k: d[k] for k in MOSS} | {k: s.get(k) for k in SPEED}
            ids = {x["id"] for x in p["per_sample"] if x["cer_valid"] and x["cp_cer_valid"]}
            der = (d.get("speaker_timestamp_der"), d.get("speaker_timestamp_der_skipped_parse_error"))
            out.setdefault("moss_td/" + os.path.basename(os.path.dirname(path)), []).append((ok, m, ids, der))
    return out
def summary(rows):
    complete = all(ok and None not in m.values() for ok, m, _, _ in rows)
    means = {k: statistics.mean(r[1][k] for r in rows) for k in rows[0][1]} if complete else {}
    return complete, means, set.intersection(*(r[2] for r in rows))
new = load(sys.argv[1])
old = load(sys.argv[2]) if len(sys.argv) > 2 else {}
for key, rows in sorted(new.items()):
    complete, means, ids = summary(rows)
    print(f"{key}\n  new runs={len(rows)} complete={complete}")
    omeans = {}
    if key in old:
        ocomplete, omeans, oids = summary(old[key])
        print(f"  old runs={len(old[key])} complete={ocomplete}")
        if key.startswith("moss_td/"):
            print(f"  valid IDs of old missing in new: {len(oids - ids)} {sorted(oids - ids)[:10]}")
    for k, v in means.items():
        line = f"  {k:26} new {v:.6f}"
        if k in omeans:
            line += f"  old {omeans[k]:.6f}"
            if omeans[k]:
                line += f"  ratio {v / omeans[k]:.4f}"
        print(line)
    if key.startswith("moss_td/"):
        print(f"  speaker_timestamp_der, parse errors per run: {[r[3] for r in rows]}")
EOF
```

### Reference numbers

These show a healthy run. They are never gates. On one H100 at concurrency 32 (docstring of `benchmarks/eval/benchmark_asr_seedtts.py`), Qwen3-ASR EN reaches 97.9 samples/s with WER 0.0122, and Fun-ASR EN about 127.5 samples/s with WER 0.0171. On two H100s in CI (the second pytest attempt inside CI job 113922224972), MOSS-Transcribe-Diarize on `movies800times` gave CER 5.82%, cpCER 13.13%, 784 CER valid samples and 58.167 req/s.

## Step 8: Repeat and compare

A loop is a fixed list of passes. Run the `LOOP` line of the loop. Then, for each pass in its table, run the pass line, Step 4, and the blocks of Step 5 and Step 6. The `warm` passes are never scored (see [Cold start](#cold-start)).

### Baseline loop

Run once at task start:

```bash
export LOOP=baseline
```

| Pass | Line |
|---|---|
| 1 | `export V=base RUN=warm` |
| 2 | `export V=base RUN=1` |
| 3 | `export V=base RUN=2` |
| 4 | `export V=base RUN=3` |

Then print the baseline:

```bash
python /base/results/means.py /base/results/baseline/base
```

Every workload key must show `runs=3` and `complete=True`.

### Paired loop

A paired loop measures the candidate and the last accepted version in turn. RUN 2 runs the candidate first:

| Pass | Line |
|---|---|
| 1 | `export V=ref RUN=warm` |
| 2 | `export V=cand RUN=warm` |
| 3 | `export V=ref RUN=1` |
| 4 | `export V=cand RUN=1` |
| 5 | `export V=cand RUN=2` |
| 6 | `export V=ref RUN=2` |
| 7 | `export V=ref RUN=3` |
| 8 | `export V=cand RUN=3` |

### Noise check loop

Run this once, before the first candidate, while `/ref` and `/cand` are both at `BASE_SHA`. Run the paired loop with this `LOOP` line:

```bash
export LOOP=noise
```

Then run:

```bash
python /base/results/means.py /base/results/noise/cand /base/results/noise/ref
```

Both are the same code, so every throughput, latency and RTF `ratio` must be between 0.97 and 1.03. If one is outside, the machine is too noisy. Stop and report the output.

### Candidate loop

Edit the code in `/cand` and commit it there. Name the target workload key in a `Target-workload` trailer, chosen before measuring and never changed after. For example:

```bash
git -C /cand add -A
git -C /cand commit -m "Qwen3-ASR: describe the change here" -m "Target-workload: qwen3/asr_en.json"
git -C /cand log -1 --format='%(trailers:key=Target-workload,valueonly)'
```

The last line must print one workload key from [What is measured](#what-is-measured). Run the venv block of Step 2 with `cd /cand` and the `SAME_DEPS` check. Then run the paired loop with this `LOOP` line:

```bash
export LOOP=$(git -C /cand rev-parse --short HEAD)
```

Then compare the candidate with the last accepted version and with the baseline:

```bash
python /base/results/means.py /base/results/$LOOP/cand /base/results/$LOOP/ref
python /base/results/means.py /base/results/$LOOP/cand /base/results/baseline/base
```

Apply the [acceptance rules](#acceptance-rules) to these two outputs. Decide by the printed `new`, `old` and `ratio` values, with no rerun.

### Accept or reject

On accept, the candidate becomes the last accepted version:

```bash
git -C /ref checkout --detach $(git -C /cand rev-parse HEAD)
```

Then run the venv block of Step 2 with `cd /ref`. The next candidate starts from this commit in `/cand`.

On reject, put `/cand` back at the last accepted version:

```bash
git -C /cand checkout --detach $(git -C /ref rev-parse HEAD)
```

Then run the venv block of Step 2 with `cd /cand` and the `SAME_DEPS` check.

## Acceptance rules

A candidate is accepted only when every rule holds for every workload that ran in its loop.

**Fixed setup.** Once the loops start, the revisions of the models, datasets, scoring code and dependencies never change, and the task keeps one GPU and one CPU set. Every candidate runs its full paired loop. Never reuse stored numbers for the last accepted version. If a client command fails on the baseline, stop and report it.

**Coverage.** Every version in both outputs shows `runs=3` and `complete=True`, which means the full set with zero failed requests in every scored run.

**Correctness**, read from the output against the baseline only:

- `corpus_wer` new is at most old + 0.0002. Across 5 CI jobs on the same model, EN WER differed by at most 0.0001.
- `cer` and `cp_cer` new are at most old + 0.001 (fractions, so 0.1 percentage points). The first and second pytest attempts inside CI job 113922224972 gave `movies800times` CER of 5.87% and 5.82%.
- `cer_valid_samples` and `cp_cer_valid_samples` have a ratio of at least 1.0000.
- `valid IDs of old missing in new` is `0`. Every sample ID valid in the baseline must be valid in the candidate.
- Report the `speaker_timestamp_der` line in the PR. It checks that diarization still works.

**Performance**, read from both outputs:

| Workload | Rule |
|---|---|
| Every workload that ran | `throughput_samples_per_s` and `throughput_qps` ratio at least 0.97. `latency_mean_s`, `latency_p95_s` and `rtf_mean` ratio at most 1.03. Against both the last accepted version and the baseline |
| The target workload | Throughput ratio against the last accepted version at least 1.02 |

Comparing against both versions stops small losses from adding up.

## What does not count

These are not optimizations:

- Changing the client, dataset, concurrency, `language`, `response_format` or warmup.
- Changing the input the model sees (resampling, feature extraction, cutting audio).
- Changing decoding defaults (temperature, maximum length, EOS) so outputs get shorter.
- Lower precision for weights, activations or the KV cache, or turning on TF32.
- Reusing any result across requests based on audio content, file path or sample ID (transcripts, audio features, encoder outputs, KV prefixes). In the SeedTTS client, `--warmup` sends the same audio as the scored pass, so such a cache would hit every time.
- Special branches for SeedTTS or `movies800times`.
- Server startup warmup that reads dataset files. It may only use synthetic input.
- Using more resources, such as a second GPU, MPS, a different CPU set, or more threads than the CPUs in `CPUSET`.
- Copying changes from an existing PR.

## Cold start

On 2026-10-09, 5 sampled stage 1 jobs reached only 43.2 to 44.7 req/s on `movies800times` in their first pytest attempt, below the 51.877 gate. After the server restarted for the second attempt, they reached 57.0 to 59.1 req/s and passed. The cause is not known. A compile or file cache on disk is the likely suspect. CI sets `TORCHINDUCTOR_CACHE_DIR` to `$OMNI_CI_HOME/.torchinductor`. Here SGLang-Omni uses its default, `~/.cache/sglang-omni/torchinductor`. This is why the `warm` passes are never scored.

## Before opening a PR

**Where code goes.** Put new code in the matching model directory under `sglang_omni/models` where possible. Changes to a shared hot path get strict CI review.

**Default on.** An optimization must be on by default.

**Acceptance tests stay unchanged.** A PR does not change the test commands, the evaluation and metric code under `benchmarks/`, or the CI under `.github/` and `tests/test_model/`. These are the acceptance criteria. You may add or change matching unit tests under `tests/unit_test/`, but never loosen a check.

**Reproducible.** Every PR is reproduced before it is accepted.

**Make the PR branch.** A PR carries the last accepted version, rebased onto current main. Make it between loops. In the task container:

```bash
git -C /base fetch origin
git -C /cand switch -c asr-hc-$(git -C /ref rev-parse --short HEAD)
git -C /cand rebase origin/main
```

If the rebase stops with a conflict, run `git -C /cand rebase --abort` and report it. Then run the venv block of Step 2 with `cd /cand`. Skip the `SAME_DEPS` check here, since main may have changed dependencies.

**ASR CI.** A PR must pass the ASR CI. The tests need two GPUs. On the host, stop the task container and start a CI container on `CI_GPUS`:

```bash
docker stop asr-hc
docker run -d --name asr-ci-pr \
    --gpus "\"device=$CI_GPUS\"" \
    -v /dev/shm:/dev/shm \
    -v $HOME/.cache/huggingface:/root/.cache/huggingface \
    -v $HOME/asr-hc/base:/base -v $HOME/asr-hc/cand:/cand \
    -e HF_TOKEN \
    $IMAGE sleep infinity
docker exec -it asr-ci-pr bash
```

Inside the CI container, run the tests on the PR branch in `/cand`:

```bash
nvidia-smi -L
cd /cand && . .venv/bin/activate
export PYTHONPATH=$PWD
export OMNI_CI_HOME=$HOME/omni_ci
export SGLANG_OMNI_ROUTER_BIN=$(bash .github/scripts/prepare_rust_router.sh build)
ASR_CI_MODEL=qwen3 python -m pytest tests/test_model/test_asr_ci_seedtts.py -v -s -x
ASR_CI_MODEL=fun python -m pytest tests/test_model/test_asr_ci_seedtts.py -v -s -x
ASR_CI_MODEL=whisper python -m pytest tests/test_model/test_asr_ci_seedtts.py -v -s -x
python -m pytest tests/test_model/test_asr_ci_multi_speaker.py -v -s -x
```

`nvidia-smi -L` prints exactly two GPUs. `prepare_rust_router.sh` needs network access to install rustup and build the router under `$HOME/rust-router`. All four pytest lines must pass. If one fails, do not open the PR, and report it.

Do not wrap local runs in `run_flaky_pytest.sh`. It runs a failing test up to three times, so a green result can hide a failure.

When the tests pass, remove the CI container and push the branch from the host:

```bash
docker rm -f asr-ci-pr
git -C $HOME/asr-hc/base push https://github.com/$GITHUB_USER/sglang-omni.git $(cut -d' ' -f2 $HOME/asr-hc/base/.git/worktrees/cand/HEAD)
```

Open a PR from that branch of your fork against `main` of sgl-project/sglang-omni. Fill in every section of `.github/pull_request_template.md`, with the two `means.py` outputs of each accepted loop. Ask for one label. If every changed path is under `sglang_omni/models/fun_asr/`, ask for `run-fun-asr`. If every changed path is under `sglang_omni/models/whisper_asr/`, ask for `run-whisper-asr`. Otherwise ask for `run-qwen3-asr`. A maintainer adds the label.

Then go back to the task container:

```bash
docker start asr-hc
docker exec -it asr-hc bash
```

Run the container block of [Variables](#variables) again. Put `/cand` back at the last accepted version:

```bash
git -C /cand checkout --detach $(git -C /ref rev-parse HEAD)
```

Then run the venv block of Step 2 with `cd /cand` and the `SAME_DEPS` check.
