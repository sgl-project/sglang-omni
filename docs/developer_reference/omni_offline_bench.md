# MiniCPM-o offline benchmark runbook

This page measures MiniCPM-o 4.5 offline serving in SGLang-Omni on one GPU. It also shows how to run the MiniCPM-o CI stages on a GPU pair. Full duplex serving is covered in the [Full-Duplex-Bench v1.5 runbook](full_duplex_bench.md).

Many commands end with a check that prints a fixed word in capitals, such as `READY` or `GPU_FREE`. If the word does not print, the step did not finish and its numbers are not usable.

## What is Measured

Two workloads, both on one GPU at concurrency 16:

| Workload | Input and output | Samples | Server parameters |
|---|---|---|---|
| Video-AMME Talker | Video plus a spoken question in, text plus speech out | The first 10 questions of `Video_AMME_ci` | As in CI stage 10 |
| seed-tts voice clone | Text in, cloned speech out | The 1088 sample EN full set | As in CI stage 2 |

Every metric is a JSON field. Paths are under the run directory `$OUT`.

| File | Fields |
|---|---|
| `videoamme/videoamme_results.json` | Accuracy `summary.accuracy` (0 to 1), failed requests `summary.failed`, prompt tokens `per_sample[].prompt_tokens` by `sample_id`, and the speed fields below under `speed.` |
| `seedtts/speed_results.json` | Failed requests `summary.failed_requests`, prompt tokens `per_request[].prompt_tokens` by `id`, output tokens `per_request[].completion_tokens`, and the speed fields below under `summary.` |
| Speed fields | Throughput `throughput_qps` (req/s), latency `latency_mean_s` and `latency_p95_s` (s), `rtf_mean`, audio duration mean `audio_duration_mean_s` (s), audio throughput `audio_throughput_s_per_s` (seconds of audio per second) |
| `videoamme/wer_summary.json` | The WER fields below, at top level |
| `seedtts/wer_results.json` | The WER fields below, under `summary.` |
| WER fields | WER `wer_corpus` (a fraction, 0.005 is 0.5 points), samples above 50% WER `n_above_50_pct_wer`, WER corpus without those samples `wer_below_50_corpus` (the CI metric), unscored samples `skipped` |
| `seedtts/similarity_results.json` | SIM `summary.speaker_similarity_mean` (cosine × 100), `summary.skipped` |
| `seedtts/utmos_results.json` | UTMOS `summary.utmos_mean` (1 to 5), `summary.skipped` |

Higher is better for Accuracy, throughput, SIM and UTMOS. Lower is better for latency, RTF and WER. Audio throughput shows whether a req/s change comes with shorter or longer speech.

WER uses Qwen3-ASR-1.7B served by SGLang-Omni, as in CI. SIM and UTMOS run without a server.

## Setup

One container holds a GPU pair. Single GPU measurements use the first GPU, pinned to its CPU group. The CI stages use both GPUs. Never run the two at the same time. Downloads, caches and results live in `$BENCH_DIR` on the host.

The host needs `python3`, `lscpu` and `nvidia-smi`. The CI stages build the router on first use, which needs internet access to rustup and crates and takes several minutes.

### Start the container

On the host:

```bash
export BENCH_DIR=/data/omni-bench
export GPU_A=0 GPU_B=1
mkdir -p "$BENCH_DIR/hf" "$BENCH_DIR/omni-cache" "$BENCH_DIR/results" "$BENCH_DIR/sglang-omni"
```

Make the CPU groups. Each GPU gets 14 cores on its NUMA node with their SMT siblings, skipping cores 0 and 1. `BENCH_CPUS` is the group of `GPU_A`. `CI_CPUS` covers both GPUs. It must print `CPUS_OK`.

```bash
python3 - "$GPU_A" "$GPU_B" > "$BENCH_DIR/cpus.env" <<'PY'
import subprocess, sys
from collections import defaultdict
run = lambda cmd: subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
siblings, core_node, free, groups = defaultdict(list), {}, defaultdict(list), {}
for line in run(["lscpu", "-p=CPU,CORE,NODE"]).splitlines():
    if not line.startswith("#"):
        cpu, core, node = line.split(",")
        siblings[int(core)].append(int(cpu))
        core_node[int(core)] = int(node or 0)
for core in sorted(core_node):
    if core not in (0, 1):
        free[core_node[core]].append(core)
for line in run(["nvidia-smi", "--query-gpu=index,pci.bus_id", "--format=csv,noheader"]).splitlines():
    index, bus = (x.strip() for x in line.split(","))
    node = max(int(open(f"/sys/bus/pci/devices/{bus[-12:].lower()}/numa_node").read()), 0)
    if len(free[node]) >= 14:
        group, free[node] = free[node][:14], free[node][14:]
        groups[index] = sorted(c for core in group for c in siblings[core])
a, b = groups[sys.argv[1]], groups[sys.argv[2]]
print("BENCH_CPUS=" + ",".join(map(str, a)))
print("CI_CPUS=" + ",".join(map(str, sorted(a + b))))
PY
cat "$BENCH_DIR/cpus.env" && . "$BENCH_DIR/cpus.env" && test -n "$BENCH_CPUS" && echo CPUS_OK
```

A `KeyError` means a GPU of the pair gets no group. Set `GPU_A=2 GPU_B=3` (then `4 5`, then `6 7`) and run it again. Compare with `lscpu -p=CPU,CORE,NODE` and `nvidia-smi topo -m` if no pair works.

Start the container with the CI image:

```bash
docker run -d --init --ipc host --cap-add=SYS_PTRACE \
    --gpus "\"device=$GPU_A,$GPU_B\"" --cpuset-cpus "$CI_CPUS" --env-file "$BENCH_DIR/cpus.env" \
    -v "$BENCH_DIR/hf:/root/.cache/huggingface" \
    -v "$BENCH_DIR/omni-cache:/root/.cache/sglang-omni" \
    -v "$BENCH_DIR/results:/results" \
    -v "$BENCH_DIR/sglang-omni:/sglang-omni" \
    --name omni-bench \
    hongccc/sglang-omni@sha256:ebe4239e29a764ee3a2806385c061c5fd438a26f01458e503d3822dcba5790df \
    sleep infinity
```

Open a shell with this command whenever you need one. After a host reboot, run `docker start omni-bench` first. All later commands run inside the container.

```bash
docker exec -it omni-bench bash
```

### Check out and build the venv

Set `COMMIT` to the commit to measure. The default below measures the fetched main branch. Build the venv with the CI script, which ends with `Fresh environment ready`. The script deletes and rebuilds `OMNI_CI_HOME`.

```bash
git clone https://github.com/sgl-project/sglang-omni.git /sglang-omni
export COMMIT=origin/main
git -C /sglang-omni checkout --detach "$COMMIT"
cd /sglang-omni && OMNI_CI_HOME=/root/omni_ci bash .github/scripts/prepare_omni_venv.sh omni
```

After every checkout, this must print `VENV_OK`. If it does not, run `prepare_omni_venv.sh` again. Source changes need no rebuild, because the package is installed editable.

```bash
cd /sglang-omni && source .github/scripts/omni_ci_deps_hash.sh && test "$(omni_ci_deps_hash)" = "$(cat /root/omni_ci/.deps-hash)" && echo VENV_OK
```

### Download models and data

```bash
cd /sglang-omni && . omni/bin/activate
python -c "from huggingface_hub import snapshot_download as d; d('openbmb/MiniCPM-o-4_5', revision='503e754207c94da6bb26850b4469f367c9ea3582')"
python -c "from huggingface_hub import snapshot_download as d; d('Qwen/Qwen3-ASR-1.7B', revision='7278e1e70fe206f11671096ffdd38061171dd6e5')"
python -c "from huggingface_hub import snapshot_download as d; d('zhaochenyang20/Video_AMME_ci', repo_type='dataset', revision='7a37507f1b53416b9cb6641378e5a098ea535ab8')"
python -c "from huggingface_hub import snapshot_download as d; d('zhaochenyang20/Video_MME_ci', repo_type='dataset', revision='833bd815c628ff277911bea3b1563545b21d5e27')"
python -m benchmarks.dataset.prepare --dataset seedtts
python -m benchmarks.dataset.prepare --dataset seedtts-50
python -m benchmarks.dataset.prepare --dataset mmmu-ci-50
python -m benchmarks.dataset.prepare --dataset mmsu-ci-2000
python -m benchmarks.metrics.speaker_similarity_assets --warm-cache
python -m benchmarks.metrics.utmos --warm-cache
```

`Video_MME_ci` holds the videos that Video-AMME points to. `seedtts-50`, `mmmu-ci-50` and `mmsu-ci-2000` are only for the CI stages. A download by commit does not set `refs/main`, so set those references for offline loading:

```bash
H=/root/.cache/huggingface/hub
mkdir -p $H/models--openbmb--MiniCPM-o-4_5/refs $H/models--Qwen--Qwen3-ASR-1.7B/refs $H/datasets--zhaochenyang20--Video_AMME_ci/refs $H/datasets--zhaochenyang20--Video_MME_ci/refs
echo -n 503e754207c94da6bb26850b4469f367c9ea3582 > $H/models--openbmb--MiniCPM-o-4_5/refs/main
echo -n 7278e1e70fe206f11671096ffdd38061171dd6e5 > $H/models--Qwen--Qwen3-ASR-1.7B/refs/main
echo -n 7a37507f1b53416b9cb6641378e5a098ea535ab8 > $H/datasets--zhaochenyang20--Video_AMME_ci/refs/main
echo -n 833bd815c628ff277911bea3b1563545b21d5e27 > $H/datasets--zhaochenyang20--Video_MME_ci/refs/main
```

Verify that every pinned snapshot loads offline at its revision. It must print `REVISIONS_OK`. seed-tts loads at `27f4c1adee83b5b29b7c4b375f6b976324bda308`, pinned as `SEEDTTS_DATASET_REVISION` in `benchmarks/dataset/prepare.py`.

```bash
cd /sglang-omni && . omni/bin/activate
HF_HUB_OFFLINE=1 python - <<'PY'
from huggingface_hub import constants, snapshot_download
from benchmarks.dataset.seedtts import load_seedtts_samples
assert constants.HF_HUB_CACHE == "/root/.cache/huggingface/hub", constants.HF_HUB_CACHE
pins = [("openbmb/MiniCPM-o-4_5", "model", "503e754207c94da6bb26850b4469f367c9ea3582"),
        ("Qwen/Qwen3-ASR-1.7B", "model", "7278e1e70fe206f11671096ffdd38061171dd6e5"),
        ("zhaochenyang20/Video_AMME_ci", "dataset", "7a37507f1b53416b9cb6641378e5a098ea535ab8"),
        ("zhaochenyang20/Video_MME_ci", "dataset", "833bd815c628ff277911bea3b1563545b21d5e27")]
for repo, kind, sha in pins:
    path = snapshot_download(repo, repo_type=kind)
    assert path.endswith(sha), (repo, path)
    print(repo, sha)
n = len(load_seedtts_samples("zhaochenyang20/seed-tts-eval-arrow", None, split="en"))
assert n == 1088, n
print("REVISIONS_OK")
PY
```

## Run the benchmark

### The shell block

Run it at the start of every shell used for single GPU measurement.

```bash
taskset -pc "$BENCH_CPUS" $$
cd /sglang-omni && . omni/bin/activate
unset SGLANG_OMNI_TORCH_COMPILE_DEFAULT PYTORCH_ALLOC_CONF
export CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 SGLANG_OMNI_STRICT_PORT=1 SGLANG_OMNI_STARTUP_TIMEOUT=1800
```

Torch compile stays on (the default) and `PYTORCH_ALLOC_CONF` stays unset. A cold start with compile needs more than the default 600 second startup timeout. Stay in `/sglang-omni`, because `benchmarks` is imported from the working directory.

### One run

A run measures the checkout once on both workloads, each with a fresh server, then scores it. Start with `run0`, then repeat this section with `run1`, `run2` and `run3`, as explained in [Measuring accurately](#measuring-accurately). For `run0`, stop after the seed-tts server is stopped.

```bash
export OUT=/results/$(git rev-parse --short HEAD)/run0
```

It must print `RUN_READY`. `OUT` must be new and port 8000 must be free.

```bash
test ! -e "$OUT" && ! curl -s -o /dev/null localhost:8000/health && mkdir -p "$OUT/videoamme" "$OUT/seedtts" && git rev-parse HEAD > "$OUT/commit.txt" && git status --porcelain > "$OUT/dirty.txt" && echo RUN_READY
```

#### Video-AMME Talker

Start the server. It must print `READY`. `/tmp/omni_server.pid` holds the server's process group, so the stop command stops all of it.

```bash
setsid bash -c 'echo $$ > "$0"; exec "$@"' /tmp/omni_server.pid \
    sgl-omni serve \
    --model-path openbmb/MiniCPM-o-4_5 --model-name minicpmo \
    --thinker.engine.mem_fraction_static 0.55 --talker.engine.mem_fraction_static 0.15 \
    --thinker.factory.max_seq_len 32768 \
    --port 8000 > "$OUT/videoamme.log" 2>&1 &
timeout 1800 bash -c 'until curl -sf localhost:8000/health > /dev/null; do sleep 5; done' && echo READY || echo NOT_READY
```

Run the client. As in CI, WER is computed later from the ASR server.

```bash
python - --model minicpmo --port 8000 --repo-id zhaochenyang20/Video_AMME_ci --max-samples 10 \
    --video-fps 2 --video-max-frames 128 --video-max-pixels 401408 --max-tokens 256 --temperature 0 \
    --timeout-s 500 --max-concurrency 16 --enable-audio --output-dir "$OUT/videoamme" <<'PY'
import argparse, asyncio
from benchmarks.eval import benchmark_omni_videoamme as m
parser = argparse.ArgumentParser()
m.add_video_eval_args(parser, repo_help="")
cfg = m.video_eval_config_from_args(parser.parse_args())
asyncio.run(m.run_videoamme_eval(cfg, compute_wer=False))
PY
```

Stop the server. It must print `GPU_FREE` (memory on the GPU below 1 GiB).

```bash
kill -- -"$(cat /tmp/omni_server.pid)"
timeout 600 bash -c 'until [ "$(nvidia-smi -i 0 --query-gpu=memory.used --format=csv,noheader,nounits)" -lt 1024 ]; do sleep 5; done' && echo GPU_FREE || echo GPU_BUSY
```

#### seed-tts voice clone

Start the server. It must print `READY`.

```bash
setsid bash -c 'echo $$ > "$0"; exec "$@"' /tmp/omni_server.pid \
    sgl-omni serve \
    --model-path openbmb/MiniCPM-o-4_5 --model-name minicpmo \
    --thinker.engine.mem_fraction_static 0.55 --talker.engine.mem_fraction_static 0.15 \
    --thinker.factory.max_seq_len 8192 \
    --port 8000 > "$OUT/seedtts.log" 2>&1 &
timeout 1800 bash -c 'until curl -sf localhost:8000/health > /dev/null; do sleep 5; done' && echo READY || echo NOT_READY
```

Run the client:

```bash
python -m benchmarks.eval.benchmark_omni_seedtts --generate-only \
    --meta zhaochenyang20/seed-tts-eval-arrow --model minicpmo --port 8000 \
    --voice-clone --reference-audio-field audio.ref_audio --max-concurrency 16 \
    --output-dir "$OUT/seedtts"
```

Stop the server. It must print `GPU_FREE`.

```bash
kill -- -"$(cat /tmp/omni_server.pid)"
timeout 600 bash -c 'until [ "$(nvidia-smi -i 0 --query-gpu=memory.used --format=csv,noheader,nounits)" -lt 1024 ]; do sleep 5; done' && echo GPU_FREE || echo GPU_BUSY
```

#### Score

Start Qwen3-ASR. It must print `READY`.

```bash
setsid bash -c 'echo $$ > "$0"; exec "$@"' /tmp/omni_server.pid \
    sgl-omni serve \
    --model-path Qwen/Qwen3-ASR-1.7B --model-name Qwen/Qwen3-ASR-1.7B \
    --port 8000 > "$OUT/asr.log" 2>&1 &
timeout 1800 bash -c 'until curl -sf localhost:8000/health > /dev/null; do sleep 5; done' && echo READY || echo NOT_READY
```

seed-tts WER:

```bash
python -m benchmarks.eval.benchmark_omni_seedtts --transcribe-only \
    --meta zhaochenyang20/seed-tts-eval-arrow --model minicpmo --port 8000 \
    --asr-concurrency 4 --output-dir "$OUT/seedtts"
```

Video-AMME Talker WER, with the function CI stage 10 calls:

```bash
python - "$OUT/videoamme" <<'PY'
import json, sys
from benchmarks.tasks.asr import compute_text_audio_consistency_from_records
d = sys.argv[1]
records = json.load(open(f"{d}/videoamme_results.json"))["per_sample"]
wer = compute_text_audio_consistency_from_records(records, "en", "cuda:0", asr_router_port=8000, asr_concurrency=4)
json.dump(wer["summary"], open(f"{d}/wer_summary.json", "w"), indent=2)
PY
```

Stop the ASR server. It must print `GPU_FREE`.

```bash
kill -- -"$(cat /tmp/omni_server.pid)"
timeout 600 bash -c 'until [ "$(nvidia-smi -i 0 --query-gpu=memory.used --format=csv,noheader,nounits)" -lt 1024 ]; do sleep 5; done' && echo GPU_FREE || echo GPU_BUSY
```

SIM and UTMOS:

```bash
python -m benchmarks.eval.benchmark_omni_seedtts --similarity-only \
    --meta zhaochenyang20/seed-tts-eval-arrow --model minicpmo --output-dir "$OUT/seedtts"
python -m benchmarks.eval.benchmark_omni_seedtts --utmos-only \
    --meta zhaochenyang20/seed-tts-eval-arrow --model minicpmo --output-dir "$OUT/seedtts"
```

### Check coverage and collect the metrics

This prints every field from [What is Measured](#what-is-measured), checks the speech files and writes `$OUT/metrics.json`. The last line is `RUN_COMPLETE` when every request and every scoring stage covered all 10 Video-AMME and 1088 seed-tts samples, with unique IDs, one WAV per sample, 0 failed requests, 0 skipped samples and no empty or NaN audio. A missing JSON field raises an error.

```bash
python - "$OUT" <<'PY'
import glob, json, sys
import numpy as np
import soundfile as sf
d = sys.argv[1]
J = lambda p: json.load(open(f"{d}/{p}"))
v, vw, s = J("videoamme/videoamme_results.json"), J("videoamme/wer_summary.json"), J("seedtts/speed_results.json")
w, sim, u = (J(f"seedtts/{n}_results.json")["summary"] for n in ("wer", "similarity", "utmos"))
m = {"videoamme.accuracy": v["summary"]["accuracy"], "videoamme.failed": v["summary"]["failed"],
     "seedtts.failed": s["summary"]["failed_requests"], "seedtts.sim": sim["speaker_similarity_mean"],
     "seedtts.sim_skipped": sim["skipped"], "seedtts.utmos": u["utmos_mean"], "seedtts.utmos_skipped": u["skipped"],
     "seedtts.completion_tokens_sum": sum(r["completion_tokens"] or 0 for r in s["per_request"])}
for k in ["throughput_qps", "latency_mean_s", "latency_p95_s", "rtf_mean", "audio_duration_mean_s", "audio_throughput_s_per_s"]:
    m[f"videoamme.{k}"], m[f"seedtts.{k}"] = v["speed"][k], s["summary"][k]
for k in ["wer_corpus", "wer_below_50_corpus", "n_above_50_pct_wer", "skipped"]:
    m[f"videoamme.wer.{k}"], m[f"seedtts.wer.{k}"] = vw[k], w[k]
audio = [sf.read(p) for p in sorted(glob.glob(f"{d}/*/audio/*.wav"))]
rates = sorted({int(r) for _, r in audio})
m["audio.files"] = len(audio)
m["audio.empty_or_nan"] = sum(int(a.size == 0 or bool(np.isnan(a).any())) for a, _ in audio)
for k, val in m.items():
    print(k, val)
print("audio.rates", rates)
clean = len(audio) > 0 and all(m[k] == 0 for k in m if k.endswith(("failed", "skipped", "empty_or_nan")))
coverage = (len(v["per_sample"]) == v["summary"]["total_samples"] == vw["evaluated"] == 10
            and len({r["sample_id"] for r in v["per_sample"]}) == 10
            and len(s["per_request"]) == s["summary"]["completed_requests"] == w["evaluated"] == sim["evaluated"] == u["evaluated"] == 1088
            and len({r["id"] for r in s["per_request"]}) == 1088
            and len(glob.glob(f"{d}/videoamme/audio/*.wav")) == 10
            and len(glob.glob(f"{d}/seedtts/audio/*.wav")) == 1088)
print("coverage.complete", coverage)
json.dump({"metrics": m, "rates": rates, "complete": clean and coverage}, open(f"{d}/metrics.json", "w"))
print("RUN_COMPLETE" if clean and coverage else "RUN_INCOMPLETE")
PY
```

A run is incomplete when a step printed `NOT_READY` or `GPU_BUSY`, or this script does not print `RUN_COMPLETE`. Stop any server, move the run aside and repeat it:

```bash
kill -- -"$(cat /tmp/omni_server.pid)"
mv "$OUT" "$OUT-incomplete"
```

## Measuring accurately

- **Fixed GPU and CPUs.** Measure on one GPU, with the shell pinned to that GPU's CPU group by the shell block. The server and the client share those 28 logical CPUs. Do not run the CI stages or anything else on the pair at the same time.
- **Fresh server per workload.** Every workload in every run starts its own server and waits for GPU memory to drop below 1 GiB after stopping it. Never reuse a server across runs.
- **Compile and cold start.** Torch compile is on. The compile cache is in `$BENCH_DIR/omni-cache/torchinductor`. It is kept across containers and commits. The first server start with an empty cache is much slower than later starts. Run `run0` first to fill the cache and discard it. Run `run0` again after any change to model code. Delete that directory to measure a true cold start.
- **Client warmup.** Before the timed window, both clients send 16 warmup requests (the concurrency), all with the first sample. A failed warmup request stops the client. Keep this default so the timed batch does not absorb cold work.
- **Three runs.** Run `run1`, `run2` and `run3` after `run0`. Report the mean and the spread. This prints mean, sample standard deviation and the three values for each metric:

```bash
D=/results/$(git rev-parse --short HEAD)
python - "$D/run1" "$D/run2" "$D/run3" <<'PY'
import json, statistics, sys
runs = [json.load(open(f"{d}/metrics.json")) for d in sys.argv[1:]]
assert all(r["complete"] for r in runs), "a run is incomplete"
for k in runs[0]["metrics"]:
    vals = [r["metrics"][k] for r in runs]
    print(k, round(statistics.mean(vals), 6), round(statistics.stdev(vals), 6), vals)
PY
```

- **Same session.** Keep the GPU, CPU allocation and container configuration fixed. Record any session or allocation boundary when comparing runs, because it can change measurement noise.
- **Not comparable with CI gates.** CI runs these workloads on two GPUs behind a router with two workers, with compile off, on `seedtts-50` instead of the full set, and retries failed stages. The thresholds in `tests/test_model/omni_ci_config.py` do not apply to single GPU numbers.

To find where the time goes, see the `model-profiling` skill in this repository.

## Running the CI stages locally

This runs the 10 MiniCPM-o stage files of Omni model CI (`.github/workflows/test-qwen3-omni-ci.yaml`) as they are, on both GPUs with compile off. Stage 11 runs only for Qwen3-Omni. Open a new shell without the shell block, and set the output directory:

```bash
export CI_OUT=/results/$(git -C /sglang-omni rev-parse --short HEAD)/ci/run1
```

The prepare block sets the variables of the CI workflow and the `omni-setup` action that change results. Hugging Face stays offline and cache paths stay at their defaults. A new shell is limited to the container CPU set, so `OMNI_CI_CPUSET` reads it back in the format the CI pin script expects. It must print `CI_READY`.

```bash
cd /sglang-omni && . omni/bin/activate
export OMNI_CI_MODEL=minicpmo SGLANG_OMNI_TORCH_COMPILE_DEFAULT=0 CUDA_VISIBLE_DEVICES=0,1
export HF_HUB_OFFLINE=1
export OMNI_CI_CPUSET="$(taskset -pc $$ | awk '{print $NF}')" OMNI_CI_HOME=/root/omni_ci PYTHONPATH="$PWD"
export PYTORCH_ALLOC_CONF=expandable_segments:True NCCL_NVLS_ENABLE=0
export TORCHINDUCTOR_CACHE_DIR="$OMNI_CI_HOME/.torchinductor"
export FLASHINFER_WORKSPACE_BASE=/root FLASHINFER_JIT_DEBUG=0
export SGLANG_OMNI_ROUTER_BIN="$(bash .github/scripts/prepare_rust_router.sh build)"
test -x "$SGLANG_OMNI_ROUTER_BIN" && test ! -e "$CI_OUT" && mkdir -p "$CI_OUT" && git rev-parse HEAD > "$CI_OUT/commit.txt" && echo CI_READY
```

The test block writes a log and an exit code per stage, and clears the GPUs after each stage so a crashed stage does not hold them for the next one:

```bash
python -m pytest tests/test_model/test_qwen3_omni_thinker_length.py -v -s -x > "$CI_OUT/thinker_length.log" 2>&1; echo $? > "$CI_OUT/thinker_length.rc"; bash .github/scripts/delete_gpu_process.sh --kill-orphans > /dev/null
python -m pytest tests/test_model/test_qwen3_omni_tts_ci.py -v -s -x > "$CI_OUT/tts_ci.log" 2>&1; echo $? > "$CI_OUT/tts_ci.rc"; bash .github/scripts/delete_gpu_process.sh --kill-orphans > /dev/null
python -m pytest tests/test_model/test_qwen3_omni_mmmu_ci.py -v -s -x > "$CI_OUT/mmmu_ci.log" 2>&1; echo $? > "$CI_OUT/mmmu_ci.rc"; bash .github/scripts/delete_gpu_process.sh --kill-orphans > /dev/null
python -m pytest tests/test_model/test_qwen3_omni_mmmu_talker_ci.py -v -s -x > "$CI_OUT/mmmu_talker_ci.log" 2>&1; echo $? > "$CI_OUT/mmmu_talker_ci.rc"; bash .github/scripts/delete_gpu_process.sh --kill-orphans > /dev/null
python -m pytest tests/test_model/test_qwen3_omni_mmsu_ci.py -v -s -x > "$CI_OUT/mmsu_ci.log" 2>&1; echo $? > "$CI_OUT/mmsu_ci.rc"; bash .github/scripts/delete_gpu_process.sh --kill-orphans > /dev/null
python -m pytest tests/test_model/test_qwen3_omni_mmsu_talker_ci.py -v -s -x > "$CI_OUT/mmsu_talker_ci.log" 2>&1; echo $? > "$CI_OUT/mmsu_talker_ci.rc"; bash .github/scripts/delete_gpu_process.sh --kill-orphans > /dev/null
python -m pytest tests/test_model/test_qwen3_omni_videomme_ci.py -v -s -x > "$CI_OUT/videomme_ci.log" 2>&1; echo $? > "$CI_OUT/videomme_ci.rc"; bash .github/scripts/delete_gpu_process.sh --kill-orphans > /dev/null
python -m pytest tests/test_model/test_qwen3_omni_videomme_talker_ci.py -v -s -x > "$CI_OUT/videomme_talker_ci.log" 2>&1; echo $? > "$CI_OUT/videomme_talker_ci.rc"; bash .github/scripts/delete_gpu_process.sh --kill-orphans > /dev/null
python -m pytest tests/test_model/test_qwen3_omni_videoamme_ci.py -v -s -x > "$CI_OUT/videoamme_ci.log" 2>&1; echo $? > "$CI_OUT/videoamme_ci.rc"; bash .github/scripts/delete_gpu_process.sh --kill-orphans > /dev/null
python -m pytest tests/test_model/test_qwen3_omni_videoamme_talker_tp2_ci.py -v -s -x > "$CI_OUT/videoamme_talker_tp2_ci.log" 2>&1; echo $? > "$CI_OUT/videoamme_talker_tp2_ci.rc"; bash .github/scripts/delete_gpu_process.sh --kill-orphans > /dev/null
grep -H . "$CI_OUT"/*.rc
```

An exit code of 0 is a pass. All stages except `thinker_length` and `tts_ci` print an `Accuracy:` or `Overall accuracy:` line in their log.

## Reference results

There are no published numbers yet for one GPU at concurrency 16 with the commands above. These come from other setups:

| Workload | Source | Setup | Results |
|---|---|---|---|
| Video-AMME Talker | [#2580](https://github.com/sgl-project/sglang-omni/issues/2580), main `5ed8a8d` | One H200, concurrency 8, 50 questions, seed 0 (CI sends no seed), text accuracy only with no WER | Accuracy 0.68, 0.33 req/s |
| Video-AMME Talker | [#2552](https://github.com/sgl-project/sglang-omni/pull/2552), CI stage 10, 7 runs | Two H100s, DP 2, compile off | 0.643 to 0.668 req/s, Accuracy 0.7 in every run, WER 0.004032 in 6 runs and 0.008065 in one excluded run (one word apart) |
| seed-tts voice clone | [#2532](https://github.com/sgl-project/sglang-omni/pull/2532), main plus that PR before merge | One H200, CI server parameters, concurrency 16, 1088 samples, MPS on | 14.22 req/s, WER corpus without samples above 50% 1.47%, SIM 49.78 |

The #2532 run used MPS. The commands on this page do not.

## Notes

- **The first start is slow.** A start with an empty compile cache takes much longer than a warm start. One run starts three servers.
- **Video-AMME Talker is one batch.** Ten requests do not fill 16 slots, so its throughput is mostly the completion time of that one batch. Small changes in one request move it.
- **seed-tts output length varies.** The client samples at temperature 0.7 without a seed, as CI does. One long output can move throughput at concurrency 16 a lot. Check `seedtts.completion_tokens_sum` before reading a throughput change. Passing `--seed 0` controls sampling variation, but does not guarantee identical outputs under concurrent scheduling. Record the seed and output lengths. CI sends no seed, so that setting differs from CI.
- **CI differs from these runs.** CI wraps each stage in `.github/scripts/run_flaky_pytest.sh`, which runs a failed stage up to 3 times. The local test block does not, so a stage that fails only sometimes shows up here. CI gives the GPU pair 28 logical CPUs. This page gives it 56. Omni CI draws MiniCPM-o or Qwen3-Omni at random; the `run-omni-minicpm-o-4.5` label, with `run-omni-ci` or `run-all-ci`, selects MiniCPM-o.
- **Stages may fail on main.** A stage can fail on main. Run it on an earlier commit to tell an existing failure from a new one.
- **Some datasets are CI subsets.** `Video_AMME_ci`, `Video_MME_ci`, `seed-tts-eval-50-arrow`, `mmmu-ci-50` and `mmsu-ci-2000` are CI subsets hosted on Hugging Face. Their results are not full benchmark estimates.
- **Some CI subsets are not pinned.** `seed-tts-eval-50-arrow`, `mmmu-ci-50` and `mmsu-ci-2000` load at whatever revision is cached. When this page was written, Hugging Face main was `351f38c8d5ea36deda18dd9218645cda8f5c91b3`, `ff72fd69cc7e0719e04a0ddb12d160de89c6fefe` and `5ae6ed4343a89566be6dd90023d529f1bd97802d`.
