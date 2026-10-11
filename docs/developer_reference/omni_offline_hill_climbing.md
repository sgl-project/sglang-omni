# MiniCPM-o offline hill climbing runbook

This page is for hill climbing MiniCPM-o 4.5 offline serving in SGLang-Omni. The goal is to change the runtime so the measured metrics get better, with no regression anywhere else. Full duplex serving is covered in [Full-Duplex-Bench v1.5 runbook](full_duplex_bench.md).

Run every command as written, in order. Each check prints a fixed word such as `READY`. If the word does not print, stop and report the output to the maintainers.

## What is Measured

**Code scope.** New code stays inside `sglang_omni/models/minicpm_o`. Other models do not change. A change to a shared hot path (for example `OmniScheduler` in `sglang_omni/scheduling/omni_scheduler.py`) is tested hard in CI.

**Main workloads.** Both run on one GPU at concurrency 16.

| Workload | Input and output | Samples |
|---|---|---|
| Video-AMME Talker | Video plus a spoken question in, text plus speech out | The first 10 questions of `Video_AMME_ci`, as in CI |
| seed-tts voice clone | Text in, cloned speech out | The 1088 sample EN full set |

Ten requests do not fill 16 slots, so Video-AMME Talker throughput is mostly the completion time of that one batch. Each candidate names one workload as its target. The target must get faster, and the other must not get slower.

**CI stages.** Stages 1 to 10 of Omni model CI with `minicpmo` (`.github/workflows/test-qwen3-omni-ci.yaml`) must not regress. Stage 11 runs only for Qwen3-Omni.

**Baseline.** Main `921ea2c83acbfd7e9247ff38d63d8963b7572b7d` (`921ea2c8`). It does not move with main. MiniCPM-o GPU CI has not run since [#2667](https://github.com/sgl-project/sglang-omni/pull/2667) merged, so the baseline itself may fail some CI stages.

## Layout and variables

One container holds a GPU pair. Steps 4 to 6 measure on the first GPU, pinned to its CPU group. Step 7 runs the CI stages on both GPUs. Never run the two at the same time. Checkouts, downloads and results live in `$HILL_DIR` on the host.

The container holds three checkouts of one git repository. `/base` is the baseline, and the client and all scoring run from it. `/ref` is the last accepted version, the branch `accepted`. `/cand` is the candidate. These are all the variables:

| Variable | Set in | Meaning |
|---|---|---|
| `HILL_DIR` | Step 1 | `/data/omni-hill` on the host |
| `GPU_A`, `GPU_B` | Step 1 | The host GPU pair. Steps 4 to 6 use `GPU_A` |
| `PASS_CPUS`, `CI_CPUS` | Step 1 | CPU groups of `GPU_A` and of both GPUs |
| `CAND_SHA` | Step 3, the shell block | The commit of `/cand` |
| `CHECKOUT`, `OUT`, `CI_OUT` | A line from Step 6 or 7 | The checkout, and the pass or round directory |

## Step 1: Start the container

On the host:

Set `GIT_NAME` and `GIT_EMAIL` to the identity you use for repository commits.

```bash
export HILL_DIR=/data/omni-hill
export GPU_A=0 GPU_B=1
export GIT_NAME="Your Name"
export GIT_EMAIL=you@example.com
mkdir -p "$HILL_DIR/hf" "$HILL_DIR/metric-cache" "$HILL_DIR/results"
mkdir -p "$HILL_DIR/base" "$HILL_DIR/ref" "$HILL_DIR/cand"
```

Make the CPU groups. Each GPU gets 14 cores on its NUMA node with their SMT siblings, skipping cores 0 and 1. It must print `CPUS_OK`.

```bash
python3 - "$GPU_A" "$GPU_B" > "$HILL_DIR/cpus.env" <<'PY'
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
print("PASS_CPUS=" + ",".join(map(str, a)))
print("CI_CPUS=" + ",".join(map(str, sorted(a + b))))
PY
cat "$HILL_DIR/cpus.env" && . "$HILL_DIR/cpus.env" && test -n "$PASS_CPUS" && echo CPUS_OK
```

A `KeyError` means a GPU of the pair gets no group. Then set `GPU_A=2 GPU_B=3` (then `4 5`, then `6 7`) and run the step again. If `6 7` also fails, stop and report the output of `lscpu -p=CPU,CORE,NODE` and `nvidia-smi topo -m`. Start the container with the CI image and options:

```bash
docker run -d --init --ipc host --cap-add=SYS_PTRACE \
    --gpus "\"device=$GPU_A,$GPU_B\"" --cpuset-cpus "$CI_CPUS" --env-file "$HILL_DIR/cpus.env" \
    -e GIT_NAME -e GIT_EMAIL \
    -v "$HILL_DIR/hf:/root/.cache/huggingface" \
    -v "$HILL_DIR/metric-cache:/root/.cache/sglang-omni" \
    -v "$HILL_DIR/results:/results" \
    -v "$HILL_DIR/base:/base" -v "$HILL_DIR/ref:/ref" -v "$HILL_DIR/cand:/cand" \
    --name omni-hill \
    hongccc/sglang-omni@sha256:ebe4239e29a764ee3a2806385c061c5fd438a26f01458e503d3822dcba5790df \
    sleep infinity
```

Open a shell with this command, every time you need one. After a host reboot, run `docker start omni-hill` first.

```bash
docker exec -it omni-hill bash
```

All later commands run in this shell.

## Step 2: Set up the checkouts

```bash
git clone https://github.com/sgl-project/sglang-omni.git /base
git -C /base checkout --detach 921ea2c83acbfd7e9247ff38d63d8963b7572b7d
git -C /base branch accepted 921ea2c83acbfd7e9247ff38d63d8963b7572b7d
git -C /base worktree add --detach /ref accepted
git -C /base worktree add --detach /cand accepted
git -C /cand config user.name "$GIT_NAME"
git -C /cand config user.email "$GIT_EMAIL"
```

Create one venv per checkout with the CI script `.github/scripts/prepare_omni_venv.sh`. Each checkout gets the venv `omni`, as in CI. Each command ends with `Fresh environment ready`.

```bash
cd /base && OMNI_CI_HOME=/root/venv-base bash .github/scripts/prepare_omni_venv.sh omni
cd /ref && OMNI_CI_HOME=/root/venv-ref bash .github/scripts/prepare_omni_venv.sh omni
cd /cand && OMNI_CI_HOME=/root/venv-cand bash .github/scripts/prepare_omni_venv.sh omni
```

Both dependency comparisons must print nothing:

```bash
diff <(/base/omni/bin/python -m pip freeze --exclude-editable) <(/ref/omni/bin/python -m pip freeze --exclude-editable)
diff <(/base/omni/bin/python -m pip freeze --exclude-editable) <(/cand/omni/bin/python -m pip freeze --exclude-editable)
```

## Step 3: Download models and data

```bash
cd /base && . omni/bin/activate
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

`Video_MME_ci` holds the videos that Video-AMME points to, so Step 4 needs it too. `seedtts-50`, `mmmu-ci-50` and `mmsu-ci-2000` are for Step 7. A download by commit does not set `refs/main`, so set those references for offline loading:

```bash
H=/root/.cache/huggingface/hub
mkdir -p $H/models--openbmb--MiniCPM-o-4_5/refs $H/models--Qwen--Qwen3-ASR-1.7B/refs $H/datasets--zhaochenyang20--Video_AMME_ci/refs $H/datasets--zhaochenyang20--Video_MME_ci/refs
echo -n 503e754207c94da6bb26850b4469f367c9ea3582 > $H/models--openbmb--MiniCPM-o-4_5/refs/main
echo -n 7278e1e70fe206f11671096ffdd38061171dd6e5 > $H/models--Qwen--Qwen3-ASR-1.7B/refs/main
echo -n 7a37507f1b53416b9cb6641378e5a098ea535ab8 > $H/datasets--zhaochenyang20--Video_AMME_ci/refs/main
echo -n 833bd815c628ff277911bea3b1563545b21d5e27 > $H/datasets--zhaochenyang20--Video_MME_ci/refs/main
```

Check the pinned revisions. Every line must print its `OK_` word. Runs load these repositories offline through `refs/main`, except seed-tts, which `benchmarks/dataset/seedtts.py` loads at a pinned revision.

```bash
test "$(cat ~/.cache/huggingface/hub/models--openbmb--MiniCPM-o-4_5/refs/main)" = 503e754207c94da6bb26850b4469f367c9ea3582 && echo OK_MINICPMO
test "$(cat ~/.cache/huggingface/hub/models--Qwen--Qwen3-ASR-1.7B/refs/main)" = 7278e1e70fe206f11671096ffdd38061171dd6e5 && echo OK_ASR
test "$(cat ~/.cache/huggingface/hub/datasets--zhaochenyang20--Video_AMME_ci/refs/main)" = 7a37507f1b53416b9cb6641378e5a098ea535ab8 && echo OK_VIDEOAMME
test "$(cat ~/.cache/huggingface/hub/datasets--zhaochenyang20--Video_MME_ci/refs/main)" = 833bd815c628ff277911bea3b1563545b21d5e27 && echo OK_VIDEOMME
```

**The shell block.** Run it at the start of every new shell for Steps 4 to 6.

```bash
taskset -pc "$PASS_CPUS" $$
cd /base && . /base/omni/bin/activate
export CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 SGLANG_OMNI_STRICT_PORT=1 SGLANG_OMNI_STARTUP_TIMEOUT=1800
export CAND_SHA="$(git -C /cand rev-parse HEAD)"
echo "${SGLANG_OMNI_TORCH_COMPILE_DEFAULT-unset} ${PYTORCH_ALLOC_CONF-unset}"
```

It must print `unset unset`: torch compile stays on (the default) and `PYTORCH_ALLOC_CONF` stays unset. A cold start with compile needs more than the default 600 second startup timeout. Stay in `/base`, because `benchmarks` is imported from the working directory.

## Step 4: Run one measurement pass

One pass measures one checkout once on both workloads, with fresh servers.

### 4.1 Choose the pass

Run the shell block. Run the first line from Step 6 whose `OUT` directory does not exist yet. The next command must print `PASS_READY`. The checkout must be clean, `OUT` must be new and port 8000 must be free.

```bash
test -z "$(git -C "$CHECKOUT" status --porcelain)" && test ! -e "$OUT" && ! curl -s -o /dev/null localhost:8000/health && mkdir -p "$OUT/videoamme" "$OUT/seedtts" && git -C "$CHECKOUT" rev-parse HEAD > "$OUT/commit.txt" && echo PASS_READY
```

### 4.2 Video-AMME Talker

Parameters match CI stage 10. Start the server. It must print `READY`. `/tmp/omni_server.pid` holds the server's process group, so the stop command stops all of it.

```bash
setsid bash -c 'echo $$ > "$0"; exec "$@"' /tmp/omni_server.pid \
    "$CHECKOUT/omni/bin/sgl-omni" serve \
    --model-path openbmb/MiniCPM-o-4_5 --model-name minicpmo \
    --thinker.engine.mem_fraction_static 0.55 --talker.engine.mem_fraction_static 0.15 \
    --thinker.factory.max_seq_len 32768 \
    --port 8000 > "$OUT/videoamme.log" 2>&1 &
timeout 1800 bash -c 'until curl -sf localhost:8000/health > /dev/null; do sleep 5; done' && echo READY || echo NOT_READY
```

Run the client. As in CI, WER comes later, from the ASR server.

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

Stop the server. It must print `GPU_FREE` (memory on the first GPU below 1 GiB).

```bash
kill -- -"$(cat /tmp/omni_server.pid)"
timeout 600 bash -c 'until [ "$(nvidia-smi -i 0 --query-gpu=memory.used --format=csv,noheader,nounits)" -lt 1024 ]; do sleep 5; done' && echo GPU_FREE || echo GPU_BUSY
```

### 4.3 seed-tts voice clone

Server parameters match CI stage 2. Start the server. It must print `READY`.

```bash
setsid bash -c 'echo $$ > "$0"; exec "$@"' /tmp/omni_server.pid \
    "$CHECKOUT/omni/bin/sgl-omni" serve \
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

### 4.4 Score

Start Qwen3-ASR, always from `/base`. It must print `READY`.

```bash
setsid bash -c 'echo $$ > "$0"; exec "$@"' /tmp/omni_server.pid \
    /base/omni/bin/sgl-omni serve \
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

SIM and UTMOS run without a server:

```bash
python -m benchmarks.eval.benchmark_omni_seedtts --similarity-only \
    --meta zhaochenyang20/seed-tts-eval-arrow --model minicpmo --output-dir "$OUT/seedtts"
python -m benchmarks.eval.benchmark_omni_seedtts --utmos-only \
    --meta zhaochenyang20/seed-tts-eval-arrow --model minicpmo --output-dir "$OUT/seedtts"
```

## Step 5: Read the results

Every metric is a JSON field. Paths are under `$OUT`.

| File | Fields |
|---|---|
| `videoamme/videoamme_results.json` | Accuracy `summary.accuracy` (0 to 1), failed requests `summary.failed`, prompt tokens `per_sample[].prompt_tokens` by `sample_id`, and the speed fields below under `speed.` |
| `seedtts/speed_results.json` | Failed requests `summary.failed_requests`, prompt tokens `per_request[].prompt_tokens` by `id`, output tokens `per_request[].completion_tokens`, and the speed fields below under `summary.` |
| Speed fields | Throughput `throughput_qps` (req/s), latency `latency_mean_s` and `latency_p95_s`, `rtf_mean`, Audio duration mean `audio_duration_mean_s`, audio throughput `audio_throughput_s_per_s` |
| `videoamme/wer_summary.json` | The WER fields below, at top level |
| `seedtts/wer_results.json` | The WER fields below, under `summary.` |
| WER fields | WER `wer_corpus` (a fraction, 0.005 is 0.5 points), >50% WER samples `n_above_50_pct_wer`, WER corpus (excl >50%) `wer_below_50_corpus` (for comparison with CI only), unscored samples `skipped` |
| `seedtts/similarity_results.json` | SIM `summary.speaker_similarity_mean` (cosine × 100), `summary.skipped` |
| `seedtts/utmos_results.json` | UTMOS `summary.utmos_mean` (1 to 5), `summary.skipped` |

Higher is better for Accuracy, throughput, SIM and UTMOS. Lower is better for latency, RTF and WER. Failed and skipped counts must be 0.

This prints every field above, checks the speech files and writes `$OUT/metrics.json`. The last line must be `PASS_VALID`.

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
valid = len(audio) > 0 and all(m[k] == 0 for k in m if k.endswith(("failed", "skipped", "empty_or_nan")))
coverage = (len(v["per_sample"]) == v["summary"]["total_samples"] == vw["evaluated"] == 10
            and len({r["sample_id"] for r in v["per_sample"]}) == 10
            and len(s["per_request"]) == s["summary"]["completed_requests"] == w["evaluated"] == sim["evaluated"] == u["evaluated"] == 1088
            and len({r["id"] for r in s["per_request"]}) == 1088
            and len(glob.glob(f"{d}/videoamme/audio/*.wav")) == 10
            and len(glob.glob(f"{d}/seedtts/audio/*.wav")) == 1088)
valid = valid and coverage
print("coverage.complete", coverage)
tokens = {f"v/{r['sample_id']}": r["prompt_tokens"] for r in v["per_sample"]}
tokens.update({f"s/{r['id']}": r["prompt_tokens"] for r in s["per_request"]})
json.dump({"metrics": m, "rates": rates, "valid": valid, "prompt_tokens": tokens}, open(f"{d}/metrics.json", "w"))
print("PASS_VALID" if valid else "PASS_INVALID")
PY
```

**Invalid pass.** A pass is invalid when Step 4 printed `NOT_READY` or `GPU_BUSY`, or this script does not print `PASS_VALID`. Stop any server and move the pass aside:

```bash
kill -- -"$(cat /tmp/omni_server.pid)"
mv "$OUT" "$OUT-invalid"
```

Run the pass again right away with the same line. If it is invalid again, stop and report both directories to the maintainers.

## Step 6: Repeat and compare

A pass is Step 4 then Step 5. Run 0 warms the compile cache and is not scored. A metric's value is its plain mean over runs 1, 2 and 3.

### Baseline, once at the start of the task

Run four passes, with these lines in this order:

```bash
export CHECKOUT=/base OUT=/results/baseline/run0
export CHECKOUT=/base OUT=/results/baseline/run1
export CHECKOUT=/base OUT=/results/baseline/run2
export CHECKOUT=/base OUT=/results/baseline/run3
```

### Each candidate

Start from the last accepted version:

```bash
git -C /cand checkout --detach accepted
git -C /ref checkout --detach accepted
```

Make the change in `/cand`. Commit it with the target in the message. The target cannot change later.

```bash
git -C /cand add -A
```

For a Video-AMME Talker target:

```bash
git -C /cand commit -m "target: videoamme"
```

For a seed-tts target:

```bash
git -C /cand commit -m "target: seedtts"
```

It must print `CAND_CLEAN`.

```bash
test -z "$(git -C /cand status --porcelain)" && echo CAND_CLEAN
```

Run the shell block again to read the new `CAND_SHA`. Then run eight passes in this order:

```bash
export CHECKOUT=/ref  OUT=/results/cand-$CAND_SHA/ref/run0
export CHECKOUT=/cand OUT=/results/cand-$CAND_SHA/cand/run0
export CHECKOUT=/ref  OUT=/results/cand-$CAND_SHA/ref/run1
export CHECKOUT=/cand OUT=/results/cand-$CAND_SHA/cand/run1
export CHECKOUT=/cand OUT=/results/cand-$CAND_SHA/cand/run2
export CHECKOUT=/ref  OUT=/results/cand-$CAND_SHA/ref/run2
export CHECKOUT=/ref  OUT=/results/cand-$CAND_SHA/ref/run3
export CHECKOUT=/cand OUT=/results/cand-$CAND_SHA/cand/run3
```

### Compare

This prints the base, ref and cand means, one line per [Acceptance](#acceptance) rule (`PASS`, `FAIL` or `REVIEW`), and a verdict. A missing file raises an error. Stop and treat that comparison as `REJECT`.

```bash
python - /results/baseline "/results/cand-$CAND_SHA" <<'PY'
import json, subprocess, sys
target = subprocess.check_output(["git", "-C", "/cand", "log", "-1", "--format=%s"], text=True).strip().removeprefix("target: ")
other = {"videoamme": "seedtts", "seedtts": "videoamme"}[target]
load = lambda g: [json.load(open(f"{g}/run{n}/metrics.json")) for n in (1, 2, 3)]
runs = {"base": load(sys.argv[1]), "ref": load(f"{sys.argv[2]}/ref"), "cand": load(f"{sys.argv[2]}/cand")}
B, R, C = ((lambda k, g=g: sum(r["metrics"][k] for r in runs[g]) / 3) for g in ("base", "ref", "cand"))
print("target", target, "| metric base ref cand")
for k in runs["base"][0]["metrics"]:
    print(k, round(B(k), 6), round(R(k), 6), round(C(k), 6))
status = []
def rule(name, ok, review=False):
    status.append("PASS" if ok else ("REVIEW" if review else "FAIL"))
    print(status[-1], name)
first = runs["base"][0]
for g, rs in runs.items():
    for i, r in enumerate(rs, 1):
        rule(f"{g} run{i} is valid", r["valid"])
for i, r in enumerate(runs["cand"], 1):
    rule(f"cand run{i} prompt tokens and sample rates equal baseline",
         r["prompt_tokens"] == first["prompt_tokens"] and r["rates"] == first["rates"])
rule("videoamme.accuracy >= baseline", C("videoamme.accuracy") >= B("videoamme.accuracy"))
for wl in ("videoamme", "seedtts"):
    k = f"{wl}.wer.wer_corpus"
    rule(f"{k} <= max(baseline * 1.25, baseline + 0.005)", C(k) <= max(B(k) * 1.25, B(k) + 0.005))
    rule(f"{k} not above baseline", C(k) <= B(k), review=True)
    k = f"{wl}.wer.n_above_50_pct_wer"
    rule(f"{k} <= baseline", C(k) <= B(k))
    for k in (f"{wl}.latency_mean_s", f"{wl}.latency_p95_s", f"{wl}.rtf_mean"):
        rule(f"{k} <= 1.03 * ref and baseline", C(k) <= 1.03 * R(k) and C(k) <= 1.03 * B(k))
    k = f"{wl}.audio_duration_mean_s"
    rule(f"{k} within 5% of baseline", abs(C(k) / B(k) - 1) <= 0.05, review=True)
for k in ("seedtts.sim", "seedtts.utmos"):
    rule(f"{k} >= 0.97 * baseline", C(k) >= 0.97 * B(k))
    rule(f"{k} not below baseline", C(k) >= B(k), review=True)
t, o = f"{target}.throughput_qps", f"{other}.throughput_qps"
rule(f"{t} >= 1.02 * ref", C(t) >= 1.02 * R(t))
rule(f"{o} >= 0.97 * ref and baseline", C(o) >= 0.97 * R(o) and C(o) >= 0.97 * B(o))
print("ACCEPT" if all(s == "PASS" for s in status) else ("REJECT" if "FAIL" in status else "REVIEW"))
PY
```

`ACCEPT`: go on to Step 7. `REVIEW`: send the full output to the maintainers and wait for their decision before Step 7. `REJECT`: start the next candidate.

## Step 7: Run the CI stages

This runs the 10 CI stage files as they are, three times in `/base` and in `/cand`, on both GPUs with compile off as in CI. It is required before acceptance and before a PR. Open a new shell, without the shell block, and run:

```bash
export CAND_SHA="$(git -C /cand rev-parse HEAD)"
```

Run six rounds. Each round is its line below, then the prepare block, then the test block.

```bash
export CHECKOUT=/base CI_OUT=/results/cand-$CAND_SHA/ci/base/run1
export CHECKOUT=/cand CI_OUT=/results/cand-$CAND_SHA/ci/cand/run1
export CHECKOUT=/cand CI_OUT=/results/cand-$CAND_SHA/ci/cand/run2
export CHECKOUT=/base CI_OUT=/results/cand-$CAND_SHA/ci/base/run2
export CHECKOUT=/base CI_OUT=/results/cand-$CAND_SHA/ci/base/run3
export CHECKOUT=/cand CI_OUT=/results/cand-$CAND_SHA/ci/cand/run3
```

The prepare block matches the CI workflow and the `omni-setup` action, except that Hugging Face stays offline. It must print `CI_READY`.

```bash
cd "$CHECKOUT" && . omni/bin/activate
export OMNI_CI_MODEL=minicpmo SGLANG_OMNI_TORCH_COMPILE_DEFAULT=0 CUDA_VISIBLE_DEVICES=0,1
export HF_HUB_OFFLINE=1
export OMNI_CI_CPUSET="$CI_CPUS" OMNI_CI_HOME="$HOME/omni_ci" PYTHONPATH="$PWD"
export PYTORCH_ALLOC_CONF=expandable_segments:True NCCL_NVLS_ENABLE=0
export TORCHINDUCTOR_CACHE_DIR="$OMNI_CI_HOME/.torchinductor"
export FLASHINFER_WORKSPACE_BASE=/root FLASHINFER_JIT_DEBUG=0
export SGLANG_OMNI_ROUTER_BIN="$(bash .github/scripts/prepare_rust_router.sh build)"
test -x "$SGLANG_OMNI_ROUTER_BIN" && test ! -e "$CI_OUT" && mkdir -p "$CI_OUT" && git rev-parse HEAD > "$CI_OUT/commit.txt" && echo CI_READY
```

The test block:

```bash
python -m pytest tests/test_model/test_qwen3_omni_thinker_length.py -v -s -x > "$CI_OUT/thinker_length.log" 2>&1; echo $? > "$CI_OUT/thinker_length.rc"
python -m pytest tests/test_model/test_qwen3_omni_tts_ci.py -v -s -x > "$CI_OUT/tts_ci.log" 2>&1; echo $? > "$CI_OUT/tts_ci.rc"
python -m pytest tests/test_model/test_qwen3_omni_mmmu_ci.py -v -s -x > "$CI_OUT/mmmu_ci.log" 2>&1; echo $? > "$CI_OUT/mmmu_ci.rc"
python -m pytest tests/test_model/test_qwen3_omni_mmmu_talker_ci.py -v -s -x > "$CI_OUT/mmmu_talker_ci.log" 2>&1; echo $? > "$CI_OUT/mmmu_talker_ci.rc"
python -m pytest tests/test_model/test_qwen3_omni_mmsu_ci.py -v -s -x > "$CI_OUT/mmsu_ci.log" 2>&1; echo $? > "$CI_OUT/mmsu_ci.rc"
python -m pytest tests/test_model/test_qwen3_omni_mmsu_talker_ci.py -v -s -x > "$CI_OUT/mmsu_talker_ci.log" 2>&1; echo $? > "$CI_OUT/mmsu_talker_ci.rc"
python -m pytest tests/test_model/test_qwen3_omni_videomme_ci.py -v -s -x > "$CI_OUT/videomme_ci.log" 2>&1; echo $? > "$CI_OUT/videomme_ci.rc"
python -m pytest tests/test_model/test_qwen3_omni_videomme_talker_ci.py -v -s -x > "$CI_OUT/videomme_talker_ci.log" 2>&1; echo $? > "$CI_OUT/videomme_talker_ci.rc"
python -m pytest tests/test_model/test_qwen3_omni_videoamme_ci.py -v -s -x > "$CI_OUT/videoamme_ci.log" 2>&1; echo $? > "$CI_OUT/videoamme_ci.rc"
python -m pytest tests/test_model/test_qwen3_omni_videoamme_talker_tp2_ci.py -v -s -x > "$CI_OUT/videoamme_talker_tp2_ci.log" 2>&1; echo $? > "$CI_OUT/videoamme_talker_tp2_ci.rc"
```

All stages except `thinker_length` and `tts_ci` print an `Accuracy:` or `Overall accuracy:` line. After the six rounds, this prints per stage the exit codes and mean Accuracy of each side, and a verdict. `PASS` means cand matched base. `FAIL` means cand failed more rounds or has lower Accuracy. `BASE_FAILS` and `UNVERIFIED` mean base itself failed or crashed.

```bash
python - "/results/cand-$CAND_SHA/ci" <<'PY'
import re, sys
root = sys.argv[1]
stages = ["thinker_length", "tts_ci", "mmmu_ci", "mmmu_talker_ci", "mmsu_ci", "mmsu_talker_ci",
          "videomme_ci", "videomme_talker_ci", "videoamme_ci", "videoamme_talker_tp2_ci"]
def side(name, stage):
    codes, accs = [], []
    for n in (1, 2, 3):
        d = f"{root}/{name}/run{n}"
        codes.append(int(open(f"{d}/{stage}.rc").read()))
        accs += [float(a) for a in re.findall(r"^\s*(?:Overall accuracy|Accuracy):\s+([0-9.]+)", open(f"{d}/{stage}.log").read(), re.M)]
    return codes, (sum(accs) / len(accs) if accs else None)
for stage in stages:
    bc, ba = side("base", stage)
    cc, ca = side("cand", stage)
    if any(c not in (0, 1) for c in bc):
        verdict = "UNVERIFIED"
    elif all(c == 0 for c in bc):
        verdict = "PASS" if all(c == 0 for c in cc) and (ba is None or (ca is not None and ca >= ba)) else "FAIL"
    else:
        verdict = "BASE_FAILS" if sum(c != 0 for c in cc) <= sum(c != 0 for c in bc) else "FAIL"
    print(stage, "base", bc, "cand", cc, "acc_base", ba, "acc_cand", ca, verdict)
PY
```

## Step 8: Record the result

A candidate is accepted when Step 6 printed `ACCEPT` (or the maintainers approved a `REVIEW`), and every Step 7 stage printed `PASS`. An exception requires explicit maintainer approval for a documented existing baseline failure. Any new candidate failure still blocks acceptance. Record it and list every accepted commit:

```bash
git -C /base branch -f accepted "$(git -C /cand rev-parse HEAD)"
git -C /base log --oneline 921ea2c83acbfd7e9247ff38d63d8963b7572b7d..accepted
```

At the end of the task, run `docker rm -f omni-hill` on the host. The checkouts and accepted commits stay in `$HILL_DIR/base`, `$HILL_DIR/ref` and `$HILL_DIR/cand`. The results stay in `$HILL_DIR/results`.

## Acceptance

Small samples are for quick trials only. A version is accepted only after Steps 4 to 7, measured fresh next to the last accepted version. Never reuse earlier numbers. The Step 6 and Step 7 scripts apply every rule below. CI numbers do not apply, because CI turns compile off and uses two GPUs.

**Correctness, against the baseline.**

- Every run contains all 10 Video-AMME and 1088 seed-tts samples, with unique sample IDs and one WAV per sample. Generation and all scoring stages cover those counts with 0 failed requests and 0 skipped samples. A missing JSON field is a failure.
- No speech output is empty or has NaN values, and the sample rate equals the baseline.
- Video-AMME Talker Accuracy is not below the baseline, with no tolerance.
- Prompt tokens equal the baseline for every sample.
- WER (`wer_corpus`) is at most max(baseline × 1.25, baseline + 0.005). The >50% WER sample count is not above the baseline.
- SIM and UTMOS are at least baseline × 0.97.
- WER, SIM or UTMOS worse than the baseline but within tolerance, or an Audio duration mean change over 5%, goes to the maintainers. Audio throughput shows whether a req/s gain comes from shorter speech.

**Performance, against the last accepted version.**

- Target workload Throughput is at least 1.02 × the last accepted version.
- The other workload's Throughput is at least 0.97 × the last accepted version and 0.97 × the baseline.
- For both workloads, latency mean, latency p95 and RTF mean are at most 1.03 × each of those two versions. Checking both stops small regressions from adding up.

**CI stages.** The Step 7 verdicts apply. Step 7 runs without the CI retry wrapper (`.github/scripts/run_flaky_pytest.sh`), so a stage that fails only sometimes on the baseline counts as failing. For a `BASE_FAILS` or `UNVERIFIED` stage, record why, on which samples and with which values, from the logs under `/results/cand-$CAND_SHA/ci/`. Compare the actual failures, since matching failure counts do not show that the causes match. Wait for explicit approval of a specific baseline failure before Step 8. Reviewing the logs alone does not waive a stage, and any new failure blocks acceptance.

The 1.02 factor and the tolerances are provisional. They will be tuned after baseline and A/A runs on the target machine.

## Prior work

There are no numbers yet for one GPU at concurrency 16. These come from other setups.

- **Video-AMME Talker.** [#2580](https://github.com/sgl-project/sglang-omni/pull/2580) (main `5ed8a8d`), one H200, concurrency 8, 50 questions: Accuracy 0.68, 0.33 req/s. In [#2552](https://github.com/sgl-project/sglang-omni/pull/2552), CI stage 10 (two H100s, DP 2, compile off) ran 7 times: 0.643 to 0.668 req/s and Accuracy 0.7 every time. WER was 0.004032 in 6 runs and 0.008065 in one, a difference of one word.
- **seed-tts.** [#2532](https://github.com/sgl-project/sglang-omni/pull/2532), one H200, the CI server parameters, concurrency 16, 1088 voice clone samples: 14.22 req/s, WER corpus (excl >50%) 1.47%, SIM 49.78. That run used MPS, which stays off here.

Read #2580 first, the Video-AMME profile. At concurrency 8, CPU preprocessing is serial at about 3.0 s per request, and the GPU is busy only 17.1% to 27.6% of the time. Then read [#2273](https://github.com/sgl-project/sglang-omni/pull/2273) and [#2399](https://github.com/sgl-project/sglang-omni/pull/2399) (TTS and code2wav), and the optimization tracker [#2284](https://github.com/sgl-project/sglang-omni/pull/2284), which lists the areas already claimed. Related open PRs are #2316, #2480, #2487 and #2589 (preprocessing), #2529, #2528, #2330 and #2353 (talker), and #2356 (HiFT). Build on existing work. Copying a change from an existing PR does not count.

## Rules for PRs

- **Default on.** An optimization is on by default. If one method beats another with no tradeoff, make it the default and remove the old path where possible. Optional paths behind flags make the code unreadable, and would change the evaluation commands.
- **Fixed evaluation.** A PR does not change the evaluation commands, the evaluation and metric code under `benchmarks/`, or the CI under `.github/` and `tests/test_model/`. Adding or changing matching unit tests under `tests/unit_test/` is fine.
- **Local benchmark fixes.** If a command does not run, you may patch the benchmark locally. The patch changes how it runs, never what it measures. Save the diff, rerun the baseline with it, use it for ref and cand from then on, and keep it out of the PR.
- **Numerics.** A change that alters numerics or has a tradeoff (for example a kernel swap, or a batch composition that loses questions) goes in its own PR with a per sample accuracy comparison. The maintainers decide at review. Step 6 alone never accepts it.
- **CI label.** MiniCPM-o CI runs only with the `run-minicpmo` label. Ask for it in the PR description. The maintainers reproduce every PR.

## What does not count

| Category | Examples |
|---|---|
| Changing the measurement | Client, runner, metrics and request parameters (fps 2, 128 frames, 401408 pixels, `max_tokens` 256, temperature, question count, concurrency) |
| Changing the input | Resize interpolation and count, pixel dtype, video decoding, audio resampling and features, reference audio handling |
| Changing audio quality or precision | code2wav `n_timesteps=10` and `inference_cfg_rate=0.7`, moving any part to FP8, INT8, BF16 or FP16, turning on TF32 |
| Shorter outputs | Chat template, default thinker and talker sampling parameters, maximum length, EOS |
| Caches and special cases for the evaluation | Reusing final outputs across requests, cache keys from file path, sample ID or prompt text, caches kept across restarts, reading files before they are requested, special branches for Video-AMME or seed-tts |
| More resources | A second GPU, MPS, changing CPU affinity, more threads than the given CPUs. #2399 moved code2wav to a second GPU on H20 and gained 35% audio throughput at concurrency 64. That is not a runtime optimization |

## Notes

- **The first start is slow.** In #2580 on one H200 (28 CPUs), a server with an empty compile cache took about 10.5 minutes to become healthy, and about 5.4 to 5.8 minutes after that. One pass starts three servers.
- **seed-tts output length varies.** The client samples at temperature 0.7 without a seed, as CI does. One long output can move throughput at concurrency 16 a lot. Compare `seedtts.completion_tokens_sum` in the Step 6 output to see a gain from shorter outputs.
