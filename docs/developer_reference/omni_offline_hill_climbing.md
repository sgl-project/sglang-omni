# MiniCPM-o offline hill climbing runbook

This page is for hill climbing MiniCPM-o 4.5 offline serving in SGLang-Omni: changing the runtime so a fixed evaluation gets faster while every other workload stays at least as good. Full duplex serving is covered separately.

Unlike TTS hill climbing, this task must protect many workloads. The scripts, the baseline and the scoring are fixed for the whole task.

## Scope

- **Code.** New code stays inside `sglang_omni/models/minicpm_o`. Other models do not change. A change to a shared hot path (for example `OmniScheduler` in `sglang_omni/scheduling/omni_scheduler.py`) is tested hard in CI.
- **Protected stages.** Stages 1 to 10 of Omni model CI with `minicpmo` (`.github/workflows/test-qwen3-omni-ci.yaml`). Stage 11 runs only for Qwen3-Omni. The target workload must improve clearly. Every other workload must not get worse.

## Main workloads

Two workloads matter most. Both run on one GPU at concurrency 16.

| Workload | Input and output | Samples |
|---|---|---|
| Video-AMME Talker | Video plus a spoken question in, text plus speech out | The first 10 questions of `Video_AMME_ci`, as in CI |
| seed-tts voice clone | Text in, cloned speech out | The 1088 sample EN full set |

Ten requests do not fill 16 slots, so Video-AMME Talker throughput is mostly the completion time of that one batch. The 50 question set is a possible extension. Confirm it with the maintainers first, and do not switch sets in the middle of a task. The other 8 stages are guarded by the CI pytest files as they are.

## Baseline and pinned inputs

The baseline is main `921ea2c8`. It does not move with main. It includes [#2667](https://github.com/sgl-project/sglang-omni/pull/2667), which changed a shared hot path. MiniCPM-o GPU CI has not run since that merge, so the baseline itself may fail some stages. [Other CI stages](#other-ci-stages) says how to handle that.

Model and dataset revisions are in the command comments. Check them before running. Record the ones not listed (the step 4 datasets, and the SIM and UTMOS weights) after the first download. These revisions and the dependencies stay fixed for the whole task.

## Commands

Run the Environment, Server helpers and Steps 1 to 3 blocks in order in one shell inside the single GPU container. Later blocks use the functions and variables of earlier ones. They are examples: fill in the gaps their comments name before a real run.

### Environment

```bash
# Environment setup example. For more detail, read docs/cookbook/minicpm_o.md and tests/test_model/conftest.py
# CI image hongccc/sglang-omni@sha256:ebe4239e29a764ee3a2806385c061c5fd438a26f01458e503d3822dcba5790df, start the container with --init and --ipc host
# CPU group i: on the NUMA node of GPU i, skip cores 0 and 1, then take 14 physical cores in order plus their SMT siblings, 28 CPUs in total
# Groups do not overlap. Write the split to cpusets.txt and keep it for the whole task. One container per GPU: --gpus device=i --cpuset-cpus <group i>
# Three checkouts: /base is the baseline, /ref is the last accepted version, /cand is the candidate. The client, Qwen3-ASR, SIM and UTMOS always come from /base
git clone https://github.com/sgl-project/sglang-omni.git /base && git -C /base checkout 921ea2c83acbfd7e9247ff38d63d8963b7572b7d
git -C /base worktree add --detach /ref 921ea2c83acbfd7e9247ff38d63d8963b7572b7d
git -C /base worktree add --detach /cand 921ea2c83acbfd7e9247ff38d63d8963b7572b7d
# First run the baseline with all three on 921ea2c8. After that, /ref checks out the last accepted version and /cand the candidate
for t in /base /ref /cand; do
  ( cd $t && uv venv --system-site-packages .venv -p /usr/bin/python3.12 &&   # follows .github/scripts/prepare_omni_venv.sh
    echo 'import site; site.addsitedir("/opt/sglang/lib/python3.12/site-packages")' > .venv/lib/python3.12/site-packages/sglang-image.pth &&
    . .venv/bin/activate && python .github/scripts/omni_missing_dependencies.py --extra minicpm-o pyproject.toml | xargs -r -d '\n' -n 1 python -m pip install &&
    python .github/scripts/omni_missing_dependencies.py --overrides pyproject.toml | xargs -r -d '\n' python -m pip install --no-deps &&
    uv pip install --no-deps -e . )
done
diff <(/base/.venv/bin/python -m pip freeze --exclude-editable) <(/cand/.venv/bin/python -m pip freeze --exclude-editable)   # should be empty
cd /base && . .venv/bin/activate

hf download openbmb/MiniCPM-o-4_5                              # 503e754207c94da6bb26850b4469f367c9ea3582
hf download Qwen/Qwen3-ASR-1.7B                                # 7278e1e70fe206f11671096ffdd38061171dd6e5, the ASR model for WER
hf download zhaochenyang20/Video_AMME_ci --repo-type dataset   # 7a37507f1b53416b9cb6641378e5a098ea535ab8
hf download zhaochenyang20/Video_MME_ci --repo-type dataset    # 833bd815c628ff277911bea3b1563545b21d5e27
python -m benchmarks.dataset.prepare --dataset seedtts          # 27f4c1adee83b5b29b7c4b375f6b976324bda308
for d in seedtts-50 mmmu-ci-50 mmsu-ci-2000; do python -m benchmarks.dataset.prepare --dataset $d; done   # for step 4
python -m benchmarks.metrics.speaker_similarity_assets --warm-cache
python -m benchmarks.metrics.utmos --warm-cache   # UTMOS weights, fetched before going offline
# If any refs/main differs from the commits above, stop and report it
export HF_HUB_OFFLINE=1 SGLANG_OMNI_STRICT_PORT=1 SGLANG_OMNI_STARTUP_TIMEOUT=1800   # a cold start with compile takes longer than the default 600 seconds
# Do not set SGLANG_OMNI_TORCH_COMPILE_DEFAULT. On one GPU these two workloads keep compile on, which is the default
```

### Server helpers

```bash
# Example server start and stop. Before a real run, add a port check, cleanup on errors and a timeout on the GPU memory wait
up() {  # up <log name> <checkout> <serve args>: start a server and wait for /health
  setsid "$2/.venv/bin/sgl-omni" serve "${@:3}" --port 8000 > "$OUT/$1.log" 2>&1 &
  PID=$!
  timeout 1800 bash -c 'until curl -sf localhost:8000/health > /dev/null; do sleep 5; done' || exit 1
}
down() {  # kill the whole process group and wait until GPU memory drops below 1 GiB
  kill -- "-$PID"
  wait "$PID" || true
  until [ $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits) -lt 1024 ]; do sleep 5; done
}
M="--model-path openbmb/MiniCPM-o-4_5 --model-name minicpmo --thinker.engine.mem_fraction_static 0.55 --talker.engine.mem_fraction_static 0.15"
S="--meta zhaochenyang20/seed-tts-eval-arrow --model minicpmo"
```

### Steps 1 to 3: main workloads

```bash
# At the start of the task, run once with VERSIONS=base. After that, run each candidate with VERSIONS="ref cand", alternating the two versions
# Each RUN starts a fresh server for each workload. RUN 0 is not scored (it warms the compile cache). RUNs 1 to 3 are scored
VERSIONS=${VERSIONS:-base}
for RUN in 0 1 2 3; do
  ORDER=$VERSIONS
  if [ "$VERSIONS" = "ref cand" ] && [ "$RUN" = 2 ]; then
    ORDER="cand ref"
  else
    :
  fi
  for V in $ORDER; do
    OUT=results/$(git -C /$V rev-parse --short HEAD)/run$RUN; mkdir -p $OUT/videoamme $OUT/seedtts

    # 1. Video-AMME Talker. Server and request parameters match CI stage 10, on one GPU
    up videoamme /$V $M --thinker.factory.max_seq_len 32768
    python - --model minicpmo --port 8000 --repo-id zhaochenyang20/Video_AMME_ci --max-samples 10 \
        --video-fps 2 --video-max-frames 128 --video-max-pixels 401408 --max-tokens 256 --temperature 0 \
        --timeout-s 500 --max-concurrency 16 --enable-audio --output-dir $OUT/videoamme <<'PY'
import argparse, asyncio
from benchmarks.eval import benchmark_omni_videoamme as m
parser = argparse.ArgumentParser()
m.add_video_eval_args(parser, repo_help="")
cfg = m.video_eval_config_from_args(parser.parse_args())
m.wait_for_service(f"http://{cfg.host}:{cfg.port}")
res = asyncio.run(m.run_videoamme_eval(cfg, compute_wer=False))
m.print_videomme_accuracy_summary(res["summary"], cfg.model, title="Video-AMME Accuracy")
m.print_speed_summary(res["speed"], cfg.model, cfg.max_concurrency, title="Video-AMME Speed")
PY
    down

    # 2. seed-tts voice clone. Server parameters match CI stage 2
    up seedtts /$V $M --thinker.factory.max_seq_len 8192
    python -m benchmarks.eval.benchmark_omni_seedtts --generate-only $S --port 8000 \
        --voice-clone --reference-audio-field audio.ref_audio --max-concurrency 16 --output-dir $OUT/seedtts
    down

    # 3. Scoring. Qwen3-ASR starts from /base (CI also stops MiniCPM-o before it starts ASR)
    up asr /base --model-path Qwen/Qwen3-ASR-1.7B --model-name Qwen/Qwen3-ASR-1.7B
    python -m benchmarks.eval.benchmark_omni_seedtts --transcribe-only $S --port 8000 \
        --asr-concurrency 4 --output-dir $OUT/seedtts
    python - $OUT/videoamme <<'PY'   # the same function CI stage 10 calls
import json, sys
from benchmarks.metrics.wer import print_wer_summary
from benchmarks.tasks.asr import compute_text_audio_consistency_from_records
d = sys.argv[1]
records = json.load(open(f"{d}/videoamme_results.json"))["per_sample"]
wer = compute_text_audio_consistency_from_records(records, "en", "cuda:0", asr_router_port=8000, asr_concurrency=4)
print_wer_summary(wer["summary"], "minicpmo")
json.dump(wer["summary"], open(f"{d}/wer_summary.json", "w"), indent=2)
PY
    down
    python -m benchmarks.eval.benchmark_omni_seedtts --similarity-only $S --output-dir $OUT/seedtts
    python -m benchmarks.eval.benchmark_omni_seedtts --utmos-only $S --output-dir $OUT/seedtts
  done
done
```

### Step 4: other CI stages

Step 4 runs in its own two GPU container. Run the Environment block there first.

```bash
# 4. The other stages use the CI pytest files as they are. Before a PR, run them three times each in /base and /cand, with compile off as in CI
# One container per two GPUs: GPU 0,1, 2,3, 4,5 and 6,7. Set TASK_CI_CPUSET to the union of the cpusets of those two GPUs (the loop passes it on as OMNI_CI_CPUSET)
# Do not run this at the same time as the single GPU runs above
for CI_RUN in 1 2 3; do
  CI_ORDER="base cand"
  if [ "$CI_RUN" = 2 ]; then
    CI_ORDER="cand base"
  else
    :
  fi
  for V in $CI_ORDER; do
    ( cd "/$V" && . .venv/bin/activate
      export OMNI_CI_MODEL=minicpmo SGLANG_OMNI_TORCH_COMPILE_DEFAULT=0 CUDA_VISIBLE_DEVICES=0,1
      export OMNI_CI_CPUSET="${TASK_CI_CPUSET:?set TASK_CI_CPUSET to the cpuset of the two GPUs first}"
      export OMNI_CI_HOME="$HOME/omni_ci"
      SGLANG_OMNI_ROUTER_BIN=$(bash .github/scripts/prepare_rust_router.sh build) || exit 1
      export SGLANG_OMNI_ROUTER_BIN
      for T in thinker_length tts_ci mmmu_ci mmmu_talker_ci mmsu_ci mmsu_talker_ci \
               videomme_ci videomme_talker_ci videoamme_ci videoamme_talker_tp2_ci; do
        CI_RESULTS="results/$(git rev-parse --short HEAD)/ci_run$CI_RUN"
        mkdir -p "$CI_RESULTS"
        python -m pytest "tests/test_model/test_qwen3_omni_$T.py" -v -s -x > "$CI_RESULTS/$T.log" 2>&1
        TEST_EXIT_CODE=$?
        printf '%s\n' "$TEST_EXIT_CODE" > "$CI_RESULTS/$T.exitcode"
      done
    ) || exit 1
  done
done
```

## Reading the results

Read every metric from JSON. Steps 1 to 3 write under `/base/results/<commit>/run<N>/`:

| Workload | Files |
|---|---|
| Video-AMME Talker | `summary` and `speed` in `videoamme/videoamme_results.json`, and `videoamme/wer_summary.json` |
| seed-tts | `seedtts/speed_results.json`, `seedtts/wer_results.json`, `seedtts/similarity_results.json` and `seedtts/utmos_results.json` |

Step 4 writes `<test>.log` and `<test>.exitcode` for each stage under `/<checkout>/results/<commit>/ci_run<N>/`.

There are no numbers yet for one GPU at concurrency 16. These results come from other setups and are for reference only:

- **Video-AMME Talker.** [#2580](https://github.com/sgl-project/sglang-omni/pull/2580) (main `5ed8a8d`), one H200, concurrency 8, 50 questions: Accuracy 0.68, 0.33 req/s. CI stage 10 uses two H100s, DP 2, compile off, the first 10 questions and concurrency 16. In [#2552](https://github.com/sgl-project/sglang-omni/pull/2552) it ran 7 times: 0.643 to 0.668 qps and Accuracy 0.7 every time. WER was 0.004032 in 6 runs and 0.008065 in one, a difference of one word.
- **seed-tts.** [#2532](https://github.com/sgl-project/sglang-omni/pull/2532), one H200, the CI server parameters, concurrency 16, 1088 voice clone samples: 14.22 req/s, WER corpus (excl >50%) 1.47%, SIM 49.78. That run used MPS, which stays off here.

## Where to start

The profiling skill is `.claude/skills/model-profiling`. Read [#2580](https://github.com/sgl-project/sglang-omni/pull/2580) first. It is the Video-AMME profile: at concurrency 8, CPU preprocessing is serial at about 3.0 s per request, and the GPU is busy only 17.1% to 27.6% of the time. Then read [#2273](https://github.com/sgl-project/sglang-omni/pull/2273) and [#2399](https://github.com/sgl-project/sglang-omni/pull/2399) (TTS and code2wav), and [#2284](https://github.com/sgl-project/sglang-omni/pull/2284), the optimization tracker, which lists the areas already claimed.

Related open sglang-omni PRs are #2316, #2480, #2487 and #2589 (preprocessing), #2529, #2528, #2330 and #2353 (talker), and #2356 (HiFT). Build on existing work. Copying a change from an existing PR does not count.

## Modes

**Mode 2** (optimization without an outside reference) is open now. The target workload gets better, and the other workloads and stages do not regress, as defined in [Acceptance](#acceptance).

**Mode 1** (measured against vllm-omni) is not open yet. vllm-omni supports MiniCPM-o 4.5, but its request format differs. Video and audio go in `video_url` and `audio_url` inside `messages`, and the speech is in `choices[1]`. How the video is packed also has a large effect on accuracy. vllm-omni changed only the packing, and MiniCPM-o 4.5 on Daily-Omni went from 66.75% to 78.45% (vllm-omni [#5293](https://github.com/vllm-project/vllm-omni/pull/5293), [#5606](https://github.com/vllm-project/vllm-omni/pull/5606) and [#5625](https://github.com/vllm-project/vllm-omni/pull/5625)). So Video-AMME Talker is Mode 2 only.

seed-tts is the most likely first Mode 1 workload. vllm-omni's perf CI (`tests/dfx/perf/tests/test_minicpmo_4_5.json`) uses it, but at concurrency 1, 4 and 8 with 32, 64 and 128 requests, so its numbers are not a direct target. Mode 1 for seed-tts opens once a fixed vllm-omni client exists and has numbers from the same GPU and the same CPU group.

## Acceptance

Small samples are for quick trials only. A version is accepted only after the full evaluation. At the start of the task, run the baseline once and take the mean of RUN 1 to 3. Every later comparison with the baseline uses this mean.

Measure each candidate again alongside the last accepted version, alternating the two in the same allocation. Never reuse numbers stored from an earlier run. Before measuring, name the target workload in the commit message. It cannot change after the measurement.

### Correctness, against the baseline

- Every run has 0 failed requests. WER, SIM and UTMOS cover every sample, with 0 skipped.
- No audio output is empty or has NaN values. Each one has the right sample rate and was written during this run.
- A missing JSON field counts as a failure.
- Video-AMME Talker mean Accuracy is not below the baseline. There is no tolerance.
- Prompt tokens match the baseline for each sample ID.
- Scoring uses the full corpus WER (`wer_corpus`), with no samples dropped. WER corpus (excl >50%) (`wer_below_50_corpus`) is only for comparison with CI and for debugging.
- The number of >50% WER samples (`n_above_50_pct_wer`) is not above the baseline.
- WER is at most max(baseline × 1.25, baseline + 0.005). 0.005 is 0.5 percentage points.
- SIM and UTMOS are at least baseline × 0.97.
- A change of more than 5% in Audio duration mean needs a look. Also report audio throughput (`audio_throughput_s_per_s`) to check whether a req/s gain comes from shorter speech.

### Performance, against the last accepted version

- The mean Throughput (req/s) of the target workload is at least 1.02 × the last accepted version.
- The Throughput of the other workload is at least 0.97 × the last accepted version and 0.97 × the baseline.
- For both workloads, latency mean, latency p95 and RTF mean are at most 1.03 × each of those two versions.

The 0.97 and 1.03 bounds are the same as in the TTS hill climbing task. Checking against both versions stops many small regressions from adding up to a large one.

### Other CI stages

Before a PR, run the 10 stages of step 4 three times each on the baseline and the candidate. Step 4 runs pytest without the CI retry wrapper (`.github/scripts/run_flaky_pytest.sh`), so it is stricter than CI. A stage that fails only sometimes on the baseline counts as failing on the baseline.

- A stage that passes all three times on the baseline must pass all three times on the candidate, with mean Accuracy not below the baseline.
- For a stage that already fails on the baseline, record why, on which samples and with which metric values. After the maintainers review that record, the stage no longer blocks. The candidate still must not be worse: no more failures and no fewer samples.
- Mark a stage without a valid baseline as unverified.

The 1.02 factor and the tolerances above are provisional. They will be tuned after baseline and A/A runs on the target machine. The A/A in #2580 used concurrency 8 and 50 questions and does not apply. Until then, report a candidate whose quality drops but stays within tolerance to the maintainers. It is not accepted automatically.

## Rules for PRs

- **Default on.** An optimization is on by default. If one method beats another with no tradeoff, make it the default and remove the old path where possible. Optional code paths behind flags make the repository grow until nobody can read it. Default on also means the evaluation commands stay the same.
- **Fixed evaluation.** A PR does not change the evaluation commands, the evaluation and metric code under `benchmarks/`, or the CI under `.github/` and `tests/test_model/`. Adding or changing the matching unit tests under `tests/unit_test/` is fine.
- **Local benchmark fixes.** If a command does not run, you may patch the benchmark locally so it runs. The patch changes only how it runs, never what it measures: inputs, request parameters, samples and scoring stay the same. Save the diff and rerun the baseline three times with it. After that, ref and cand both use the patched benchmark with no further changes. Keep the patch out of the PR.

## What does not count

| Category | Examples |
|---|---|
| Changing the measurement | Client, runner, metrics and request parameters (fps 2, 128 frames, 401408 pixels, `max_tokens` 256, temperature, question count, concurrency) |
| Changing the input | Resize interpolation and count, pixel dtype, video decoding, audio resampling and features, reference audio handling |
| Changing audio quality or precision | code2wav `n_timesteps=10` and `inference_cfg_rate=0.7`, moving any part to FP8, INT8, BF16 or FP16, turning on TF32 |
| Shorter outputs | Chat template, default thinker and talker sampling parameters, maximum length, EOS |
| Caches and special cases for the evaluation | Reusing final outputs across requests, cache keys from file path, sample ID or prompt text, caches kept across restarts, reading files before they are requested, special branches for Video-AMME or seed-tts |
| More resources | A second GPU, MPS, changing CPU affinity, more threads than the given CPUs. #2399 moved code2wav to a second GPU on H20 and gained 35% audio throughput at concurrency 64. That is not a runtime optimization |

A change that alters numerics or has a tradeoff (for example a kernel swap or a new batch composition that loses questions) can go in its own PR, with an accuracy comparison for each sample. The maintainers decide at review. The automatic loop does not accept it.

## Notes

- **The first start is slow.** Torch compile is on by default. In #2580 on one H200 (28 CPUs), a server with an empty cache took about 10.5 minutes to become healthy, and about 5.4 to 5.8 minutes after that.
- **Every run starts a fresh server.** A second pass on the same server is faster because of the encoder cache and the radix prefix cache.
- **CI numbers are not the standard.** CI turns compile off and uses two GPUs, so its numbers and thresholds do not apply to one GPU with compile on.
- **Any reproducible optimization is welcome.** Open a PR against sglang-omni. The maintainers review and reproduce every PR. MiniCPM-o CI runs only when the PR has the `run-minicpmo` label, so ask for it in the PR description and a maintainer adds it.
- **Hardware and focus.** The maintainers mostly use H100 and H200. Most gains so far come from the SGLang-Omni runtime. Kernel work is rarely involved.
