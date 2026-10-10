# ASR hill climbing runbook

This page is for hill climbing ASR serving in SGLang-Omni: making it faster against a fixed evaluation while correctness stays the same. The workload is simple. Audio goes in and text comes out. Correctness is WER (CER and cpCER for MOSS-Transcribe-Diarize). Performance is throughput, latency and RTF.

The work must keep both stages of the ASR CI (`.github/workflows/test-asr-ci.yaml`) passing.

## What is measured

| Stage | When it runs | Model | Data | Concurrency |
|---|---|---|---|---|
| 1 | Every PR | MOSS-Transcribe-Diarize, multiple speakers | `movies800times` (800 samples, run once without and once with streaming), `aishell4_long` (20), `googletime` (25) | 16 |
| 2 | Every PR, one model picked at random | Qwen3-ASR-1.7B, Whisper-large-v3 or Fun-ASR-Nano | SeedTTS EN (1088) and ZH (2020) | 32 |

Stage 1 also has `aishell4_long90`, one 90 minute sample stitched together from `aishell4_long` clips. It only runs through pytest (see [Before opening a PR](#before-opening-a-pr)).

CI runs on two H100s (Rust router with DP 2). Here one server on one GPU is enough. The datasets and concurrency stay the same as CI.

## Prior work

Read this before you start. For profiling, use the skill in `.claude/skills/model-profiling`. Read the existing profiles: [#1887](https://github.com/sgl-project/sglang-omni/issues/1887) (Qwen3-ASR), [#1888](https://github.com/sgl-project/sglang-omni/issues/1888) (Whisper), [#1941](https://github.com/sgl-project/sglang-omni/issues/1941) (Fun-ASR), [#1886](https://github.com/sgl-project/sglang-omni/issues/1886) (MOSS-Transcribe-Diarize), [#1324](https://github.com/sgl-project/sglang-omni/issues/1324) (Qwen3-ASR high concurrency roadmap) and the [Qwen3-ASR concurrency profile](qwen3_asr_concurrency_profile.md). Also read the open PRs: [#2609](https://github.com/sgl-project/sglang-omni/pull/2609) (Qwen3-ASR GC tail latency), [#2333](https://github.com/sgl-project/sglang-omni/pull/2333) (MOSS-Transcribe-Diarize decoder compile off by default), [#2446](https://github.com/sgl-project/sglang-omni/pull/2446) (MOSS-Transcribe-Diarize prefill and decode split) and [#2296](https://github.com/sgl-project/sglang-omni/pull/2296) (Fun-ASR context length). Copying changes from an existing PR does not count.

## Run the benchmark

The block below sets up the environment and runs the full loop. Comments explain each part. It uses the CI image, three checkouts and a fixed CPU group per GPU.

```bash
# For more setup detail, read docs/cookbook/ and .github/workflows/test-asr-ci.yaml
# CI image hongccc/sglang-omni@sha256:ebe4239e29a764ee3a2806385c061c5fd438a26f01458e503d3822dcba5790df, one container per GPU
# CPUs: group i comes from the NUMA node of GPU i (nvidia-smi topo -m). Skip cores 0 and 1, take 14 physical cores in order from the lowest numbered one, plus their SMT siblings, 28 CPUs in total
# Cut as many groups as fit, at most 8, with no overlap. Write the split to cpusets.txt and keep it for the whole task. Start each container with --gpus device=i --cpuset-cpus <group i>

# Three checkouts: /base is the baseline, /ref is the last accepted version, /cand is the candidate. The client always uses the /base venv
git clone https://github.com/sgl-project/sglang-omni.git /base && git -C /base checkout 921ea2c83acbfd7e9247ff38d63d8963b7572b7d
git -C /base worktree add --detach /ref 921ea2c83acbfd7e9247ff38d63d8963b7572b7d
git -C /base worktree add --detach /cand 921ea2c83acbfd7e9247ff38d63d8963b7572b7d
# After this, check out the last accepted version in /ref and the candidate commit in /cand
for t in /base /ref /cand; do
  ( cd $t && uv venv --system-site-packages .venv -p /usr/bin/python3.12 &&   # follows .github/scripts/prepare_omni_venv.sh
    echo 'import site; site.addsitedir("/opt/sglang/lib/python3.12/site-packages")' > .venv/lib/python3.12/site-packages/sglang-image.pth &&
    . .venv/bin/activate && python .github/scripts/omni_missing_dependencies.py pyproject.toml | xargs -r -d '\n' -n 1 python -m pip install &&
    python .github/scripts/omni_missing_dependencies.py --overrides pyproject.toml | xargs -r -d '\n' python -m pip install --no-deps &&
    uv pip install --no-deps -e . )
done
diff <(/base/.venv/bin/python -m pip freeze --exclude-editable) <(/cand/.venv/bin/python -m pip freeze --exclude-editable)   # must be empty, a candidate may not change dependencies
cd /base && . .venv/bin/activate

python -m benchmarks.dataset.prepare --dataset seedtts   # 27f4c1adee83b5b29b7c4b375f6b976324bda308
hf download Qwen/Qwen3-ASR-1.7B                          # 7278e1e70fe206f11671096ffdd38061171dd6e5
hf download openai/whisper-large-v3                      # 06f233fe06e710322aca913c1bc4249a0d71fce1
hf download FunAudioLLM/Fun-ASR-Nano-2512-hf             # 7ef6cafdc445b5beff0b730cbeb51952c793ca6e
hf download OpenMOSS-Team/MOSS-Transcribe-Diarize        # 704aa4a9c304e8520be88901e0d1960158ef5b15
# These are the revisions in the 2026-10-09 CI logs. If ~/.cache/huggingface/hub/models--*/refs/main differs, stop and report it
# The three MOSS-Transcribe-Diarize datasets are private. Download them once with an HF_TOKEN that has access. Their revision is not pinned, so record refs/main and keep it from then on
export HF_HUB_OFFLINE=1 SGLANG_OMNI_STRICT_PORT=1

# Start and stop example. Add your own port checks, cleanup after errors and a timeout on GPU memory release
up() {  # up <log name> <checkout> <serve args>, starts a server and waits for /health
  setsid "$2/.venv/bin/sgl-omni" serve "${@:3}" --port 8000 > "$OUT/$1.log" 2>&1 &
  PID=$!
  timeout 1800 bash -c 'until curl -sf localhost:8000/health > /dev/null; do sleep 5; done' || exit 1
}
down() {  # kills the whole process group and waits until GPU memory is below 1 GiB
  kill -- "-$PID"
  wait "$PID" || true
  until [ $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits) -lt 1024 ]; do sleep 5; done
}

# At task start, run once with VERSIONS=base. For every candidate after that, use VERSIONS="ref cand", which alternates the two versions
# Every RUN starts a fresh server for each version. RUN 0 is not scored (see Cold start below). RUN 1 to 3 are scored
# If only one model directory under sglang_omni/models changed, run only that model. Otherwise run all of them
# For one SeedTTS model, set MODELS to it and skip part 2. For MOSS-Transcribe-Diarize only, set MODELS=""
# Results go to results/<version>-<commit>/, so /base and /ref at the same commit do not overwrite each other
MODELS="Qwen/Qwen3-ASR-1.7B openai/whisper-large-v3 FunAudioLLM/Fun-ASR-Nano-2512-hf"
VERSIONS=${VERSIONS:-base}
for RUN in 0 1 2 3; do
  ORDER=$VERSIONS
  if [ "$VERSIONS" = "ref cand" ] && [ "$RUN" = 2 ]; then
    ORDER="cand ref"
  else
    :
  fi
  for V in $ORDER; do
    # 1. SeedTTS ASR
    for MODEL in $MODELS; do
      OUT=results/$V-$(git -C /$V rev-parse --short HEAD)/run$RUN/${MODEL##*/}; mkdir -p $OUT
      up server /$V --model-path $MODEL --model-name $MODEL
      for LANG in en zh; do
        python -m benchmarks.eval.benchmark_asr_seedtts --port 8000 --model-path $MODEL --lang $LANG \
            --concurrencies 32 --repeats 1 --warmup \
            --dataset-revision 27f4c1adee83b5b29b7c4b375f6b976324bda308 --output $OUT/asr_$LANG.json
      done
      down
    done
    # 2. MOSS-Transcribe-Diarize. Server args match the CI launcher log, with a fresh server for each dataset
    for D in "movies800times" "movies800times --stream" "aishell4_long" "googletime"; do
      OUT=results/$V-$(git -C /$V rev-parse --short HEAD)/run$RUN/moss_td/${D// /_}; mkdir -p $OUT
      up server /$V --model-path OpenMOSS-Team/MOSS-Transcribe-Diarize --model-name OpenMOSS-Team/MOSS-Transcribe-Diarize \
          --asr.engine.max_running_requests 16 --asr.engine.cuda_graph_max_bs 16 --mem-fraction-static 0.8
      python -m benchmarks.eval.benchmark_asr_transcribe_diarize --use-existing-server --port 8000 \
          --dataset $D --max-concurrency 16 --output-dir $OUT
      down
    done
  done
done
```

After every new checkout in `/ref` or `/cand`, rerun the venv loop and the `pip freeze` diff for that checkout.

Each script sends the requests and computes the scores. In the SeedTTS client, `--warmup` first sends the full dataset once and does not score it. The MOSS-Transcribe-Diarize runs have no warmup pass. CI warms up with only 64 requests and runs on two GPUs, so numbers from this loop are never compared with the CI gates. They are compared only with your own baseline.

## Results

Read every metric at full precision from the JSON files, not from the printed lines.

| Workload | File | Fields |
|---|---|---|
| SeedTTS | `asr_*.json` | `results[0]`, take `.mean` of each metric, plus `evaluated`, `total` and `skipped` |
| MOSS-Transcribe-Diarize coverage | `transcribe_diarize_results.json` | `summary` |
| MOSS-Transcribe-Diarize CER, cpCER, valid sample counts, speaker timestamp DER | `transcribe_diarize_results.json` | `diarization_metrics_percent`. These are percentages, so divide by 100 before comparing |
| MOSS-Transcribe-Diarize performance | `transcribe_diarize_speed_results.json` | `speed`, including `failed_requests` |

The valid sample IDs are the `per_sample` entries of `transcribe_diarize_results.json` whose `cer_valid` and `cp_cer_valid` are true.

Output looks roughly like this ([CI run 37914129116](https://github.com/sgl-project/sglang-omni/actions/runs/37914129116), 2026-10-09, Qwen3-ASR EN, two H100s):

```text
============================================================
                  ASR WER Benchmark Result
============================================================
  ASR model:                     Qwen/Qwen3-ASR-1.7B
  Language:                      en
  Evaluated / Total:             1088/1088
  WER (corpus, micro-avg):       0.0120 (1.20%)
  Latency mean (s):              0.1250194393312139
  Latency p95 (s):               0.16246931133209724
  RTF mean:                      0.027117304365458155
  ASR concurrency:               32
  ASR throughput (samples/s):    251.11287530943812
============================================================
```

For one GPU, the docstring at the top of `benchmarks/eval/benchmark_asr_seedtts.py` gives these numbers on one H100 at concurrency 32: Qwen3-ASR 97.9 samples/s with WER 0.0122, and Fun-ASR about 127.5 samples/s with WER 0.0171.

MOSS-Transcribe-Diarize on `movies800times` (CI job 113922224972, attempt 2, two H100s): CER 5.82%, cpCER 13.13%, 784 CER valid samples, throughput 58.167 req/s, latency mean and p95 0.273 s and 0.625 s, RTF mean 0.0269.

## Acceptance rules

The goal is that performance and correctness only move forward.

**Fixed setup.** The baseline is main at `921ea2c8` and does not follow main. Once the loop runs, the revisions of the model, tokenizer, datasets, scoring resources and dependencies never change. Each group keeps one GPU and one set of 28 CPUs for the whole task. Every number is the mean of RUN 1 to 3. While exploring you may use a small sample (for example `--max-samples`) to find a direction. Run the full loop before accepting a version or opening a PR.

**Paired measurement.** Run the baseline once at the start. After that, measure each candidate against the last accepted version on the same GPU and CPU group, in one paired loop. Never reuse stored numbers for the last accepted version. Before measuring, name the target workload in the commit message (one model's EN or ZH, or one MOSS-Transcribe-Diarize dataset). Do not change it after measuring.

**Correctness** is compared with the baseline only:

- Every run covers the full set (1088/1088, 2020/2020, 800/800 and so on) with zero failed requests. For SeedTTS, `evaluated` equals `total` and `skipped` is 0. For MOSS-Transcribe-Diarize, `speed.failed_requests` is 0.
- SeedTTS `corpus_wer` is at most baseline + 0.0002. Across 5 CI jobs on the same model, EN WER differed by at most 0.0001.
- MOSS-Transcribe-Diarize CER and cpCER are at most baseline + 0.001. The two attempts of job 113922224972 gave `movies800times` CER of 5.87% and 5.82%.
- `cer_valid_samples` and `cp_cer_valid_samples` are not below the baseline. Save the IDs of the valid samples and compare the IDs, not only the counts.
- Also save the `speaker_timestamp_der*` fields of `diarization_metrics_percent`, including `speaker_timestamp_der_skipped_parse_error`, to check that diarization still works.

**Performance:**

| Workload | Rule |
|---|---|
| The target workload | Throughput at least 1.02 times the last accepted version |
| Every other workload | Throughput at least 0.97 times both the last accepted version and the baseline. Latency mean, latency p95 and RTF mean at most 1.03 times both |

Comparing against both versions stops small losses from adding up over many steps. These tolerances are provisional. Before the first candidate, run the paired loop with the same source in `/ref` and `/cand` on the target machine (see [Cold start](#cold-start)). Set the tolerances once from that spread, then keep them fixed for the whole task.

**Close calls.** When a result is near a threshold or the runs disagree, repeat the same paired loop in full (RUN 0 to 3) and score both passes together. Keep all earlier results.

## What does not count

These are not optimizations:

- Changing the client, dataset, concurrency, `language`, `response_format` or warmup.
- Changing the input the model sees (resampling, feature extraction, cutting audio).
- Changing decoding defaults (temperature, maximum length, EOS) so outputs get shorter.
- Lower precision for weights, activations or the KV cache, or turning on TF32.
- Reusing any result across requests based on audio content, file path or sample ID (transcripts, audio features, encoder outputs, KV prefixes). In the SeedTTS client, `--warmup` sends the same audio as the scored pass, so such a cache would hit every time.
- Special branches for SeedTTS or `movies800times`.
- Server startup warmup that reads dataset files. It may only use synthetic input.
- Using more resources, such as a second GPU, MPS, a different CPU affinity, or more threads than the assigned CPUs.

## Cold start

Cold start has a large effect. On 2026-10-09, 5 sampled stage 1 jobs reached only 43.2 to 44.7 req/s on `movies800times` in attempt 1, below the 51.877 gate. After the server restarted on attempt 2, they reached 57.0 to 59.1 req/s and passed (attempt 2 of 3). The cause is not known yet. A compile cache or file cache on disk is the likely suspect.

For this reason RUN 0 is not scored. Record the state of both caches before each loop (for example what is in the compile cache directory, and whether the model and dataset files were already in the page cache). RUN 0 only warms the caches, and the cold start problem may still be there. Before accepting a gain, run the paired loop a few times with the same source in `/ref` and `/cand`, and check that the scored rounds are stable.

## Modes

**Mode 1 (with a reference)** compares against vllm-omni. It currently applies to Qwen3-ASR and Whisper-large-v3 only. vLLM itself serves ASR on `/v1/audio/transcriptions` (no `--omni` needed), and `benchmark_asr_seedtts` runs against it without code changes.

1. Install vllm-omni into a separate venv as the [v0.30.0 release](https://github.com/vllm-project/vllm-omni/releases/tag/v0.30.0) describes, on the same machine, GPU and CPU group. Save its `pip freeze`.
2. Start `vllm serve Qwen/Qwen3-ASR-1.7B --port 8000` (`openai/whisper-large-v3` for Whisper). Record the defaults that actually take effect.
3. Send a few requests first and check that both sides match on revision, language, precision, decoding settings, audio input and normalization.
4. Run the client of the "1. SeedTTS ASR" part of the loop. Do not pass `--stream`, because the two servers use different streaming event formats.

Measure the reference values once at task start and keep them fixed. A model moves to Mode 2 if vLLM's WER is more than 0.0002 above the baseline, or if the baseline is already no slower than vLLM. The goal is to match or beat vLLM. Each step is still accepted by the [acceptance rules](#acceptance-rules).

Fun-ASR and MOSS-Transcribe-Diarize do not use Mode 1 for now. Fun-ASR runs on vLLM with a different converted checkpoint (Fun-ASR-Nano-2512-vllm). For MOSS-Transcribe-Diarize, it is not yet confirmed that the speaker labels can be read from `verbose_json`.

**Mode 2 (open)** has no reference and applies to all four models. The same acceptance rules apply.

## Before opening a PR

**Where code goes.** Put new code in the matching model directory under `sglang_omni/models` where possible. Changes to a shared hot path get strict CI review.

**Default on.** An optimization must be on by default.

**Acceptance tests stay unchanged.** A PR does not change the test commands, the evaluation and metric code under `benchmarks/`, or the CI under `.github/` and `tests/test_model/`. These are the acceptance criteria. You may add or change matching unit tests under `tests/unit_test/`, but never loosen a check.

**Local benchmark fixes.** If a command does not run, you may fix the benchmark locally. Change only how it runs, never the inputs, request parameters, sample range or scoring. Save the diff and use it to measure the baseline three times again. After that the script does not change, and the diff does not go into the PR.

**Reproducible.** Any optimization counts if it reproduces. Every PR is reproduced before it is accepted.

**ASR CI.** A PR must pass the ASR CI. The stage 2 pytest calls the functions in `benchmark_asr_seedtts`. On a local machine with two GPUs, run from the checkout that holds the PR branch with its venv active (for example `cd /cand && . .venv/bin/activate`):

```bash
export OMNI_CI_HOME=$HOME/omni_ci
export SGLANG_OMNI_ROUTER_BIN=$(bash .github/scripts/prepare_rust_router.sh build)
ASR_CI_MODEL=qwen3 python -m pytest tests/test_model/test_asr_ci_seedtts.py -v -s -x
python -m pytest tests/test_model/test_asr_ci_multi_speaker.py -v -s -x
```

Keep the two `export` lines separate. Otherwise `prepare_rust_router.sh` cannot read `OMNI_CI_HOME`. Use `ASR_CI_MODEL=fun` or `ASR_CI_MODEL=whisper` for the other models.

Do not wrap local runs in `run_flaky_pytest.sh`. It retries up to three times, so a green result can hide a failure.

State in the PR which model to run. A maintainer then adds one of the `run-qwen3-asr`, `run-fun-asr` or `run-whisper-asr` labels.
