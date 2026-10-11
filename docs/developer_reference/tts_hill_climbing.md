# TTS hill climbing runbook

This page is for hill climbing TTS serving in SGLang-Omni: make the measured metrics better without regressions. Correctness is WER, from transcribing the speech with Qwen3-ASR-1.7B. Performance is throughput, latency and RTF. The code must stay general, but the only workload to protect is the four benchmark commands on this page.

## What is Measured

All four commands use the SeedTTS EN set (`zhaochenyang20/seed-tts-eval-arrow`, 1088 samples), non streaming, at concurrency 16.

| Model | Request shape | Command flags | Output directory name |
|---|---|---|---|
| `Qwen/Qwen3-TTS-12Hz-1.7B-Base` | Voice clone | `--ref-format references` | `qwen3_tts_base_en` |
| `OpenMOSS-Team/MOSS-TTS-v1.5` | Voice clone with a duration target | `--ref-format references --token-count auto` | `moss_tts_en` |
| `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice` | Named voice, no reference audio | `--no-ref-audio --voice Ryan` | `qwen3_tts_customvoice_en` |
| `FunAudioLLM/Fun-CosyVoice3-0.5B-2512` | Voice clone | none | `fun_cosyvoice3_en` |

Each command starts the TTS server itself, sends 16 warmup requests and then the 1088 scored ones, stops the server, starts Qwen3-ASR-1.7B on the same port and computes WER. One GPU is enough.

The server uses the benchmark's engine defaults (`max_running_requests` 64 and `cuda_graph_max_bs` 64), not the model's own serve defaults. `server_logs/tts_server.log` in each output directory shows them.

The work must also keep the [TTS CI](#tts-ci) passing.

## Prior Work

For profiling, use the skill in `.claude/skills/model-profiling`. Read [#1883](https://github.com/sgl-project/sglang-omni/issues/1883) (Fun-CosyVoice3 profile), [#1233](https://github.com/sgl-project/sglang-omni/issues/1233) (MOSS-TTS optimization tracker) and [#1707](https://github.com/sgl-project/sglang-omni/issues/1707) (MOSS-TTS delay sampling CUDA graph).

Merged and already in the baseline: [#2524](https://github.com/sgl-project/sglang-omni/pull/2524) (CosyVoice3 idle speaker encoder workers), [#2555](https://github.com/sgl-project/sglang-omni/pull/2555) (CosyVoice3 reference features between stages), [#2176](https://github.com/sgl-project/sglang-omni/pull/2176) (CosyVoice3 S3 tokenizer cuDNN algorithms), [#2299](https://github.com/sgl-project/sglang-omni/pull/2299) (CosyVoice3 generation length contract) and [#2525](https://github.com/sgl-project/sglang-omni/pull/2525) (MOSS-TTS input embedding accumulator).

Merged after the baseline: [#2632](https://github.com/sgl-project/sglang-omni/pull/2632) (CosyVoice3 DiT conv and Mish Triton kernel), [#2631](https://github.com/sgl-project/sglang-omni/pull/2631) (CosyVoice3 streaming Flow CUDA graphs), [#2662](https://github.com/sgl-project/sglang-omni/pull/2662) (MOSS-TTS reference encoding refactor) and [#2636](https://github.com/sgl-project/sglang-omni/pull/2636) (stage CPU binding). Do not redo them. They come in when you rebase for a PR (Step 15).

Open PRs: [#2443](https://github.com/sgl-project/sglang-omni/pull/2443) (CosyVoice3 streaming HiFT compile), [#2164](https://github.com/sgl-project/sglang-omni/pull/2164) (CosyVoice3 partial QK RoPE), [#1693](https://github.com/sgl-project/sglang-omni/pull/1693) (CosyVoice3 batched S3 tokenizer), [#2625](https://github.com/sgl-project/sglang-omni/pull/2625) (CosyVoice3 token ids through a pinned buffer), [#2626](https://github.com/sgl-project/sglang-omni/pull/2626) (CosyVoice3 repetition aware sampling), [#2449](https://github.com/sgl-project/sglang-omni/pull/2449) (CosyVoice3 same GPU DP), [#2472](https://github.com/sgl-project/sglang-omni/pull/2472) and [#2473](https://github.com/sgl-project/sglang-omni/pull/2473) (Qwen3-TTS admission and vocoder bootstrap), [#2669](https://github.com/sgl-project/sglang-omni/pull/2669) (Qwen3-TTS reference codec state cache), [#2658](https://github.com/sgl-project/sglang-omni/pull/2658) (Qwen3-TTS first chunk decode) and [#2659](https://github.com/sgl-project/sglang-omni/pull/2659) (Qwen3-TTS prompt rows and constants). Copying changes from an existing PR does not count.

## Names Used on This Page

| Name | Meaning |
|---|---|
| Baseline commit | main at `921ea2c83acbfd7e9247ff38d63d8963b7572b7d`. It does not follow main |
| `/work/base`, `/work/ref`, `/work/cand` | Checkouts of the baseline, the last accepted version and the candidate. You edit and commit only in `/work/cand` |
| B0 | The scored runs of `/work/base`, measured once in Step 10 |
| Bk | The last accepted version, measured again beside each candidate. Before the first acceptance, its code is the baseline |
| Pass | One measurement session in `/work/results/<pass name>/`. Never reuse a name |
| RUN 0 to 3 | The runs of a pass. RUN 0 only warms caches. Every number is the mean of RUN 1 to 3 |

## One Time Setup

### Step 1: choose the GPU and CPUs

Use one GPU and one fixed set of 28 CPUs for the whole task. On the host, run:

```bash
nvidia-smi topo -m
lscpu -e=CPU,NODE,CORE
```

In the first output, read the NUMA affinity of GPU 0. In the second, on that node, skip the two lowest numbered physical cores and take the next 14. The CPU set is those cores plus their SMT siblings (same `CORE` value), 28 CPUs. If SMT is off, take 28 physical cores instead.

Set these variables on the host. Replace the `CPUSET` value with the set you found.

```bash
GPU=0
CPUSET=2-15,114-127
WORK=$HOME/tts_hill_climbing
```

### Step 2: start the container

The image is the TTS CI image. Everything under `/work` lives in `$WORK` on the host.

```bash
mkdir -p $WORK $HOME/.cache
docker run -d --name tts-hill-climbing \
    --gpus device=$GPU --cpuset-cpus $CPUSET \
    -v /dev/shm:/dev/shm \
    -v $WORK:/work \
    -v $HOME/.cache:/root/.cache \
    --entrypoint sleep \
    hongccc/sglang-omni@sha256:ebe4239e29a764ee3a2806385c061c5fd438a26f01458e503d3822dcba5790df \
    infinity
docker exec -it tts-hill-climbing bash
```

Every later command runs inside the container. Open another shell with the last line.

### Step 3: write the variables file

Set `GIT_NAME` and `GIT_EMAIL` to the identity you use for repository commits.

```bash
cat > /work/env.sh <<'EOF'
export BASE_COMMIT=921ea2c83acbfd7e9247ff38d63d8963b7572b7d
export COSYVOICE_PATH=/work/CosyVoice
export COSYVOICE_COMMIT=074ca6dc9e80a2f424f1f74b48bdd7d3fea531cc
export SGLANG_OMNI_STRICT_PORT=1
export GIT_NAME="Your Name"
export GIT_EMAIL=you@example.com
EOF
source /work/env.sh
```

Run `source /work/env.sh` in every new shell. Do not set `SGLANG_OMNI_TORCH_COMPILE_DEFAULT`, so torch compile stays on (CI turns it off).

### Step 4: install sox and the CosyVoice sources

Fun-CosyVoice3 and Qwen3-TTS need sox. Fun-CosyVoice3 imports the CosyVoice and Matcha-TTS sources, as in `.github/scripts/prepare_omni_venv.sh`.

```bash
apt-get update && apt-get install -y sox
git clone --filter=blob:none --no-checkout https://github.com/FunAudioLLM/CosyVoice.git $COSYVOICE_PATH
git -C $COSYVOICE_PATH checkout --detach $COSYVOICE_COMMIT
git -C $COSYVOICE_PATH submodule update --init --depth=1 third_party/Matcha-TTS
git -C $COSYVOICE_PATH/third_party/Matcha-TTS rev-parse HEAD
```

The last line must print `dd9105b34bf2be2230f4aa1e4769fb586a3c824e`.

### Step 5: create the three checkouts

```bash
git clone https://github.com/sgl-project/sglang-omni.git /work/base
git -C /work/base checkout --detach $BASE_COMMIT
git -C /work/base worktree add --detach /work/ref $BASE_COMMIT
git -C /work/base worktree add --detach /work/cand $BASE_COMMIT
git -C /work/cand config user.name "$GIT_NAME"
git -C /work/cand config user.email "$GIT_EMAIL"
```

### Step 6: create a venv in each checkout

Run this block three times, with `T=/work/base`, then `T=/work/ref`, then `T=/work/cand`. Later steps say when to run it again.

```bash
T=/work/base
cd $T
rm -rf .venv
uv venv --system-site-packages .venv -p /usr/bin/python3.12
echo 'import site; site.addsitedir("/opt/sglang/lib/python3.12/site-packages")' > .venv/lib/python3.12/site-packages/sglang-image.pth
printf '%s\n' $COSYVOICE_PATH $COSYVOICE_PATH/third_party/Matcha-TTS > .venv/lib/python3.12/site-packages/cosyvoice.pth
source .venv/bin/activate
python .github/scripts/omni_missing_dependencies.py --extra fun-cosyvoice3 pyproject.toml | xargs -r -d '\n' -n 1 python -m pip install
python .github/scripts/omni_missing_dependencies.py --overrides pyproject.toml | xargs -r -d '\n' python -m pip install --no-deps
uv pip install --no-deps -e .
uv pip install --no-deps sox einops
uv pip install --no-deps qwen-tts==0.1.1
deactivate
```

This follows `.github/scripts/prepare_omni_venv.sh`, but installs only the `fun-cosyvoice3` extra. The last two install lines come from `docs/cookbook/qwen3_tts.md`. Never install them with dependencies.

Then check that the three venvs match. Both commands must print nothing.

```bash
diff <(/work/base/.venv/bin/python -m pip freeze --exclude-editable) <(/work/ref/.venv/bin/python -m pip freeze --exclude-editable)
diff <(/work/base/.venv/bin/python -m pip freeze --exclude-editable) <(/work/cand/.venv/bin/python -m pip freeze --exclude-editable)
```

### Step 7: download the data, models and scoring weights

Each model is pinned to its `refs/main` commit on Hugging Face on 2026-10-10.

```bash
cd /work/base
source .venv/bin/activate
python -m benchmarks.dataset.prepare --dataset seedtts
hf download Qwen/Qwen3-TTS-12Hz-1.7B-Base --revision fd4b254389122332181a7c3db7f27e918eec64e3
hf download Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice --revision 0c0e3051f131929182e2c023b9537f8b1c68adfe
hf download OpenMOSS-Team/MOSS-TTS-v1.5 --revision cdd3b911b1585e3f2dbc7775ef10f9926f58850a
hf download OpenMOSS-Team/MOSS-Audio-Tokenizer --revision 3cd226ba2947efa357ef453bcad111b6eafba782
hf download FunAudioLLM/Fun-CosyVoice3-0.5B-2512 --revision 29e01c4e8d000f4bcd70751be16fa94bf3d85a18
hf download Qwen/Qwen3-ASR-1.7B --revision 7278e1e70fe206f11671096ffdd38061171dd6e5
python -m benchmarks.metrics.speaker_similarity_assets --warm-cache
python -m benchmarks.metrics.utmos --warm-cache
deactivate
```

`benchmarks/dataset/prepare.py` pins the dataset revision `27f4c1adee83b5b29b7c4b375f6b976324bda308`. `MOSS-Audio-Tokenizer` is the codec MOSS-TTS-v1.5 loads. The last two lines fetch the SIM and UTMOS weights that TTS CI uses.

Offline, a model name loads the commit in `refs/main`. A download by commit does not write that file, so write it:

```bash
H=/root/.cache/huggingface/hub
mkdir -p $H/models--Qwen--Qwen3-TTS-12Hz-1.7B-Base/refs $H/models--Qwen--Qwen3-TTS-12Hz-1.7B-CustomVoice/refs $H/models--OpenMOSS-Team--MOSS-TTS-v1.5/refs $H/models--OpenMOSS-Team--MOSS-Audio-Tokenizer/refs $H/models--FunAudioLLM--Fun-CosyVoice3-0.5B-2512/refs $H/models--Qwen--Qwen3-ASR-1.7B/refs
echo -n fd4b254389122332181a7c3db7f27e918eec64e3 > $H/models--Qwen--Qwen3-TTS-12Hz-1.7B-Base/refs/main
echo -n 0c0e3051f131929182e2c023b9537f8b1c68adfe > $H/models--Qwen--Qwen3-TTS-12Hz-1.7B-CustomVoice/refs/main
echo -n cdd3b911b1585e3f2dbc7775ef10f9926f58850a > $H/models--OpenMOSS-Team--MOSS-TTS-v1.5/refs/main
echo -n 3cd226ba2947efa357ef453bcad111b6eafba782 > $H/models--OpenMOSS-Team--MOSS-Audio-Tokenizer/refs/main
echo -n 29e01c4e8d000f4bcd70751be16fa94bf3d85a18 > $H/models--FunAudioLLM--Fun-CosyVoice3-0.5B-2512/refs/main
echo -n 7278e1e70fe206f11671096ffdd38061171dd6e5 > $H/models--Qwen--Qwen3-ASR-1.7B/refs/main
```

Then go offline for the rest of the task:

```bash
echo 'export HF_HUB_OFFLINE=1' >> /work/env.sh
source /work/env.sh
```

### Step 8: write the compare script

`python /work/compare.py NEW OLD` prints, for each model and field, the three scored runs of NEW, their mean, the mean of OLD and the ratio of the two means. Optional FILE FIELD limit it to one field, and a SUFFIX after them is added to the model directories of NEW.

```bash
cat > /work/compare.py <<'EOF'
import json
import sys
from pathlib import Path

MODELS = ["qwen3_tts_base_en", "moss_tts_en", "qwen3_tts_customvoice_en", "fun_cosyvoice3_en"]
FIELDS = {
    "speed_results.json": ["throughput_qps", "latency_mean_s", "latency_median_s", "latency_p95_s",
                           "latency_p99_s", "rtf_mean", "rtf_median", "rtf_p95", "rtf_p99",
                           "completed_requests", "failed_requests", "max_token_hits", "audio_duration_mean_s"],
    "wer_results.json": ["wer_corpus", "evaluated", "skipped", "n_above_50_pct_wer"],
}

def runs(version_dir, model, file, field):
    return [json.loads((Path(version_dir) / f"run{n}" / model / file).read_text())["summary"][field] for n in (1, 2, 3)]

new_dir, old_dir = sys.argv[1], sys.argv[2]
if len(sys.argv) > 3:
    FIELDS = {sys.argv[3]: [sys.argv[4]]}
suffix = sys.argv[5] if len(sys.argv) > 5 else ""
for model in MODELS:
    print(f"\n{model}\n  {'field':24}{'run1':>10}{'run2':>10}{'run3':>10}{'mean':>10}{'old mean':>10}{'ratio':>8}")
    for file, fields in FIELDS.items():
        for field in fields:
            new = runs(new_dir, model + suffix, file, field)
            new_mean, old_mean = sum(new) / 3, sum(runs(old_dir, model, file, field)) / 3
            ratio = f"{new_mean / old_mean:8.3f}" if old_mean else "     n/a"
            print(f"  {field:24}" + "".join(f"{v:10.4g}" for v in new) + f"{new_mean:10.4g}{old_mean:10.4g}{ratio}")
EOF
```

## Run the Benchmark

### Step 9: one run of the four commands

Only the first three lines of this block change. Set them to the values the calling step gives: `PASS` is the pass name, `V` is the checkout (`base`, `ref` or `cand`) and `RUN` is 0, 1, 2 or 3. Always run all four models in this order, including the ones the candidate does not touch.

```bash
PASS=b0
V=base
RUN=0
OUT=/work/results/$PASS/$V/run$RUN
cd /work/$V
source .venv/bin/activate
python -m benchmarks.eval.benchmark_tts_seedtts --meta zhaochenyang20/seed-tts-eval-arrow \
    --model Qwen/Qwen3-TTS-12Hz-1.7B-Base --port 8000 --ref-format references \
    --output-dir $OUT/qwen3_tts_base_en --lang en --max-concurrency 16
python -m benchmarks.eval.benchmark_tts_seedtts --meta zhaochenyang20/seed-tts-eval-arrow \
    --model OpenMOSS-Team/MOSS-TTS-v1.5 --port 8000 --ref-format references --token-count auto \
    --output-dir $OUT/moss_tts_en --lang en --max-concurrency 16
python -m benchmarks.eval.benchmark_tts_seedtts --meta zhaochenyang20/seed-tts-eval-arrow \
    --model Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice --port 8000 --no-ref-audio --voice Ryan \
    --output-dir $OUT/qwen3_tts_customvoice_en --lang en --max-concurrency 16
python -m benchmarks.eval.benchmark_tts_seedtts --meta zhaochenyang20/seed-tts-eval-arrow \
    --model FunAudioLLM/Fun-CosyVoice3-0.5B-2512 --port 8000 \
    --output-dir $OUT/fun_cosyvoice3_en --lang en --max-concurrency 16
deactivate
```

Each command starts and stops its own servers, so every run starts fresh. The TTS server, the client and Qwen3-ASR all run on the checkout under test.

If a command fails, do not score that pass. Redo the whole pass under its name plus `b` (for example `c01b`), from `mkdir`. If it fails again on `cand` only, reject the candidate (Step 12, step 9). If it fails on `base` or `ref`, stop and report it.

### Step 10: measure the baseline B0

Do this once, at the start.

1. Create the pass directory and the record of accepted versions. `mkdir` fails if the pass name was used before.

   ```bash
   mkdir -p /work/results
   mkdir /work/results/b0
   echo "b0 $BASE_COMMIT /work/results/b0/base" > /work/accepted.txt
   ```

2. Run Step 9 four times with `PASS=b0` and `V=base`: first `RUN=0`, then `RUN=1`, `RUN=2` and `RUN=3`.
3. Print B0. Both paths are the same, so every ratio is 1.

   ```bash
   python /work/compare.py /work/results/b0/base /work/results/b0/base
   ```

4. Check B0 against the correctness rules in [Acceptance Rules](#acceptance-rules). If B0 fails the 0.02 WER line for a model, stop and report it. Fun-CosyVoice3 once measured 0.0183, so check all three runs.

`/work/accepted.txt` has one line per accepted version: pass name, commit and result directory. The last line is always Bk.

### Step 11: check the noise once

Do this once, before the first candidate, while all checkouts hold the baseline commit.

1. Run a paired pass (Step 13) three times, with `PASS=noise1`, then `PASS=noise2`, then `PASS=noise3`.
2. Compare each pass with the first one:

   ```bash
   python /work/compare.py /work/results/noise1/cand /work/results/noise1/ref
   python /work/compare.py /work/results/noise2/cand /work/results/noise1/ref
   python /work/compare.py /work/results/noise3/cand /work/results/noise1/ref
   ```

If any throughput, latency or RTF ratio is below 0.97 or above 1.03, stop and report it to the maintainers. The bounds would then pass or fail on noise.

### Step 12: measure a candidate

1. Commit the change in `/work/cand`. The message starts with the one target model directory, which never changes after measuring.

   ```bash
   git -C /work/cand add -A
   git -C /work/cand commit -m "fun_cosyvoice3: <one line that describes the change>"
   ```

2. Check that `benchmarks/` and `pyproject.toml` match the baseline. Each command must print nothing.

   ```bash
   git -C /work/cand status --short
   git -C /work/cand diff --stat $BASE_COMMIT -- benchmarks pyproject.toml
   ```

3. Run Step 6 with `T=/work/cand`, then the second `pip freeze` diff of Step 6. It must print nothing.
4. Set the pass name and the result path for the fresh Bk comparison. Use `c01` for the first candidate, `c02` for the second, and so on.

   ```bash
   PASS=c01
   BK=/work/results/$PASS/ref
   ```

5. Run the paired pass in Step 13 with this `PASS`. Both versions start fresh for every run, and RUN 1 to 3 are scored.
6. Compare with Bk, then with B0:

   ```bash
   python /work/compare.py /work/results/$PASS/cand $BK
   python /work/compare.py /work/results/$PASS/cand /work/results/b0/base
   ```

7. Apply the [Acceptance Rules](#acceptance-rules) to both comparisons. Keep all runs. Do not replace a scored run because its result is unfavorable.
8. Check for shared code, which the client and Qwen3-ASR also import (for example `sglang_omni.admission`). Use the target model directory in place of `fun_cosyvoice3`:

   ```bash
   git -C /work/cand diff --stat $(git -C /work/ref rev-parse HEAD) -- . ':!sglang_omni/models/fun_cosyvoice3'
   ```

   If it prints anything, also score the audio from `/work/base`. Score a copy, because `generated.json` holds absolute paths to the original audio. Run this block with `R=.../run1`, then `run2`, then `run3`:

   ```bash
   R=/work/results/$PASS/cand/run1
   cp -r $R/qwen3_tts_base_en $R/qwen3_tts_base_en.base_asr
   cp -r $R/moss_tts_en $R/moss_tts_en.base_asr
   cp -r $R/qwen3_tts_customvoice_en $R/qwen3_tts_customvoice_en.base_asr
   cp -r $R/fun_cosyvoice3_en $R/fun_cosyvoice3_en.base_asr
   cd /work/base
   source .venv/bin/activate
   python -m benchmarks.eval.benchmark_tts_seedtts --model Qwen/Qwen3-TTS-12Hz-1.7B-Base --output-dir $R/qwen3_tts_base_en.base_asr --transcribe-only
   python -m benchmarks.eval.benchmark_tts_seedtts --model OpenMOSS-Team/MOSS-TTS-v1.5 --output-dir $R/moss_tts_en.base_asr --transcribe-only
   python -m benchmarks.eval.benchmark_tts_seedtts --model Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice --output-dir $R/qwen3_tts_customvoice_en.base_asr --transcribe-only
   python -m benchmarks.eval.benchmark_tts_seedtts --model FunAudioLLM/Fun-CosyVoice3-0.5B-2512 --output-dir $R/fun_cosyvoice3_en.base_asr --transcribe-only
   deactivate
   ```

   WER needs only the saved audio, so other flags are left out. The mean `wer_corpus` of the copies must also pass the WER rule:

   ```bash
   python /work/compare.py /work/results/$PASS/cand $BK wer_results.json wer_corpus .base_asr
   ```

9. If the candidate passes every rule, go to Step 14. If not, reject it:

   ```bash
   git -C /work/cand reset --hard $(git -C /work/ref rev-parse HEAD)
   ```

### Step 13: a paired pass

A paired pass runs `/work/ref` and `/work/cand` in one pass, alternating. Every candidate uses it, as do Steps 11 and 15.

1. Check the commits. Outside Step 15, the first line must match the commit on the last line of `/work/accepted.txt`.

   ```bash
   git -C /work/ref rev-parse HEAD
   git -C /work/cand log -1 --oneline
   ```

2. Create the pass. Replace `c01` with the name the calling step gives.

   ```bash
   PASS=c01
   mkdir /work/results/$PASS
   ```

3. Run Step 9 eight times with this `PASS`, with `V` and `RUN` in exactly this order: `ref 0`, `cand 0`, `ref 1`, `cand 1`, `cand 2`, `ref 2`, `ref 3`, `cand 3`.
4. Compare the candidate with the `ref` side of the same pass:

   ```bash
   python /work/compare.py /work/results/$PASS/cand /work/results/$PASS/ref
   ```

### Step 14: accept a candidate

When a candidate passes every rule:

1. Record it. `PASS` is the paired pass the candidate was judged on, such as `c01`. Its commit is the new Bk. The stored results are evidence, and the next candidate measures this version again. The second line keeps the ratios to B0.

   ```bash
   echo "$PASS $(git -C /work/cand rev-parse HEAD) /work/results/$PASS/cand" >> /work/accepted.txt
   python /work/compare.py /work/results/$PASS/cand /work/results/b0/base > /work/results/$PASS/ratio_to_b0.txt
   ```

2. Move `/work/ref` to the accepted commit:

   ```bash
   git -C /work/ref checkout --detach $(git -C /work/cand rev-parse HEAD)
   ```

3. Run Step 6 with `T=/work/ref`.

The next candidate is a new commit on top of this one.

## Results

Read every metric from the JSON files in `/work/results/<pass>/<version>/run<N>/<model directory>/`, not from the printed lines.

| Metric | File | Field |
|---|---|---|
| Throughput | `speed_results.json` | `summary.throughput_qps` |
| Latency | `speed_results.json` | `summary.latency_mean_s`, `latency_median_s`, `latency_p95_s`, `latency_p99_s` |
| RTF | `speed_results.json` | `summary.rtf_mean`, `rtf_median`, `rtf_p95`, `rtf_p99` |
| Requests, length cap hits, audio length | `speed_results.json` | `summary.completed_requests`, `failed_requests`, `max_token_hits`, `audio_duration_mean_s` |
| WER | `wer_results.json` | `summary.wer_corpus`, `evaluated`, `skipped`, `n_above_50_pct_wer` |
| Per sample transcript and WER | `wer_results.json` | `per_sample` |

SIM and UTMOS are not scored here. TTS CI gates them (`similarity_mean_min`, `utmos_mean_min`). The CI references in `tests/test_model/tts_ci_config.py` come from a different deployment. Compare only with your own B0 and Bk.

## Acceptance Rules

Performance and correctness only move forward.

**Fixed setup.** Once B0 is measured, the baseline commit, the revisions of models, dataset and dependencies, the GPU and the CPU set never change. While exploring, you may add `--max-samples` to find a direction. Run the full Step 12 before accepting a version or opening a PR.

**One model at a time.** Each candidate optimizes one target model, and the other three must not regress. The current focus is Fun-CosyVoice3.

**Correctness**, for every model:

- Every scored run covers the full set: `completed_requests` is 1088, `failed_requests` is 0, `evaluated` is 1088 and `skipped` is 0.
- The mean `wer_corpus` of RUN 1 to 3 is at most 0.02 (2 percent absolute). Use the full corpus WER. WER corpus (excl >50%) is for debugging only.
- The line is absolute, so a WER rise that stays under 0.02 passes. Report any model whose mean `wer_corpus` rises more than 0.005 above B0.
- Look closer, with a per sample comparison in the PR, when `n_above_50_pct_wer` is above Bk, or when `audio_duration_mean_s` is more than 5 percent away from Bk. A throughput gain that comes with shorter audio is not a gain.

**Performance**, as ratios of the mean against the freshly measured Bk and the original B0:

| Model | Rule |
|---|---|
| The target model | `throughput_qps` ratio above 1 against Bk |
| All four models | `throughput_qps` ratio at least 0.97 against both Bk and B0 |
| All four models | Latency mean, median, p95 and p99, and RTF mean, median, p95 and p99, ratio at most 1.03 against both Bk and B0 |

**Uncertain results.** Every candidate already has a paired pass. If the scored runs disagree on whether a rule passes, report all runs to the maintainers and wait for a decision. A positive mean alone does not establish a stable improvement.

**Cold start.** Torch compile is on by default for the SGLang engine stages (`SGLANG_OMNI_TORCH_COMPILE_DEFAULT` in `server_args_builder.py`), and the Fun-CosyVoice3 DiT always compiles at startup. Inductor keeps compiled graphs in `/tmp/torchinductor_<user>`, shared by all checkouts, and a changed graph compiles again. So RUN 0 is not scored.

**Seeds.** The commands set no sampler seed, so one request that runs to the length cap can move a whole run. Check `max_token_hits` and `latency_p99_s` of each run before trusting a throughput change.

## What Does Not Count

These are not optimizations:

- Changing the four commands or the client, including the dataset, concurrency, warmup, `--ref-format`, `--voice`, `--token-count`, `response_format` or adding a seed.
- Changing the input the model sees (text normalization, reference audio resampling or cutting, reference feature extraction).
- Changing decoding defaults (temperature, top p, top k, repetition penalty, maximum length, EOS) so outputs get shorter. This includes the Fun-CosyVoice3 generation length contract from #2299.
- Lowering audio quality settings, for example fewer Euler steps in the Fun-CosyVoice3 Flow model ([#2479](https://github.com/sgl-project/sglang-omni/pull/2479)).
- Lower precision for weights, activations or the KV cache, or turning on TF32.
- Reusing any result across requests based on the text, reference audio, file path or sample ID (generated speech, speech tokens, speaker embeddings, KV prefixes). The warmup sends the first sample 16 times, and the scored pass sends it again.
- Caching more in the reference caches already in the baseline (Fun-CosyVoice3 `request_builders.py`, Qwen3-TTS `get_speaker_artifact_cache`, MOSS-TTS `ref_audio_cache`), raising their limits or changing their keys. Open #2669 is such a change.
- Special branches for SeedTTS or for one of the four request shapes.
- Server startup warmup that reads dataset files. It may only use synthetic input.
- Using more resources, such as a second GPU, MPS, a different CPU set, or more threads than the assigned CPUs.

A change that alters numerics or has a tradeoff can go in its own PR, with a quality comparison for each sample. The maintainers decide at review. This loop does not accept it.

## Before Opening a PR

**Where code goes.** Put new code in the model directory under `sglang_omni/models` where possible. Shared hot paths get strict review.

**Default on.** An optimization is on by default. If method A beats method B with no tradeoff, remove B where possible. Optional paths behind flags make the code hard to read.

**Acceptance tests stay unchanged.** A PR does not change the test commands, the evaluation and metric code under `benchmarks/`, or the CI under `.github/` and `tests/test_model/`. You may add or change matching unit tests under `tests/unit_test/`, but never loosen a check.

**Local benchmark fixes.** If a command does not run, you may fix the benchmark locally. Change only how it runs, never the inputs, request parameters, sample range or scoring. Apply the same diff to all three checkouts and measure B0 again (Step 10, new pass name). The diff never changes after that and never goes into the PR.

**Reproducible.** Every PR is reproduced before it is accepted, mostly on H100 and H200.

### Step 15: rebase for a PR

A PR goes to current main, which has moved (for example #2632 and #2662). The PR holds every accepted commit since the baseline. Use `pr01` for the first PR, `pr02` for the second, and so on.

1. Put main in `/work/ref`. Put Bk in `/work/cand` on a new branch and rebase it onto main.

   ```bash
   BK_COMMIT=$(tail -n 1 /work/accepted.txt | cut -d ' ' -f 2)
   git -C /work/base fetch origin main
   git -C /work/ref checkout --detach origin/main
   git -C /work/cand checkout --detach $BK_COMMIT
   git -C /work/cand switch -c pr01
   git -C /work/cand rebase origin/main
   ```

   If the rebase stops on a conflict, run `git -C /work/cand rebase --abort` and ask the maintainers.

2. Check that neither main nor the candidate changed the TTS client, scoring code or dependencies. Both commands must print nothing. If the first prints anything, stop and ask the maintainers.

   ```bash
   git -C /work/ref diff --stat $BASE_COMMIT -- benchmarks/benchmarker benchmarks/dataset benchmarks/eval/benchmark_tts_seedtts.py benchmarks/metrics benchmarks/tasks pyproject.toml
   git -C /work/cand diff --stat origin/main -- benchmarks pyproject.toml
   ```

3. Run Step 6 with `T=/work/ref`, then with `T=/work/cand`. Both commands below must print nothing. If they differ, stop and ask the maintainers.

   ```bash
   diff <(/work/base/.venv/bin/python -m pip freeze --exclude-editable) <(/work/ref/.venv/bin/python -m pip freeze --exclude-editable)
   diff <(/work/ref/.venv/bin/python -m pip freeze --exclude-editable) <(/work/cand/.venv/bin/python -m pip freeze --exclude-editable)
   ```

4. Run a paired pass (Step 13) with `PASS=pr01`. The PR reports this pass, and the gain must hold there. Changes from main, such as #2636, apply to both sides.
5. Push the branch `pr01` to your fork and open a PR against main. Fill in `.github/pull_request_template.md`, and put the step 4 compare output under Benchmark & Profiling.
6. Before the next candidate, move both checkouts back to Bk:

   ```bash
   git -C /work/ref checkout --detach $BK_COMMIT
   git -C /work/cand checkout --detach $BK_COMMIT
   ```

   Then run Step 6 with `T=/work/ref` and with `T=/work/cand`.

### TTS CI

A PR must pass the TTS CI. CI runs one preset per PR on two H100s and deploys differently from this loop:

- A Rust router with two workers, and compile off for the SGLang engine stages (the third `export` below).
- The `qwen3-tts` and `qwen3-tts-custom-voice` presets run the vocoder in its own process (`--vocoder.process vocoder`, GPU memory fractions 0.85 and 0.10). The loop keeps the default colocated layout.
- The CI `moss` preset in `tests/test_model/tts_ci_config.py` is MOSS-TTS-Local-Transformer-v1.5, and CI also runs Higgs TTS.
- The `qwen3-tts-custom-voice` preset gates nothing yet (`gate_thresholds=False`). A green run only shows that it runs.
- Stage 4 (`tests/test_model/test_tts_serving_ci.py`) always runs Higgs.
- Stage 5 (`tests/test_ci/test_tts_mps_dp2.py`) runs MPS on every run, with `higgs_h100_dp3` when CI picks `higgs` and `moss_local_h100_dp2` (MOSS-TTS-Local) otherwise. It runs only in CI.

So a Fun-CosyVoice3 PR that touches shared code is also gated on Higgs and MOSS-TTS-Local.

To run CI locally, do it after Step 15 step 5 and before step 6, so `/work/cand` holds the PR branch. CI uses GPU 0 and GPU 1. Set `CI_CPUSET` to the CPUs of their NUMA node, minus the two lowest numbered physical cores and their siblings (56 CPUs with SMT). On the host, run:

```bash
CI_CPUSET=2-29,114-141
docker stop tts-hill-climbing
docker run -d --name tts-ci \
    --gpus '"device=0,1"' --cpuset-cpus $CI_CPUSET \
    -v /dev/shm:/dev/shm \
    -v $WORK:/work \
    -v $HOME/.cache:/root/.cache \
    --entrypoint sleep \
    hongccc/sglang-omni@sha256:ebe4239e29a764ee3a2806385c061c5fd438a26f01458e503d3822dcba5790df \
    infinity
docker exec -it tts-ci bash
```

Do not set `OMNI_CI_CPUSET`. `--cpuset-cpus` already pins the container. Inside it, run:

```bash
source /work/env.sh
cd /work/cand
source .venv/bin/activate
HF_HUB_OFFLINE=0 hf download OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5
HF_HUB_OFFLINE=0 hf download OpenMOSS-Team/MOSS-Audio-Tokenizer-v2
HF_HUB_OFFLINE=0 hf download bosonai/higgs-tts-3-4b
export OMNI_CI_HOME=/work/omni_ci
export SGLANG_OMNI_ROUTER_BIN=$(bash .github/scripts/prepare_rust_router.sh build)
export SGLANG_OMNI_TORCH_COMPILE_DEFAULT=0
TTS_CI_MODEL=cosyvoice3 python -m pytest tests/test_model/test_tts_ci.py -v -s -x --concurrency 16
python -m pytest tests/test_model/test_tts_serving_ci.py -v -s -x
```

The downloads are for the `moss` and `higgs` presets and stage 4. Keep the `export` lines separate, or `prepare_rust_router.sh` cannot read `OMNI_CI_HOME`. The router build needs network and lands in `/work/rust-router`, so later containers reuse it. Without `--tts-stage`, `test_tts_ci.py` runs stages 1 to 3 (non streaming, streaming, consistency). For the other presets, use `TTS_CI_MODEL=qwen3-tts`, `qwen3-tts-custom-voice`, `moss` or `higgs`. For the two Qwen3-TTS presets, CI also runs the latency stage, for example `TTS_CI_MODEL=qwen3-tts python -m pytest tests/test_model/test_tts_latency_ci.py -v -s -x`.

When done, run `docker rm -f tts-ci` and `docker start tts-hill-climbing` on the host.

Do not wrap local runs in `run_flaky_pytest.sh`. It runs up to three attempts, so a green result can hide a failure.

Without a label, CI picks `higgs`, `moss`, `qwen3-tts` or `cosyvoice3` from a sha256 digest of the GitHub run ID. A rerun keeps the model, and a new run picks again. State in the PR which model to run. A maintainer then adds one of the labels `run-cosyvoice3`, `run-qwen3-tts`, `run-qwen3-tts-custom-voice`, `run-moss` or `run-higgs`.
