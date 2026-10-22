# TTS benchmark runbook

This page runs the SeedTTS EN benchmark against four TTS models served by SGLang-Omni. It measures throughput, latency, RTF and WER. WER comes from transcribing the generated speech with Qwen3-ASR-1.7B.

Every run uses `python -m benchmarks.eval.benchmark_tts_seedtts`. The client is `benchmarks/eval/benchmark_tts_seedtts.py`, and the scoring code is in `benchmarks/tasks/tts.py` and `benchmarks/metrics/`.

## What is Measured

All four commands use the SeedTTS EN set (`zhaochenyang20/seed-tts-eval-arrow`, 1088 samples), non streaming, at concurrency 16.

| Model | Request shape | Command flags | Output directory name |
|---|---|---|---|
| `Qwen/Qwen3-TTS-12Hz-1.7B-Base` | Voice clone | `--ref-format references` | `qwen3_tts_base_en` |
| `OpenMOSS-Team/MOSS-TTS-v1.5` | Voice clone with a duration target | `--ref-format references --token-count auto` | `moss_tts_en` |
| `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice` | Named voice, no reference audio | `--no-ref-audio --voice Ryan` | `qwen3_tts_customvoice_en` |
| `FunAudioLLM/Fun-CosyVoice3-0.5B-2512` | Voice clone | none | `fun_cosyvoice3_en` |

Each command does the whole cycle by itself:

1. Starts the TTS server on port 8000.
2. Sends 16 warmup requests (the default warmup equals the concurrency). They are not scored.
3. Sends the 1088 scored requests, at most 16 in flight.
4. Stops the TTS server.
5. Starts Qwen3-ASR-1.7B on the same port, transcribes the saved audio, computes WER and stops it.

One GPU is enough. The TTS server and Qwen3-ASR run one after the other on it.

The server uses the benchmark's engine settings (`--max-running-requests 64` and `--cuda-graph-max-bs 64`), not the model's own serve defaults. `server_logs/tts_server.log` in each output directory shows the exact server command. Fun-CosyVoice3 requests send no `max_new_tokens`, so the server applies the model's own length rule. The other models send 2048.

Every metric is in a JSON file in the model's output directory, under `summary`:

| Metric | File | Field | Unit |
|---|---|---|---|
| Throughput | `speed_results.json` | `throughput_qps` | Completed requests per second of wall clock |
| Latency | `speed_results.json` | `latency_mean_s`, `latency_median_s`, `latency_p95_s`, `latency_p99_s` | Seconds per request, end to end |
| RTF | `speed_results.json` | `rtf_mean`, `rtf_median`, `rtf_p95`, `rtf_p99` | Request latency over output audio length |
| Coverage | `speed_results.json` | `completed_requests`, `failed_requests` | Requests |
| Length cap hits | `speed_results.json` | `max_token_hits` | Requests that reported a length stop |
| Finish reasons seen | `speed_results.json` | `finish_reason_observed` | Requests that reported a finish reason |
| Audio length | `speed_results.json` | `audio_duration_mean_s` | Seconds |
| WER | `wer_results.json` | `wer_corpus` | Fraction (0.0144 is 1.44 percent) |
| WER coverage | `wer_results.json` | `evaluated`, `skipped` | Samples |
| Runaway samples | `wer_results.json` | `n_above_50_pct_wer` | Samples with WER above 50 percent |

`wer_results.json` also holds `per_sample` (each transcript and its WER), and `speed_results.json` holds `per_request`. Read the numbers from these files, not from the printed lines. `wer_corpus` includes every sample. `wer_below_50_corpus` leaves out samples above 50 percent WER and is for debugging only.

`max_token_hits` counts only requests that reported a finish reason. It is complete only when `finish_reason_observed` equals `completed_requests`.

SIM and UTMOS are not scored by these commands.

## One Time Setup

### Choose the GPU and CPUs

Use one GPU and one fixed set of CPUs for every run you want to compare. On the host, run:

```bash
nvidia-smi topo -m
lscpu -e=CPU,NODE,CORE
```

In the first output, read the NUMA node of your GPU. In the second, pick a set of physical cores on that node plus their SMT siblings (same `CORE` value). Leave the lowest numbered cores to the host. Keep the set the same for every run.

Set these on the host. Replace the values with yours. `NODE` is the GPU's NUMA node.

```bash
GPU=0
CPUSET=2-15,114-127
NODE=0
WORK=$HOME/tts_bench
```

### Start the container

Use the TTS CI image. Everything under `/work` lives in `$WORK` on the host.

```bash
mkdir -p $WORK $HOME/.cache
docker run -d --name tts-bench \
    --gpus device=$GPU --cpuset-cpus $CPUSET --cpuset-mems $NODE \
    -v /dev/shm:/dev/shm \
    -v $WORK:/work \
    -v $HOME/.cache:/root/.cache \
    --entrypoint sleep \
    hongccc/sglang-omni@sha256:06ed306707bcf3d9e9beb5ba7ce66f09a666cac11c23455ece50c34199c0e985 \
    infinity
docker exec -it tts-bench bash
```

Every later command runs inside the container. Open another shell with the last line. Check the environment first. `HOME` must be `/root`, and `uv`, `hf` and `git` must be found.

```bash
echo $HOME $HF_HOME
which uv hf git
```

### Write the variables file

```bash
cat > /work/env.sh <<'EOF'
export COSYVOICE_PATH=/work/CosyVoice
export COSYVOICE_COMMIT=074ca6dc9e80a2f424f1f74b48bdd7d3fea531cc
export SGLANG_OMNI_STRICT_PORT=1
export CUDA_VISIBLE_DEVICES=0
EOF
source /work/env.sh
```

Run `source /work/env.sh` in every new shell.

- `SGLANG_OMNI_STRICT_PORT=1` makes the server fail instead of moving to another port, so the client never talks to a leftover server.
- `CUDA_VISIBLE_DEVICES=0` is required. After each server stops, the benchmark waits for the GPU memory to be released, and it refuses to run without this variable. Inside the container the GPU is index 0.
- Do not set `SGLANG_OMNI_TORCH_COMPILE_DEFAULT`. Torch compile then stays on, as in a normal deployment (CI turns it off).

### Install sox and the CosyVoice sources

Fun-CosyVoice3 and Qwen3-TTS need sox. Fun-CosyVoice3 imports the CosyVoice and Matcha-TTS sources, as in `.github/scripts/prepare_omni_venv.sh`.

```bash
apt-get update && apt-get install -y sox
git clone --filter=blob:none --no-checkout https://github.com/FunAudioLLM/CosyVoice.git $COSYVOICE_PATH
git -C $COSYVOICE_PATH checkout --detach $COSYVOICE_COMMIT
git -C $COSYVOICE_PATH submodule update --init --depth=1 third_party/Matcha-TTS
git -C $COSYVOICE_PATH/third_party/Matcha-TTS rev-parse HEAD
```

The last line must print `dd9105b34bf2be2230f4aa1e4769fb586a3c824e`.

### Check out and install SGLang-Omni

Replace `main` with the commit you want to measure.

```bash
git clone https://github.com/sgl-project/sglang-omni.git /work/sglang-omni
cd /work/sglang-omni
git checkout --detach main
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

This follows `.github/scripts/prepare_omni_venv.sh`, but installs only the `fun-cosyvoice3` extra. The last two install lines come from `docs/cookbook/qwen3_tts.md`. Never install them with dependencies. `qwen-tts` would replace the pinned Transformers.

### Download the data and models

Each model is pinned to its Hugging Face `main` commit on 2026-10-10.

```bash
cd /work/sglang-omni
source .venv/bin/activate
python -m benchmarks.dataset.prepare --dataset seedtts
hf download Qwen/Qwen3-TTS-12Hz-1.7B-Base --revision fd4b254389122332181a7c3db7f27e918eec64e3
hf download Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice --revision 0c0e3051f131929182e2c023b9537f8b1c68adfe
hf download OpenMOSS-Team/MOSS-TTS-v1.5 --revision cdd3b911b1585e3f2dbc7775ef10f9926f58850a
hf download OpenMOSS-Team/MOSS-Audio-Tokenizer --revision 3cd226ba2947efa357ef453bcad111b6eafba782
hf download FunAudioLLM/Fun-CosyVoice3-0.5B-2512 --revision 29e01c4e8d000f4bcd70751be16fa94bf3d85a18
hf download Qwen/Qwen3-ASR-1.7B --revision 7278e1e70fe206f11671096ffdd38061171dd6e5
deactivate
```

`MOSS-Audio-Tokenizer` is the codec that MOSS-TTS-v1.5 loads.

The dataset is pinned in code. `benchmarks/dataset/prepare.py` sets `SEEDTTS_DATASET_REVISION = "27f4c1adee83b5b29b7c4b375f6b976324bda308"`, and the loader uses it even though the dataset's `main` has moved.

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

Check the pins. The loop prints each model's `refs/main` and the snapshot directories it has. The pinned revision must have a matching snapshot directory. Other cached snapshots may also be present. The last line must list `27f4c1adee83b5b29b7c4b375f6b976324bda308`.

```bash
H=/root/.cache/huggingface/hub
for d in $H/models--Qwen--Qwen3-TTS-12Hz-1.7B-Base $H/models--Qwen--Qwen3-TTS-12Hz-1.7B-CustomVoice $H/models--OpenMOSS-Team--MOSS-TTS-v1.5 $H/models--OpenMOSS-Team--MOSS-Audio-Tokenizer $H/models--FunAudioLLM--Fun-CosyVoice3-0.5B-2512 $H/models--Qwen--Qwen3-ASR-1.7B; do
    echo "$d refs/main=$(cat $d/refs/main) snapshots=$(ls $d/snapshots | tr '\n' ' ')"
done
ls $H/datasets--zhaochenyang20--seed-tts-eval-arrow/snapshots
```

Then go offline, so no run fetches a newer revision:

```bash
echo 'export HF_HUB_OFFLINE=1' >> /work/env.sh
source /work/env.sh
```

`HF_HUB_OFFLINE` does not cover ModelScope. Each Fun-CosyVoice3 server builds the upstream `CosyVoice3` object to load Flow and HiFT, and its constructor also loads the wetext normalizer from ModelScope. That request runs on every start, even with the files cached. If ModelScope is unreachable or rate limited, `tts_server.log` shows `no frontend is avaliable`. SGLang-Omni never uses that normalizer. It encodes the raw text with its own tokenizer (`sglang_omni/models/fun_cosyvoice3/utils.py`), so the outputs are the same either way and the run stays valid.

### Write the summary script

`python /work/summary.py DIR` reads `DIR/run1` to `DIR/run3` and prints, for each model and field, the three values, their mean and their sample standard deviation. It also prints a coverage line for each run.

```bash
cat > /work/summary.py <<'EOF'
import json
import statistics
import sys
from pathlib import Path

MODELS = ["qwen3_tts_base_en", "moss_tts_en", "qwen3_tts_customvoice_en", "fun_cosyvoice3_en"]
FIELDS = {
    "speed_results.json": ["throughput_qps", "latency_mean_s", "latency_median_s", "latency_p95_s",
                           "latency_p99_s", "rtf_mean", "rtf_median", "rtf_p95", "rtf_p99",
                           "max_token_hits", "finish_reason_observed", "audio_duration_mean_s"],
    "wer_results.json": ["wer_corpus", "n_above_50_pct_wer"],
}
EXPECTED = 1088
RUNS = (1, 2, 3)

root = Path(sys.argv[1])


def summary(run, model, file):
    path = root / f"run{run}" / model / file
    return json.loads(path.read_text())["summary"] if path.exists() else None


for model in MODELS:
    print(f"\n{model}")
    for run in RUNS:
        speed = summary(run, model, "speed_results.json")
        wer = summary(run, model, "wer_results.json")
        if speed is None or wer is None:
            print(f"  run{run}: INCOMPLETE, result file missing")
            continue
        counts = (speed["completed_requests"], speed["failed_requests"], wer["evaluated"], wer["skipped"])
        status = "complete" if counts == (EXPECTED, 0, EXPECTED, 0) else "INCOMPLETE"
        print(f"  run{run}: {status}, completed {counts[0]}, failed {counts[1]}, evaluated {counts[2]}, "
              f"skipped {counts[3]}, finish reasons {speed['finish_reason_observed']}")
    print(f"  {'field':24}{'run1':>10}{'run2':>10}{'run3':>10}{'mean':>10}{'std':>10}")
    for file, fields in FIELDS.items():
        summaries = [summary(run, model, file) for run in RUNS]
        if None in summaries:
            continue
        for field in fields:
            values = [s[field] for s in summaries]
            print(f"  {field:24}" + "".join(f"{v:10.4g}" for v in values)
                  + f"{statistics.mean(values):10.4g}{statistics.stdev(values):10.4g}")
EOF
```

## Run the Benchmark

### Smoke test

Run one model on 32 samples first. It catches setup problems (offline loading, GPU cleanup, ASR) in minutes. Its numbers are not comparable with full runs.

```bash
source /work/env.sh
cd /work/sglang-omni
source .venv/bin/activate
python -m benchmarks.eval.benchmark_tts_seedtts --meta zhaochenyang20/seed-tts-eval-arrow \
    --model FunAudioLLM/Fun-CosyVoice3-0.5B-2512 --port 8000 \
    --output-dir /work/results/smoke/fun_cosyvoice3_en --lang en --max-concurrency 16 --max-samples 32
deactivate
```

It passes when `/work/results/smoke/fun_cosyvoice3_en` holds both `speed_results.json` and `wer_results.json`.

### Write the run script

`bash /work/run_tts.sh CHECKOUT OUT` runs the four commands once, from the checkout `CHECKOUT`, into the new directory `OUT`. It stops at the first error and refuses to reuse a directory.

```bash
cat > /work/run_tts.sh <<'EOF'
set -e
CHECKOUT=$1
OUT=$2
mkdir -p $(dirname $OUT)
mkdir $OUT
source /work/env.sh
cd $CHECKOUT
source .venv/bin/activate
git rev-parse HEAD > $OUT/commit.txt
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
EOF
```

Each command starts and stops its own servers, so every run starts from a fresh server. To measure fewer models, delete the other commands from the script. Keep the order of the rest.

### Run one measurement

One measurement is RUN 0 to RUN 3. RUN 0 only warms caches and is never scored. Give each measurement a new `NAME`, for example the commit you measure.

```bash
NAME=m1-$(git -C /work/sglang-omni rev-parse --short HEAD)
for RUN in 0 1 2 3; do
    bash /work/run_tts.sh /work/sglang-omni /work/results/$NAME/run$RUN || break
done
python /work/summary.py /work/results/$NAME
```

### Coverage checks

A run is complete only when the summary prints `complete` for it. That means, for every model:

- `completed_requests` is 1088 and `failed_requests` is 0 in `speed_results.json`.
- `evaluated` is 1088 and `skipped` is 0 in `wer_results.json`.

Also check that `finish_reason_observed` equals `completed_requests`. If it is lower, `max_token_hits` undercounts.

If a command exits with an error, or a run is `INCOMPLETE`, do not use that measurement. Start it again under a new `NAME`. Do not swap in a single rerun. Read `server_logs/tts_server.log` and `server_logs/asr_server.log` in the model's output directory for the cause.

## Measuring Accurately

**Fixed GPU and CPUs.** Use the same GPU, CPU set, image and model revisions for every number you compare. The container pins the CPUs with `--cpuset-cpus` and the memory node with `--cpuset-mems`. Run `nvidia-smi` on the host, not in the container, to check that no other process uses the GPU. A container sees only its own processes.

**Fresh server per run.** Each command starts its own TTS server and Qwen3-ASR server, so no state carries over between runs. `--repeats` with `--generate-only` measures several passes on one server. Those passes share a warm server, so they are not independent runs.

**Warmup.** The 16 warmup requests run at full concurrency before timing starts. Cold work on the first batches then does not land in the scored window. Keep the default. `--warmup 0` moves that work into the scored numbers.

**Compile and cold start.** Torch compile is on by default for the Qwen3-TTS and MOSS-TTS engine stages (`SGLANG_OMNI_TORCH_COMPILE_DEFAULT` in `sglang_omni/scheduling/sglang_backend/server_args_builder.py`). The Fun-CosyVoice3 engine stage never compiles, but its DiT compiles at startup. Inductor keeps compiled graphs in `/root/.cache/sglang-omni/torchinductor`, which is on the `$HOME/.cache` mount and survives the container. A changed graph compiles again, and the first run after a change pays for it. That is why RUN 0 is not scored.

**Three scored runs.** Report the mean of RUN 1 to 3 and their spread (the `std` column, or all three values). A single run can move a lot. The commands set no sampler seed, so one request that runs to the length cap can change a whole run. Check `max_token_hits` and `latency_p99_s` of each run before you trust a throughput change.

**Run to run spread.** Repeat the whole measurement under new names to see the run to run spread on your machine.

**The scorer runs from the same checkout.** The TTS client and Qwen3-ASR run from the same checkout as the server. A change to shared code (for example `sglang_omni.admission`) also changes the ASR side.

**Audio length.** Throughput and RTF depend on how much audio each request makes. Compare `audio_duration_mean_s` along with them. A faster run with shorter audio did less work.

For profiling, use the skill in `.claude/skills/model-profiling`.

## Reference Results

These numbers were not measured with the setup on this page. Use them only to see whether your numbers are in a sensible range. Compare only numbers you measured yourself on one machine.

| Model | `throughput_qps` | `latency_mean_s` | `rtf_mean` | `wer_corpus` | Source and setup |
|---|---:|---:|---:|---:|---|
| Qwen3-TTS-12Hz-1.7B-Base | 32.85 | 0.484 | 0.1193 | 0.0113 | CI host reference in `tests/test_model/tts_ci_config.py` (`QWEN3_TTS_VC_NON_STREAM_P95`, `QWEN3_TTS_VC_WER_MAX_CORPUS`). H100, router over two workers, compile off, vocoder in its own process |
| Fun-CosyVoice3-0.5B-2512 | 10.995 | 1.451 | 0.3256 | 0.0144 | CI host reference in `tests/test_model/tts_ci_config.py` (`COSYVOICE3_VC_NON_STREAM_P95`, `COSYVOICE3_VC_WER_MAX_CORPUS`). H100, five full sample rounds at `c00c327`, router over two workers, compile off |
| MOSS-TTS-v1.5 | 2.312 | 3.449 | 0.811 | 0.0168 | `docs/cookbook/moss_tts.md`. One H200, concurrency 8, WER from Whisper-large-v3 |

Qwen3-TTS-12Hz-1.7B-CustomVoice has no reliable reference. `docs/cookbook/qwen3_tts.md` lists 14.788 qps, and `QWEN3_TTS_CUSTOM_VOICE_NON_STREAM_REFERENCE` in `tts_ci_config.py` lists 34.153 qps for the same source.

## Comparing with TTS CI

TTS CI (`.github/workflows/test-tts-ci.yaml`) deploys differently from this page. Do not compare numbers from this page with CI numbers.

- CI runs a Rust router over two workers on two H100s. This page runs one server on one GPU.
- CI sets `SGLANG_OMNI_TORCH_COMPILE_DEFAULT=0` (`.github/scripts/ci_env.sh`). This page keeps compile on.
- The `qwen3-tts` and `qwen3-tts-custom-voice` presets run the vocoder in its own process (`--vocoder.process vocoder`, GPU memory fractions 0.85 and 0.10). This page keeps the default colocated layout.
- The CI `moss` preset is MOSS-TTS-Local-Transformer-v1.5, not MOSS-TTS-v1.5. CI also runs Higgs TTS.
- CI scores SIM and UTMOS. This page does not.
- Stages 1 to 4 run through `.github/scripts/run_flaky_pytest.sh`, which makes up to three attempts, so a green stage can hide a failed attempt. The latency stage makes one attempt.

## Notes

- **SeedTTS EN is a public set.** Its 1088 samples are not a sample of real traffic. Concurrency 16 non streaming is one operating point.
- **MOSS-TTS-v1.5 runs here at concurrency 16.** Its cookbook measures it at 8 and reports higher WER above that.
- **WER depends on the ASR model.** Numbers scored with Whisper-large-v3 (as in some cookbook tables) do not compare with Qwen3-ASR numbers.
- **Some samples run away.** A few utterances loop until the length cap. They raise `wer_corpus`, `max_token_hits`, `latency_p99_s` and `audio_duration_mean_s` together. Check `n_above_50_pct_wer` and the `per_sample` entries when one run differs from the others.
- **Changing the client changes the workload.** The dataset, concurrency, warmup, `--ref-format`, `--voice`, `--token-count`, `--response-format` and `--seed` all change what is measured. Keep them fixed across everything you compare.
- **Use GPU index 0 inside the container.** After each server stops, the GPU cleanup scans for orphan processes on the visible GPU. With a host GPU other than 0 it may miss them, but the memory wait still works.
