# TTS benchmark runbook

This page runs the SeedTTS EN benchmark against four TTS models served by SGLang-Omni. It measures throughput, latency, RTF and WER. WER comes from transcribing the generated speech with Qwen3-ASR-1.7B.

Every run uses `python -m benchmarks.eval.benchmark_tts_seedtts`. The client is `benchmarks/eval/benchmark_tts_seedtts.py`, and the scoring code is in `benchmarks/tasks/tts.py` and `benchmarks/metrics/`.

## Speculative TTS sweep

The shared speculative sweep accepts `tts-seedtts-en` and `tts-seedtts-zh`. These workloads synthesize SeedTTS target text with its reference audio, then transcribe the saved output using Qwen3-ASR-1.7B. English reports WER; Chinese reports CER through the same character-normalizing scorer used by the TTS client.

After installing the model prerequisites below, run Fun-CosyVoice3 with its DSpark draft on one dedicated GPU:

```bash
CUDA_VISIBLE_DEVICES=0 python -m benchmarks.eval.speculative_sweep \
    --draft-config examples/configs/fun_cosyvoice3_dspark.yaml \
    --workload tts-seedtts-zh \
    --output-dir results/sweep/cosyvoice3_zh \
    -- --stream
```

The defaults are concurrency `1,2,4,8,16,32,64`, three measured repeats at each concurrency, warmup and ABBA order. Each leg starts a fresh TTS server. The baseline config sets `disable_speculative_decoding: true` on every stage that the serving loader identifies as running a draft; both sides disable the drafted stage's prefix cache unless `--prefix-cache` is passed. Use a new output directory for every invocation. For a smoke test, add `--concurrencies 1,2 --repeats 1 --order AB --max-samples 8` before the `--` separator. A smoke test does not establish tail latency or full-corpus quality.

Every TTS leg first generates and saves every load's audio. After all generation legs finish and the last TTS server releases the GPU, the driver starts one ASR server to score the saved loads from every leg. It uses a separate port, `--port + len(--order)`: 8704 for the default ABBA sweep, or 8702 for an AB smoke test. ASR execution is outside the TTS timing window, and its log is separate from the TTS log. `--asr-model-path` and `--asr-concurrency` before the separator configure this evaluator. Model request flags such as `--stream`, `--seed`, `--ref-format` and `--no-ref-audio` go after the separator. Routing, workload selection, concurrency and execution-phase flags are owned by the driver and cannot be overridden there.

Server options cannot be passed after `--`: put `quantization`, `max_running_requests`, `max_queued_requests` and `cuda_graph_max_bs` in the `--draft-config` YAML under `stages.<stage>.engine` (`stages.tts_engine.engine` for Fun-CosyVoice3). Select that file with `--draft-config`, not a client-side `--server-config`. The sweep also rejects client-side `--server-timeout` and `--skip-gpu-cleanup`, because the driver owns server startup and cleanup; these are lifecycle controls, not engine YAML settings.

The example names `yuekai/cosyvoice3_llm_dspark` and omits `speculative_draft_model_revision`. When the server builds the draft stage's engine arguments, it fills in the revision for that draft repository from `sglang_omni/speculative/draft_registry.py`. The Fun-CosyVoice3 target and the default Qwen3-ASR evaluator use model IDs without explicit revisions. For a published comparison, pin those two checkpoints separately and record all three resolved revisions along with the code and environment versions.

Keep sampling and vocoder settings identical between sides. The TTS client preserves Fun-CosyVoice3's server-derived output budget unless `--max-new-tokens` is explicitly passed.

The root `summary.md` and `sweep.json` compare throughput, completion latency p95, first-audio latency p95 when streaming, error rate and generation failures. The shared evaluator retains `asr_server.log` in the output root. Each leg retains `server.log`, `generate.log` and a directory per load: `c1` with one repeat, or `c1_r1`, `c1_r2`, etc. with multiple repeats. Each load keeps `speed_results.json`, `generated.json`, `audio/`, `wer_results.json` and `transcribe.log`. RTF, audio seconds per second, output duration, finish reasons, filtered error rates and severe-error counts remain in these client artifacts. Each synthesis repeat samples total memory used on the dedicated GPU during generation, including warmup. The report shows the largest observed peak in MiB across repeats and loads. If no samples are available the peak is missing. ASR transcription is outside the sampling window.

Throughput for each repeat is completed requests divided by the unrounded measured wall time in `speed_results.json`. Results without a positive wall time cannot qualify for `k`. Measurements are averaged across repeats, then compared across legs, as for the ASR workloads. Generation failures remain separate from transcription failures. Missing first-output timings for a streaming load, untranscribed generated audio, or a repeat with no scored samples marks the load incomplete and excludes it from `k` and the performance checks. The Roadmap accuracy row and reported accuracy ranges use only concurrencies complete on both sides. The per-load table retains partial scores and lists the reasons for exclusion below it. Transcribed samples whose reference normalizes to empty remain scoring skips; their timings still count. Missing files, malformed records or mismatched sample coverage fail the run instead of producing a successful report. `k` is the highest complete measured concurrency with at least 1.2x throughput; it does not certify every default-on gate.

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

The server uses the benchmark's engine settings (`--max-running-requests 64` and `--cuda-graph-max-bs 64`), not the model's own serve defaults. `server_logs/tts_server.log` in each output directory shows the exact server command. Qwen3-TTS and Fun-CosyVoice3 profiles omit `max_new_tokens`, so the server applies their model-specific length rules. Other model profiles send 2048.

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
    hongccc/sglang-omni@sha256:d2fbd80ec42c8a6d9dff4b8a4fdbee35d50c5a1b41aa8dc872925da6e0c82f6a \
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

## Mixed Streaming and Non Streaming Load

Real traffic mixes streaming and non streaming requests, and some scheduler problems only show under that mix. `--mixture NS+S` runs NS non streaming and S streaming closed loop clients against one server at the same time. Each client sends a request, waits for the whole response and sends the next one at once. `48+0` measures non streaming alone and `0+16` streaming alone, with the same method and the same result fields as a mixture load such as `48+16`.

Each load runs this timeline:

1. Warmup. Every client sends one request with the first sample of its class. It stays out of the results and `generated.json`, its audio is saved under the plain row id and not scored, and a failed warmup request stops the run. `--warmup 0` skips it.
2. Settle. All clients start together and run for `--mixture-settle-s` seconds (default 10). Requests that finish in this time are recorded but not counted.
3. Window. The next `--mixture-duration-s` seconds (default 120) are measured.
4. Drain. Clients stop sending when the window ends and the requests in flight finish. They are recorded but not counted.

A summary counts the requests whose response completed inside the window, and every throughput divides by the window length. Both classes walk the sample pool that `--max-samples` and `--sample-offset` select, in dataset order, non streaming from the first row and streaming from the middle row, and start over at the end. Request ids get a `#n<k>` or `#s<k>` suffix, so every saved WAV and `generated.json` entry is unique.

One load writes to `--output-dir` and then scores WER on every request in `generated.json`, as a normal run does:

```bash
python -m benchmarks.eval.benchmark_tts_seedtts --meta zhaochenyang20/seed-tts-eval-arrow \
    --model FunAudioLLM/Fun-CosyVoice3-0.5B-2512 --port 8000 --lang en \
    --mixture 48+16 --output-dir /work/results/mix/cosyvoice3_48_16
```

Several loads, or `--repeats` above 1, need `--generate-only`. The loads run in the given order on one server. Each point goes to `ns<NS>_s<S>`, or `ns<NS>_s<S>_r<k>` with repeats, and `mixture_sweep.json` holds the mean, min, max and count of the main fields of each class across repeats, and every repeat's full summaries under `per_repeat`:

```bash
python -m benchmarks.eval.benchmark_tts_seedtts --meta zhaochenyang20/seed-tts-eval-arrow \
    --model FunAudioLLM/Fun-CosyVoice3-0.5B-2512 --port 8000 --lang en \
    --mixture 48+16,16+16,48+0,0+16,0+48 --generate-only --fingerprint --seed 1234 \
    --output-dir /work/results/mix/cosyvoice3_sweep
```

For a smoke test, add `--mixture-duration-s 20 --mixture-settle-s 2 --max-samples 32`. Its numbers are not comparable with full runs.

`--mixture` cannot be combined with `--concurrency` or `--max-concurrency`, `--concurrencies`, `--stream`, `--request-rate`, `--arrival-seed`, `--sustained-overshoot`, `--subtalker-dosample-ratio`, a `--warmup` other than 0, or the `--transcribe-only`, `--similarity-only` and `--utmos-only` phases. Run those phases later without `--mixture` on the output directory of one load, or on each point directory of a sweep. `wer_results.json` then still records the run as a mixture, and speaker similarity scores each repeated row against the reference audio of its dataset row. `--response-format` applies to the non streaming class, and the streaming class always reads raw PCM. A load above `--max-running-requests` plus `--max-queued-requests` needs `--warmup 0`, because the warmup sends one request per client at once and the server rejects the excess. The AuK profile sets a concurrency and a warmup, so AuK cannot run `--mixture`.

### Mixture Fields

`speed_results.json` of a mixture point keeps every usual key and adds `summary_by_class` and `mixture`:

| Metric | Field | Unit |
|---|---|---|
| Total audio | `summary.audio_throughput_s_per_s` | Audio seconds per second of window, both classes |
| Non streaming throughput | `summary_by_class.non_streaming.throughput_qps`, `audio_throughput_s_per_s` | Requests per second, audio seconds per second |
| Non streaming latency | `summary_by_class.non_streaming.latency_median_s`, `latency_p95_s`, `latency_p99_s` | Seconds per request, end to end |
| Streams finished | `summary_by_class.streaming.throughput_qps` | Streams per second |
| First audio | `summary_by_class.streaming.audio_ttfp_median_s`, `audio_ttfp_p95_s`, `audio_ttfp_p99_s` | Seconds from send to the first audio chunk |
| Playback underrun | `summary_by_class.streaming.max_playback_underrun_p99_s`, `max_playback_underrun_max_s` | Seconds of the longest playback gap in a stream |
| Longest window with no stream finishing | `summary_by_class.streaming.stream_finish_gap_max_s` | Seconds, both window edges counted, the whole window when no stream finished in it |
| Requests outside the window | `mixture.outside_window_requests`, `mixture.outside_window_failed_requests` | Requests per class |
| Pool wraps | `mixture.pool_cycles` | Requests issued per class divided by the pool size |

Each class summary has the fields of a normal `summary`, computed over that class only. A class that completes nothing inside the window reports `throughput_qps` and `audio_throughput_s_per_s` of 0, so starved streams read as the worst value and not as a missing one. The latency fields of the combined `summary` mix both classes, and its first audio fields come from the streams only, so read them per class. A class with no clients has no entry in `summary_by_class`. `per_request` lists every request after warmup with `request_class`, plus `sent_at_s` and `finished_at_s` in seconds from the window start, so a request is in the window when `0 <= finished_at_s <= mixture.duration_s`. A failed request stops its client for the rest of the load. Failures outside the window are in `mixture.outside_window_failed_requests`, and the `failed_requests` of a `mixture_sweep.json` row counts both.

A load is complete when `summary.failed_requests` and every `mixture.outside_window_failed_requests` entry are 0. Also check that `finish_reason_observed` equals `completed_requests` in each class. If it is lower, `max_token_hits` undercounts, which is always the case for streams with `--no-collect-stream-outcomes`.

In `mixture_sweep.json`, `config` describes the whole sweep: `mixture` lists the loads, the settle and the window, and `concurrency` and `warmup` are null (`warmup` is 0 with `--warmup 0`) because they change with the load. Each point's `speed_results.json` records its own. A mixture row gets `normalized_work` when the sweep also ran an `N+0` and a `0+M` load that completed work. It is the non streaming audio throughput divided by the highest `N+0` audio throughput, plus the streaming audio throughput divided by the highest `0+M` one. Streaming audio costs more vocoder time per second, so a change that moves work to the streams can lower total audio while normalized work stays the same (#3668).

### Mixture Caveats

- Mixture numbers are not comparable with the single pass numbers on this page. The pool starts over, so rows repeat and hit server caches more often. On one H100 with Fun-CosyVoice3, the non streaming class at 48+16 walked the 1088 rows 3.2 to 3.7 times per load, as `mixture.pool_cycles` reports.
- The faster build in an A/B measures more requests and reaches further into the pool. Both builds send the same request sequence, but their windows cover different rows. Streams start halfway through the pool, so at 96+16 on one H100 the build before #3668, which finished about half as many streams, drew them from rows with about 3 percent more audio each. Check audio per request per class before you read a small throughput change.
- `normalized_work` divides by the single mode loads of the same sweep, so in an A/B it also carries each sweep's single mode noise. To compare builds, divide both by the same references, such as the mean of both builds' single mode throughput.
- Use the same `--seed` setting on both sides. Seeded server generation is not byte identical across runs, so compare `max_token_hits` and `output_tokens_total` per class before you trust a throughput change.
- One Python event loop drives every client. Keep the CPU pinning from the setup above, and run the same build twice to see the run to run spread before you read small differences.
- Every WAV is saved. At about 160 audio seconds per second, which Fun-CosyVoice3 reaches at 48+16 on one H100, a 120 s window holds about 19,000 audio seconds, about 0.9 GB at 24 kHz 16 bit mono.
- Tails of a build that starves streams vary widely between legs. On one H100 at 48+16, the build before #3668 gave a max underrun p99 of 15.1 s in one leg and 7.7 s in the other, while its total audio stayed within 0.2 percent and its first audio and non streaming latency medians within 1.5 percent. Even its streaming audio moved 7 percent, from 18.4 to 19.8 audio seconds per second. Read such tails as a sign of starvation, not as a precise number.
- #3668 measured a 7 to 9 percent speed difference between two H100s. Compare only legs on the same GPU, start each leg with a fresh server, and run the builds in a balanced order such as A, B, B, A.

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
