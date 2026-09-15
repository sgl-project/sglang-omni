# Restage: residency planning

Restage answers one deployment question: given a pipeline, a GPU budget and a
workload with an SLO, where should each GPU process live, how many copies of
it, and should copies share a card. It ranks a small family of residency
shapes from two probes on one GPU, then measures only the top-ranked shapes
against the shipped layout. The method follows the Restage planner by Yueying
Li and Yuanfan Chen; SGLang-Omni contributes the schema-level materialization
of candidates and the measurement harness.

## Model

A shape assigns replica device tuples to every GPU process. Its utility is the
aggregate flow F the busiest GPU still sustains:

```
sum over instances i on GPU g of  F / (n_i * T_i)  <=  C_g
F  <=  n_p * T_p * min(1, pool_p / kappa_p)   for the KV-holding process p
```

`T_i` is the throughput one instance sustains inside the SLO, `n_i` the
replica count of its process, `C_g` is 1 when the GPU serves one pipeline
copy and `k * d(mode)` when `k` copies compete under time slicing (`d` = 0.58)
or MPS (`d` = 0.85). `kappa` is the concurrency at which the SLO water level
`c / T + (c + 2.33 sqrt(c)) * delta / D = 1` is reached; `pool` is how many
requests of `L` tokens the KV pool holds at the declared memory fraction.

Four shape families are ranked: replicate the whole pipeline (dedicated or
colocated), give the backbone its own GPUs and replicate the tails on
dedicated GPUs, consolidate the tails on one shared GPU, and tensor-parallel a
backbone whose weights exceed one GPU. Every candidate passes the serving
placement checks before it is ranked.

## Calibrate

```bash
sgl-omni autotune calibrate \
  --config examples/configs/qwen3_tts_1_7b.yaml \
  --spec examples/configs/restage/qwen3_tts_calibrate.json \
  --output restage/calibration
```

The spec names the model, the task (`tts` or `asr`), the samples (explicit
rows, or a SeedTTS `source` the dataset loader stages), and the stage
geometry the probes cannot see: `weights_gib` per GPU stage and
`kv_bytes_per_token` on the engine stage. The command launches the shipped
pipeline once on the caller's visible GPU, sends three streaming requests at
concurrency 1 to read the arrival stall and the engine share of service time
from `X-Engine-Time`, then runs a closed-loop sweep at the listed
concurrencies and keeps the throughput of the largest one whose rtf p99 is
inside the SLO. `constants.json` records every number with its provenance;
stage throughputs derived from the share are `PREDICTED` until a measured
cell replaces them.

## Plan

```bash
sgl-omni autotune plan \
  --config examples/configs/qwen3_tts_1_7b.yaml \
  --constants restage/calibration/constants.json \
  --search-space examples/configs/restage/qwen3_tts_search.json \
  --output restage/plan --top 3
```

The search space lists the device budget and optional configuration
dimensions, each a group of ordinary serving overrides such as
`{"vocoder.process": "vocoder"}`. Dimensions may not set `mps`, `gpu`,
`tp_size`, memory fractions or `processes.*`: the shapes own those. The
output holds `ranking.json` with every shape, its predicted utility, the
binding GPU or pool, and rejections; `baseline.yaml` for the shipped
single-copy layout; and one YAML per exported candidate. No server starts.

Give the serving YAML an admission queue wide enough for open-loop bursts
(for Qwen3-TTS, `stages.tts_engine.engine.max_queued_requests: 256`);
otherwise the campaign measures the queue cap, not the placement.

## Run

```bash
sgl-omni autotune run --spec examples/configs/restage/qwen3_tts_campaign.json --output restage/campaign
```

The campaign measures the baseline and the exported candidates on the
caller's GPUs with one workload, SLO and load grid, and the same seeded
Poisson arrivals for every candidate. Without explicit `rates`, the grid
brackets the baseline's and the best candidate's predicted requests/s at
0.5, 0.75 and 1.0. TTS trials transcribe their saved audio with the ASR model
named in the spec after the TTS server stops, on a separate port; ASR trials
score the returned transcript. `selection.json`
ranks candidates by the highest rate grid prefix that passed every repeat,
then median goodput, and reports how often the measured order agrees with the
prediction. `recommended.yaml` is written only when a candidate has a
passing rate. Set `run_identity` and pass `--resume` to continue an
interrupted campaign; completed trials are reused and the interrupted cell
reruns in a new attempt directory.
