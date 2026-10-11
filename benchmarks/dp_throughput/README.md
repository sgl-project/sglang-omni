# Same-card DP / server-throughput profiling

Measurement tooling to profile serving **throughput** and **GPU idle** for an omni
TTS pipeline under same-card data parallelism (DP=N replicas on one GPU), with and
without CUDA MPS, driven either by a manual client split or through the real
`sglang-omni` router. The goal is a **single shared methodology** so results are
comparable across people instead of every ad-hoc harness measuring a different thing.

This directory contains **only measurement/benchmark tooling**. It does not change the
router, the models, or the serving path. One reproduction step (fitting more than one
replica on a card) needs a one-line local patch that is intentionally NOT included here;
see "Fitting more than one replica per card" below.

## Methodology (the shared measurement standard)

- **Primary metric is device-level SM Active %, not throughput.** Throughput (qps) is
  easy to mis-measure: it moves with batch size, per-request audio length, co-tenant CPU
  load, and client-side artifacts. We read the GPU directly with `nsys --gpu-metrics`
  (NVIDIA device-level Speed-Of-Light counters, 10 kHz). SM Active % is measured over the
  **whole physical GPU across all processes on it**, so it stays comparable when several
  replicas share a card (a per-process CUPTI timeline cannot see the other replica).
  **GPU idle fraction ~= 100 - SM Active %.** DRAM and Tensor % come along for the bound
  diagnosis.
- **Throughput is reported too, but as corroboration.** Use it for scaling ratios
  (does DP=2 roughly double?), not as the sole signal.
- **Client control (rule out the client being the bottleneck), 3 checks:**
  1. **Plateau**: a single-server concurrency sweep must flatten (more offered
     concurrency stops raising qps) => the server is saturated, not starved. (`conc_sweep.sh`)
  2. **Saturation in-window**: server `#running-req` sits near the engine cap with
     `#queue-req > 0` for the whole capture window => it always had work. (`analyze_saturation.py`,
     run automatically by `run_condition.sh`)
  3. **Independent clients**: the load generator is a continuous closed-loop (each of
     `conc` workers re-issues the instant its previous request completes, no inter-batch
     drain), and you run one client process per replica so N clients are N independent
     GILs. If the same clients reach a higher number in another condition, the client is
     not the limiter.

## Layout

| file | what |
|---|---|
| `common.sh` | shared config (MODEL, CARD, cores, ...); sourced by the shell scripts |
| `load_client.py` | continuous closed-loop seed-tts load generator (one per replica) |
| `launch_server.sh` | launch one omni server, pinned, on `$CARD` |
| `wait_health.sh` | poll `/health` |
| `run_condition.sh` | one condition: N clients + nsys device capture + parse + throughput + saturation |
| `parse_gpu_metrics.py` | nsys report -> device SM% / DRAM% / Tensor% |
| `analyze_saturation.py` | server `#running-req` / `#queue-req` in the capture window |
| `conc_sweep.sh` | concurrency plateau test |
| `scan_dp.sh` | DP=1..N sweep, optional MPS |
| `mps.sh` | start/stop a per-user MPS daemon |
| `router_bench/` | measure the stock `sglang-omni` router driving same-card replicas |

## Setup

```bash
cd <repo-root>                      # scripts import the repo without installing (PYTHONPATH=repo)
export MODEL=<served model path or HF id>     # an omni TTS pipeline model
export CARD=2                        # physical GPU index (nvidia-smi) to run everything on
export NUMA_NODE=0                   # NUMA node of $CARD  (check: nvidia-smi topo -m)
export SERVER_CORES=0-15             # NUMA-local cores for server replicas
export CLIENT_CORES=16-23            # cores for load clients, disjoint from SERVER_CORES
export OUTDIR=./dp_runs
# GPU_METRICS_SET defaults to gh100 (Hopper: H100/H200). Set to your arch's nsys set otherwise.
```

Requirements: `nsys` (Nsight Systems), `numactl`, and the repo's benchmark deps (the
scripts reuse `benchmarks/eval/benchmark_omni_seedtts.py` and `benchmarks/dataset`).

## Usage

**1) Client-control plateau (single server):**
```bash
bash benchmarks/dp_throughput/launch_server.sh 8801
bash benchmarks/dp_throughput/wait_health.sh 8801
bash benchmarks/dp_throughput/conc_sweep.sh 8801 8,16,32,48,64,96
# qps should flatten at high conc => server-saturated (not client-starved)
```

**2) One measured condition (already-running server(s)):**
```bash
# 1 server on 8801, drive with 1 client @ conc 96:
OUTDIR=./dp_runs bash benchmarks/dp_throughput/run_condition.sh base 60 8801 96 16-23 ./dp_runs/base
```

**3) Same-card DP sweep (manual split):**
```bash
# DP=1..3, MPS off, 2 reps. Needs the mem-fraction patch for DP>1 (see below).
MEM_FRACS=0.40,0.40,0.27 bash benchmarks/dp_throughput/scan_dp.sh 3 0 2
```

**4) With MPS (unprivileged per-user daemon):**
```bash
MEM_FRACS=0.40,0.40,0.27 bash benchmarks/dp_throughput/scan_dp.sh 3 1 2   # use_mps=1
```

**5) Through the real router (compare manual split vs router):**
```bash
# one rep per call (fresh MPS each); router spawns N same-card workers itself:
USE_MPS=1 bash benchmarks/dp_throughput/router_bench/run_router_condition.sh 2 1
USE_MPS=1 bash benchmarks/dp_throughput/router_bench/run_router_condition.sh 3 1
```

## Reading results (per condition dir under `$OUTDIR`)

- `sol_<label>.txt` : device **SM Active %** (GPU busy; idle ~= 100 - this), DRAM %, Tensor %.
- `run.log` : `AGGREGATE qps_window` (sum across clients) + per-client cv (steadiness) +
  the `SATURATION` block (`#running-req` med/min/max, `#queue-req`).
- router runs also print `router cpu%` (its single async event loop; ~100 % = one core maxed).

## MPS notes and pitfalls

- MPS runs **unprivileged** if the GPU compute mode is `Default` (Volta+): the per-user
  control daemon and pipe live under `$MPS_PIPE` (default `/tmp/dpt_mps_pipe`). `mps.sh start`
  / `mps.sh stop`. No `sudo`, no system config change.
- **Never `kill -9` an MPS client (a server/worker).** It corrupts the MPS server, and later
  workers then fail with `cudaErrorMpsRpcFailure`. Stop clients gracefully (SIGTERM) before
  stopping MPS. The scripts do this.
- **Start MPS with a fresh pipe dir.** A reused pipe dir accumulates crash state and poisons
  new workers. `mps.sh start` clears it.
- **router + MPS is unstable across back-to-back reps**: the 2nd sustained rep tends to crash
  a worker with `cudaErrorMpsRpcFailure`. Run one rep per fresh MPS daemon (the router script
  does this). Unprivileged users cannot `nvidia-smi --gpu-reset`, so if MPS gets wedged, use a
  brand-new pipe dir.

## Fitting more than one replica per card (manual patch, NOT part of this PR)

Two replicas on one 80 GB card OOM because the AR engine hardcodes its KV pool fraction.
To reproduce DP>1 you must make that fraction configurable. This is a **core-code change,
so it is deliberately excluded from this tooling PR**; apply it locally to reproduce:

```diff
# sglang_omni/models/higgs_tts/stages.py  (create_sglang_tts_engine_executor overrides)
-        mem_fraction_static=0.85,
+        mem_fraction_static=float(os.environ.get("HIGGS_MEM_FRAC", "0.85")),
```

Then `MEM_FRACS`/`MEM_FRAC` flow through to each replica via `HIGGS_MEM_FRAC`. Rule of thumb
that fits on 80 GB: DP1/DP2 at 0.40, DP3 at 0.27; DP4 does not fit (the 4th replica cannot
get enough for its weights under the fraction-of-free-memory model). Without the patch the
tooling still works for single-replica and multi-card runs.

## Notes

- Numbers from the runs this tooling produced (Higgs TTS 4B, 1x H100) live in the PR
  description; treat SM%/idle as the primary result and throughput as corroboration.
- Everything is pinned (NUMA-local server cores, disjoint client cores); set
  `SERVER_CORES`/`CLIENT_CORES`/`NUMA_NODE` to match your card's affinity (`nvidia-smi topo -m`).
