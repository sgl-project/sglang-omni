# Same-GPU Data Parallelism with CUDA MPS

> TL;DR: Same-GPU DP with CUDA MPS can substantially increase throughput. In the pinned TTS tests below, saturated DP2 and DP3 configurations reached 1.4 to 2.1x the tuned single-replica throughput.

A common data-parallel deployment assigns one GPU to each replica. When a tuned replica still leaves substantial GPU headroom, colocating multiple replicas on the same GPU can improve per-GPU throughput.

Same-GPU data parallelism runs several complete serving replicas on one GPU and lets [CUDA MPS](https://docs.nvidia.com/deploy/mps/index.html) share the GPU between them. This is a conditional and ongoing optimization. We are excited to share it and call for the community to join the exploration.

## Deploy

The steps below are one continuous flow. We provide a script `examples/launch_same_gpu_dp.sh` to wrap them behind a per-run state directory (`up/verify/down/list`). It records replica PIDs, process groups, ports, and logs, refuses to start when ports are taken or another run is recorded on the same GPU, health-gates sequential startup, surfaces each replica's logged KV allocation, compares expected replica processes against the MPS client list, and tears down only the processes it recorded. The manual steps below explain each mechanism, checkpoint, and failure mode; the launcher is a convenience wrapper around them, not a production supervisor, and it does not remove your responsibility to check actual KV capacity, MPS attachment, and saturation. Detailed instructions are as follows:

1. **Choose the GPU and NUMA node.**

```bash
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader
GPU_ID=0
BUS=$(nvidia-smi --query-gpu=pci.bus_id --format=csv,noheader -i $GPU_ID)
BUS=${BUS,,}; BUS=${BUS:4}
NODE=$(cat /sys/bus/pci/devices/$BUS/numa_node)   # if -1, set the node explicitly
numactl -H | grep "node $NODE cpus"
```

Pick a GPU that is idle, then find its NUMA node from the PCI bus id (drm card ordinals do not always match nvidia-smi ordinals), and pin replicas and memory to that node.

2. **Start a private MPS daemon.**

```bash
export CUDA_MPS_PIPE_DIRECTORY=/tmp/mps-$USER-gpu$GPU_ID/pipe
export CUDA_MPS_LOG_DIRECTORY=/tmp/mps-$USER-gpu$GPU_ID/log
mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
nvidia-cuda-mps-control -d
echo get_default_active_thread_percentage | nvidia-cuda-mps-control   # sanity: responds
```

Isolate this experiment behind a private MPS pipe directory. Every colocated replica in the experiment must use the same directory, or it may fall back to ordinary CUDA context time-slicing.

On a multi-GPU node, the right MPS topology depends on your deployment and should be validated there: NVIDIA's control daemon is often run once per node, and this guide's performance case study below covers only a single H100. Before relying on any cross-GPU result, confirm how the daemon, its pipe and log directories, and each replica's `CUDA_VISIBLE_DEVICES` map to physical GPUs in your setup.

3. **Launch replicas sequentially, and check each KV pool.**

```bash
GPU_ID=0; NODE=0
CORE_BLOCKS=("0-15" "16-31")            # one block per replica, non-overlapping
MF=0.42
i=0
for PORT in 8801 8802; do
  CUDA_VISIBLE_DEVICES=$GPU_ID \
  numactl --cpunodebind=$NODE --membind=$NODE -C "${CORE_BLOCKS[$i]}" \
    sgl-omni serve \
      --model-path bosonai/higgs-tts-3-4b \
      --mem-fraction-static $MF \
      --host 127.0.0.1 --port $PORT --model-name higgs > "replica_$i.log" 2>&1 &
  until [ "$(curl -s -o /dev/null -w '%{http_code}' -m 3 127.0.0.1:$PORT/health)" = 200 ]; do sleep 6; done
  i=$((i+1))
done
```

The example above uses the **same** `--mem-fraction-static` for every replica. Launch one at a time, wait for `/health`, and confirm each replica's `KV Cache is allocated. #tokens: ...` line is workable. Tested starting points on the 80 GB H100 Higgs configuration are 2 replicas at `mf=0.42` with 16 cores each, or 3 at `mf=0.27` with 10 cores each.

Identical `--mem-fraction-static` flags do **not** mean identical KV capacity. `--mem-fraction-static` is evaluated at each replica's init time against **remaining** GPU memory, roughly `mem_fraction × free_memory` after weights and other fixed overheads. It is a per-replica request against whatever is left, not an additive share of the card. Because replicas start sequentially, earlier ones have already reserved memory, so later ones see a smaller free pool and allocate fewer KV tokens even when every flag is the same (in one run, three sequential `mf=0.27` replicas received 97,503 / 53,149 / 20,961 KV tokens).

To control KV size more precisely, set an explicit common token cap with `--max-total-tokens`, at or below the smallest capacity any replica can actually satisfy:

```bash
sgl-omni serve \
  --mem-fraction-static 0.42 \
  --max-total-tokens <COMMON_FEASIBLE_TOKEN_CAP> \
  ...
```

The launcher passes the same cap to every replica when `MAX_TOTAL_TOKENS` is set:

```bash
MAX_TOTAL_TOKENS=<COMMON_FEASIBLE_TOKEN_CAP> \
  bash examples/launch_same_gpu_dp.sh up
```

Leave `MAX_TOTAL_TOKENS` unset to retain the engine's auto-profiled behavior. Still read each replica's `KV Cache is allocated. #tokens: ...` line; if a replica cannot meet the cap, lower the common cap or reduce the replica count.

Intuition suggests that equal KV capacity across colocated replicas is a prerequisite for peak efficiency. We do not yet have a mechanism that starts several servers on one GPU and guarantees identical KV pools without an explicit common cap, and we have not validated that equal allocation is optimal under all loads. **Upcoming experiments will measure how uneven per-replica memory budgets on the same GPU affect aggregate throughput.**

4. **Drive every replica to saturation.**

To reach maximum throughput, feed replicas one by one: keep sending to a replica until it is saturated (`#queue-req > 0`), then move to the next. Aggregate concurrency alone can leave some replicas under-driven. SGLang Omni's router does not yet support this fill-one-then-next behavior; that is planned for later work. Additionally, as mentioned in the previous section, once every colocated replica can be given a stable, equal KV pool, sequential filling should no longer be needed: with fully equivalent workers, random routing is the most efficient way to keep the pool saturated.

5. **Verify MPS attachment.**

MPS should be verified carefully. Four things are easy to conflate: env vars set, daemon running, an MPS server exists, and the replica processes you launched are actually attached as clients. Only the last makes the comparison valid, and a replica that missed the pipe directory falls back to time-slicing without any error. The launcher writes the server-to-client PID mapping to `mps_attach.txt` and fails if any replica has no attached client; to check manually:

```bash
echo get_server_list | nvidia-cuda-mps-control
for SRV in $(echo get_server_list | nvidia-cuda-mps-control); do
  echo "server $SRV clients:"; echo "get_client_list $SRV" | nvidia-cuda-mps-control
done
ps -o pid,pgid,cmd -p <client pids>     # confirm they are your replica processes
grep -c MpsRpc <your replica logs>      # must total 0
```

6. **Route traffic.**

For easy deployment, you can register each replica endpoint with the [Omni Router](omni_router.md). Keep the router's `--max-connections` at least as large as the total offered concurrency. However, as we said, the current router does not support sequential filling, i.e., the fill-one-then-next scheduling strategy. To reach maximum throughput, you can manually route traffic to the replicas one by one.

7. **Tear down safely.**

On a shared host, only touch processes you launched, and never treat "the GPU is empty" as the success condition. Stop new traffic, then SIGTERM each tracked replica process group (`kill -TERM -- -<pgid>`, which also reaps the `multiprocessing-fork` stage workers) and wait until they exit. Confirm the MPS client list is empty (`get_client_list <server>`): the pipe is private to your run, so any remaining client is outstanding work even when its PID no longer matches a tracked group, and live clients must be gone before the daemon quits, or the MPS server can enter an RPC-failure state that outlasts your run. Only then quit the daemon (`echo quit | nvidia-cuda-mps-control`), and SIGKILL surviving tracked groups only as a last resort. `examples/launch_same_gpu_dp.sh down` follows this order and keeps the state directory whenever cleanup cannot be confirmed.

Setting up and tearing down MPS is more involved than running a single replica, but in the pinned H100 Higgs tests the throughput gain was substantial. The table below shows the nominal completed-run ranges; the full accounting, including the failed and degraded runs, is in the case study.


| Configuration | Nominal throughput | Relative to single |
|---|---:|---:|
| Single c96 | 21.7 to 22.1 qps | 1.0x |
| DP2 + MPS, 2 x c64 | 31.5 to 37.7 qps | 1.4 to 1.7x |
| DP3 + MPS, 3 x c64 | 39.9 to 46.9 qps | 1.8 to 2.1x |

These commands and `--mem-fraction-static` starting points are from an 80 GB H100 with Higgs and are not fixed recommendations for other GPUs. On an H200 you would re-determine the replica count, a common feasible token cap, `--mem-fraction-static`, CPU allocation, and saturation concurrency for that card. H200 may fit a larger KV budget or additional replicas, but this guide does not prescribe unverified values: repeat the sizing and saturation procedure and inspect every replica's actual allocation.


## How We Found This

This recipe grew out of the serving profiling in [#907](https://github.com/sgl-project/sglang-omni/issues/907). Our profiling found substantial unused GPU capacity across several omni serving workloads, with strong host-dispatch-bound evidence in the tested ASR setup. From there we ran same-GPU DP experiments on [Higgs](https://sgl-project.github.io/sglang-omni/cookbook/higgs_tts.html) and [Moss](https://sgl-project.github.io/sglang-omni/cookbook/moss_tts_local.html) TTS models.

| Experiment | GPU signal | Controlled observation | Result | Interpretation |
|---|---|---|---|---|
| ASR single replica | GPU timeline 94.3% idle | throughput 0.90x at SM clock 0.455x; 0.31x at host CPU near 0.25x | sensitive to CPU, not to GPU compute | strong host-dispatch-bound causal evidence in this ASR setup |
| Higgs tuned single | SM Active about 29%, GPU idle about 71% | throughput plateaued, worker fully driven | 1.00x normalized | clear reclaimable GPU headroom, but not the full ASR causal closure |
| Higgs DP2 without MPS | SM Active about 37 to 38%, GPU idle about 62 to 63% | added a second same-card server process | about 1.24x normalized | the second process reclaims part of the idle gap; host scheduling and long-tail batching can both contribute |
| Higgs DP with MPS | see the pinned case study in Evaluate | each replica saturated, MPS attachment confirmed | 1.4 to 2.1x nominal, repeated | MPS-enabled saturated runs produced the largest gains observed in the later pinned tests. |

ASR is the strongest host-bound evidence. Higgs started as a gray zone but clearly leaves GPU headroom at a tuned single replica. Running several replicas as separate processes changes host execution, scheduling, and long-tail behavior, and it is not the same as enlarging one replica's batch. Without MPS the CUDA contexts mostly time-slice and recover only part of the idle; MPS lets kernels from different processes run concurrently when resources permit, and the later MPS-enabled saturated runs produced the largest gains observed in the pinned tests.

## Reproduce the results

We release our early results and the guidance to reproduce them below.

### Prepare the baseline

The single-replica baseline decides whether same-GPU DP is worth it, and an under-driven baseline makes DP look better than it is. Tune and measure one replica first, then treat its throughput, latency, and GPU utilization as the number every DP configuration has to beat.

* **Sweep concurrency to the plateau.** Raise client concurrency until throughput stops climbing, and read the scheduler log lines (`#running-req`, `#queue-req`) at each step rather than assuming a good operating point.
* **Know the admission limit.** Higgs serves with `max_running_requests=64` and `cuda_graph_max_bs=64` by default; both can be raised via `sgl-omni serve --max_running_requests N --cuda_graph_max_bs N` (the CUDA-graph capture range must cover the admission limit, and raising it costs capture memory). Whether the default cap binds depends on the runtime, so check the queue, do not assume.
* **Separate client from server.** Client concurrency is not the active generation batch: requests beyond the admission limit wait in the scheduler queue, and requests also spend time in the other pipeline stages.
* **Prerequisites.** NVIDIA CUDA MPS available with GPU compute mode `Default`, so a per-user daemon needs no root; enough GPU memory for every replica, each sized with `--mem-fraction-static` plus a roughly fixed per-replica overhead (weights, codec, MPS context); non-overlapping CPU core blocks, one per replica, on the GPU's NUMA node (on SMT machines logical CPUs `N` and `N + ncores` are often the same physical core, so check `lscpu -e=CPU,CORE,NODE`); and enough offered concurrency to saturate each replica, not just the pool.


### Evaluate

Whether same-GPU DP helps is easy to measure incorrectly, so hold the comparison to the same discipline for every configuration:

| Control | Why it matters |
|---|---|
| tune the single replica to its throughput plateau | keeps the baseline from being artificially weak |
| hold total GPU and CPU resources fixed | separates replica splitting from simply adding resources |
| give each replica dedicated CPU cores | keeps replicas from contending for host dispatch |
| saturate each replica separately | keeps the DP pool from being under-fed |
| pin software and runtime settings | makes the comparison reproducible |
| report latency and unsuccessful runs | avoids showing only the best throughput |

## Case Study on H100 with Higgs TTS Model

One H100 80 GB (driver 580.126.20 / CUDA 13), sglang-omni `a78de4cb`, sglang `0.5.12.post1`, `bosonai/higgs-tts-3-4b` (snapshot `7556c17e`), `/v1/audio/speech`, seed-tts-eval EN, 300 samples per client, default `max_running_requests=64` / `cuda_graph_max_bs=64`, 32 server cores of the GPU's NUMA node split per replica, one client per replica on the SMT-sibling cores, fresh servers per run, interleaved on a shared host. Every attempted run is reported.

| Configuration | Nominal throughput | Relative to single | Run outcome |
|---|---:|---:|---|
| Single c96 | 21.7 to 22.1 qps | 1.0x | 4/4 completed |
| DP2 + MPS, 2 x c64 | 31.5 to 37.7 qps | 1.4 to 1.7x | 3 nominal of 5 attempts |
| DP3 + MPS, 3 x c64 | 39.9 to 46.9 qps | 1.8 to 2.1x | 2 nominal and 1 degraded of 4 attempts |

The failures: one DP2 benchmark run hit `cudaErrorMpsRpcFailure`, and one DP2 and one DP3 replica failed to start, all coinciding with host-load spikes. One DP3 run completed every request but at 13.3 qps, so it is marked degraded rather than excluded. The core-pinned single stayed within a few percent across all runs, and DP3 was not clearly repeatably better than DP2.

Note: the `--max-total-tokens` option makes per-replica KV sizing more explicit and comparable. It is not a direct fix for `cudaErrorMpsRpcFailure`, and the launch and runtime failure rate has not been re-measured with it in place; the failures in the table reflect the runs as recorded.

The #907 profiling, this repeated case study, and the reviewer verification below are three separate measurement series. They ran on different dates and load, and in some cases different software, so they should not be compared by absolute QPS; the differences between roughly 61, 21, and 29.9 qps are not attributed to a single cause.

> A separate reviewer verification on the same pinned software revision measured 29.9, 59.7, and 64.5 qps for single, DP2, and DP3. Absolute throughput differed between the two runtime environments, including different observed admission behavior, so the two series should not be combined. Both nevertheless showed a clear DP gain once every configuration was saturated.

To measure your own setup, check whether one tuned replica is below GPU saturation under your real workload before adopting DP:

```bash
nvidia-smi dmon -i $GPU_ID -s um -d 5                        # coarse utilization
nsys profile --gpu-metrics-devices $GPU_ID --gpu-metrics-set gh100 \
  -d 60 -o one_replica -f true sleep 63                      # device-level SM-active
```

Low SM activity at the tuned single replica's peak may indicate reclaimable headroom; confirm it with a controlled DP comparison before relying on it. If SM activity is already near the ceiling, stop here.

## Limits and next steps

1. **Generality is not fully validated.** Beyond the pinned H100 Higgs case study, we also ran related experiments on H200 and used SGLang to serve Qwen3-4B directly; both lines of work largely confirmed the same-GPU DP gains. Space and time limit how completely we can present those results here, and the measurements are not yet as polished as we would like. We believe same-GPU DP is a promising direction for smaller models on GPUs with ample memory and compute headroom, but the experimental coverage is still incomplete.

2. **Strictly equal per-replica KV size.** Sequential starts with the same `--mem-fraction-static` do not yield the same KV pool. Set each replica's KV size directly (e.g. a common `--max-total-tokens`) so capacities are strictly identical; intuition favors that for peak efficiency, but equal-versus-asymmetric budgets still need dedicated experiments and a sizing procedure that generalizes across cards.

3. **Router and scheduler still need a deeper dive.** Both the router and the SGLang Omni scheduler need further optimization. On the router side, better routing strategies for a colocated pool are clearly required. On the scheduler side, a more ambitious question is whether we can borrow the spirit of LLM prefill–decode (PD) disaggregation: keep one large shared KV cache and let multiple replicas share it. That direction is extremely challenging, and we believe the potential payoff is correspondingly large.

Same-GPU DP with MPS can recover idle GPU time on host- or dispatch-bound serving today, but broader validation and the work above are still unfinished. If this direction interests you, or you have results from other models, GPUs, or workloads that confirm or challenge these findings, we would like to work with you.
