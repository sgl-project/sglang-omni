# CPU allocator: topology-aware core planning for colocated serving

Host-bound speech serving degrades sharply when the work sharing its CPUs is
placed anywhere it likes: past a contention threshold throughput can collapse
to ~40% of what the same server manages when that work is confined to a
bounded set of cores (see the measurement tables in the introducing PR). The
CPU allocator derives that placement for every process the server owns,
instead of the manual `numactl`/`taskset` recipes.

## Usage

```bash
sgl-omni serve --model-path <model> --cpu-allocator static
```

Modes:

| mode | behavior |
| --- | --- |
| `off` (default) | No affinity change; identical to today. |
| `static` | Plan once at startup from the CPU topology and pin every stage process. |
| `auto` | Plan the same way, but apply it only while the pipeline is short of CPU next to foreign load, and release it when that load goes away. |

At startup the allocator discovers the CPU/NUMA/SMT topology, grants whole
physical cores (both SMT siblings together) exclusively to declared serial
dispatch loops, spreads those grants across NUMA nodes by remaining
capacity, and leaves everything else on the node's shared pool. Grants are
not tied to the GPU's NUMA node: an ablation with identical exclusive
carving on the GPU-local vs the far socket measured no difference on H200,
so locality anchoring buys nothing there while capacity spreading prevents
every service from piling onto node 0. The plan and any degradations are
logged as one JSON line (`cpu_alloc plan: ...`).

The universe is `sched_getaffinity`, so a container cpuset or an outer
`taskset` bounds the plan. Exclusivity applies to the processes of this
server; keeping *other* tenants off those cores is the job of the outer
cpuset (cgroup, Docker `--cpuset-cpus`, or Kubernetes CPU manager).

## Choosing between static and auto

`static` applies the plan for the process lifetime. `auto` applies it only
while the tree is starved next to foreign load, after three consecutive
samples where it holds less than 60% of its declared cores while at least one
foreign core is busy, and releases it after three samples with no foreign
load. On our measurements applying the plan is throughput neutral in both
directions, so `auto` is there for operators who would rather the masks not
exist while the box is quiet, not because pinning under contention was worth
anything on its own.

## What the plan is worth

What moves throughput is that colocated work is confined to a bounded set of
cores, not that the serial loop holds exclusive ones. With a neighbour held
to the shared pool, Fun-ASR measured 125.5 QPS planned against 125.9
unplanned, both about 2.7x the 46.5 the same server manages when that
neighbour roams the whole cpuset. So the allocator earns its place by placing
every process this server owns, correctly and automatically, rather than by
the exclusive grant itself.

## Model declarations

A model opts in by declaring per-stage host costs
(`PipelineConfig.stage_cpu_costs()`): `serial-loop` for a dispatch loop that
needs exclusive physical cores, `gpu-bound` for everything else. A stage
without a declaration is `gpu-bound`, so a model that declares nothing is a
no-op even when the allocator is enabled.

Inside a container the plan uses the container's own CPU set
(`sched_getaffinity`) while reading core and SMT relationships from sysfs,
which stays visible and truthful under `--cpuset-cpus`. When a cpuset hands
out only one sibling of a physical core, only that sibling is planned; the
other one belongs to whoever owns the outer cpuset.

Shipped declarations:

| model | declared stages (exclusive physical cores) |
| --- | --- |
| Higgs TTS | tts_engine 1, vocoder 1 |
| Qwen3-ASR | asr 4 |
| Fun-ASR | asr 5 |
| Whisper | asr 4 |
| MOSS-TTS-Local | tts_engine 1, vocoder 1 |
| Qwen3-TTS | tts_engine 2, vocoder 1 |
| Fish S2-Pro | tts_engine 1, vocoder 1 |
| dots.tts | latent_engine 1, vocoder 1 |

## Capacity planning for colocated deployments

How many services fit on one machine safely: sum the exclusive cores of each
service's declaration plus a shared-pool allowance (2+ physical cores per
service), per NUMA node. The plan CLI computes NUMA/SMT-correct partitions:

```bash
# Replica core blocks for a same-GPU DP pool on GPU 0's NUMA node
python -m sglang_omni.cpu_alloc plan --replicas 3 --gpu-id 0
# Full topology dump for audits
python -m sglang_omni.cpu_alloc topology
```

Give each colocated service its own outer cpuset and enable the allocator
inside it:

```bash
# Docker: one lane per service, allocator partitions within the lane
docker run --cpuset-cpus 0-15,112-127 ... \
  sgl-omni serve --model-path FunAudioLLM/Fun-ASR-Nano-2512-hf --cpu-allocator static
```

```yaml
# Kubernetes: static CPU manager gives the pod an exclusive cpuset;
# the allocator partitions inside it (requests==limits, integral CPUs).
resources:
  requests: {cpu: "32", memory: "64Gi"}
  limits: {cpu: "32", memory: "64Gi"}
```

`examples/mps_dp/autodp.sh` uses the same planner for its per-replica
`CORE_BLOCKS` automatically.

## Observing contention in production

`GET /host_contention` reports foreign CPU load on the server's allowed
cpuset (the CI cpuset-contention sampling idea, ported to serving):

```json
{
  "cpuset": "0-15,112-127",
  "foreign_busy_cores_last": 0.1,
  "foreign_busy_cores_window_peak": 11.8,
  "foreign_busy_cores_peak": 11.8,
  "own_busy_cores_last": 4.2
}
```

A latency regression with `foreign_busy_cores_window_peak` near zero is a
real regression; one with a large peak is core theft.
