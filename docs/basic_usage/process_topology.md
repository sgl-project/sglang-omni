# Process Topology, Replicas, and GPU Sharing

Four mechanisms decide how an SGLang-Omni pipeline occupies its GPUs. They
compose, and each answers a different question:

1. **Process topology** defines which stages run together.
2. **Process replicas** define how many copies run.
3. **Placement** defines where they run.
4. **CUDA MPS** improves kernel scheduling between processes on one GPU.
5. **CUDA IPC weight sharing** removes duplicate weight copies on one GPU.

| Mechanism | Configured per | What it changes | What it does not do |
|---|---|---|---|
| Process topology | `StageConfig.process` | non-TP logical Process membership; TP materializes per rank | replica count, placement |
| Process replica | `PipelineConfig.processes[<name>]` | copies a whole logical Process, sticky per request | model parallelism for one request, MPS |
| Placement | `replica_devices` | the GPU each replica or rank lands on | GPU context scheduling |
| CUDA MPS | `mps` | kernel overlap between colocated CUDA contexts | routing, weight or KV sharing |
| CUDA IPC weight sharing | `weight_share` | followers alias the leader's immutable weights | KV, CUDA graphs, sampler, request state |

Which one you want:

| Your question | Mechanism | What it will not solve |
|---|---|---|
| Which stages share one OS process and its local state? | Process topology | replica count and GPU placement |
| Which bottleneck Process needs more capacity? | Process replica | model parallelism for a single request |
| Which GPU does each replica run on? | `replica_devices` | CUDA context scheduling |
| How do colocated processes use idle compute? | CUDA MPS | replica creation and request routing |
| Full replicas on one GPU do not fit in VRAM? | CUDA IPC weight sharing | KV, CUDA graph, and request-state sharing |

## Stages and process topology

A stage is one logical execution unit of the pipeline DAG. It declares its
factory, wiring (`next`, `stream_to`, `wait_for`), GPU, TP size, runtime
resources, and process name.

For a non-TP stage, `StageConfig.process` defines Process membership. Stages
sharing a Process Name share an OS process, Python heap, asyncio event loop,
and local dispatch path; GPU members also share one CUDA context. A TP stage
owns its logical Process and materializes one OS process per rank.

A logical Process is a grouping, spawn, and placement boundary. It is not a
recovery boundary: any child stage process dying still stops the pipeline.

```yaml
stages:
  talker_ar:
    process: talker_ar
  code2wav:
    process: code2wav
```

Inspect the resolved result with `sgl-omni config resolve --config <config.yaml>
--show config`.

## Process replicas

A replica copies a whole logical Process, not a single stage. Stages inside one
Process are copied together under the same replica index. The runtime names
physical instances `<name>@rN`, while models and routes keep referring to
logical names.

At admission the coordinator picks one replica per replicated Process the
request touches. That binding rides the message and stays fixed for the whole
request lifetime, across payload, stream, completion, and abort paths. The
default policy is per-Process thread-safe round robin, chosen independently for
each Process.

```yaml
processes:
  talker_ar:
    num_replicas: 2
    replica_devices: [1, 2]
  code2wav:
    num_replicas: 2
    replica_devices: [1, 2]
```

## Placement

`replica_devices` sets the GPU each replica uses, including replica 0, and
overrides the placement of every GPU stage inside that replica. CPU stages are
unaffected and stay on the host.

A GPU Process with `num_replicas > 1` must declare `replica_devices`: `N`
device ids for a non-TP Process, `N x T` for a Process with TP size `T`. Every
GPU stage in that Process must also come from a factory that declares a
`gpu_id` parameter, otherwise startup refuses the placement by name; not all
model stages do yet.
Different replicas may repeat a device id, which is how same-GPU data
parallelism is expressed:

```yaml
processes:
  code2wav:
    num_replicas: 2
    replica_devices: [1, 1]
```

That places two non-TP replicas on GPU 1. It does not mean two ranks of one TP
replica may share a GPU.

When `replica_devices` colocates Process groups on one GPU, every GPU stage
involved must declare `gpu_memory_fraction`, whether or not the general
colocation check is enabled. The value is a placement-time budget, not a
runtime memory limit:

```yaml
stages:
  code2wav:
    gpu_memory_fraction: 0.014
```

## CUDA MPS

MPS only schedules multiple CUDA contexts on one GPU. It does not create
replicas, choose routes, or share weights, KV, or CUDA graphs. When a single
context already saturates the GPU, MPS can add contention and tail latency
rather than throughput.

The runtime manages the daemon itself. Modes (`--mps` on the CLI or `mps:` in
the config, default `off`):

* `off`: MPS is never touched.
* `auto`: enabled on every GPU hosting two or more single-GPU, non-TP CUDA
  processes of this pipeline, including process replicas.
* `on`: one eligible process is enough, and an MPS-incapable platform is a hard
  error instead of a warning.

```bash
sgl-omni serve --config <config.yaml> --mps auto --port 8091
```

The daemon is shared per physical GPU, keyed by device UUID. See
[Same-GPU Data Parallelism with CUDA MPS](mps_dp.md) for the full lifecycle,
verification, and operator notes.

## CUDA IPC weight sharing

Weight sharing removes duplicate weight VRAM; it does not change scheduling.
Within one sharing group, the lowest replica index is the leader: it loads the
checkpoint and publishes CUDA-IPC handles. Followers build the same module tree
with dummy weights and alias the leader's immutable parameters and buffers by
assignment. KV cache, CUDA graphs, sampler state, request state, and any tensor
the architecture's share policy marks replica-private stay per replica.

Use it to make a replica count fit, or to free VRAM for KV, CUDA graphs, and
more replicas. It is a capacity mechanism, not a throughput optimization.

```bash
sgl-omni serve --config <config.yaml> --mps on --weight-share on --port 8091
```

`weight_share` is `off` or `on` (default `off`). When `on`, every logical
Process whose replicas repeat a GPU id becomes a sharing group on that GPU, and
the runtime assigns roles itself; `SGLANG_OMNI_WEIGHT_SHARE` must not be set in
the environment. Replicas of the same Process that are alone on their GPU keep
loading their own weights and still serve requests normally.

Requirements, all checked before any process is spawned:

* the sharing Process has exactly one SGLang engine stage, with `tp=pp=1`;
* that stage pins `max_total_tokens`, because a follower attaches after its
  dummy weights are freed and memory profiling cannot derive a stable KV
  budget;
* the architecture is on the share-policy allowlist, which the engine enforces
  when the leader loads.

Because sharing rides on process replicas, it is available to a model only when
that model's GPU stage factories accept `gpu_id` as described under Placement.
Models on the allowlist whose factories do not can still share weights through
the multi-serve `examples/mps_dp/launch.sh` recipe.

Followers are spawned only after every leader is ready, and are shut down
before their leader. Sharing is a whole-group lifecycle: the leader must
outlive its followers, online weight updates are refused while sharing is
active, and a dead leader fails the pipeline rather than serving aliased
memory. Restart the pipeline as a whole.

Weight sharing does not require MPS, and MPS does not require weight sharing.
`--mps on --weight-share on` is the usual same-GPU DP combination: MPS gives
the replicas kernel overlap, weight sharing gives them room to fit.

## Putting it together

Configure in this order:

1. decide which stages share a process;
2. choose the replica count;
3. place each replica on a GPU;
4. enable MPS when processes share a GPU;
5. enable weight sharing when duplicate weights are the capacity limit.

### Replicas across GPUs

```yaml
config_cls: Qwen3OmniSpeechPipelineConfig
name: qwen3-omni-speech-replica2
model_path: Qwen/Qwen3-Omni-30B-A3B-Instruct

stages:
  talker_ar:
    gpu_memory_fraction: 0.123
  code2wav:
    gpu_memory_fraction: 0.014

processes:
  talker_ar:
    num_replicas: 2
    replica_devices: [1, 2]
  code2wav:
    num_replicas: 2
    replica_devices: [1, 2]
```

Resolved layout:

```
GPU 0: image_encoder + audio_encoder + thinker
GPU 1: talker_ar@r0 + code2wav@r0
GPU 2: talker_ar@r1 + code2wav@r1
```

Run it with `sgl-omni serve --config
examples/configs/qwen3_omni_speech_replica2.yaml --port 8091`; the full file is
[`qwen3_omni_speech_replica2.yaml`](https://github.com/sgl-project/sglang-omni/blob/main/examples/configs/qwen3_omni_speech_replica2.yaml).

### Replicas on one GPU

Repeat the device id and declare the memory budget for every GPU stage on
that card. Without `--mps`, the replicas time-slice the GPU. MOSS TTS local is
used here because its engine factory accepts `gpu_id` and its architecture is
on the weight-share allowlist, so the same file also serves the next example:

```yaml
config_cls: MossTTSLocalPipelineConfig
name: mossl
model_path: OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5

stages:
  preprocessing:
    gpu: 0
    gpu_memory_fraction: 0.05
  tts_engine:
    gpu: 0
    gpu_memory_fraction: 0.35
    engine:
      mem_fraction_static: 0.30
      max_total_tokens: 30000
  vocoder:
    gpu: 0
    gpu_memory_fraction: 0.15

processes:
  pipeline:
    num_replicas: 2
    replica_devices: [0, 0]
```

### Replicas on one GPU with MPS and weight sharing

Same config, plus the two runtime flags. `max_total_tokens` above is what
weight sharing requires of the engine stage:

```bash
sgl-omni serve --config <config.yaml> --mps on --weight-share on --port 8091
```

The leader holds the shared weights; the follower attaches over CUDA IPC and
carries only its own KV, graphs, and request state.

## Performance and correctness

Replicas mainly help under queueing and higher concurrency. At low concurrency
they can be slightly slower, because the parallelism gain does not cover
routing and process overhead. MPS helps only when colocated processes have
overlappable GPU work. Weight sharing saves memory and does not by itself
improve scheduling.

Validate a topology change on: serial output consistency, concurrent routing
and request isolation, abort and recovery, clean exit of processes, ports, and
GPU memory, and, when enabled, MPS attachment and the weight-share lifecycle.

## Migration

Removed interfaces and their replacements are covered in
[Process Topology Migration](process_topology_migration.md): express process
membership with `StageConfig.process`, and replica count and placement with the
top-level `processes` block.
