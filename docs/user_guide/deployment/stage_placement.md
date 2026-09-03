# Stage placement

SGLang-Omni serves a model as a pipeline of heterogeneous stages. Placement
decides which GPU or CPU owns each stage, which stages share a device or
process, and which inter-stage transfers cross a device or process boundary.

Start from a checked-in model configuration. Placement is part of a validated
deployment, not a theoretical GPU-memory packing exercise.

## Read the pipeline first

Every model cookbook lists its high-level topology. The declarative
`PipelineConfig` is the source of truth for exact stage names, GPU assignments,
process groups, tensor parallelism, streaming edges, and terminal stages.

Typical stage characteristics differ:

| Stage type | Common constraint |
|---|---|
| Preprocessing and response formatting | CPU latency and worker concurrency |
| Multimodal encoder | Activation memory and bursty GPU compute |
| Autoregressive engine | Weights, KV cache, decode latency, CUDA Graph memory |
| Acoustic or codec tail | Fixed per-request state and batched GPU compute |
| Vocoder/code2wav | Streaming cadence, workspace, and first-audio latency |

Measure the actual pipeline rather than assuming that the largest checkpoint
name identifies the bottleneck.

## Placement strategies

### Disaggregated

Place heavy or independently scaling stages on separate GPUs. This avoids
device contention and lets each stage use a larger memory budget, but transfers
cross device boundaries and may add single-request latency.

### Colocated

Share one GPU between stages when their combined weights, KV pools, runtime
state, graph captures, workspaces, and activation peaks fit. Colocation can
reduce transfer overhead and GPU count, but only an explicit validated memory
budget makes the configuration safe.

### Tensor parallel

Shard one stage across multiple GPUs when a single device cannot fit it or when
the model's validated topology benefits from tensor parallelism. Tensor
parallelism inside a stage is separate from pipeline placement between stages.
The stage's GPU list and TP size must agree.

## Process placement and GPU placement

These are separate decisions:

- `StageConfig.gpu` selects CPU, one GPU, or a TP GPU list.
- `StageConfig.process` groups non-TP stages into OS processes.
- stages can share a GPU while remaining in different processes;
- stages in one process can use process-local dispatch on eligible edges;
- TP ranks always run in exclusive rank processes.

Process colocation does not merge stage ownership. Routing, scheduling, aborts,
streaming, and terminal completion remain stage-level responsibilities.

## Memory budgets

Budget every GPU resident, not just the AR model:

- model weights and quantization metadata;
- KV cache and maximum running requests;
- CUDA Graph and compilation buffers;
- encoder, vocoder, or acoustic model weights;
- per-request fixed pools;
- activations, communication buffers, and allocator fragmentation.

Some pipelines auto-size an AR KV pool when neighboring stages use known typed
resource budgets. An explicitly pinned `mem_fraction_static` can bypass that
carve-out, so leave headroom for colocated stages yourself.

Runtime preparation rejects colocated process groups that span multiple GPUs
or omit required explicit budgets. A configuration passing startup validation
still needs workload validation at its intended concurrency and request length.

## Select a topology

1. Start with the cookbook's recommended checked-in config.
2. Confirm the exact stage topology and terminal outputs.
3. Measure per-stage steady and peak memory, utilization, and latency at the
   target traffic pattern.
4. Isolate a stage when it is the measured contention bottleneck or cannot fit
   within a safe shared budget.
5. Colocate light stages only when the combined peak allocation and performance
   are validated.
6. Re-run correctness, streaming, latency, throughput, and overload checks.
7. Record hardware, stage-to-GPU mapping, process mapping, and config revision.

Single-stream and concurrent traffic can prefer different topologies. A
cross-GPU handoff can hurt single-request latency even when isolating a busy
stage improves throughput and tail latency at concurrency.

## Qwen3-Omni example

Qwen3-Omni provides single-GPU colocated H20, H100, and H200 profiles with
explicit per-stage memory fractions. Its standard disaggregated speech layout
keeps the thinker and talker on separate GPUs and shares code2wav with the
thinker. Use the [Qwen3-Omni cookbook](../../cookbook/qwen3_omni.md) and
[Omni model usage](../../basic_usage/qwen3_omni.md) for validated commands and
the measured model-specific placement tradeoffs.

Do not copy Qwen3-Omni fractions to another model or GPU. The typed config is a
qualification point for that exact pipeline and hardware tier.

## Related documentation

- [Pipeline architecture](../../developer_reference/pipeline.md)
- [Configuration reference](../../developer_reference/config.md)
- [Admission control](../advanced_features/admission_control.md)
- [Benchmark methodology](../../benchmarks/methodology.md)
- [Supported models](../../supported_models.md)
