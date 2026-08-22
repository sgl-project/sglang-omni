# Config

SGLang-Omni uses declarative config as the contract between model-specific
pipeline definitions and the model-agnostic runtime. `PipelineConfig` describes
the whole pipeline: model path, stage list, endpoints, relay backend, and global
runtime overrides. `StageConfig` describes one logical stage: how to construct
it, where it runs, where its normal results go, and whether it participates in
fan-in or streaming edges.

The config layer is intentionally static. It should make topology, placement,
and stage construction visible before the runtime starts; request-time behavior
belongs in stages, schedulers, model runners, and model-local payload logic.

## Declarative Config

Pipelines are declared with `PipelineConfig` and `StageConfig`.

Example:

```python
# Every non-TP stage must declare `process` explicitly — there is no implicit
# default. Each stage below runs in its own OS process; multiple stages can
# share an OS process by giving them the same `process` value (see
# `Qwen3OmniSpeechColocatedPipelineConfig` for that pattern).
stages = [
    StageConfig(
        name="preprocessing",
        process="preprocessing",
        factory="...create_preprocessing_executor",
        next=["image_encoder", "audio_encoder", "mm_aggregate"],
        project_payload={
            "image_encoder": "...project_preprocessing_to_image_encoder",
            "audio_encoder": "...project_preprocessing_to_audio_encoder",
            "mm_aggregate": "...project_preprocessing_to_mm_aggregate",
        },
    ),
    StageConfig(
        name="mm_aggregate",
        process="mm_aggregate",
        factory="...create_aggregate_executor",
        wait_for=["preprocessing", "image_encoder", "audio_encoder"],
        merge_fn="...merge_for_thinker",
        next="thinker",
    ),
    StageConfig(
        name="thinker",
        process="thinker",
        factory="...create_sglang_thinker_executor_from_config",
        factory_args={"speech_enabled": True},
        gpu=0,
        next=["decode", "talker_ar"],
        stream_to=["talker_ar"],
    ),
    StageConfig(
        name="decode",
        process="decode",
        factory="...create_decode_executor",
        terminal=True,
    ),
    StageConfig(
        name="code2wav",
        process="code2wav",
        factory="...create_code2wav_scheduler",
        gpu=1,
        terminal=True,
    ),
]
```

## `StageConfig` Reference

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `name` | `str` | required | Unique stage identifier. |
| `factory` | `str` | required | Dotted import path to the stage factory. |
| `factory_args` | `dict[str, Any]` | `{}` | Explicit arguments forwarded to the factory after runtime overrides and typed runtime fields are merged. Signature-dependent defaults such as `model_path`, `gpu_id`, and `total_gpu_memory_fraction` are not written here during runtime prep; the worker injects them after importing the factory. `gpu_id` is owned by placement and is **rejected** here (set the device via `gpu` instead). |
| `next` | `str`, `list[str]`, or `None` | `None` | Static downstream stage or stages for normal result routing. |
| `terminal` | `bool` | `False` | Marks a stage as terminal; terminal results are sent to the coordinator. |
| `route_fn` | `str` or `None` | `None` | Dotted function path for request-aware result routing. The function receives `(request_id, stage_output)` and returns a downstream stage name or list of stage names. |
| `gpu` | `int`, `list[int]`, or `None` | `None` | GPU id for the stage. `None` means CPU placement. A list is used for tensor parallel ranks. |
| `tp_size` | `int` | `1` | Number of tensor-parallel ranks. Must match `len(gpu)` when `gpu` is a list. |
| `process` | `str` or `None` | `None` | OS process group identifier. Non-TP stages with the same `process` value share a single OS process; today every non-TP stage must declare one explicitly (see also `_validate_general`). For TP stages, `process` is optional, must not be shared with another stage, and acts as a prefix for the derived rank-process names (`{process}_tp{rank}`); if unset, the stage name is used as the prefix. |
| `wait_for` | `list[str]` or `None` | `None` | Upstream stages required before this stage can execute a request. |
| `wait_for_fn` | `str` or `None` | `None` | Dotted function path for request-aware fan-in source selection. The function receives `(request_id, from_stage, payload)` and returns the active subset of `wait_for`, or `None` when the payload does not determine the subset yet. |
| `merge_fn` | `str` or `None` | `None` | Dotted import path to the fan-in merge function. Required when `wait_for` is set. |
| `stream_to` | `list[str]` | `[]` | Static superset of streaming targets for chunks such as hidden states or codec codes. This is parallel to normal result routing. |
| `stream_done_to_fn` | `str` or `None` | `None` | Dotted function path for request-aware stream-completion targets. The function receives `(request_id, stage_output)` and returns the active subset of `stream_to` targets that should receive the final done signal. |
| `project_payload` | `dict[str, str]` | `{}` | Optional target-stage to dotted projection function mapping used before writing a downstream payload. |
| `comm` | `CommConfig` or `None` | `None` | Per-stage communication pool and Mooncake options. Transport selection is derived by `CommRouter` from locality and placement. |

Routing rule: set exactly one of `next` or `terminal=True`. `route_fn` is an
optional request-aware override for stages that already declare `next`; keep
`next` as the static topology declaration for validation. `route_fn` is not
valid on terminal stages and must return downstream stage targets, not `None`.
Fan-in follows the same static-superset pattern: keep `wait_for` as the full set
of possible upstream stages, and use `wait_for_fn` only to select the active
per-request subset. The returned subset must be non-empty and contained in
`wait_for`; returning `None` keeps the request pending until another upstream
payload can resolve the active set.
When using `stream_done_to_fn`, keep `stream_to` as the static superset because
runtime prep derives stream receivers and same-GPU stream paths from it.

Derived from stages:

- `entry_stage`: defaults to the first stage unless explicitly set on
  `PipelineConfig`
- `terminal_stages`: computed from stages with `terminal=True`
- `gpu_placement`: computed from stages with `gpu` set
- communication transport: inferred from stage locality and placement
  (`local_object`, `cuda_ipc`, host shared memory, or `mooncake`)

## `PipelineConfig` Reference

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `model_path` | `str` | required | Hugging Face model id or local checkpoint path. |
| `stages` | `list[StageConfig]` | required | Ordered logical stage definitions. The first stage is the default entry stage. |
| `name` | `str` or `None` | `model_path` | Pipeline name. Used for reporting and runtime identification. |
| `entry_stage` | `str` or `None` | first stage | Optional override for the stage that receives new requests. |
| `processes` | `dict[str, ProcessConfig]` | `{}` | Sparse replica policy keyed by Process Name. Member stages come from `StageConfig.process`, so this never repeats them. Unlisted processes run one replica. |
| `runtime_overrides` | `dict[str, dict[str, Any]]` | `{}` | Per-stage factory argument overrides applied during runtime prep. |
| `env_defaults` | `dict[str, str]` | `{}` | Environment defaults applied before stage factory imports. Existing process values take precedence. |
| `endpoints` | `EndpointsConfig` | IPC defaults | Endpoint allocation settings. `base_path` controls where Unix-domain sockets are created. |
| `terminal_stages_fn` | `str` or `None` | `None` | Dotted function path for request-aware terminal-stage resolution. The function receives the normalized `OmniRequest` and returns terminal stage names for that request, or `None` to use static terminals. |
| `config_cls` | `str` or `None` | class name | Stored automatically and used when loading a saved config file. |

Derived values are computed from stages, not manually maintained:

- `resolved_entry_stage`: `entry_stage` if set, otherwise the first stage name
- `terminal_stages`: all stages with `terminal=True`
- `gpu_placement`: stage name to GPU id or TP GPU list for stages with `gpu`

`CommConfig` is the per-stage communication tuning block. It contains
`slot_size_mb`, `credits`, `mooncake_protocol`, `mooncake_hostname`, and
`mooncake_device_name`. It does not select a transport backend.

### Logical Processes and Replicas

`StageConfig.process` is the only declaration of process membership. Stages that
name the same process share one OS process; different names produce different
processes. A TP stage owns its process outright: it falls back to the stage name
when `process` is unset, and it cannot be shared with another stage.

The config compiler groups stages by Process Name, attaches the sparse
`processes` replica policy, and validates the resulting topology once. A
cross-process edge derived from `next`, `stream_to`, or `wait_for` is rejected
only when the model lists it in `process_local_edges()`. Membership is final
after this step; GPU memory fractions remain explicit stage configuration.

`ProcessConfig` is the value type of that mapping:

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `num_replicas` | `int >= 1` | `1` | Number of whole-process instances. |
| `replica_devices` | `int`, `list[int]`, comma-separated `str`, or `None` | `None` | Device ids for every replica, replica 0 included. A non-TP process with GPU stages needs one id per replica; a TP process needs `num_replicas x tp_size` ids. A pure CPU process must leave this unset. |

For example:

```yaml
stages:
  - name: decode
    process: tail
  - name: postprocess
    process: tail

processes:
  tail:
    num_replicas: 3
```

A replica copies a whole Process. With `num_replicas: N`, each member stage gets
N instances, and instances with the same index form one complete Process replica
named `P@rN` (`decode@r0` and `postprocess@r0` in `tail@r0`). TP rank processes
are named `P@rN_tpM`. The `@rN` suffix is reserved: user-declared stage and
process names must not use it.

`replica_devices` covers every replica including index 0, and overrides
`StageConfig.gpu` for the GPU stages of that process. This holds at
`num_replicas: 1` as well: the process keeps its unsuffixed names and produces
no replica bindings, but its GPU stages still move to the declared device. A
non-TP process with GPU stages needs one device id per replica; a TP process
with size T needs N x T ids, split into per-replica rank groups. A pure CPU
process must not declare devices.

Every GPU-stage factory governed by `replica_devices` must accept an explicit
`gpu_id` keyword argument. Runtime preparation rejects factories without that
contract instead of translating placement into a legacy `device` argument.

Because a replicated process resolves its devices here rather than from
`StageConfig.gpu`, replica placement is included in GPU colocation validation.
When `require_memory_fraction_for_colocation` is `true`, every GPU shared by
multiple process groups requires explicit
`runtime.resources.total_gpu_memory_fraction` values. Setting it to `false`
disables that requirement only for colocation declared by `StageConfig.gpu`.
Sharing involving final placement from `replica_devices` always requires
fractions, including when `num_replicas: 1`. Fractions that are provided still
count toward `max_total_gpu_memory_fraction_per_gpu`.

At admission the coordinator picks one replica per replicated Process and
projects that index onto the Process's member stages, so members always agree
and different processes choose independently. The default round-robin policy
advances every replicated Process once per admission, so Processes with equal
replica counts stay index-aligned. Unequal counts are legal, but their selected
indices periodically diverge; stream-connected stages may then route across
GPUs through the relay even when equal-index replicas are placed on the same
GPU. Use equal counts when a topology relies on same-index, same-GPU pairing.

## Runtime Prep and Runner

Runtime prep builds the resolved state used by the runner:

- validate stage names and static topology
- compute the entry stage and terminal stages
- allocate ZMQ endpoints
- carry dotted factory, merge, route, and projection paths into worker specs
- merge `factory_args` with `runtime_overrides` and typed runtime fields without
  importing stage factories
- prepare signature-dependent defaults such as `model_path`, `gpu_id`, and
  `total_gpu_memory_fraction`; the worker injects them after importing the
  factory when the factory accepts them
- build relay config from stage placement and relay backend
- wire stream targets and same-GPU stream fast paths

Serving uses `MultiProcessPipelineRunner` for both single-process and
multi-process topologies. Runtime prep compiles the logical process plan first,
expands process replicas, then resolves GPU placement, and finally builds the
physical process topology:

- every non-TP stage must declare `process` explicitly — there is no implicit
  default. Configs saved before this refactor are not auto-migrated: set
  `process="pipeline"` on every non-TP stage to recover the historical
  single-process behavior, or use any other shared/distinct process name to
  opt into the declarative multi-stage-per-process layout. See
  [Process Topology Migration](../basic_usage/process_topology_migration.md) for
  replacements for the removed CLI and config entries;
- explicit `stage.process` groups non-TP stages declaratively.

A process group may contain CPU stages and stages on at most one GPU. When
`require_memory_fraction_for_colocation` is enabled, multiple process groups may
share the same GPU only when GPU-stage memory budgets are explicit. Disabling
that requirement permits missing budgets; any budgets that are provided must
still fit the configured placement limit.

```text
pipeline/
|-- stage_workers.py    # StageLaunchConfig, subprocess entrypoint, StageGroup
|-- runtime_config.py   # endpoint/runtime-dir/placement prep
`-- mp_runner.py        # Cross-stage orchestration and coordinator ownership
```

The child process does not recompile the pipeline. The main process builds
fully resolved, picklable stage/process specs; the child imports stage
factories, builds schedulers, constructs `Stage` objects, signals ready, and
runs one or more non-TP stages in the same event loop.

## Tensor Parallelism

Tensor parallelism inside a stage is orthogonal to pipeline parallelism between
stages.

```python
StageConfig(
    name="thinker",
    factory="...",
    gpu=[0, 1, 2, 3],
    tp_size=4,
)
```

For `tp_size > 1`, the runner derives one process per TP rank. Each process runs
the stage scheduler and model worker with a different `tp_rank` and GPU. NCCL
collectives inside model forward keep TP ranks in lockstep. `StageConfig.process`
is optional for TP stages; if set, it names the logical TP process and acts as
the prefix for the derived per-rank process names (`{process}_tp{rank}`); if
unset, the stage name is used for both. TP ranks always own their OS process
exclusively — a TP stage's process group cannot host any other stage, regardless
of whether `process` is set or unset.

Only rank 0 owns external stage IO:

- rank 0 receives ZMQ messages from the coordinator or previous stage
- rank 0 fans work and aborts out to follower ranks
- all ranks make the same scheduling decisions
- only rank 0 sends downstream results or terminal completions

Each TP stage gets its own NCCL port allocation so multiple TP groups can exist
inside one pipeline.
