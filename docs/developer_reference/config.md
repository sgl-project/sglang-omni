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
| `factory_args` | `dict[str, Any]` | `{}` | Arguments forwarded to the factory. Runtime prep may inject `model_path` and `gpu_id` if the factory accepts them and they are not already set. |
| `next` | `str`, `list[str]`, or `None` | `None` | Static downstream stage or stages for normal result routing. |
| `terminal` | `bool` | `False` | Marks a stage as terminal; terminal results are sent to the coordinator. |
| `route_fn` | `str` or `None` | `None` | Dotted function path for request-aware result routing. The function receives `(request_id, stage_output)` and returns a downstream stage name or list of stage names. |
| `gpu` | `int`, `list[int]`, or `None` | `None` | GPU id for the stage. `None` means CPU placement. A list is used for tensor parallel ranks. |
| `tp_size` | `int` | `1` | Number of tensor-parallel ranks. Must match `len(gpu)` when `gpu` is a list. |
| `process` | `str` or `None` | `None` | OS process group identifier. Non-TP stages with the same `process` value share a single OS process; today every non-TP stage must declare one explicitly (see also `_validate_general`). For TP stages, `process` is optional and acts as a prefix for the derived rank-process names (`{process}_tp{rank}`); if unset, the stage name is used as the prefix. |
| `wait_for` | `list[str]` or `None` | `None` | Upstream stages required before this stage can execute a request. |
| `merge_fn` | `str` or `None` | `None` | Dotted import path to the fan-in merge function. Required when `wait_for` is set. |
| `stream_to` | `list[str]` | `[]` | Static superset of streaming targets for chunks such as hidden states or codec codes. This is parallel to normal result routing. |
| `stream_done_to_fn` | `str` or `None` | `None` | Dotted function path for request-aware stream-completion targets. The function receives `(request_id, stage_output)` and returns the active subset of `stream_to` targets that should receive the final done signal. |
| `project_payload` | `dict[str, str]` | `{}` | Optional target-stage to dotted projection function mapping used before writing a downstream payload. |
| `relay` | `RelayConfig` or `None` | `None` | Per-stage relay override. If unset, relay device and defaults are inferred from stage placement and `PipelineConfig.relay_backend`. |

Routing rule: set exactly one of `next` or `terminal=True`. `route_fn` is an
optional request-aware override for stages that already declare `next`; keep
`next` as the static topology declaration for validation. `route_fn` is not
valid on terminal stages and must return downstream stage targets, not `None`.
When using `stream_done_to_fn`, keep `stream_to` as the static superset because
runtime prep derives stream receivers and same-GPU stream paths from it.

Derived from stages:

- `entry_stage`: defaults to the first stage unless explicitly set on
  `PipelineConfig`
- `terminal_stages`: computed from stages with `terminal=True`
- `gpu_placement`: computed from stages with `gpu` set
- relay device: explicit `StageConfig.relay.device` when present; otherwise
  inferred by runtime prep from `gpu` and `relay_backend`

## `PipelineConfig` Reference

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `model_path` | `str` | required | Hugging Face model id or local checkpoint path. |
| `stages` | `list[StageConfig]` | required | Ordered logical stage definitions. The first stage is the default entry stage. |
| `name` | `str` or `None` | `model_path` | Pipeline name. Used for reporting and runtime identification. |
| `entry_stage` | `str` or `None` | first stage | Optional override for the stage that receives new requests. |
| `relay_backend` | one of `shm`, `nccl`, `nixl`, `mooncake` | `shm` | Global relay backend used when creating per-stage relays. |
| `fused_stages` | `list[list[str]]` | `[]` | Validated as adjacent stage groups, but fusion is not implemented yet. |
| `runtime_overrides` | `dict[str, dict[str, Any]]` | `{}` | Per-stage factory argument overrides applied during runtime prep. |
| `endpoints` | `EndpointsConfig` | IPC defaults | Endpoint allocation settings: `scheme`, `base_path`, and `base_port`. |
| `completion_endpoint` | `str` or `None` | `None` | Optional explicit coordinator completion endpoint. |
| `abort_endpoint` | `str` or `None` | `None` | Optional explicit coordinator abort broadcast endpoint. |
| `terminal_stages_fn` | `str` or `None` | `None` | Dotted function path for request-aware terminal-stage resolution. The function receives the normalized `OmniRequest` and returns terminal stage names for that request, or `None` to use static terminals. |
| `config_cls` | `str` or `None` | class name | Stored automatically and used when loading a saved config file. |

Derived values are computed from stages, not manually maintained:

- `resolved_entry_stage`: `entry_stage` if set, otherwise the first stage name
- `terminal_stages`: all stages with `terminal=True`
- `gpu_placement`: stage name to GPU id or TP GPU list for stages with `gpu`

`RelayConfig` is the per-stage data-transfer override. It currently contains
`slot_size_mb`, `credits`, `rank`, `world_size`, and `device`.

## Runtime Prep and Runner

Runtime prep builds the resolved state used by the runner:

- validate stage names and static topology
- compute the entry stage and terminal stages
- allocate ZMQ endpoints
- resolve dotted factory, merge, and projection functions
- merge `factory_args` with `runtime_overrides`
- inject global values such as `model_path` and `gpu_id` into factory args when
  accepted by the factory
- build relay config from stage placement and relay backend
- wire stream targets and same-GPU stream fast paths

Serving uses `MultiProcessPipelineRunner` for both single-process and
multi-process topologies. Runtime prep first resolves GPU placement, then
process topology:

- every non-TP stage must declare `process` explicitly — there is no implicit
  default. Configs saved before this refactor are not auto-migrated: set
  `process="pipeline"` on every non-TP stage to recover the historical
  single-process behavior, or use any other shared/distinct process name to
  opt into the declarative multi-stage-per-process layout;
- explicit `stage.process` groups non-TP stages declaratively.

A process group may contain CPU stages and stages on at most one GPU. Multiple
process groups may share the same GPU only when GPU-stage memory budgets are
explicit and fit the configured placement limit.

```text
pipeline/
|-- stage_process.py    # StageProcessSpec and subprocess entrypoint
|-- stage_group.py      # StageGroup lifecycle for topology process groups
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
is optional for TP stages; if set, it acts as the prefix for the derived
per-rank process names (`{process}_tp{rank}`); if unset, the stage name is used
as the prefix. TP ranks always own their OS process exclusively — a TP stage's
process group cannot host any other stage, regardless of whether `process` is
set or unset.

Only rank 0 owns external stage IO:

- rank 0 receives ZMQ messages from the coordinator or previous stage
- rank 0 fans work and aborts out to follower ranks
- all ranks make the same scheduling decisions
- only rank 0 sends downstream results or terminal completions

Each TP stage gets its own NCCL port allocation so multiple TP groups can exist
inside one pipeline.
