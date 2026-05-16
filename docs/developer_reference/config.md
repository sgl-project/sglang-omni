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
stages = [
    StageConfig(
        name="preprocessing",
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
        factory="...create_aggregate_executor",
        wait_for=["preprocessing", "image_encoder", "audio_encoder"],
        merge_fn="...merge_for_thinker",
        next="thinker",
    ),
    StageConfig(
        name="thinker",
        factory="...create_sglang_thinker_executor_from_config",
        factory_args={"speech_enabled": True},
        gpu=0,
        next=["decode", "talker_ar"],
        stream_to=["talker_ar"],
    ),
    StageConfig(name="decode", factory="...create_decode_executor", terminal=True),
    StageConfig(
        name="code2wav",
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
| `factory_args` | `dict[str, Any]` | `{}` | Arguments forwarded to the factory. The compiler may inject `model_path` and `gpu_id` if the factory accepts them and they are not already set. |
| `next` | `str`, `list[str]`, or `None` | `None` | Static downstream stage or stages for normal result routing. |
| `terminal` | `bool` | `False` | Marks a stage as terminal; terminal results are sent to the coordinator. |
| `gpu` | `int`, `list[int]`, or `None` | `None` | GPU id for the stage. `None` means CPU placement. A list is used for tensor parallel ranks. |
| `tp_size` | `int` | `1` | Number of tensor-parallel ranks. Must match `len(gpu)` when `gpu` is a list. |
| `wait_for` | `list[str]` or `None` | `None` | Upstream stages required before this stage can execute a request. |
| `merge_fn` | `str` or `None` | `None` | Dotted import path to the fan-in merge function. Required when `wait_for` is set. |
| `stream_to` | `list[str]` | `[]` | Streaming targets for chunks such as hidden states or codec codes. This is parallel to normal result routing. |
| `project_payload` | `dict[str, str]` | `{}` | Optional target-stage to dotted projection function mapping used before writing a downstream payload. |
| `relay` | `RelayConfig` or `None` | `None` | Per-stage relay override. If unset, relay device and defaults are inferred from stage placement and `PipelineConfig.relay_backend`. |

Routing rule: in current code, use `next` for static downstream routing or
`terminal=True` for a terminal stage. Validation requires at least one of those;
use exactly one by convention because terminal stages do not route downstream.
The refactor tracking draft includes a `route_fn` field for future data-driven
routing, but the current `StageConfig` schema does not accept `route_fn`.

Derived from stages:

- `entry_stage`: defaults to the first stage unless explicitly set on
  `PipelineConfig`
- `terminal_stages`: computed from stages with `terminal=True`
- `gpu_placement`: computed from stages with `gpu` set
- relay device: explicit `StageConfig.relay.device` when present; otherwise
  inferred by the compiler from `gpu` and `relay_backend`

## `PipelineConfig` Reference

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `model_path` | `str` | required | Hugging Face model id or local checkpoint path. |
| `stages` | `list[StageConfig]` | required | Ordered logical stage definitions. The first stage is the default entry stage. |
| `name` | `str` or `None` | `model_path` | Pipeline name. Used for reporting and runtime identification. |
| `entry_stage` | `str` or `None` | first stage | Optional override for the stage that receives new requests. |
| `relay_backend` | one of `shm`, `nccl`, `nixl`, `mooncake` | `shm` | Global relay backend used when creating per-stage relays. |
| `fused_stages` | `list[list[str]]` | `[]` | Validated as adjacent stage groups, but fusion is not implemented yet. |
| `runtime_overrides` | `dict[str, dict[str, Any]]` | `{}` | Per-stage factory argument overrides applied by the compiler. |
| `endpoints` | `EndpointsConfig` | IPC defaults | Endpoint allocation settings: `scheme`, `base_path`, and `base_port`. |
| `completion_endpoint` | `str` or `None` | `None` | Optional explicit coordinator completion endpoint. |
| `abort_endpoint` | `str` or `None` | `None` | Optional explicit coordinator abort broadcast endpoint. |
| `config_cls` | `str` or `None` | class name | Stored automatically and used when loading a saved config file. |

Derived values are computed from stages, not manually maintained:

- `resolved_entry_stage`: `entry_stage` if set, otherwise the first stage name
- `terminal_stages`: all stages with `terminal=True`
- `gpu_placement`: stage name to GPU id or TP GPU list for stages with `gpu`

`RelayConfig` is the per-stage data-transfer override. It currently contains
`slot_size_mb`, `credits`, `rank`, `world_size`, and `device`.

## Compiler and Runners

The compiler prepares runtime wiring from a pipeline config:

- validate stage names and static topology
- compute the entry stage and terminal stages
- allocate ZMQ endpoints
- resolve dotted factory, merge, and projection functions
- merge `factory_args` with `runtime_overrides`
- inject global values such as `model_path` and `gpu_id` into factory args when
  accepted by the factory
- build relay config from stage placement and relay backend
- wire stream targets and same-GPU stream fast paths

Single-process serving uses `compile_pipeline_core()` to build a coordinator and
in-process `Stage` objects.

Multi-process serving uses `MultiProcessPipelineRunner`. It prepares the same
runtime state, then builds one `StageGroup` per logical stage. A stage group
owns one or more OS processes for that logical stage.

```text
pipeline/
|-- stage_process.py    # StageProcessSpec and subprocess entrypoint
|-- stage_group.py      # StageGroup lifecycle for a logical stage
`-- mp_runner.py        # Cross-stage orchestration and coordinator ownership
```

The child process does not recompile the pipeline. The main process builds a
fully resolved, picklable `StageProcessSpec`; the child imports the stage
factory, builds the scheduler, constructs the `Stage`, signals ready, and runs.

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

For `tp_size > 1`, `StageGroup` spawns one process per TP rank. Each process
runs the stage scheduler and model worker with a different `tp_rank` and GPU.
NCCL collectives inside model forward keep TP ranks in lockstep.

Only rank 0 owns external stage IO:

- rank 0 receives ZMQ messages from the coordinator or previous stage
- rank 0 fans work and aborts out to follower ranks
- all ranks make the same scheduling decisions
- only rank 0 sends downstream results or terminal completions

Each TP stage gets its own NCCL port allocation so multiple TP groups can exist
inside one pipeline.
