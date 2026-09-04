# Config

SGLang-Omni uses declarative config as the contract between model-specific
pipeline definitions and the model-agnostic runtime. `PipelineConfig` describes
the whole pipeline: model path, stage list, endpoints, and relay backend.
`StageConfig` describes one logical stage: how to construct it, where it runs,
where its normal results go, and whether it participates in fan-in or streaming
edges.

The config layer is intentionally static. It should make topology, placement,
and stage construction visible before the runtime starts; request-time behavior
belongs in stages, schedulers, model runners, and model-local payload logic.

## Declarative Config

Pipelines are declared with `PipelineConfig` and `StageConfig` in the model's
`config.py`. Stage topology — which stages exist, how they route, where
requests enter — lives here and only here; config files and CLI flags override
settings on these stages but never add or remove them.

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
        factory_path="...create_preprocessing_executor",
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
        factory_path="...create_aggregate_executor",
        wait_for=["preprocessing", "image_encoder", "audio_encoder"],
        merge_fn="...merge_for_thinker",
        next="thinker",
    ),
    EngineStageConfig(               # drives an SGLang engine, so engine.* exists
        name="thinker",
        process="thinker",
        factory_path="...create_sglang_thinker_executor_from_config",
        factory=FactoryArgs(max_seq_len=8192),
        gpu=0,
        next=["decode", "talker_ar"],
        stream_to=["talker_ar"],
    ),
    StageConfig(
        name="decode",
        process="decode",
        factory_path="...create_decode_executor",
        terminal=True,
    ),
]
```

## Consumer Groups

Stage settings are grouped by the module that consumes them:

| Group | Consumer | Examples |
| --- | --- | --- |
| stage top level | parent process: placement, process planning, wiring | `gpu`, `tp_size`, `process`, `gpu_memory_fraction` |
| `engine.*` | SGLang `ServerArgs` (only on `EngineStageConfig` stages) | `mem_fraction_static`, `max_running_requests`, `disable_cuda_graph` |
| `factory.*` | the stage factory's signature | `dtype`, `max_seq_len`, `max_concurrency`, `enable_async_decode` |

Each group declares its commonly tuned fields, which validate eagerly. **Any
other key passes through untouched**: the group vocabularies belong to their
consumers, so the entry side never format-checks unknown keys. A free-form
`factory.*` key reaches the stage factory under its own name; a free-form
`engine.*` key travels in the `server_args_overrides` mapping to SGLang,
which rules on it. This is how factory-specific knobs are passed — what used
to be a `factory_args` entry is now written under `factory.*`:

```yaml
stages:
  latent_engine:
    factory:
      num_steps: 4          # not a declared FactoryArgs field; passed to the
                            # factory as num_steps=4, validated by its signature
```

A value the factory does not accept is an error at stage construction, not a
silent no-op.

## Setting Values: YAML and CLI

There are exactly two user-facing spellings of one path language.

**YAML** — the `stages:` mapping, keyed by stage name:

```yaml
config_cls: MossTTSPipelineConfig
model_path: OpenMOSS-Team/MOSS-TTS

stages:
  tts_engine:
    tp_size: 2
    engine:
      mem_fraction_static: 0.7
  vocoder:
    factory:
      dtype: bfloat16
```

Entries merge by name: a field the file does not write keeps the model's
default. Naming a stage the config class does not define is an error that
lists the real stage names.

**CLI** — dotted flags with the `stages.` prefix implied; the flag starts
from the stage name exactly as the mapping does:

```bash
sgl-omni serve --config omni.yaml \
    --tts_engine.tp_size 2 \
    --tts_engine.engine.mem_fraction_static 0.7 \
    --vocoder.factory.dtype bfloat16 \
    --vocoder.process vocoder
```

CLI text is coerced by the declared field type; free-form group keys fall
back to YAML scalar parsing (`true` → bool, `7` → int). Writing one path
twice at the same precedence is refused, never silently last-one-wins.
The command line outranks the config file; an explicit dotted path outranks
the broadcast flag below.

**Shared values** — the `shared:` selector list writes one value into several
stages at once:

```yaml
shared:
  - select:
      stages: [preprocessing, latent_engine]   # or engine: true, exclude: [...]
    factory:
      num_steps: 4
```

The entry expands to one patch per matched stage before resolution. An
explicit per-stage entry (under `stages:` or as a dotted flag) overrides the
expansion; two `shared` entries writing one leaf conflict.

**Broadcast flag** — `--mem-fraction-static 0.7` fans one value out to every
SGLang engine stage's `engine.mem_fraction_static`. It is the one convenience
flag left; a dotted per-stage spelling overrides it without being a conflict.

**Inspection** — `sgl-omni config resolve` prints the configuration a launch
with the same arguments would use (`--show config|diff|provenance`), and
`sgl-omni config explain PATH` names the source that set a value and what it
overrode. Both run the same merge as `serve`.

## `StageConfig` Reference

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `name` | `str` | required | Unique stage identifier. Set by the config class; a config file addresses stages by name and never renames them. |
| `factory_path` | `str` | required | Dotted import path to the stage factory. |
| `engine` | `EngineArgs` or `None` | `None` | SGLang ServerArgs overrides. Exists only on `EngineStageConfig` stages; writing it elsewhere is a path error. |
| `factory` | `FactoryArgs` | empty | Constructor kwargs for the stage factory, passed by field name. Unknown keys pass through. |
| `next` | `str`, `list[str]`, or `None` | `None` | Static downstream stage or stages for normal result routing. |
| `terminal` | `bool` | `False` | Marks a stage as terminal; terminal results are sent to the coordinator. |
| `route_fn` | `str` or `None` | `None` | Dotted function path for request-aware result routing. The function receives `(request_id, stage_output)` and returns a downstream stage name or list of stage names. |
| `gpu` | `int`, `list[int]`, or `None` | `None` | GPU id for the stage. `None` means CPU placement. A list is used for tensor parallel ranks. |
| `tp_size` | `int` | `1` | Number of tensor-parallel ranks. Must match `len(gpu)` when `gpu` is a list. |
| `gpu_memory_fraction` | `float` or `None` | `None` | Per-stage-rank budget as a fraction of total physical GPU memory. Required per stage when multiple processes share one GPU. |
| `process` | `str` or `None` | `None` | OS process group identifier. Non-TP stages with the same `process` value share a single OS process; every non-TP stage must declare one explicitly. For TP stages, `process` is optional and acts as a prefix for the derived rank-process names (`{process}_tp{rank}`); if unset, the stage name is used as the prefix. |
| `env` | `dict[str, str]` | `{}` | Per-stage env defaults applied in this stage's worker process at spawn; never overrides `os.environ`. |
| `wait_for` | `list[str]` or `None` | `None` | Upstream stages required before this stage can execute a request. |
| `wait_for_fn` | `str` or `None` | `None` | Dotted function path for request-aware fan-in source selection. |
| `merge_fn` | `str` or `None` | `None` | Dotted import path to the fan-in merge function. Required when `wait_for` is set. |
| `stream_to` | `list[str]` | `[]` | Static superset of streaming targets for chunks such as hidden states or codec codes. |
| `stream_done_to_fn` | `str` or `None` | `None` | Dotted function path for request-aware stream-completion targets. |
| `project_payload` | `dict[str, str]` | `{}` | Optional target-stage to dotted projection function mapping used before writing a downstream payload. |
| `comm` | `CommConfig` or `None` | `None` | Per-stage communication pool and Mooncake options. |

Routing rule: set exactly one of `next` or `terminal=True`. `route_fn` is an
optional request-aware override for stages that already declare `next`.
Fan-in follows the same static-superset pattern: keep `wait_for` as the full
set of possible upstream stages, and use `wait_for_fn` only to select the
active per-request subset. When using `stream_done_to_fn`, keep `stream_to`
as the static superset because runtime prep derives stream receivers from it.

## `PipelineConfig` Reference

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `model_path` | `str` | required | Hugging Face model id or local checkpoint path. |
| `stages` | `list[StageConfig]` | required | Stage definitions. Config files override fields on them by name and cannot add or remove stages. |
| `name` | `str` or `None` | `model_path` | Pipeline name. Used for reporting and runtime identification. |
| `entry_stage` | `str` or `None` | first stage | Declared by the config class when the first stage is not the entry. Not settable from a config file or the CLI. |
| `fused_stages` | `list[list[str]]` | `[]` | Adjacent linear stage groups to colocate in one runtime process. |
| `env_defaults` | `dict[str, str]` | `{}` | Environment defaults applied before stage factory imports. Existing process values take precedence. |
| `mps` | `off`, `on`, or `auto` | `off` | Native CUDA MPS policy for eligible same-GPU worker processes. |
| `endpoints` | `EndpointsConfig` | IPC defaults | Endpoint allocation settings. |
| `placement` | `PlacementConfig` | defaults | Placement planning limits, e.g. `max_total_gpu_memory_fraction_per_gpu`. |
| `terminal_stages_fn` | `str` or `None` | `None` | Dotted function path for request-aware terminal-stage resolution. |
| `config_cls` | `str` | class name | Stored automatically and used when loading a saved config file. |

Class-level hooks a model config may declare:

- `stage_config_types: ClassVar[dict[str, type[StageConfig]]]` — the
  `StageConfig` subclass a named stage compiles paths against; mapping a
  stage to `EngineStageConfig` is what makes `engine.*` exist on it.
- `stage_factory_kwargs(stage_name)` — constructor kwargs the pipeline author
  passes to the factory in code (wiring, not configuration). A user-set
  `factory.*` value with the same name overrides the hook's value.
- `tensor_parallel_server_args_overrides(stage_name, tp_size)` /
  `topology_gated_custom_all_reduce_stages()` — engine overrides derived from
  the resolved TP topology at launch.

Derived values are computed from stages, not manually maintained:
`resolved_entry_stage`, `terminal_stages`, `gpu_placement`.

### Stage Fusion

`fused_stages` is a framework-level colocation hint. It keeps every listed
logical stage as a normal `Stage`; it does not create a synthetic scheduler.
At runtime prep, each fused group adds a process-colocation constraint, and
ordinary Stage routing can then use process-local dispatch for eligible hops.
A group must be adjacent, linear, non-TP, and fit on at most one GPU.

## How Values Reach a Factory

Two channels feed a stage factory, resolved in the parent process and applied
against the factory signature in the worker:

```text
PipelineConfig.stage_factory_kwargs(name)      # author channel: code wiring
stage.factory.*                                # config channel: by field name
stage.engine.*  ->  server_args_overrides      # config channel: one dict to SGLang
```

Per key, the config channel wins over the author channel;
`server_args_overrides` merges per key the same way. A configured key the
factory does not accept raises at construction. Standard kwargs
(`model_path`, `gpu_id`, `total_gpu_memory_fraction`) are injected only when
the factory signature declares them; `gpu_id` is owned by placement and is
rejected from the author channel.

## Runtime Prep and Runner

Runtime prep builds the resolved state used by the runner:

- validate stage names and static topology
- compute the entry stage and terminal stages
- allocate ZMQ endpoints
- carry dotted factory, merge, route, and projection paths into worker specs
- resolve both kwargs channels without importing stage factories
- build relay config from stage placement and relay backend
- wire stream targets and same-GPU stream fast paths

Serving uses `MultiProcessPipelineRunner` for both single-process and
multi-process topologies. Runtime prep first resolves GPU placement, then
process topology: every non-TP stage must declare `process` explicitly, and
explicit `stage.process` groups non-TP stages declaratively. A process group
may contain CPU stages and stages on at most one GPU. Multiple process groups
may share the same GPU only when per-stage `gpu_memory_fraction` budgets are
explicit and fit the configured placement limit.

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

```bash
sgl-omni serve --model-path ... --thinker.tp_size 4 --thinker.gpu "[0, 1, 2, 3]"
```

For `tp_size > 1`, the runner derives one process per TP rank. Each process runs
the stage scheduler and model worker with a different `tp_rank` and GPU. NCCL
collectives inside model forward keep TP ranks in lockstep. `StageConfig.process`
is optional for TP stages; if set, it acts as the prefix for the derived
per-rank process names (`{process}_tp{rank}`). TP ranks always own their OS
process exclusively.

Only rank 0 owns external stage IO:

- rank 0 receives ZMQ messages from the coordinator or previous stage
- rank 0 fans work and aborts out to follower ranks
- all ranks make the same scheduling decisions
- only rank 0 sends downstream results or terminal completions

Each TP stage gets its own NCCL port allocation so multiple TP groups can exist
inside one pipeline.
