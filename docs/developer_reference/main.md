# Architecture

SGLang-Omni is the multi-stage runtime for omni models: models that accept
mixed text, image, audio, and video inputs and may emit text, audio, or other
modalities.

## System Overview

```text
HTTP API -> Client -> Coordinator -> Stage -> Scheduler -> ModelRunner -> model forward
```


| Layer                             | Duty                                                                                   |
| ----------------------------------- | ---------------------------------------------------------------------------------------- |
| [HTTP API](./apiserver_design.md) | OpenAI-compatible request and response schemas, SSE framing, HTTP errors               |
| [Client](./apiserver_design.md)   | `GenerateRequest` to `OmniRequest`, result aggregation, audio encoding                 |
| [Coordinator](./pipeline.md)      | Request lifecycle, entry-stage submission, terminal result collection, abort broadcast |
| [Stage](./pipeline.md)            | Control-plane IO, relay IO, fan-in, stream routing, scheduler inbox/outbox bridging    |
| [Scheduler](./pipeline.md)        | Per-stage execution loop and failure propagation to stage outbox                       |
| [ModelRunner](./pipeline.md)      | AR forward preparation, model forward dispatch, output extraction                      |
| [Communication](./communication.md) | Control-plane messages and relay data transfer between stages                         |

Refer to the layer-specific document for specific design details.

## Directory Layout

```text
sglang_omni/
|-- pipeline/       # Inter-stage orchestration, stages, coordinator, processes
|-- scheduling/     # Scheduler loops and inbox/outbox message types
|-- model_runner/   # Shared model runner abstractions for AR stages
|-- models/         # Model-specific configs, stages, request builders, modules
|-- config/         # PipelineConfig, StageConfig, config manager, compiler
|-- relay/          # Data transfer backends
|-- serve/          # HTTP server and OpenAI-compatible API adapter
|-- client/         # Internal client used by API adapters
`-- proto/          # Request, payload, stage, and control-plane message types
```

## Model Directory Convention

Model-specific code should stay under `sglang_omni/models/<model>/`.

Recommended layout:

```text
models/<model>/
|-- config.py             # PipelineConfig subclass and StageConfig list
|-- stages.py             # stage factories
|-- routing.py            # optional data-driven routing helpers
|-- request_builders.py   # inter-stage payload transforms
|-- payload_types.py      # typed model-specific payload state
|-- callbacks.py          # feedback callbacks or strategy, when needed
`-- components/           # model modules, processors, vocoders, adapters
```

Only model-local behavior belongs here. The framework-owned layers are still
`Stage`, `Coordinator`, schedulers, model-runner bases, relay, compiler, and
runners.
