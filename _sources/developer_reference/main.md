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
| [TTS Integration](./tts_model_integration.md) | Checklist and lifecycle rules for adding TTS model families                         |

Refer to the layer-specific document for specific design details.

## Directory Layout

```text
sglang_omni/
|-- pipeline/       # Inter-stage orchestration, stages, coordinator, processes
|-- scheduling/     # Scheduler loops and inbox/outbox message types
|-- model_runner/   # Shared model runner abstractions for AR stages
|-- models/         # Model-specific configs, stages, request builders, modules
|-- config/         # PipelineConfig, StageConfig, config manager, topology
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
`Stage`, `Coordinator`, schedulers, model-runner bases, relay, runtime prep, and
runners.

## Naming and lint checks

Use public names for classes, functions, methods, and attributes defined in
`sglang_omni/`. The leading-underscore lint hook checks every Python file under
that directory, including new model packages. It flags class and function
definitions, every attribute read or write, and string names passed to
`getattr`. It ignores vendor copies, dunder names such as `__init__`, a lone
`_`, and definitions inside functions.

A name inherited from an upstream object still fails the check. Keep that
spelling, and mark the line:

```python
session_request._omni_prompt_cache_key = adapter_request._omni_prompt_cache_key  # noqa: leading-underscore
```

When a local definition must keep a name because an external interface
requires it, put `# noqa: leading-underscore` on that line and explain why:

```python
class ExternalAdapter(ExternalBase):
    # ExternalBase calls this hook by its exact name.
    def _required_external_hook(self):  # noqa: leading-underscore
        return self.state
```

The exemption keeps the definition out of lint violations and automatic renames.
Its references retain the same name. A comment such as
`# skip leading-underscore class and function names` is not an exemption.

The pre-commit hook runs `scripts/check_leading_underscore.py --fix`, which renames
violating definitions and supported references within the same file. It does not
update callers in other files. Review the diff, preserve third-party API names,
and update any affected cross-file callers explicitly. To check without editing:

```bash
python scripts/check_leading_underscore.py
```

Run `pre-commit run --all-files` before submitting a change.
