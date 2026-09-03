---
orphan: true
---

# Model cookbook template

Copy this template when adding a model cookbook. Keep the required sections,
remove inapplicable optional subsections, and replace every placeholder before
publishing the page.

````markdown
# Model name

One sentence describing the model and its primary use.

## Overview

| Item | Value |
|---|---|
| Task | TTS / ASR / Omni / Music / Generation |
| Checkpoint(s) | `organization/model` |
| Endpoint(s) | `/v1/...` |
| Pipeline | preprocessing → engine → vocoder |
| Input / output | ... → ... |
| Streaming | Direction and transport / No |
| Validated hardware | Accelerator model, or Not recorded |

## Prerequisites

Follow the shared installation guide. List only model-specific packages,
system dependencies, checkpoint access, or setup required before deployment.

## Deploy

Provide one canonical checked-in configuration or command.

```bash
sgl-omni serve \
  --model-path organization/model \
  --port 8000
```

If the model requires a checked-in configuration, include its real
`examples/configs/` path in this command.

## Send a request

Provide one minimal working request. Include curl and Python only when both add
value.

```bash
curl ...
```

## Capabilities

Include only relevant model-specific subsections, for example voice cloning,
language hints, streaming, long audio, diarization, multimodal input, or voice
design.

## Configuration

Document behavior that differs from shared runtime defaults. Do not copy the
complete server configuration reference.

## Limitations

- List concrete unsupported or constrained behavior.

## Benchmark

Provide the canonical benchmark command and link to the shared benchmark
methodology.

Include one command that uses an existing benchmark entry point and the
arguments needed for this model.

## Related documentation

- API or serving guide
- Runtime feature guide
- Deployment guide
- Benchmark methodology
- Runnable examples
- Developer documentation
````
