# Documentation Contribution Guide

SGLang-Omni follows SGLang's task-oriented documentation structure while
treating multimodal pipelines, stage placement, streaming behavior, and
hardware qualification as first-class concepts.

This guide defines where information belongs and how to write model
documentation. Apply it to new pages and to existing pages when they are
materially updated. Do not reorganize unrelated legacy pages only to match the
physical layout below.

## Information architecture

The user-facing documentation has these sections:

| Section | Question it answers |
|---|---|
| Get started | How do I install SGLang-Omni and send one request? |
| Basic usage | How do I use a public API or common model workflow? |
| Advanced features | How does a reusable runtime capability work? |
| Deployment | How do I place and size a multi-stage pipeline? |
| Supported models | Which models are supported, and on which accelerators? |
| Cookbook | How do I deploy and use one model? |
| Benchmarks | How do we measure correctness and performance? |
| Developer guide | How does the system work, and how do I extend it? |
| References | What does a configuration field, CLI flag, or error mean? |

The directory structure may migrate incrementally. Navigation and content
ownership define the contract even when a legacy page still has an older path.

## Content boundaries

### Get started

Keep installation, the first successful request, supported platforms, and
release notes here. Do not turn this section into a complete feature guide.

### Basic usage

Document endpoint fields, response formats, streaming protocols, and generic
error semantics here. A cookbook should mention only model-specific endpoint
behavior, such as an unsupported route.

### Advanced features

Document reusable behavior such as deterministic inference, admission control,
batching, prefill CUDA Graph, MPS/DP, stage offload, colocation, and weight
sharing here. Cookbooks state whether a model supports a feature and explain
only model-specific behavior.

### Deployment

Document stage placement, memory tuning, colocation, and multi-GPU resource
planning here. Keep model-specific validated topologies in the cookbook and
link to the shared deployment guidance.

### Supported models

Maintain a compact model-family matrix and a separate accelerator matrix in
[Supported models](./supported_models.md). Keep model support independent from
validation evidence. Never apply evidence from one model/backend combination
to a broader family or accelerator. Store exact checkpoints, revisions,
configuration overrides, hardware, and evidence links in
[Model qualification evidence](./developer_reference/model_qualification.md).

### Cookbook

A cookbook is an operational recipe for one model. It contains the model's
prerequisites, one canonical deployment, pipeline, first request, capabilities,
deviations from shared defaults, known limitations, canonical benchmark
command, and links to shared documentation. Use the
[model cookbook template](./cookbook/template.md).

Do not use a cookbook as a complete API reference, generic feature guide,
benchmark methodology document, CI qualification database, performance
investigation report, or implementation design document. Keep CI workflow,
worker-count, and qualification evidence details out of the primary metadata.

### Benchmarks

Separate benchmark content into three layers:

1. A cookbook gives the canonical command for that model.
2. Benchmark documentation defines datasets, metrics, warmup, concurrency,
   streaming methodology, and reproducibility requirements.
3. A qualification report records the exact commit, model revision, hardware,
   dependencies, launch configuration, parameters, results, and analysis.

Do not keep large current-main result tables or tuning histories in permanent
cookbook prose.

### Developer guide

Keep architecture, pipeline lifecycle, stage interfaces, communication,
profiling, and model integration details here. Operational instructions belong
in a cookbook or deployment guide.

### References

Keep factual configuration, CLI, and error definitions concise. Tutorials and
model recommendations belong elsewhere.

## Sources of truth

Documentation explains how to use the system; it does not redefine facts owned
by code, configuration, CI, or benchmark artifacts.

| Information | Source of truth |
|---|---|
| Supported model registration | Model and pipeline registry |
| API request fields | Request schema and API implementation |
| CLI flags | CLI and configuration implementation |
| Runtime defaults | Runtime configuration |
| Model defaults | Model adapter and pipeline configuration |
| Recommended launch configuration | Checked-in example config |
| Benchmark commands | Benchmark implementation |
| CI qualification | Model CI definition |
| Performance numbers | Benchmark artifact or qualification report |
| Exact model qualification | Model CI definition, report, or benchmark artifact cataloged in the qualification page |
| Cookbook | Operational explanation and model-specific guidance |

Link to the stronger source when practical. If a cookbook recommends an
override, explain why that model needs it instead of copying the entire shared
configuration reference.

## Supported-model schema

The primary model-family table uses these fields:

| Field | Meaning |
|---|---|
| Model | Public model or checkpoint family |
| Task | TTS, ASR, Omni, Music, Generation, or another concrete task |
| Endpoint | Public serving endpoint |
| Streaming | Direction and transport, or No with a short qualification |
| Status | Experimental or Supported |
| Cookbook | The operational recipe |

The accelerator table keeps three claims separate:

| Field | Meaning |
|---|---|
| Backend implementation | Whether an accelerator integration exists in current main |
| Expected model scope | Which model/backend combinations are intended to work |
| Validation | Whether those combinations are CI tested, manually validated, experimental, not recorded, or unsupported |
| Documentation / evidence | Installation guidance, CI, or the implementation source supporting the claim |

Detailed qualification belongs in
[Model qualification evidence](./developer_reference/model_qualification.md),
with one row per exact checkpoint and configuration. Record the checkpoint and
revision when pinned, material launch overrides, hardware, validation type,
and CI workflow, report, or benchmark evidence.

Model status describes the maintenance expectation:

- **Experimental**: an implementation exists, but its documented support
  contract is not yet considered stable.
- **Supported**: the configuration is maintained and expected to work.

Accelerator validation describes model/backend runtime evidence:

- **CI tested**: recurring model-level CI runs on the named accelerator.
- **Manually validated**: current documentation records an end-to-end run, but
  no recurring model gate covers it.
- **Experimental**: backend and model-specific implementation exists without
  recurring CI or a durable manual validation record in current main.
- **Not recorded**: backend implementation exists, but no user-facing model
  support set is recorded.
- **Unsupported**: end-to-end model serving is not supported on that backend.

Model status and accelerator validation are independent. A platform abstraction
or checked-in profile proves implementation scope, not runtime validation.

## Hardware claims

State what was tested, not what might fit. Prefer "Validated on: H100" over
"Minimum hardware: 80 GB GPU." If no runtime evidence is recorded, write "Not
recorded" rather than inferring support from model size, a backend abstraction,
or a checked-in profile.

## Writing style

- Use active voice and second person for procedures.
- Use sentence-case headings.
- Explain what a feature is before explaining how to configure it.
- Put prerequisites before commands.
- Keep examples copy/pasteable and use realistic values.
- Prefer one canonical example over several nearly identical variants.
- Reuse checked-in examples and configuration files when practical.
- Verify flags, defaults, API fields, and runtime behavior against their source
  of truth.
- Link to shared documentation instead of duplicating it.
- Use comments in examples only when they explain a non-obvious constraint.

Avoid marketing language, filler introductions, speculative support claims,
duplicated API tables, large historical benchmark tables, and implementation
details that do not help a user operate the model.

## New-model checklist

A new model should include:

- [ ] A model-family entry with evidence-based maturity.
- [ ] An accelerator-matrix update when the model changes a backend's expected
      or validated scope.
- [ ] A qualification-evidence row for each exact CI or manual validation
      claim.
- [ ] A cookbook based on the standard template.
- [ ] A first-class pipeline topology.
- [ ] Validated hardware, or an explicit statement that it is not yet recorded.
- [ ] A checked-in server configuration or a copy/pasteable launch command.
- [ ] A runnable client or curl request.
- [ ] A streaming example when streaming is supported.
- [ ] Model-specific capabilities and known limitations.
- [ ] A canonical benchmark command.
- [ ] CI or manual validation linked only when current-main evidence exists.
- [ ] Links to shared API, runtime, deployment, and benchmark documentation.
