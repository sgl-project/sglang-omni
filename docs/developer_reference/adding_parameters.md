# Adding a Parameter

A practical guide for wiring a new setting into SGLang-Omni's configuration
surface: where it belongs, how it is declared and validated, and how a value
travels from YAML or the command line into the code that reads it. For the
reference description of the config layer itself, see [Config](config.md).

## First question: who consumes the value?

Stage settings are grouped by consumer, and the group decides everything else
— the spelling, the validation site, and the code that receives the value.
There are exactly three consumers:

| Consumer | Where it lives | Examples |
|---|---|---|
| The parent process (placement, process planning) | Stage top level | `gpu`, `tp_size`, `process`, `gpu_memory_fraction` |
| The stage factory (constructor kwargs) | `factory.*` | `max_concurrency`, `dtype`, `prefill_coalesce_requests` |
| The SGLang engine (ServerArgs) | `engine.*` | `mem_fraction_static`, `max_running_requests`, `disable_radix_cache` |

Every group answers to the same two spellings — one path language, written
twice:

```yaml
# YAML: the stages: mapping, keyed by stage name
stages:
  tts_engine:
    tp_size: 2
    factory:
      max_concurrency: 8
    engine:
      mem_fraction_static: 0.7
```

```bash
# CLI: dotted flags, the stages. prefix implied
sgl-omni serve --config omni.yaml \
    --tts_engine.tp_size 2 \
    --tts_engine.factory.max_concurrency 8 \
    --tts_engine.engine.mem_fraction_static 0.7
```

Both spellings become per-leaf patches and resolve through the same machinery,
so precedence (CLI beats file beats model default), duplicate detection, and
`sgl-omni config explain` provenance all come for free. You never write any of
that yourself.

## Case 1: a factory kwarg used by one model

Most parameters are this case, and it requires **no schema change at all**.
The `factory` group accepts undeclared keys and passes them through untouched;
the factory's signature is the check.

1. Add the parameter to your stage factory's signature, with its default:

```python
# sglang_omni/models/dots_tts/stages.py
def create_vocoder_executor(
    model_path: str,
    *,
    stream_slots: int = 16,
    ...
) -> DotsTTSStreamingVocoder:
```

2. There is no step 2. Users can immediately set it:

```bash
sgl-omni serve ... --vocoder.factory.stream_slots 8
```

The runtime overlays `factory.*` values onto the author's kwargs by name and
**refuses a key the factory does not accept** — a typo like
`--vocoder.factory.stream_slotz 8` fails at launch with the stage and
parameter named. This only works because factories declare every parameter
explicitly: never add a `**kwargs` catch-all to a stage factory, it turns
both typos and silently-ignored settings into no-ops.

### Validating a model-specific parameter

A free-form key is not locked out of the mainstream validation pipeline.
When the parameter has rules worth declaring — a range, an enum, eager
validation — type it, exactly as a shared field would be: subclass the group,
declare the field with its static constraints, and type the stage with it.

```python
class VocoderFactoryArgs(FactoryArgs):
    stream_slots: int | None = Field(default=None, ge=1, le=64)

class VocoderStageConfig(StageConfig):
    factory: VocoderFactoryArgs = Field(default_factory=VocoderFactoryArgs)

class MyPipelineConfig(PipelineConfig):
    stage_config_types: ClassVar[dict[str, type[StageConfig]]] = {
        "vocoder": VocoderStageConfig,
    }
```

`stages.vocoder.factory.stream_slots` is now a typed path with identical
treatment to a field declared on `FactoryArgs` itself, scoped to one stage:
the static range is enforced at resolution, the lossless conversion rule
applies to CLI text and YAML scalars, and `config explain` enumerates it.
This is the same pattern `EngineStageConfig` uses, and the same principle as
the shared groups: constraints are declarations on fields, not hand-written
checks. (A subclass carrying real fields is not an empty shell.)

The other two sites cover what a field declaration cannot express — both run
at resolve time too, because the resolver rebuilds and re-validates the
pipeline class on every merge:

- **Cross-field and cross-stage rules** go in the pipeline class's
  `model_post_init` — Ming-TTS validates its audio-decode cadence and batch
  contract there, and Ming-Omni its GPU-collision rule.
- **Rules that need the consumer's runtime state** stay in the consumer —
  the dots vocoder refuses a `stream_slots` that disagrees with the latent
  engine's admission limit, a relationship only known at launch.

A parameter with no rules beyond "the factory accepts it" needs none of
this; the signature default and the built-in unknown-kwarg refusal are
enough.

## Case 2: a factory kwarg shared across models

When a knob is common enough that several pipelines tune it — batching,
concurrency, coalescing — declare it on `FactoryArgs` in
`sglang_omni/config/schema.py` so it validates eagerly and shows up in path
enumeration:

```python
class FactoryArgs(BaseModel):
    ...
    max_concurrency: int | None = Field(default=None, ge=1)
    prefill_coalesce_wait_ms: float | None = Field(default=None, gt=0)
```

Rules for a declared field:

- **Default is `None`, meaning "unset".** The factory's own signature default
  stays in charge; only a set value is passed through. Never duplicate the
  factory's default into the schema — that would be a second source of truth.
- **Declare the range statically** with `Field(ge=/gt=/lt=/le=)`, or
  `Literal[...]` for a string enum, or `Field(min_length=1)` for a non-empty
  string. Pydantic enforces the constraint at construction and resolution,
  and path compilation carries the same constraint into CLI/YAML conversion —
  one declaration, every channel. Do not write the range as a hand-rolled
  `model_post_init` check; that is reserved for rules a declaration cannot
  express (see below).
- **Type shape is already handled.** Conversion is lossless-only: an int fits
  a float field and `0`/`1` fit a bool field, but a bool is refused on a
  numeric field and a float is refused on an int field, for CLI text and
  native YAML scalars alike. You get this for free.

`model_post_init` on the group is only for what a declaration cannot say:
cross-field rules, or advisory warnings (`prefill_coalesce_requests=1` warns
that the value is a no-op).

## Case 3: an engine (ServerArgs) setting

`engine.*` maps onto SGLang's ServerArgs for the stage's engine, and only
exists on engine stages — a stage declares itself one by using
`EngineStageConfig` (via `stage_config_types` in the pipeline class). Writing
`engine.*` on any other stage is a path error, so users cannot set values
that nothing would read.

- Any ServerArgs key already works as a free-form passthrough:
  `--tts_engine.engine.disable_radix_cache true`. Whether the key exists is
  SGLang's call, made when the engine starts.
- Declare a key on `EngineArgs` only when it is commonly tuned and deserves
  eager validation, e.g. `mem_fraction_static: Field(gt=0, lt=1)`.

Consumers read the set keys with `stage.engine.overrides()` — a dict of
exactly what was written, so SGLang's own defaults govern everything else.

## Case 4: a stage-placement setting

Fields read by the parent process before any stage code runs — placement,
process grouping, TP shape — live at the stage top level in `StageConfig`.
Add one only when the launcher or planner genuinely consumes it. Same
validation split as everywhere else: static `Field` constraints for ranges
(`tp_size: Field(default=1, ge=1)`), `model_post_init` for cross-field shape
(a TP stage's `gpu` list must match `tp_size` and be unique).

Do **not** re-check these rules downstream. The planner and topology walker
trust a validated config; a constraint has exactly one site.

## Case 5: a pipeline-level setting

Whole-pipeline values (`model_path`, `name`, `placement.*`) are top-level
`PipelineConfig` fields, written without a stage prefix (`--model_path ...`,
YAML top level). Model-specific pipeline classes may add their own fields the
same way (see `MossTTSLocalPipelineConfig`'s cache and cuda-graph fields).
Cross-stage invariants for one model belong in that pipeline class's
`model_post_init` — e.g. Ming-Omni refusing a talker GPU that collides with
the thinker's TP range.

## Values users do not set

Two mechanisms exist for values that are derived rather than configured.
Reach for them last.

**Author-derived factory kwargs** — `stage_factory_kwargs(stage_name)` on the
pipeline class returns launch-time constructor kwargs for one stage. Use it
when the pipeline author knows better than a static default (e.g. qwen3-tts
pinning deterministic-inference settings). Two hard rules: the config channel
wins per key (an explicit `factory.*` value overrides the hook's), and the
hook must not read *other* stages' config — cross-stage sharing is the user's
declaration to make, in YAML:

```yaml
# shared: writes one value into several stages, by selector
shared:
  - select: {engine: true}          # every SGLang engine stage
    engine:
      mem_fraction_static: 0.6
  - select: {stages: [talker, vocoder]}
    factory:
      dtype: bfloat16
```

An explicit per-stage write always outranks a `shared:` expansion.

**Post-merge derivation** — a value that only exists after every source has
been merged (e.g. `disable_custom_all_reduce` derived from the resolved TP
size and GPU topology in `serve`). A derivation is a fallback: it must only
fill keys no source set, never override an explicit value, and
`config resolve`/`explain` must run the same derivation so the preview equals
the launch. If you find yourself wanting a third such mechanism, stop and ask
whether the value can simply be a config default.

## Where validation goes: the one-site rule

Every rule has exactly one home, chosen by what the rule needs to see:

| Rule needs | Site | Example |
|---|---|---|
| Only the value | Static `Field` constraint / `Literal` — on the shared group, or a per-stage group subclass for one model | `mem_fraction_static: Field(gt=0, lt=1)`; `VocoderFactoryArgs.stream_slots` |
| The right conversion | Nothing — lossless coercion is built in | bool refused on int fields |
| Sibling fields on one object | `model_post_init` on that model | TP `gpu` list matches `tp_size` |
| Several stages of one pipeline | The pipeline class's `model_post_init` | Ming GPU-collision check; Ming-TTS audio-decode contract |
| The consumer's runtime state | The consumer, at the point of use | vocoder `stream_slots` vs. latent engine |
| The factory's parameter list | Nothing — the signature check is built in | unknown `factory.*` key refused |

Anti-patterns, each removed from this codebase at least once — do not
reintroduce them:

- **A second validation site.** If serve pre-checks a range the schema also
  checks, one of them is wrong the day the rule changes.
- **A capability whitelist.** "Factories that support X" lists go stale; the
  factory signature already says what it accepts.
- **Reserved fields and speculative enums.** Every field and every enum
  member must have a consumer today.
- **Empty subclass shells.** A `StageConfig` subclass must carry a real
  difference (like `EngineStageConfig.engine_stage`); a shell that only
  renames is noise.
- **Compat for unreleased spellings.** Migration hints cover spellings that
  shipped on `main`; an abstraction that only ever existed on your branch is
  deleted without a trace.
- **Topology in user config.** Which stages exist, how they route, where
  requests enter — that is the model's `config.py`, never YAML or CLI.

## Checklist for a new parameter

1. Name the consumer; that names the group and the spelling.
2. Free-form first: a model-specific factory kwarg needs only a signature
   parameter. Declare on `FactoryArgs`/`EngineArgs` only for cross-model
   knobs.
3. Ranges and enums are static declarations (`Field(...)`, `Literal`), not
   code.
4. Cross-field rules go in the owning model's `model_post_init`; nothing is
   checked twice.
5. Verify the whole surface end to end:

```bash
sgl-omni config resolve --model-path <model> --<stage>.factory.<name> <value>
sgl-omni config explain stages.<stage>.factory.<name> --model-path <model>
```

`resolve` must show the value where the consumer will read it, and `explain`
must attribute it to your source. Both commands build the identical patch set
a launch builds, so if the preview is right, the launch is right.

6. Tests: pin the acceptance path (value reaches the consumer) and the
   refusal path (out-of-range and, for free-form keys, the unknown-kwarg
   refusal), in the suite of the layer that owns the rule.
