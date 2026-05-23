# TTS Model Integration

Notes on adding a new TTS model family for native `/v1/audio/speech` serving.
Read [main.md](./main.md) first for the broader stage / scheduler / coordinator
picture; this page only covers the TTS-specific pieces.

## Order of work

Roughly the steps to add a new model. Each one is fleshed out in a section
below.

1. Pick the HF architecture string (`config.json::architectures[0]`).
2. Scaffold `sglang_omni/models/<name>/` with `__init__.py` + `config.py`. Set
   `architecture` on a `PipelineConfig` subclass and export `EntryClass`. The
   registry finds the model from here.
3. If the upstream HF config is not in stock `transformers`, call
   `AutoConfig.register("<model_type>", <Config>)` at import time or in the
   AR factory.
4. Write the SGLang model class in `sglang_model.py` (or split it like Higgs
   does), then add one line to
   `sglang_omni/model_runner/sglang_model_runner.py::_register_omni_model` so
   SGLang can resolve the architecture.
5. Implement the three stage factories in `stages.py`. The AR factory builds
   server args via `build_sglang_server_args`, hands them to
   `create_sglang_infrastructure`, and returns an `OmniScheduler`.
6. Write `request_builders.py` and `payload_types.py`. Wire abort cleanup
   into every scheduler that touches shared state.
7. Add `examples/configs/<name>.yaml` and list the model in
   [docs/basic_usage/tts.md](../basic_usage/tts.md).
8. Add the GPU-free unit tests listed at the bottom.

## Layout

A new model lives under `sglang_omni/models/<name>/`. The files most TTS
models end up with:

```text
__init__.py           # package marker; subpackages here are auto-discovered
config.py             # PipelineConfig subclass, stage list, EntryClass
stages.py             # factory functions referenced by StageConfig.factory
request_builders.py   # adapt the incoming request into scheduler input
payload_types.py      # typed state passed between stages
sglang_model.py       # SGLang-side model class registered under the HF arch
model_runner.py       # custom AR runner, only when the default does not fit
```

Not every model needs every file (Higgs splits its model into `model.py` +
`modeling.py`; Voxtral keeps its pipeline submodules under `pipeline/`). Use
whatever shape fits, but keep model code out of the framework layers.

## Pipeline shape

The minimum useful pipeline is three stages. Qwen3-TTS, Voxtral-TTS and S2-Pro
all keep this shape; Qwen3-TTS and S2-Pro call the AR stage `tts_engine`, while
Voxtral uses the analogous `tts_generation` name.

1. **preprocessing** - validate the request, fetch and tokenize references,
   build prompt state. Keep heavy CPU/GPU work here so the AR loop is not held
   up by it.
2. **tts_engine** - autoregressive generation of audio or codec tokens. Use
   `OmniScheduler` whenever you want SGLang's KV cache, batching, abort
   handling, and request limits. Reach for a custom model runner only when the
   forward really is model-specific.
3. **vocoder** - codes-to-waveform. A `SimpleScheduler` with `batch_compute_fn`
   handles most batched vocoders. Use a streaming scheduler when audio needs
   to leave the server before generation finishes.

Two variants in the tree are worth knowing about:

- Insert an extra **audio_encoder** stage between preprocessing and `tts_engine`
  when you need to run a heavy encoder on the AR device once per request;
  Higgs TTS does this for its multi-codebook reference embed.
- For per-chunk streaming from engine to vocoder, set `stream_to=["vocoder"]`
  on the engine `StageConfig` and `can_accept_stream_before_payload=True` on
  the vocoder `StageConfig`. S2-Pro is the reference for this.

Wire all of the above declaratively in `config.py` (stage order, terminal
flags, GPU placement, fan-out). Then expose `EntryClass = YourPipelineConfig`
at module scope and set `architecture: ClassVar[str] = "<HFArch>"` on the
class. `sglang_omni/models/registry.py` walks every subpackage of
`sglang_omni/models/`, picks up each `EntryClass`, and matches the
`architecture` attribute against the model's HF config; no manual list to
edit anywhere.

Once the code side works, drop a runnable launch file under
`examples/configs/<name>.yaml` and add the model to
[docs/basic_usage/tts.md](../basic_usage/tts.md) so users have something to
point `sgl-omni serve --config` at.

### SGLang wiring

The `tts_engine` factory has to stand up an SGLang worker on its GPU. Two
shared helpers do the heavy lifting:

- `build_sglang_server_args(checkpoint_dir, ...)` from
  `sglang_omni.scheduling.sglang_backend`
- `create_sglang_infrastructure(server_args, gpu_id, *, model_arch_override=...)`
  from `sglang_omni.scheduling.bootstrap`, which returns the
  `(model_worker, tree_cache, req_to_token_pool, token_to_kv_pool_allocator,
  prefill_mgr, decode_mgr, model_config)` tuple that `OmniScheduler` expects

Two pieces of glue still have to be added by hand:

- Insert your SGLang model class into `ModelRegistry.models[...]` inside
  `sglang_omni/model_runner/sglang_model_runner.py::_register_omni_model`.
  Pass the same key as `model_arch_override` when the HF architecture string
  does not match the class name you registered.
- If upstream weights don't load cleanly into your SGLang module, add a
  `weight_loader.py` (see Higgs's `DiscreteWeightMapper` for the shape).
  Most models don't need one.

## Where the request comes in

`POST /v1/audio/speech` builds a `GenerateRequest` from the OpenAI payload in
`sglang_omni/serve/openai_api.py::build_speech_generate_request`. That request
enters the pipeline and your model's request builder turns it into whatever
the AR scheduler needs.

Two things are easy to get wrong at this boundary:

- **Endpoint defaults silently override model defaults.** The HTTP layer fills
  in a single set of sampling defaults (currently the S2-Pro values). For any
  other model, those values look exactly like the user explicitly asking for
  them. Pass an `explicit_generation_params` list (or equivalent) through the
  request and have your request builder distinguish "user set this" from "the
  endpoint filled it in". The same trick is needed for any field where the
  endpoint has an opinion.
- **Input is heterogeneous.** TTS clients send text under several names
  (`input`, `text`, sometimes a chat-style structure) and references under
  several shapes (`ref_audio` + `ref_text`, or a `references[]` list).
  Normalize what you accept inside the request builder, validate required
  references there, and keep that logic out of the AR stage so a bad request
  fails before anything touches the GPU.

The builder should hand the scheduler a typed dataclass (see the
`payload_types.py` files for examples), not a free-form dict; the AR stage
reads these fields on every step.

## Scheduler contracts

Every scheduler exposes the same four methods plus two queues, and Stage code
relies on it:

```python
inbox: Queue[IncomingMessage]
outbox: Queue[OutgoingMessage]
start() -> None
stop() -> None
abort(request_id: str) -> None
```

Pick by responsibility:

- `SimpleScheduler` - a single callable, optionally batched via
  `batch_compute_fn`. Right for preprocessing and most vocoders.
- `OmniScheduler` - wraps SGLang's prefill/decode managers, KV cache, and
  request-limit machinery. Use it for AR generation.
- A custom scheduler - when neither of the above fits (stateful streaming
  vocoder, custom detokenizer, etc.).

The part that bites people is abort cleanup. Anything you store outside the
scheduler keyed by `request_id`, such as a prepared-tensor stash from
preprocessing to AR, a session handle, or a reference cache, has to be freed on
three paths:

- the preprocessing stage aborts before handing off
- the AR stage aborts before its request builder consumes the handoff
- preprocessing finishes *after* the request was already aborted, so the
  result gets dropped on the floor

Wire the same cleanup function as `abort_callback` on every scheduler that
touches the shared state, and make it idempotent; it will be called more
than once for the same id, by design.

## Tensors and devices

Keep tensors on the device they were produced on until something actually
needs CPU bytes. The recurring mistake is calling `.cpu()` inside the request
builder "just to be safe"; it costs a sync per request and the AR runner has
to move the tensor right back.

Practical line:

- Preprocessing should create prompt and reference tensors on the AR
  device/dtype, or normalize them exactly once before the decode handoff.
- Scheduler request data should carry the normalized tensors as device
  tensors. Do not store CPU copies in AR request data unless the target
  consumer is explicitly CPU-bound.
- Detach tensors when no gradients are needed, and cast only at clear
  ownership boundaries such as preprocessing output, feedback-buffer writes,
  or final vocoder/HTTP serialization.
- CPU materialization for metadata, such as a stable cache-key hash, should
  produce metadata only; it should not replace the device tensor used by
  prefill/decode.

One more trap: if your prefix includes continuous embeddings spliced into the
token stream (Higgs and Qwen3-TTS both do this), the radix cache key has to
be derived from the embedding content. Two different prompts that happen to
share the same placeholder token IDs will otherwise alias to the same KV
prefix, and you will see one user's audio leak into another's.

## What to test before review

GPU-free unit tests covering each rule already stated above:

- request boundary - sampling-default preservation and required-input
  validation (see "Where the request comes in")
- scheduler request data - device/dtype invariants and the abort-cleanup
  race paths listed under "Scheduler contracts"
- stage-local behavior - whichever of vocoder batching or streaming you
  picked

For end-to-end quality, run the shared TTS benchmark:

```bash
python -m benchmarks.eval.benchmark_tts_seedtts --help
```

Report WER/CER, sample count, throughput, and `rtf_mean`. If the model trails
on a particular language or split, write a sentence on the likely cause
(sampling config, codec/vocoder version, text normalization, eval setup)
instead of just leaving the number to speak for itself.
