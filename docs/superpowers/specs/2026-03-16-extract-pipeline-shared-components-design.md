# Extract Pipeline Shared Components

**Date:** 2026-03-16
**Status:** Draft
**Approach:** Protocol-Based Composition (Option C)

## Problem

`models/qwen3_omni/` contains ~6800 lines of model-specific code across 21 Python files. Analysis shows ~1200 lines are generic pipeline plumbing (streaming bridge loops, executor factories, tensor utils, event decoding) that got baked into the model directory during initial development. Adapting a new model currently requires ~6000+ lines; much of that is copy-paste of this plumbing.

## Goal

Extract reusable pipeline building blocks into the framework layer so that adapting a new omni model (potentially with a completely different architecture) requires ~2800 lines instead of ~6300.

**Non-goals:**
- Abstracting model internals (attention, MoE, RoPE, weight loading)
- Abstracting HF-specific preprocessing (processor, tokenizer)
- Changing the SGLang AR backend interface
- Refactoring `engine_io.py` (M-RoPE, multimodal token handling — too model-specific)

## Constraints

- Next model may have a completely different architecture (not Thinker+Talker four-stage)
- SGLang remains the AR inference backend
- Must not break existing Qwen3-Omni pipeline
- Abstractions must be composable, not inheritance-heavy — a model that doesn't use streaming AR should not need to touch `streaming.py`

## Design

### New File 1: `executors/streaming.py` (~250 lines)

Protocol-based building blocks for streaming AR executors. Extracted from `models/qwen3_omni/components/talker_executor.py` (947 lines).

#### Protocols

```python
class ChunkClassifier(Protocol):
    """Classify inbound StreamItems as upstream data or feedback.

    Only called for StreamItem instances. StreamSignal handling (done/error)
    is built into run_streaming_bridge() itself — the bridge checks is_done
    and error fields before calling classify().
    """
    def classify(self, item: StreamItem) -> Literal["upstream", "feedback"]: ...

class TrailingBuffer(Protocol):
    """Accumulate upstream hidden states during streaming decode.

    Only responsible for buffer-side readiness. Engine-side readiness
    (e.g. scheduler status == WAITING_FEEDBACK) is checked by the bridge
    loop itself before calling ready_for_feedback().
    """
    def append(self, request_id: str, chunk: StreamItem) -> None: ...
    def mark_done(self, request_id: str) -> None: ...
    def ready_for_feedback(self, request_id: str, step_index: int) -> bool: ...

class FeedbackSender(Protocol):
    """Send feedback chunks to the engine's feedback mailbox.

    from_stage is fixed at construction time (e.g. CODE_PREDICTOR_STAGE
    for Qwen3). The bridge loop calls send() with the chunk data and ID.
    """
    def send(self, request_id: str, chunk_id: int, data: torch.Tensor) -> None: ...
```

#### Generic Bridge Loop

```python
async def run_streaming_bridge(
    request_id: str,
    stream_queue: StreamQueue,
    classifier: ChunkClassifier,
    trailing_buffer: TrailingBuffer,
    feedback_sender: FeedbackSender,
    state: StreamingRequestState,
    engine_ready_fn: Callable[[str], bool],
    *,
    poll_interval: float = 0.01,
    is_aborted: Callable[[], bool] = lambda: False,
) -> None:
    """Generic bridge loop: poll stream_queue -> classify -> route.

    Signal handling (built-in):
    - StreamSignal with error: raises
    - StreamSignal with is_done from upstream: calls trailing_buffer.mark_done()
    - StreamSignal from other sources: ignored

    Chunk routing (via classifier):
    - "upstream" items -> trailing_buffer.append()
    - "feedback" items -> pending feedbacks queue

    Feedback flushing:
    - Checks engine_ready_fn(request_id) first (engine-side readiness,
      e.g. scheduler status == WAITING_FEEDBACK)
    - Then checks trailing_buffer.ready_for_feedback() (buffer-side readiness)
    - If both pass, pops from pending queue and calls feedback_sender.send()
    """
```

This is a standalone async function, not a method on a base class. It takes protocols as parameters, making it usable by any streaming AR executor without inheritance.

The `engine_ready_fn` callback separates engine-side readiness (scheduler status) from buffer-side readiness (`TrailingBuffer`), keeping the protocol implementable without engine internals.

#### Request Lifecycle Manager

```python
class StreamingRequestManager:
    """Manages per-request state, bridge tasks, and result tasks.

    Provides generic implementations of add_request/get_result/abort.
    The model-specific part (building the initial engine request) is
    passed as a callback.

    The stream_queue is set externally via set_stream_queue() — the pipeline
    compiler wires this up during stage initialization (same pattern as the
    current TalkerStreamingExecutor._stream_queue).

    Feedback mailbox wiring (creating the internal StreamQueue, setting it on
    the engine, enabling feedback on the iteration controller) is handled in
    set_feedback_mailbox(). The executor delegates this call to the manager.
    """
    def __init__(self, engine, ...): ...

    def set_stream_queue(self, queue: StreamQueue) -> None:
        """Called by pipeline compiler to wire inbound stream queue."""
        ...

    def set_feedback_mailbox(self, mailbox: StreamQueue) -> None:
        """Wire inbound stream queue + enable feedback on engine.

        Creates internal feedback mailbox, sets it on engine, and
        enables feedback on the scheduler's iteration controller.
        """
        ...

    async def add_request(
        self,
        payload: StagePayload,
        *,
        build_fn: Callable[..., EngineInput],
        classifier: ChunkClassifier,
        buffer: TrailingBuffer,
        sender: FeedbackSender,
        min_initial_chunks: int = 1,
    ) -> None: ...

    async def get_result(self) -> StagePayload: ...
    async def abort(self, request_id: str) -> None: ...
```

#### Qwen3 Usage After Refactoring

```python
class TalkerStreamingExecutor(Executor):
    def __init__(self, engine, ...):
        self._manager = StreamingRequestManager(engine, ...)
        self._classifier = QwenChunkClassifier(thinker_stage=THINKER_STAGE)
        self._buffer = QwenTrailingBuffer(engine, im_end_token_id, project_fn=...)
        self._sender = QwenFeedbackSender(feedback_mailbox, from_stage=CODE_PREDICTOR_STAGE)

    async def add_request(self, payload):
        await self._manager.add_request(
            payload,
            build_fn=self._build_initial_request,
            classifier=self._classifier,
            buffer=self._buffer,
            sender=self._sender,
        )

    async def get_result(self): return await self._manager.get_result()
    async def abort(self, rid): await self._manager.abort(rid)

    # Lifecycle: delegate to manager
    async def start(self): await self._manager.start()
    async def stop(self): await self._manager.stop()
    def set_stream_fn(self, fn): self._manager.set_stream_fn(fn)
    def set_feedback_mailbox(self, mb): self._manager.set_feedback_mailbox(mb)
    async def stream(self, rid): ...  # delegate to engine.stream()

    # --- Model-specific hooks (all that remains) ---
    def _build_initial_request(self, payload, chunks, thinker_done): ...
    def _project_assistant_chunk(self, chunk): ...
    # ... token loading, embedding projection, speaker resolution, etc.
```

**Line count:** `talker_executor.py` 958 -> ~450 lines.

### New File 2: `executors/factories.py` (~150 lines)

Generic executor factory helpers. Extracted from `models/qwen3_omni/pipeline/stages.py` (637 lines).

#### Encoder Executor Factory

```python
def create_encoder_executor(
    *,
    stage_name: str,
    model: torch.nn.Module,
    device: str,
    state_loader: Callable[[StagePayload], Any],
    request_builder: Callable[[Any, str], Any],
    result_applier: Callable[[Any, str, Any], None],
    state_storer: Callable[[StagePayload, Any], StagePayload],
) -> EngineExecutor:
    """Wrap any single-pass model (encoder, embedder) as an EngineExecutor.

    Extracted from stages.py _create_encoder_executor (lines 73-93).
    """
```

#### SGLang Config Builder

```python
def build_sglang_executor_args(
    model_path: str,
    *,
    context_length: int = 8192,
    model_arch_override: str | None = None,
    weight_prefix: str | None = None,
    overrides: dict[str, Any] | None = None,
) -> "ServerArgs":
    """Build SGLang ServerArgs from plain config dict.

    Wraps build_sglang_server_args() with a cleaner interface.
    Extracted from stages.py create_sglang_thinker_executor_from_config (lines 332-353).
    """
```

#### Hidden State Stream Adapter

```python
def make_hidden_state_stream_adapter(
    *,
    stream_fn: Callable | None,
    hidden_key: str = "hidden_states",
    split_fn: Callable | None = None,
) -> Callable:
    """Create a stream adapter that extracts hidden states from AR output
    and sends them to stream targets.

    Extracted from stages.py make_thinker_stream_adapter (lines 356-423)
    and make_talker_ar_stream_adapter (lines 426-447).

    split_fn defaults to split_dual_layer_hidden() for dual-layer capture.
    Pass a custom split_fn for single-layer or other layouts.
    """

def split_dual_layer_hidden(
    hidden: dict | torch.Tensor | None,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Split {embed: T, layer_N: T} dict into (embed, layer_hidden).

    Handles fallback keys: "embed", 0, "0".
    Extracted from stages.py lines 367-388.
    """
```

**Qwen3 `stages.py` after refactoring:** 637 -> ~350 lines. Model-specific code that remains:
- `create_preprocessing_executor()` — instantiates `Qwen3OmniPreprocessor`
- `create_sglang_thinker_executor()` — Qwen3 token extraction, `_stream_builder` callback
- `create_talker_ar_executor()` — Qwen3 config extraction (token IDs, codec vocab, speaker map)
- `create_decode_executor()` — uses model-specific `decode_events` output types

### New File 3: `proto/events.py` (~25 lines)

Move `OmniEvent` and `OmniEventType` from `models/qwen3_omni/io.py` — these types are model-agnostic.

```python
"""Streaming event types for omni pipelines."""

OmniEventType = Literal[
    "text_delta", "text_final",
    "audio_chunk", "audio_final",
    "image",
    "video_chunk", "video_final",
    "debug", "final",
]

@dataclass
class OmniEvent:
    """Streaming-friendly event emitted by decode logic."""
    type: OmniEventType
    modality: str
    payload: dict[str, Any]
    is_final: bool = False
```

`models/qwen3_omni/io.py` changes to re-export:
```python
from sglang_omni.proto.events import OmniEvent, OmniEventType  # noqa: F401
```

### New File 4: `preprocessing/utils.py` (~30 lines)

Tensor coercion helpers extracted from `models/qwen3_omni/pipeline/merge.py` (lines 15-38).

```python
"""Tensor coercion utilities for preprocessing pipelines."""

def as_tensor(value: Any, dtype: torch.dtype | None = None) -> torch.Tensor | None:
    """Coerce value to tensor with optional dtype conversion. Returns None if value is None."""

def as_tensor_list(value: Any) -> list[torch.Tensor] | None:
    """Normalize a single tensor or list of tensors to list[Tensor]."""

def is_non_empty(tensor: torch.Tensor | None) -> bool:
    """Check that tensor exists and has at least one element."""
```

### Existing File Change: `pipeline/event_decode.py` (moved, signature relaxed)

`decode_events()` moves from `models/qwen3_omni/pipeline/merge.py` (lines 248-316) to `sglang_omni/pipeline/event_decode.py` (~70 lines).

The function currently takes `PipelineState` and `ThinkerOutput` (defined in `models/qwen3_omni/io.py`). To avoid a framework→model circular dependency, the signature is relaxed to accept plain dicts:

```python
def decode_events(
    *,
    thinker_out: dict[str, Any],     # was ThinkerOutput (TypedDict)
    stream_state: dict[str, Any],    # was state.stream_state via PipelineState
    tokenizer: Any,
    eos_token_id: int | None,
    step: int,
) -> Iterable[OmniEvent]:
```

The caller in `merge.py` changes from `decode_events(state=state, ...)` to `decode_events(stream_state=state.stream_state, ...)`. The function body only accesses `state.stream_state` anyway, so this is a clean break.

`merge.py` changes to import from the new location. Line count: 316 -> ~220.

Note: there is no `sglang_omni/runtime/` top-level package. `sglang_omni/pipeline/` is the correct home since `decode_events` is a pipeline utility used by stage executors.

## File Change Summary

### New files

| File | Lines | Source |
|------|-------|--------|
| `executors/streaming.py` | ~250 | `talker_executor.py` lifecycle + bridge + feedback |
| `executors/factories.py` | ~150 | `stages.py` encoder factory, SGLang config, stream adapters |
| `proto/events.py` | ~25 | `io.py` OmniEvent/OmniEventType |
| `preprocessing/utils.py` | ~30 | `merge.py` tensor helpers |
| `pipeline/event_decode.py` | ~70 | `merge.py` decode_events (signature relaxed to plain dicts) |
| **Total new framework** | **~525** | |

### Modified files

| File | Before | After | Delta |
|------|--------|-------|-------|
| `models/qwen3_omni/components/talker_executor.py` | 958 | ~450 | -508 |
| `models/qwen3_omni/pipeline/stages.py` | 641 | ~350 | -291 |
| `models/qwen3_omni/pipeline/merge.py` | 316 | ~220 | -96 |
| `models/qwen3_omni/io.py` | 121 | ~105 | -16 |
| **Total model delta** | | | **-911** |

### Untouched files (model-specific, not worth abstracting)

- `talker.py` (1008) — model architecture
- `thinker.py` (822) — model architecture + hidden capture
- `talker_input.py` (244) — 95% Qwen3-specific chat template logic
- `config.py` (220) — pipeline wiring declarations
- `engine_io.py` (421) — M-RoPE, multimodal tokens, codec request building
- `code2wav_executor.py` (258) — vocoder
- `code_predictor_executor.py` (219) — code predictor
- `image_encoder.py` (182), `audio_encoder.py` (89) — thin wrappers
- `preprocessor.py` (340) — HF processor integration
- `common.py` (35) — `load_thinker_config` helper
- `pipeline/state_io.py` (16) — `load_state`/`store_state`
- `pipeline/next_stage.py` (90) — stage name constants and routing
- `__init__.py` files (23 total)

### Net effect

- **Model directory:** ~6800 -> ~5900 lines (-13%)
- **Framework growth:** +455 lines
- **New model adaptation cost:** ~6000+ -> ~2800 lines (-55%)

## Reuse Matrix for New Models

| Component | Architecture A (similar streaming) | Architecture B/C (different) |
|-----------|------|------|
| `streaming.py` protocols + bridge | Full reuse — implement 3 protocols | Skip entirely |
| `StreamingRequestManager` | Full reuse | Skip entirely |
| `factories.create_encoder_executor` | Full reuse | Full reuse (any encoder) |
| `factories.build_sglang_executor_args` | Full reuse | Full reuse (SGLang backend) |
| `factories.make_hidden_state_stream_adapter` | Full reuse or custom split_fn | Skip if no hidden state streaming |
| `proto/events.py` OmniEvent | Full reuse | Full reuse |
| `preprocessing/utils.py` | Full reuse | Full reuse |
| `pipeline/event_decode.py` | Full reuse | Full reuse (any tokenizer-based decoding) |

## Implementation Order

1. **`proto/events.py`** + update `io.py` re-export — zero-risk, no logic change
2. **`preprocessing/utils.py`** + update `merge.py` imports — zero-risk, no logic change
3. **`pipeline/event_decode.py`** + update `merge.py` — move + relax signature (`PipelineState` -> `dict`)
4. **`executors/factories.py`** + update `stages.py` — replace factory calls; verify all call sites (including `config.py`)
5. **`executors/streaming.py`** + refactor `talker_executor.py` — highest risk, most value

Steps 1-2 are pure moves (no logic changes). Step 3 is a move + minor signature change. Step 4 is medium risk (factory signatures change, must verify all callers). Step 5 is the main refactoring effort.

## Testing Strategy

- Existing e2e speech test (`test_speech_e2e_real.py`) covers the full pipeline
- Existing HF alignment test (`test_hf_alignment.py`) covers talker logit accuracy
- No new unit tests required for pure moves (steps 1-3)
- Steps 4-5: run existing tests to verify behavioral equivalence
- Verify: pipeline startup, text-only inference, speech inference with streaming
