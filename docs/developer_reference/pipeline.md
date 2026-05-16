## Pipeline Overview

### Coordinator

`Coordinator` is the global request router. It registers stage endpoints, sends
new requests to the entry stage, receives `CompleteMessage` and `StreamMessage`
events, and resolves client futures or streams.

Key responsibilities:

- route new requests to `entry_stage`
- track request state: pending, running, completed, failed, aborted
- collect terminal stage completions
- merge results when a pipeline has multiple terminal stages, such as `decode`
  and `code2wav`
- broadcast abort messages to all stages

The coordinator is stage-implementation agnostic. In a tensor
parallel stage group, it only talks to rank 0. Peer ranks stay internal to the
stage group.

### Stage

`Stage` is an IO shell. It handles all inter-stage communication. It receives control messages, reads
and writes relay payloads, performs fan-in when needed, and pushes all executable
work into `scheduler.inbox`.

```python
class Stage:
    def __init__(
        self,
        name,
        control_plane,
        relay,
        get_next,
        input_handler,
        scheduler,
        stream_targets,
        same_gpu_targets,
    ):
        self.scheduler = scheduler
```

Stage responsibilities:

- receive `SubmitMessage`, `DataReadyMessage`, `ShutdownMessage`, and profiler
  control messages over ZMQ
- receive `AbortMessage` over the coordinator broadcast channel
- read and write full `StagePayload` objects through relay
- aggregate inputs with `AggregatedInput` for fan-in stages
- route normal results to downstream stages or the coordinator
- route streaming chunks, including same-GPU CUDA IPC and cross-GPU relay
- drain `scheduler.outbox` and convert scheduler output into control-plane
  messages

The important invariant is that `Stage` does not branch on scheduler type.
`SimpleScheduler`, `OmniScheduler`, and streaming schedulers all present the
same surface.

### Scheduler

All schedulers implement the same interface:

```python
class Scheduler:
    inbox: Queue[IncomingMessage]
    outbox: Queue[OutgoingMessage]

    def start(self) -> None: ...
    def stop(self) -> None: ...
    def abort(self, request_id: str) -> None: ...
```

Scheduler messages are used to communicate with stage layer:

```python
class IncomingMessage:
    request_id: str
    type: Literal["new_request", "stream_chunk", "stream_done"]
    data: Any

class OutgoingMessage:
    request_id: str
    type: Literal["result", "stream", "error"]
    data: Any
    target: str | None
    metadata: dict[str, Any] | None
```

#### OmniScheduler

`OmniScheduler` is used by autoregressive stages. It composes with SGLang's
upstream scheduler. The goal is to reuse SGLang's
batch selection, KV cache management, prefill/decode scheduling, tree cache, and
overlap scheduling while keeping SGLang-Omni's transport, request objects, and
streaming behavior outside the upstream scheduler.

#### SimpleScheduler

`SimpleScheduler` is for non-AR stages such as preprocessing, encoders,
aggregation, and decode. It has no KV cache and no SGLang batching. The loop is:

```text
inbox.get() -> compute function -> outbox.put(result or error)
```

It supports a batch compute function for stages where local batching is
useful.

#### Code2WavScheduler

`Code2WavScheduler` is a streaming vocoder scheduler. It handles:

- `new_request`: initialize per-request state
- `stream_chunk`: accumulate and decode code chunks
- `stream_done`: flush remaining audio and emit a final result

### Model Runner

The model runner layer owns the AR forward path. The design target is:

```text
ForwardBatch -> before/custom forward hooks -> model forward -> post hook -> output processing
```

The shared base runner owns common mechanics: `ForwardBatch` construction,
sampling, logit processing, repetition penalty handling, output processing, and
conversion into scheduler output.

#### ThinkerModelRunner

`ThinkerModelRunner` is for Qwen-omni thinker-style AR models. Its model-specific job is
to prepare the forward batch by injecting multimodal embeddings such as image,
video, audio, and deepstack inputs before the model forward.

#### FeedbackARModelRunner

The refactor design identifies a shared `FeedbackARModelRunner` role for AR
models whose next decode step depends on feedback produced by the previous step
inside the same model runner. Qwen3-Omni talker and Fish Audio S2-Pro both fit
this shape; Qwen3 currently implements the pattern in its talker runner.

The abstraction covers self-contained feedback loops only:

- write previous-step feedback into model buffers before forward
- run the AR backbone and secondary head inside model `forward()`
- extract codebook outputs and feedback tensors after forward
- push stream or result output to the scheduler outbox

Cross-stage feedback, where the producer and consumer live in different
schedulers and communicate through relay, is out of scope for this runner.

The design groups model-specific feedback behavior into a small strategy:

```python
class FeedbackStrategy:
    def write_buffers(self, model, schedule_batch, requests) -> None: ...
    def extract_output(self, model, schedule_batch, requests, outbox) -> None: ...
    def prefill_forward(self, tp_worker, forward_batch, ...) -> object | None: ...
```
