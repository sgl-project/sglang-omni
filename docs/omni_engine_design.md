# Omni Engine Design

A unified, composable engine architecture for multi-modal model serving (Encoder, AR, Decoder, DiT).

## Design Principles

### Core Principle: Clear Separation > Leaky Abstractions

1. **Generic components know nothing about model specifics**
   - Scheduler doesn't know about tokens, positions, KV cache
   - ModelRunner doesn't know what `batch_data` contains

2. **Model-specific logic lives in dedicated components**
   - BatchPlanner: request selection + batch_data layout
   - ResourceManager: allocate/free model resources
   - IterationController: update request state + completion checks
   - InputPreparer: batch_data → model inputs
   - OutputProcessor: model outputs → RequestOutput

3. **Opaque data passing**
   - `Request.data`: model-specific, opaque to Scheduler
   - `SchedulerOutput.batch_data`: built by BatchPlanner, consumed by InputPreparer
   - `RequestOutput.data`: model-specific output

### Learned from vLLM v2
1. **Separation of Concerns**: Scheduler, Executor, ModelRunner as distinct components
2. **Contract-based Communication**: SchedulerOutput as the contract between scheduler and runner
3. **Async-first**: Non-blocking execution with futures

### Learned from MiniSGL
1. **Scheduler Owns State**: Single source of truth, runner is stateless
2. **Simple Batch Lifecycle**: Rebuild per step (optimize later if needed)

See `docs/omni_engine_design_notes.md` for short decisions and constraints.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              OmniEngine                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                      Scheduler (Generic)                            │   │
│  │                                                                      │   │
│  │  - requests: dict[str, Request]                                     │   │
│  │  - waiting / running queues                                         │   │
│  │  - delegates to BatchPlanner / ResourceManager / IterationController │   │
│  │                                                                      │   │
│  │  schedule() → SchedulerOutput                                       │   │
│  │  update(SchedulerOutput, ModelRunnerOutput)                         │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                           │                                 │
│                                           │ SchedulerOutput                 │
│                                           │   - requests: list[Request]     │
│                                           │   - batch_data: Any (opaque)    │
│                                           │   - step_id: int                │
│                                           ▼                                 │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                      ModelRunner (Generic)                          │   │
│  │                                                                      │   │
│  │  execute(SchedulerOutput) → ModelRunnerOutput                       │   │
│  │      ├── InputPreparer.prepare(batch_data)   ← model-specific       │   │
│  │      ├── model.forward()                                            │   │
│  │      └── OutputProcessor.process()           ← model-specific       │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

Model-Specific Components:
├── BatchPlanner (select_requests, build_batch)
├── ResourceManager (can_allocate, allocate, free)
├── IterationController (update_request, is_finished)
├── InputPreparer (batch_data → model inputs)
├── OutputProcessor (model outputs → RequestOutput)
└── RequestData (EncoderRequestData, ARRequestData, DiTRequestData)
```

---

## Component Responsibilities

| Component | Knows About | Doesn't Know About |
|-----------|-------------|-------------------|
| Scheduler | Request lifecycle, queues, abort | Tokens, tensors, resource details |
| BatchPlanner | Selection strategy, batch_data layout | Request lifecycle management |
| ResourceManager | Resource accounting (KV, memory) | Batch layout, model outputs |
| IterationController | Per-request state updates, finish checks | Batch selection, resource allocation |
| ModelRunner | How to call model | What batch_data contains |
| InputPreparer | How to convert batch_data → tensors | Scheduling decisions |
| OutputProcessor | How to extract per-request output | Request lifecycle |

---

## Request Lifecycle

```
                    add_request()
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                        WAITING                                   │
│  - Request received, queued for scheduling                      │
│  - BatchPlanner selects + ResourceManager allocates             │
└───────┬───────────────────────────────┬─────────────────────────┘
        │ schedule() + allocate         │ abort()
        ▼                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                        RUNNING                                   │
│  - Actively being processed by ModelRunner                      │
│  - For AR: multiple iterations until EOS                        │
│  - For Encoder: single iteration                                │
│  - For DiT: fixed N iterations                                  │
└───────────────┬─────────────────────────────────────────────────┘
                │ IterationController.is_finished() returns True
                ▼
┌─────────────────────────────────────────────────────────────────┐
│                        FINISHED                                  │
│  - ResourceManager.free() called (free resources)               │
│  - Future resolved with result                                  │
└─────────────────────────────────────────────────────────────────┘
                ▲
                │
┌─────────────────────────────────────────────────────────────────┐
│                        ABORTED                                   │
│  - Resources freed (if allocated)                               │
│  - Future resolved with abort                                   │
└─────────────────────────────────────────────────────────────────┘
```

---

## Engine ABC

```python
# ═══════════════════════════════════════════════════════════════════════════
# engines/base.py - Engine abstract base class
# ═══════════════════════════════════════════════════════════════════════════

from abc import ABC, abstractmethod
from typing import Any

class Engine(ABC):
    """Abstract base class for all engines."""

    @abstractmethod
    async def add_request(self, request_id: str, data: Any) -> None:
        """Add a request for processing."""
        ...

    @abstractmethod
    async def get_result(self, request_id: str) -> Any:
        """Get result for a request (blocks until ready)."""
        ...

    @abstractmethod
    async def abort(self, request_id: str) -> None:
        """Abort a request."""
        ...

    @abstractmethod
    async def start(self) -> None:
        """Start the engine processing loop."""
        ...

    @abstractmethod
    async def stop(self) -> None:
        """Stop the engine processing loop."""
        ...
```

---

## Core Types (Generic - Model Agnostic)

```python
# ═══════════════════════════════════════════════════════════════════════════
# types.py - Generic types only (no model-specific fields)
# ═══════════════════════════════════════════════════════════════════════════

from dataclasses import dataclass
from enum import Enum, auto
from typing import Any

class RequestStatus(Enum):
    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()
    ABORTED = auto()

@dataclass
class Request:
    """
    Generic request container.

    The Scheduler only cares about:
    - request_id: identity
    - status: lifecycle state

    Everything else is stored in `data` (opaque to Scheduler).
    This includes per-request params and simple output lists (streaming deferred).
    """
    request_id: str
    status: RequestStatus = RequestStatus.WAITING
    data: Any = None  # Model-specific, opaque to Scheduler

    # Timestamps (generic)
    arrival_time: float = 0.0
    finish_time: float | None = None

@dataclass
class SchedulerOutput:
    """
    Generic contract between Scheduler and ModelRunner.

    - requests: which requests to process
    - batch_data: opaque, built by BatchPlanner, consumed by InputPreparer
    """
    requests: list[Request]
    batch_data: Any  # Opaque - built by BatchPlanner, consumed by InputPreparer
    step_id: int = 0  # For abort safety / stale updates

    @property
    def num_requests(self) -> int:
        return len(self.requests)

    @property
    def request_ids(self) -> list[str]:
        return [r.request_id for r in self.requests]

@dataclass
class RequestOutput:
    """
    Generic output for a single request.

    The `data` field contains model-specific output
    (tokens, embeddings, latents, etc.)
    """
    request_id: str
    data: Any = None  # Model-specific output
    finished: bool = False
    finish_reason: str | None = None  # "stop", "length", "abort"

@dataclass
class ModelRunnerOutput:
    """Generic output from ModelRunner."""
    outputs: dict[str, RequestOutput]  # request_id → output
```

---

## Scheduler (Generic)

```python
# ═══════════════════════════════════════════════════════════════════════════
# scheduler.py - Generic scheduler (knows nothing about model specifics)
# ═══════════════════════════════════════════════════════════════════════════

from collections import deque
import asyncio
import time
import logging

logger = logging.getLogger(__name__)

class Scheduler:
    """
    Generic request scheduler.

    Responsibilities:
    - Manage request lifecycle (WAITING → RUNNING → FINISHED/ABORTED)
    - Delegate selection and resource usage to BatchPlanner/ResourceManager
    - Delegate iteration updates to IterationController
    - Produce SchedulerOutput for ModelRunner

    Does NOT know about:
    - Input formats (tokens, latents, etc.)
    - Model-specific batching logic
    - Resource details (KV cache, etc.)
    """

    def __init__(
        self,
        batch_planner: "BatchPlanner",
        resource_manager: "ResourceManager",
        iteration_controller: "IterationController",
        max_running: int = 256,
    ):
        self.batch_planner = batch_planner
        self.resource_manager = resource_manager
        self.iteration_controller = iteration_controller
        self.max_running = max_running

        # Request state
        self.requests: dict[str, Request] = {}
        self.waiting: deque[str] = deque()
        self.running: list[str] = []

        # Result futures (created lazily in get_result)
        self._futures: dict[str, asyncio.Future] = {}
        self._step_id = 0
        self._aborted_this_step: set[str] = set()

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────

    def add_request(self, request_id: str, data: Any) -> None:
        """Add a new request with model-specific data."""
        request = Request(
            request_id=request_id,
            data=data,
            arrival_time=time.time(),
        )
        self.requests[request_id] = request
        self.waiting.append(request_id)
        # Note: Future created lazily in get_result() to avoid event loop issues

    def abort_request(self, request_id: str) -> None:
        """Abort a request."""
        request = self.requests.get(request_id)
        if request is None:
            return
        self._aborted_this_step.add(request_id)
        self._finish_request(request, status=RequestStatus.ABORTED)

    def has_requests(self) -> bool:
        """Check if there are any requests to process."""
        return len(self.waiting) > 0 or len(self.running) > 0

    async def get_result(self, request_id: str) -> Request:
        """Wait for a request to complete."""
        if request_id not in self.requests:
            raise KeyError(f"Unknown request: {request_id}")

        # Create future lazily (requires running event loop)
        if request_id not in self._futures:
            self._futures[request_id] = asyncio.get_running_loop().create_future()

        # If already finished or aborted, resolve immediately
        request = self.requests[request_id]
        if request.status in (RequestStatus.FINISHED, RequestStatus.ABORTED):
            return request

        return await self._futures[request_id]

    # ─────────────────────────────────────────────────────────────────────────
    # Core Scheduling
    # ─────────────────────────────────────────────────────────────────────────

    def schedule(self) -> SchedulerOutput | None:
        """Schedule next batch. Returns None if no work."""
        if not self.waiting and not self.running:
            return None

        self._step_id += 1
        self._aborted_this_step.clear()

        waiting_reqs = [self.requests[req_id] for req_id in self.waiting]
        running_reqs = [self.requests[req_id] for req_id in self.running]

        # BatchPlanner handles selection AND resource allocation.
        selected = self.batch_planner.select_requests(
            waiting_reqs,
            running_reqs,
            self.resource_manager,
        )

        if not selected:
            return None

        # Move newly scheduled from waiting to running.
        for req in selected:
            if req.request_id in self.waiting:
                self.waiting.remove(req.request_id)
                self.running.append(req.request_id)
                req.status = RequestStatus.RUNNING

        batch_data = self.batch_planner.build_batch(selected)

        return SchedulerOutput(
            requests=selected,
            batch_data=batch_data,
            step_id=self._step_id,
        )

    def update(
        self,
        scheduler_output: SchedulerOutput,
        model_output: ModelRunnerOutput
    ) -> list[Request]:
        """
        Update state from model output.
        Returns list of finished requests.
        """
        finished = []

        for request in scheduler_output.requests:
            if request.request_id in self._aborted_this_step:
                continue

            output = model_output.outputs.get(request.request_id)
            if output is None:
                logger.warning("No output for request_id=%s", request.request_id)
                continue

            # Update via iteration controller (model-specific)
            self.iteration_controller.update_request(request, output)

            # Check completion via iteration controller
            if self.iteration_controller.is_finished(request, output):
                self._finish_request(request)
                finished.append(request)

        return finished

    def _finish_request(
        self,
        request: Request,
        status: RequestStatus = RequestStatus.FINISHED,
    ) -> None:
        """Clean up finished/aborted request."""
        was_running = request.status == RequestStatus.RUNNING
        request.status = status
        request.finish_time = time.time()

        if was_running:
            self.resource_manager.free(request)

        # Remove from queues
        if request.request_id in self.running:
            self.running.remove(request.request_id)
        if request.request_id in self.waiting:
            self.waiting.remove(request.request_id)

        # Resolve future if someone is waiting
        if request.request_id in self._futures:
            future = self._futures[request.request_id]
            if not future.done():
                future.set_result(request)
```

---

## Runtime Protocols (Model-Specific Interfaces)

```python
# ═══════════════════════════════════════════════════════════════════════════
# runtime/interfaces.py - Model-specific protocols
# ═══════════════════════════════════════════════════════════════════════════

from typing import Protocol, Any

class BatchPlanner(Protocol):
    """Selects requests and builds batch data."""

    def select_requests(
        self,
        waiting: list[Request],
        running: list[Request],
        resource_manager: "ResourceManager",
    ) -> list[Request]:
        """
        Select which requests to include in this batch.
        Also responsible for resource allocation via resource_manager.

        - AR: separate prefill vs decode
        - DiT: group by timestep
        - Encoder: simple FCFS
        """
        ...

    def build_batch(self, requests: list[Request]) -> Any:
        """Build model-specific batch data."""
        ...


class ResourceManager(Protocol):
    """Manages resources (memory, KV cache, etc.)."""

    def can_allocate(self, request: Request) -> bool: ...
    def allocate(self, request: Request) -> None: ...
    def free(self, request: Request) -> None: ...


class IterationController(Protocol):
    """Controls iteration (when is request done?)."""

    def update_request(self, request: Request, output: RequestOutput) -> None: ...
    def is_finished(self, request: Request, output: RequestOutput) -> bool: ...
```

---

## Reusable Implementations (Common)

```python
# ═══════════════════════════════════════════════════════════════════════════
# runtime/common.py - Reusable components
# ═══════════════════════════════════════════════════════════════════════════

class SimpleResourceManager:
    """Counting-based resource manager."""
    def __init__(self, max_count: int = 32):
        self.max_count = max_count
        self._count = 0

    def can_allocate(self, request: Request) -> bool:
        return self._count < self.max_count

    def allocate(self, request: Request) -> None:
        self._count += 1

    def free(self, request: Request) -> None:
        self._count = max(0, self._count - 1)


class SinglePassIterationController:
    """For Encoder - always done in one pass."""
    def update_request(self, request: Request, output: RequestOutput) -> None:
        request.data.embeddings = output.data

    def is_finished(self, request: Request, output: RequestOutput) -> bool:
        return True


class EosIterationController:
    """For AR - stop at EOS or max length."""
    def __init__(self, eos_token_id: int | list[int], max_length: int = 2048):
        self.eos_token_ids = (
            [eos_token_id] if isinstance(eos_token_id, int) else eos_token_id
        )
        self.max_length = max_length

    def update_request(self, request: Request, output: RequestOutput) -> None:
        token = output.data
        if isinstance(output.data, tuple):
            token, past_kv = output.data
            request.data.past_key_values = past_kv
        request.data.output_ids.append(token)
        if request.data.num_computed_tokens == 0:
            request.data.num_computed_tokens = len(request.data.input_ids)
        else:
            request.data.num_computed_tokens += 1

    def is_finished(self, request: Request, output: RequestOutput) -> bool:
        token = output.data
        if isinstance(output.data, tuple):
            token, _ = output.data
        if token in self.eos_token_ids:
            return True
        if request.data.max_new_tokens is not None:
            if len(request.data.output_ids) >= request.data.max_new_tokens:
                return True
        return request.data.num_computed_tokens >= self.max_length


class FixedStepsIterationController:
    """For DiT - fixed number of steps."""
    def __init__(self, num_steps: int):
        self.num_steps = num_steps

    def update_request(self, request: Request, output: RequestOutput) -> None:
        request.data.latents = output.data
        request.data.current_step += 1

    def is_finished(self, request: Request, output: RequestOutput) -> bool:
        return request.data.current_step >= self.num_steps
```

---

## ModelRunner (Generic)

```python
# ═══════════════════════════════════════════════════════════════════════════
# model_runner.py - Generic model runner
# ═══════════════════════════════════════════════════════════════════════════

import torch
from typing import Protocol, Any

class InputPreparer(Protocol):
    """Converts SchedulerOutput to model inputs (batch_data + request params)."""

    def prepare(
        self,
        scheduler_output: SchedulerOutput,
        device: torch.device,
    ) -> dict[str, Any]:
        """
        Convert opaque batch_data to model input dict.

        For Encoder: padded input_ids + attention_mask
        For AR: input_ids + positions + block_table + ...
        For DiT: latents + timesteps + ...
        """
        ...

class OutputProcessor(Protocol):
    """Converts model outputs to RequestOutputs."""

    def process(
        self,
        model_output: Any,
        scheduler_output: SchedulerOutput
    ) -> dict[str, RequestOutput]:
        """
        Convert model output to per-request outputs.

        For Encoder: extract embeddings per request
        For AR: sample tokens per request
        For DiT: extract denoised latents per request
        """
        ...

class ModelRunner:
    """
    Generic model executor.

    Responsibilities:
    - Convert SchedulerOutput to model inputs (via InputPreparer)
    - Execute model forward pass
    - Convert model outputs to RequestOutputs (via OutputProcessor)

    Completely stateless. All state lives in Scheduler.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        input_preparer: InputPreparer,
        output_processor: OutputProcessor,
        device: torch.device = torch.device("cuda"),
    ):
        self.model = model.to(device).eval()
        self.input_preparer = input_preparer
        self.output_processor = output_processor
        self.device = device

    def execute(self, scheduler_output: SchedulerOutput) -> ModelRunnerOutput:
        """Execute model on batch."""
        # 1. Prepare inputs (model-specific)
        model_inputs = self.input_preparer.prepare(
            scheduler_output,
            self.device,
        )

        # 2. Forward pass
        with torch.inference_mode():
            model_output = self.model(**model_inputs)

        # 3. Process outputs (model-specific)
        request_outputs = self.output_processor.process(
            model_output,
            scheduler_output
        )

        return ModelRunnerOutput(outputs=request_outputs)
```

---

## OmniEngine

```python
# ═══════════════════════════════════════════════════════════════════════════
# engine.py - OmniEngine combining scheduler and runner
# ═══════════════════════════════════════════════════════════════════════════

import asyncio
from typing import Any
from ..base import Engine

class OmniEngine(Engine):
    """
    Unified engine for all model types.

    Execution model:
    - Busy loop: schedule() → execute() → update()
    - Async-friendly: add_request() and get_result() are async
    """

    def __init__(
        self,
        scheduler: Scheduler,
        model_runner: ModelRunner,
    ):
        self.scheduler = scheduler
        self.model_runner = model_runner

        self._running = False
        self._loop_task: asyncio.Task | None = None

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────

    async def add_request(self, request_id: str, data: Any) -> None:
        """Add a request for processing."""
        self.scheduler.add_request(request_id, data)

    async def get_result(self, request_id: str) -> Any:
        """Get result for a request (blocks until ready)."""
        request = await self.scheduler.get_result(request_id)
        return request.data

    async def abort(self, request_id: str) -> None:
        """Abort a request."""
        self.scheduler.abort_request(request_id)

    # ─────────────────────────────────────────────────────────────────────────
    # Lifecycle
    # ─────────────────────────────────────────────────────────────────────────

    async def start(self) -> None:
        """Start the engine processing loop."""
        self._running = True
        self._loop_task = asyncio.create_task(self._run_loop())

    async def stop(self) -> None:
        """Stop the engine processing loop."""
        self._running = False
        if self._loop_task:
            await self._loop_task
            self._loop_task = None

    # ─────────────────────────────────────────────────────────────────────────
    # Processing Loop
    # ─────────────────────────────────────────────────────────────────────────

    async def _run_loop(self) -> None:
        """Main processing loop."""
        while self._running:
            await self._step()
            await asyncio.sleep(0)  # Yield to other coroutines

    async def _step(self) -> bool:
        """Execute one step. Returns True if work was done."""
        # 1. Schedule
        scheduler_output = self.scheduler.schedule()

        if scheduler_output is None:
            await asyncio.sleep(0.001)  # Brief sleep when idle
            return False

        # 2. Execute (run in executor to not block event loop)
        loop = asyncio.get_event_loop()
        model_output = await loop.run_in_executor(
            None,
            self.model_runner.execute,
            scheduler_output
        )

        # 3. Update state
        self.scheduler.update(scheduler_output, model_output)

        return True
```

---

## Model-Specific Implementations

### Encoder

```python
# ═══════════════════════════════════════════════════════════════════════════
# runtime/encoder.py - Encoder-specific implementation
# ═══════════════════════════════════════════════════════════════════════════

from dataclasses import dataclass
import torch

# ─────────────────────────────────────────────────────────────────────────────
# Encoder-specific data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class EncoderRequestData:
    """Encoder-specific request data (stored in Request.data)."""
    input_ids: torch.Tensor
    embeddings: torch.Tensor | None = None  # Filled after execution

@dataclass
class EncoderBatchData:
    """Encoder-specific batch data (SchedulerOutput.batch_data)."""
    input_ids_list: list[torch.Tensor]
    seq_lens: list[int]

# ─────────────────────────────────────────────────────────────────────────────
# BatchPlanner
# ─────────────────────────────────────────────────────────────────────────────

class EncoderBatchPlanner:
    """
    BatchPlanner for encoder models.

    Characteristics:
    - Single forward pass (no iteration)
    - No KV cache
    - Simple FCFS selection
    """

    def __init__(self, max_batch_size: int = 32):
        self.max_batch_size = max_batch_size

    def select_requests(
        self,
        waiting: list[Request],
        running: list[Request],
        resource_manager: ResourceManager,
    ) -> list[Request]:
        selected = []
        for req in waiting:
            if len(selected) >= self.max_batch_size:
                break
            if resource_manager.can_allocate(req):
                resource_manager.allocate(req)
                selected.append(req)
        return selected

    def build_batch(self, requests: list[Request]) -> EncoderBatchData:
        return EncoderBatchData(
            input_ids_list=[r.data.input_ids for r in requests],
            seq_lens=[len(r.data.input_ids) for r in requests],
        )

# IterationController: SinglePassIterationController (runtime/common.py)

# ─────────────────────────────────────────────────────────────────────────────
# InputPreparer
# ─────────────────────────────────────────────────────────────────────────────

class EncoderInputPreparer:
    """Converts EncoderBatchData to model inputs."""

    def __init__(self, pad_token_id: int = 0):
        self.pad_token_id = pad_token_id

    def prepare(self, scheduler_output: SchedulerOutput, device: torch.device) -> dict:
        batch_data: EncoderBatchData = scheduler_output.batch_data
        max_len = max(batch_data.seq_lens)
        batch_size = len(batch_data.input_ids_list)

        input_ids = torch.full(
            (batch_size, max_len),
            self.pad_token_id,
            dtype=torch.long,
            device=device
        )
        attention_mask = torch.zeros(
            (batch_size, max_len),
            dtype=torch.float,
            device=device
        )

        for i, ids in enumerate(batch_data.input_ids_list):
            seq_len = len(ids)
            input_ids[i, :seq_len] = ids.to(device)
            attention_mask[i, :seq_len] = 1.0

        return {"input_ids": input_ids, "attention_mask": attention_mask}

# ─────────────────────────────────────────────────────────────────────────────
# OutputProcessor
# ─────────────────────────────────────────────────────────────────────────────

class EncoderOutputProcessor:
    """Extracts embeddings from encoder output."""

    def __init__(self, pooling: str = "last"):
        self.pooling = pooling  # "last", "mean", "cls"

    def process(
        self,
        model_output: Any,
        scheduler_output: SchedulerOutput
    ) -> dict[str, RequestOutput]:
        hidden_states = model_output.last_hidden_state  # [batch, seq, hidden]
        batch_data: EncoderBatchData = scheduler_output.batch_data

        outputs = {}
        for i, request in enumerate(scheduler_output.requests):
            seq_len = batch_data.seq_lens[i]

            if self.pooling == "last":
                emb = hidden_states[i, seq_len - 1]
            elif self.pooling == "mean":
                emb = hidden_states[i, :seq_len].mean(dim=0)
            else:  # cls
                emb = hidden_states[i, 0]

            outputs[request.request_id] = RequestOutput(
                request_id=request.request_id,
                data=emb,
                finished=True,
            )

        return outputs
```

### AR (Autoregressive)

```python
# ═══════════════════════════════════════════════════════════════════════════
# runtime/ar.py - AR-specific implementation
# ═══════════════════════════════════════════════════════════════════════════

from dataclasses import dataclass, field
import torch

# ─────────────────────────────────────────────────────────────────────────────
# AR-specific data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ARRequestData:
    """AR-specific request data (stored in Request.data)."""
    input_ids: torch.Tensor
    output_ids: list[int] = field(default_factory=list)
    num_computed_tokens: int = 0
    max_new_tokens: int | None = None
    temperature: float = 0.0

    # For paged attention (optional, can start with simple KV cache)
    block_ids: list[int] = field(default_factory=list)

    # For simple HF-style KV cache
    past_key_values: tuple | None = None

@dataclass
class ARBatchData:
    """AR-specific batch data (SchedulerOutput.batch_data)."""
    input_ids: torch.Tensor          # [num_tokens]
    positions: torch.Tensor          # [num_tokens]
    seq_lens: list[int]              # Total length per sequence
    query_lens: list[int]            # New tokens this step

    # For paged attention
    block_table: list[list[int]] | None = None
    context_lens: list[int] | None = None

    # For simple HF-style KV cache
    past_key_values_list: list[tuple] | None = None

# ─────────────────────────────────────────────────────────────────────────────
# BatchPlanner (HF KV cache)
# ─────────────────────────────────────────────────────────────────────────────

class ARBatchPlanner:
    """
    AR BatchPlanner using HF-style KV cache.
    """

    def __init__(self, max_batch_size: int = 32, max_tokens: int = 8192):
        self.max_batch_size = max_batch_size
        self.max_tokens = max_tokens

    def select_requests(
        self,
        waiting: list[Request],
        running: list[Request],
        resource_manager: ResourceManager,
    ) -> list[Request]:
        selected = []
        total_tokens = 0

        # 1. Prioritize decode (running requests) - 1 token each
        for req in running:
            if len(selected) >= self.max_batch_size:
                break
            selected.append(req)
            total_tokens += 1

        # 2. Add prefill (waiting) if no decode or if budget allows
        if not selected:
            for req in waiting:
                if len(selected) >= self.max_batch_size:
                    break
                tokens_needed = len(req.data.input_ids)
                if total_tokens + tokens_needed > self.max_tokens:
                    break
                if resource_manager.can_allocate(req):
                    resource_manager.allocate(req)
                    selected.append(req)
                    total_tokens += tokens_needed

        return selected

    def build_batch(self, requests: list[Request]) -> ARBatchData:
        all_input_ids = []
        all_positions = []
        seq_lens = []
        query_lens = []
        past_key_values_list = []

        for request in requests:
            data: ARRequestData = request.data
            is_prefill = data.num_computed_tokens == 0

            if is_prefill:
                # Prefill: all input tokens
                ids = data.input_ids
                num_new = len(ids)
                pos = torch.arange(num_new)
            else:
                # Decode: last generated token
                last_token = data.output_ids[-1]
                ids = torch.tensor([last_token])
                num_new = 1
                pos = torch.tensor([data.num_computed_tokens])

            all_input_ids.append(ids)
            all_positions.append(pos)
            seq_lens.append(data.num_computed_tokens + num_new)
            query_lens.append(num_new)
            past_key_values_list.append(data.past_key_values)

        return ARBatchData(
            input_ids=torch.cat(all_input_ids),
            positions=torch.cat(all_positions),
            seq_lens=seq_lens,
            query_lens=query_lens,
            past_key_values_list=past_key_values_list,
        )

# ResourceManager: ARResourceManager (runtime/ar.py)

# IterationController: EosIterationController (runtime/common.py)

# ─────────────────────────────────────────────────────────────────────────────
# InputPreparer (single request for now)
# ─────────────────────────────────────────────────────────────────────────────

class ARInputPreparer:
    """AR input preparer for HF models (single request)."""

    def prepare(self, scheduler_output: SchedulerOutput, device: torch.device) -> dict:
        # For simplicity, assume single request
        # TODO: Handle batching with attention masks

        batch_data: ARBatchData = scheduler_output.batch_data
        input_ids = batch_data.input_ids.unsqueeze(0).to(device)

        past_kv = None
        if batch_data.past_key_values_list and batch_data.past_key_values_list[0] is not None:
            past_kv = batch_data.past_key_values_list[0]

        return {
            "input_ids": input_ids,
            "past_key_values": past_kv,
            "use_cache": True,
        }

# ─────────────────────────────────────────────────────────────────────────────
# OutputProcessor
# ─────────────────────────────────────────────────────────────────────────────

class AROutputProcessor:
    """AR output processor with per-request sampling."""

    def process(
        self,
        model_output: Any,
        scheduler_output: SchedulerOutput
    ) -> dict[str, RequestOutput]:
        logits = model_output.logits  # [batch, seq, vocab]
        past_key_values = model_output.past_key_values

        # Single request for now
        request = scheduler_output.requests[0]
        temperature = request.data.temperature

        if temperature <= 0.0:
            next_token = logits[:, -1, :].argmax(dim=-1).item()
        else:
            probs = torch.softmax(logits[:, -1, :] / temperature, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()

        return {
            request.request_id: RequestOutput(
                request_id=request.request_id,
                data=(next_token, past_key_values),
                finished=False,  # IterationController decides this
            )
        }
```

### DiT (Diffusion Transformer)

```python
# ═══════════════════════════════════════════════════════════════════════════
# runtime/dit.py - DiT-specific implementation
# ═══════════════════════════════════════════════════════════════════════════

from dataclasses import dataclass
import torch

# ─────────────────────────────────────────────────────────────────────────────
# DiT-specific data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DiTRequestData:
    """DiT-specific request data (stored in Request.data)."""
    latents: torch.Tensor            # Current latents
    condition: torch.Tensor | None   # Text/image condition
    current_step: int = 0

@dataclass
class DiTBatchData:
    """DiT-specific batch data (SchedulerOutput.batch_data)."""
    latents: torch.Tensor            # [batch, C, H, W]
    timesteps: torch.Tensor          # [batch]
    conditions: torch.Tensor | None  # [batch, seq, hidden]

# ─────────────────────────────────────────────────────────────────────────────
# BatchPlanner
# ─────────────────────────────────────────────────────────────────────────────

class DiTBatchPlanner:
    """
    BatchPlanner for Diffusion Transformer models.

    Characteristics:
    - Fixed number of denoising steps
    - No KV cache
    - Can group by step number
    """

    def __init__(self, max_batch_size: int = 16):
        self.max_batch_size = max_batch_size

    def select_requests(
        self,
        waiting: list[Request],
        running: list[Request],
        resource_manager: ResourceManager,
    ) -> list[Request]:
        selected = []
        for req in running:
            if len(selected) >= self.max_batch_size:
                break
            selected.append(req)
        if not selected:
            for req in waiting:
                if len(selected) >= self.max_batch_size:
                    break
                if resource_manager.can_allocate(req):
                    resource_manager.allocate(req)
                    selected.append(req)
        return selected

    def build_batch(self, requests: list[Request]) -> DiTBatchData:
        latents = torch.stack([r.data.latents for r in requests])
        timesteps = torch.tensor([r.data.current_step for r in requests])

        conditions = None
        if requests[0].data.condition is not None:
            conditions = torch.stack([r.data.condition for r in requests])

        return DiTBatchData(
            latents=latents,
            timesteps=timesteps,
            conditions=conditions,
        )

# IterationController: FixedStepsIterationController (runtime/common.py)

# ─────────────────────────────────────────────────────────────────────────────
# InputPreparer
# ─────────────────────────────────────────────────────────────────────────────

class DiTInputPreparer:
    """Converts DiTBatchData to model inputs."""

    def __init__(self, scheduler: Any):  # Diffusion scheduler
        self.scheduler = scheduler

    def prepare(self, scheduler_output: SchedulerOutput, device: torch.device) -> dict:
        batch_data: DiTBatchData = scheduler_output.batch_data
        # Convert step index to actual timestep value
        timesteps = self.scheduler.timesteps[batch_data.timesteps]

        return {
            "hidden_states": batch_data.latents.to(device),
            "timestep": timesteps.to(device),
            "encoder_hidden_states": batch_data.conditions.to(device) if batch_data.conditions is not None else None,
        }

# ─────────────────────────────────────────────────────────────────────────────
# OutputProcessor
# ─────────────────────────────────────────────────────────────────────────────

class DiTOutputProcessor:
    """Processes DiT output (denoising step)."""

    def __init__(self, scheduler: Any):  # Diffusion scheduler
        self.scheduler = scheduler

    def process(
        self,
        model_output: Any,
        scheduler_output: SchedulerOutput
    ) -> dict[str, RequestOutput]:
        # model_output is noise prediction
        noise_pred = model_output.sample
        batch_data: DiTBatchData = scheduler_output.batch_data

        outputs = {}
        for i, request in enumerate(scheduler_output.requests):
            # Apply scheduler step to get denoised latents
            step_output = self.scheduler.step(
                noise_pred[i],
                batch_data.timesteps[i],
                batch_data.latents[i],
            )

            outputs[request.request_id] = RequestOutput(
                request_id=request.request_id,
                data=step_output.prev_sample,
                finished=False,
            )

        return outputs
```

---

## Factory Functions

```python
# ═══════════════════════════════════════════════════════════════════════════
# factory.py - Factory functions for creating engines
# ═══════════════════════════════════════════════════════════════════════════

def create_encoder_engine(
    model: torch.nn.Module,
    tokenizer: Any,
    max_batch_size: int = 32,
    device: str = "cuda",
) -> OmniEngine:
    """
    Create an encoder engine.

    Example:
        engine = create_encoder_engine(bert_model, tokenizer)
        await engine.start()

        # Create request data
        input_ids = tokenizer.encode("Hello world", return_tensors="pt")
        data = EncoderRequestData(input_ids=input_ids[0])

        await engine.add_request("req-1", data)
        result = await engine.get_result("req-1")
    """
    scheduler = Scheduler(
        batch_planner=EncoderBatchPlanner(max_batch_size),
        resource_manager=SimpleResourceManager(max_batch_size),
        iteration_controller=SinglePassIterationController(),
        max_running=max_batch_size,
    )

    model_runner = ModelRunner(
        model=model,
        input_preparer=EncoderInputPreparer(
            pad_token_id=tokenizer.pad_token_id or 0
        ),
        output_processor=EncoderOutputProcessor(pooling="last"),
        device=torch.device(device),
    )

    return OmniEngine(scheduler=scheduler, model_runner=model_runner)

def create_ar_engine(
    model: torch.nn.Module,
    tokenizer: Any,
    max_seq_len: int = 2048,
    device: str = "cuda",
) -> OmniEngine:
    """
    Create an AR engine (single request, HF KV cache).

    Example:
        engine = create_ar_engine(llama_model, tokenizer)
        await engine.start()

        input_ids = tokenizer.encode("Once upon a time", return_tensors="pt")
        data = ARRequestData(
            input_ids=input_ids[0],
            max_new_tokens=256,
            temperature=0.7,
        )

        await engine.add_request("req-1", data)
        result = await engine.get_result("req-1")  # ARRequestData with output_ids
    """
    scheduler = Scheduler(
        batch_planner=ARBatchPlanner(),
        resource_manager=ARResourceManager(max_count=1),  # Or PagedKVCacheManager
        iteration_controller=EosIterationController(
            tokenizer.eos_token_id or 2,
            max_seq_len,
        ),
    )

    model_runner = ModelRunner(
        model=model,
        input_preparer=ARInputPreparer(),
        output_processor=AROutputProcessor(),
        device=torch.device(device),
    )

    return OmniEngine(scheduler=scheduler, model_runner=model_runner)
```

---

## File Structure

```
sglang_omni/
├── engines/
│   ├── base.py                           # Engine ABC
│   │
│   └── omni/
│       ├── __init__.py                   # Public exports
│       │
│       ├── types.py                      # Generic types only
│       │   - Request
│       │   - RequestStatus
│       │   - SchedulerOutput
│       │   - RequestOutput
│       │   - ModelRunnerOutput
│       │
│       ├── scheduler.py                  # Generic Scheduler
│       │
│       ├── model_runner.py               # Generic ModelRunner
│       │   - InputPreparer protocol
│       │   - OutputProcessor protocol
│       │
│       ├── engine.py                     # OmniEngine
│       │
│       ├── runtime/                      # Model-type-specific support
│       │   ├── __init__.py
│       │   ├── interfaces.py             # BatchPlanner/ResourceManager/IterationController protocols
│       │   ├── common.py                 # Reusable implementations
│       │   ├── encoder.py                # Encoder support (see below)
│       │   ├── ar.py                     # AR support
│       │   └── dit.py                    # DiT support
│       │
│       └── factory.py                    # create_*_engine functions
```

### What Each Runtime File Contains

```python
# runtime/encoder.py - Everything needed to support Encoder models

# 1. Data structures (what Request.data and batch_data contain)
@dataclass
class EncoderRequestData: ...

@dataclass
class EncoderBatchData: ...

# 2. Selection logic (when/how to batch)
class EncoderBatchPlanner: ...

# 3. Input/Output transformation (batch_data ↔ tensors)
class EncoderInputPreparer: ...
class EncoderOutputProcessor: ...
```

Note: The actual nn.Module (BERT, LLaMA, DiT) comes from **outside** — passed into the factory:

```python
# User provides the actual model
from transformers import BertModel
bert = BertModel.from_pretrained("bert-base")

# Factory wires it up with Encoder-specific support
engine = create_encoder_engine(
    model=bert,           # ← Actual nn.Module from user
    tokenizer=tokenizer,
)
```

So `runtime/encoder.py` doesn't contain BERT — it contains the logic to **schedule and batch requests** for any encoder model.

---

## Execution Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           OmniEngine._run_loop()                            │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 1. SCHEDULE                                                                  │
│                                                                              │
│    scheduler.schedule()                                                      │
│    ├── BatchPlanner.select_requests()                                       │
│    │   └── ResourceManager.allocate()                                       │
│    └── BatchPlanner.build_batch() → model-specific batch_data               │
│                                                                              │
│    Output: SchedulerOutput                                                  │
│    ├── requests: [Request, ...]                                             │
│    └── batch_data: Any (opaque, model-specific)                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 2. EXECUTE                                                                   │
│                                                                              │
│    model_runner.execute(scheduler_output)                                   │
│    ├── input_preparer.prepare(batch_data) → model inputs                    │
│    ├── model(**inputs) → model outputs                                      │
│    └── output_processor.process() → RequestOutputs                          │
│                                                                              │
│    Output: ModelRunnerOutput                                                │
│    └── outputs: {req_id: RequestOutput(data=...)}                           │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 3. UPDATE                                                                    │
│                                                                              │
│    scheduler.update(scheduler_output, model_output)                         │
│    ├── For each request:                                                    │
│    │   ├── IterationController.update_request() → update Request.data       │
│    │   └── if IterationController.is_finished(): _finish_request()          │
│    │       └── ResourceManager.free() → free resources                      │
│    └── Resolve futures for finished requests                                │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
                              [Loop continues]
```

---

## Implementation Order

1. **Phase 1**: `types.py` - Generic types
2. **Phase 2**: `scheduler.py` - Generic Scheduler
3. **Phase 3**: `model_runner.py` - Generic ModelRunner
4. **Phase 4**: `engine.py` - OmniEngine
5. **Phase 5**: `runtime/encoder.py` + preparers/processors - Test with BERT
6. **Phase 6**: `runtime/ar.py` (HF KV cache) - Test with LLaMA
7. **Phase 7**: Upgrade AR to batched/paged attention
8. **Phase 8**: `runtime/dit.py` - Test with DiT

---

## Notes on Preprocess

Preprocess (image resize, mel spectrum, tokenization, etc.) is intentionally **not** part of this design. It can be:

1. **Done by caller before `add_request()`** - Simplest approach
2. **A separate PreprocessRunner** - For async CPU preprocessing
3. **A separate stage in a multi-stage pipeline** - PreprocessStage → EncoderStage → LLMStage

This keeps the core engine focused on GPU execution.
