# SPDX-License-Identifier: Apache-2.0
"""Validate CUDA graph batch coverage for SGLang-backed generation stages."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _BufferProbe:
    """Per-model ``(label, fn)`` extractors reading the allocated buffer first dim."""

    extractors: tuple[tuple[str, Callable[[Any], int]], ...]
    note: str = ""


_BUFFER_PROBES: dict[str, _BufferProbe] = {
    "HiggsTTSModel": _BufferProbe(
        (
            ("_sampler_pool.seeds", lambda m: m._sampler_pool.seeds.shape[0]),
            ("_cg_codes_BN", lambda m: m._cg_codes_BN.shape[0]),
            ("_cg_active_last_codes", lambda m: m._cg_active_last_codes.shape[0]),
        ),
        note="sampler pool = max_running_requests + 1 (one reserved padding row)",
    ),
    "Qwen3TTSTalker": _BufferProbe(
        (("_feedback_buffer", lambda m: m._feedback_buffer.shape[0]),)
    ),
    "MossTTSDelaySGLangModel": _BufferProbe(
        (
            (
                "_decode_input_embedding.weight",
                lambda m: m._decode_input_embedding.weight.shape[0],
            ),
        )
    ),
    "MossTTSLocalSGLangModel": _BufferProbe(
        (
            (
                "_decode_input_embedding.weight",
                lambda m: m._decode_input_embedding.weight.shape[0],
            ),
        )
    ),
    "S2ProSGLangTextModel": _BufferProbe(
        (("_vq_codes", lambda m: m._vq_codes.shape[0]),),
        note="allocated only after setup_vq_decode()",
    ),
    "VoxtralSGLangTTSModel": _BufferProbe(
        (
            (
                "_decode_input_embed_buffer",
                lambda m: m._decode_input_embed_buffer.shape[0],
            ),
        )
    ),
    "Qwen3OmniTalker": _BufferProbe(
        (("_feedback_buffer", lambda m: m._feedback_buffer.shape[0]),)
    ),
}


@dataclass(frozen=True)
class CudaGraphBatchReport:
    """Outcome of validating one stage's CUDA graph batch sizing."""

    stage: str
    max_running_requests: int | None
    cuda_graph_max_bs: int | None
    captured_bs: list[int] | None
    request_slots: int | None
    buffer_capacity: int | None
    buffer_source: str | None
    ok: bool
    findings: list[str] = field(default_factory=list)

    @property
    def max_captured_bs(self) -> int | None:
        return max(self.captured_bs) if self.captured_bs else None

    def format(self) -> str:
        """Render a human-readable multi-line report."""
        lines = [f"Stage: {self.stage}"]
        lines.append(
            "  serving config:     "
            f"max_running_requests={self.max_running_requests}, "
            f"cuda_graph_max_bs={self.cuda_graph_max_bs}"
        )
        lines.append(
            f"  captured graph bs:  {self.captured_bs} "
            f"(max captured = {self.max_captured_bs})"
        )
        lines.append(f"  request slots:      {self.request_slots}")
        lines.append(
            f"  model-side buffers: {self.buffer_capacity} " f"[{self.buffer_source}]"
        )
        lines.append(f"  VERDICT: {'OK' if self.ok else 'MISMATCH'}")
        for finding in self.findings:
            lines.append(f"    - {finding}")
        return "\n".join(lines)


def evaluate_cuda_graph_batch_sizing(
    *,
    stage: str,
    max_running_requests: int | None,
    cuda_graph_max_bs: int | None,
    captured_bs: list[int] | None,
    request_slots: int | None,
    buffer_capacity: int | None,
    buffer_source: str | None = None,
) -> CudaGraphBatchReport:
    """Compare the three batch-size facts and produce a verdict."""
    findings: list[str] = []
    ok = True

    captured = [b for b in (captured_bs or []) if b is not None]
    max_captured = max(captured) if captured else None

    if buffer_capacity is not None and max_captured is not None:
        if max_captured > buffer_capacity:
            ok = False
            findings.append(
                f"graphs captured up to bs={max_captured} but model-side "
                f"buffer holds {buffer_capacity}; capture/replay above "
                f"{buffer_capacity} overruns the model buffer."
            )
    elif buffer_capacity is None:
        findings.append(
            f"model-side buffer not read ({buffer_source}); validated serving "
            f"config vs. captured sizes only."
        )

    if (
        buffer_capacity is not None
        and max_running_requests is not None
        and buffer_capacity < max_running_requests
    ):
        ok = False
        findings.append(
            f"model-side buffer ({buffer_capacity}) is smaller than "
            f"max_running_requests ({max_running_requests}); peak concurrency "
            f"cannot be served."
        )

    if ok and not findings:
        findings.append(
            "captured sizes and model-side buffer track the serving config."
        )

    return CudaGraphBatchReport(
        stage=stage,
        max_running_requests=max_running_requests,
        cuda_graph_max_bs=cuda_graph_max_bs,
        captured_bs=captured or None,
        request_slots=request_slots,
        buffer_capacity=buffer_capacity,
        buffer_source=buffer_source,
        ok=ok,
        findings=findings,
    )


def read_captured_bs(model_runner: object) -> list[int] | None:
    """Read the actually-captured CUDA graph batch sizes, or None if absent."""
    try:
        captured = model_runner.graph_runner.capture_bs
    except AttributeError:
        logger.debug(
            "cuda_graph_batch_validator: model_runner.graph_runner.capture_bs "
            "missing (CUDA graphs disabled)."
        )
        return None
    try:
        sizes = sorted(int(b) for b in captured)
    except (TypeError, ValueError):
        logger.warning(
            "cuda_graph_batch_validator: capture_bs is not an iterable of "
            "ints (%r).",
            captured,
        )
        return None
    return sizes or None


def read_model_buffer_capacity(model: object) -> tuple[int | None, str]:
    """Return ``(min_capacity, source)`` over the model's registered buffers.

    A model may register several per-request buffers; the safe capacity is the
    smallest one, so all that resolve are read and the minimum is returned.
    """
    if model is None:
        return None, "no model object on runner"

    cls = type(model).__name__
    probe = _BUFFER_PROBES.get(cls)
    if probe is None:
        return None, f"no buffer probe registered for model class {cls!r}"

    candidates = [("model", model)]
    try:
        inner = model.model
    except AttributeError:
        inner = None
    if inner is not None and inner is not model:
        candidates.append(("model.model", inner))

    smallest: int | None = None
    source = ""
    for where, obj in candidates:
        for label, extract in probe.extractors:
            try:
                dim = int(extract(obj))
            except (AttributeError, TypeError, ValueError, IndexError):
                continue
            if smallest is None or dim < smallest:
                smallest = dim
                source = f"{where}.{label}.shape[0]"

    if smallest is None:
        labels = ", ".join(label for label, _ in probe.extractors)
        return None, (
            f"model class {cls!r} registered but none of its buffers resolved "
            f"({labels}); buffer may not be allocated yet"
        )

    if probe.note:
        source += f" ({probe.note})"
    return smallest, source


def validate_stage(
    stage_name: str,
    model_runner: object,
    *,
    buffer_capacity: int | None = None,
) -> CudaGraphBatchReport:
    """Validate one SGLang-backed stage's batch sizing from its live runner."""
    try:
        server_args = model_runner.server_args
        max_running_requests = server_args.max_running_requests
        cuda_graph_max_bs = server_args.cuda_graph_max_bs
    except AttributeError:
        max_running_requests = None
        cuda_graph_max_bs = None

    try:
        request_slots = model_runner.req_to_token_pool.size
    except AttributeError:
        request_slots = None

    captured_bs = read_captured_bs(model_runner)

    try:
        model = model_runner.model
    except AttributeError:
        model = None
    if buffer_capacity is None:
        buffer_capacity, buffer_source = read_model_buffer_capacity(model)
    else:
        buffer_source = "caller-provided"

    model_cls = type(model).__name__ if model is not None else "unknown-model"

    return evaluate_cuda_graph_batch_sizing(
        stage=f"{stage_name} ({model_cls})",
        max_running_requests=max_running_requests,
        cuda_graph_max_bs=cuda_graph_max_bs,
        captured_bs=captured_bs,
        request_slots=request_slots,
        buffer_capacity=buffer_capacity,
        buffer_source=buffer_source,
    )


_SGLANG_FACTORY_MARKERS = (
    "create_sglang",
    "_thinker_executor_from_config",
    "_talker_ar_executor_from_config",
    "create_generation_executor",
)


def sglang_stage_names(pipeline_config: object) -> list[str]:
    """Return the names of stages whose factory is SGLang-backed."""
    try:
        stages = pipeline_config.stages or []
    except AttributeError:
        return []
    names: list[str] = []
    for stage in stages:
        try:
            name = stage.name
            factory = stage.factory or ""
        except AttributeError:
            continue
        if name and any(marker in factory for marker in _SGLANG_FACTORY_MARKERS):
            names.append(name)
    return names


@dataclass(frozen=True)
class CustomGraphReport:
    """Frame-coverage report for a non-SGLang stage with its own CUDA graphs."""

    stage: str
    captured_frames: list[int] | None
    slot_capacity: int | None
    ok: bool = True
    findings: list[str] = field(default_factory=list)

    def format(self) -> str:
        lines = [f"Stage: {self.stage} (custom graph)"]
        lines.append(f"  captured frames (T): {self.captured_frames}")
        lines.append(f"  slot capacity:       {self.slot_capacity}")
        lines.append("  VERDICT: OK (coverage report)")
        for finding in self.findings:
            lines.append(f"    - {finding}")
        return "\n".join(lines)


@dataclass(frozen=True)
class NoGraphReport:
    """Report for a stage that has no CUDA graph to validate."""

    stage: str
    reason: str = ""
    ok: bool = True

    def format(self) -> str:
        lines = [f"Stage: {self.stage} (no CUDA graph)"]
        lines.append("  VERDICT: OK (no CUDA graph enabled; nothing to flag)")
        if self.reason:
            lines.append(f"    - {self.reason}")
        return "\n".join(lines)


def _read_custom_graph_report(
    stage_name: str, scheduler: object
) -> CustomGraphReport | None:
    """Coverage report for the MOSS streaming vocoder, or None if not one."""
    try:
        session = scheduler._session
        has_graph = session.has_cuda_graph_runner()
    except AttributeError:
        return None
    if not has_graph:
        return None

    try:
        captured = sorted(session.captured_frames())
    except AttributeError:
        captured = None
    try:
        slot_capacity = session._batch_size
    except AttributeError:
        slot_capacity = None

    findings: list[str] = []
    if captured:
        findings.append(
            f"captured {len(captured)} frame-size graph(s) up to T={max(captured)}; "
            f"uncaptured frame sizes fall back to eager."
        )
    findings.append(
        f"graphs are keyed by frame count T against a fixed slot pool "
        f"({slot_capacity}); batch is bounded by the slot allocator, not by a "
        f"serving max_running_requests, so the SGLang batch-overrun check does "
        f"not apply here."
    )
    return CustomGraphReport(
        stage=f"{stage_name} ({type(scheduler).__name__})",
        captured_frames=captured,
        slot_capacity=slot_capacity,
        findings=findings,
    )


def _resolve_sglang_model_runner(scheduler: object) -> object | None:
    """Return the SGLang ModelRunner from a scheduler, or None if non-SGLang."""
    try:
        return scheduler.tp_worker.model_runner
    except AttributeError:
        pass

    try:
        return scheduler._model_runner.tp_worker.model_runner
    except AttributeError:
        return None


def validate_stage_scheduler(
    stage_name: str,
    scheduler: object,
):
    """Validate any stage from its scheduler; always returns a report."""
    model_runner = _resolve_sglang_model_runner(scheduler)
    if model_runner is not None:
        if read_captured_bs(model_runner) is None:
            try:
                model = model_runner.model
            except AttributeError:
                model = None
            model_cls = type(model).__name__ if model is not None else "unknown-model"
            return NoGraphReport(
                stage=f"{stage_name} ({model_cls})",
                reason="SGLang stage with no captured CUDA graph "
                "(cuda_graph disabled or capture skipped).",
            )
        return validate_stage(stage_name, model_runner)

    custom = _read_custom_graph_report(stage_name, scheduler)
    if custom is not None:
        return custom

    return NoGraphReport(
        stage=f"{stage_name} ({type(scheduler).__name__})",
        reason="stage captures no CUDA graphs.",
    )


def validate_stages(stage_schedulers):
    """Validate every ``(stage_name, scheduler)`` pair; one report each."""
    return [
        validate_stage_scheduler(stage_name, scheduler)
        for stage_name, scheduler in stage_schedulers
    ]
