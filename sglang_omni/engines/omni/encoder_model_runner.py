# SPDX-License-Identifier: Apache-2.0
"""EncoderModelRunner - optional cache-aware model executor."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import hashlib
import logging
from typing import TYPE_CHECKING, Any, Iterable

import torch

from .runtime.encoder import EncoderBatchData
from .types import ModelRunnerOutput, RequestOutput, SchedulerOutput

if TYPE_CHECKING:
    from .runtime.interfaces import InputPreparer, OutputProcessor

logger = logging.getLogger(__name__)


@dataclass
class _CacheEntry:
    data: Any
    finished: bool
    finish_reason: str | None


def _iter_request_indices(indices: Iterable[int]) -> list[int]:
    return list(indices)


def _hash_tensor(value: torch.Tensor) -> str:
    cpu = value.detach().contiguous().cpu()
    payload = cpu.numpy().tobytes()
    meta = f"{cpu.dtype}|{tuple(cpu.shape)}".encode("utf-8")
    return hashlib.sha256(meta + payload).hexdigest()


def _hash_value(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return _hash_tensor(value)
    if isinstance(value, (list, tuple)):
        parts = [_hash_value(v) for v in value]
        if any(p is None for p in parts):
            return None
        return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()
    if isinstance(value, dict):
        items = []
        for key in sorted(value.keys()):
            hashed = _hash_value(value[key])
            if hashed is None:
                return None
            items.append(f"{key}={hashed}")
        return hashlib.sha256("|".join(items).encode("utf-8")).hexdigest()
    try:
        return hashlib.sha256(str(value).encode("utf-8")).hexdigest()
    except Exception:
        return None


def _detach_value(value: Any, *, device: torch.device | None) -> Any:
    if isinstance(value, torch.Tensor):
        value = value.detach()
        if device is not None:
            value = value.to(device=device)
        return value
    if isinstance(value, dict):
        return {k: _detach_value(v, device=device) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_detach_value(v, device=device) for v in value)
    return value


class EncoderModelRunner:
    """ModelRunner with optional encoder output caching."""

    def __init__(
        self,
        model: torch.nn.Module,
        input_preparer: InputPreparer,
        output_processor: OutputProcessor,
        *,
        device: torch.device | str = "cuda",
        use_cache: bool = False,
        cache_size: int | None = None,
        cache_device: torch.device | str | None = None,
    ):
        if isinstance(device, str):
            device = torch.device(device)
        if isinstance(cache_device, str):
            cache_device = torch.device(cache_device)

        self.device = device
        self.input_preparer = input_preparer
        self.output_processor = output_processor
        self.use_cache = use_cache
        self.cache_size = cache_size
        self.cache_device = cache_device

        self._cache: OrderedDict[str, _CacheEntry] = OrderedDict()

        self.model = model.to(device)
        self.model.eval()

    def clear_cache(self) -> None:
        self._cache.clear()

    def _get_cache_key(self, request: Any) -> str | None:
        data = getattr(request, "data", None)
        if data is None:
            return None
        cache_key = getattr(data, "cache_key", None)
        if cache_key is None and isinstance(data, dict):
            cache_key = data.get("cache_key")
        if cache_key is not None:
            return str(cache_key)
        input_dict = getattr(data, "input_dict", None)
        if input_dict is None and isinstance(data, dict):
            input_dict = data.get("input_dict")
        if input_dict is not None:
            return _hash_value(input_dict)
        input_ids = getattr(data, "input_ids", None)
        if input_ids is None and isinstance(data, dict):
            input_ids = data.get("input_ids")
        return _hash_value(input_ids)

    def _slice_batch_data(
        self, batch_data: Any, indices: list[int]
    ) -> EncoderBatchData | list[Any] | None:
        if isinstance(batch_data, EncoderBatchData):
            return EncoderBatchData(
                input_ids_list=[batch_data.input_ids_list[i] for i in indices],
                seq_lens=[batch_data.seq_lens[i] for i in indices],
            )
        if isinstance(batch_data, list):
            return [batch_data[i] for i in indices]
        if isinstance(batch_data, tuple):
            return [batch_data[i] for i in indices]
        return None

    def _touch_cache(self, key: str) -> None:
        if key in self._cache:
            self._cache.move_to_end(key)

    def _set_cache(self, key: str, output: RequestOutput) -> None:
        if not self.use_cache:
            return
        entry = _CacheEntry(
            data=_detach_value(output.data, device=self.cache_device),
            finished=bool(output.finished),
            finish_reason=output.finish_reason,
        )
        self._cache[key] = entry
        self._cache.move_to_end(key)
        if self.cache_size is not None:
            while len(self._cache) > self.cache_size:
                self._cache.popitem(last=False)

    def _execute_batch(self, scheduler_output: SchedulerOutput) -> dict[str, RequestOutput]:
        model_inputs = self.input_preparer.prepare(scheduler_output, self.device)
        with torch.inference_mode():
            model_output = self.model(**model_inputs)
        return self.output_processor.process(model_output, scheduler_output)

    def execute(self, scheduler_output: SchedulerOutput) -> ModelRunnerOutput:
        if not self.use_cache or scheduler_output.num_requests == 0:
            outputs = self._execute_batch(scheduler_output)
            for request in scheduler_output.requests:
                key = self._get_cache_key(request)
                if key is not None and request.request_id in outputs:
                    self._set_cache(key, outputs[request.request_id])
            req_ids = [req.request_id for req in scheduler_output.requests]
            req_id_to_index = {req_id: idx for idx, req_id in enumerate(req_ids)}
            return ModelRunnerOutput(
                outputs=outputs,
                req_ids=req_ids,
                req_id_to_index=req_id_to_index,
            )

        cached_outputs: dict[str, RequestOutput] = {}
        uncached_requests: list[Any] = []
        uncached_indices: list[int] = []
        request_keys: list[str | None] = []

        for idx, request in enumerate(scheduler_output.requests):
            key = self._get_cache_key(request)
            request_keys.append(key)
            if key is None or key not in self._cache:
                uncached_requests.append(request)
                uncached_indices.append(idx)
                continue
            entry = self._cache[key]
            self._touch_cache(key)
            cached_outputs[request.request_id] = RequestOutput(
                request_id=request.request_id,
                data=entry.data,
                finished=entry.finished,
                finish_reason=entry.finish_reason,
            )

        if not uncached_requests:
            req_ids = [req.request_id for req in scheduler_output.requests]
            req_id_to_index = {req_id: idx for idx, req_id in enumerate(req_ids)}
            return ModelRunnerOutput(
                outputs=cached_outputs,
                req_ids=req_ids,
                req_id_to_index=req_id_to_index,
            )

        sliced_batch_data = self._slice_batch_data(
            scheduler_output.batch_data, _iter_request_indices(uncached_indices)
        )
        if sliced_batch_data is None:
            outputs = self._execute_batch(scheduler_output)
            for request in scheduler_output.requests:
                key = self._get_cache_key(request)
                if key is not None and request.request_id in outputs:
                    self._set_cache(key, outputs[request.request_id])
            req_ids = [req.request_id for req in scheduler_output.requests]
            req_id_to_index = {req_id: idx for idx, req_id in enumerate(req_ids)}
            return ModelRunnerOutput(
                outputs=outputs,
                req_ids=req_ids,
                req_id_to_index=req_id_to_index,
            )

        sub_output = SchedulerOutput(
            requests=uncached_requests,
            batch_data=sliced_batch_data,
            step_id=scheduler_output.step_id,
        )
        outputs = self._execute_batch(sub_output)
        merged_outputs = dict(cached_outputs)
        merged_outputs.update(outputs)

        for request in uncached_requests:
            key = self._get_cache_key(request)
            if key is None:
                continue
            output = outputs.get(request.request_id)
            if output is None:
                continue
            self._set_cache(key, output)

        req_ids = [req.request_id for req in scheduler_output.requests]
        req_id_to_index = {req_id: idx for idx, req_id in enumerate(req_ids)}
        return ModelRunnerOutput(
            outputs=merged_outputs,
            req_ids=req_ids,
            req_id_to_index=req_id_to_index,
        )

