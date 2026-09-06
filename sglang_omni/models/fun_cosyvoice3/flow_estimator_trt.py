# SPDX-License-Identifier: Apache-2.0
"""Opt-in TensorRT engine for the Fun-CosyVoice3 Flow DiT estimator."""

from __future__ import annotations

import hashlib
import logging
import os
import queue
from typing import Any

import torch

logger = logging.getLogger(__name__)

_CFG_BATCH = 2
_MEL_DIM = 80
_DEFAULT_ONNX_CANDIDATES = (
    "flow.decoder.estimator.fp32.onnx",
    "flow.decoder.estimator.onnx",
    "flow.decoder.estimator.autocast_fp16.onnx",
)


def _trt_logger():
    import tensorrt as trt

    return trt.Logger(trt.Logger.WARNING)


def _is_fp16_onnx(onnx_path: str) -> bool:
    name = os.path.basename(onnx_path).lower()
    return "fp16" in name or "autocast" in name


def resolve_flow_estimator_onnx(checkpoint_dir: str) -> str:
    for name in _DEFAULT_ONNX_CANDIDATES:
        path = os.path.join(checkpoint_dir, name)
        if os.path.isfile(path) and os.path.getsize(path) > 0:
            return path
    tried = ", ".join(_DEFAULT_ONNX_CANDIDATES)
    raise FileNotFoundError(
        f"No Flow estimator ONNX found under {checkpoint_dir!r}; looked for {tried}"
    )


def _resolve_plan_path(onnx_path: str) -> str:
    cache_dir = os.environ.get("COSYVOICE3_TRT_CACHE") or os.path.join(
        os.path.expanduser("~"), ".cache", "sglang-omni", "cosyvoice3_trt"
    )
    os.makedirs(cache_dir, exist_ok=True)
    try:
        dev_name = torch.cuda.get_device_name()
    except (RuntimeError, AssertionError):
        dev_name = "unknown"
    import tensorrt as trt

    st = os.stat(onnx_path)
    key = (
        f"{os.path.abspath(onnx_path)}|{st.st_size}|{int(st.st_mtime)}|"
        f"{dev_name}|trt{trt.__version__}|maxb{_CFG_BATCH}"
    )
    digest = hashlib.sha1(key.encode()).hexdigest()[:16]
    return os.path.join(cache_dir, f"flow_estimator_{digest}.plan")


def _dynamic_shapes(time: int) -> dict[str, tuple[int, ...]]:
    # note (guozhihao-224): official ONNX freezes CFG batch=2 and static t/spks;
    # only profile the time dim on x/mask/mu/cond.
    return {
        "x": (_CFG_BATCH, _MEL_DIM, time),
        "mask": (_CFG_BATCH, 1, time),
        "mu": (_CFG_BATCH, _MEL_DIM, time),
        "cond": (_CFG_BATCH, _MEL_DIM, time),
    }


def _cfg_pair_shapes(frames: int) -> dict[str, tuple[int, ...]]:
    return {
        "x": (_CFG_BATCH, _MEL_DIM, frames),
        "mask": (_CFG_BATCH, 1, frames),
        "mu": (_CFG_BATCH, _MEL_DIM, frames),
        "t": (_CFG_BATCH,),
        "spks": (_CFG_BATCH, _MEL_DIM),
        "cond": (_CFG_BATCH, _MEL_DIM, frames),
    }


def _require_cfg_pair_inputs(
    x: torch.Tensor,
    mask: torch.Tensor,
    mu: torch.Tensor,
    t: torch.Tensor,
    spks: torch.Tensor,
    cond: torch.Tensor,
) -> dict[str, tuple[int, ...]]:
    shapes = _cfg_pair_shapes(int(x.shape[2]))
    for (name, want), tensor in zip(
        shapes.items(), (x, mask, mu, t, spks, cond), strict=True
    ):
        got = tuple(tensor.shape)
        if got != want:
            raise ValueError(
                f"Flow-estimator TensorRT input {name} has shape {got}, "
                f"expected {want}"
            )
    return shapes


def _convert_onnx_to_trt(
    onnx_path: str,
    plan_path: str,
    *,
    strongly_typed: bool,
) -> None:
    import tensorrt as trt

    min_time, opt_time, max_time = 4, 500, 3000
    logger.info(
        "Building Flow-estimator TensorRT engine from %s "
        "(CFG batch=%d, time %d..%d, %s)",
        onnx_path,
        _CFG_BATCH,
        min_time,
        max_time,
        "strongly-typed/fp16" if strongly_typed else "fp32+FP16",
    )
    trt_logger = _trt_logger()
    builder = trt.Builder(trt_logger)
    if strongly_typed:
        network = builder.create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
        )
    else:
        network = builder.create_network(0)
    parser = trt.OnnxParser(network, trt_logger)
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            errs = "; ".join(str(parser.get_error(i)) for i in range(parser.num_errors))
            raise ValueError(f"Failed to parse {onnx_path}: {errs}")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 32)
    if not strongly_typed:
        config.set_flag(trt.BuilderFlag.FP16)

    profile = builder.create_optimization_profile()
    mins = _dynamic_shapes(min_time)
    opts = _dynamic_shapes(opt_time)
    maxs = _dynamic_shapes(max_time)
    for name in mins:
        profile.set_shape(name, mins[name], opts[name], maxs[name])
    config.add_optimization_profile(profile)

    engine_bytes = builder.build_serialized_network(network, config)
    if engine_bytes is None:
        raise RuntimeError(
            f"TensorRT failed to build Flow-estimator engine from {onnx_path}"
        )
    tmp = plan_path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(engine_bytes)
    os.replace(tmp, plan_path)
    logger.info("Wrote Flow-estimator TensorRT engine to %s", plan_path)


def _canonicalize_device(device: str | torch.device) -> torch.device:
    dev = torch.device(device)
    if dev.type == "cuda" and dev.index is None:
        return torch.device("cuda", torch.cuda.current_device())
    return dev


class FlowEstimatorTRT:
    def __init__(
        self,
        engine: Any,
        device: str | torch.device,
        *,
        io_dtype: torch.dtype,
        trt_concurrent: int = 1,
    ) -> None:
        self.trt_engine = engine
        self.io_dtype = io_dtype
        self.max_batch = _CFG_BATCH
        self.device = _canonicalize_device(device)
        self._pool: queue.Queue = queue.Queue(maxsize=trt_concurrent)
        for _ in range(trt_concurrent):
            ctx = engine.create_execution_context()
            if ctx is None:
                raise RuntimeError(
                    "failed to create TRT execution context (out of memory?)"
                )
            stream = torch.cuda.Stream(device=self.device)
            self._pool.put([ctx, stream])

    def acquire_estimator(self) -> tuple[list[Any], Any]:
        return self._pool.get(), self.trt_engine

    def release_estimator(self, context: Any, stream: Any) -> None:
        self._pool.put([context, stream])

    def execute(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        mu: torch.Tensor,
        t: torch.Tensor,
        spks: torch.Tensor,
        cond: torch.Tensor,
        streaming: bool = False,
    ) -> torch.Tensor:
        del streaming
        return execute_flow_estimator(self, x, mask, mu, t, spks, cond)


def _enqueue_once(
    estimator: FlowEstimatorTRT,
    x: torch.Tensor,
    mask: torch.Tensor,
    mu: torch.Tensor,
    t: torch.Tensor,
    spks: torch.Tensor,
    cond: torch.Tensor,
) -> torch.Tensor:
    if _canonicalize_device(x.device) != estimator.device:
        raise RuntimeError(
            "Flow-estimator TensorRT device is "
            f"{estimator.device}, got tensors on {x.device}"
        )
    shapes = _require_cfg_pair_inputs(x, mask, mu, t, spks, cond)
    [context, stream], trt_engine = estimator.acquire_estimator()
    caller_stream = torch.cuda.current_stream(estimator.device)
    stream.wait_stream(caller_stream)
    try:
        with torch.cuda.stream(stream):
            inputs = tuple(
                tensor.to(estimator.io_dtype).contiguous()
                for tensor in (x, mask, mu, t, spks, cond)
            )
            out = torch.empty_like(inputs[0])
            for name, shape in shapes.items():
                context.set_input_shape(name, shape)
            bound = (*inputs, out)
            for index, tensor in enumerate(bound):
                context.set_tensor_address(
                    trt_engine.get_tensor_name(index), tensor.data_ptr()
                )
            if context.execute_async_v3(stream.cuda_stream) is not True:
                raise RuntimeError("Flow-estimator TensorRT execute_async_v3 failed")
            for tensor in bound:
                if tensor.is_cuda:
                    tensor.record_stream(stream)
        caller_stream.wait_stream(stream)
        if out.is_cuda:
            out.record_stream(caller_stream)
        return out.to(x.dtype)
    finally:
        estimator.release_estimator(context, stream)


def _run_estimator(
    estimator: Any,
    x: torch.Tensor,
    mask: torch.Tensor,
    mu: torch.Tensor,
    t: torch.Tensor,
    spks: torch.Tensor,
    cond: torch.Tensor,
) -> torch.Tensor:
    # note (guozhihao-224): FlowEstimatorTRT.execute calls execute_flow_estimator;
    # enqueue here to avoid recursion. Test doubles implement execute() instead.
    if isinstance(estimator, FlowEstimatorTRT):
        return _enqueue_once(estimator, x, mask, mu, t, spks, cond)
    return estimator.execute(x, mask, mu, t, spks, cond)


def _take_cfg_pairs(
    tensors: tuple[torch.Tensor, ...],
    start: int,
    end: int,
    request_batch: int,
) -> tuple[torch.Tensor, ...]:
    uncond = request_batch
    return tuple(
        torch.cat([tensor[start:end], tensor[uncond + start : uncond + end]], dim=0)
        for tensor in tensors
    )


def execute_flow_estimator(
    estimator: Any,
    x: torch.Tensor,
    mask: torch.Tensor,
    mu: torch.Tensor,
    t: torch.Tensor,
    spks: torch.Tensor,
    cond: torch.Tensor,
) -> torch.Tensor:
    cfg_batch = int(x.shape[0])
    if cfg_batch < 2 or cfg_batch % 2:
        raise ValueError(
            f"Flow estimator CFG batch must be even and >= 2, got {cfg_batch}"
        )
    # note (guozhihao-224): packed Flow may pass a broadcast timestep (1,);
    # official ONNX/TRT freezes t at CFG batch=2, so expand before enqueue.
    if int(t.shape[0]) == 1 and cfg_batch > 1:
        t = t.expand(cfg_batch).contiguous()
    max_batch = int(estimator.max_batch)
    if cfg_batch <= max_batch:
        return _run_estimator(estimator, x, mask, mu, t, spks, cond)

    # note (guozhihao-224): packed CFG is [cond_0..B, uncond_0..B]; chunk by
    # request pair (slicing the first N rows mixes two conditionals).
    request_batch = cfg_batch // 2
    max_requests = max(1, max_batch // 2)
    out = torch.empty_like(x)
    for start in range(0, request_batch, max_requests):
        end = min(start + max_requests, request_batch)
        chunks = _take_cfg_pairs(
            (x, mask, mu, t, spks, cond), start, end, request_batch
        )
        y = _run_estimator(estimator, *chunks)
        n = end - start
        out[start:end] = y[:n]
        out[request_batch + start : request_batch + end] = y[n:]
    return out


def is_flow_estimator_trt(estimator: Any) -> bool:
    if isinstance(estimator, torch.nn.Module):
        return False
    if isinstance(estimator, FlowEstimatorTRT):
        return True
    return hasattr(estimator, "execute")


def build_flow_estimator_trt(
    onnx_path: str,
    device: str | torch.device,
    *,
    trt_concurrent: int = 1,
) -> FlowEstimatorTRT:
    try:
        import tensorrt as trt
    except ImportError as exc:
        raise RuntimeError(
            "enable_flow_estimator_trt requires the tensorrt package. "
            "Install NVIDIA TensorRT in the serving environment."
        ) from exc

    strongly_typed = _is_fp16_onnx(onnx_path)
    io_dtype = torch.float16 if strongly_typed else torch.float32
    plan_path = _resolve_plan_path(onnx_path)
    if not os.path.exists(plan_path) or os.path.getsize(plan_path) == 0:
        _convert_onnx_to_trt(onnx_path, plan_path, strongly_typed=strongly_typed)

    runtime = trt.Runtime(_trt_logger())
    with open(plan_path, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())
    if engine is None:
        raise RuntimeError(
            f"Failed to deserialize Flow-estimator TensorRT engine {plan_path}"
        )
    logger.info(
        "Loaded Flow-estimator TensorRT engine (%s, %.1f MiB, max_batch=%d)",
        plan_path,
        os.path.getsize(plan_path) / (1 << 20),
        _CFG_BATCH,
    )
    return FlowEstimatorTRT(
        engine,
        device,
        io_dtype=io_dtype,
        trt_concurrent=trt_concurrent,
    )
