# SPDX-License-Identifier: Apache-2.0
"""Process-local SGLang diffusion runtime for the image decoder."""

from __future__ import annotations

import secrets
import socket
from collections.abc import Iterator
from contextlib import contextmanager, nullcontext
from typing import Literal

import torch
import torch.distributed as dist


class DecoderRuntimeHandle:
    """Own the SGLang diffusion state used by one decoder rank."""

    def __init__(
        self,
        device: torch.device,
        dtype: torch.dtype,
        attention_backend: str,
        parallel_state,
        server_args_module,
        runtime_context,
        precision,
        *,
        sp_rank: int = 0,
        sp_size: int = 1,
        ulysses_degree: int = 1,
        ring_degree: int = 1,
    ) -> None:
        self.device = device
        self.dtype = dtype
        self.attention_backend = attention_backend
        self.sp_rank = sp_rank
        self.sp_size = sp_size
        self.ulysses_degree = ulysses_degree
        self.ring_degree = ring_degree
        self.group: dist.ProcessGroup | None = None
        self.cpu_group: dist.ProcessGroup | None = None
        self._parallel_state = parallel_state
        self._server_args_module = server_args_module
        self._runtime_context = runtime_context
        self._precision = precision
        self._world_started = False
        self._model_started = False
        self._published = False
        self._closed = False

    @contextmanager
    def compute_context(self):
        if self._closed:
            raise RuntimeError("Decoder runtime is closed")
        device_context = (
            torch.cuda.device(self.device)
            if self.device.type == "cuda"
            else nullcontext()
        )
        with device_context:
            state = self._precision._mixed_precision_state
            missing = object()
            previous = getattr(state, "state", missing)
            self._precision.set_mixed_precision_policy(
                param_dtype=self.dtype, reduce_dtype=torch.float32
            )
            try:
                yield
            finally:
                if previous is missing:
                    del state.state
                else:
                    state.state = previous

    def validate(self) -> None:
        if self._closed:
            raise RuntimeError("Decoder runtime is closed")
        ps = self._parallel_state
        if not dist.is_initialized() or not ps.model_parallel_is_initialized():
            raise RuntimeError("SGLang decoder runtime is not initialized")
        topology = (
            ps.get_world_size(),
            ps.get_tp_world_size(),
            ps.get_sp_world_size(),
            ps.get_sp_parallel_rank(),
            ps.get_ulysses_parallel_world_size(),
            ps.get_ring_parallel_world_size(),
        )
        expected = (
            self.sp_size,
            1,
            self.sp_size,
            self.sp_rank,
            self.ulysses_degree,
            self.ring_degree,
        )
        if topology != expected:
            raise RuntimeError(
                f"SGLang decoder topology {topology} does not match {expected}"
            )
        group = ps.get_sp_group().device_group
        if self.group is not None and self.group is not group:
            raise RuntimeError("Decoder process group changed after initialization")
        self.group = group
        self.cpu_group = ps.get_sp_group().cpu_group
        if self._precision.get_compute_dtype() != self.dtype:
            raise RuntimeError(
                "Decoder runtime compute dtype does not match model dtype"
            )
        if (
            self.device.type == "cuda"
            and torch.cuda.current_device() != self.device.index
        ):
            raise RuntimeError("Decoder CUDA device does not match the runtime device")

    @property
    def is_leader(self) -> bool:
        return self.sp_rank == 0

    @contextmanager
    def preparation(self, phase: str) -> Iterator[None]:
        if self.sp_size == 1:
            yield
            return
        error = None
        try:
            yield
        except Exception as exc:
            error = exc
        failures: list[str | None] = [None] * self.sp_size
        dist.all_gather_object(
            failures,
            None if error is None else f"{type(error).__name__}: {error}",
            group=self.cpu_group,
        )
        if any(failure is not None for failure in failures):
            raise RuntimeError(
                f"Decoder {phase} failed across ranks: {failures}"
            ) from error

    def request_seed(
        self, metadata: tuple[int, int, str, int, int], seed: int | None
    ) -> int | None:
        self.validate()
        if self.sp_size == 1:
            return seed
        request = (metadata, seed)
        requests: list[tuple[tuple[int, int, str, int, int], int | None] | None] = [
            None
        ] * self.sp_size
        dist.all_gather_object(requests, request, group=self.cpu_group)
        if any(candidate != request for candidate in requests):
            raise ValueError("Decoder ranks received inconsistent request settings")
        shared = [
            (
                seed
                if seed is not None
                else (secrets.randbits(63) if self.is_leader else None)
            )
        ]
        dist.broadcast_object_list(
            shared, src=dist.get_global_rank(self.cpu_group, 0), group=self.cpu_group
        )
        return shared[0]

    def broadcast_features(self, features: torch.Tensor) -> torch.Tensor:
        if self.sp_size > 1:
            dist.broadcast(
                features, src=dist.get_global_rank(self.group, 0), group=self.group
            )
        return features

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            if self._model_started:
                self._parallel_state.destroy_model_parallel()
        finally:
            try:
                if self._world_started:
                    self._parallel_state.destroy_distributed_environment()
            finally:
                if self._published:
                    try:
                        self._server_args_module.set_global_server_args(None)
                    finally:
                        self._runtime_context.reset_context()

    def __enter__(self):
        if self._closed:
            raise RuntimeError("Decoder runtime is closed")
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        try:
            self.close()
        except BaseException as cleanup:
            if exc is not None:
                raise exc from cleanup
            raise


def initialize_decoder_runtime(
    model_path: str,
    *,
    gpu_id: int | None = 0,
    dtype: torch.dtype = torch.bfloat16,
    attention_backend: str = "torch_sdpa",
    dist_timeout: int = 180,
    sp_rank: int = 0,
    sp_size: int = 1,
    stage_role: Literal["single", "leader", "follower"] = "single",
    nccl_port: int | None = None,
    ulysses_degree: int = 1,
    ring_degree: int = 1,
) -> DecoderRuntimeHandle:
    """Initialize the upstream diffusion runtime for one decoder process."""
    if gpu_id is not None and (type(gpu_id) is not int or gpu_id < 0):
        raise ValueError("Decoder gpu_id must be a nonnegative CUDA ordinal or None")
    if type(dist_timeout) is not int or dist_timeout < 1:
        raise ValueError("Decoder dist_timeout must be positive seconds")
    if not attention_backend or attention_backend == "auto":
        raise ValueError("Decoder attention_backend must be explicit")
    if any(type(n) is not int or n < 1 for n in (sp_size, ulysses_degree, ring_degree)):
        raise ValueError("Decoder parallel degrees must be positive integers")
    if type(sp_rank) is not int or not 0 <= sp_rank < sp_size:
        raise ValueError("Decoder sp_rank must be in [0, sp_size)")
    if sp_size != ulysses_degree * ring_degree:
        raise ValueError("Decoder sp_size must equal ulysses_degree * ring_degree")
    expected_role = (
        "single" if sp_size == 1 else ("leader" if sp_rank == 0 else "follower")
    )
    if stage_role != expected_role:
        raise ValueError(
            f"Decoder rank {sp_rank}/{sp_size} requires role {expected_role}"
        )
    if nccl_port is not None and (
        type(nccl_port) is not int or not 1 <= nccl_port <= 65535
    ):
        raise ValueError("Decoder nccl_port must be an integer in [1, 65535]")
    if sp_size > 1 and nccl_port is None:
        raise ValueError("SP workers require a shared nccl_port")
    precisions = {
        torch.float32: "fp32",
        torch.float16: "fp16",
        torch.bfloat16: "bf16",
    }
    if dtype not in precisions:
        raise ValueError("Unsupported decoder runtime dtype")
    if dist.is_initialized():
        raise RuntimeError(
            "Decoder initialization requires a dedicated process with no existing groups"
        )

    if nccl_port is None:
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            nccl_port = sock.getsockname()[1]
    port = nccl_port
    address = f"127.0.0.1:{port}"

    from sglang.multimodal_gen import utils as precision
    from sglang.multimodal_gen.configs.pipeline_configs.zimage import (
        ZImagePipelineConfig,
    )
    from sglang.multimodal_gen.runtime.distributed import parallel_state as ps
    from sglang.multimodal_gen.runtime.server_args import server_args as args_module
    from sglang.srt import runtime_context
    from sglang.srt.server_args import ServerArgs as SrtServerArgs

    if (
        ps.world_group_is_initialized()
        or ps.model_parallel_is_initialized()
        or args_module._global_server_args is not None
        or runtime_context.get_context()._server_args is not None
    ):
        raise RuntimeError("Decoder initialization found existing SGLang runtime state")

    device = torch.device("cpu" if gpu_id is None else f"cuda:{gpu_id}")
    runtime = DecoderRuntimeHandle(
        device,
        dtype,
        attention_backend,
        ps,
        args_module,
        runtime_context,
        precision,
        sp_rank=sp_rank,
        sp_size=sp_size,
        ulysses_degree=ulysses_degree,
        ring_degree=ring_degree,
    )
    try:
        if gpu_id is not None:
            torch.cuda.set_device(gpu_id)
        args = args_module.ServerArgs(
            model_path=model_path,
            backend=args_module.Backend.SGLANG,
            pipeline_config=ZImagePipelineConfig(dit_precision=precisions[dtype]),
            performance_mode="manual",
            num_gpus=sp_size,
            tp_size=1,
            sp_degree=sp_size,
            dp_size=1,
            ulysses_degree=ulysses_degree,
            ring_degree=ring_degree,
            kv_gather_degree=1,
            sp_split_auto=False,
            enable_cfg_parallel=False,
            cfg_parallel_degree=1,
            attention_backend=attention_backend,
            use_fsdp_inference=False,
            dit_cpu_offload=False,
            dit_layerwise_offload=False,
            vae_cpu_offload=False,
            text_encoder_cpu_offload=False,
            image_encoder_cpu_offload=False,
            enable_torch_compile=False,
            enable_breakable_cuda_graph=False,
            nccl_port=port,
            dist_init_addr=address,
            dist_timeout=dist_timeout,
        )
        runtime._published = True
        args_module.set_global_server_args(args)
        runtime._world_started = True
        ps.init_distributed_environment(
            world_size=sp_size,
            rank=sp_rank,
            distributed_init_method=f"tcp://{address}",
            local_rank=gpu_id or 0,
            backend="gloo" if gpu_id is None else "nccl",
            device_id=None if gpu_id is None else device,
            timeout=dist_timeout,
        )
        runtime._model_started = True
        ps.initialize_model_parallel(
            data_parallel_size=1,
            classifier_free_guidance_degree=1,
            sequence_parallel_degree=sp_size,
            ulysses_degree=ulysses_degree,
            ring_degree=ring_degree,
            tensor_parallel_degree=1,
            pipeline_parallel_degree=1,
            vae_parallel_size=0,
            backend="gloo" if gpu_id is None else "nccl",
        )
        runtime_context.publish(
            SrtServerArgs(model_path="dummy", tp_size=1),
            role="diffusion_gpu_worker",
        )
        with runtime.compute_context():
            runtime.validate()
    except BaseException as error:
        try:
            runtime.close()
        except BaseException as cleanup:
            raise error from cleanup
        raise
    return runtime
