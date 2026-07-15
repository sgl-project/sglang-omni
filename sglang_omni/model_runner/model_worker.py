from __future__ import annotations

import logging
import os
import socket
from dataclasses import dataclass
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

from sglang_omni.quantization import (
    needs_quant_config_normalization,
    normalize_quant_config,
    resolve_quant_config,
)

if TYPE_CHECKING:
    from sglang.srt.configs.model_config import ModelConfig
    from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)


@dataclass
class ModelWorkerConfig:
    model_arch_override: str | None = None
    weight_prefix: str | None = None
    nccl_port: int | None = None
    total_gpu_memory_fraction: float | None = None


_ARCH_CONFIG_MAP: dict[str, tuple[str, str | None]] = {
    "BailingMoeV2ForCausalLM": ("llm_config", None),
    "Qwen3OmniTalker": ("talker_config", "text_config"),
    "Qwen3OmniThinkerForCausalLM": ("thinker_config", "text_config"),
    "Qwen3ASRForConditionalGeneration": ("thinker_config", "text_config"),
    "Qwen3TTSTalker": ("talker_config", None),
    "MossTTSDelaySGLangModel": ("language_config", None),
    "MossTTSLocalSGLangModel": ("language_config", None),
    "MossTranscribeDiarizeForConditionalGeneration": ("text_config", None),
}


class ModelWorker:
    def __init__(
        self,
        config: ModelWorkerConfig,
        server_args: ServerArgs,
        gpu_id: int,
        tp_rank: int = 0,
    ):
        self.server_args = server_args
        self.model_arch_override = config.model_arch_override
        self.weight_prefix = config.weight_prefix
        self.nccl_port = config.nccl_port
        self.total_gpu_memory_fraction = config.total_gpu_memory_fraction

        self.gpu_id = gpu_id
        self.tp_rank = tp_rank
        self._init_model_config()
        self._configure_backend_policy()
        self._init_model_runner()
        self._init_dllm_algorithm()

        self.device = self.model_runner.device
        from sglang.srt.utils import broadcast_pyobj, set_random_seed

        self.random_seed = broadcast_pyobj(
            [server_args.random_seed],
            self.tp_rank,
            self.model_runner.tp_group.cpu_group,
        )[0]
        set_random_seed(self.random_seed)

    def _init_model_config(self):
        if self.model_arch_override == "BailingMoeV2ForCausalLM":
            from sglang_omni.models.ming_omni.registration import (
                register_ming_hf_config,
            )

            register_ming_hf_config()

        from sglang.srt.configs.model_config import ModelConfig

        self.model_config = ModelConfig.from_server_args(
            server_args=self.server_args,
            model_path=self.server_args.model_path,
            model_revision=self.server_args.revision,
            is_draft_model=False,
        )

        if self.model_arch_override is not None:
            self._apply_arch_override(self.model_config, self.model_arch_override)

    @staticmethod
    def _apply_arch_override(model_config: ModelConfig, arch: str) -> None:
        """Override model config for a sub-model architecture."""
        model_config.hf_config.architectures = [arch]
        if arch == "WhisperForConditionalGeneration":
            cfg = model_config.hf_config
            model_config.hf_text_config = cfg
            model_config.is_encoder_decoder = True
            model_config.hidden_size = int(cfg.d_model)
            model_config.num_attention_heads = int(cfg.decoder_attention_heads)
            model_config.num_key_value_heads = int(cfg.decoder_attention_heads)
            model_config.num_hidden_layers = int(cfg.decoder_layers)
            model_config.num_attention_layers = int(cfg.decoder_layers) * 2
            model_config.vocab_size = int(cfg.vocab_size)
            model_config.head_dim = int(cfg.d_model) // int(cfg.decoder_attention_heads)
            model_config.v_head_dim = model_config.head_dim
            return
        entry = _ARCH_CONFIG_MAP.get(arch)
        if entry is None:
            return
        sub_config_attr, text_config_attr = entry
        sub_cfg = getattr(model_config.hf_config, sub_config_attr, None)
        if sub_cfg is None:
            return
        text_cfg = getattr(sub_cfg, text_config_attr) if text_config_attr else sub_cfg
        model_config.hf_text_config = text_cfg
        model_config.num_attention_heads = text_cfg.num_attention_heads
        model_config.num_key_value_heads = text_cfg.num_key_value_heads
        model_config.hidden_size = text_cfg.hidden_size
        model_config.num_hidden_layers = text_cfg.num_hidden_layers

    def _configure_backend_policy(self) -> None:
        # Apply Omni-specific quantization adapters (stage-local checkpoint name
        # normalization) before SGLang builds its quant config, then run the
        # model_worker backend policy.
        _apply_omni_quantization_adapters(self.model_config)

        effective_quantization = _apply_model_worker_backend_policy(
            self.server_args,
            self.model_config,
            self.model_arch_override,
        )
        _initialize_model_worker_backend_globals(
            self.server_args,
            self.model_config,
            effective_quantization,
        )

    def get_memory_pool(self):
        return (
            self.model_runner.req_to_token_pool,
            self.model_runner.token_to_kv_pool_allocator,
        )

    def get_worker_info(self):
        max_total_num_tokens = self.model_runner.max_total_num_tokens
        max_req_len = min(self.server_args.context_length - 1, max_total_num_tokens - 1)
        max_req_input_len = max_req_len - 1
        req_pool = self.model_runner.req_to_token_pool
        kv_pool = self.model_runner.token_to_kv_pool_allocator
        return (
            max_total_num_tokens,
            self.server_args.max_prefill_tokens,
            self.server_args.max_running_requests,
            self.server_args.max_queued_requests,
            max_req_len,
            max_req_input_len,
            self.random_seed,
            self.device,
            req_pool.size,
            req_pool.max_context_len,
            kv_pool.size,
        )

    def get_tp_group(self):
        return self.model_runner.tp_group

    def get_attention_tp_group(self):
        return self.model_runner.attention_tp_group

    def get_attention_tp_cpu_group(self):
        return self.model_runner.attention_tp_group.cpu_group

    def get_pad_input_ids_func(self):
        return getattr(self.model_runner.model, "pad_input_ids", None)

    def _init_model_runner(self):
        from .sglang_model_runner import SGLModelRunner

        nccl_port = (
            self.nccl_port if self.nccl_port is not None else _resolve_nccl_port()
        )
        self.model_runner = SGLModelRunner(
            model_config=self.model_config,
            server_args=self.server_args,
            gpu_id=self.gpu_id,
            tp_rank=self.tp_rank,
            moe_ep_rank=0,
            moe_ep_size=1,
            pp_rank=0,
            pp_size=1,
            nccl_port=nccl_port,
            model_arch_override=self.model_arch_override,
            weight_prefix=self.weight_prefix,
            total_gpu_memory_fraction=self.total_gpu_memory_fraction,
        )

    def _init_dllm_algorithm(self):
        if self.server_args.dllm_algorithm is None:
            self.dllm_algorithm = None
            return

        from sglang.srt.dllm.algorithm.base import DllmAlgorithm

        self.dllm_algorithm = DllmAlgorithm.from_server_args(self.server_args)

    def forward_batch_generation(
        self,
        forward_batch,
    ):
        from sglang.srt.managers.scheduler import GenerationBatchResult

        if self.dllm_algorithm is not None:
            logits_output, next_token_ids, can_run_cuda_graph = self.dllm_algorithm.run(
                self.model_runner, forward_batch
            )
            return GenerationBatchResult(
                logits_output=logits_output,
                next_token_ids=next_token_ids,
                can_run_cuda_graph=can_run_cuda_graph,
            )

        out = self.model_runner.forward(forward_batch=forward_batch)
        logits_output, can_run_cuda_graph = out.logits_output, out.can_run_graph
        batch_result = GenerationBatchResult(
            logits_output=logits_output,
            can_run_cuda_graph=can_run_cuda_graph,
            expert_distribution_metrics=out.expert_distribution_metrics,
        )
        return batch_result

    def model_info(self) -> dict[str, Any]:
        return {
            "model_path": self.server_args.model_path,
            "load_format": self.server_args.load_format,
            "weight_version": self.server_args.weight_version,
            "tp_rank": self.tp_rank,
            "tp_size": self.server_args.tp_size,
            "model_arch_override": self.model_arch_override,
            "supports_weight_update": hasattr(
                self.model_runner, "update_weights_from_disk"
            ),
            "supports_weight_checker": True,
        }

    def update_weights_from_disk(self, payload: dict[str, Any]) -> tuple[bool, str]:
        model_path = payload.get("model_path")
        if not model_path:
            return False, "model_path is required"
        update = getattr(self.model_runner, "update_weights_from_disk", None)
        if update is None:
            return False, "model runner does not support update_weights_from_disk"
        load_format = payload.get("load_format") or self.server_args.load_format
        success, message = update(
            model_path,
            load_format,
            recapture_cuda_graph=bool(payload.get("recapture_cuda_graph", False)),
        )
        if success:
            runner_args = getattr(self.model_runner, "server_args", None)
            setattr(self.server_args, "model_path", model_path)
            setattr(self.server_args, "load_format", load_format)
            if runner_args is not None:
                setattr(runner_args, "model_path", model_path)
                setattr(runner_args, "load_format", load_format)
            model_config = getattr(self.model_runner, "model_config", None)
            if model_config is not None:
                setattr(model_config, "model_path", model_path)

            weight_version = payload.get("weight_version")
            if weight_version is not None:
                setattr(self.server_args, "weight_version", weight_version)
                if runner_args is not None:
                    setattr(runner_args, "weight_version", weight_version)
        return bool(success), str(message)

    def update_weights_from_tensor(self, payload: dict[str, Any]) -> tuple[bool, str]:
        if payload.get("serialized_named_tensors") is not None:
            return (
                False,
                "update_weights_from_tensor requires a tensor data plane; "
                "Omni admin control plane only carries metadata",
            )
        return self._call_optional_weight_method("update_weights_from_tensor", payload)

    def init_weights_update_group(self, payload: dict[str, Any]) -> tuple[bool, str]:
        init = getattr(self.model_runner, "init_weights_update_group", None)
        if init is None:
            return False, "model runner does not support init_weights_update_group"
        master_address = payload.get("master_address")
        master_port = payload.get("master_port")
        world_size = payload.get("world_size")
        if not master_address or master_port is None or world_size is None:
            return False, "master_address, master_port and world_size are required"
        try:
            master_port_int = int(master_port)
            rank_offset_int = int(payload.get("rank_offset", 0))
            world_size_int = int(world_size)
        except (TypeError, ValueError):
            return False, "master_port, rank_offset and world_size must be integers"
        success, message = init(
            master_address,
            master_port_int,
            rank_offset_int,
            world_size_int,
            payload.get("group_name") or "weight_update_group",
            backend=payload.get("backend") or "nccl",
        )
        return bool(success), str(message)

    def destroy_weights_update_group(self, payload: dict[str, Any]) -> tuple[bool, str]:
        destroy = getattr(self.model_runner, "destroy_weights_update_group", None)
        if destroy is None:
            return False, "model runner does not support destroy_weights_update_group"
        success, message = destroy(payload.get("group_name") or "weight_update_group")
        return bool(success), str(message)

    def update_weights_from_distributed(
        self, payload: dict[str, Any]
    ) -> tuple[bool, str]:
        update = getattr(self.model_runner, "update_weights_from_distributed", None)
        if update is None:
            return (
                False,
                "model runner does not support update_weights_from_distributed",
            )
        names = payload.get("names")
        dtypes = payload.get("dtypes")
        shapes = payload.get("shapes")
        if names is None or dtypes is None or shapes is None:
            return False, "names, dtypes and shapes are required"
        # Pydantic already guards type/None at the HTTP boundary; this length
        # check is the one guard that matters — sglang zips names/dtypes/shapes
        # and silently truncates to the shortest, under-broadcasting weights.
        name_count = len(names)
        dtype_count = len(dtypes)
        shape_count = len(shapes)
        if name_count == 0 or dtype_count == 0 or shape_count == 0:
            return False, "names, dtypes and shapes must be non-empty"
        if name_count != dtype_count or name_count != shape_count:
            return False, "names, dtypes and shapes must have the same length"
        success, message = update(
            names,
            dtypes,
            shapes,
            payload.get("group_name") or "weight_update_group",
            load_format=payload.get("load_format"),
        )
        if success:
            weight_version = payload.get("weight_version")
            if weight_version is not None:
                setattr(self.server_args, "weight_version", weight_version)
                runner_args = getattr(self.model_runner, "server_args", None)
                if runner_args is not None:
                    setattr(runner_args, "weight_version", weight_version)
        return bool(success), str(message)

    def weights_checker(self, action: str) -> dict[str, Any]:
        checker = getattr(self, "_strict_weight_checker", None)
        if checker is None:
            from sglang_omni.model_runner.weight_checker import StrictWeightChecker

            checker = StrictWeightChecker(self.model_runner)
            self._strict_weight_checker = checker
        return checker.run(action)

    def _call_optional_weight_method(
        self,
        method_name: str,
        payload: dict[str, Any],
    ) -> tuple[bool, str]:
        method = getattr(self.model_runner, method_name, None)
        if method is None:
            return False, f"model runner does not support {method_name}"
        recv_req = SimpleNamespace(**payload)
        success, message = method(recv_req)
        return bool(success), str(message)


def _resolve_nccl_port() -> int:
    master_port = os.environ.get("MASTER_PORT")
    if master_port:
        return int(master_port)

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("", 0))
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            port = sock.getsockname()[1]
    except PermissionError:
        # Some restricted CI / sandbox environments do not allow ephemeral socket
        # binding during test-time configuration. Fall back to a stable default so
        # callers still receive a valid NCCL port choice.
        port = 29500

    os.environ["MASTER_PORT"] = str(port)
    return port


def _apply_model_worker_backend_policy(
    server_args: ServerArgs,
    model_config: ModelConfig,
    model_arch_override: str | None,
) -> str | None:
    """Apply Omni backend policy after checkpoint quantization is known."""

    effective_quantization = _normalize_quantization(
        getattr(model_config, "quantization", None)
    )
    server_quantization = _normalize_quantization(server_args.quantization)
    if server_quantization is not None:
        effective_quantization = server_quantization

    moe_runner_backend = server_args.moe_runner_backend
    is_qwen3_omni_arch = model_arch_override in (
        "Qwen3OmniTalker",
        "Qwen3OmniThinkerForCausalLM",
    )
    if is_qwen3_omni_arch and server_args.ep_size != 1:
        raise ValueError(
            "Qwen3-Omni ModelWorker does not support expert parallelism; "
            "use ep_size=1."
        )
    has_moe = _model_config_has_moe(model_config)
    has_native_fp8_block_quant = _model_config_has_native_fp8_block_quant(model_config)

    if (
        model_arch_override == "Qwen3OmniTalker"
        and effective_quantization is None
        and moe_runner_backend == "auto"
    ):
        # Note:(Chenchen Hong) flashinfer_cutlass MoE deadlocks CUDA-graph
        # capture on H20 (no H20 kernel coverage); triton captures cleanly there.
        server_args.moe_runner_backend = (
            "triton" if _is_h20_device() else "flashinfer_cutlass"
        )
        moe_runner_backend = server_args.moe_runner_backend

    if (
        is_qwen3_omni_arch
        and effective_quantization == "fp8"
        and has_moe
        and moe_runner_backend == "auto"
        and has_native_fp8_block_quant
        and _is_fp8_cutlass_moe_supported()
    ):
        server_args.moe_runner_backend = "cutlass"
        moe_runner_backend = server_args.moe_runner_backend

    if (
        is_qwen3_omni_arch
        and effective_quantization == "fp8"
        and has_moe
        and moe_runner_backend == "cutlass"
    ):
        if not has_native_fp8_block_quant:
            raise ValueError(
                "Qwen3-Omni FP8 CUTLASS MoE requires a native serialized "
                "block-FP8 checkpoint with weight_block_size."
            )

    if (
        is_qwen3_omni_arch
        and effective_quantization == "fp8"
        and moe_runner_backend == "flashinfer_cutlass"
    ):
        raise ValueError(
            "Qwen3-Omni native FP8 checkpoints cannot use "
            "moe_runner_backend='flashinfer_cutlass'. Leave the backend as "
            "'auto' so Omni selects a native-FP8-compatible MoE runner."
        )

    fp8_gemm_backend = _normalize_quantization(server_args.fp8_gemm_runner_backend)
    if (
        model_arch_override == "Qwen3OmniTalker"
        and effective_quantization == "fp8"
        and has_native_fp8_block_quant
        and fp8_gemm_backend in (None, "auto")
    ):
        # Projected talker prefill has request-dependent FP8 dense GEMM shapes
        # outside decode CUDA graph replay; DeepGEMM can otherwise JIT there.
        server_args.fp8_gemm_runner_backend = "triton"
        fp8_gemm_backend = server_args.fp8_gemm_runner_backend

    server_quantization = server_args.quantization
    logger.info(
        f"Configured SGLang backend policy: arch={model_arch_override} "
        f"effective_quantization={effective_quantization} "
        f"server_quantization={server_quantization} "
        f"moe_runner_backend={moe_runner_backend} "
        f"fp8_gemm_backend={fp8_gemm_backend}"
    )
    return effective_quantization


def _normalize_quantization(value: object) -> str | None:
    if value is None:
        return None
    return str(value).lower()


def _model_config_has_moe(model_config: ModelConfig) -> bool:
    config_to_check = getattr(model_config, "hf_text_config", None)
    if config_to_check is None:
        hf_config = getattr(model_config, "hf_config", None)
        config_to_check = getattr(hf_config, "text_config", hf_config)
    return hasattr(config_to_check, "num_experts_per_tok")


def _model_config_has_native_fp8_block_quant(model_config: ModelConfig) -> bool:
    quant_dict = resolve_quant_config(getattr(model_config, "hf_config", None))
    if quant_dict is None:
        return False
    return (
        _normalize_quantization(quant_dict.get("quant_method")) == "fp8"
        and quant_dict.get("weight_block_size") is not None
    )


def _is_h20_device() -> bool:
    """True only on NVIDIA H20 (word-boundary match so "H200" isn't caught)."""
    try:
        import re

        import torch

        if not torch.cuda.is_available():
            return False
        return bool(re.search(r"\bH20\b", torch.cuda.get_device_name(0)))
    except Exception:
        return False


def _is_fp8_cutlass_moe_supported() -> bool:
    """Mirror pinned SGLang 0.5.12.post1 FP8 CUTLASS MoE assertions."""
    try:
        from sglang.srt.layers.quantization.fp8_utils import cutlass_fp8_supported
        from sglang.srt.utils import is_sm90_supported, is_sm100_supported
    except ImportError:
        return False

    return bool(
        cutlass_fp8_supported() and (is_sm90_supported() or is_sm100_supported())
    )


def _apply_omni_quantization_adapters(model_config: ModelConfig) -> None:
    """Apply Omni-specific quantization adapters before SGLang builds its config.

    SGLang owns detection, config parsing, layer construction, and post-load
    hooks. The only Omni-specific step needed here is stage-local checkpoint
    name normalization for methods whose per-block quant names are matched
    against runtime module names, currently AutoRound.
    """
    quant_dict = resolve_quant_config(getattr(model_config, "hf_config", None))
    if quant_dict is None:
        return

    if needs_quant_config_normalization(quant_dict):
        normalize_quant_config(model_config)


def _initialize_model_worker_backend_globals(
    server_args: ServerArgs,
    model_config: ModelConfig,
    effective_quantization: str | None,
) -> None:
    """Initialize backend globals needed by direct workers before model loading."""

    if _model_config_has_moe(model_config):
        from sglang.srt.layers.moe import initialize_moe_config

        initialize_moe_config(server_args)

    if effective_quantization == "fp8":
        from sglang.srt.layers.quantization.fp8_utils import initialize_fp8_gemm_config

        initialize_fp8_gemm_config(server_args)
