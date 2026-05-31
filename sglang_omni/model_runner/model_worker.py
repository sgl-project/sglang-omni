from __future__ import annotations

import logging
import os
import socket
from dataclasses import dataclass
from typing import TYPE_CHECKING

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
    "Qwen3TTSTalker": ("talker_config", None),
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
    server_quantization = _normalize_quantization(
        getattr(server_args, "quantization", None)
    )
    if server_quantization is not None:
        effective_quantization = server_quantization

    moe_runner_backend = getattr(server_args, "moe_runner_backend", "auto")
    is_qwen3_omni_arch = model_arch_override in (
        "Qwen3OmniTalker",
        "Qwen3OmniThinkerForCausalLM",
    )
    if is_qwen3_omni_arch and getattr(server_args, "ep_size", 1) != 1:
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
        server_args.moe_runner_backend = "flashinfer_cutlass"
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

    server_quantization = getattr(server_args, "quantization", None)
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
    quant_config = _get_hf_quantization_config(model_config)
    if quant_config is None:
        return False
    quant_method = _get_config_value(quant_config, "quant_method")
    weight_block_size = _get_config_value(quant_config, "weight_block_size")
    return (
        _normalize_quantization(quant_method) == "fp8" and weight_block_size is not None
    )


def _get_hf_quantization_config(model_config: ModelConfig) -> object | None:
    hf_config = getattr(model_config, "hf_config", None)
    quant_config = getattr(hf_config, "quantization_config", None)
    if quant_config is not None:
        return quant_config

    hf_text_config = getattr(model_config, "hf_text_config", None)
    return getattr(hf_text_config, "quantization_config", None)


def _get_config_value(config: object, key: str) -> object | None:
    if isinstance(config, dict):
        return config.get(key)
    return getattr(config, key, None)


def _is_fp8_cutlass_moe_supported() -> bool:
    """Mirror pinned SGLang 0.5.8 FP8 CUTLASS MoE assertions."""
    try:
        from sglang.srt.layers.quantization.fp8_utils import cutlass_fp8_supported
        from sglang.srt.utils import is_sm90_supported, is_sm100_supported
    except ImportError:
        return False

    return bool(
        cutlass_fp8_supported() and (is_sm90_supported() or is_sm100_supported())
    )


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
