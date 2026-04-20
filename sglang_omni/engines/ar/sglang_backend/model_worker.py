from __future__ import annotations

import os
import socket
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sglang.srt.configs.model_config import ModelConfig
    from sglang.srt.server_args import ServerArgs


@dataclass
class ModelWorkerConfig:
    model_arch_override: str | None = None
    weight_prefix: str | None = None
    nccl_port: int | None = None


# Architecture -> (sub_config_attr, text_config_attr) for composite models.
# When text_config_attr is None, the sub_config IS the text config itself.
_ARCH_CONFIG_MAP: dict[str, tuple[str, str | None]] = {
    "Qwen3OmniTalker": ("talker_config", "text_config"),
    "BailingMoeV2ForCausalLM": ("llm_config", None),
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

        self.gpu_id = gpu_id
        self.tp_rank = tp_rank
        self._init_model_config()
        self._init_model_runner()

        self.device = self.model_runner.device
        from sglang.srt.utils import broadcast_pyobj, set_random_seed

        self.random_seed = broadcast_pyobj(
            [server_args.random_seed],
            self.tp_rank,
            self.model_runner.tp_group.cpu_group,
        )[0]
        set_random_seed(self.random_seed)

    @property
    def tp_cpu_group(self):
        """NCCL CPU process group for TP broadcast operations."""
        return self.model_runner.tp_group.cpu_group

    @property
    def tp_size(self) -> int:
        return self.server_args.tp_size

    def _init_model_config(self):
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

        entry = _ARCH_CONFIG_MAP.get(arch)
        if entry is None:
            return
        sub_config_attr, text_config_attr = entry
        sub_cfg = getattr(model_config.hf_config, sub_config_attr, None)
        if sub_cfg is None:
            return
        # When text_config_attr is None, the sub_config IS the text config
        if text_config_attr is not None:
            text_cfg = getattr(sub_cfg, text_config_attr, None)
        else:
            text_cfg = sub_cfg
        if text_cfg is None:
            return
        model_config.hf_text_config = text_cfg
        model_config.num_attention_heads = text_cfg.num_attention_heads
        model_config.num_key_value_heads = text_cfg.num_key_value_heads
        model_config.hidden_size = text_cfg.hidden_size
        model_config.num_hidden_layers = text_cfg.num_hidden_layers

    def get_memory_pool(self):
        return (
            self.model_runner.req_to_token_pool,
            self.model_runner.token_to_kv_pool_allocator,
        )

    def _init_model_runner(self):
        from .model_runner import SGLModelRunner

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
        )

    def forward_batch_generation(
        self,
        forward_batch,
    ):
        from sglang.srt.managers.scheduler import GenerationBatchResult

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

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("", 0))
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        port = sock.getsockname()[1]

    os.environ["MASTER_PORT"] = str(port)
    return port
