# SPDX-License-Identifier: Apache-2.0
"""MiniMax Music 3 SGLang engine builder — OmniScheduler integration."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from sglang_omni.scheduling.engine_factory import TtsEngineBuilder
from sglang_omni.scheduling.generation_batch_policy import build_default_cuda_graph_bs

logger = logging.getLogger(__name__)

_AUDIO_WEIGHT_PREFIXES = ("model.audio_decoder.", "model.audio_extra_embedding.")


def _rvq_graph_buckets(max_running_requests: int) -> list[int]:
    """Batch buckets to capture the RVQ depth pass at."""
    buckets, size = [], 2
    while size < 2 * max_running_requests:
        buckets.append(size)
        size *= 2
    buckets.append(2 * max_running_requests)
    return buckets


class MiniMaxMusic3EngineBuilder(TtsEngineBuilder):
    model_name = "minimax_music3"
    context_length = 10240
    model_arch_override = "Qwen3ForCausalLM"

    def __init__(self, *, max_running_requests: int = 16) -> None:
        self.max_running_requests = int(max_running_requests)
        if self.max_running_requests <= 0:
            raise ValueError("MiniMax Music 3 max_running_requests must be positive")
        self._model_runner: Any | None = None
        self._checkpoint_root: str | None = None

    def resolve_checkpoint(self, model_path: str) -> str:
        from .checkpoint import resolve_checkpoint

        paths = resolve_checkpoint(model_path)
        self._checkpoint_root = str(paths.root)
        return str(paths.qwen_dir)

    def pre_infra_setup(self, checkpoint_dir: str) -> None:
        self._normalize_backbone_config(Path(checkpoint_dir) / "config.json")
        self._filter_audio_weights()

    def generation_defaults(self, *, dtype: str) -> dict[str, Any]:
        return {
            "disable_cuda_graph": False,
            "disable_overlap_schedule": True,
            "disable_radix_cache": True,
            "enable_torch_compile": False,
            "max_running_requests": self.max_running_requests,
            "chunked_prefill_size": 0,
            "mem_fraction_static": 0.50,
            "dtype": dtype,
            "trust_remote_code": False,
        }

    def adjust_overrides(self, overrides: dict[str, Any]) -> None:
        if int(overrides.get("tp_size", 1)) != 1:
            raise ValueError("MiniMax Music 3 does not support TP")
        requested = int(
            overrides.get("max_running_requests", self.max_running_requests)
        )
        if requested <= 0:
            raise ValueError("MiniMax Music 3 max_running_requests must be positive")
        self.max_running_requests = requested
        rows = 2 * requested
        overrides["max_running_requests"] = rows
        overrides["cuda_graph_max_bs"] = rows
        overrides["cuda_graph_bs"] = build_default_cuda_graph_bs(rows)
        overrides["disable_radix_cache"] = True
        overrides["chunked_prefill_size"] = 0
        if not bool(overrides.get("disable_cuda_graph", False)):
            overrides["enable_return_hidden_states"] = True

    def setup_model(
        self,
        *,
        model_worker: Any,
        checkpoint_dir: str,
        device: str,
        gpu_id: int,
        server_args: Any,
    ) -> None:
        del checkpoint_dir, device, gpu_id
        from sglang_omni.scheduling.generation_batch_policy import (
            get_decode_cuda_graph_max_bs,
        )

        from .sglang_model import attach_minimax_modules, enable_graph_feedback

        assert self._checkpoint_root is not None
        model = model_worker.model_runner.model
        attach_minimax_modules(model, self._checkpoint_root)
        if not bool(server_args.disable_cuda_graph):
            enable_graph_feedback(
                model,
                max(
                    int(server_args.max_running_requests),
                    int(get_decode_cuda_graph_max_bs(server_args) or 0),
                ),
            )
        model.eval()

    def setup_model_resources(
        self, model: Any, server_args: Any, *, generation_cuda_graph_enabled: bool
    ) -> None:
        del generation_cuda_graph_enabled
        from .sglang_model import enable_rvq_depth_cuda_graph

        del server_args
        enable_rvq_depth_cuda_graph(
            model, _rvq_graph_buckets(self.max_running_requests)
        )

    def make_scheduler(self, **kwargs: Any) -> Any:
        from .scheduler import MiniMaxMusic3Scheduler

        return MiniMaxMusic3Scheduler(
            tp_worker=kwargs.pop("model_worker"),
            abort_callback=self.make_abort_callback(),
            request_finished_callback=self.make_request_finished_callback(),
            **kwargs,
            **self.extra_scheduler_kwargs(),
        )

    def make_model_runner(self, model_worker: Any, output_proc: Any) -> Any:
        from .model_runner import MiniMaxMusic3ModelRunner

        self._model_runner = MiniMaxMusic3ModelRunner(model_worker, output_proc)
        return self._model_runner

    def make_adapters(self, model: Any) -> tuple[Any, Any]:
        del model
        from transformers import AutoTokenizer

        from .checkpoint import resolve_checkpoint
        from .prompt import validate_tokenizer_ids
        from .sglang_request_builder import (
            apply_minimax_result,
            build_sglang_minimax_request,
        )

        assert self._checkpoint_root is not None
        paths = resolve_checkpoint(self._checkpoint_root)
        tokenizer = AutoTokenizer.from_pretrained(
            str(paths.tokenizer_dir), trust_remote_code=False
        )
        validate_tokenizer_ids(tokenizer)

        def build_request(payload: Any) -> Any:
            return build_sglang_minimax_request(payload, tokenizer)

        return build_request, apply_minimax_result

    def make_abort_callback(self) -> Any | None:
        assert self._model_runner is not None
        return self._model_runner.reset_request

    def extra_scheduler_kwargs(self) -> dict[str, Any]:
        from .sglang_request_builder import build_stream_output

        return {
            "stream_output_builder": build_stream_output,
            "enable_async_decode": False,
        }

    @staticmethod
    def _normalize_backbone_config(config_path: Path) -> None:
        """Rewrite the backbone config so HuggingFace resolves a Qwen3 config."""
        config = json.loads(config_path.read_text())
        if config.get("model_type") == "qwen3":
            return
        backup_path = config_path.with_suffix(".json.bak")
        if not backup_path.exists():
            backup_path.write_text(config_path.read_text())
        config["model_type"] = "qwen3"
        config_path.unlink()
        config_path.write_text(json.dumps(config, indent=2))
        logger.info(
            f"MiniMax Music 3: rewrote {config_path} model_type to qwen3 (backup at {backup_path})"
        )

    @staticmethod
    def _filter_audio_weights() -> None:
        from sglang.srt.models.qwen3 import Qwen3ForCausalLM

        if getattr(Qwen3ForCausalLM.load_weights, "_minimax_filtered", False):
            return
        load_weights = Qwen3ForCausalLM.load_weights

        def filtered_load_weights(self, weights):
            return load_weights(
                self,
                (
                    (name, tensor)
                    for name, tensor in weights
                    if not name.startswith(_AUDIO_WEIGHT_PREFIXES)
                ),
            )

        filtered_load_weights._minimax_filtered = True
        Qwen3ForCausalLM.load_weights = filtered_load_weights
        logger.info("MiniMax Music 3: Qwen3 weight loader now skips audio-module keys")


__all__ = ["MiniMaxMusic3EngineBuilder"]
