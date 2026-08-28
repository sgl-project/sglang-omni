# SPDX-License-Identifier: Apache-2.0
"""Torch MPS runner for Qwen3-ASR."""

from __future__ import annotations

import gc
import json
import logging
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from transformers import Qwen3Config, Qwen3ForCausalLM

from sglang_omni.model_runner.base import ModelRunner

logger = logging.getLogger(__name__)

_MROPE_ONLY_KEYS = frozenset({"interleaved", "mrope_interleaved", "mrope_section"})


def _resolve_model_path(model_path: str) -> Path:
    path = Path(model_path).expanduser()
    if path.is_dir():
        return path.resolve()

    from huggingface_hub import snapshot_download

    return Path(snapshot_download(model_path))


def _text_config(model_path: Path) -> Qwen3Config:
    root_config = json.loads((model_path / "config.json").read_text())
    text_config = dict(root_config["thinker_config"]["text_config"])
    for field in ("rope_parameters", "rope_scaling"):
        rope_config = text_config.get(field)
        if isinstance(rope_config, dict):
            text_config[field] = {
                key: value
                for key, value in rope_config.items()
                if key not in _MROPE_ONLY_KEYS
            }
    return Qwen3Config(**text_config)


def _load_language_weights(
    language_model: Qwen3ForCausalLM,
    model_path: Path,
) -> None:
    expected = set(language_model.state_dict())
    state_dict = {}
    weight_files = sorted(model_path.glob("*.safetensors"))
    if not weight_files:
        raise ValueError(f"Qwen3-ASR checkpoint has no safetensors in {model_path}")
    for weight_file in weight_files:
        with safe_open(weight_file, framework="pt", device="cpu") as f:
            for checkpoint_name in f.keys():
                if checkpoint_name.startswith(
                    "thinker.model."
                ) or checkpoint_name.startswith("thinker.lm_head."):
                    model_name = checkpoint_name.removeprefix("thinker.")
                    if model_name in expected:
                        state_dict[model_name] = f.get_tensor(checkpoint_name)

    missing = expected - set(state_dict)
    if missing:
        raise ValueError(
            "Qwen3-ASR Torch MPS language checkpoint is incomplete: "
            f"{sorted(missing)[:10]}"
        )
    language_model.load_state_dict(state_dict, strict=True, assign=True)

    rotary = language_model.model.rotary_emb
    rope_config = language_model.config.rope_parameters
    if not isinstance(rope_config, dict):
        rope_config = language_model.config.rope_scaling
    rope_theta = float(rope_config["rope_theta"])
    head_dim = int(language_model.config.head_dim)
    inv_freq = 1.0 / (
        rope_theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
    )
    rotary.inv_freq = inv_freq
    rotary.original_inv_freq = inv_freq.clone()


def install_torch_mps_language_model(model: Any, model_path: str) -> None:
    """Replace SGLang's Torch-native LM with the pinned HF Torch implementation."""
    checkpoint = _resolve_model_path(model_path)
    old_parameter = next(model.language_model.parameters())
    device = old_parameter.device
    dtype = old_parameter.dtype

    del model.language_model
    gc.collect()
    torch.mps.empty_cache()

    with torch.device("meta"):
        language_model = Qwen3ForCausalLM(_text_config(checkpoint))
    _load_language_weights(language_model, checkpoint)
    model.language_model = language_model.eval().to(device=device, dtype=dtype)
    logger.info("Installed Qwen3-ASR Hugging Face Torch LM on %s (%s)", device, dtype)


class Qwen3ASRTorchMpsModelRunner(ModelRunner):
    """Run one Qwen3-ASR request through Torch audio and Hugging Face Qwen3."""

    def __init__(self, tp_worker: Any, output_processor: Any):
        super().__init__(tp_worker, output_processor)
        self._past_key_values: dict[str, Any] = {}

    def lookahead_eligible(self, batch: Any) -> bool:
        del batch
        return False

    @staticmethod
    def _one_request(requests: list[Any]) -> Any:
        if len(requests) != 1:
            raise RuntimeError(
                "Qwen3-ASR Torch MPS currently requires max_running_requests=1"
            )
        return requests[0]

    def _next_token_result(self, next_token_ids: torch.Tensor) -> Any:
        from sglang.srt.managers.scheduler import GenerationBatchResult

        return GenerationBatchResult(
            logits_output=None,
            next_token_ids=next_token_ids,
            can_run_cuda_graph=False,
        )

    @torch.inference_mode()
    def custom_prefill_forward(
        self,
        forward_batch: Any,
        schedule_batch: Any,
        requests: list[Any],
    ) -> Any:
        del forward_batch
        scheduler_request = self._one_request(requests)
        req = scheduler_request.data.req
        mm_inputs = req.multimodal_inputs
        if mm_inputs is None or len(mm_inputs.mm_items) != 1:
            raise ValueError("Qwen3-ASR Torch MPS requires exactly one audio item")
        item = mm_inputs.mm_items[0]
        if item.feature is None or item.pad_value is None:
            raise ValueError(
                "Qwen3-ASR Torch MPS requires audio features and pad value"
            )
        if mm_inputs.audio_token_id is None:
            raise ValueError("Qwen3-ASR Torch MPS is missing its audio token ID")

        token_ids = [int(token_id) for token_id in schedule_batch.input_ids.tolist()]
        pad_value = int(item.pad_value)
        audio_token_id = int(mm_inputs.audio_token_id)
        normalized_ids = [
            audio_token_id if token_id == pad_value else token_id
            for token_id in token_ids
        ]
        audio_positions = [
            index
            for index, token_id in enumerate(normalized_ids)
            if token_id == audio_token_id
        ]
        if not audio_positions:
            raise ValueError("Qwen3-ASR Torch MPS prefill has no audio placeholders")
        audio_start = audio_positions[0]
        if audio_positions != list(
            range(audio_start, audio_start + len(audio_positions))
        ):
            raise ValueError(
                "Qwen3-ASR Torch MPS audio placeholders must be contiguous"
            )

        language_model = self.model.language_model
        input_ids = torch.tensor(
            [normalized_ids],
            dtype=torch.long,
            device=self.device,
        )
        input_embeddings = language_model.model.embed_tokens(input_ids)
        audio_features = self.model.get_audio_feature([item]).to(
            device=self.device,
            dtype=input_embeddings.dtype,
        )
        if audio_features.shape != (
            1,
            len(audio_positions),
            input_embeddings.shape[-1],
        ):
            raise ValueError(
                "Qwen3-ASR Torch MPS audio embedding shape does not match its "
                f"placeholder span: {tuple(audio_features.shape)}"
            )
        input_embeddings[0, audio_start : audio_start + len(audio_positions), :] = (
            audio_features[0]
        )

        output = language_model(
            inputs_embeds=input_embeddings,
            use_cache=True,
            logits_to_keep=1,
        )
        self._past_key_values[scheduler_request.request_id] = output.past_key_values
        return self._next_token_result(output.logits[:, -1, :].argmax(dim=-1))

    @torch.inference_mode()
    def custom_decode_forward(
        self,
        forward_batch: Any,
        schedule_batch: Any,
        requests: list[Any],
    ) -> Any:
        del forward_batch
        scheduler_request = self._one_request(requests)
        request_id = scheduler_request.request_id
        try:
            past_key_values = self._past_key_values[request_id]
        except KeyError as exc:
            raise RuntimeError(
                f"Qwen3-ASR Torch MPS decode has no cache for {request_id}"
            ) from exc

        input_ids = schedule_batch.input_ids.reshape(1, 1).to(
            device=self.device,
            dtype=torch.long,
        )
        output = self.model.language_model(
            input_ids=input_ids,
            past_key_values=past_key_values,
            use_cache=True,
            logits_to_keep=1,
        )
        self._past_key_values[request_id] = output.past_key_values
        return self._next_token_result(output.logits[:, -1, :].argmax(dim=-1))

    def on_request_finished(self, request_id: str, req_data: Any) -> None:
        del req_data
        self._past_key_values.pop(request_id, None)

    def abort_request(self, request_id: str) -> None:
        self._past_key_values.pop(request_id, None)


__all__ = [
    "Qwen3ASRTorchMpsModelRunner",
    "install_torch_mps_language_model",
]
