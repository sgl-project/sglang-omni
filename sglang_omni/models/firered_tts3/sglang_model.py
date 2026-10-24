# SPDX-License-Identifier: Apache-2.0
"""SGLang Qwen3 backbone with the FireRedTTS3 float32 flow side path."""

from __future__ import annotations

from collections.abc import Iterable

import torch
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.models.qwen3 import Qwen3ForCausalLM
from torch import nn
from transformers import Qwen3Config

from sglang_omni.models.firered_tts3.dit import DiT
from sglang_omni.models.firered_tts3.hf_config import FireRedTTS3Config
from sglang_omni.models.firered_tts3.modules import PatchEncoder
from sglang_omni.models.firered_tts3.weights import (
    build_dit,
    build_patch_encoder,
    route_root_key,
    strip_backbone_prefix,
)
from sglang_omni.models.weight_loader import default_weight_loader
from sglang_omni.utils.json import JsonValue

BACKBONE_CONFIG_FIELDS = (
    "vocab_size",
    "hidden_size",
    "intermediate_size",
    "num_hidden_layers",
    "num_attention_heads",
    "num_key_value_heads",
    "head_dim",
    "hidden_act",
    "max_position_embeddings",
    "rms_norm_eps",
    "rope_theta",
    "tie_word_embeddings",
    "attention_bias",
    "max_window_layers",
    "use_sliding_window",
    "sliding_window",
    "bos_token_id",
    "eos_token_id",
)


def backbone_qwen3_config(config: FireRedTTS3Config) -> Qwen3Config:
    return Qwen3Config(
        **{name: getattr(config, name) for name in BACKBONE_CONFIG_FIELDS}
    )


class FireRedTTS3SGLangModel(nn.Module):
    """Compose the SGLang Qwen3 backbone with the FireRedTTS3 side path."""

    def __init__(
        self,
        config: FireRedTTS3Config,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        if not isinstance(config, FireRedTTS3Config):
            raise ValueError(
                f"FireRedTTS3 requires the registered FireRedTTS3Config; got {type(config).__name__}"
            )
        else:
            pass
        self.config: FireRedTTS3Config = config
        self.backbone: Qwen3ForCausalLM = Qwen3ForCausalLM(
            backbone_qwen3_config(config),
            quant_config=quant_config,
            prefix=f"{prefix}.backbone" if prefix else "backbone",
        )
        # note (kermit): Cast side-path parameters before checkpoint copies can round their values.
        self.patch_encoder: PatchEncoder = build_patch_encoder(config).to(
            dtype=torch.float32
        )
        self.dit: DiT = build_dit(config).to(dtype=torch.float32)
        self.dit_head: nn.Linear = nn.Linear(config.hidden_size, config.dit_hidden_size)
        self.stop_head: nn.Linear = nn.Linear(config.hidden_size, 1)
        self.spk_proj_llm: nn.Linear | None = (
            nn.Linear(config.spk_in_dim, config.hidden_size)
            if config.spk_in_dim is not None
            else None
        )
        self.spk_proj_dit: nn.Linear | None = (
            nn.Linear(config.spk_in_dim, config.spk_in_dim)
            if config.spk_in_dim is not None
            else None
        )
        for module in (
            self.dit_head,
            self.stop_head,
            self.spk_proj_llm,
            self.spk_proj_dit,
        ):
            if module is not None:
                module.to(dtype=torch.float32)
            else:
                pass
        self.llm_hidden_size: int = int(config.hidden_size)
        self.redae_dim: int = int(config.redae_dim)
        self.patch_size: int = int(config.patch_size)
        self.history_patches: int = int(config.num_history_patches)
        self.history_length: int = self.history_patches * self.patch_size
        self.weights_locked: bool = False
        self.text_logits_requested: bool = False
        self.graph_feedback_buffer: torch.Tensor | None = None

    def get_input_embeddings(self) -> nn.Module:
        return self.backbone.get_input_embeddings()

    def enable_graph_feedback(self, max_batch_size: int) -> None:
        """Route decode feedback embeddings through a persistent buffer."""
        if max_batch_size <= 0:
            raise ValueError("FireRedTTS3 graph feedback buffer needs a positive size")
        else:
            pass
        parameter = next(self.backbone.parameters())
        self.graph_feedback_buffer = torch.zeros(
            (int(max_batch_size), self.llm_hidden_size),
            device=parameter.device,
            dtype=parameter.dtype,
        )

    def warmup_side_path(self) -> None:
        """Warm the fp32 side-path modules with their hot shapes."""
        device = next(self.backbone.parameters()).device
        with torch.inference_mode():
            prompt = torch.zeros(1, 4 * self.patch_size, self.redae_dim, device=device)
            self.patch_encoder(prompt)
            feedback = torch.zeros(1, self.patch_size, self.redae_dim, device=device)
            self.patch_encoder(feedback)
            hidden = torch.zeros(
                1, self.history_patches + 1, self.llm_hidden_size, device=device
            )
            self.dit_head(hidden)
            self.stop_head(hidden[:, -1:])
            if self.spk_proj_llm is not None and self.spk_proj_dit is not None:
                spk = torch.zeros(1, int(self.config.spk_in_dim), device=device)
                self.spk_proj_llm(spk)
                self.spk_proj_dit(spk)
            else:
                pass
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        else:
            pass

    @property
    def weight_update_blocked_reason(self) -> str | None:
        if self.weights_locked:
            return "FireRedTTS3 weight updates are disabled after Flow execution is initialized; restart the worker with new weights"
        else:
            return None

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        reason = self.weight_update_blocked_reason
        if reason is not None:
            raise RuntimeError(reason)
        else:
            pass
        backbone_weights: list[tuple[str, torch.Tensor]] = []
        side_params = dict(self.side_path_parameters())
        loaded: set[str] = set()
        for name, tensor in weights:
            bucket = route_root_key(name)
            if bucket == "backbone":
                backbone_weights.append((strip_backbone_prefix(name), tensor))
                continue
            else:
                pass
            if bucket == "lm_head":
                continue
            else:
                pass
            if bucket == "unknown":
                raise ValueError(f"unexpected FireRedTTS3 checkpoint key {name!r}")
            else:
                pass
            parameter = side_params.get(name)
            assert (
                parameter is not None
            ), f"FireRedTTS3 checkpoint key {name!r} routed to {bucket} but no such side-path parameter exists"
            loader = getattr(parameter, "weight_loader", default_weight_loader)
            loader(parameter, tensor)
            loaded.add(name)
        self.backbone.load_weights(backbone_weights)
        device = next(self.backbone.parameters()).device
        for module in self.side_path_modules():
            module.to(device=device, dtype=torch.float32)
        return loaded

    def side_path_modules(self) -> list[nn.Module]:
        modules: list[nn.Module] = [
            self.patch_encoder,
            self.dit,
            self.dit_head,
            self.stop_head,
        ]
        if self.spk_proj_llm is not None:
            modules.append(self.spk_proj_llm)
        else:
            pass
        if self.spk_proj_dit is not None:
            modules.append(self.spk_proj_dit)
        else:
            pass
        return modules

    def side_path_parameters(self) -> Iterable[tuple[str, nn.Parameter]]:
        for module, prefix in (
            (self.patch_encoder, "patch_encoder"),
            (self.dit, "dit"),
            (self.dit_head, "dit_head"),
            (self.stop_head, "stop_head"),
            (self.spk_proj_llm, "spk_proj_llm"),
            (self.spk_proj_dit, "spk_proj_dit"),
        ):
            if module is None:
                continue
            else:
                pass
            for name, parameter in module.named_parameters():
                yield (f"{prefix}.{name}", parameter)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor | None = None,
        **kwargs: JsonValue | torch.Tensor,
    ) -> LogitsProcessorOutput:
        """Accept SGLang's optional forward arguments and return backbone hidden states."""
        # note (kermit): Decode graphs read the same fixed buffer that the runner fills before replay.
        if (
            self.graph_feedback_buffer is not None
            and forward_batch.forward_mode.is_decode()
        ):
            input_embeds = self.graph_feedback_buffer[: input_ids.shape[0]]
        elif input_embeds is None:
            input_embeds = forward_batch.input_embeds
        else:
            pass
        hidden_states = self.backbone.model(
            input_ids=input_ids,
            positions=positions,
            forward_batch=forward_batch,
            input_embeds=input_embeds,
        )
        # note (kermit): Captured decode keeps real logits available for later text sampling.
        if self.config.use_lm_head and (
            self.text_logits_requested
            or (
                self.graph_feedback_buffer is not None
                and forward_batch.forward_mode.is_decode()
            )
        ):
            return self.backbone.logits_processor(
                input_ids, hidden_states, self.backbone.lm_head, forward_batch
            )
        else:
            pass
        if forward_batch.forward_mode.is_extend():
            extend_seq_lens = forward_batch.extend_seq_lens
            request_count = (
                int(extend_seq_lens.numel()) if extend_seq_lens is not None else 1
            )
        else:
            request_count = int(hidden_states.shape[0])
        # note (kermit): Audio-only steps consume hidden states and replace the sampled control token.
        return LogitsProcessorOutput(
            next_token_logits=hidden_states.new_empty((request_count, 1)),
            hidden_states=hidden_states,
        )


EntryClass = FireRedTTS3SGLangModel
__all__ = ["EntryClass", "FireRedTTS3SGLangModel", "backbone_qwen3_config"]
