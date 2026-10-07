# SPDX-License-Identifier: Apache-2.0
"""Fun-CosyVoice3 pipeline state."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import Field, dataclass

import torch
from numpy.typing import ArrayLike

from sglang_omni.scheduling.pipeline_state import DeclarativeStateBase, WireSpec, wire
from sglang_omni.scheduling.typed_tensor import encode_typed_tensor


@dataclass
class FunCosyVoice3State(DeclarativeStateBase):
    """Per-request state for Fun-CosyVoice3 generation."""

    sample_rate: int = wire(24000, codec="int")

    text: str = wire("", codec="str")
    language: str = wire("auto", codec="str_or")
    instructions: str | None = None
    ref_audio: object = None
    ref_text: str | None = None
    stream: bool = wire(False, codec="bool")
    speed: float = wire(1.0, codec="float")
    seed: int | None = None
    generation_kwargs: Mapping[str, object] = wire(default_factory=dict, codec="dict")
    flow_embedding: ArrayLike | torch.Tensor | None = wire(None, codec="tensor_list")
    flow_prompt_speech_token: ArrayLike | torch.Tensor | None = wire(
        None, codec="tensor_list"
    )
    flow_prompt_speech_feat: ArrayLike | torch.Tensor | None = wire(
        None, codec="typed_tensor"
    )
    audio_codes: ArrayLike | torch.Tensor | None = wire(None, codec="tensor_list")
    audio_samples: object = wire(None, codec="tensor_list")

    def encode_field(
        self,
        data: dict[str, object],
        f: Field[object],
        spec: WireSpec,
        emit: str,
    ) -> None:
        if f.name == "flow_prompt_speech_feat":
            prompt_features = self.flow_prompt_speech_feat
            if (
                isinstance(prompt_features, torch.Tensor)
                and prompt_features.dtype == torch.float32
                and prompt_features.device.type == "cpu"
                and prompt_features.numel() > 0
                and prompt_features.ndim > 0
                and torch.get_default_dtype() == torch.float32
                and torch.get_default_device().type == "cpu"
                and bool(torch.isfinite(prompt_features).all())
            ):
                # note (Codex): Each message owns its reference storage on local routes.
                data[f.name] = prompt_features.detach().clone()
                return
            else:
                spec = WireSpec(emit=spec.emit, codec="tensor_list")
        else:
            pass
        super().encode_field(data, f, spec, emit)

    def to_terminal_dict(self) -> dict[str, object]:
        completion_fields = self.to_dict()
        prompt_features = completion_fields.get("flow_prompt_speech_feat")
        if isinstance(prompt_features, torch.Tensor):
            # note (Codex): Terminal messages use MessagePack without tensor relay.
            completion_fields.pop("flow_prompt_speech_feat")
            completion_fields.update(
                encode_typed_tensor(prompt_features, key="flow_prompt_speech_feat")
            )
        else:
            pass
        return completion_fields

    @classmethod
    def from_dict(cls: type[FunCosyVoice3State], data: object) -> FunCosyVoice3State:
        if isinstance(data, dict) and "flow_prompt_speech_feat" in data:
            legacy_features = data["flow_prompt_speech_feat"]
            remaining_fields = dict(data)
            remaining_fields.pop("flow_prompt_speech_feat")
            for suffix in ("bytes", "shape", "dtype"):
                remaining_fields.pop(f"flow_prompt_speech_feat_{suffix}", None)
            state = super().from_dict(remaining_fields)
            state.flow_prompt_speech_feat = legacy_features
            return state
        else:
            return super().from_dict(data)
