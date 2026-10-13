"""Reuse the shared talker prompt layout with MLX projections and embeddings."""

import mlx.core as mx
import mlx.nn as nn
import torch
from mlx_vlm.models.qwen3_omni_moe.talker import Talker

from sglang_omni.models.qwen3_omni.components.talker_prefill import (
    TalkerPrefillBuilder,
    TalkerPromptPrefill,
)
from sglang_omni.models.qwen3_omni.mlx.loading import load_component
from sglang_omni.models.qwen3_omni.mlx.tensors import mlx_to_torch, torch_to_mlx
from sglang_omni.models.qwen3_omni.payload_types import Qwen3OmniPipelineState
from sglang_omni.pipeline.stage.stream_queue import StreamItem
from sglang_omni.proto import StagePayload


class TorchProjection:
    def __init__(self, module: nn.Module, *, embedding: bool = False) -> None:
        self.module = module
        self.embedding = embedding

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        with mx.stream(mx.new_thread_local_stream(mx.gpu)):
            values = torch_to_mlx(tensor)
            if self.embedding:
                values = values.astype(mx.int32)
            else:
                pass
            return mlx_to_torch(self.module(values))


class MlxTalkerProjectionAdapter:
    def __init__(self, talker: Talker) -> None:
        self.config = talker.config
        self.activation_dtype = torch.float32
        self.text_projection = TorchProjection(talker.text_projection)
        self.hidden_projection = TorchProjection(talker.hidden_projection)
        self.codec_embedding = TorchProjection(
            talker.model.codec_embedding, embedding=True
        )

    def get_input_embeddings(self) -> TorchProjection:
        return self.codec_embedding


class MlxTalkerPrefillBuilder(TalkerPrefillBuilder):
    mlx_embedding: TorchProjection | None = None

    def build_prompt_prefill(
        self,
        payload: StagePayload,
        thinker_chunks: list[StreamItem],
        *,
        thinker_done: bool,
    ) -> TalkerPromptPrefill:
        state = Qwen3OmniPipelineState.from_dict(payload.data)
        prompt_ids = state.prompt["input_ids"].reshape(-1)
        if self.build_multimodal_mask(prompt_ids).any() and (
            not thinker_chunks
            or "prompt_hidden_layer" not in thinker_chunks[0].metadata
        ):
            raise ValueError("MLX media prompts require thinker hidden-state captures")
        else:
            pass
        speaker = str(payload.request.params.get("speaker", "Ethan")).lower()
        if speaker not in self.speaker_map:
            raise ValueError(f"Unknown Qwen3-Omni speaker {speaker!r}")
        else:
            pass
        return super().build_prompt_prefill(
            payload, thinker_chunks, thinker_done=thinker_done
        )

    def load_prompt_token_embeddings(self, token_ids: torch.Tensor) -> torch.Tensor:
        if self.mlx_embedding is None:
            self.mlx_embedding = TorchProjection(
                load_component(self.model_path, "embedding"), embedding=True
            )
        else:
            pass
        return self.mlx_embedding(token_ids)

    def prompt_hidden_states(
        self, prompt_embed: torch.Tensor, thinker_chunks: list[StreamItem]
    ) -> torch.Tensor:
        first = thinker_chunks[0]
        if "prompt_hidden_layer" not in first.metadata:
            return torch.zeros_like(prompt_embed)
        else:
            pass
        if (
            first.metadata["prompt_hidden_layer"]
            != self.model.config.accept_hidden_layer
        ):
            raise ValueError("MLX thinker capture layer does not match talker")
        else:
            pass
        if (
            not isinstance(first.data, torch.Tensor)
            or first.data.shape != prompt_embed.shape
        ):
            raise ValueError("MLX prompt captures must cover every prompt token")
        else:
            pass
        return first.data.to(dtype=prompt_embed.dtype)
