"""Omni multimodal prefill on the shared native MLX worker lifecycle."""

import json
from pathlib import Path
from typing import Protocol

import mlx.core as mx
import numpy as np
import torch
from mlx_lm.models.cache import KVCache
from mlx_vlm.models.qwen3_omni_moe.language import (
    Qwen3OmniMoeThinkerTextRotaryEmbedding,
)
from sglang.srt.hardware_backend.mlx.kv_cache import (
    ContiguousAttentionKVCache,
    MLXAttentionWrapper,
)
from sglang.srt.hardware_backend.mlx.model_runner import (
    MlxModelRunner,
    MlxPendingDecode,
    MlxPendingPrefill,
)
from sglang.srt.hardware_backend.mlx.sampling import MlxLogprobSpec
from sglang.srt.managers.schedule_batch import Req, ScheduleBatch

from sglang_omni.model_runner.mlx_model_worker import MlxSchedulerModelRunner
from sglang_omni.models.qwen3_omni.mlx.loading import load_component
from sglang_omni.models.qwen3_omni.mlx.tensors import (
    mlx_to_torch,
    request_positions,
    torch_to_mlx,
)
from sglang_omni.models.weight_loader import resolve_model_path
from sglang_omni.scheduling.sglang_backend.request_data import SGLangARRequestData


class LogitsHook(Protocol):
    def __call__(
        self, logits: np.ndarray[tuple[int, ...], np.dtype[np.float32]]
    ) -> np.ndarray[tuple[int, ...], np.dtype[np.float32]]: ...


class OmniAttentionWrapper(MLXAttentionWrapper):
    """Preserve explicit M-RoPE instead of the text-only batched wrapper."""

    @property
    def rotary_emb(self) -> Qwen3OmniMoeThinkerTextRotaryEmbedding:
        return (
            self._inner.rotary_emb
        )  # noqa: leading-underscore - native wrapper contract

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | str | None = None,
        cache: KVCache | ContiguousAttentionKVCache | None = None,
        position_ids: mx.array | None = None,
        position_embeddings: tuple[mx.array, mx.array] | None = None,
    ) -> mx.array:
        return self._inner(  # noqa: leading-underscore - native wrapper contract
            x,
            mask=mask,
            cache=cache,
            position_ids=position_ids,
            position_embeddings=position_embeddings,
        )


class OmniMlxRunner(MlxModelRunner):
    component = "language"

    def _load_model(self) -> None:  # noqa: leading-underscore - native runner hook
        if not self.disable_radix_cache:
            raise ValueError("Qwen3-Omni MLX requires disable_radix_cache=True")
        else:
            pass
        self.model_path = str(resolve_model_path(self.model_path))
        self.model = load_component(self.model_path, self.component)
        for index, layer in enumerate(self.model.model.layers):
            layer.self_attn = OmniAttentionWrapper(layer.self_attn, index)
        self.position_deltas: dict[str, int] = {}
        raw = json.loads((Path(self.model_path) / "config.json").read_text())
        self.placeholder_token_ids = {
            modality: raw["thinker_config"][f"{modality}_token_id"]
            for modality in ("image", "video", "audio")
        }
        self.accept_hidden_layer = raw["talker_config"]["accept_hidden_layer"]

    def remove_request(self, req_id: str) -> None:
        self.position_deltas.pop(req_id, None)
        super().remove_request(req_id)

    def clear(self) -> None:
        self.position_deltas.clear()
        super().clear()

    def prefill_start(
        self,
        req_id: str,
        new_token_ids: list[int],
        full_token_ids: list[int],
        prefix_slot_ids: list[int],
        new_slot_ids: list[int],
        req_pool_idx: int,
        req: Req | None = None,
        needs_logits: bool = True,
        logit_edit_row: mx.array | None = None,
        logprob_spec: MlxLogprobSpec | None = None,
    ) -> MlxPendingPrefill:
        if req is None:
            raise ValueError("Omni MLX requires its scheduler request")
        else:
            pass
        if (
            prefix_slot_ids
            or len(new_token_ids) != len(full_token_ids)
            or not needs_logits
        ):
            raise ValueError(
                "Omni MLX does not support prefix caching or chunked prefill"
            )
        else:
            pass
        if logit_edit_row is not None or logprob_spec is not None:
            raise ValueError("Omni MLX currently supports greedy thinker decoding only")
        else:
            pass
        if self.position_deltas.keys() - {req_id}:
            raise ValueError("Omni MLX supports one active request per AR stage")
        else:
            pass
        model_inputs = req.omni_model_inputs or {}
        token_ids = list(new_token_ids)
        pad_values = model_inputs.get("pad_values", {})
        for modality, pad in pad_values.items():
            token_ids = [
                self.placeholder_token_ids[modality] if token == pad else token
                for token in token_ids
            ]
        if any(token < 0 or token >= self.model.args.vocab_size for token in token_ids):
            raise ValueError("Missing modality placeholder metadata in Omni MLX prompt")
        else:
            pass
        input_ids = mx.array([token_ids], dtype=mx.int32)
        embeddings = self.model.model.embed_tokens(input_ids)
        visual_positions: list[int] = []
        visual_layers: list[tuple[list[int], list[torch.Tensor]]] = []
        for modality, token_id in self.placeholder_token_ids.items():
            positions = [i for i, token in enumerate(token_ids) if token == token_id]
            features = model_inputs.get(f"{modality}_embeds")
            if features is None:
                if positions:
                    raise ValueError(
                        f"{modality} placeholders require encoder embeddings"
                    )
                else:
                    pass
                continue
            else:
                pass
            if features.shape != (
                len(positions),
                embeddings.shape[-1],
            ):
                raise ValueError(
                    f"{modality} embeddings do not match placeholder count/width"
                )
            else:
                pass
            embeddings[:, mx.array(positions), :] = torch_to_mlx(features).astype(
                embeddings.dtype
            )
            if modality in ("image", "video"):
                visual_positions.extend(positions)
                layers = model_inputs.get(f"{modality}_deepstack_visual_embeds")
                if layers is None:
                    layers = model_inputs.get("deepstack_visual_embeds")
                else:
                    pass
                if layers is None:
                    raise ValueError(f"{modality} requires DeepStack residuals")
                else:
                    visual_layers.append((positions, layers))
            else:
                pass
        deepstack = None
        if visual_layers:
            layer_count = len(self.model.config.vision_config.deepstack_visual_indexes)
            deepstack = mx.zeros(
                (1, len(token_ids), layer_count, embeddings.shape[-1]),
                dtype=embeddings.dtype,
            )
            for positions, layers in visual_layers:
                if len(layers) != layer_count:
                    raise ValueError(
                        "Visual DeepStack layer count does not match model"
                    )
                else:
                    pass
                for index, features in enumerate(layers):
                    if tuple(features.shape) != (len(positions), embeddings.shape[-1]):
                        raise ValueError(
                            "Visual DeepStack rows do not match placeholders"
                        )
                    else:
                        pass
                    deepstack[:, mx.array(positions), index, :] = torch_to_mlx(
                        features
                    ).astype(embeddings.dtype)
        else:
            pass
        positions, delta = request_positions(req, len(token_ids))
        cache = (
            self._acquire_cache()
        )  # noqa: leading-underscore - native cache ownership
        # note (Copilot): Explicit positions and cleared library state prevent cross-request reuse.
        self.model._rope_deltas = (
            None  # noqa: leading-underscore - library mutable position state
        )
        self.model._position_ids = (
            None  # noqa: leading-underscore - library mutable position state
        )
        output = self.model(
            input_ids,
            inputs_embeds=embeddings,
            cache=cache,
            position_ids=positions,
            deepstack_visual_embeds=deepstack,
            output_hidden_state_idx=(
                self.accept_hidden_layer
                if visual_positions or model_inputs.get("audio_embeds") is not None
                else None
            ),
        )
        if output.hidden_states is not None:
            req.omni_mlx_prompt_hidden = mlx_to_torch(output.hidden_states[0])
            req.omni_mlx_capture_layer = self.accept_hidden_layer
        else:
            pass
        self.position_deltas[req_id] = delta
        return MlxPendingPrefill(
            lazy_token=mx.argmax(output.logits[:, -1, :], axis=-1),
            cache=cache,
            req_id=req_id,
            full_token_ids=token_ids,
            req_pool_idx=req_pool_idx,
            synced_offset=0,
        )

    def decode_batch_start(
        self,
        req_ids: list[str],
        edit_rows: mx.array | None = None,
        logprob_spec: MlxLogprobSpec | None = None,
        logits_hook: LogitsHook | None = None,
    ) -> MlxPendingDecode:
        if (
            len(req_ids) != 1
            or edit_rows is not None
            or logprob_spec is not None
            or logits_hook is not None
        ):
            raise ValueError("Omni MLX requires single-request greedy decode")
        else:
            pass
        request_id = req_ids[0]
        cache = self._req_caches[
            request_id
        ]  # noqa: leading-underscore - native cache ownership
        position = cache[0].offset + self.position_deltas[request_id]
        output = self.model(
            mx.array(
                [[self._req_token_ids[request_id][-1]]], dtype=mx.int32
            ),  # noqa: leading-underscore - native token ownership
            cache=cache,
            position_ids=mx.full((3, 1, 1), position, dtype=mx.int32),
        )
        return MlxPendingDecode(
            lazy_tokens=mx.argmax(output.logits[:, -1, :], axis=-1),
            req_ids=req_ids,
            caches=[cache],
        )


class OmniTalkerMlxRunner(OmniMlxRunner):
    component = "talker"


def make_qwen3_omni_thinker_mlx_runner_class() -> type[OmniMlxRunner]:
    return OmniMlxRunner


def make_qwen3_omni_talker_mlx_runner_class() -> type[OmniTalkerMlxRunner]:
    return OmniTalkerMlxRunner


class OmniMlxSchedulerModelRunner(MlxSchedulerModelRunner):
    def lookahead_eligible(self, batch: ScheduleBatch) -> bool:
        return False

    def abort_request(self, request_id: str) -> None:
        self.tp_worker._mlx_runner.remove_request(
            request_id
        )  # noqa: leading-underscore - native worker contract

    def on_request_finished(
        self, request_id: str, req_data: SGLangARRequestData
    ) -> None:
        self.abort_request(request_id)
        req_data.req.omni_mlx_prompt_hidden = None
