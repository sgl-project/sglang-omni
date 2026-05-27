# SPDX-License-Identifier: Apache-2.0
"""Higgs-owned Qwen3 backbone with decode-only ``torch.compile`` hook.

Mirrors :class:`sglang.srt.models.qwen2.Qwen2Model.forward` so we can pick
between eager ``self.layers`` (prefill) and ``self._compiled_decode_layers``
(decode, populated by :func:`stages._compile_higgs_backbone`).

When syncing with newer sglang, diff ``Qwen2Model.forward`` against the
loop here and port any new logic.
"""

from __future__ import annotations

from typing import Optional, Union

import torch
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors
from sglang.srt.models.qwen3 import Qwen3ForCausalLM, Qwen3Model
from sglang.srt.utils import add_prefix


class HiggsQwen3Model(Qwen3Model):
    """``Qwen3Model`` with a decode-only compile-layer indirection."""

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> Union[torch.Tensor, PPProxyTensors]:
        if self.pp_group.is_first_rank:
            hidden_states = (
                self.embed_tokens(input_ids) if input_embeds is None else input_embeds
            )
            residual = None
        else:
            assert pp_proxy_tensors is not None
            hidden_states = pp_proxy_tensors["hidden_states"]
            residual = pp_proxy_tensors["residual"]

        # Prefill keeps ``self.layers`` because its shape varies per request,
        # which would force dynamo recompiles. The ``_compiled_max_decode_bs``
        # hook is kept for future workarounds; with the eager pre-warmup the
        # full decode bs range is safe (see ``issue_565_torch_compile_result.md``).
        compiled = getattr(self, "_compiled_decode_layers", None)
        max_bs = getattr(self, "_compiled_max_decode_bs", None)
        forward_mode = getattr(forward_batch, "forward_mode", None)
        is_decode = bool(
            forward_mode is not None
            and getattr(forward_mode, "is_decode", lambda: False)()
        )
        bs = hidden_states.shape[0]
        use_compiled = (
            is_decode and compiled is not None and (max_bs is None or bs <= max_bs)
        )
        layers = compiled if use_compiled else self.layers

        aux_hidden_states: list[torch.Tensor] = []
        for i in range(self.start_layer, self.end_layer):
            if i in self.layers_to_capture:
                aux_hidden_states.append(
                    hidden_states + residual if residual is not None else hidden_states
                )
            hidden_states, residual = layers[i](
                positions, hidden_states, forward_batch, residual
            )
        if not self.pp_group.is_last_rank:
            return PPProxyTensors(
                {"hidden_states": hidden_states, "residual": residual}
            )

        if hidden_states.shape[0] != 0:
            if residual is None:
                hidden_states = self.norm(hidden_states)
            else:
                hidden_states, _ = self.norm(hidden_states, residual)

        if len(aux_hidden_states) == 0:
            return hidden_states
        return hidden_states, aux_hidden_states


class HiggsQwen3ForCausalLM(Qwen3ForCausalLM):
    """``Qwen3ForCausalLM`` with the inner model replaced by :class:`HiggsQwen3Model`."""

    def __init__(self, config, quant_config=None, prefix: str = "") -> None:
        super().__init__(config=config, quant_config=quant_config, prefix=prefix)
        # Re-tie ``lm_head`` after the model swap when ``tie_word_embeddings``
        # had pointed it at the old ``embed_tokens``.
        was_tied = (
            self.pp_group.is_last_rank
            and getattr(self, "lm_head", None) is self.model.embed_tokens
        )
        old_model = self.model
        self.model = HiggsQwen3Model(
            config,
            quant_config=quant_config,
            prefix=add_prefix("model", prefix),
        )
        if was_tied:
            self.lm_head = self.model.embed_tokens
        del old_model


__all__ = ["HiggsQwen3Model", "HiggsQwen3ForCausalLM"]
