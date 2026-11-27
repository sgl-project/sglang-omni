# SPDX-License-Identifier: Apache-2.0
"""Adapt VoiceChat's Thinker to the Ascend Mamba2 backend in SGLang 0.5.23."""

from dataclasses import replace

import torch
from sglang.srt.layers.attention.hybrid_linear_attn_backend import Mamba2AttnBackend
from sglang.srt.layers.attention.mamba.mamba import MambaMixer2
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.utils import is_npu

if is_npu():
    from sglang.srt.hardware_backend.npu.attention.ascend_hybrid_linear_attn_backend import (
        AscendMambaAttnBackendBase,
    )

    class VoiceChatAscendMamba2Backend(AscendMambaAttnBackendBase, Mamba2AttnBackend):
        def forward(
            self,
            mixer: MambaMixer2,
            hidden_states: torch.Tensor,
            output: torch.Tensor | None,
            layer_id: int,
            forward_batch: ForwardBatch,
            mup_vector: torch.Tensor | None = None,
            use_triton_causal_conv: bool = False,
        ) -> torch.Tensor:
            del use_triton_causal_conv
            if forward_batch.mamba_track_mask is not None:
                raise RuntimeError(
                    "VoiceChat on Ascend requires Mamba checkpoint tracking to be disabled"
                )
            else:
                pass
            layer_cache = self.req_to_token_pool.mamba2_layer_cache(layer_id)
            # note (celestial-micha): The NPU pool places the convolution window before the channels.
            mixer_cache = replace(
                layer_cache, conv=[layer_cache.conv[0].transpose(-1, -2)]
            )
            mixer_output, _, _ = mixer.forward(
                hidden_states=hidden_states,
                output=output,
                layer_cache=mixer_cache,
                metadata=self.forward_metadata,
                mup_vector=mup_vector,
                use_triton_causal_conv=False,
            )
            return mixer_output

else:
    pass
