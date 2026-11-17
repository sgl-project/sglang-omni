# SPDX-License-Identifier: Apache-2.0
"""Small production-class Fish codec for CPU decoder and scheduler tests."""

from dataclasses import dataclass
from functools import partial

import torch

from sglang_omni.models.fishaudio_drama3.stateful_decoder import StatefulDACDecoder
from sglang_omni.models.fishaudio_s2_pro.fish_speech.models.dac.modded_dac import (
    DAC,
    ModelArgs,
    WindowLimitedTransformer,
)
from sglang_omni.models.fishaudio_s2_pro.fish_speech.models.dac.rvq import (
    DownsampleResidualVectorQuantize,
)


@dataclass(kw_only=True)
class CpuCodecDeltaGraph:
    backend: StatefulDACDecoder
    codes: torch.Tensor
    slot_index: torch.Tensor
    frame_count: int

    def replay(self) -> torch.Tensor:
        return self.backend.decode_slot(self.codes, self.slot_index, self.frame_count)


class CpuCodecGraphBoundary:
    def capture(
        self,
        backend: StatefulDACDecoder,
        codes: torch.Tensor,
        device_slot_index: torch.Tensor,
        frame_count: int,
    ) -> CpuCodecDeltaGraph:
        return CpuCodecDeltaGraph(
            backend=backend,
            codes=codes,
            slot_index=device_slot_index,
            frame_count=frame_count,
        )

    def wait_before_submit(self) -> None:
        pass

    def record_submission(self) -> None:
        pass

    def synchronize(self) -> None:
        pass


def tiny_fishaudio_codec() -> DAC:
    torch.manual_seed(37)
    quantizer = DownsampleResidualVectorQuantize(
        input_dim=16,
        n_codebooks=9,
        codebook_dim=4,
        codebook_size=16,
        semantic_codebook_size=16,
        downsample_factor=(2, 2),
        post_module=WindowLimitedTransformer(
            ModelArgs(
                n_layer=1,
                n_head=2,
                dim=16,
                head_dim=8,
                intermediate_size=32,
                dropout_rate=0,
                attn_dropout_rate=0,
            ),
            input_dim=16,
            window_size=8,
            causal=True,
        ),
    )
    return (
        DAC(
            encoder_dim=8,
            encoder_rates=[2, 2],
            latent_dim=16,
            decoder_dim=32,
            decoder_rates=[2, 2],
            quantizer=quantizer,
            encoder_transformer_layers=[0, 0],
            decoder_transformer_layers=[0, 0],
            transformer_general_config=partial(ModelArgs, window_size=8),
            causal=True,
        )
        .eval()
        .float()
    )
