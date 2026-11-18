"""Standalone Zipformer streaming encoder derived from X-ASR's Apache-2.0 recipe.

Checkpoint loading and end-to-end compatibility remain experimental.
"""

import torch
from torch import nn

from sglang_omni.vendor.xasr.scaling_converter import convert_scaled_to_non_scaled
from sglang_omni.vendor.xasr.subsampling import Conv2dSubsampling
from sglang_omni.vendor.xasr.zipformer import Zipformer2


class StreamingZipformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder_embed = Conv2dSubsampling(
            in_channels=80, out_channels=192, dropout=0.0
        )
        self.encoder = Zipformer2(
            output_downsampling_factor=2,
            downsampling_factor=(1, 2, 4, 8, 4, 2),
            num_encoder_layers=(2, 2, 4, 5, 4, 2),
            encoder_dim=(192, 256, 512, 768, 512, 256),
            encoder_unmasked_dim=(192, 192, 256, 256, 256, 192),
            query_head_dim=32,
            pos_head_dim=4,
            value_head_dim=12,
            pos_dim=48,
            num_heads=(4, 4, 4, 8, 4, 4),
            feedforward_dim=(512, 768, 1536, 2048, 1536, 768),
            cnn_module_kernel=(31, 31, 15, 15, 15, 31),
            dropout=0.0,
            causal=True,
            chunk_size=(24,),
            left_context_frames=(256,),
        )
        self.encoder_proj = nn.Linear(768, 512)

    def initial_states(self, batch, device):
        states = self.encoder.get_init_states(batch, device)
        states.append(torch.zeros(batch, 128, 3, 19, device=device))
        states.append(torch.zeros(batch, dtype=torch.int64, device=device))
        return states

    def forward(self, x, states):
        n = x.size(0)
        lengths = torch.full((n,), 61, device=x.device, dtype=torch.int64)
        x, lengths, embed_state = self.encoder_embed.streaming_forward(
            x=x, x_lens=lengths, cached_left_pad=states[-2]
        )
        mask = (
            states[-1].unsqueeze(1) <= torch.arange(256, device=x.device).expand(n, 256)
        ).flip(1)
        mask = torch.cat(
            (mask, torch.zeros(n, 24, dtype=torch.bool, device=x.device)), dim=1
        )
        out, _, next_states = self.encoder.streaming_forward(
            x=x.permute(1, 0, 2),
            x_lens=lengths,
            states=states[:-2],
            src_key_padding_mask=mask,
        )
        return self.encoder_proj(out.permute(1, 0, 2)), [
            *next_states,
            embed_state,
            states[-1] + lengths,
        ]

    def inference_conversion(self):
        return convert_scaled_to_non_scaled(self, inplace=True, is_onnx=True).eval()
