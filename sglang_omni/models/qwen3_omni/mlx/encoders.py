"""Native MLX encoder and vocoder tensor adapters."""

import math

import mlx.core as mx
import torch
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
    _get_feat_extract_output_lengths,
)

from sglang_omni.models.qwen3_omni.components.audio_encoder import (
    pack_padded_audio_features,
)
from sglang_omni.models.qwen3_omni.components.image_encoder import ImageEncoderOutput
from sglang_omni.models.qwen3_omni.mlx.loading import load_component
from sglang_omni.models.qwen3_omni.mlx.tensors import mlx_to_torch, torch_to_mlx


class MlxImageEncoder:
    def __init__(self, model_path: str) -> None:
        self.visual = load_component(model_path, "vision")
        config = self.visual.config
        self.spatial_merge_size = config.spatial_merge_size
        self.out_hidden_size = config.out_hidden_size
        self.deepstack_layers = len(config.deepstack_visual_indexes)
        self.visual_dtype_bytes = 4

    def __call__(
        self,
        *,
        pixel_values: torch.Tensor | None = None,
        image_grid_thw: torch.Tensor | None = None,
        pixel_values_videos: torch.Tensor | None = None,
        video_grid_thw: torch.Tensor | None = None,
        video_second_per_grid: torch.Tensor | None = None,
        use_audio_in_video: bool | None = None,
    ) -> ImageEncoderOutput:
        # note (Copilot): Timing/interleaving stay in shared mm_inputs for M-RoPE;
        # the vision tower consumes only patches and their spatial/temporal grid.
        outputs: ImageEncoderOutput = {}
        with mx.stream(mx.new_thread_local_stream(mx.gpu)):
            for modality, pixels, grid in (
                ("image", pixel_values, image_grid_thw),
                ("video", pixel_values_videos, video_grid_thw),
            ):
                if pixels is None and grid is None:
                    continue
                else:
                    pass
                if pixels is None or grid is None:
                    raise ValueError(f"{modality} requires pixels and grid metadata")
                else:
                    pass
                if grid.ndim != 2 or grid.shape[1] != 3 or torch.any(grid <= 0):
                    raise ValueError(f"Invalid {modality} grid")
                else:
                    pass
                if int(grid.prod(-1).sum()) != pixels.shape[0]:
                    raise ValueError(f"{modality} patch count does not match grid")
                else:
                    pass
                features, layers = self.visual(
                    torch_to_mlx(pixels),
                    grid_thw=torch_to_mlx(grid).astype(mx.int32),
                )
                outputs[f"{modality}_embeds"] = mlx_to_torch(features)
                outputs[f"{modality}_grid_thw"] = grid.cpu()
                outputs[f"{modality}_token_counts"] = (
                    grid.cpu().prod(-1) // self.spatial_merge_size**2
                )
                outputs[f"deepstack_visual_embeds_{modality}"] = [
                    mlx_to_torch(layer) for layer in layers
                ]
        return outputs


class MlxAudioEncoder:
    def __init__(self, model_path: str) -> None:
        self.audio_tower = load_component(model_path, "audio")

    def __call__(
        self,
        *,
        input_features: torch.Tensor,
        feature_attention_mask: torch.Tensor | None = None,
        audio_feature_lengths: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if feature_attention_mask is not None:
            mask = feature_attention_mask.cpu().bool()
            audio_feature_lengths = mask.sum(-1)
            if input_features.ndim != 3 or mask.shape != (
                input_features.shape[0],
                input_features.shape[-1],
            ):
                raise ValueError(
                    "Audio features and mask must agree in batch/time dimensions"
                )
            else:
                pass
            expected = (
                torch.arange(mask.shape[-1])[None, :] < audio_feature_lengths[:, None]
            )
            if not torch.equal(mask, expected):
                raise ValueError(
                    "Omni MLX audio masks must describe valid frame prefixes"
                )
            else:
                pass
            input_features = pack_padded_audio_features(
                input_features, mask, audio_feature_lengths
            )
        else:
            pass
        if audio_feature_lengths is None or torch.any(audio_feature_lengths <= 0):
            raise ValueError("Audio input requires positive valid feature lengths")
        else:
            pass
        if input_features.ndim != 2 or input_features.shape[-1] != int(
            audio_feature_lengths.sum()
        ):
            raise ValueError(
                "Packed audio features do not match declared frame lengths"
            )
        else:
            pass
        with mx.stream(mx.new_thread_local_stream(mx.gpu)):
            features = self.audio_tower(
                torch_to_mlx(input_features),
                feature_lens=torch_to_mlx(audio_feature_lengths).astype(mx.int32),
            )
            output = mlx_to_torch(features)
        lengths = _get_feat_extract_output_lengths(audio_feature_lengths)
        if output.shape[0] != int(lengths.sum()):
            raise ValueError("Audio encoder output does not match token counts")
        else:
            pass
        return {
            "audio_embeds": output,
            "audio_feature_lengths": audio_feature_lengths.cpu(),
            "audio_output_lengths": lengths.cpu(),
        }


class MlxCode2Wav:
    def __init__(self, model_path: str) -> None:
        self.model = load_component(model_path, "code2wav")
        self.config = self.model.config
        self.total_upsample = math.prod(
            self.config.upsampling_ratios + self.config.upsample_rates
        )

    def __call__(self, codes: torch.Tensor) -> torch.Tensor:
        if codes.ndim != 3 or codes.shape[1] != self.config.num_quantizers:
            raise ValueError("Code2wav requires [batch, quantizers, frames] codes")
        else:
            pass
        if torch.any(codes < 0) or torch.any(codes >= self.config.codebook_size):
            raise ValueError("Code2wav received an invalid codec token")
        else:
            pass
        with mx.stream(mx.new_thread_local_stream(mx.gpu)):
            return mlx_to_torch(self.model(torch_to_mlx(codes).astype(mx.int32)))
