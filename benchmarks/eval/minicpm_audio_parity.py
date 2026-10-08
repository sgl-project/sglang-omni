"""Compare frozen audio encoder outputs and isolate attention mask differences."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import librosa
import torch
from transformers import AutoConfig, AutoProcessor
from transformers.dynamic_module_utils import get_class_from_dynamic_module

from sglang_omni.models.minicpm_o.components.audio_encoder import (
    MiniCPMOAudioEncoder,
    chunked_causal_mask,
    feature_lens_after_pooling,
)
from sglang_omni.models.weight_loader import load_weights_by_prefix


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--audio", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backend", choices=["official", "omni"], required=True)
    arguments = parser.parse_args()
    processor = AutoProcessor.from_pretrained(
        arguments.model_path, trust_remote_code=True
    )
    waveform, sample_rate = librosa.load(arguments.audio, sr=16000, mono=True)
    processed = processor("<audio>./</audio>", audios=[[waveform]], return_tensors="pt")
    features = processed["audio_features"].to("cuda", dtype=torch.bfloat16)
    lengths = torch.hstack(processed["audio_feature_lens"]).to("cuda")
    if arguments.backend == "official":
        config = AutoConfig.from_pretrained(
            arguments.model_path, trust_remote_code=True
        )
        config.audio_config._attn_implementation = "sdpa"
        encoder_class = get_class_from_dynamic_module(
            "modeling_minicpmo.MiniCPMWhisperEncoder", arguments.model_path
        )
        encoder = (
            encoder_class(config.audio_config).eval().to("cuda", dtype=torch.bfloat16)
        )
        encoder.load_state_dict(
            load_weights_by_prefix(arguments.model_path, prefix=("apm.",))
        )
        projector_class = get_class_from_dynamic_module(
            "modeling_minicpmo.MultiModalProjector", arguments.model_path
        )
        projector = (
            projector_class(
                in_dim=config.audio_config.d_model, out_dim=config.hidden_size
            )
            .eval()
            .to("cuda", dtype=torch.bfloat16)
        )
        projector.load_state_dict(
            load_weights_by_prefix(
                arguments.model_path, prefix=("audio_projection_layer.",)
            )
        )
        model_class = get_class_from_dynamic_module(
            "modeling_minicpmo.MiniCPMO", arguments.model_path
        )
        reference = SimpleNamespace(
            apm=encoder,
            audio_projection_layer=projector,
            audio_avg_pooler=torch.nn.AvgPool1d(config.audio_pool_step),
            audio_encoder_layer=-1,
        )
        reference.subsequent_chunk_mask = model_class.subsequent_chunk_mask
        reference._get_feat_extract_output_lengths = (
            model_class._get_feat_extract_output_lengths.__get__(reference)
        )
        reference.config = config
        with torch.inference_mode():
            embeddings = model_class.get_audio_embedding(
                reference,
                {"audio_features": features, "audio_feature_lens": [lengths]},
                chunk_length=config.audio_chunk_length,
            )
        torch.save(
            {
                "official": torch.cat(embeddings[0]).cpu(),
                "features": features.cpu(),
                "lengths": lengths.cpu(),
            },
            arguments.output,
        )
    else:
        encoder = MiniCPMOAudioEncoder(arguments.model_path)
        golden = torch.load(arguments.output, weights_only=True)
        with torch.inference_mode():
            native = encoder(audio_features=features, audio_feature_lens=lengths)[
                "audio_embeds"
            ]
            sequence_length = (features.shape[-1] + 1) // 2
            positions = torch.arange(sequence_length, device=features.device)
            visible = positions[None, :] < lengths[:, None]
            allowed = (
                chunked_causal_mask(
                    sequence_length, encoder.chunk_num_frame, encoder.device
                )[None]
                & visible[:, None, :]
            )
            mask = (
                torch.where(allowed, 0.0, float("-inf")).to(encoder.dtype).unsqueeze(1)
            )
            states, cache = encoder.apm(features, mask)
            projected = encoder.audio_projection_layer(states)
            pooled = encoder.audio_avg_pooler(projected.transpose(1, 2)).transpose(1, 2)
            pooled_lengths = feature_lens_after_pooling(
                lengths, encoder.audio_pool_step
            )
            keep = (
                torch.arange(pooled.shape[1], device=features.device)[None]
                < pooled_lengths[:, None]
            )
            matched = pooled[keep]
        reference = golden["official"].to(native.device)
        print(
            json.dumps(
                {
                    "mel_equal": torch.equal(golden["features"], features.cpu()),
                    "lengths_equal": torch.equal(golden["lengths"], lengths.cpu()),
                    "native_mean_absolute_error": (native.float() - reference.float())
                    .abs()
                    .mean()
                    .item(),
                    "official_mask_mean_absolute_error": (
                        matched.float() - reference.float()
                    )
                    .abs()
                    .mean()
                    .item(),
                    "native_max_absolute_error": (native.float() - reference.float())
                    .abs()
                    .max()
                    .item(),
                    "official_mask_max_absolute_error": (
                        matched.float() - reference.float()
                    )
                    .abs()
                    .max()
                    .item(),
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
