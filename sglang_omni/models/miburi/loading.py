# SPDX-License-Identifier: Apache-2.0
"""Build the gesture LM and its three motion codecs from the MIBURI checkpoint."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import yaml
from huggingface_hub import snapshot_download
from safetensors import safe_open

from sglang_omni.models.miburi.architecture import GestureLMSpec, MotionCodecSpec
from sglang_omni.models.miburi.components.gesture_model import GestureModel
from sglang_omni.models.miburi.components.motion_codec import MotionCodec
from sglang_omni.models.miburi.motion import BodyPartMasks

IDLE_DIR = "tokens_idle_slices"
CHECKPOINT_PREFIX = "module."
CODEC_CONFIG_KEYS = ("upperbodycodec_ckpt", "lowerbodycodec_ckpt", "facecodec_ckpt")
CODEC_FEATURE_KEYS = ("upperlower_nfeats", "lowertrans_nfeats", "face_nfeats")
# The face codec has half the transformer layers and heads the experiment config names.
CODEC_DEPTH_DIVISORS = (1, 1, 2)
# Gesture LM tensors serving never reads: the voice-activity head and the codebook copies only train, and the depth stack's memory attention has one key, so its query side (norm3, in_projs) and keys are inert.
UNUSED_LM_KEYS = re.compile(
    r"vad_predictor\..*"
    r"|gesture_codec_layers\..*"
    r"|depth_transformer\.layers\.\d+\.norm3\..*"
    r"|depth_transformer\.layers\.\d+\.cross_attns\.\d+\.(in_projs|key_projs)\..*"
)
# The codecs' encoder side.
UNUSED_CODEC_KEYS = re.compile(
    r"seanet_encoder\..*|transformer_encoder\..*|quantizer\.input_proj\..*"
)
# Checkpoint key fragments → GestureModel's names.
LM_KEY_RENAMES = (
    ("text_procemb.", "text_embedding."),
    ("audio_procemb.", "audio_embeddings."),
    ("temp_condproj.", "temporal_condition_projs."),
    ("dep_condproj.", "depth_condition_projs."),
    ("spk_tempemb.", "temporal_speaker_embedding."),
    ("spk_depemb.", "depth_speaker_embeddings."),
    ("temporal_gemb.", "temporal_code_embeddings."),
    ("depformer_gemb.", "depth_code_embeddings."),
    ("temporal_gproj.", "temporal_code_projs."),
    ("depformer_gproj.", "depth_code_projs."),
    ("depformer_in.", "depth_hidden_projs."),
    ("out_norm.", "temporal_out_norm."),
    ("depformer_classifier.", "depth_classifiers."),
    (".in_projs.", ".query_projs."),
    (".val_projs.", ".value_projs."),
    (".norm3.", ".cross_attn_norms."),
    (".memory_projs.0.", ".memory_fusion."),
    (".ca_gating.", ".cross_attn_feed_forward."),
    (".gating.", ".feed_forward."),
)
# Checkpoint key fragments → the PersonaPlex components' and MotionCodec's names.
CODEC_KEY_RENAMES = (
    ("._codebook.", ".codebook."),
    ("transformer_decoder.transformer.", "transformer."),
    (".self_attn.in_projs.0.", ".self_attn.in_proj."),
    (".self_attn.out_projs.0.", ".self_attn.out_proj."),
    ("seanet_decoder.model.", "decoder."),
    (".conv.conv", ".conv"),
    (".convtr.convtr", ".convtr"),
)


@dataclass(frozen=True, kw_only=True)
class GestureSampling:
    """Guidance and per-stack sampling; temperature 0 picks the most likely code."""

    cfg_coef: float
    temporal_temperature: float
    temporal_top_p: float
    depth_temperature: float
    depth_top_p: float


@dataclass(kw_only=True)
class GestureRuntime:
    """The loaded gesture LM, its upper, lower and face codecs, and what every session shares."""

    model: GestureModel
    codecs: tuple[MotionCodec, MotionCodec, MotionCodec]
    sampling: GestureSampling
    speaker_id: int
    idle_upper_codes: torch.Tensor
    idle_face_codes: torch.Tensor
    masks: BodyPartMasks
    motion_fps: float
    device: torch.device


def fetch(checkpoint: str, patterns: list[str]) -> Path:
    """A local checkpoint directory as is, else the files of the Hub repo that match patterns."""
    path = Path(checkpoint)
    if path.is_dir():
        return path
    else:
        return Path(snapshot_download(checkpoint, allow_patterns=patterns))


def read_checkpoint(path: Path) -> dict[str, torch.Tensor]:
    with safe_open(str(path), framework="pt", device="cpu") as weights:
        return {
            key.removeprefix(CHECKPOINT_PREFIX): weights.get_tensor(key)
            for key in weights.keys()
        }


def rename_key(key: str, renames: tuple[tuple[str, str], ...]) -> str:
    for old, new in renames:
        key = key.replace(old, new)
    return key


def gesture_lm_state(
    checkpoint: dict[str, torch.Tensor], spec: GestureLMSpec
) -> dict[str, torch.Tensor]:
    """The gesture LM tensors serving reads, under GestureModel's keys."""
    state = {
        rename_key(key, LM_KEY_RENAMES): tensor
        for key, tensor in checkpoint.items()
        if UNUSED_LM_KEYS.fullmatch(key) is None
    }
    for codebook in range(spec.depth_steps):
        # GestureModel ties these codebooks' temporal tables to the depth path's; the checkpoint's own are dead weights.
        state[f"temporal_code_embeddings.{codebook}.weight"] = checkpoint[
            f"depformer_gemb.{codebook}.weight"
        ]
    return state


def motion_codec_state(checkpoint: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """A codec's decoder tensors under MotionCodec's keys, weight norms folded."""
    state = {}
    for key, tensor in checkpoint.items():
        if UNUSED_CODEC_KEYS.fullmatch(key) is not None or key.endswith(".weight_g"):
            pass
        elif key.endswith(".weight_v"):
            stem = key.removesuffix(".weight_v")
            # Note (Junnan Li): Folded by the call weight norm computes the weight with, the only form that reproduces it bit for bit; the norm runs over every axis but 0, the input channels of the transposed convolution.
            state[f"{rename_key(stem, CODEC_KEY_RENAMES)}.weight"] = (
                torch._weight_norm(  # noqa: leading-underscore  # torch's weight-norm kernel
                    tensor, checkpoint[f"{stem}.weight_g"], 0
                )
            )
        else:
            state[rename_key(key, CODEC_KEY_RENAMES)] = tensor
    return state


def load_motion_codecs(
    root: Path,
    config: dict[str, object],
    codec_paths: list[str],
    spec: GestureLMSpec,
    device: torch.device,
) -> tuple[MotionCodec, MotionCodec, MotionCodec]:
    """The upper, lower and face codecs the experiment config names."""
    codecs = []
    for path, features_key, codebooks, divisor in zip(
        codec_paths,
        CODEC_FEATURE_KEYS,
        spec.codebooks_per_part,
        CODEC_DEPTH_DIVISORS,
        strict=True,
    ):
        codec = MotionCodec(
            MotionCodecSpec(
                features=config[features_key],
                codebooks=codebooks,
                num_layers=config["transformer_layers"] // divisor,
                num_heads=config["transformer_heads"] // divisor,
                residual_blocks=config["convblock_layers"],
                upsample_ratio=config["frame_chunk_size"],
            )
        )
        codec.load_state_dict(motion_codec_state(read_checkpoint(root / path)))
        codecs.append(codec.to(device))
    return tuple(codecs)


def load_gesture_runtime(
    *,
    checkpoint: str,
    experiment: str,
    speaker_id: int,
    cfg_coef: float,
    temporal_temperature: float,
    temporal_top_p: float,
    depth_temperature: float,
    depth_top_p: float,
    device: torch.device,
) -> GestureRuntime:
    """Build the experiment's gesture LM and codecs; every checkpoint tensor is loaded or named unused in UNUSED_LM_KEYS / UNUSED_CODEC_KEYS."""
    experiment_dir = f"experiments/{experiment}"
    root = fetch(checkpoint, [f"{experiment_dir}/*"])
    config = yaml.safe_load((root / experiment_dir / "config.yaml").read_text())
    speakers = json.loads(
        (root / experiment_dir / "speaker_id_to_index.json").read_text()
    )
    if speaker_id not in speakers.values():
        raise ValueError(
            f"speaker_id {speaker_id} is not a speaker of {experiment}: {speakers}"
        )
    else:
        pass
    codec_paths = [config[name] for name in CODEC_CONFIG_KEYS]
    root = fetch(
        checkpoint,
        [f"{experiment_dir}/*"] + [f"{Path(path).parent}/*" for path in codec_paths],
    )
    spec = GestureLMSpec(
        temporal_layers=config["gestureformer_layers"],
        temporal_heads=config["gestureformer_heads"],
        depth_layers=config["gestureformer_depformer_layers"],
        depth_heads=config["gestureformer_depformer_heads"],
    )
    model = GestureModel(spec)
    model.load_state_dict(
        gesture_lm_state(
            read_checkpoint(root / experiment_dir / Path(config["test_ckpt"]).name),
            spec,
        )
    )

    def idle_codes(name: str) -> torch.Tensor:
        with np.load(
            root / experiment_dir / IDLE_DIR / name, allow_pickle=False
        ) as slices:
            return torch.from_numpy(slices["0"]).to(device=device, dtype=torch.long)

    return GestureRuntime(
        model=model.to(device),
        codecs=load_motion_codecs(root, config, codec_paths, spec, device),
        sampling=GestureSampling(
            cfg_coef=cfg_coef,
            temporal_temperature=temporal_temperature,
            temporal_top_p=temporal_top_p,
            depth_temperature=depth_temperature,
            depth_top_p=depth_top_p,
        ),
        speaker_id=speaker_id,
        idle_upper_codes=idle_codes("idle_upper_tokens_slices.npz"),
        idle_face_codes=idle_codes("idle_face_tokens_slices.npz"),
        masks=BodyPartMasks.smplx(),
        motion_fps=float(config["motion_fps"]),
        device=device,
    )


__all__ = [
    "GestureRuntime",
    "GestureSampling",
    "load_gesture_runtime",
]
