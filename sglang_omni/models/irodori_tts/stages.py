# SPDX-License-Identifier: Apache-2.0
"""Native SGLang Omni inference stage for Irodori-TTS."""

from __future__ import annotations

import base64
import binascii
import json
import logging
import math
import mimetypes
import secrets
from contextlib import ExitStack
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from safetensors import safe_open
from safetensors.torch import load_file

from sglang_omni.models.irodori_tts.request_builders import (
    IrodoriSynthesisRequest,
    build_irodori_request,
)
from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.simple_scheduler import SimpleScheduler
from sglang_omni.utils.audio_payload import audio_waveform_payload
from sglang_omni.utils.device import resolve_concrete_device
from sglang_omni.models.irodori_tts.codec import (
    DACVAECodec,
    patchify_latent,
    unpatchify_latent,
)
from sglang_omni.models.irodori_tts.model_config import (
    ModelConfig,
    merge_dataclass_overrides,
)
from sglang_omni.models.irodori_tts.duration import build_duration_features
from sglang_omni.models.irodori_tts.dit import TextToLatentRFDiT
from sglang_omni.models.irodori_tts.flow_matching import sample_euler_rf_cfg
from sglang_omni.models.irodori_tts.text_normalization import normalize_text
from sglang_omni.models.irodori_tts.tokenizer import PretrainedTextTokenizer

logger = logging.getLogger(__name__)


def checkpoint_file(model_path: str) -> Path:
    resolved_path = Path(model_path).expanduser()
    if resolved_path.is_dir():
        resolved_path = resolved_path / "model.safetensors"
    else:
        pass
    if not resolved_path.is_file():
        raise FileNotFoundError(f"Irodori model.safetensors not found: {resolved_path}")
    else:
        pass
    return resolved_path.resolve()


def checkpoint_configuration(
    path: Path,
) -> tuple[dict[str, object], dict[str, object] | None]:
    with safe_open(str(path), framework="pt", device="cpu") as checkpoint:
        metadata = checkpoint.metadata() or {}
    raw_model_configuration = metadata.get("config_json")
    if raw_model_configuration is None:
        raise ValueError(f"Irodori checkpoint is missing config_json metadata: {path}")
    else:
        pass
    model_configuration = json.loads(raw_model_configuration)
    if not isinstance(model_configuration, dict):
        raise ValueError(f"Irodori config_json metadata must be an object: {path}")
    else:
        pass
    raw_text_configuration = metadata.get("text_encoder_config_json")
    if raw_text_configuration is None:
        text_configuration = None
    else:
        text_configuration = json.loads(raw_text_configuration)
        if not isinstance(text_configuration, dict):
            raise ValueError(
                f"Irodori text_encoder_config_json metadata must be an object: {path}"
            )
        else:
            pass
    return model_configuration, text_configuration


def precision_dtype(precision: str, device: torch.device) -> torch.dtype:
    normalized_precision = precision.strip().lower()
    if normalized_precision == "fp32":
        return torch.float32
    elif normalized_precision == "bf16":
        if device.type not in ("cuda", "xpu"):
            raise ValueError("Irodori bf16 precision requires a CUDA or XPU device")
        else:
            pass
        return torch.bfloat16
    else:
        raise ValueError(f"Unsupported Irodori precision: {precision}")


def tokenizer_source(checkpoint_path: Path, fallback_repo: str) -> tuple[str, bool]:
    tokenizer_path = checkpoint_path.parent / "tokenizer"
    if (tokenizer_path / "tokenizer_config.json").is_file():
        return str(tokenizer_path), True
    else:
        return fallback_repo, False


def load_model(
    checkpoint_path: Path,
    device: torch.device,
    model_precision: str,
    codec_repo: str,
    codec_precision: str,
) -> tuple[
    ModelConfig,
    TextToLatentRFDiT,
    PretrainedTextTokenizer,
    PretrainedTextTokenizer | None,
    DACVAECodec,
    int,
    int,
    float,
]:
    raw_configuration, text_encoder_configuration = checkpoint_configuration(
        checkpoint_path
    )
    inference_configuration: dict[str, object] = {}
    for key in ("max_text_len", "max_caption_len", "ref_max_seconds"):
        value = raw_configuration.pop(key, None)
        if value is not None:
            inference_configuration[key] = value
        else:
            pass
    model_configuration = merge_dataclass_overrides(
        ModelConfig(), raw_configuration, section="Irodori checkpoint model_config"
    )
    model = TextToLatentRFDiT(
        model_configuration,
        pretrained_backbone_config=text_encoder_configuration,
        load_pretrained_backbone_weights=(
            not model_configuration.use_pretrained_text_encoder
        ),
    )
    model_state = load_file(str(checkpoint_path), device="cpu")
    model.load_state_dict(
        model_state,
        assign=model_configuration.use_pretrained_text_encoder,
    )
    model_dtype = precision_dtype(model_precision, device)
    model = model.to(device=device, dtype=model_dtype)
    model.eval()

    text_tokenizer_path, text_tokenizer_is_local = tokenizer_source(
        checkpoint_path, model_configuration.text_tokenizer_repo
    )
    tokenizer = PretrainedTextTokenizer.from_pretrained(
        text_tokenizer_path,
        add_bos=bool(model_configuration.text_add_bos),
        local_files_only=text_tokenizer_is_local,
        revision=(
            None
            if text_tokenizer_is_local
            else model_configuration.text_encoder_revision
        ),
    )
    if (
        not model_configuration.use_pretrained_text_encoder
        and tokenizer.vocab_size != model_configuration.text_vocab_size
    ):
        raise ValueError(
            "Irodori text tokenizer vocabulary size does not match the checkpoint"
        )
    else:
        pass

    caption_tokenizer = None
    if model_configuration.use_caption_condition:
        caption_tokenizer_path, caption_tokenizer_is_local = tokenizer_source(
            checkpoint_path, model_configuration.caption_tokenizer_repo_resolved
        )
        caption_tokenizer = PretrainedTextTokenizer.from_pretrained(
            caption_tokenizer_path,
            add_bos=model_configuration.caption_add_bos_resolved,
            local_files_only=caption_tokenizer_is_local,
            revision=(
                None
                if caption_tokenizer_is_local
                else model_configuration.text_encoder_revision
            ),
        )
    else:
        pass

    codec_device = str(device)
    codec_dtype = precision_dtype(codec_precision, device)
    codec = DACVAECodec.load(
        repo_id=codec_repo,
        device=codec_device,
        dtype=codec_dtype,
        deterministic_encode=True,
        deterministic_decode=True,
    )
    if model_configuration.latent_dim != codec.latent_dim:
        raise ValueError(
            "Irodori checkpoint latent dimension does not match "
            "the selected DACVAE codec"
        )
    else:
        pass
    default_text_max_len = int(inference_configuration.get("max_text_len", 256))
    default_caption_max_len = int(
        inference_configuration.get("max_caption_len", default_text_max_len)
    )
    default_max_ref_seconds = float(
        inference_configuration.get("ref_max_seconds", 30.0)
    )
    return (
        model_configuration,
        model,
        tokenizer,
        caption_tokenizer,
        codec,
        default_text_max_len,
        default_caption_max_len,
        default_max_ref_seconds,
    )


def reference_paths(
    request: IrodoriSynthesisRequest,
    temporary_directory: str,
) -> list[str]:
    audio_paths: list[str] = []
    for reference_index, reference in enumerate(request.references):
        if reference.audio_path is not None:
            audio_paths.append(reference.audio_path)
        else:
            pass
        if reference.audio_base64 is not None:
            extension = mimetypes.guess_extension(reference.media_type) or ".wav"
            audio_path = (
                Path(temporary_directory)
                / f"reference-{reference_index}{extension}"
            )
            try:
                audio_bytes = base64.b64decode(reference.audio_base64, validate=True)
            except binascii.Error as error:
                raise ValueError(
                    "Irodori reference audio is not valid base64"
                ) from error
            audio_path.write_bytes(audio_bytes)
            audio_paths.append(str(audio_path))
        else:
            pass
    return audio_paths


def prepare_reference(
    audio_paths: list[str],
    model_configuration: ModelConfig,
    model: TextToLatentRFDiT,
    codec: DACVAECodec,
    max_ref_seconds: float,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    if not model_configuration.use_speaker_condition_resolved:
        return None, None
    else:
        pass
    if not audio_paths:
        reference_length = max(1, int(model_configuration.speaker_patch_size))
        reference_latent = torch.zeros(
            (1, reference_length, model_configuration.patched_latent_dim),
            device=model.device,
            dtype=model.dtype,
        )
        reference_mask = torch.zeros(
            (1, reference_length), device=model.device, dtype=torch.bool
        )
        return reference_latent, reference_mask
    else:
        pass

    latent_pieces = [codec.encode_file(audio_path) for audio_path in audio_paths]
    reference_latent = torch.cat(latent_pieces, dim=1)
    hop_length = int(codec.model.hop_length)
    max_reference_steps = max(
        1,
        math.ceil(max_ref_seconds * codec.sample_rate / hop_length),
    )
    reference_latent = reference_latent[:, :max_reference_steps]
    reference_latent = patchify_latent(
        reference_latent, model_configuration.latent_patch_size
    ).to(device=model.device, dtype=model.dtype)
    if reference_latent.shape[1] == 0:
        raise ValueError("Irodori reference audio produced an empty latent sequence")
    else:
        pass
    reference_mask = torch.ones(
        (1, reference_latent.shape[1]), device=model.device, dtype=torch.bool
    )
    return reference_latent, reference_mask


def find_flattening_point(
    latent: torch.Tensor,
    window_size: int = 20,
    std_threshold: float = 0.05,
    mean_threshold: float = 0.1,
) -> int:
    if latent.ndim != 2:
        raise ValueError(
            f"Expected Irodori latent shape (T, D), got {tuple(latent.shape)}"
        )
    else:
        pass
    total_steps = int(latent.shape[0])
    if total_steps <= 0 or window_size <= 0:
        return total_steps
    else:
        pass
    padding = torch.zeros(
        (window_size, latent.shape[1]), device=latent.device, dtype=latent.dtype
    )
    padded_latent = torch.cat([latent, padding], dim=0)
    for index in range(padded_latent.shape[0] - window_size):
        window = padded_latent[index : index + window_size]
        if (
            window.std(unbiased=False) < std_threshold
            and torch.abs(window.mean()) < mean_threshold
        ):
            return index
        else:
            pass
    return total_steps


def synthesize(
    payload: StagePayload,
    *,
    model_configuration: ModelConfig,
    model: TextToLatentRFDiT,
    tokenizer: PretrainedTextTokenizer,
    caption_tokenizer: PretrainedTextTokenizer | None,
    codec: DACVAECodec,
    max_text_len: int,
    max_caption_len: int,
    max_ref_seconds: float,
    max_seconds: float,
) -> StagePayload:
    request = build_irodori_request(payload)
    normalized_text = normalize_text(request.text).strip()
    if not normalized_text:
        raise ValueError("Irodori text became empty after normalization")
    else:
        pass
    number_of_steps = 40 if request.num_steps is None else request.num_steps
    if number_of_steps <= 0:
        raise ValueError("Irodori num_steps must be positive")
    else:
        pass
    duration_scale = 1.0 if request.duration_scale is None else request.duration_scale
    if duration_scale <= 0:
        raise ValueError("Irodori duration_scale must be positive")
    else:
        pass
    seed = secrets.randbits(63) if request.seed is None else request.seed

    text_ids, text_mask = tokenizer.batch_encode(
        [normalized_text], max_length=max_text_len
    )
    text_ids = text_ids.to(model.device)
    text_mask = text_mask.to(model.device)
    caption_ids = None
    caption_mask = None
    if model_configuration.use_caption_condition:
        if caption_tokenizer is None:
            raise RuntimeError("Irodori caption tokenizer was not loaded")
        else:
            pass
        caption_text = "" if request.caption is None else request.caption.strip()
        caption_ids, caption_mask = caption_tokenizer.batch_encode(
            [caption_text], max_length=max_caption_len
        )
        if not caption_text:
            caption_mask.zero_()
        else:
            pass
        caption_ids = caption_ids.to(model.device)
        caption_mask = caption_mask.to(model.device)
    else:
        pass

    with TemporaryDirectory(prefix="sglang-omni-irodori-") as temporary_directory:
        audio_paths = reference_paths(request, temporary_directory)
        reference_latent, reference_mask = prepare_reference(
            audio_paths,
            model_configuration,
            model,
            codec,
            max_ref_seconds,
        )
        with torch.inference_mode():
            if request.seconds is not None:
                if request.seconds <= 0:
                    raise ValueError("Irodori seconds must be positive")
                else:
                    pass
                selected_seconds = min(max_seconds, max(0.5, request.seconds))
                target_samples = max(1, int(selected_seconds * codec.sample_rate))
                latent_steps = math.ceil(target_samples / int(codec.model.hop_length))
            elif model_configuration.use_duration_predictor:
                has_speaker = torch.zeros((1,), dtype=torch.bool, device=model.device)
                if reference_mask is not None:
                    has_speaker = reference_mask.any(dim=1)
                else:
                    pass
                duration_features = build_duration_features(
                    [normalized_text],
                    token_counts=text_mask.sum(dim=1),
                    max_text_len=max_text_len,
                    has_speaker=has_speaker,
                ).to(model.device)
                (
                    duration_text_state,
                    duration_text_mask,
                    duration_speaker_state,
                    duration_speaker_mask,
                    duration_caption_state,
                    duration_caption_mask,
                ) = model.encode_conditions(
                    text_input_ids=text_ids,
                    text_mask=text_mask,
                    ref_latent=reference_latent,
                    ref_mask=reference_mask,
                    caption_input_ids=caption_ids,
                    caption_mask=caption_mask,
                )
                predicted_log_frames = model.predict_duration_log_frames(
                    text_state=duration_text_state,
                    text_mask=duration_text_mask,
                    speaker_state=duration_speaker_state,
                    speaker_mask=duration_speaker_mask,
                    caption_state=duration_caption_state,
                    caption_mask=duration_caption_mask,
                    duration_features=duration_features,
                    has_speaker=has_speaker,
                    has_caption=(
                        torch.full(
                            (1,),
                            request.caption is not None,
                            dtype=torch.bool,
                            device=model.device,
                        )
                        if model_configuration.use_caption_condition
                        else None
                    ),
                )
                predicted_frames = float(
                    torch.expm1(predicted_log_frames).mean().item()
                )
                min_frames = max(
                    1,
                    math.ceil(0.5 * codec.sample_rate / int(codec.model.hop_length)),
                )
                max_frames = max(
                    1,
                    math.floor(
                        max_seconds
                        * codec.sample_rate
                        / int(codec.model.hop_length)
                    ),
                )
                latent_steps = min(
                    max_frames,
                    max(min_frames, round(predicted_frames * duration_scale)),
                )
                target_samples = latent_steps * int(codec.model.hop_length)
            else:
                selected_seconds = min(max_seconds, 30.0)
                target_samples = int(selected_seconds * codec.sample_rate)
                latent_steps = math.ceil(target_samples / int(codec.model.hop_length))

            patch_size = model_configuration.latent_patch_size
            patched_steps = math.ceil(latent_steps / patch_size)
            generated_patches = sample_euler_rf_cfg(
                model=model,
                text_input_ids=text_ids,
                text_mask=text_mask,
                ref_latent=reference_latent,
                ref_mask=reference_mask,
                sequence_length=patched_steps,
                caption_input_ids=caption_ids,
                caption_mask=caption_mask,
                num_steps=number_of_steps,
                cfg_scale_text=(
                    3.0 if request.cfg_scale_text is None else request.cfg_scale_text
                ),
                cfg_scale_caption=(
                    3.0
                    if request.cfg_scale_caption is None
                    else request.cfg_scale_caption
                ),
                cfg_scale_speaker=(
                    5.0
                    if request.cfg_scale_speaker is None
                    else request.cfg_scale_speaker
                ),
                cfg_guidance_mode="independent",
                seed=int(seed),
            )
            generated_latent = unpatchify_latent(
                generated_patches,
                patch_size=patch_size,
                latent_dim=model_configuration.latent_dim,
            )[:, :latent_steps]
            flattening_step = find_flattening_point(generated_latent[0])
            if flattening_step > 0:
                target_samples = min(
                    target_samples, flattening_step * int(codec.model.hop_length)
                )
            else:
                pass
            generated_audio = codec.decode_latent(generated_latent)[
                0, :, :target_samples
            ]

    payload.data.update(
        audio_waveform_payload(
            generated_audio,
            sample_rate=codec.sample_rate,
            modality="audio",
            source_hint="Irodori-TTS",
        )
    )
    return payload



def synthesize_batch(
    payloads: list[StagePayload],
    *,
    model_configuration: ModelConfig,
    model: TextToLatentRFDiT,
    tokenizer: PretrainedTextTokenizer,
    caption_tokenizer: PretrainedTextTokenizer | None,
    codec: DACVAECodec,
    max_text_len: int,
    max_caption_len: int,
    max_ref_seconds: float,
    max_seconds: float,
    max_batch_size: int,
    max_batch_tokens: int,
) -> list[StagePayload | BaseException]:
    """Batch compatible Irodori requests in one rectified-flow and codec pass."""
    if not payloads:
        return []
    else:
        pass

    results: list[StagePayload | BaseException | None] = [None] * len(payloads)
    items: list[dict[str, object]] = []

    with ExitStack() as temporary_directories:
        for request_index, payload in enumerate(payloads):
            try:
                request = build_irodori_request(payload)
                normalized_text = normalize_text(request.text).strip()
                if not normalized_text:
                    raise ValueError("Irodori text became empty after normalization")
                else:
                    pass
                number_of_steps = 40 if request.num_steps is None else request.num_steps
                if number_of_steps <= 0:
                    raise ValueError("Irodori num_steps must be positive")
                else:
                    pass
                duration_scale = (
                    1.0 if request.duration_scale is None else request.duration_scale
                )
                if duration_scale <= 0:
                    raise ValueError("Irodori duration_scale must be positive")
                else:
                    pass
                if request.seconds is not None and request.seconds <= 0:
                    raise ValueError("Irodori seconds must be positive")
                else:
                    pass
                if (
                    model_configuration.use_caption_condition
                    and caption_tokenizer is None
                ):
                    raise RuntimeError("Irodori caption tokenizer was not loaded")
                else:
                    pass

                temporary_directory = temporary_directories.enter_context(
                    TemporaryDirectory(prefix="sglang-omni-irodori-")
                )
                audio_paths = reference_paths(request, temporary_directory)
                reference_latent, reference_mask = prepare_reference(
                    audio_paths,
                    model_configuration,
                    model,
                    codec,
                    max_ref_seconds,
                )
                items.append(
                    {
                        "request_index": request_index,
                        "payload": payload,
                        "request": request,
                        "normalized_text": normalized_text,
                        "number_of_steps": int(number_of_steps),
                        "duration_scale": float(duration_scale),
                        "seed": (
                            secrets.randbits(63)
                            if request.seed is None
                            else int(request.seed)
                        ),
                        "cfg_scales": (
                            3.0
                            if request.cfg_scale_text is None
                            else float(request.cfg_scale_text),
                            3.0
                            if request.cfg_scale_caption is None
                            else float(request.cfg_scale_caption),
                            5.0
                            if request.cfg_scale_speaker is None
                            else float(request.cfg_scale_speaker),
                        ),
                        "reference_latent": reference_latent,
                        "reference_mask": reference_mask,
                    }
                )
            except Exception as exc:
                results[request_index] = exc
            else:
                pass

    if not items:
        return [
            result
            if result is not None
            else RuntimeError("Irodori batch did not produce a result")
            for result in results
        ]
    else:
        pass

    def run_individually(retry_items: list[dict[str, object]]) -> None:
        for item in retry_items:
            request_index = int(item["request_index"])
            try:
                results[request_index] = synthesize(
                    item["payload"],
                    model_configuration=model_configuration,
                    model=model,
                    tokenizer=tokenizer,
                    caption_tokenizer=caption_tokenizer,
                    codec=codec,
                    max_text_len=max_text_len,
                    max_caption_len=max_caption_len,
                    max_ref_seconds=max_ref_seconds,
                    max_seconds=max_seconds,
                )
            except Exception as exc:
                results[request_index] = exc
            else:
                pass

    for batch_row, item in enumerate(items):
        item["batch_row"] = batch_row

    try:
        text_ids, text_mask = tokenizer.batch_encode(
            [str(item["normalized_text"]) for item in items],
            max_length=max_text_len,
        )
        text_ids = text_ids.to(model.device)
        text_mask = text_mask.to(model.device)
        caption_ids = None
        caption_mask = None
        if model_configuration.use_caption_condition:
            if caption_tokenizer is None:
                raise RuntimeError("Irodori caption tokenizer was not loaded")
            else:
                pass
            caption_texts = [
                "" if item["request"].caption is None else item["request"].caption.strip()
                for item in items
            ]
            caption_ids, caption_mask = caption_tokenizer.batch_encode(
                caption_texts,
                max_length=max_caption_len,
            )
            for row, caption_text in enumerate(caption_texts):
                if not caption_text:
                    caption_mask[row].zero_()
                else:
                    pass
            caption_ids = caption_ids.to(model.device)
            caption_mask = caption_mask.to(model.device)
        else:
            pass

        reference_latent = None
        reference_mask = None
        if model_configuration.use_speaker_condition_resolved:
            reference_parts = [item["reference_latent"] for item in items]
            reference_masks = [item["reference_mask"] for item in items]
            if any(part is None for part in reference_parts) or any(
                mask is None for mask in reference_masks
            ):
                raise RuntimeError("Irodori speaker conditioning is missing references")
            else:
                pass
            max_reference_length = max(int(part.shape[1]) for part in reference_parts)
            reference_latent = torch.zeros(
                (
                    len(items),
                    max_reference_length,
                    model_configuration.patched_latent_dim,
                ),
                device=model.device,
                dtype=model.dtype,
            )
            reference_mask = torch.zeros(
                (len(items), max_reference_length),
                device=model.device,
                dtype=torch.bool,
            )
            for row, (part, mask) in enumerate(
                zip(reference_parts, reference_masks, strict=True)
            ):
                length = int(part.shape[1])
                reference_latent[row, :length] = part[0]
                reference_mask[row, :length] = mask[0]
        else:
            pass

        predicted_frames: list[float | None] = [None] * len(items)
        needs_duration = model_configuration.use_duration_predictor and any(
            item["request"].seconds is None for item in items
        )
        if needs_duration:
            if model_configuration.use_speaker_condition_resolved:
                has_speaker = reference_mask.any(dim=1)
            else:
                has_speaker = torch.zeros(
                    (len(items),), device=model.device, dtype=torch.bool
                )
            duration_features = build_duration_features(
                [str(item["normalized_text"]) for item in items],
                token_counts=text_mask.sum(dim=1),
                max_text_len=max_text_len,
                has_speaker=has_speaker,
            ).to(model.device)
            with torch.inference_mode():
                (
                    duration_text_state,
                    duration_text_mask,
                    duration_speaker_state,
                    duration_speaker_mask,
                    duration_caption_state,
                    duration_caption_mask,
                ) = model.encode_conditions(
                    text_input_ids=text_ids,
                    text_mask=text_mask,
                    ref_latent=reference_latent,
                    ref_mask=reference_mask,
                    caption_input_ids=caption_ids,
                    caption_mask=caption_mask,
                )
                has_caption = (
                    torch.tensor(
                        [
                            item["request"].caption is not None
                            for item in items
                        ],
                        device=model.device,
                        dtype=torch.bool,
                    )
                    if model_configuration.use_caption_condition
                    else None
                )
                duration_prediction = model.predict_duration_log_frames(
                    text_state=duration_text_state,
                    text_mask=duration_text_mask,
                    speaker_state=duration_speaker_state,
                    speaker_mask=duration_speaker_mask,
                    caption_state=duration_caption_state,
                    caption_mask=duration_caption_mask,
                    duration_features=duration_features,
                    has_speaker=has_speaker,
                    has_caption=has_caption,
                )
                predicted_values = (
                    torch.expm1(duration_prediction.float())
                    .reshape(len(items), -1)
                    .mean(dim=1)
                    .tolist()
                )
            for row, predicted_value in enumerate(predicted_values):
                predicted_frames[row] = float(predicted_value)
        else:
            pass
    except Exception:
        logger.exception(
            "Irodori batch preparation failed; retrying %d items individually",
            len(items),
        )
        run_individually(items)
        return [
            result
            if result is not None
            else RuntimeError("Irodori batch fallback did not produce a result")
            for result in results
        ]

    usable_items: list[dict[str, object]] = []
    for item in items:
        request = item["request"]
        row = int(item["batch_row"])
        try:
            if request.seconds is not None:
                selected_seconds = min(max_seconds, max(0.5, request.seconds))
                target_samples = max(1, int(selected_seconds * codec.sample_rate))
                latent_steps = math.ceil(
                    target_samples / int(codec.model.hop_length)
                )
            elif model_configuration.use_duration_predictor:
                value = predicted_frames[row]
                if value is None:
                    raise RuntimeError("Irodori duration prediction is missing")
                else:
                    pass
                min_frames = max(
                    1,
                    math.ceil(
                        0.5 * codec.sample_rate / int(codec.model.hop_length)
                    ),
                )
                max_frames = max(
                    1,
                    math.floor(
                        max_seconds
                        * codec.sample_rate
                        / int(codec.model.hop_length)
                    ),
                )
                latent_steps = min(
                    max_frames,
                    max(
                        min_frames,
                        round(float(value) * float(item["duration_scale"])),
                    ),
                )
                target_samples = latent_steps * int(codec.model.hop_length)
            else:
                selected_seconds = min(max_seconds, 30.0)
                target_samples = int(selected_seconds * codec.sample_rate)
                latent_steps = math.ceil(
                    target_samples / int(codec.model.hop_length)
                )
            item["target_samples"] = int(target_samples)
            item["latent_steps"] = int(latent_steps)
            item["patched_steps"] = math.ceil(
                int(latent_steps) / model_configuration.latent_patch_size
            )
            ref = item["reference_latent"]
            item["reference_steps"] = 0 if ref is None else int(ref.shape[1])
            usable_items.append(item)
        except Exception as exc:
            results[int(item["request_index"])] = exc
        else:
            pass

    parameter_groups: dict[tuple[object, ...], list[dict[str, object]]] = {}
    for item in usable_items:
        key = (item["number_of_steps"], *item["cfg_scales"])
        parameter_groups.setdefault(key, []).append(item)

    max_batch_size = max(int(max_batch_size), 1)
    max_batch_tokens = max(int(max_batch_tokens), 1)

    def split_by_work(items_to_split: list[dict[str, object]]) -> list[list[dict[str, object]]]:
        sorted_items = sorted(
            items_to_split,
            key=lambda item: int(item["patched_steps"])
            + int(item["reference_steps"]),
        )
        groups: list[list[dict[str, object]]] = []
        current: list[dict[str, object]] = []
        current_min_length = 0
        current_max_length = 0
        for item in sorted_items:
            work_length = int(item["patched_steps"]) + int(item["reference_steps"])
            next_max_length = max(current_max_length, work_length)
            next_cost = (len(current) + 1) * next_max_length
            length_ratio_too_large = (
                bool(current)
                and next_max_length > max(1, current_min_length) * 2
            )
            if current and (
                len(current) >= max_batch_size
                or next_cost > max_batch_tokens
                or length_ratio_too_large
            ):
                groups.append(current)
                current = []
                current_min_length = 0
                current_max_length = 0
            else:
                pass
            if not current:
                current_min_length = work_length
                current_max_length = work_length
            else:
                current_min_length = min(current_min_length, work_length)
                current_max_length = max(current_max_length, work_length)
            current.append(item)
        if current:
            groups.append(current)
        else:
            pass
        return groups

    def run_group(group: list[dict[str, object]]) -> None:
        batch_rows = [int(item["batch_row"]) for item in group]
        patched_lengths = [int(item["patched_steps"]) for item in group]
        max_patched_length = max(patched_lengths)
        logger.debug(
            "Irodori batched sampling requests=%d padded_latent_frames=%d",
            len(group),
            max_patched_length,
        )
        group_latent_mask = torch.zeros(
            (len(group), max_patched_length),
            device=model.device,
            dtype=torch.bool,
        )
        for row, length in enumerate(patched_lengths):
            group_latent_mask[row, :length] = True
        else:
            pass
        group_reference_latent = (
            None if reference_latent is None else reference_latent[batch_rows]
        )
        group_reference_mask = (
            None if reference_mask is None else reference_mask[batch_rows]
        )
        group_caption_ids = None if caption_ids is None else caption_ids[batch_rows]
        group_caption_mask = None if caption_mask is None else caption_mask[batch_rows]
        with torch.inference_mode():
            generated_patches = sample_euler_rf_cfg(
                model=model,
                text_input_ids=text_ids[batch_rows],
                text_mask=text_mask[batch_rows],
                ref_latent=group_reference_latent,
                ref_mask=group_reference_mask,
                sequence_length=max_patched_length,
                caption_input_ids=group_caption_ids,
                caption_mask=group_caption_mask,
                num_steps=int(group[0]["number_of_steps"]),
                cfg_scale_text=float(group[0]["cfg_scales"][0]),
                cfg_scale_caption=float(group[0]["cfg_scales"][1]),
                cfg_scale_speaker=float(group[0]["cfg_scales"][2]),
                cfg_guidance_mode="independent",
                seed=[int(item["seed"]) for item in group],
                latent_mask=group_latent_mask,
            )
            generated_latent = unpatchify_latent(
                generated_patches,
                patch_size=model_configuration.latent_patch_size,
                latent_dim=model_configuration.latent_dim,
            )
            max_latent_length = max(int(item["latent_steps"]) for item in group)
            generated_latent = generated_latent[:, :max_latent_length]
            flattening_steps: list[int] = []
            for row, item in enumerate(group):
                latent_length = int(item["latent_steps"])
                generated_latent[row, latent_length:] = 0
                flattening_steps.append(
                    find_flattening_point(generated_latent[row, :latent_length])
                )
            else:
                pass
            generated_audio = codec.decode_latent(generated_latent)
            for row, item in enumerate(group):
                target_samples = int(item["target_samples"])
                flattening_step = flattening_steps[row]
                if flattening_step > 0:
                    target_samples = min(
                        target_samples,
                        flattening_step * int(codec.model.hop_length),
                    )
                else:
                    pass
                item["payload"].data.update(
                    audio_waveform_payload(
                        generated_audio[row, :, :target_samples],
                        sample_rate=codec.sample_rate,
                        modality="audio",
                        source_hint="Irodori-TTS",
                    )
                )
                results[int(item["request_index"])] = item["payload"]

    for parameter_group in parameter_groups.values():
        for group in split_by_work(parameter_group):
            try:
                run_group(group)
            except Exception:
                logger.exception(
                    "Irodori batched synthesis failed for batch size %d; "
                    "retrying each request individually",
                    len(group),
                )
                if model.device.type == "cuda":
                    torch.cuda.empty_cache()
                else:
                    pass
                run_individually(group)

    return [
        result
        if result is not None
        else RuntimeError("Irodori batch did not produce a result")
        for result in results
    ]


def create_irodori_executor(
    model_path: str,
    *,
    device: str | None = None,
    gpu_id: int | None = None,
    model_precision: str,
    codec_precision: str,
    codec_repo: str,
    max_seconds: float,
    max_batch_size: int,
    max_batch_wait_ms: int,
    max_batch_tokens: int,
) -> SimpleScheduler[StagePayload, StagePayload]:
    resolved_device = torch.device(str(resolve_concrete_device(device, gpu_id)))
    resolved_checkpoint = checkpoint_file(model_path)
    (
        model_configuration,
        model,
        tokenizer,
        caption_tokenizer,
        codec,
        max_text_len,
        max_caption_len,
        max_ref_seconds,
    ) = load_model(
        resolved_checkpoint,
        resolved_device,
        model_precision,
        codec_repo,
        codec_precision,
    )

    def run_synthesis(payload: StagePayload) -> StagePayload:
        return synthesize(
            payload,
            model_configuration=model_configuration,
            model=model,
            tokenizer=tokenizer,
            caption_tokenizer=caption_tokenizer,
            codec=codec,
            max_text_len=max_text_len,
            max_caption_len=max_caption_len,
            max_ref_seconds=max_ref_seconds,
            max_seconds=max_seconds,
        )

    def run_synthesis_batch(
        payloads: list[StagePayload],
    ) -> list[StagePayload | BaseException]:
        return synthesize_batch(
            payloads,
            model_configuration=model_configuration,
            model=model,
            tokenizer=tokenizer,
            caption_tokenizer=caption_tokenizer,
            codec=codec,
            max_text_len=max_text_len,
            max_caption_len=max_caption_len,
            max_ref_seconds=max_ref_seconds,
            max_seconds=max_seconds,
            max_batch_size=max_batch_size,
            max_batch_tokens=max_batch_tokens,
        )

    logger.info(
        "Loaded native Irodori-TTS model from %s on %s with model precision %s "
        "and codec precision %s; request batching is enabled (max batch size %d)",
        resolved_checkpoint,
        resolved_device,
        model_precision,
        codec_precision,
        max_batch_size,
    )
    return SimpleScheduler(
        run_synthesis,
        batch_compute_fn=run_synthesis_batch,
        max_batch_size=max_batch_size,
        max_batch_wait_ms=max_batch_wait_ms,
        max_concurrency=1,
    )
