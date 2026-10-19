# SPDX-License-Identifier: Apache-2.0
"""Single-request Irodori conditioning and synthesis helpers."""

import base64
import binascii
import math
import mimetypes
import secrets
from collections.abc import Sequence
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from sglang_omni.models.irodori_tts.codec import (
    DACVAECodec,
    patchify_latent,
    unpatchify_latent,
)
from sglang_omni.models.irodori_tts.dit import TextToLatentRFDiT
from sglang_omni.models.irodori_tts.flow_matching import sample_euler_rf_cfg
from sglang_omni.models.irodori_tts.model_config import ModelConfig
from sglang_omni.models.irodori_tts.request_builders import (
    IrodoriSynthesisRequest,
    build_irodori_request,
)
from sglang_omni.models.irodori_tts.text_processing import (
    build_duration_features,
    normalize_text,
)
from sglang_omni.models.irodori_tts.tokenizer import PretrainedTextTokenizer
from sglang_omni.proto import StagePayload
from sglang_omni.utils.audio_payload import audio_waveform_payload


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
                Path(temporary_directory) / f"reference-{reference_index}{extension}"
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


def find_flattening_points(
    latent_batch: torch.Tensor,
    latent_lengths: Sequence[int],
    window_size: int = 20,
    std_threshold: float = 0.05,
    mean_threshold: float = 0.1,
) -> list[int]:
    """Find each row's first quiet window with a single device synchronization."""
    if latent_batch.ndim != 3:
        raise ValueError(
            f"Expected Irodori latent batch shape (B, T, D), got {tuple(latent_batch.shape)}"
        )
    else:
        pass
    batch_size, total_steps, latent_dim = latent_batch.shape
    if len(latent_lengths) != batch_size:
        raise ValueError("Irodori latent lengths must match the batch size")
    else:
        pass
    lengths = torch.as_tensor(
        latent_lengths, device=latent_batch.device, dtype=torch.long
    ).clamp(min=0, max=total_steps)
    if total_steps == 0 or latent_dim == 0 or window_size <= 0:
        return [int(length) for length in lengths.tolist()]
    else:
        pass

    # Accumulate in fp32 to keep rolling statistics stable for bf16/fp16 latents.
    values = latent_batch.float()
    time_sum = values.sum(dim=-1)
    time_squared_sum = values.square().sum(dim=-1)
    zero_column = torch.zeros((batch_size, 1), device=values.device, dtype=values.dtype)
    prefix_sum = torch.cat((zero_column, time_sum.cumsum(dim=1)), dim=1)
    prefix_squared_sum = torch.cat((zero_column, time_squared_sum.cumsum(dim=1)), dim=1)

    starts = torch.arange(total_steps, device=values.device).expand(batch_size, -1)
    ends = torch.minimum(starts + window_size, lengths[:, None])
    window_sum = prefix_sum.gather(1, ends) - prefix_sum.gather(1, starts)
    window_squared_sum = prefix_squared_sum.gather(1, ends) - prefix_squared_sum.gather(
        1, starts
    )
    window_lengths = (ends - starts).clamp(min=0).to(values.dtype)
    element_count = (window_lengths * latent_dim).clamp(min=1.0)
    window_mean = window_sum / element_count
    window_variance = (window_squared_sum / element_count - window_mean.square()).clamp(
        min=0.0
    )
    matches = (
        (starts < lengths[:, None])
        & (window_variance.sqrt() < std_threshold)
        & (window_mean.abs() < mean_threshold)
    )
    sentinel = torch.full_like(starts, total_steps)
    first_match = torch.where(matches, starts, sentinel).amin(dim=1)
    # Transfer results and lengths together so a batch causes one host sync.
    packed_results = torch.stack((first_match, lengths), dim=1).tolist()
    return [
        int(match) if int(match) < int(length) else int(length)
        for match, length in packed_results
    ]


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
    return find_flattening_points(
        latent.unsqueeze(0),
        [int(latent.shape[0])],
        window_size=window_size,
        std_threshold=std_threshold,
        mean_threshold=mean_threshold,
    )[0]


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
    speed = 1.0 if request.speed is None else request.speed
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
                selected_seconds = min(max_seconds, max(0.5, request.seconds / speed))
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
                        max_seconds * codec.sample_rate / int(codec.model.hop_length)
                    ),
                )
                latent_steps = min(
                    max_frames,
                    max(min_frames, round(predicted_frames * duration_scale / speed)),
                )
                target_samples = latent_steps * int(codec.model.hop_length)
            else:
                selected_seconds = min(max_seconds, 30.0 / speed)
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
