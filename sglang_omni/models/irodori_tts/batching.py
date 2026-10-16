# SPDX-License-Identifier: Apache-2.0
"""Length-aware batching and per-request error isolation for Irodori-TTS."""

import logging
import math
import secrets
from contextlib import ExitStack
from tempfile import TemporaryDirectory

import torch

from sglang_omni.models.irodori_tts import synthesis
from sglang_omni.models.irodori_tts.codec import DACVAECodec, unpatchify_latent
from sglang_omni.models.irodori_tts.dit import TextToLatentRFDiT
from sglang_omni.models.irodori_tts.flow_matching import sample_euler_rf_cfg
from sglang_omni.models.irodori_tts.model_config import ModelConfig
from sglang_omni.models.irodori_tts.request_builders import build_irodori_request
from sglang_omni.models.irodori_tts.text_processing import (
    build_duration_features,
    normalize_text,
)
from sglang_omni.models.irodori_tts.tokenizer import PretrainedTextTokenizer
from sglang_omni.proto import StagePayload
from sglang_omni.utils.audio_payload import audio_waveform_payload

logger = logging.getLogger(__name__)


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
                speed = 1.0 if request.speed is None else request.speed
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
                audio_paths = synthesis.reference_paths(request, temporary_directory)
                reference_latent, reference_mask = synthesis.prepare_reference(
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
                        "speed": float(speed),
                        "seed": (
                            secrets.randbits(63)
                            if request.seed is None
                            else int(request.seed)
                        ),
                        "cfg_scales": (
                            (
                                3.0
                                if request.cfg_scale_text is None
                                else float(request.cfg_scale_text)
                            ),
                            (
                                3.0
                                if request.cfg_scale_caption is None
                                else float(request.cfg_scale_caption)
                            ),
                            (
                                5.0
                                if request.cfg_scale_speaker is None
                                else float(request.cfg_scale_speaker)
                            ),
                        ),
                        "reference_latent": reference_latent,
                        "reference_mask": reference_mask,
                    }
                )
            except Exception as exc:
                results[request_index] = exc

    if not items:
        return [
            (
                result
                if result is not None
                else RuntimeError("Irodori batch did not produce a result")
            )
            for result in results
        ]
    else:
        pass

    def run_individually(retry_items: list[dict[str, object]]) -> None:
        for item in retry_items:
            request_index = int(item["request_index"])
            try:
                results[request_index] = synthesis.synthesize(
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
                (
                    ""
                    if item["request"].caption is None
                    else item["request"].caption.strip()
                )
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
                        [item["request"].caption is not None for item in items],
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
            (
                result
                if result is not None
                else RuntimeError("Irodori batch fallback did not produce a result")
            )
            for result in results
        ]

    usable_items: list[dict[str, object]] = []
    for item in items:
        request = item["request"]
        row = int(item["batch_row"])
        try:
            if request.seconds is not None:
                selected_seconds = min(
                    max_seconds, max(0.5, request.seconds / float(item["speed"]))
                )
                target_samples = max(1, int(selected_seconds * codec.sample_rate))
                latent_steps = math.ceil(target_samples / int(codec.model.hop_length))
            elif model_configuration.use_duration_predictor:
                value = predicted_frames[row]
                if value is None:
                    raise RuntimeError("Irodori duration prediction is missing")
                else:
                    pass
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
                    max(
                        min_frames,
                        round(
                            float(value)
                            * float(item["duration_scale"])
                            / float(item["speed"])
                        ),
                    ),
                )
                target_samples = latent_steps * int(codec.model.hop_length)
            else:
                selected_seconds = min(max_seconds, 30.0 / float(item["speed"]))
                target_samples = int(selected_seconds * codec.sample_rate)
                latent_steps = math.ceil(target_samples / int(codec.model.hop_length))
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

    parameter_groups: dict[tuple[object, ...], list[dict[str, object]]] = {}
    for item in usable_items:
        key = (item["number_of_steps"], *item["cfg_scales"])
        parameter_groups.setdefault(key, []).append(item)

    max_batch_size = max(int(max_batch_size), 1)
    max_batch_tokens = max(int(max_batch_tokens), 1)

    def split_by_work(
        items_to_split: list[dict[str, object]],
    ) -> list[list[dict[str, object]]]:
        sorted_items = sorted(
            items_to_split,
            key=lambda item: int(item["patched_steps"]) + int(item["reference_steps"]),
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
                bool(current) and next_max_length > max(1, current_min_length) * 2
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
        max_reference_length = max(int(item["reference_steps"]) for item in group)
        group_reference_latent = (
            None
            if reference_latent is None
            else reference_latent[batch_rows, :max_reference_length]
        )
        group_reference_mask = (
            None
            if reference_mask is None
            else reference_mask[batch_rows, :max_reference_length]
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
            latent_lengths = [int(item["latent_steps"]) for item in group]
            valid_latent = (
                torch.arange(generated_latent.shape[1], device=generated_latent.device)[
                    None, :
                ]
                < torch.tensor(latent_lengths, device=generated_latent.device)[:, None]
            )
            generated_latent.masked_fill_(~valid_latent[:, :, None], 0)
            flattening_steps = synthesis.find_flattening_points(
                generated_latent,
                latent_lengths,
            )
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
        (
            result
            if result is not None
            else RuntimeError("Irodori batch did not produce a result")
        )
        for result in results
    ]
