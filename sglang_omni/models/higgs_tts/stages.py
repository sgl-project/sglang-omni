# SPDX-License-Identifier: Apache-2.0
"""Stage factories for the Higgs TTS pipeline.

Pipeline shape::

    preprocessing → audio_encoder → tts_engine → vocoder

- ``create_preprocessing_executor``: text tokenize + (if raw audio path)
  load waveform; fast path also delay-encodes client-supplied
  ``reference_codes`` and builds the prompt. Returns a
  :class:`ThreadedSimpleScheduler` for CPU-heavy work.
- ``create_audio_encoder_executor``: GPU codec encode for the raw-audio
  path → delayed ref codes + prompt assembly. No-op on the fast path.
- ``create_sglang_tts_engine_executor``: runs :class:`HiggsTTSModel` under
  sglang's worker; the model runner computes the fused multi-codebook
  embedding inline in prefill from ``reference_codes_delayed`` and overlays
  it at ``-100`` placeholder positions. Returns a :class:`OmniScheduler`.
- ``create_vocoder_executor``: creates the Higgs vocoder scheduler, preserving
  batched non-streaming decode and incremental streaming audio chunks.
"""

from __future__ import annotations

import base64
import logging
import os
import threading
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torchaudio.functional as F_audio
from tokenizers import Tokenizer
from transformers import PreTrainedTokenizerFast

from sglang_omni.models.higgs_tts import fusion_reference
from sglang_omni.models.higgs_tts.payload_types import HiggsTtsState
from sglang_omni.models.higgs_tts.text_tokenizer import HiggsTokenizerAdapter
from sglang_omni.models.higgs_tts.utils import (
    apply_delay_pattern,
    get_or_load_codec,
    load_audio_to_24k,
    resolve_checkpoint,
    to_codes_TN,
)
from sglang_omni.models.higgs_tts.vocoder_scheduler import (
    DEFAULT_HIGGS_INITIAL_CHUNK_FRAMES,
    DEFAULT_HIGGS_STREAM_FOLLOWUP_STRIDE,
    DEFAULT_HIGGS_STREAM_STRIDE,
    HiggsStreamingVocoderScheduler,
)

# _REF_PATH_HASH_MEMO is the shared memo object, re-exported so tests can
# reset it; the underscored alias keeps this module's historical API.
from sglang_omni.preprocessing.cache_key import _REF_PATH_HASH_MEMO  # noqa: F401
from sglang_omni.preprocessing.cache_key import hash_bytes, hash_media_item
from sglang_omni.preprocessing.cache_key import (
    reference_path_cache_key as _reference_path_cache_key,
)
from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.reference_encoder import (
    ReferenceEncodeService,
    TensorReferenceEncodeHook,
)
from sglang_omni.scheduling.simple_scheduler import SimpleScheduler
from sglang_omni.scheduling.speaker_cache import (
    SpeakerCacheKey,
    get_speaker_artifact_cache,
)
from sglang_omni.scheduling.stage_cache import StageOutputCache
from sglang_omni.scheduling.threaded_simple_scheduler import ThreadedSimpleScheduler
from sglang_omni.utils.device import resolve_device_spec

logger = logging.getLogger(__name__)


# The codec emits 25 frames/s (hop 960 @ 24 kHz); a historical "75 Hz" note
# was a conservative cap, not the real rate. Each reference frame becomes one
# prompt placeholder token, and the engine admits
# ``prompt + max_new_tokens <= context_length - 1`` (4095, see
# ``HiggsTtsEngineBuilder.context_length``). With the default
# max_new_tokens=2048 the prompt budget is ~2047 tokens, so references past
# ~80 s are guaranteed to be rejected by the engine AFTER paying the full
# download/decode/encode cost — reject them up front instead. Chunked prefill
# of the multi-codebook prompt is unsafe (sampler state machine has no
# rollback); the context budget keeps every admitted prompt well under
# chunked_prefill_size.
_CODEC_FRAMES_PER_SEC = 25
_MAX_REF_AUDIO_SEC = 80
_ENGINE_CONTEXT_BUDGET = 4095

# Mirrors the tts_engine stage's default max_new_tokens cap (config.py). The
# engine clamps every request to min(requested, cap) in request_builders just
# before admission, so budget pre-checks must use the clamped value — checking
# the raw client value would reject requests the cap would have saved. A
# deployment that RAISES the engine cap only makes this pre-check laxer,
# which is safe: the engine's own admission check still applies.
_MAX_NEW_TOKENS_CAP = 2048


def _effective_max_new_tokens(requested: Any) -> int:
    return min(int(requested), _MAX_NEW_TOKENS_CAP)


# Per-window/segment target for over-long raw references: the crop length in
# "trim" mode, the per-segment length in "split_fuse" mode. <= 0 disables
# long-reference handling entirely and restores the hard _MAX_REF_AUDIO_SEC
# rejection. Read once at import so every request in a process behaves
# identically (content-hash cache keys stay stable).
_REF_TRIM_SECONDS = float(os.environ.get("HIGGS_REF_TRIM_SECONDS", "30"))

# How a single raw reference LONGER than _REF_TRIM_SECONDS is handled:
#   "whole" (default) — encode the clip as one reference, up to the hard
#       _MAX_REF_AUDIO_SEC cap. Keeps every feature and does no fusion work.
#   "split_fuse" — split into equal segments and blend them through
#       reference-space voice fusion, so the timbre comes from the whole clip
#       via a short hybrid reference rather than a long prompt.
#   "trim" — keep only the best _REF_TRIM_SECONDS window (lossy).
#
# "whole" is the default because measurement, not the design's expectation,
# decides between the first two. split_fuse was written on the premise that
# its calibration build amortizes over later requests for the same voice --
# which requires those requests to land on the pod holding the cache, and
# there is no affinity routing, so in practice every request pays a cold
# build. Cold, on a card constrained to the deployment's size, an 80 s
# single-reference clone costs 10.1 s through split_fuse and 5.4 s encoded
# whole, and whole stays ahead at every concurrency level up to saturation
# (at 4 concurrent: 30.0 s vs 39.7 s wall). Splitting does make the encode
# itself 34% cheaper -- 2x40 s batched is 99 ms against 150 ms for 1x80 s --
# but that is 50 ms of a multi-second request, while the calibration
# synthesis it adds is seconds.
#
# split_fuse remains available, and remains the right shape if reference
# caching ever becomes effective (affinity routing, or a shared cache), or
# for references beyond the hard cap: it is the only mode whose prompt cost
# does not grow with the reference.
# Falls back to "trim" when the legacy "logits" fusion mode is active
# (fanning one voice out into K sibling rows costs K× KV for nothing).
_REF_LONG_MODE = os.environ.get("HIGGS_REF_LONG_MODE", "whole").strip().lower()
if _REF_LONG_MODE not in ("whole", "split_fuse", "trim"):
    raise ValueError(
        f"HIGGS_REF_LONG_MODE must be 'whole', 'split_fuse' or 'trim', "
        f"got {_REF_LONG_MODE!r}"
    )
_SPLIT_FUSE_MAX_SEGMENTS = 4
# Segments with a lower speech-active frame ratio are dropped from the blend
# (fusing near-silence in would only dilute the timbre and risk failing the
# calibration F0 gate).
_SPLIT_FUSE_MIN_ACTIVE_RATIO = 0.15


# Total reference audio one batched codec forward may carry. Peak allocation
# scales with the batch's total samples (measured on a 4090: 89 MiB for 5 s,
# 533 MiB for 30 s, 1419 MiB for 80 s), so budgeting by seconds rather than by
# item count bounds the peak whatever mix of lengths arrives.
_CODEC_BATCH_MAX_SEC = float(os.environ.get("HIGGS_CODEC_BATCH_MAX_SEC", "80"))


def _batches_within_budget(
    wavs: list[torch.Tensor], *, sample_rate: int = 24000
) -> list[list[torch.Tensor]]:
    """Split ``wavs`` into consecutive batches under the audio-seconds budget.

    A batch always contains at least one waveform, so an item longer than the
    budget is encoded alone rather than dropped.
    """
    budget = max(1, int(_CODEC_BATCH_MAX_SEC * sample_rate))
    batches: list[list[torch.Tensor]] = []
    current: list[torch.Tensor] = []
    total = 0
    for wav in wavs:
        length = int(wav.shape[-1])
        if current and total + length > budget:
            batches.append(current)
            current, total = [], 0
        current.append(wav)
        total += length
    if current:
        batches.append(current)
    return batches


def _long_reference_mode() -> str:
    if _REF_TRIM_SECONDS <= 0:
        return "whole"  # segment/window handling disabled -> hard cap applies
    if _REF_LONG_MODE == "split_fuse" and fusion_reference.fusion_mode() != "reference":
        return "trim"
    return _REF_LONG_MODE


_REF_CODE_CACHE_MAX_ITEMS = 256
_REF_CODE_CACHE_MAX_BYTES = 256 * 1024 * 1024
_REF_WAVEFORM_CACHE_MAX_ITEMS = 256
_REF_WAVEFORM_CACHE_MAX_BYTES = 512 * 1024 * 1024
_VOCODER_COMPILE_WARMUP_FRAME_COUNTS = (1, 8)

# note (kaige li): preprocessing folds these into HiggsTtsState and nothing
# downstream reads request.inputs again. Leaving them on the request re-pickles
# the raw reference audio into the payload header on every cross-process hop
# (audio_encoder -> tts_engine, tts_engine -> vocoder).
_CONSUMED_REFERENCE_INPUT_KEYS = frozenset(
    {"reference_audio", "references", "reference_codes"}
)


def _frame_activity(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    """Per-frame speech-activity flags and RMS energy over 50 ms frames.

    The activity floor is adaptive: well below typical speech RMS, well
    above silence/noise floors relative to this clip's own loud frames.
    Returns ``(active_flags, rms_energy, frame_size_in_samples)``.
    """
    frame = 24000 // 20  # 50 ms analysis frames
    n_frames = len(x) // frame
    energy = np.sqrt(
        np.mean(
            x[: n_frames * frame].reshape(n_frames, frame).astype(np.float64) ** 2,
            axis=1,
        )
    )
    floor = max(1e-4, 0.1 * float(np.percentile(energy, 95)))
    return (energy >= floor).astype(np.int64), energy, frame


def _trim_reference_waveform(wav: torch.Tensor) -> torch.Tensor:
    """Crop an over-long reference to its best ``_REF_TRIM_SECONDS`` window.

    Timbre cloning saturates well below the context budget, so instead of
    rejecting (or blindly head/middle-cutting) a long reference we keep the
    contiguous window with the most speech in it: frames are scored by RMS
    energy against an adaptive floor, the window with the highest count of
    active frames wins, and total energy breaks ties. Fully deterministic —
    the same audio always crops to the same samples, so the content-hash
    cache keys and radix fingerprints downstream stay stable.
    """
    if _REF_TRIM_SECONDS <= 0:
        return wav
    target = int(_REF_TRIM_SECONDS * 24000)
    length = int(wav.shape[-1])
    if length <= target:
        return wav

    x = wav.reshape(-1).contiguous().numpy()
    active, energy, frame = _frame_activity(x)
    n_frames = len(energy)
    window = max(1, target // frame)
    if n_frames <= window:
        return wav[..., :target]

    csum_active = np.concatenate(([0], np.cumsum(active)))
    csum_energy = np.concatenate(([0.0], np.cumsum(energy)))
    starts = np.arange(n_frames - window + 1)
    active_in_window = csum_active[starts + window] - csum_active[starts]
    energy_in_window = csum_energy[starts + window] - csum_energy[starts]

    best_active = active_in_window.max()
    candidates = np.flatnonzero(active_in_window == best_active)
    # Ties broken by energy, then by earliest start (both deterministic).
    start_frame = int(candidates[np.argmax(energy_in_window[candidates])])
    # window*frame can be < target when target is not a frame multiple; clamp
    # the start so the slice always returns exactly `target` samples even
    # when the last window wins.
    start = min(start_frame * frame, length - target)
    logger.warning(
        "reference audio cropped from %.1fs to its best %.1fs window "
        "(offset %.1fs, %d/%d speech-active frames)",
        length / 24000,
        target / 24000,
        start / 24000,
        int(best_active),
        window,
    )
    return wav[..., start : start + target]


def _split_reference_for_fusion(wav: torch.Tensor) -> list[torch.Tensor] | None:
    """Split one over-long reference into equal same-speaker segments for
    reference-space fusion, so the whole clip contributes to the timbre.

    Returns ``None`` when the clip already fits a single reference. Segments
    share one exact length so the audio encoder can batch them in a single
    GPU forward. Segments carrying almost no speech are dropped from the
    blend; if that leaves fewer than two, the most speech-active segment is
    returned alone (callers fall back to ordinary single-reference cloning
    with it). Fully deterministic, like ``_trim_reference_waveform``.
    """
    target = int(_REF_TRIM_SECONDS * 24000)
    length = int(wav.shape[-1])
    if _REF_TRIM_SECONDS <= 0 or length <= target:
        return None
    k = min(-(-length // target), _SPLIT_FUSE_MAX_SEGMENTS)
    seg_len = length // k
    flat = wav.reshape(-1)
    segments = [flat[i * seg_len : (i + 1) * seg_len] for i in range(k)]

    active, _, frame = _frame_activity(flat.contiguous().numpy())
    ratios: list[float] = []
    for i in range(k):
        f0 = (i * seg_len) // frame
        f1 = min(((i + 1) * seg_len) // frame, len(active))
        ratios.append(float(active[f0:f1].mean()) if f1 > f0 else 0.0)

    kept = [
        seg
        for seg, ratio in zip(segments, ratios)
        if ratio >= _SPLIT_FUSE_MIN_ACTIVE_RATIO
    ]
    if len(kept) < 2:
        return [segments[int(np.argmax(ratios))]]
    if len(kept) < len(segments):
        logger.warning(
            "reference split-fusion: dropped %d of %d segments with "
            "speech-active ratio below %.2f",
            len(segments) - len(kept),
            len(segments),
            _SPLIT_FUSE_MIN_ACTIVE_RATIO,
        )
    return kept


def _check_prompt_budget(prompt_len: int, max_new_tokens: int, *, what: str) -> None:
    """Reject a request the engine is guaranteed to reject, before it pays
    for encode/prefill — and say why in terms the caller can act on."""
    total = prompt_len + max_new_tokens
    if total <= _ENGINE_CONTEXT_BUDGET:
        return
    raise ValueError(
        f"{what} needs {prompt_len} prompt tokens + {max_new_tokens} "
        f"generation tokens = {total}, over the engine's "
        f"{_ENGINE_CONTEXT_BUDGET}-token context. Shorten the reference "
        f"audio ({_CODEC_FRAMES_PER_SEC} prompt tokens per second of audio) "
        f"or the target text, or lower max_new_tokens."
    )


def _reference_audio_cache_key(reference_audio: Any) -> str | None:
    """Safe source key for preprocessing waveform-cache lookup."""
    if isinstance(reference_audio, (str, Path)):
        return _reference_path_cache_key(reference_audio)
    if not isinstance(reference_audio, dict):
        return None
    path = reference_audio.get("audio_path") or reference_audio.get("path")
    if path:
        return _reference_path_cache_key(path)
    if "bytes" in reference_audio:
        data = reference_audio["bytes"]
        if isinstance(data, str):
            data = data.encode()
        return hash_media_item(data)
    encoded = reference_audio.get("base64") or reference_audio.get("data")
    if encoded is None:
        return None
    raw = base64.b64decode(encoded) if isinstance(encoded, str) else bytes(encoded)
    return hash_media_item(raw)


def _without_consumed_reference_media(inputs: Any) -> Any:
    """Return inputs with the reference media preprocessing already consumed."""
    if not isinstance(inputs, dict):
        return inputs
    return {
        key: value
        for key, value in inputs.items()
        if key not in _CONSUMED_REFERENCE_INPUT_KEYS
    }


def _fusion_ref_entries(inputs: dict) -> list[dict] | None:
    """Detect a voice-fusion request and normalize its reference entries.

    A request is a fusion request when ``references`` is a list of >= 2 entries
    *and at least one entry carries a ``weight``* (the weight is what marks the
    intent to blend rather than the legacy "first ref wins" behavior). Returns a
    list of ``{"audio": <source>, "codes": <raw [T,N] or None>, "weight": float,
    "reference_text": str | None}`` — one per voice — or ``None`` for the
    ordinary single-reference path.

    ``audio`` is whatever ``load_audio_to_24k`` accepts (path / url / dict with
    bytes|base64|data); ``codes`` is set instead when the entry supplies
    pre-encoded ``reference_codes``. Exactly one of the two is non-None.
    """
    refs = inputs.get("references")
    if not isinstance(refs, list) or len(refs) < 2:
        return None
    if not any(isinstance(r, dict) and r.get("weight") is not None for r in refs):
        return None

    entries: list[dict] = []
    for i, r in enumerate(refs):
        if not isinstance(r, dict):
            raise ValueError(f"references[{i}] must be an object for voice fusion")
        weight = r.get("weight")
        weight = 1.0 if weight is None else float(weight)
        if weight < 0:
            raise ValueError(f"references[{i}].weight must be >= 0, got {weight}")
        reference_text = r.get("text") or r.get("reference_text")
        codes = r.get("reference_codes")
        if codes is not None:
            entries.append(
                {
                    "audio": None,
                    "codes": codes,
                    "weight": weight,
                    "reference_text": reference_text,
                }
            )
            continue
        if "bytes" in r or "base64" in r or "data" in r:
            audio: Any = r
        else:
            audio = r.get("audio_path") or r.get("path")
        if audio is None:
            raise ValueError(
                f"references[{i}] must supply audio_path/path/bytes/base64/data "
                f"or reference_codes"
            )
        entries.append(
            {
                "audio": audio,
                "codes": codes,
                "weight": weight,
                "reference_text": reference_text,
            }
        )
    return entries


def _reference_code_cache_key_from_waveform(
    waveform: torch.Tensor, sample_rate: int
) -> str:
    """Content key for the reference-code cache after audio decode/resample.

    Hashing the waveform consumed by the codec keeps cache reuse tied to actual
    audio content across local files, bytes/base64 payloads, and URL refs.
    """
    wav = waveform.detach().cpu().contiguous().float()
    meta = f"sr:{int(sample_rate)}|shape:{tuple(wav.shape)}"
    return f"waveform:{meta}:{hash_bytes(wav.numpy().tobytes())}"


def _uploaded_voice_cache_key(
    reference_audio: Any,
    *,
    artifact_kind: str,
) -> SpeakerCacheKey | None:
    if not isinstance(reference_audio, dict):
        return None
    voice_name = reference_audio.get("uploaded_voice_name")
    created_at = reference_audio.get("uploaded_voice_created_at")
    if voice_name is None or created_at is None:
        return None
    return SpeakerCacheKey(
        model_type="higgs_tts",
        voice_name=str(voice_name),
        voice_version=int(created_at),
        artifact_kind=artifact_kind,
    )


def _state_uploaded_voice_cache_key(
    state: HiggsTtsState,
    *,
    artifact_kind: str,
) -> SpeakerCacheKey | None:
    if state.uploaded_voice_name is None or state.uploaded_voice_created_at is None:
        return None
    return SpeakerCacheKey(
        model_type="higgs_tts",
        voice_name=state.uploaded_voice_name,
        voice_version=int(state.uploaded_voice_created_at),
        artifact_kind=artifact_kind,
    )


class _HiggsReferenceInput:
    """Waveform plus its content key computed at preprocessing time."""

    __slots__ = ("waveform", "content_key")

    def __init__(self, waveform: torch.Tensor, content_key: str | None) -> None:
        self.waveform = waveform
        self.content_key = content_key


class _HiggsReferenceEncodeHook(TensorReferenceEncodeHook[_HiggsReferenceInput]):
    """Encode delayed 24 kHz reference codes keyed by waveform content."""

    model_revision = ""
    encoder_id = "higgs_codec_delayed"
    artifact_kind = "reference_codes"
    storage_dtype = torch.int32
    output_dtype = torch.long

    def __init__(self, codec: Any, *, num_codebooks: int, model_identity: str):
        self._codec = codec
        self._num_codebooks = int(num_codebooks)
        self.model_id = str(model_identity)
        self.encoder_config_hash = f"nq{self._num_codebooks}"

    def input_key(self, item: _HiggsReferenceInput) -> str | None:
        return item.content_key

    def encode_one(self, item: _HiggsReferenceInput) -> torch.Tensor:
        ref_codes_TN = self._codec.encode_reference(
            item.waveform, sample_rate=24000
        ).to(torch.long)
        if ref_codes_TN.ndim != 2 or ref_codes_TN.shape[1] != self._num_codebooks:
            raise ValueError(
                f"codec output must be [T, {self._num_codebooks}], got "
                f"{tuple(ref_codes_TN.shape)}"
            )
        return apply_delay_pattern(ref_codes_TN)


def create_preprocessing_executor(
    model_path: str,
    *,
    num_codebooks: int = 8,
    codebook_size: int = 1026,
    max_concurrency: int = 16,
):
    """CPU stage: text tokenize + optional ref-audio file IO.

    Builds the full prompt + delays the codes when the client supplied
    pre-encoded ``reference_codes``. When raw audio is supplied, defers
    codec encoding (and prompt assembly) to the audio_encoder stage —
    only the loaded waveform is shipped forward.

    Reference media is dropped from ``request.inputs`` once folded into the
    state, so downstream cross-process hops stop re-pickling the raw audio
    into the payload header.
    """
    checkpoint_dir = resolve_checkpoint(model_path)

    # Note:(Chenchen Hong) Load tokenizer.json directly to avoid checkpoint metadata drift.
    raw = Tokenizer.from_file(os.path.join(checkpoint_dir, "tokenizer.json"))
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=raw)
    adapter = HiggsTokenizerAdapter(tokenizer)
    # Runs on a ThreadedSimpleScheduler pool for preprocessing;
    reference_waveform_cache = StageOutputCache(
        max_size=_REF_WAVEFORM_CACHE_MAX_ITEMS,
        max_bytes=_REF_WAVEFORM_CACHE_MAX_BYTES,
    )
    reference_waveform_cache_lock = threading.Lock()
    speaker_cache = get_speaker_artifact_cache()

    def _load_fusion_waveform(audio: Any) -> torch.Tensor:
        """Load a fusion reference's audio source to a ``[1, 1, L]`` 24 kHz tensor.

        Cropping here is opt-in, like it is for a single reference. It used to
        be unconditional, which quietly discarded 50 s of an 80 s source --
        62% of the audio whose timbre the caller asked to blend -- in the one
        path this feature exists for. Nothing forces it: a source's
        calibration prompt is its reference plus CAL_MAX_NEW_TOKENS, so 80 s
        comes to 2768 of the engine's 4095 tokens and fits, and the anchor
        pass is truncated by length already. A source that genuinely cannot
        fit is rejected by _check_prompt_budget with a message saying so,
        which is the caller's decision to make rather than ours to make
        silently.
        """
        waveform_np, sample_rate = load_audio_to_24k(audio)
        wav = torch.from_numpy(waveform_np)
        if sample_rate != 24000:
            wav = F_audio.resample(wav, sample_rate, 24000)
        if _long_reference_mode() == "trim":
            wav = _trim_reference_waveform(wav)
        if wav.shape[-1] > _MAX_REF_AUDIO_SEC * 24000:
            raise ValueError(
                f"a fusion reference is too long "
                f"({wav.shape[-1] / 24000:.1f}s). Keep references at or "
                f"under {_MAX_REF_AUDIO_SEC}s, or enable automatic cropping "
                f"(HIGGS_REF_TRIM_SECONDS > 0)."
            )
        return wav.view(1, 1, -1).contiguous().float()

    def _build_fusion_state(
        text: str, specs: list[dict], params: dict, *, auto_split: bool = False
    ) -> HiggsTtsState:
        """Turn >= 2 weighted references into a ``fusion_refs`` state.

        Pre-encoded entries are delayed + prompt-built here; raw-audio entries
        carry their waveform forward (``codes_delayed`` stays None) for the
        audio_encoder GPU stage to encode.

        Mode "reference" (default): each ref additionally gets a calibration
        prompt (its voice reading the fixed calibration sentence) and the
        state carries ``fusion_build`` — the final request's prompt parts,
        assembled around the hybrid reference inside the engine stage. Mode
        "logits" keeps only the legacy sibling prompts.
        """
        reference_mode = fusion_reference.fusion_mode() == "reference"
        cal_text = fusion_reference.calibration_text() if reference_mode else None

        fusion_refs: list[dict[str, Any]] = []
        for i, spec in enumerate(specs):
            entry: dict[str, Any] = {
                "weight": float(spec["weight"]),
                "reference_text": spec.get("reference_text"),
                "codes_delayed": None,
                "prompt_token_ids": None,
                "cal_prompt_token_ids": None,
                "waveform": None,
            }
            codes = spec.get("codes")
            if codes is not None:
                codes_TN = to_codes_TN(codes, num_codebooks)
                if codes_TN is None:
                    raise ValueError(f"references[{i}].reference_codes is empty")
                if codes_TN.shape[0] > _ENGINE_CONTEXT_BUDGET:
                    raise ValueError(
                        f"references[{i}].reference_codes too long "
                        f"({codes_TN.shape[0]} frames = "
                        f"~{codes_TN.shape[0] / _CODEC_FRAMES_PER_SEC:.0f}s at "
                        f"{_CODEC_FRAMES_PER_SEC} frames/s); it can never fit "
                        f"the engine's {_ENGINE_CONTEXT_BUDGET}-token context."
                    )
                delayed = apply_delay_pattern(codes_TN)
                entry["codes_delayed"] = delayed.tolist()
                entry["prompt_token_ids"] = adapter.build_prompt(
                    text,
                    num_ref_tokens=delayed.shape[0],
                    reference_text=spec.get("reference_text"),
                )
                if not reference_mode:
                    # The per-sibling prompt only reaches the engine in the
                    # legacy "logits" mode; the default "reference" mode
                    # serves a short hybrid-reference prompt instead (upper
                    # bound checked once below).
                    _check_prompt_budget(
                        len(entry["prompt_token_ids"]),
                        _effective_max_new_tokens(params.get("max_new_tokens", 2048)),
                        what=f"fusion references[{i}]",
                    )
                if reference_mode:
                    entry["cal_prompt_token_ids"] = adapter.build_prompt(
                        cal_text,
                        num_ref_tokens=delayed.shape[0],
                        reference_text=spec.get("reference_text"),
                    )
                    _check_prompt_budget(
                        len(entry["cal_prompt_token_ids"]),
                        fusion_reference.CAL_MAX_NEW_TOKENS,
                        what=f"fusion references[{i}] calibration",
                    )
            else:
                wav = spec.get("waveform")
                entry["waveform"] = (
                    wav if wav is not None else _load_fusion_waveform(spec["audio"])
                )
            fusion_refs.append(entry)

        fusion_build = None
        if reference_mode:
            # The hybrid reference reads the calibration sentence, so it IS
            # the final request's reference transcript.
            prefix, suffix = adapter.build_prompt_parts(text, reference_text=cal_text)
            fusion_build = {
                "cal_text": cal_text,
                "final_prompt_prefix": prefix,
                "final_prompt_suffix": suffix,
            }
            # The hybrid reference's row count is unknown until the engine
            # builds it, so only reject requests that are doomed even before
            # a single hybrid-reference row is added (i.e. the target text
            # alone blows the context).
            _check_prompt_budget(
                len(prefix) + len(suffix),
                _effective_max_new_tokens(params.get("max_new_tokens", 2048)),
                what="fusion request (before hybrid-reference rows)",
            )
            if auto_split:
                # Degradation path for auto-split builds: if the calibration
                # F0 gate exhausts every seed, the engine serves one raw
                # segment as a plain single reference — that prompt has no
                # reference transcript, so ship a second prompt-part pair.
                fb_prefix, fb_suffix = adapter.build_prompt_parts(
                    text, reference_text=None
                )
                fusion_build["fallback_prompt_prefix"] = fb_prefix
                fusion_build["fallback_prompt_suffix"] = fb_suffix

        return HiggsTtsState(
            prompt_token_ids=[],
            fusion_refs=fusion_refs,
            fusion_build=fusion_build,
            target_text=text,
            num_codebooks=num_codebooks,
            codebook_size=codebook_size,
            max_new_tokens=int(params.get("max_new_tokens", 2048)),
            temperature=float(params.get("temperature", 1.0)),
            top_p=params.get("top_p"),
            top_k=params.get("top_k"),
            seed=params.get("seed"),
        )

    def _preprocess(payload: StagePayload) -> StagePayload:
        inputs = payload.request.inputs or {}
        params = payload.request.params or {}
        if isinstance(inputs, str):
            inputs = {"text": inputs}

        # Voice fusion: >= 2 weighted references → blend at the output layer.
        fusion_specs = _fusion_ref_entries(inputs)
        if fusion_specs is not None:
            text = inputs.get("input") or inputs.get("text") or ""
            payload.data = _build_fusion_state(text, fusion_specs, params).to_dict()
            payload.request.inputs = _without_consumed_reference_media(
                payload.request.inputs
            )
            return payload

        raw_refs = inputs.get("references")
        if raw_refs and isinstance(raw_refs, list):
            first = raw_refs[0]
            if isinstance(first, dict):
                inputs = dict(inputs)
                if first.get("text") and not inputs.get("reference_text"):
                    inputs["reference_text"] = first["text"]
                if inputs.get("reference_audio") is None:
                    if "bytes" in first or "base64" in first or "data" in first:
                        inputs["reference_audio"] = first
                    else:
                        inputs["reference_audio"] = first.get(
                            "audio_path"
                        ) or first.get("path")

        text = inputs.get("input") or inputs.get("text") or ""
        reference_text = inputs.get("reference_text") or None
        ref_codes_TN = to_codes_TN(inputs.get("reference_codes"), num_codebooks)
        if ref_codes_TN is not None and ref_codes_TN.shape[0] > _ENGINE_CONTEXT_BUDGET:
            raise ValueError(
                f"reference_codes is too long ({ref_codes_TN.shape[0]} frames = "
                f"~{ref_codes_TN.shape[0] / _CODEC_FRAMES_PER_SEC:.0f}s at "
                f"{_CODEC_FRAMES_PER_SEC} frames/s); it can never fit the "
                f"engine's {_ENGINE_CONTEXT_BUDGET}-token context."
            )

        waveform_tensor = None
        reference_code_cache_key = None
        uploaded_voice_name = None
        uploaded_voice_created_at = None
        if ref_codes_TN is None and inputs.get("reference_audio") is not None:
            reference_audio = inputs["reference_audio"]
            speaker_waveform_cache_key = _uploaded_voice_cache_key(
                reference_audio,
                artifact_kind="reference_waveform",
            )
            if speaker_waveform_cache_key is not None:
                uploaded_voice_name = speaker_waveform_cache_key.voice_name
                uploaded_voice_created_at = speaker_waveform_cache_key.voice_version
                cached_reference = speaker_cache.get(speaker_waveform_cache_key)
                if cached_reference is not None:
                    waveform_tensor, reference_code_cache_key = cached_reference
                    waveform_tensor = waveform_tensor.clone()
            else:
                reference_source_key = _reference_audio_cache_key(reference_audio)
                with reference_waveform_cache_lock:
                    cached_reference = reference_waveform_cache.get(
                        reference_source_key
                    )
                if cached_reference is not None:
                    cached_waveform, reference_code_cache_key = cached_reference
                    waveform_tensor = cached_waveform.clone()
            if waveform_tensor is None:
                waveform_np, sample_rate = load_audio_to_24k(reference_audio)
                wav = torch.from_numpy(waveform_np)
                if sample_rate != 24000:
                    wav = F_audio.resample(wav, sample_rate, 24000)
                mode = _long_reference_mode()
                if mode in ("trim", "whole"):
                    if mode == "trim":
                        wav = _trim_reference_waveform(wav)
                    # Both serve the clip as one prompt, so the context
                    # length is the binding constraint.
                    max_sec = float(_MAX_REF_AUDIO_SEC)
                else:
                    # split_fuse keeps the full clip; the engine only ever
                    # sees per-segment prompts, so the cap is what the
                    # segment budget can absorb, not the context length.
                    max_sec = _SPLIT_FUSE_MAX_SEGMENTS * _REF_TRIM_SECONDS
                if wav.shape[-1] > max_sec * 24000:
                    raise ValueError(
                        f"reference_audio is too long "
                        f"({wav.shape[-1] / 24000:.1f}s). Keep it at or under "
                        f"{max_sec:.0f}s, or enable automatic long-reference "
                        f"handling (HIGGS_REF_TRIM_SECONDS > 0)."
                    )
                waveform_tensor = wav.view(1, 1, -1).contiguous().float()
                reference_code_cache_key = _reference_code_cache_key_from_waveform(
                    waveform_tensor, 24000
                )
                if speaker_waveform_cache_key is not None:
                    speaker_cache.put(
                        speaker_waveform_cache_key,
                        (waveform_tensor.clone(), reference_code_cache_key),
                    )
                elif reference_source_key is not None:
                    with reference_waveform_cache_lock:
                        reference_waveform_cache.put(
                            reference_source_key,
                            (waveform_tensor.clone(), reference_code_cache_key),
                        )

        # Long single reference in split_fuse mode: blend equal segments of
        # the clip through reference-space fusion so the whole recording
        # contributes to the timbre (instead of cropping to one window).
        if waveform_tensor is not None and _long_reference_mode() == "split_fuse":
            segments = _split_reference_for_fusion(waveform_tensor)
            if segments is not None and len(segments) >= 2:
                if reference_text:
                    logger.warning(
                        "reference split-fusion ignores the provided "
                        "reference_text: segments do not match the full-clip "
                        "transcript, and the hybrid reference reads the "
                        "calibration sentence instead"
                    )
                specs = [
                    {
                        "weight": 1.0,
                        "reference_text": None,
                        "codes": None,
                        "waveform": seg.view(1, 1, -1).contiguous().float(),
                    }
                    for seg in segments
                ]
                payload.data = _build_fusion_state(
                    text, specs, params, auto_split=True
                ).to_dict()
                payload.request.inputs = _without_consumed_reference_media(
                    payload.request.inputs
                )
                return payload
            if segments is not None:
                # Only one segment carries speech: use it as an ordinary
                # single reference. The consumed audio changed, so recompute
                # the content key, drop the uploaded-voice identity (its
                # cached artifacts describe the full clip, not this segment)
                # and drop the transcript (it describes the full clip too).
                waveform_tensor = segments[0].view(1, 1, -1).contiguous().float()
                reference_code_cache_key = _reference_code_cache_key_from_waveform(
                    waveform_tensor, 24000
                )
                uploaded_voice_name = None
                uploaded_voice_created_at = None
                reference_text = None

        if ref_codes_TN is not None:
            delayed = apply_delay_pattern(ref_codes_TN)
            prompt_ids = adapter.build_prompt(
                text,
                num_ref_tokens=delayed.shape[0],
                reference_text=reference_text,
            )
            _check_prompt_budget(
                len(prompt_ids),
                _effective_max_new_tokens(params.get("max_new_tokens", 2048)),
                what="reference_codes request",
            )
            ref_codes_delayed: list[list[int]] | None = delayed.tolist()
            target_text_for_encoder = None
            reference_text_for_encoder = None
        elif waveform_tensor is None:
            prompt_ids = adapter.build_prompt(
                text, num_ref_tokens=0, reference_text=reference_text
            )
            _check_prompt_budget(
                len(prompt_ids),
                _effective_max_new_tokens(params.get("max_new_tokens", 2048)),
                what="zero-shot request",
            )
            ref_codes_delayed = None
            target_text_for_encoder = None
            reference_text_for_encoder = None
        else:
            prompt_ids = []
            ref_codes_delayed = None
            target_text_for_encoder = text
            reference_text_for_encoder = reference_text

        state = HiggsTtsState(
            prompt_token_ids=prompt_ids,
            reference_codes_delayed=ref_codes_delayed,
            reference_waveform=waveform_tensor,
            reference_code_cache_key=reference_code_cache_key,
            target_text=target_text_for_encoder,
            reference_text=reference_text_for_encoder,
            uploaded_voice_name=uploaded_voice_name,
            uploaded_voice_created_at=uploaded_voice_created_at,
            num_codebooks=num_codebooks,
            codebook_size=codebook_size,
            max_new_tokens=int(params.get("max_new_tokens", 2048)),
            temperature=float(params.get("temperature", 1.0)),
            top_p=params.get("top_p"),
            top_k=params.get("top_k"),
            seed=params.get("seed"),
            return_logprob=bool(params.get("return_logprob", False)),
            return_omni_rollout=bool(params.get("return_omni_rollout", False)),
        )
        payload.data = state.to_dict()
        payload.request.inputs = _without_consumed_reference_media(
            payload.request.inputs
        )
        return payload

    return ThreadedSimpleScheduler(_preprocess, max_concurrency=max_concurrency)


def create_audio_encoder_executor(
    model_path: str,
    *,
    device: str = "cuda:0",
    gpu_id: int | None = None,
    dtype: str = "bfloat16",
    num_codebooks: int = 8,
    max_concurrency: int = 2,
):
    """GPU stage: codec-encode raw ref audio → delayed codes + prompt assembly.

    No-op when preprocessing already produced ``reference_codes_delayed`` (the
    client-supplied pre-encoded fast path). Codec weights are extracted from
    the TTS checkpoint itself (bundled at ``tied.embedding.modality_embeddings``).
    """
    device = resolve_device_spec(device, gpu_id)
    checkpoint_dir = resolve_checkpoint(model_path)
    raw = Tokenizer.from_file(os.path.join(checkpoint_dir, "tokenizer.json"))
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=raw)
    adapter = HiggsTokenizerAdapter(tokenizer)

    codec = get_or_load_codec(checkpoint_dir, device, dtype)
    codec.model.acoustic_encoder = torch.compile(
        codec.model.acoustic_encoder, mode="default", dynamic=True
    )
    # The HuBERT semantic tower (12 transformer layers at 50 fps) dominates
    # long-reference encode time; compile it like the acoustic encoder. This
    # is a speed optimization, not a correctness requirement, so a codec that
    # does not expose the tower (a test double, or an upstream rename) warns
    # instead of failing the stage.
    if hasattr(codec.model, "semantic_model"):
        codec.model.semantic_model = torch.compile(
            codec.model.semantic_model, mode="default", dynamic=True
        )
    else:
        logger.warning(
            "codec exposes no 'semantic_model'; skipping its torch.compile "
            "(long-reference encoding will be slower)"
        )
    # Two warm-up shapes: the second, different length flips torch.compile's
    # dynamic=True specialization into the generalized dynamic-shape kernel,
    # so live traffic with arbitrary reference lengths mostly avoids
    # recompiles — recompiles serialize process-wide on Dynamo's compile_lock
    # and would stall concurrent encodes.
    for warmup_seconds in (1, 2):
        codec.encode_reference(
            torch.zeros(warmup_seconds * codec.SAMPLE_RATE),
            sample_rate=codec.SAMPLE_RATE,
        )
    reference_service = ReferenceEncodeService(
        _HiggsReferenceEncodeHook(
            codec,
            num_codebooks=num_codebooks,
            model_identity=checkpoint_dir,
        ),
        max_items=_REF_CODE_CACHE_MAX_ITEMS,
        max_bytes=_REF_CODE_CACHE_MAX_BYTES,
        log_prefix="Higgs ref cache",
    )
    speaker_cache = get_speaker_artifact_cache()
    # Content-keyed cache of fusion-ref codes (int32 delayed rows): repeated
    # requests for the same (auto-split) reference skip the codec GPU forward
    # entirely — the engine-side fused-reference cache only helps AFTER this
    # stage has already re-encoded every segment.
    fusion_code_cache = StageOutputCache(
        max_size=_REF_CODE_CACHE_MAX_ITEMS,
        max_bytes=_REF_CODE_CACHE_MAX_BYTES,
    )
    fusion_code_cache_lock = threading.Lock()

    def _validate_ref_codes(ref_codes_TN: torch.Tensor) -> torch.Tensor:
        if ref_codes_TN.ndim != 2 or ref_codes_TN.shape[1] != num_codebooks:
            raise ValueError(
                f"codec output must be [T, {num_codebooks}], got "
                f"{tuple(ref_codes_TN.shape)}"
            )
        return ref_codes_TN

    def _encode_one(waveform: torch.Tensor) -> list[list[int]]:
        """Codec-encode one 24 kHz mono waveform → delayed code rows."""
        ref_codes_TN = _validate_ref_codes(
            codec.encode_reference(waveform, sample_rate=24000).to(torch.long)
        )
        return apply_delay_pattern(ref_codes_TN).tolist()

    def _encode(payload: StagePayload) -> StagePayload:
        state = HiggsTtsState.from_dict(payload.data)

        # Voice fusion: encode each reference that preprocessing left as a raw
        # waveform, delay it, and prebuild that sibling's prompt (mode
        # "logits") plus its calibration prompt (mode "reference", where the
        # engine clones each voice reading the calibration sentence before
        # morphing them into one hybrid reference). Pre-encoded refs already
        # carry their prompts and pass through untouched.
        if state.fusion_refs:
            cal_text = (state.fusion_build or {}).get("cal_text")
            pending: list[dict] = []
            wavs: list[torch.Tensor] = []
            for ref in state.fusion_refs:
                if ref.get("codes_delayed") is not None:
                    continue
                wav = ref.pop("waveform", None)
                if wav is None:
                    raise ValueError(
                        "fusion ref has neither codes_delayed nor waveform"
                    )
                if not isinstance(wav, torch.Tensor):
                    wav = torch.as_tensor(wav, dtype=torch.float32)
                pending.append(ref)
                wavs.append(wav)
            keys = [_reference_code_cache_key_from_waveform(w, 24000) for w in wavs]
            with fusion_code_cache_lock:
                cached = [fusion_code_cache.get(k) for k in keys]
            rows_list: list[list[list[int]] | None] = [
                c.tolist() if c is not None else None for c in cached
            ]
            miss = [i for i, rows in enumerate(rows_list) if rows is None]
            miss_wavs = [wavs[i] for i in miss]
            if (
                len(miss_wavs) > 1
                and hasattr(codec, "encode_batch")
                and len({int(w.shape[-1]) for w in miss_wavs}) == 1
            ):
                # Equal-length refs (the auto-split segments of one long
                # reference) encode in a single batched GPU forward instead
                # of N sequential ones.
                #
                # Chunked by total audio, because the batch is not otherwise
                # bounded: a request may carry any number of references, and
                # equal lengths are the common case (the split segments of one
                # clip, or same-duration clips from one recording session), so
                # the batching condition above is met precisely when the batch
                # is largest. Sixteen 80 s references reached one HuBERT
                # forward asking for 2.15 GiB and took the whole stage's budget
                # out with it. The per-chunk ceiling is one reference at the
                # length cap, whose peak allocation is measured and provisioned
                # for; a chunk always holds at least one reference so an
                # over-long single item still makes progress.
                encoded = []
                for batch in _batches_within_budget(miss_wavs):
                    encoded.extend(
                        apply_delay_pattern(_validate_ref_codes(c.to(torch.long)))
                        for c in codec.encode_batch(batch)
                    )
            else:
                encoded = [
                    apply_delay_pattern(
                        _validate_ref_codes(
                            codec.encode_reference(w, sample_rate=24000).to(torch.long)
                        )
                    )
                    for w in miss_wavs
                ]
            for i, delayed in zip(miss, encoded):
                rows_list[i] = delayed.tolist()
                with fusion_code_cache_lock:
                    fusion_code_cache.put(keys[i], delayed.to(torch.int32))
            for ref, delayed_rows in zip(pending, rows_list):
                ref["codes_delayed"] = delayed_rows
                ref["prompt_token_ids"] = adapter.build_prompt(
                    state.target_text or "",
                    num_ref_tokens=len(delayed_rows),
                    reference_text=ref.get("reference_text"),
                )
                if not cal_text:
                    # Sibling prompts reach the engine only in "logits" mode
                    # (cal_text is set exactly in the default reference mode).
                    _check_prompt_budget(
                        len(ref["prompt_token_ids"]),
                        _effective_max_new_tokens(state.max_new_tokens),
                        what="fusion reference",
                    )
                if cal_text:
                    ref["cal_prompt_token_ids"] = adapter.build_prompt(
                        cal_text,
                        num_ref_tokens=len(delayed_rows),
                        reference_text=ref.get("reference_text"),
                    )
                    _check_prompt_budget(
                        len(ref["cal_prompt_token_ids"]),
                        fusion_reference.CAL_MAX_NEW_TOKENS,
                        what="fusion reference calibration",
                    )
            state.target_text = None
            payload.data = state.to_dict()
            return payload

        waveform = state.reference_waveform
        if waveform is None:
            return payload

        # note (luojiaxuan): Uploaded voices stay on the versioned speaker cache
        # invalidated by voice re-upload; everything else rides the shared service.
        speaker_code_cache_key = _state_uploaded_voice_cache_key(
            state,
            artifact_kind="reference_codes",
        )
        cached_delayed = (
            speaker_cache.get(speaker_code_cache_key)
            if speaker_code_cache_key is not None
            else None
        )
        if cached_delayed is not None:
            delayed_rows = cached_delayed.tolist()
        else:
            delayed = reference_service.get_or_encode(
                _HiggsReferenceInput(waveform, state.reference_code_cache_key),
                desc=state.uploaded_voice_name or "ad-hoc reference",
            )
            delayed_rows = delayed.tolist()
            if speaker_code_cache_key is not None:
                speaker_cache.put(
                    speaker_code_cache_key, delayed.detach().to("cpu", torch.int32)
                )
        state.reference_codes_delayed = delayed_rows
        state.prompt_token_ids = adapter.build_prompt(
            state.target_text or "",
            num_ref_tokens=len(delayed_rows),
            reference_text=state.reference_text,
        )
        _check_prompt_budget(
            len(state.prompt_token_ids),
            _effective_max_new_tokens(state.max_new_tokens),
            what="reference_audio request",
        )
        state.reference_waveform = None
        state.reference_code_cache_key = None
        state.target_text = None
        state.reference_text = None
        payload.data = state.to_dict()
        return payload

    # max_concurrency > 1 overlaps one request's CPU work (resample, semantic
    # feature extraction, tolist) with another's GPU kernels; the GPU kernels
    # themselves serialize on the shared default stream. _encode is re-entrant:
    # codec encode is stateless inference, and both reference_service and
    # speaker_cache take their own locks.
    return SimpleScheduler(_encode, max_concurrency=max_concurrency)


def create_sglang_tts_engine_executor(
    model_path: str,
    *,
    device: str = "cuda:0",
    max_new_tokens: int | None = 2048,
    max_running_requests: int = 64,
    cuda_graph_max_bs: int = 64,
    server_args_overrides: dict[str, Any] | None = None,
    enable_async_decode: bool = False,
    async_decode_min_batch_size: int = 2,
    stream_stride: int = DEFAULT_HIGGS_STREAM_STRIDE,
    stream_followup_stride: int = DEFAULT_HIGGS_STREAM_FOLLOWUP_STRIDE,
    initial_chunk_frames: int = DEFAULT_HIGGS_INITIAL_CHUNK_FRAMES,
    prefill_coalesce_requests: int = 0,
    prefill_coalesce_wait_ms: float = 60.0,
    total_gpu_memory_fraction: float | None = None,
):
    """sglang-backed AR engine for Higgs TTS."""
    from sglang_omni.models.higgs_tts.engine_builder import HiggsTtsEngineBuilder

    return HiggsTtsEngineBuilder(
        max_new_tokens=max_new_tokens,
        max_running_requests=max_running_requests,
        cuda_graph_max_bs=cuda_graph_max_bs,
        enable_async_decode=enable_async_decode,
        async_decode_min_batch_size=async_decode_min_batch_size,
        stream_stride=stream_stride,
        stream_followup_stride=stream_followup_stride,
        initial_chunk_frames=initial_chunk_frames,
        prefill_coalesce_requests=prefill_coalesce_requests,
        prefill_coalesce_wait_ms=prefill_coalesce_wait_ms,
        total_gpu_memory_fraction=total_gpu_memory_fraction,
    ).build(
        model_path,
        device=device,
        server_args_overrides=server_args_overrides,
    )


def create_vocoder_executor(
    model_path: str,
    *,
    device: str = "cuda:0",
    dtype: str = "bfloat16",
    vocoder_decode_batch_size: int = 16,
    max_batch_wait_ms: int = 2,
    stream_stride: int = DEFAULT_HIGGS_STREAM_STRIDE,
    stream_followup_stride: int = DEFAULT_HIGGS_STREAM_FOLLOWUP_STRIDE,
    initial_chunk_frames: int = DEFAULT_HIGGS_INITIAL_CHUNK_FRAMES,
    stream_overlap_tokens: int = 8,
    stream_holdback_tokens: int = 4,
    compile_decode: bool = False,
    decode_cuda_graph_frame_counts: tuple[int, ...] = (),
):
    """Decode Higgs delayed codes to a mono 24 kHz waveform.

    Codec weights are extracted from the TTS checkpoint itself.
    """
    if compile_decode and decode_cuda_graph_frame_counts:
        raise ValueError(
            "compile_decode and decode_cuda_graph_frame_counts are mutually exclusive"
        )
    # decode_cuda_graph_frame_counts must cover every window size the streaming
    # scheduler can submit, or those windows fall back to eager decode (warned
    # only once per distinct missed frame count, so easy to miss in serving
    # logs). The reachable set is a joint function of
    # stream_stride/stream_followup_stride/stream_overlap_tokens/
    # stream_holdback_tokens, the codec's codebook count, and the engine
    # stage's flush cadence (HiggsTTSModelRunner._initial/_next_stream_flush
    # rows) — no sound closed form exists from this stage's arguments alone,
    # so there is deliberately no startup validation here. The default
    # tuple(range(1, 151)) in config.py covers the default 75+75 strides with
    # margin; when overriding strides, re-derive the domain empirically.
    checkpoint_dir = resolve_checkpoint(model_path)
    codec = get_or_load_codec(checkpoint_dir, device, dtype)
    if compile_decode:
        eager_decode = codec.model.decode
        try:
            codec.model.decode = torch.compile(eager_decode, dynamic=True)
            warm_codes_TN = torch.zeros(
                (
                    max(_VOCODER_COMPILE_WARMUP_FRAME_COUNTS),
                    int(codec.model.config.num_quantizers),
                ),
                dtype=torch.long,
                device="cpu",
            )
            # Note: (stephenkgli) match serving's contiguous [T, N] layout and
            # warm the zero-one-specialized batch and frame-count classes.
            for frame_count in _VOCODER_COMPILE_WARMUP_FRAME_COUNTS:
                frame_codes_TN = warm_codes_TN[:frame_count]
                codec.decode(frame_codes_TN)
                codec.decode_batch([frame_codes_TN, frame_codes_TN])
        except Exception:
            logger.warning(
                "torch.compile of the codec decode failed; falling back to the "
                "eager vocoder decode",
                exc_info=True,
            )
            codec.model.decode = eager_decode
    elif decode_cuda_graph_frame_counts:
        # This is an explicitly selected performance contract. Failing startup
        # is preferable to silently serving through the eager path and
        # discovering the regression only in a latency/throughput CI job.
        codec.capture_decode_cuda_graphs(
            tuple(int(value) for value in decode_cuda_graph_frame_counts)
        )

    return HiggsStreamingVocoderScheduler(
        codec,
        max_batch_size=vocoder_decode_batch_size,
        max_batch_wait_ms=max_batch_wait_ms,
        stream_stride=stream_stride,
        stream_followup_stride=stream_followup_stride,
        initial_chunk_frames=initial_chunk_frames,
        stream_overlap_tokens=stream_overlap_tokens,
        stream_holdback_tokens=stream_holdback_tokens,
    )


__all__ = [
    "create_audio_encoder_executor",
    "create_preprocessing_executor",
    "create_sglang_tts_engine_executor",
    "create_vocoder_executor",
]
