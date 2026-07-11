# SPDX-License-Identifier: Apache-2.0
"""Request mapping helpers for MOSS-TTS Delay."""

from __future__ import annotations

import base64
import collections
import hashlib
import io
import re
import time
from dataclasses import dataclass, field
from typing import Any

import torch

from sglang_omni.models.moss_tts.payload_types import MossTTSState
from sglang_omni.proto import StagePayload
from sglang_omni.sampling.seed import derive_sampling_seed, new_random_sampling_seed
from sglang_omni.scheduling.prepared_request_queue import PreparedRequestQueue
from sglang_omni.scheduling.types import ARRequestData
from sglang_omni.utils.audio_payload import audio_data_uri_from_reference

MOSS_TTS_DEFAULT_MAX_NEW_TOKENS = 4096
_MOSS_TTS_PREPARED_MARKER = "_moss_tts_prepared_request"
_TOKEN_PREFIX_RE = re.compile(r"^\$\{token:(\d+)\}")
_TOKEN_PREFIX_START_RE = re.compile(r"^\$\{token:")
_DATA_URI_RE = re.compile(r"^data:audio/[^;,]+;base64,(?P<data>.+)$", re.DOTALL)
_INF_DELAY = -1


def _new_moss_tts_sampling_seed() -> int:
    return new_random_sampling_seed()


def derive_moss_tts_sampling_seed(public_seed: int) -> int:
    """Derive a stable per-request sampling seed from a public ``seed``.

    Mirrors the Qwen3-TTS pattern: ``multinomial_with_seed`` combines this
    per-request seed with a per-(step, channel) position, so a row's sampled
    token depends only on its own seed and position -- never on its batch
    neighbours. This makes ``seed`` reproducible at any batch size, not just 1.
    """
    return derive_sampling_seed("moss-tts", int(public_seed))


_GENERATION_FIELDS = (
    "max_new_tokens",
    "temperature",
    "top_p",
    "top_k",
    "repetition_penalty",
    "text_temperature",
    "text_top_p",
    "text_top_k",
    "audio_temperature",
    "audio_top_p",
    "audio_top_k",
    "audio_repetition_penalty",
)


@dataclass
class MossTTSSGLangRequestData(ARRequestData):
    """Scheduler-owned request state for MOSS-TTS Delay."""

    enforce_request_limits: bool = True
    req: Any = None
    synced: bool = False
    generation_steps: int = 0
    suppress_tokens: list[int] | None = None
    input_embeds_are_projected: bool = False
    prefill_input_embeds: torch.Tensor | None = None
    decode_input_embeds: list[torch.Tensor] = field(default_factory=list)
    stage_payload: Any = None
    state: MossTTSState = field(default_factory=MossTTSState)
    model_config: Any = None
    prompt_rows: torch.Tensor | None = None
    assistant_prefix_rows: torch.Tensor | None = None
    output_rows: list[torch.Tensor] = field(default_factory=list)
    pending_feedback_queue: Any = field(default_factory=collections.deque)
    text_temperature: float = 1.5
    text_top_p: float = 1.0
    text_top_k: int = 50
    audio_temperature: float = 1.7
    audio_top_p: float = 0.8
    audio_top_k: int = 25
    audio_repetition_penalty: float = 1.0
    seed: int | None = None
    sampling_seed: int = field(default_factory=_new_moss_tts_sampling_seed)
    delay_state: torch.Tensor | None = None
    audio_length: int = 0
    delayed_length: int = _INF_DELAY
    is_audio: bool = False
    engine_start_s: float = 0.0


@dataclass
class MossTTSPreparedRequest:
    """Heavy MOSS-TTS preprocessing output consumed by the AR scheduler."""

    state: MossTTSState
    input_ids_list: list[int]
    input_ids: torch.Tensor
    prompt_rows: torch.Tensor
    gen_kwargs: dict[str, Any]


@dataclass
class MossTTSPreprocessingContext:
    processor: Any


_QUEUE: PreparedRequestQueue[MossTTSPreprocessingContext, MossTTSPreparedRequest] = (
    PreparedRequestQueue()
)


def set_moss_tts_preprocessing_context(*, processor: Any) -> None:
    """Register the upstream MOSS processor used by preprocessing."""

    _QUEUE.set_context(MossTTSPreprocessingContext(processor=processor))


def clear_moss_tts_preprocessing_context() -> None:
    """Clear MOSS-TTS preprocessing globals, mainly for tests and reloads."""

    _QUEUE.clear_context()


def cleanup_prepared_moss_tts_request(request_id: str) -> None:
    """Drop any prepared MOSS-TTS handoff for an aborted request.

    Only tombstone (so a pending insert is later dropped) when preprocessing is
    actually in flight; an abort for a request that is not being preprocessed
    leaves nothing behind.
    """

    _QUEUE.abort(str(request_id))


def pop_prepared_moss_tts_request(
    payload: StagePayload,
) -> MossTTSPreparedRequest | None:
    data = payload.data if isinstance(payload.data, dict) else {}
    marker = data.get(_MOSS_TTS_PREPARED_MARKER)
    if marker is None:
        return None
    prepared = _QUEUE.pop(str(marker))
    if prepared is None:
        raise RuntimeError(
            "MOSS-TTS preprocessing state is missing for prepared payload "
            f"{marker!r}; the AR scheduler must not rebuild it"
        )
    return prepared


def normalize_moss_tts_inputs(inputs: Any) -> tuple[str, list[dict[str, Any]]]:
    if isinstance(inputs, str):
        return inputs, []
    if isinstance(inputs, dict):
        references = inputs.get("references") or []
        if not isinstance(references, list):
            raise ValueError("MOSS-TTS references must be a list")
        return str(inputs.get("text", inputs.get("input", ""))), [
            dict(reference) for reference in references if isinstance(reference, dict)
        ]
    return str(inputs) if inputs is not None else "", []


def resolve_moss_reference(
    references: list[dict[str, Any]],
    tts_params: dict[str, Any],
) -> tuple[Any | None, str | None]:
    reference = references[0] if references else {}
    ref_audio = (
        reference.get("audio_path")
        or reference.get("ref_audio")
        or reference.get("audio")
        or audio_data_uri_from_reference(reference)
        or tts_params.get("ref_audio")
    )
    ref_text = reference.get("text") or tts_params.get("ref_text")
    return ref_audio, str(ref_text) if ref_text is not None else None


def _resolve_optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _resolve_token_count(
    text: str,
    params: dict[str, Any],
    tts_params: dict[str, Any],
) -> tuple[str, int | None]:
    """Resolve the duration token count and return ``(clean_text, count)``.

    The upstream v1.5 processor takes the duration via ``build_user_message(
    text=..., tokens=...)`` and writes it into the ``- Tokens:`` field, so the
    inline ``${token:N}`` prefix must be stripped out of ``text`` while its value
    becomes ``tokens``. Explicit ``token_count`` / ``duration_tokens`` / ``tokens``
    params and the ``${token:N}`` prefix are all validated as ``> 0``. Other
    markup ([pause Xs], pinyin, IPA, ...) is passed through unchanged.
    """
    for source in (tts_params, params):
        for key in ("token_count", "duration_tokens", "tokens"):
            if source.get(key) is not None:
                value = source[key]
                if isinstance(value, bool):
                    raise ValueError("MOSS-TTS token_count must be an integer")
                count = int(value)
                if count <= 0:
                    raise ValueError("MOSS-TTS token_count must be > 0")
                return text, count

    match = _TOKEN_PREFIX_RE.match(text)
    if match:
        count = int(match.group(1))
        if count <= 0:
            raise ValueError("MOSS-TTS ${token:N} count must be > 0")
        return text[match.end() :].lstrip(), count
    if _TOKEN_PREFIX_START_RE.match(text):
        raise ValueError("MOSS-TTS ${token:N} count must be a positive integer")
    return text, None


def build_moss_tts_state(payload: StagePayload) -> MossTTSState:
    inputs = payload.request.inputs or {}
    params = payload.request.params or {}
    metadata = payload.request.metadata or {}
    tts_params = metadata.get("tts_params")
    if not isinstance(tts_params, dict):
        tts_params = {}

    text, references = normalize_moss_tts_inputs(inputs)
    ref_audio, ref_text = resolve_moss_reference(references, tts_params)
    language = _resolve_optional_text(
        tts_params.get("language") or params.get("language")
    )
    instructions = _resolve_optional_text(
        tts_params.get("instructions")
        or tts_params.get("instruct")
        or params.get("instructions")
        or params.get("instruct")
    )
    text, token_count = _resolve_token_count(text, params, tts_params)
    return MossTTSState(
        text=text,
        ref_audio=ref_audio,
        ref_text=ref_text,
        language=language,
        instructions=instructions,
        token_count=token_count,
        generation_kwargs=build_generation_kwargs(params, tts_params=tts_params),
    )


def build_generation_kwargs(
    params: dict[str, Any],
    *,
    tts_params: dict[str, Any],
) -> dict[str, Any]:
    explicit_generation_params = tts_params.get("explicit_generation_params")
    if isinstance(explicit_generation_params, (list, tuple, set)):
        explicit_fields = {str(field) for field in explicit_generation_params}
    else:
        explicit_fields = set()

    raw_max_new_tokens = params.get("max_new_tokens")
    if raw_max_new_tokens is None:
        max_new_tokens = MOSS_TTS_DEFAULT_MAX_NEW_TOKENS
    elif isinstance(raw_max_new_tokens, bool):
        raise ValueError(
            f"MOSS-TTS max_new_tokens must be an integer, got {raw_max_new_tokens!r}"
        )
    else:
        max_new_tokens = int(raw_max_new_tokens)

    generation_kwargs: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        # note (chenyang): the checkpoint's own generate() defaults; greedy
        # (temperature=0) collapses the codec LM into copying the reference
        # audio. Callers may override any field.
        "text_temperature": 1.5,
        "audio_temperature": 1.7,
        "text_top_p": 1.0,
        "audio_top_p": 0.8,
        "text_top_k": 50,
        "audio_top_k": 25,
        "audio_repetition_penalty": 1.0,
    }

    if "temperature" in explicit_fields and params.get("temperature") is not None:
        generation_kwargs["text_temperature"] = float(params["temperature"])
        generation_kwargs["audio_temperature"] = float(params["temperature"])
    if "top_p" in explicit_fields and params.get("top_p") is not None:
        generation_kwargs["text_top_p"] = float(params["top_p"])
        generation_kwargs["audio_top_p"] = float(params["top_p"])
    if "top_k" in explicit_fields and params.get("top_k") is not None:
        generation_kwargs["text_top_k"] = int(params["top_k"])
        generation_kwargs["audio_top_k"] = int(params["top_k"])
    if (
        "repetition_penalty" in explicit_fields
        and params.get("repetition_penalty") is not None
    ):
        generation_kwargs["audio_repetition_penalty"] = float(
            params["repetition_penalty"]
        )

    for source in (tts_params, params):
        for field in (
            "text_temperature",
            "text_top_p",
            "text_top_k",
            "audio_temperature",
            "audio_top_p",
            "audio_top_k",
            "audio_repetition_penalty",
        ):
            if source.get(field) is not None:
                value = source[field]
                generation_kwargs[field] = (
                    int(value) if field.endswith("top_k") else float(value)
                )

    seed = tts_params.get("seed")
    if seed is None:
        seed = params.get("seed")
    if seed is not None:
        generation_kwargs["seed"] = seed

    _validate_moss_tts_generation_kwargs(generation_kwargs)
    return generation_kwargs


def _validate_moss_tts_generation_kwargs(kwargs: dict[str, Any]) -> None:
    """Validate public sampling fields (MOSS uses a custom sampler that bypasses
    SGLang's SamplingParams.verify), raising ValueError on out-of-range values."""
    if int(kwargs["max_new_tokens"]) <= 0:
        raise ValueError(
            f"MOSS-TTS max_new_tokens must be > 0, got {kwargs['max_new_tokens']!r}"
        )
    for field in ("text_temperature", "audio_temperature"):
        if float(kwargs[field]) < 0:
            raise ValueError(f"MOSS-TTS {field} must be >= 0, got {kwargs[field]!r}")
    for field in ("text_top_p", "audio_top_p"):
        if not 0.0 < float(kwargs[field]) <= 1.0:
            raise ValueError(
                f"MOSS-TTS {field} must be in (0, 1], got {kwargs[field]!r}"
            )
    for field in ("text_top_k", "audio_top_k"):
        if int(kwargs[field]) < -1:
            raise ValueError(f"MOSS-TTS {field} must be >= -1, got {kwargs[field]!r}")
    if float(kwargs["audio_repetition_penalty"]) <= 0:
        raise ValueError(
            "MOSS-TTS audio_repetition_penalty must be > 0, got "
            f"{kwargs['audio_repetition_penalty']!r}"
        )
    seed = kwargs.get("seed")
    if seed is not None and (
        isinstance(seed, bool) or not isinstance(seed, int) or seed < 0
    ):
        raise ValueError(f"MOSS-TTS seed must be a non-negative integer, got {seed!r}")


def build_row_cache_key_ids(rows: torch.Tensor) -> list[int]:
    """Build stable radix-cache token ids for MOSS multi-channel prompt rows."""

    rows = rows.detach().to(dtype=torch.long, device="cpu")
    key_ids: list[int] = []
    for row in rows:
        digest = hashlib.blake2b(row.numpy().tobytes(), digest_size=8).digest()
        key_ids.append(int.from_bytes(digest, "little") & ((1 << 63) - 1))
    return key_ids


def _reference_for_processor(processor: Any, ref_audio: Any | None) -> list[Any] | None:
    if ref_audio is None:
        return None
    if not isinstance(ref_audio, str):
        return [ref_audio]
    match = _DATA_URI_RE.match(ref_audio)
    if match is None:
        return [ref_audio]

    try:
        import soundfile as sf
    except ImportError as exc:
        raise RuntimeError(
            "MOSS-TTS base64 reference audio requires soundfile to decode the data URI"
        ) from exc

    raw = base64.b64decode(match.group("data"))
    audio, sample_rate = sf.read(io.BytesIO(raw), dtype="float32", always_2d=True)
    wav = torch.from_numpy(audio.T)
    codes = processor.encode_audios_from_wav([wav], int(sample_rate))[0]
    return [codes]


def _build_processor_message(processor: Any, state: MossTTSState) -> dict[str, Any]:
    reference = _reference_for_processor(processor, state.ref_audio)
    return processor.build_user_message(
        text=state.text,
        reference=reference,
        instruction=state.instructions,
        tokens=state.token_count,
        language=state.language,
    )


def _prepare_moss_tts_request(
    payload: StagePayload,
    *,
    processor: Any,
) -> MossTTSPreparedRequest:
    state = build_moss_tts_state(payload)
    message = _build_processor_message(processor, state)
    batch = processor([[message]], mode="generation")
    input_rows = batch["input_ids"]
    if input_rows.ndim != 3 or int(input_rows.shape[0]) != 1:
        raise ValueError(
            "MOSS-TTS processor must return input_ids with shape [1, T, C]"
        )
    prompt_rows = input_rows[0].detach().to(dtype=torch.long, device="cpu")
    input_ids_list = build_row_cache_key_ids(prompt_rows)
    return MossTTSPreparedRequest(
        state=state,
        input_ids_list=input_ids_list,
        input_ids=torch.tensor(input_ids_list, dtype=torch.long),
        prompt_rows=prompt_rows,
        gen_kwargs=state.generation_kwargs,
    )


def preprocess_moss_tts_payload(payload: StagePayload) -> StagePayload:
    """Run MOSS-TTS prompt/reference preprocessing outside the AR scheduler."""

    rid = str(payload.request_id)
    context = _QUEUE.begin(rid)
    if context is None:
        raise RuntimeError(
            "MOSS-TTS preprocessing context is not initialized; "
            "create_preprocessing_executor must register it before requests run"
        )

    try:
        prepared = _prepare_moss_tts_request(payload, processor=context.processor)
    except BaseException:
        _QUEUE.fail_inflight(rid)
        raise
    # note (Yue Yin): publish fails closed; when it drops the handoff (aborted
    # mid-flight or context reset) skip the marker, so the AR stage never pops a
    # marker whose prepared state no longer exists.
    published = _QUEUE.publish(rid, prepared)

    data = prepared.state.to_dict()
    if published:
        data[_MOSS_TTS_PREPARED_MARKER] = payload.request_id
    return StagePayload(
        request_id=payload.request_id, request=payload.request, data=data
    )


def _last_equal(rows: torch.Tensor, value: int) -> int:
    matches = (rows[:, 0] == int(value)).nonzero(as_tuple=False).flatten()
    if matches.numel() == 0:
        return -1
    return int(matches[-1].item())


def _resolve_audio_payload_bounds(
    rows: torch.Tensor, cfg: Any
) -> tuple[int, int] | None:
    text = rows[:, 0].to(dtype=torch.long)
    bos_pos = (text == int(cfg.audio_start_token_id)).nonzero(as_tuple=False)
    if bos_pos.numel() == 0:
        gen_pos = (text == int(cfg.audio_assistant_gen_slot_token_id)).nonzero(
            as_tuple=False
        )
        if gen_pos.numel() == 0:
            return None
        start = int(gen_pos[0].item())
    else:
        start = int(bos_pos[0].item()) + 1

    eos_pos = (text[start:] == int(cfg.audio_end_token_id)).nonzero(as_tuple=False)
    if eos_pos.numel() > 0:
        end = start + int(eos_pos[0].item())
    else:
        end_candidates: list[int] = []
        for token_id in (
            int(cfg.audio_assistant_gen_slot_token_id),
            int(cfg.audio_assistant_delay_slot_token_id),
        ):
            matches = (text[start:] == token_id).nonzero(as_tuple=False)
            if matches.numel() > 0:
                end_candidates.append(start + int(matches[-1].item()) + 1)
        if not end_candidates:
            return None
        end = max(end_candidates)

    n_vq = int(rows.shape[1] - 1)
    if end <= start or end <= start + n_vq:
        return None
    return start, end


def _initialize_generation_state(
    data: MossTTSSGLangRequestData,
    *,
    model: Any,
) -> None:
    prompt_rows = data.prompt_rows
    if prompt_rows is None or prompt_rows.numel() == 0:
        return
    cfg = model.config
    seq_len = int(prompt_rows.shape[0])
    last_text = int(prompt_rows[-1, 0].item())
    is_continuation = last_text in (
        int(cfg.audio_start_token_id),
        int(cfg.audio_assistant_gen_slot_token_id),
    )
    audio_start_idx = _last_equal(prompt_rows, int(cfg.audio_start_token_id))
    data.is_audio = bool(is_continuation and audio_start_idx >= 0)
    data.audio_length = seq_len - audio_start_idx if data.is_audio else 0
    assistant_start_idx = _last_equal(prompt_rows, int(cfg.im_start_token_id)) + 3
    assistant_start_idx = max(0, min(assistant_start_idx, seq_len))
    data.assistant_prefix_rows = prompt_rows[assistant_start_idx:].detach().clone()
    data.state.assistant_start_length = int(data.assistant_prefix_rows.shape[0])


def build_sglang_moss_tts_request(
    payload: StagePayload,
    *,
    model: Any,
) -> MossTTSSGLangRequestData:
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.sampling.sampling_params import SamplingParams

    prepared = pop_prepared_moss_tts_request(payload)
    if prepared is None:
        raise RuntimeError(
            "MOSS-TTS AR request builder requires a payload prepared by "
            "preprocess_moss_tts_payload"
        )

    cfg = model.config
    gen_kwargs = prepared.gen_kwargs
    max_new_tokens = int(
        gen_kwargs.get("max_new_tokens", MOSS_TTS_DEFAULT_MAX_NEW_TOKENS)
    )
    sampling_params = SamplingParams(
        max_new_tokens=max_new_tokens,
        temperature=0.0,
        stop_token_ids=[int(cfg.im_end_token_id)],
    )
    sampling_params.normalize(None)
    sampling_params.verify(int(cfg.vocab_size_list[0]))

    req = Req(
        rid=payload.request_id,
        origin_input_text="",
        origin_input_ids=prepared.input_ids_list,
        sampling_params=sampling_params,
        eos_token_ids={int(cfg.im_end_token_id)},
        vocab_size=int(cfg.vocab_size_list[0]),
    )
    req.tokenizer = None
    req._input_embeds_are_projected = True
    req._codec_suppress_tokens = None

    data = MossTTSSGLangRequestData(
        input_ids=prepared.input_ids,
        max_new_tokens=max_new_tokens,
        temperature=0.0,
        output_ids=req.output_ids,
        req=req,
        state=prepared.state,
        model_config=cfg,
        prompt_rows=prepared.prompt_rows,
        text_temperature=float(gen_kwargs.get("text_temperature", 0.0)),
        text_top_p=float(gen_kwargs.get("text_top_p", 1.0)),
        text_top_k=int(gen_kwargs.get("text_top_k", -1)),
        audio_temperature=float(gen_kwargs.get("audio_temperature", 0.0)),
        audio_top_p=float(gen_kwargs.get("audio_top_p", 1.0)),
        audio_top_k=int(gen_kwargs.get("audio_top_k", -1)),
        audio_repetition_penalty=float(gen_kwargs.get("audio_repetition_penalty", 1.0)),
        seed=gen_kwargs.get("seed"),
        sampling_seed=(
            derive_moss_tts_sampling_seed(gen_kwargs["seed"])
            if gen_kwargs.get("seed") is not None
            else _new_moss_tts_sampling_seed()
        ),
        engine_start_s=time.perf_counter(),
    )
    data.input_embeds_are_projected = True
    _initialize_generation_state(data, model=model)
    data.stage_payload = payload
    return data


def apply_sglang_moss_tts_result(
    payload: StagePayload,
    data: MossTTSSGLangRequestData,
) -> StagePayload:
    state = data.state
    if data.assistant_prefix_rows is None:
        assistant_prefix_rows = torch.empty((0, 0), dtype=torch.long)
    else:
        assistant_prefix_rows = data.assistant_prefix_rows.to(dtype=torch.long)

    if data.output_rows:
        generated_rows = torch.stack(data.output_rows, dim=0).to(dtype=torch.long)
        if assistant_prefix_rows.numel() > 0:
            rows = torch.cat(
                [assistant_prefix_rows.to(generated_rows.device), generated_rows],
                dim=0,
            )
        else:
            rows = generated_rows
        bounds = _resolve_audio_payload_bounds(rows, data.model_config)
        if bounds is None:
            payload_rows = rows
        else:
            start, end = bounds
            payload_rows = rows[start:end]
            state.assistant_start_length = 0
        state.delayed_audio_codes = payload_rows[:, 1:].detach().cpu()
    else:
        n_vq = (
            int(data.prompt_rows.shape[1] - 1)
            if data.prompt_rows is not None and data.prompt_rows.ndim == 2
            else 0
        )
        state.delayed_audio_codes = torch.empty((0, n_vq), dtype=torch.long)

    state.prompt_tokens = len(data.input_ids) if data.input_ids is not None else 0
    state.completion_tokens = len(data.output_rows)
    state.engine_time_s = time.perf_counter() - data.engine_start_s
    return StagePayload(
        request_id=payload.request_id,
        request=payload.request,
        data=state.to_dict(),
    )


def make_moss_tts_scheduler_adapters(*, model: Any):
    """Build StagePayload <-> SGLang request adapters for MOSS-TTS."""

    def request_builder(payload: StagePayload) -> MossTTSSGLangRequestData:
        return build_sglang_moss_tts_request(payload, model=model)

    def result_adapter(data: MossTTSSGLangRequestData) -> StagePayload:
        return apply_sglang_moss_tts_result(data.stage_payload, data)

    return request_builder, result_adapter
