# SPDX-License-Identifier: Apache-2.0
"""Request mapping helpers for CosyVoice3 zero-shot TTS.

This module mirrors ``qwen3_tts/request_builders.py``:

  * ``preprocess_cosyvoice3_payload`` runs the proven CosyVoice frontend and
    builds the projected LM prompt embeds (upstream ``cosyvoice/llm/llm.py``
    inference path), stashing the heavy tensor CPU-side in a module-global
    context keyed by ``request_id`` (the State only carries a marker), like
    Qwen3-TTS' ``_PREPARED_REQUESTS`` + ``_QWEN3_TTS_PREPARED_MARKER``.
  * ``build_sglang_cosyvoice3_request`` turns a preprocessed payload into a
    SGLang AR request (speech-token sampling with suppress + min_new_tokens).
  * ``apply_sglang_cosyvoice3_result`` reads ``data.output_ids`` (the generated
    speech tokens) into ``state.speech_tokens`` (cf. ``moss_tts`` result adapter).
"""

from __future__ import annotations

import hashlib
import os
import threading
import time
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Callable

import torch

from sglang_omni.models.cosyvoice3.payload_types import CosyVoice3State
from sglang_omni.proto import StagePayload
from sglang_omni.sampling.seed import derive_sampling_seed
from sglang_omni.scheduling.sglang_backend import SGLangARRequestData

# --- CosyVoice3 speech-token vocabulary layout -----------------------------
# speech_token_size = 6561 (3^8 FSQ); the speech embedding / decoder vocab is
# speech_token_size + 200 = 6761. Special rows live above the real speech tokens.
SPEECH_TOKEN_SIZE = 6561
SPEECH_VOCAB_SIZE = 6761
SOS_ID = SPEECH_TOKEN_SIZE + 0  # 6561
EOS_ID = SPEECH_TOKEN_SIZE + 1  # 6562 (eos_token in CosyVoice3LM)
# CosyVoice3LM.stop_token_ids = [speech_token_size + i for i in range(200)] — generation
# ends on ANY of the 200 special tokens (sos/eos/task_id/fill/...), not just eos_token.
STOP_TOKEN_IDS = list(range(SPEECH_TOKEN_SIZE, SPEECH_VOCAB_SIZE))  # [6561..6760]
TASK_ID = SPEECH_TOKEN_SIZE + 2  # 6563
ENDOFPROMPT_TOKEN = 151646  # Qwen <|endofprompt|>; must be present in the prompt text
# CosyVoice3's zero-shot prompt layout is `<instruction><|endofprompt|><reference transcript>`
# (per the model's own zero-shot example: "You are a helpful assistant.<|endofprompt|>…").
CV3_ZERO_SHOT_INSTRUCTION = "You are a helpful assistant."

# CosyVoice3 was trained with repetition-aware sampling (RAS). Without it the native top-k/top-p
# sampler gets stuck looping a silence token and never reaches a stop token, running to
# max_new_tokens (a long trailing blank). A repetition penalty demotes the looped token so the
# model reaches its natural stop. This is a pragmatic stand-in for the faithful windowed
# `ras_sampling` (see notes). It is the DEFAULT and is overridable per request.
#
# CAVEAT: this is squared. sglang-omni's shared runner applies the penalty manually
# (model_runner/base.py `_apply_repetition_penalty`, which reads sampling_params.repetition_penalty)
# AND sglang's native `BatchedRepetitionPenalizer` reads the SAME field and applies it again, so
# the EFFECTIVE factor is `repetition_penalty ** 2`. 1.5 (effective 2.25) reproduces the reference
# RAS stop length on the zero-shot eval. This is a shared-runner property, not CV3-specific:
# qwen3_tts routes through the same manual+native path and is squared too (its ~1.05 -> ~1.10 is
# negligible), whereas fishaudio_s2_pro / moss_tts sample inside custom runners that never invoke
# the shared manual hook and so apply the penalty exactly once. The duplicate manual application
# belongs in the shared runner to fix centrally (would affect qwen3_tts identically); until then CV3
# is tuned for the squared behavior. Both paths read the one field, so it cannot be split per-model
# without a runner change.
CV3_REPETITION_PENALTY = 1.5
COSYVOICE3_DEFAULT_TEMPERATURE = 1.0

# CosyVoice min/max token-to-text ratios applied to the (non-prompt) text length.
MIN_TOKEN_TEXT_RATIO = 2
MAX_TOKEN_TEXT_RATIO = 20

# Default ras_sampling knobs (top_p .8, top_k 25) from cosyvoice3.yaml.
COSYVOICE3_DEFAULT_TOP_K = 25
COSYVOICE3_DEFAULT_TOP_P = 0.8

# AR engine context window (must match `context_length` in stages.py). The prompt embeds plus
# the generation budget must fit here, else OmniScheduler rejects the request late; we clamp
# max_len (and reject over-long prompts early) against this.
CV3_CONTEXT_LENGTH = 4096
CV3_CONTEXT_SAFETY_MARGIN = 8

_COSYVOICE3_PREPARED_MARKER = "_cosyvoice3_prepared_request"


# ---------------------------------------------------------------------------
# Prepared-request handoff (preprocessing -> AR scheduler)
# ---------------------------------------------------------------------------


@dataclass
class CosyVoice3PreparedRequest:
    """Heavy CosyVoice3 preprocessing output consumed by the AR scheduler."""

    state: CosyVoice3State
    prompt_input_embeds: torch.Tensor  # [L, H] projected LM prompt hidden states
    min_len: int
    max_len: int


@dataclass
class CosyVoice3PreprocessingContext:
    """Engine objects shared with the preprocessing stage (same process)."""

    model: Any


_PREPROCESSING_CONTEXT: CosyVoice3PreprocessingContext | None = None
_PREPARED_REQUESTS: dict[str, CosyVoice3PreparedRequest] = {}
_PREPARED_REQUESTS_LOCK = threading.Lock()


def set_cosyvoice3_preprocessing_context(*, model: Any) -> None:
    """Register the engine model used by the preprocessing stage.

    Called by ``create_sglang_tts_engine_executor`` after the real ``llm.pt``
    weights are overlaid, so the preprocessing stage builds prompt embeds from
    the loaded ``speech_embedding`` / ``text_embedding``.
    """

    global _PREPROCESSING_CONTEXT
    with _PREPARED_REQUESTS_LOCK:
        _PREPROCESSING_CONTEXT = CosyVoice3PreprocessingContext(model=model)
        _PREPARED_REQUESTS.clear()


def clear_cosyvoice3_preprocessing_context() -> None:
    """Clear CosyVoice3 preprocessing globals (mainly for tests and reloads)."""

    global _PREPROCESSING_CONTEXT
    with _PREPARED_REQUESTS_LOCK:
        _PREPROCESSING_CONTEXT = None
        _PREPARED_REQUESTS.clear()


def _prepared_request_id(payload: StagePayload) -> str | None:
    data = payload.data
    if not isinstance(data, dict):
        return None
    marker = data.get(_COSYVOICE3_PREPARED_MARKER)
    return str(marker) if marker is not None else None


def pop_prepared_cosyvoice3_request(
    payload: StagePayload,
) -> CosyVoice3PreparedRequest | None:
    """Consume the prepared request referenced by a preprocessed payload."""

    prepared_request_id = _prepared_request_id(payload)
    if prepared_request_id is None:
        return None
    with _PREPARED_REQUESTS_LOCK:
        prepared = _PREPARED_REQUESTS.pop(prepared_request_id, None)
    if prepared is None:
        raise RuntimeError(
            "CosyVoice3 preprocessing state is missing for prepared payload "
            f"{prepared_request_id!r}; the AR scheduler must not rebuild it"
        )
    return prepared


def cleanup_prepared_cosyvoice3_request(request_id: str) -> None:
    """Drop any prepared CosyVoice3 handoff state for an aborted request."""

    with _PREPARED_REQUESTS_LOCK:
        _PREPARED_REQUESTS.pop(str(request_id), None)


# ---------------------------------------------------------------------------
# Radix-cache key for a precomputed embedding prefix (copied from qwen3_tts)
# ---------------------------------------------------------------------------


def build_embedding_cache_key_ids(input_embeds: torch.Tensor) -> list[int]:
    """Build stable radix-cache token ids for a precomputed embedding prefix."""
    rows = input_embeds.detach().to(dtype=torch.float32, device="cpu")
    key_ids: list[int] = []
    for row in rows:
        digest = hashlib.blake2b(row.numpy().tobytes(), digest_size=8).digest()
        key_ids.append(int.from_bytes(digest, "little") & ((1 << 63) - 1))
    return key_ids


def cosyvoice3_suppress_tokens() -> list[int]:
    """No suppression — like the reference, CosyVoice3 samples the full speech vocab and
    ends by emitting ANY special stop token; suppressing the other 199 specials (as before)
    prevented the model from ever emitting its natural stop, so it ran to max_new_tokens
    and produced a long trailing tail. `min_new_tokens` masks the stops until min length.
    """
    return []


def _min_new_tokens_tokenizer_shim() -> SimpleNamespace:
    """Tokenizer stand-in for sglang's `min_new_tokens` penalizer.

    CosyVoice3 uses `min_new_tokens` to guarantee a minimum speech length. sglang's penalty
    reads `req.tokenizer.{additional_stop_token_ids, eos_token_id}`, but the speech engine has
    no per-request tokenizer (qwen3_tts/moss set `req.tokenizer = None` precisely because they
    do not use `min_new_tokens`). This shim exposes just those two attributes so the penalty
    can union the stop set without a real tokenizer.
    """
    return SimpleNamespace(additional_stop_token_ids=None, eos_token_id=EOS_ID)


# ---------------------------------------------------------------------------
# Preprocessing — frontend + LM prompt-embed construction
# ---------------------------------------------------------------------------


def materialize_reference_audio(
    reference: dict[str, Any],
) -> tuple[Any, Callable[[], None] | None]:
    """Resolve a reference dict to something ``torchaudio.load`` accepts.

    Returns ``(audio, cleanup)``: ``audio`` is a filesystem path (the vendored
    ``load_wav`` reads the same reference three times, so a stable path is used
    rather than a one-shot buffer). ``cleanup`` removes any temp file written for
    inline base64/bytes payloads, or ``None`` for an on-disk path.
    """
    if not isinstance(reference, dict):
        raise ValueError("CosyVoice3 requires a reference (audio + text)")

    direct = (
        reference.get("audio_path")
        or reference.get("ref_audio")
        or reference.get("audio")
    )
    if isinstance(direct, str) and os.path.exists(direct):
        return direct, None

    raw = reference.get("bytes")
    if raw is None:
        data = reference.get("base64") or reference.get("data")
        if isinstance(data, str):
            import base64

            if data.startswith("data:") and "," in data:
                data = data.split(",", 1)[1]
            raw = base64.b64decode(data)

    if raw is None:
        if isinstance(direct, str):
            # A path/URL string torchaudio may resolve at serve time.
            return direct, None
        raise ValueError(
            "CosyVoice3 requires reference audio via references[0].audio_path "
            "or inline base64/bytes data"
        )

    import tempfile

    media_type = reference.get("media_type") or "audio/wav"
    suffix = "." + media_type.split("/")[-1].split(";")[0]
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    try:
        tmp.write(raw)
        tmp.flush()
    finally:
        tmp.close()

    def _cleanup() -> None:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass

    return tmp.name, _cleanup


def _normalize_cosyvoice3_inputs(inputs: Any) -> tuple[str, dict[str, Any], str | None]:
    if isinstance(inputs, str):
        return inputs, {}, None
    if isinstance(inputs, dict):
        text = inputs.get("text", inputs.get("input", ""))
        references = inputs.get("references") or []
        if not isinstance(references, list):
            raise ValueError("CosyVoice3 references must be a list")
        reference = references[0] if references else {}
        if not isinstance(reference, dict):
            reference = {}
        ref_text = reference.get("text")
        return (
            str(text),
            reference,
            str(ref_text) if ref_text is not None else None,
        )
    return (str(inputs) if inputs is not None else ""), {}, None


# Values the OpenAI speech service materializes as GENERIC defaults for every request
# (`speech_service.py`: temperature=0.8, top_p=0.8, top_k=30, repetition_penalty=1.1). They
# must NOT override CosyVoice3's own tuned defaults unless the caller set the field explicitly
# (signalled via `tts_params["explicit_generation_params"]`). Mirrors qwen3_tts — without this,
# the API path would use rep_penalty 1.1 (effective 1.21) instead of CosyVoice3's 1.5 (2.25),
# defeating the RAS stand-in.
_CV3_IMPLICIT_SAMPLING_DEFAULTS: dict[str, set] = {
    "temperature": {1.0, 0.8},
    "top_p": {1.0, 0.8},
    "top_k": {-1, 30},
    "repetition_penalty": {1.0, 1.1},
}


def _resolve_cosyvoice3_gen_params(payload: StagePayload) -> dict[str, Any]:
    """Per-request generation params from the OmniRequest (``params`` + ``tts_params``).

    Mirrors qwen3_tts / moss_tts: honors temperature / top_k / top_p / repetition_penalty /
    max_new_tokens / seed; a field left unset (or carrying only the service's generic default,
    unless flagged in ``explicit_generation_params``) falls back to the CosyVoice3 default.
    Invalid types raise a clear request error rather than silently mangling generation.
    """
    request = payload.request
    params = getattr(request, "params", None) or {}
    metadata = getattr(request, "metadata", None) or {}
    tts_params = metadata.get("tts_params") if isinstance(metadata, dict) else None
    if not isinstance(tts_params, dict):
        tts_params = {}
    explicit = tts_params.get("explicit_generation_params")
    explicit_fields = (
        {str(f) for f in explicit}
        if isinstance(explicit, (list, tuple, set))
        else set()
    )

    def _pick(name: str) -> tuple[Any, bool]:
        # Returns (value, from_tts_params). tts_params is the direct TTS metadata channel and is
        # caller-authored; the speech service only ever materializes generic defaults into
        # `params`. So a tts_params-sourced value is inherently explicit.
        v = tts_params.get(name)
        if v is not None:
            return v, True
        return params.get(name), False

    def _sampling(name: str, default: Any, cast: Callable[[Any], Any]) -> Any:
        raw, from_tts = _pick(name)
        if raw is None:
            return default
        if isinstance(raw, bool):
            raise ValueError(f"CosyVoice3 {name} must be numeric, got {raw!r}")
        # Drop a service-materialized generic default in `params` that the caller did not flag
        # explicit. Values from tts_params are caller-authored, so they are always honored
        # (even when they happen to equal a generic default such as temperature 0.8).
        if (
            not from_tts
            and name not in explicit_fields
            and raw in _CV3_IMPLICIT_SAMPLING_DEFAULTS.get(name, ())
        ):
            return default
        try:
            return cast(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"CosyVoice3 {name} must be numeric, got {raw!r}") from exc

    seed = _pick("seed")[0]
    if isinstance(seed, bool) or (seed is not None and not isinstance(seed, int)):
        raise ValueError(f"CosyVoice3 seed must be an integer, got {seed!r}")

    max_new = _pick("max_new_tokens")[0]
    if isinstance(max_new, bool) or (
        max_new is not None and (not isinstance(max_new, int) or max_new <= 0)
    ):
        raise ValueError(
            f"CosyVoice3 max_new_tokens must be a positive integer, got {max_new!r}"
        )

    # NOTE: OpenAI `speed` is intentionally absent here — the serving layer applies
    # it generically to the encoded waveform (openai_api.py `apply_speed`), the same
    # as every TTS sibling; consuming it in the adapter too would compound to
    # speed**2.

    return {
        "temperature": _sampling("temperature", COSYVOICE3_DEFAULT_TEMPERATURE, float),
        "top_k": _sampling("top_k", COSYVOICE3_DEFAULT_TOP_K, int),
        "top_p": _sampling("top_p", COSYVOICE3_DEFAULT_TOP_P, float),
        "repetition_penalty": _sampling(
            "repetition_penalty", CV3_REPETITION_PENALTY, float
        ),
        "max_new_tokens": max_new,
        "seed": seed,
    }


def _build_cosyvoice3_prompt_embeds(
    model: Any,
    *,
    text_token: torch.Tensor,
    prompt_text_token: torch.Tensor,
    prompt_speech_token: torch.Tensor,
) -> torch.Tensor:
    """Replicate ``CosyVoice3LM.inference`` prompt assembly (llm.py:485-505).

    ``lm_input = [ speech_embedding.weight[sos], text_embedding(prompt_text ++ text),
                   speech_embedding.weight[task_id], speech_embedding(prompt_speech_token) ]``
    """
    device = model.speech_embedding.weight.device
    text_token = text_token.to(device=device, dtype=torch.long)
    prompt_text_token = prompt_text_token.to(device=device, dtype=torch.long)
    prompt_speech_token = prompt_speech_token.to(device=device, dtype=torch.long)

    with torch.no_grad():
        full_text = torch.cat([prompt_text_token, text_token], dim=1)
        if ENDOFPROMPT_TOKEN not in full_text:
            raise ValueError(
                "CosyVoice3 prompt_text is missing the <|endofprompt|> token "
                f"({ENDOFPROMPT_TOKEN}); append '<|endofprompt|>' to the reference text"
            )
        text_emb = model.text_embedding(full_text)  # [1, Lp+Lt, H]
        sos_emb = model.speech_embedding.weight[SOS_ID].reshape(1, 1, -1)
        task_id_emb = model.speech_embedding.weight[TASK_ID].reshape(1, 1, -1)
        if prompt_speech_token.shape[1] != 0:
            prompt_speech_emb = model.speech_embedding(prompt_speech_token)
        else:
            prompt_speech_emb = torch.zeros(
                1, 0, model.hidden_size, dtype=text_emb.dtype, device=device
            )
        lm_input = torch.cat(
            [sos_emb, text_emb, task_id_emb, prompt_speech_emb], dim=1
        )  # [1, L, H]
    return lm_input.squeeze(0).detach()


def preprocess_cosyvoice3_payload(
    payload: StagePayload,
    *,
    frontend: Any,
) -> StagePayload:
    """Run the CosyVoice frontend + LM prompt-embed build outside the scheduler.

    ``frontend`` is the ``CosyVoiceFrontEnd`` instance built once by
    ``create_preprocessing_executor``; the engine model comes from the shared
    preprocessing context registered by ``create_sglang_tts_engine_executor``.
    """
    with _PREPARED_REQUESTS_LOCK:
        context = _PREPROCESSING_CONTEXT
    if context is None:
        raise RuntimeError(
            "CosyVoice3 preprocessing context is not initialized; "
            "create_sglang_tts_engine_executor must register it before requests run"
        )
    model = context.model

    inputs = payload.request.inputs or {}
    text, reference, ref_text = _normalize_cosyvoice3_inputs(inputs)

    if not text or not str(text).strip():
        raise ValueError("CosyVoice3 requires non-empty input text")
    # Zero-shot voice cloning needs the reference transcript; without it the prompt
    # degrades to a bare <|endofprompt|> and the clone quality silently collapses.
    if ref_text is None or not str(ref_text).strip():
        raise ValueError(
            "CosyVoice3 zero-shot requires a reference transcript "
            "(references[0].text)"
        )

    ref_audio, cleanup = materialize_reference_audio(reference)
    try:
        # Correct layout is instruction<|endofprompt|>reference_transcript (NOT transcript
        # then marker — that mis-conditions the clone). A ref_text already containing the
        # marker is treated as a fully-preformatted prompt and used verbatim, but only if it is
        # well formed: exactly one marker with non-empty text on both sides. A degenerate form
        # like "transcript<|endofprompt|>" (nothing after) would silently mis-condition, so
        # reject it plainly rather than pass it through.
        _rt = ref_text or ""
        if "<|endofprompt|>" in _rt:
            instruction_part, _, transcript_part = _rt.partition("<|endofprompt|>")
            if (
                _rt.count("<|endofprompt|>") != 1
                or not instruction_part.strip()
                or not transcript_part.strip()
            ):
                raise ValueError(
                    "CosyVoice3 preformatted reference text must be "
                    "'<instruction><|endofprompt|><reference transcript>' with exactly one "
                    "marker and non-empty text on both sides; pass a plain transcript to use "
                    "the default instruction."
                )
            prompt_text = _rt
        else:
            prompt_text = f"{CV3_ZERO_SHOT_INSTRUCTION}<|endofprompt|>{_rt}"
        model_input = frontend.frontend_zero_shot(
            text, prompt_text, ref_audio, 24000, ""
        )
    finally:
        if cleanup is not None:
            cleanup()

    text_token = model_input["text"]  # [1, Lt]
    text_len = int(model_input["text_len"].reshape(-1)[0].item())
    prompt_text_token = model_input["prompt_text"]  # [1, Lp]
    prompt_speech_token = model_input["llm_prompt_speech_token"]  # [1, T_ref]

    model_dtype = next(model.parameters()).dtype
    # Hand off on CPU: prepared requests wait in `_PREPARED_REQUESTS` / the AR waiting
    # queue for an unbounded time under burst load, and [L, H] GPU tensors would
    # accumulate into an OOM. The model runner moves the (~MB-scale) tensor back to
    # the engine device at prefill.
    prompt_input_embeds = _build_cosyvoice3_prompt_embeds(
        model,
        text_token=text_token,
        prompt_text_token=prompt_text_token,
        prompt_speech_token=prompt_speech_token,
    ).to(dtype=model_dtype, device="cpu")

    gen = _resolve_cosyvoice3_gen_params(payload)

    min_len = max(int(text_len * MIN_TOKEN_TEXT_RATIO), 1)
    # Generation budget = the text-ratio default, capped by an explicit request max_new_tokens.
    # A cap below min_len cannot produce valid audio (CosyVoice3 needs at least min_len speech
    # tokens for this text length); reject it plainly instead of silently generating min_len
    # anyway (the final clamp floors at min_len) and pretending the cap was honored.
    explicit_cap = gen["max_new_tokens"]
    if explicit_cap is not None and int(explicit_cap) < min_len:
        raise ValueError(
            f"CosyVoice3 max_new_tokens ({int(explicit_cap)}) is below the {min_len} speech "
            f"tokens required for {text_len} text tokens; the audio would be truncated to "
            "unusable. Increase max_new_tokens or shorten the input text."
        )
    max_len = max(int(text_len * MAX_TOKEN_TEXT_RATIO), min_len)
    if explicit_cap is not None:
        max_len = min(max_len, int(explicit_cap))
    # Clamp so prompt embeds + generation fit the AR context; reject an over-long prompt early
    # (clear error) instead of letting OmniScheduler reject it after all the frontend work.
    prompt_len = int(prompt_input_embeds.shape[0])
    budget = CV3_CONTEXT_LENGTH - prompt_len - CV3_CONTEXT_SAFETY_MARGIN
    if budget < min_len:
        raise ValueError(
            f"CosyVoice3 input too long: prompt ({prompt_len}) + minimum generation "
            f"({min_len}) exceeds the {CV3_CONTEXT_LENGTH}-token context. Shorten the input "
            "text or the reference audio (CosyVoice3 is single-utterance; it does not chunk)."
        )
    max_len = max(min(max_len, budget), min_len)

    state = CosyVoice3State(
        prepared=True,
        # The flow prompt token equals the llm prompt token in zero-shot.
        prompt_speech_token=model_input["flow_prompt_speech_token"]
        .reshape(-1)
        .detach()
        .to("cpu"),
        prompt_feat=model_input["prompt_speech_feat"]
        .detach()
        .to("cpu"),  # [1,T_mel,80]
        flow_embedding=model_input["flow_embedding"].detach().to("cpu"),  # [1,192]
        min_len=min_len,
        max_len=max_len,
        top_k=int(gen["top_k"]),
        top_p=float(gen["top_p"]),
        temperature=float(gen["temperature"]),
        repetition_penalty=float(gen["repetition_penalty"]),
        seed=gen["seed"],
    )

    prepared = CosyVoice3PreparedRequest(
        state=state,
        prompt_input_embeds=prompt_input_embeds,
        min_len=min_len,
        max_len=max_len,
    )
    with _PREPARED_REQUESTS_LOCK:
        _PREPARED_REQUESTS[payload.request_id] = prepared

    data = state.to_dict()
    data[_COSYVOICE3_PREPARED_MARKER] = payload.request_id
    return StagePayload(
        request_id=payload.request_id,
        request=payload.request,
        data=data,
    )


# ---------------------------------------------------------------------------
# AR request data + builders
# ---------------------------------------------------------------------------


@dataclass
class CosyVoice3SGLangRequestData(SGLangARRequestData):
    """CosyVoice3 scheduler-owned request state."""

    prompt_input_embeds: torch.Tensor | None = None
    state: CosyVoice3State | None = None
    engine_start_s: float = 0.0


def build_sglang_cosyvoice3_request(
    payload: StagePayload,
    *,
    model: Any,
) -> CosyVoice3SGLangRequestData:
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.sampling.sampling_params import SamplingParams

    prepared = pop_prepared_cosyvoice3_request(payload)
    if prepared is None:
        raise RuntimeError(
            "CosyVoice3 AR request builder requires a payload prepared by "
            "preprocess_cosyvoice3_payload"
        )

    state = prepared.state
    vocab_size = int(getattr(model, "speech_vocab_size", SPEECH_VOCAB_SIZE))
    suppress = cosyvoice3_suppress_tokens()

    input_ids_list = build_embedding_cache_key_ids(prepared.prompt_input_embeds)
    input_ids = torch.tensor(input_ids_list, dtype=torch.long)

    sampling_params = SamplingParams(
        max_new_tokens=int(prepared.max_len),
        min_new_tokens=int(prepared.min_len),
        temperature=float(state.temperature),
        top_k=int(state.top_k),
        top_p=float(state.top_p),
        # RAS stand-in; effective factor is this ** 2 (double-applied, see CV3_REPETITION_PENALTY).
        repetition_penalty=float(state.repetition_penalty),
        stop_token_ids=list(STOP_TOKEN_IDS),
    )
    sampling_params.normalize(None)
    sampling_params.verify(vocab_size)
    # Reproducibility: a request ``seed`` routes to sglang's seeded sampler. Left None the request
    # is unseeded (nondeterministic); the base runner's `_install_sampling_seeds` returns early when
    # every row in a batch is unseeded, and only derives a request-id-stable fallback for a None row
    # when it shares a batch with seeded rows (keeping that mixed batch TP-consistent).
    sampling_params.sampling_seed = (
        derive_sampling_seed("cosyvoice3", int(state.seed))
        if state.seed is not None
        else None
    )

    req = Req(
        rid=payload.request_id,
        origin_input_text="",
        origin_input_ids=input_ids_list,
        sampling_params=sampling_params,
        eos_token_ids=set(STOP_TOKEN_IDS),
        vocab_size=vocab_size,
    )
    req.tokenizer = _min_new_tokens_tokenizer_shim()
    req._input_embeds_are_projected = True
    req._codec_suppress_tokens = tuple(suppress)

    data = CosyVoice3SGLangRequestData(
        input_ids=input_ids,
        max_new_tokens=int(prepared.max_len),
        temperature=float(state.temperature),
        top_k=int(state.top_k),
        top_p=float(state.top_p),
        output_ids=req.output_ids,
        req=req,
        prompt_input_embeds=prepared.prompt_input_embeds,
        state=state,
        engine_start_s=time.perf_counter(),
    )
    data.suppress_tokens = list(suppress)
    data.input_embeds_are_projected = True
    data.stage_payload = payload
    return data


def apply_sglang_cosyvoice3_result(
    payload: StagePayload,
    data: CosyVoice3SGLangRequestData,
) -> StagePayload:
    state = data.state if data.state is not None else CosyVoice3State()

    output_ids = list(data.output_ids or [])
    # Keep only real speech tokens (0..6560); drop EOS / any leaked special row.
    speech_tokens = [int(t) for t in output_ids if 0 <= int(t) < SPEECH_TOKEN_SIZE]

    state.speech_tokens = speech_tokens
    state.prompt_tokens = (
        int(data.input_ids.shape[0]) if data.input_ids is not None else 0
    )
    state.completion_tokens = len(output_ids)
    state.engine_time_s = time.perf_counter() - data.engine_start_s
    # Propagate the engine's stop reason so a length-truncated generation is
    # distinguishable from a natural stop-token stop downstream.
    state.finish_reason = getattr(data, "finish_reason", None)
    # Release the heavy prompt-embeds tensor eagerly: the data<->req reference cycle can
    # otherwise keep it alive until cyclic GC (OmniScheduler's generic completion cleanup does
    # not clear this adapter-specific field). NOTE: this covers the normal completion path only.
    # A request aborted BEFORE this adapter runs (e.g. cancelled while waiting, or a mid-generation
    # error) still frees the tensor via cyclic GC, since the abort callback receives only a
    # request id and cannot reach `data`. That bound matches the sibling adapters (qwen3_tts does
    # not clear the field at all); fixing the abort case would require the shared scheduler to hand
    # the request data to the abort callback.
    data.prompt_input_embeds = None
    return StagePayload(
        request_id=payload.request_id,
        request=payload.request,
        data=state.to_dict(),
    )


def make_cosyvoice3_scheduler_adapters(*, model: Any):
    """Build StagePayload <-> SGLang request adapters for CosyVoice3."""

    def request_builder(payload: StagePayload) -> CosyVoice3SGLangRequestData:
        return build_sglang_cosyvoice3_request(payload, model=model)

    def result_adapter(data: CosyVoice3SGLangRequestData) -> StagePayload:
        return apply_sglang_cosyvoice3_result(data.stage_payload, data)

    return request_builder, result_adapter
