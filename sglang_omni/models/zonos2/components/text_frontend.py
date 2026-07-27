# SPDX-License-Identifier: Apache-2.0
"""ZONOS2 TTS text frontend: text + conditioning knobs -> 2D prompt rows.

Each row is FRAME_WIDTH (10) columns: 9 audio columns (audio pad id 1025) plus
a trailing text / conditioning token column. Constants are baked to the shipped
checkpoint's text_vocab=519 layout, whose tail (offset 448..518) packs the
speaking-rate, quality, speaker-background, and accurate-mode buckets in order.
"""

from __future__ import annotations

import logging
import os
import re
import threading
from dataclasses import dataclass

import torch

from sglang_omni.scheduling.stage_cache import StageOutputCache

logger = logging.getLogger(__name__)

# Vocab / frame layout (params.json).

N_CODEBOOKS = 9
CODEBOOK_SIZE = 1024
AUDIO_VOCAB = 1026
EOA_ID = 1024
AUDIO_PAD_ID = 1025
FRAME_WIDTH = N_CODEBOOKS + 1

TEXT_VOCAB = 519  # doubles as the text-column pad / speaker-slot id

# Byte tokenizer: 192 legacy symbol ids precede the 256 byte ids.
PAD_ID, UNK_ID, BOS_ID, EOS_ID = 0, 1, 2, 3
LEGACY_SYMBOL_VOCAB_SIZE = 192
BYTE_VOCAB_SIZE = 256

SPEAKING_RATE_NUM_BUCKETS = 8
QUALITY_BUCKET_COUNTS: tuple[int, ...] = (12, 12, 12, 8, 8, 8)
SPEAKER_BACKGROUND_NUM_BUCKETS = 2
ACCURATE_MODE_NUM_BUCKETS = 1

# First conditioning id; ids below it are plain text vocab. 519-8-60-2-1 == 448.
_CONDITIONING_BASE = (
    TEXT_VOCAB
    - SPEAKING_RATE_NUM_BUCKETS
    - sum(QUALITY_BUCKET_COUNTS)
    - SPEAKER_BACKGROUND_NUM_BUCKETS
    - ACCURATE_MODE_NUM_BUCKETS
)

# Silence tokens for 0.2s at 44.1kHz (17 frames x 9 codebooks).
_SILENCE_TOKENS_0_2S = [
    [568, 778, 338, 524, 967, 360, 728, 550, 90],
    [568, 778, 10, 674, 364, 981, 741, 378, 731],
    [568, 804, 10, 674, 364, 981, 568, 378, 731],
    [568, 804, 10, 674, 364, 981, 568, 378, 731],
    [568, 804, 10, 674, 364, 981, 568, 378, 731],
    [568, 804, 10, 674, 364, 981, 568, 378, 731],
    [568, 804, 10, 674, 364, 981, 568, 378, 731],
    [568, 804, 10, 674, 364, 981, 568, 378, 731],
    [568, 804, 10, 674, 364, 981, 568, 378, 731],
    [568, 804, 10, 674, 364, 981, 568, 378, 731],
    [568, 804, 10, 674, 364, 981, 568, 378, 731],
    [568, 804, 10, 674, 364, 981, 568, 378, 731],
    [568, 804, 10, 674, 364, 981, 568, 378, 731],
    [568, 804, 10, 674, 364, 981, 568, 378, 731],
    [568, 804, 10, 674, 364, 981, 568, 378, 731],
    [568, 804, 10, 674, 364, 981, 568, 378, 731],
    [568, 778, 721, 842, 264, 974, 989, 507, 308],
]


@dataclass
class TTSSamplingParams:
    """Sampling parameters for TTS generation; defaults match the checkpoint."""

    temperature: float = 1.15
    top_k: int = 106
    top_p: float = 0.0
    min_p: float = 0.18
    repetition_penalty: float = 1.2
    max_tokens: int = 1024
    ignore_eos: bool = False
    n_codebooks: int = N_CODEBOOKS
    eoa_id: int = EOA_ID
    repetition_window: int = 50
    repetition_codebooks: int = 8


# Speech API names and legacy server codes -> NeMo language packages.
_SERVER_TO_NEMO_LANG: dict[str, str] = {
    "English": "en",
    "French": "fr",
    "German": "de",
    "Spanish": "es",
    "Italian": "it",
    "Portuguese": "pt",
    "Japanese": "ja",
    "Chinese": "zh",
    "Korean": "ko",
    "en_us": "en",
    "en_gb": "en",
    "fr_fr": "fr",
    "de": "de",
    "es": "es",
    "it": "it",
    "pt_br": "pt",
    "ja": "ja",
    "cmn": "zh",
    "ko": "ko",
}

# Upstream's own tests run Korean grammars with lower_cased input; everything
# else uses cased.
_LOWER_CASED_LANGS = {"ko"}

# zh/ja verbalizers read a cached .far but never write one (upstream quirk);
# we post-write it after first compile so later loads are fast. Note ja reads
# a "jp_" prefixed name.
_VERBALIZER_FAR_PREFIX = {"zh": "zh", "ja": "jp"}

# A digit directly followed by sentence punctuation confuses several upstream
# r1.2.0 grammars: pt's tagger raises FstOpError outright and de reads dates
# digit-by-digit. Space the punctuation off before normalization; it is
# re-attached afterwards.
_DIGIT_PUNCT_RE = re.compile(r"(\d)([.!?,;:])(?=\s|$)")
_SPACE_PUNCT_RE = re.compile(r" +([.!?,;:])(?=\s|$)")

# Moses-based punct_post_process re-attaches punctuation well for these
# languages. For the European languages it also glues currency symbols to the
# following word ("5,32 € am" -> "€am"), so there we skip moses and only
# collapse the spacing we introduced ourselves.
_MOSES_POSTPROCESS_LANGS = {"en", "zh", "ja", "ko"}

_NORMALIZER = None

# note (Yue Yin): normalize_text is a pure function of (text, language), so an
# LRU on the shared StageOutputCache spares the NeMo FST pipeline on repeated
# prompts (batched evals, retries). Cap mirrors the speaker-encoder cache (256);
# StageOutputCache is not internally locked, so keep the module lock.
_NORMALIZE_CACHE = StageOutputCache(max_size=256, cache_device="cpu")
_NORMALIZE_CACHE_LOCK = threading.Lock()


# Set once by the preprocessing stage from its ``tts_norm_cache_dir`` factory arg
# (None -> the per-user default below). Module-level to match the lazy normalizer
# singleton, which is built on the first request after the stage is constructed.
_TTS_NORM_CACHE_ROOT: str | None = None


def configure_tts_norm_cache_root(path: str | None) -> None:
    """Configure the NeMo normalizer cache dir (preprocessing ``tts_norm_cache_dir``)."""
    global _TTS_NORM_CACHE_ROOT
    _TTS_NORM_CACHE_ROOT = path or None


def _default_cache_root() -> str:
    return _TTS_NORM_CACHE_ROOT or os.path.expanduser("~/.cache/zonos2-tts-norm")


class TTSTextNormalizer:
    """Lazy per-language NeMo normalizers with .far caching.

    Construction and calls are serialized per language: the upstream
    Normalizer shares a mutable TokenParser and is not thread-safe.
    """

    def __init__(self, cache_root: str | None = None):
        self.cache_root = cache_root or _default_cache_root()
        self._normalizers: dict[str, object] = {}
        self._locks: dict[str, threading.Lock] = {}
        self._global_lock = threading.Lock()

    def _lang_lock(self, lang: str) -> threading.Lock:
        with self._global_lock:
            if lang not in self._locks:
                self._locks[lang] = threading.Lock()
            return self._locks[lang]

    def _build(self, lang: str):
        from nemo_text_processing.text_normalization.normalize import Normalizer

        input_case = "lower_cased" if lang in _LOWER_CASED_LANGS else "cased"
        # One cache dir per (lang, case): upstream .far filenames collide
        # across languages (e.g. ja's tagger writes a zh_-prefixed file).
        cache_dir = os.path.join(self.cache_root, f"{lang}_{input_case}")
        os.makedirs(cache_dir, exist_ok=True)

        logger.info("Loading TTS text normalizer for '%s' (%s)...", lang, input_case)
        normalizer = Normalizer(
            input_case=input_case,
            lang=lang,
            cache_dir=cache_dir,
            overwrite_cache=False,
        )

        prefix = _VERBALIZER_FAR_PREFIX.get(lang)
        if prefix is not None:
            far_path = os.path.join(
                cache_dir, f"{prefix}_tn_True_deterministic_verbalizer.far"
            )
            if not os.path.exists(far_path):
                from nemo_text_processing.text_normalization.en.graph_utils import (
                    generator_main,
                )

                generator_main(far_path, {"verbalize": normalizer.verbalizer.fst})
        return normalizer

    def get(self, lang: str):
        with self._lang_lock(lang):
            if lang not in self._normalizers:
                self._normalizers[lang] = self._build(lang)
            return self._normalizers[lang]

    def warmup(self, languages: list[str] | None = None) -> None:
        """Construct normalizers ahead of time (server codes or NeMo codes)."""
        langs = languages or sorted(set(_SERVER_TO_NEMO_LANG.values()))
        for lang in langs:
            lang = _SERVER_TO_NEMO_LANG.get(lang, lang)
            try:
                self.get(lang)
            except Exception:  # noqa: BLE001
                logger.exception("TTS text normalizer warmup failed for '%s'", lang)

    def normalize(self, text: str, language: str) -> str:
        """Normalize text for the given server language code.

        Returns the input unchanged for unsupported languages or on any
        normalizer error -- normalization must never fail a request.
        """
        lang = _SERVER_TO_NEMO_LANG.get(language)
        if lang is None or not text.strip():
            return text
        text_in = _DIGIT_PUNCT_RE.sub(r"\1 \2", text)
        use_moses = lang in _MOSES_POSTPROCESS_LANGS
        try:
            normalizer = self.get(lang)
            with self._lang_lock(lang):
                result = normalizer.normalize(text_in, punct_post_process=use_moses)
        except Exception:  # noqa: BLE001
            logger.exception(
                "TTS text normalization failed for lang=%s; using raw text", language
            )
            return text
        if isinstance(result, str):
            result = _SPACE_PUNCT_RE.sub(r"\1", result)
        if not isinstance(result, str) or not result.strip():
            return text
        logger.debug("TTS norm [%s]: %r -> %r", language, text, result)
        return result


def _get_normalizer():
    """Lazily build the NeMo normalizer; ``None`` if its heavy deps are missing."""
    global _NORMALIZER
    if _NORMALIZER is not None:
        return _NORMALIZER
    try:
        normalizer = TTSTextNormalizer()
        # note (Yue Yin): probe-import so a missing nemo_text_processing /
        # pynini wheel degrades to None here rather than once per request.
        import importlib

        importlib.import_module("nemo_text_processing.text_normalization.normalize")
    except Exception:  # noqa: BLE001 - missing dep must never raise
        return None
    _NORMALIZER = normalizer
    return _NORMALIZER


def normalize_text(text: str, language: str | None) -> str:
    """Written->spoken normalization; returns ``text`` unchanged on any issue.

    Gated by the preprocessing stage's ``tts_norm`` factory arg (via
    ``build_prompt_rows(normalize=...)``); not called at all when disabled.
    """
    # note (luojiaxuan): Auto and languages without a NeMo package intentionally
    # bypass normalization so model-side language inference still receives raw text.
    if not language or language not in _SERVER_TO_NEMO_LANG:
        return text
    key = (text, language)
    with _NORMALIZE_CACHE_LOCK:
        cached = _NORMALIZE_CACHE.get(key)
        if cached is not None:
            return cached
    normalizer = _get_normalizer()
    if normalizer is None:
        return text
    try:
        result = normalizer.normalize(text, language)
    except Exception:  # noqa: BLE001 - normalization must never fail a request
        result = text
    if not (isinstance(result, str) and result.strip()):
        result = text
    with _NORMALIZE_CACHE_LOCK:
        _NORMALIZE_CACHE.put(key, result)
    return result


def text_to_byte_ids(text: str) -> list[int]:
    """UTF-8 byte tokenization: BOS, (byte + 192) per byte, EOS."""
    return [
        BOS_ID,
        *(byte + LEGACY_SYMBOL_VOCAB_SIZE for byte in text.encode("utf-8")),
        EOS_ID,
    ]


def speaking_rate_token_id(bucket: int) -> int:
    if bucket < 0 or bucket >= SPEAKING_RATE_NUM_BUCKETS:
        raise ValueError(
            f"speaking_rate_bucket must be in [0, {SPEAKING_RATE_NUM_BUCKETS - 1}], "
            f"got {bucket}."
        )
    return _CONDITIONING_BASE + int(bucket)


def quality_token_id(feature_idx: int, bucket: int) -> int:
    if feature_idx < 0 or feature_idx >= len(QUALITY_BUCKET_COUNTS):
        raise ValueError(
            f"quality feature index must be in [0, {len(QUALITY_BUCKET_COUNTS) - 1}], "
            f"got {feature_idx}."
        )
    num_buckets = QUALITY_BUCKET_COUNTS[feature_idx]
    if bucket < 0 or bucket >= num_buckets:
        raise ValueError(
            f"quality bucket for feature {feature_idx} must be in "
            f"[0, {num_buckets - 1}], got {bucket}."
        )
    return (
        _CONDITIONING_BASE
        + SPEAKING_RATE_NUM_BUCKETS
        + sum(QUALITY_BUCKET_COUNTS[:feature_idx])
        + int(bucket)
    )


def _audio_pad_row(text_id: int) -> list[int]:
    return [AUDIO_PAD_ID] * N_CODEBOOKS + [int(text_id)]


def _text_rows(
    tokens: list[int],
    *,
    speaking_rate_bucket: int | None = None,
    quality_buckets=None,
) -> list[list[int]]:
    rows: list[list[int]] = []
    if speaking_rate_bucket is not None:
        rows.append(_audio_pad_row(speaking_rate_token_id(speaking_rate_bucket)))
    if quality_buckets is not None:
        for feature_idx, bucket in enumerate(quality_buckets):
            if bucket is None:
                continue
            rows.append(_audio_pad_row(quality_token_id(feature_idx, bucket)))
    rows.extend(_audio_pad_row(token) for token in tokens)
    return rows


def shear(x: torch.Tensor, pad: int) -> torch.Tensor:
    """Apply the codebook delay pattern: column ``c`` is shifted down by ``c``."""
    T, C = x.shape
    padded = x.new_full((C - 1 + T, C), pad)
    padded[C - 1 :] = x
    row_idx = (
        (C - 1)
        + torch.arange(T, device=x.device).unsqueeze(1)
        - torch.arange(C, device=x.device)
    )
    return padded.gather(0, row_idx)


def silence_prompt_tokens() -> torch.Tensor:
    """Sheared 0.2s silence prompt as ``[T, FRAME_WIDTH]`` int32 rows."""
    silence = torch.tensor(_SILENCE_TOKENS_0_2S, dtype=torch.int32)
    sheared = shear(silence[:, :N_CODEBOOKS], AUDIO_PAD_ID)
    text_col = torch.full((sheared.shape[0], 1), TEXT_VOCAB, dtype=torch.int32)
    return torch.cat([sheared, text_col], dim=1)


def make_speaker_slot(
    *,
    dtype: torch.dtype = torch.int32,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """One speaker-slot row ``[1, FRAME_WIDTH]``: audio cols 1025, text col 519."""
    slot = torch.full((1, FRAME_WIDTH), AUDIO_PAD_ID, dtype=dtype, device=device)
    slot[:, N_CODEBOOKS] = TEXT_VOCAB
    return slot


def build_prompt_rows(
    text: str,
    *,
    language: str | None = None,
    speaking_rate_bucket: int | None = None,
    quality_buckets=None,
    normalize: bool = True,
) -> torch.Tensor:
    """Build the TTS prompt rows for ``text`` as a ``[T, FRAME_WIDTH]`` int32 tensor.

    Normalize -> byte-tokenize -> prepend conditioning rows -> append silence.
    """
    if normalize:
        text = normalize_text(text, language)
    tokens = text_to_byte_ids(text)
    rows = _text_rows(
        tokens,
        speaking_rate_bucket=speaking_rate_bucket,
        quality_buckets=quality_buckets,
    )
    prompt = torch.tensor(rows, dtype=torch.int32)
    return torch.cat([prompt, silence_prompt_tokens()], dim=0)
