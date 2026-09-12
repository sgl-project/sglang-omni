# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pytest

from sglang_omni.utils.audio import load_audio
from sglang_omni.utils.g711 import ALAW, MULAW, resolve_g711_encoding, wrap_g711_as_wav

_DATA_DIR = Path(__file__).resolve().parents[2] / "data"


def _sun_au_mulaw(num_samples: int = 8000) -> bytes:
    # Sun AU header: magic, header size, data size, encoding (1 = 8-bit
    # µ-law), sample rate, channels. Built by hand because Python 3.13 drops
    # the sunau module.
    payload = b"\xff" * num_samples
    return b".snd" + struct.pack(">IIIII", 24, len(payload), 1, 8000, 1) + payload


@pytest.mark.parametrize(
    ("content_type", "filename", "expected"),
    [
        ("audio/basic", None, MULAW),
        ("audio/PCMU", "call.bin", MULAW),
        ("audio/x-mulaw; rate=8000", None, MULAW),
        ("audio/PCMA", None, ALAW),
        ("audio/pcma; rate=8000", "call.wav", ALAW),
        # Spellings nobody registered or documents are not accepted.
        ("audio/x-alaw", "call.alaw", None),
        ("audio/mulaw", None, None),
        # Generic media types defer to the filename.
        (None, "call.ulaw", MULAW),
        ("", "CALL.UL", MULAW),
        ("application/octet-stream", "call.alaw", ALAW),
        ("audio/*", "call.al", ALAW),
        # A concrete non-G.711 media type wins over the extension.
        ("audio/wav", "call.ulaw", None),
        ("audio/mpeg", None, None),
        (None, "call.wav", None),
        (None, None, None),
    ],
)
def test_resolve_encoding_from_media_type_then_filename(
    content_type, filename, expected
) -> None:
    assert resolve_g711_encoding(content_type, filename) == expected


def test_wrap_matches_the_ffmpeg_reference_container() -> None:
    # Both fixtures come from the same ffmpeg conversion: the headerless
    # payload and the µ-law WAV ffmpeg wrote around it. Wrapping the payload
    # must decode to exactly what ffmpeg's container decodes to.
    raw = (_DATA_DIR / "query_to_draw_8k.ulaw").read_bytes()
    reference = (_DATA_DIR / "query_to_draw_8k_ulaw.wav").read_bytes()

    wav = wrap_g711_as_wav(raw, MULAW)

    assert wav[:4] == b"RIFF" and wav[8:12] == b"WAVE"
    assert wav.endswith(raw)
    np.testing.assert_array_equal(load_audio(wav), load_audio(reference))


@pytest.mark.parametrize("encoding", [MULAW, ALAW])
def test_wrap_keeps_one_sample_per_byte(encoding: str) -> None:
    payload = bytes(range(256)) * 4

    samples = load_audio(wrap_g711_as_wav(payload, encoding), target_sample_rate=8000)

    assert samples.shape == (len(payload),)
    assert samples.dtype == np.float32
    assert samples.min() >= -1.0 and samples.max() <= 1.0


def test_wrap_leaves_existing_wav_untouched() -> None:
    wav = wrap_g711_as_wav(b"\xff" * 10, MULAW)

    assert wrap_g711_as_wav(wav, MULAW) is wav


def test_wrap_leaves_sun_au_untouched() -> None:
    # mimetypes maps .au and .snd to audio/basic, the same media type raw
    # µ-law uses, so a client that sniffs by extension sends AU as
    # audio/basic. AU already carries a header; wrapping it would turn that
    # header into bogus samples.
    au = _sun_au_mulaw()

    assert wrap_g711_as_wav(au, MULAW) is au


def test_wrap_rejects_unknown_encoding() -> None:
    with pytest.raises(ValueError, match="Unsupported G.711 encoding"):
        wrap_g711_as_wav(b"\x00", "pcm16")


def test_wrap_copies_the_payload_verbatim_with_a_fact_chunk() -> None:
    payload = bytes(range(256)) * 4
    wav = wrap_g711_as_wav(payload, MULAW)

    fmt_tag, channels, sample_rate, _, block_align, bits = struct.unpack(
        "<HHIIHH", wav[20:36]
    )
    assert (fmt_tag, channels, sample_rate, block_align, bits) == (7, 1, 8000, 1, 8)
    fact_at = wav.index(b"fact")
    assert struct.unpack("<II", wav[fact_at + 4 : fact_at + 12]) == (4, len(payload))
    assert wav.index(b"data") > fact_at
    assert wav.endswith(payload)


def test_wrap_pads_odd_payloads_to_keep_riff_chunks_aligned() -> None:
    wav = wrap_g711_as_wav(b"\xff" * 3, ALAW)

    assert struct.unpack("<I", wav[4:8])[0] == len(wav) - 8
    assert len(wav) % 2 == 0
