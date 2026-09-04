# SPDX-License-Identifier: Apache-2.0
"""G.711 (µ-law / A-law) telephony audio helpers."""

from __future__ import annotations

import io

import soundfile as sf

from sglang_omni.utils.audio import is_riff_wav, is_sun_au

# ITU-T G.711 fixes the telephony sample rate and both companding laws:
# https://www.itu.int/rec/T-REC-G.711
G711_SAMPLE_RATE = 8000

MULAW = "mulaw"
ALAW = "alaw"

# libsndfile subtype names; it decodes and encodes both laws for us.
_SOUNDFILE_SUBTYPES = {MULAW: "ULAW", ALAW: "ALAW"}

# Media types telephony providers and SIP stacks attach to raw G.711 bytes.
# audio/basic is 8 kHz µ-law per RFC 2046 section 4.3:
# https://www.rfc-editor.org/rfc/rfc2046#section-4.3
# audio/PCMU and audio/PCMA are the RTP payload names from RFC 4856:
# https://www.rfc-editor.org/rfc/rfc4856
# audio/x-mulaw is what Twilio Media Streams put in their mediaFormat:
# https://www.twilio.com/docs/voice/media-streams/websocket-messages
_MULAW_CONTENT_TYPES = frozenset({"audio/basic", "audio/pcmu", "audio/x-mulaw"})
_ALAW_CONTENT_TYPES = frozenset({"audio/pcma"})

# Media types that carry no format information, so the filename decides.
_GENERIC_CONTENT_TYPES = frozenset({"", "application/octet-stream", "audio/*"})

# Extensions for headerless G.711: .ul/.al are what ffmpeg and sox use
# (`ffmpeg -h demuxer=mulaw`), .ulaw/.alaw are Asterisk's recording formats:
# https://docs.asterisk.org/Getting-Started/Installing-Asterisk/Installing-Asterisk-From-Source/Exploring-Sound-Prompts/
_MULAW_EXTENSIONS = frozenset({".ulaw", ".ul"})
_ALAW_EXTENSIONS = frozenset({".alaw", ".al"})


def resolve_g711_encoding(
    content_type: str | None, filename: str | None = None
) -> str | None:
    """Work out whether a caller declared raw G.711 audio."""
    normalized = (content_type or "").split(";", 1)[0].strip().lower()
    if normalized in _MULAW_CONTENT_TYPES:
        return MULAW
    if normalized in _ALAW_CONTENT_TYPES:
        return ALAW
    if normalized not in _GENERIC_CONTENT_TYPES:
        return None

    name = (filename or "").strip().lower()
    dot = name.rfind(".")
    extension = name[dot:] if dot >= 0 else ""
    if extension in _MULAW_EXTENSIONS:
        return MULAW
    if extension in _ALAW_EXTENSIONS:
        return ALAW
    return None


def wrap_g711_as_wav(
    data: bytes, encoding: str, sample_rate: int = G711_SAMPLE_RATE
) -> bytes:
    """Put headerless G.711 bytes into a WAV container of the same encoding."""
    if is_riff_wav(data) or is_sun_au(data):
        return data
    try:
        subtype = _SOUNDFILE_SUBTYPES[encoding]
    except KeyError:
        raise ValueError(f"Unsupported G.711 encoding: {encoding!r}") from None
    pcm, _ = sf.read(
        io.BytesIO(data),
        samplerate=sample_rate,
        channels=1,
        subtype=subtype,
        format="RAW",
        dtype="float32",
    )
    buffer = io.BytesIO()
    sf.write(buffer, pcm, sample_rate, subtype=subtype, format="WAV")
    return buffer.getvalue()
