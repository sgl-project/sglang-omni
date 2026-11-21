# SPDX-License-Identifier: Apache-2.0
"""Shared request-size and timing limits for speech serving transports."""

MAX_VOICE_UPLOAD_BYTES = 10 * 1024 * 1024
MULTIPART_OVERHEAD_BYTES = 64 * 1024  # multipart framing beyond the file part
MAX_VOICE_UPLOAD_BODY_BYTES = MAX_VOICE_UPLOAD_BYTES + MULTIPART_OVERHEAD_BYTES

# One hour of audio (the audio_chunking.max_total_audio_s default) is ~110 MiB as 16 kHz mono PCM or ~82 MiB as 192 kbit/s MP3, both within 128 MiB; an hour of raw 48 kHz stereo WAV (~659 MiB) is not.
MAX_SPEECH_TO_TEXT_UPLOAD_BYTES = 128 * 1024 * 1024
MAX_SPEECH_TO_TEXT_BODY_BYTES = (
    MAX_SPEECH_TO_TEXT_UPLOAD_BYTES + MULTIPART_OVERHEAD_BYTES
)

MAX_REFERENCE_AUDIO_BYTES = 10 * 1024 * 1024
_BASE64_ENCODED_REFERENCE_AUDIO_BYTES = ((MAX_REFERENCE_AUDIO_BYTES + 2) // 3) * 4

# note (Yucheng Hu): finished raw PCM stream outcomes kept for GET
# /v1/audio/speech/{request_id}; older ones are evicted first.
MAX_SPEECH_STREAM_OUTCOMES = 1024

SPEECH_WS_CONFIG_TIMEOUT_S = 10.0
MAX_SPEECH_WS_CONFIG_MESSAGE_BYTES = _BASE64_ENCODED_REFERENCE_AUDIO_BYTES + 1024 * 1024
MAX_SPEECH_WS_TEXT_MESSAGE_BYTES = 128 * 1024
