# SPDX-License-Identifier: Apache-2.0
"""What the Qwen3-Omni node of a Session graph accepts and publishes."""

from sglang_omni.proto.streams import (
    COMMIT,
    CONFIGURE,
    IMAGE_FRAME,
    INTERRUPT,
    PCM16,
    PCM_F32,
    SESSION_INPUT_SAMPLE_RATE_HZ,
    SESSION_UNIT_MS,
    TRUNCATE,
    AudioPort,
    ImagePort,
    SessionPorts,
)

OUTPUT_SAMPLE_RATE_HZ = 24000
SESSION_PORTS = SessionPorts(
    input=(AudioPort(PCM16, SESSION_INPUT_SAMPLE_RATE_HZ), ImagePort(IMAGE_FRAME)),
    output=AudioPort(PCM_F32, OUTPUT_SAMPLE_RATE_HZ),
    unit_ms=SESSION_UNIT_MS,
    controls=frozenset({COMMIT, CONFIGURE, INTERRUPT, TRUNCATE}),
    hot_settings=frozenset({"sampling"}),
    sampling_parameters=(
        "temperature",
        "max_new_tokens",
        "talker_max_new_tokens",
        "talker_temperature",
        "talker_top_k",
        "talker_top_p",
        "talker_repetition_penalty",
        "seed",
    ),
)
