# SPDX-License-Identifier: Apache-2.0
"""Token layout for the dense Ming-3.0-tts-0.5B checkpoint."""

from __future__ import annotations

from transformers import PreTrainedTokenizerBase

DENSE_SYSTEM_PREFIX = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
DENSE_USER_PREFIX = "<|im_start|>user\n"
DENSE_USER_SUFFIX = "<|im_end|>\n"
DENSE_ASSISTANT_PREFIX = "<|im_start|>assistant\n"
DENSE_SPEAKER_START = "<|vision_start|>"
DENSE_SPEAKER_PAD = "<|vision_pad|>"
DENSE_SPEAKER_END = "<|vision_end|>\n"
DEFAULT_TTS_PROMPT = "Please generate speech based on the following description.\n"
TEXT_INPUT_PREFIX = " Text input:\n"
AUDIO_START = "<audio>"
AUDIO_PATCH = "<audioPatch>"
SPEAKER_LABEL = "  speaker_1:"

# The released checkpoint was trained against these embedding rows.
DENSE_TOKEN_IDS = {
    "<|endoftext|>": 151643,
    "<|im_start|>": 151644,
    "<|im_end|>": 151645,
    "<|vision_start|>": 151652,
    "<|vision_end|>": 151653,
    "<|vision_pad|>": 151654,
    "<audioPatch>": 151705,
    "<audio>": 151706,
}
AUDIO_PATCH_TOKEN_ID = DENSE_TOKEN_IDS["<audioPatch>"]
AUDIO_START_TOKEN_ID = DENSE_TOKEN_IDS["<audio>"]
STOP_TOKEN_ID = DENSE_TOKEN_IDS["<|im_end|>"]
DENSE_SAMPLING_TEMPERATURE = 0.0
DENSE_STOP_THRESHOLD = 0.5
# The reference loop can stop only after step index 4, the fifth latent.
DENSE_MINIMUM_DECODE_STEPS = 5
DENSE_MAXIMUM_DECODE_STEPS = 1000
DENSE_PATCH_SIZE = 1


def validate_dense_tokenizer(tokenizer: PreTrainedTokenizerBase) -> None:
    """Reject a tokenizer whose special-token rows differ from training."""
    mismatched = {
        token: (expected_id, tokenizer.convert_tokens_to_ids(token))
        for token, expected_id in DENSE_TOKEN_IDS.items()
        if tokenizer.convert_tokens_to_ids(token) != expected_id
    }
    if mismatched:
        details = ", ".join(
            f"{token}: expected {expected_id}, got {actual_id}"
            for token, (expected_id, actual_id) in sorted(mismatched.items())
        )
        raise ValueError(
            f"Tokenizer ids do not match Ming-3.0-tts-0.5B training: {details}"
        )
    else:
        pass


def build_dense_speech_prompt(
    tokenizer: PreTrainedTokenizerBase,
    text: str,
    *,
    prompt_text: str | None,
    prompt_latent_steps: int,
    include_speaker: bool,
) -> tuple[list[int], int, int]:
    """Return token ids, the speaker-pad index, and the first prompt-latent index.

    The released text-to-speech layout has no speaker slot. A slot exists only
    when a reference speaker embedding is written into it.
    """

    def encode(segment: str) -> list[int]:
        return tokenizer.encode(segment, add_special_tokens=False)

    if include_speaker:
        speaker_block = (
            encode(SPEAKER_LABEL)
            + encode(DENSE_SPEAKER_START)
            + encode(DENSE_SPEAKER_PAD)
            + encode(DENSE_SPEAKER_END)
        )
    else:
        speaker_block = []
    if prompt_latent_steps > 0 and prompt_text:
        prompt_text_ids = encode(prompt_text)
        latent_ids = encode(AUDIO_PATCH) * prompt_latent_steps
    else:
        prompt_text_ids = []
        latent_ids = []
    input_ids = (
        encode(DENSE_SYSTEM_PREFIX)
        + encode(DENSE_USER_PREFIX)
        + encode(DEFAULT_TTS_PROMPT)
        + speaker_block
        + encode(TEXT_INPUT_PREFIX)
        + prompt_text_ids
        + encode(text)
        + encode(DENSE_USER_SUFFIX)
        + encode(DENSE_ASSISTANT_PREFIX)
        + encode(AUDIO_START)
    )
    speaker_start_id = tokenizer.convert_tokens_to_ids(DENSE_SPEAKER_START)
    speaker_pad_id = tokenizer.convert_tokens_to_ids(DENSE_SPEAKER_PAD)
    speaker_positions = [
        position + 1
        for position, token_id in enumerate(input_ids)
        if token_id == speaker_start_id
    ]
    expected_speakers = 1 if include_speaker else 0
    if len(speaker_positions) != expected_speakers:
        raise RuntimeError(
            "dense Ming 3 TTS prompt has the wrong number of speaker slots"
        )
    else:
        pass
    if include_speaker and input_ids[speaker_positions[0]] != speaker_pad_id:
        raise RuntimeError("dense Ming 3 TTS prompt is missing its speaker slot")
    else:
        pass
    speaker_position = speaker_positions[0] if include_speaker else -1
    latent_position = len(input_ids) if latent_ids else -1
    input_ids.extend(latent_ids)
    return input_ids, speaker_position, latent_position
