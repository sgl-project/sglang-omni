# SPDX-License-Identifier: Apache-2.0
"""Safe tokenizer loader for Ming-Omni-TTS."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

BOS_TOKEN = "<|startoftext|>"
EOS_TOKEN = "<|endoftext|>"
PAD_TOKEN = EOS_TOKEN
CLS_TOKEN = "[CLS]"
ROLE_START_TOKEN = "<role>"
ROLE_END_TOKEN = "</role>"
AUDIO_PATCH_TOKEN = "<audioPatch>"
AUDIO_START_TOKEN = "<audio>"
END_OF_AUDIO_TOKEN = "<end_of_audio>"
SPK_START_TOKEN = "<spk>"
SPK_END_TOKEN = "</spk>"


@dataclass(frozen=True)
class MingTTSSpecialTokenIds:
    bos: int
    eos: int
    pad: int
    role_start: int
    role_end: int
    audio_patch: int
    audio_start: int
    end_of_audio: int
    spk_start: int
    spk_end: int


@dataclass(frozen=True)
class MingTTSTokenizerBundle:
    tokenizer: Any
    special: MingTTSSpecialTokenIds

    def encode_no_special(self, text: str) -> list[int]:
        return [
            int(token_id)
            for token_id in self.tokenizer.encode(text, add_special_tokens=False)
        ]


def load_ming_tts_tokenizer(
    model_path: str | Path,
    *,
    llm_config: Any | None = None,
) -> MingTTSTokenizerBundle:
    """Load the checkpoint fast tokenizer without importing remote tokenizer code."""

    from transformers import PreTrainedTokenizerFast

    tokenizer_file = Path(model_path) / "tokenizer.json"
    if not tokenizer_file.is_file():
        raise FileNotFoundError(
            "Ming-Omni-TTS requires tokenizer.json in the model directory; "
            f"missing {tokenizer_file}"
        )

    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=str(tokenizer_file),
        bos_token=BOS_TOKEN,
        eos_token=EOS_TOKEN,
        pad_token=PAD_TOKEN,
        cls_token=CLS_TOKEN,
        clean_up_tokenization_spaces=False,
        add_bos_token=False,
        add_eos_token=False,
    )

    def require_single_token(token: str) -> int:
        token_id = tokenizer.convert_tokens_to_ids(token)
        if token_id is None:
            raise ValueError(
                f"Ming-Omni-TTS tokenizer is missing required token {token!r}"
            )
        token_ids = tokenizer.encode(token, add_special_tokens=False)
        if token_ids != [token_id]:
            raise ValueError(
                "Ming-Omni-TTS tokenizer must encode required token as one id: "
                f"{token!r} -> {token_ids!r}, expected [{token_id}]"
            )
        return int(token_id)

    special = MingTTSSpecialTokenIds(
        bos=require_single_token(BOS_TOKEN),
        eos=require_single_token(EOS_TOKEN),
        pad=require_single_token(PAD_TOKEN),
        role_start=require_single_token(ROLE_START_TOKEN),
        role_end=require_single_token(ROLE_END_TOKEN),
        audio_patch=require_single_token(AUDIO_PATCH_TOKEN),
        audio_start=require_single_token(AUDIO_START_TOKEN),
        end_of_audio=require_single_token(END_OF_AUDIO_TOKEN),
        spk_start=require_single_token(SPK_START_TOKEN),
        spk_end=require_single_token(SPK_END_TOKEN),
    )

    if tokenizer.eos_token_id != special.eos:
        raise ValueError(
            "Ming-Omni-TTS tokenizer eos_token_id does not match "
            f"{EOS_TOKEN}: {tokenizer.eos_token_id} != {special.eos}"
        )
    if tokenizer.pad_token_id != special.eos:
        raise ValueError(
            "Ming-Omni-TTS tokenizer pad_token_id must match eos_token_id; "
            f"got {tokenizer.pad_token_id} and {special.eos}"
        )

    for role_prompt in ("<role>HUMAN</role>", "<role>ASSISTANT</role>"):
        token_ids = tokenizer.encode(role_prompt, add_special_tokens=False)
        if not token_ids or token_ids[0] != special.role_start:
            raise ValueError(
                f"Ming-Omni-TTS role prompt must start with {ROLE_START_TOKEN}; "
                f"{role_prompt!r} encoded as {token_ids!r}"
            )
        if token_ids[-1] != special.role_end:
            raise ValueError(
                f"Ming-Omni-TTS role prompt must end with {ROLE_END_TOKEN}; "
                f"{role_prompt!r} encoded as {token_ids!r}"
            )

    text = "Ming-Omni-TTS"
    with_special = tokenizer.encode(text, add_special_tokens=True)
    without_special = tokenizer.encode(text, add_special_tokens=False)
    if with_special != without_special:
        raise ValueError(
            "Ming-Omni-TTS tokenizer must not add implicit BOS/EOS tokens; "
            f"encode(add_special_tokens=True)={with_special!r}, "
            f"encode(add_special_tokens=False)={without_special!r}"
        )

    if llm_config is not None:
        if isinstance(llm_config, dict):
            eos_token_id = llm_config.get("eos_token_id")
            pad_token_id = llm_config.get("pad_token_id")
            vocab_size = llm_config.get("vocab_size")
        else:
            eos_token_id = getattr(llm_config, "eos_token_id", None)
            pad_token_id = getattr(llm_config, "pad_token_id", None)
            vocab_size = getattr(llm_config, "vocab_size", None)

        if eos_token_id is not None and int(eos_token_id) != special.eos:
            raise ValueError(
                "Ming-Omni-TTS tokenizer eos token does not match llm_config: "
                f"{special.eos} != {eos_token_id}"
            )
        if pad_token_id is not None and int(pad_token_id) != special.pad:
            raise ValueError(
                "Ming-Omni-TTS tokenizer pad token does not match llm_config: "
                f"{special.pad} != {pad_token_id}"
            )
        if vocab_size is not None:
            vocab_size = int(vocab_size)
            for name, token_id in special.__dict__.items():
                if token_id >= vocab_size:
                    raise ValueError(
                        "Ming-Omni-TTS special token id exceeds "
                        "llm_config.vocab_size: "
                        f"{name}={token_id}, vocab_size={vocab_size}"
                    )
    return MingTTSTokenizerBundle(tokenizer=tokenizer, special=special)
