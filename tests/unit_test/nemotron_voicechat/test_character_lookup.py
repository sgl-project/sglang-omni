# SPDX-License-Identifier: Apache-2.0
"""Character batches retain the tokenizer's token text and embedding output."""

from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer
from transformers import PreTrainedTokenizerBase, PreTrainedTokenizerFast

from sglang_omni.models.nemotron_voicechat.talker import TalkerEmbedding
from sglang_omni.models.nemotron_voicechat.talker_model_runner import (
    NemotronVoiceChatTalkerModelRunner,
    char_vocab_from_tokenizer,
)


class SparseTokenizer(PreTrainedTokenizerBase):
    def __init__(self, unknown_token: str | None = None) -> None:
        super().__init__()
        self.unknown_token: str | None = unknown_token
        self.token_strings: dict[int, str | None] = {
            0: "a",
            1: "b",
            5: "ab",
            50: "",
            60: None,
            70: "🙂🤖",
            100: "abba" * 8,
        }

    def get_vocab(self) -> dict[str, int]:
        return {
            "a": 0,
            "b": 1,
            "different_vocab_text": 5,
            "empty": 50,
            "missing": 60,
            "filtered": 70,
            "long": 100,
        }

    def convert_ids_to_tokens(
        self, ids: int, skip_special_tokens: bool = False
    ) -> str | None:
        return self.token_strings.get(ids, self.unknown_token)


@dataclass(kw_only=True)
class CharacterLookupModel:
    fusion_buffer: torch.Tensor


def make_runner(
    tokenizer: PreTrainedTokenizerBase,
) -> NemotronVoiceChatTalkerModelRunner:
    runner = NemotronVoiceChatTalkerModelRunner.__new__(
        NemotronVoiceChatTalkerModelRunner
    )
    runner.tokenizer = tokenizer
    runner.char_vocab = char_vocab_from_tokenizer(tokenizer)
    runner.char_padding_idx = len(runner.char_vocab)
    runner.model = CharacterLookupModel(fusion_buffer=torch.empty(0))
    runner.initialize_character_lookup()
    return runner


@pytest.fixture
def sparse_runner() -> NemotronVoiceChatTalkerModelRunner:
    return make_runner(SparseTokenizer())


@pytest.fixture
def byte_bpe_runner() -> NemotronVoiceChatTalkerModelRunner:
    tokenizer_backend = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer_backend.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer_backend.train_from_iterator(
        ["a hello world 中文 😀🤖", "hello 中文 😀🤖"],
        BpeTrainer(
            vocab_size=300,
            initial_alphabet=ByteLevel.alphabet(),
            special_tokens=["<unk>"],
        ),
    )
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer_backend, unk_token="<unk>"
    )
    tokenizer.add_tokens(["addedabcdef"])
    tokenizer.add_special_tokens({"additional_special_tokens": ["<special>"]})
    return make_runner(tokenizer)


def original_char_batch(
    runner: NemotronVoiceChatTalkerModelRunner, token_ids: list[int]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    device = runner.fusion_device()
    character_sequences = [
        [
            runner.char_vocab[character]
            for character in (runner.tokenizer.convert_ids_to_tokens(token_id) or "")
            if character in runner.char_vocab
        ]
        or [runner.char_padding_idx]
        for token_id in token_ids
    ]
    character_width = max(len(sequence) for sequence in character_sequences)
    character_ids = torch.full(
        (len(character_sequences), character_width),
        runner.char_padding_idx,
        dtype=torch.long,
        device=device,
    )
    for row, sequence in enumerate(character_sequences):
        character_ids[row, : len(sequence)] = torch.tensor(sequence, device=device)
    character_lengths = torch.tensor(
        [len(sequence) for sequence in character_sequences], device=device
    )
    return torch.tensor(token_ids, device=device), character_ids, character_lengths


def make_embedding(runner: NemotronVoiceChatTalkerModelRunner) -> TalkerEmbedding:
    return TalkerEmbedding(
        {
            "hidden_size": 16,
            "vocab_size": 1024,
            "char_vocab_size": runner.char_padding_idx,
            "char_encoder_config": {
                "encoder": {
                    "hidden_size": 16,
                    "intermediate_size": 32,
                    "num_hidden_layers": 1,
                    "num_attention_heads": 2,
                    "num_key_value_heads": 1,
                    "head_dim": 8,
                    "attention_dropout": 0.0,
                }
            },
        }
    ).eval()


@pytest.mark.parametrize(
    "token_ids",
    [[0], [5], [50], [60], [70], [100], [2], [0, 1, 5, 0], [100, 0, 50, 60, 70]],
)
def test_sparse_character_batches_match_original(
    sparse_runner: NemotronVoiceChatTalkerModelRunner, token_ids: list[int]
) -> None:
    actual_batch = sparse_runner.char_batch(token_ids)
    expected_batch = original_char_batch(sparse_runner, token_ids)
    for actual_tensor, expected_tensor in zip(
        actual_batch, expected_batch, strict=True
    ):
        torch.testing.assert_close(actual_tensor, expected_tensor, rtol=0, atol=0)


def test_sparse_embedding_matches_original(
    sparse_runner: NemotronVoiceChatTalkerModelRunner,
) -> None:
    embedding = make_embedding(sparse_runner)
    for token_ids in [[0], [5], [100], [50, 60, 70, 2], [100, 0, 5, 0]]:
        actual_batch = sparse_runner.char_batch(token_ids)
        expected_batch = original_char_batch(sparse_runner, token_ids)
        torch.testing.assert_close(
            embedding(*actual_batch), embedding(*expected_batch), rtol=0, atol=0
        )


def test_byte_bpe_embedding_matches_original(
    byte_bpe_runner: NemotronVoiceChatTalkerModelRunner,
) -> None:
    embedding = make_embedding(byte_bpe_runner)
    tokenizer = byte_bpe_runner.tokenizer
    token_ids = tokenizer.encode(
        "a hello 中文 😀🤖 addedabcdef <special>", add_special_tokens=False
    )
    for batch_token_ids in [[token_id] for token_id in token_ids] + [token_ids * 2]:
        actual_batch = byte_bpe_runner.char_batch(batch_token_ids)
        expected_batch = original_char_batch(byte_bpe_runner, batch_token_ids)
        for actual_tensor, expected_tensor in zip(
            actual_batch, expected_batch, strict=True
        ):
            torch.testing.assert_close(actual_tensor, expected_tensor, rtol=0, atol=0)
        torch.testing.assert_close(
            embedding(*actual_batch), embedding(*expected_batch), rtol=0, atol=0
        )


@pytest.mark.parametrize("unknown_token", [None, "ab"])
@pytest.mark.parametrize("token_ids", [[-1], [101], [0, 101, -1, 5]])
def test_unknown_ids_retain_tokenizer_behavior(
    unknown_token: str | None, token_ids: list[int]
) -> None:
    runner = make_runner(SparseTokenizer(unknown_token))
    actual_batch = runner.char_batch(token_ids)
    expected_batch = original_char_batch(runner, token_ids)
    for actual_tensor, expected_tensor in zip(
        actual_batch, expected_batch, strict=True
    ):
        torch.testing.assert_close(actual_tensor, expected_tensor, rtol=0, atol=0)


def test_fast_tokenizer_unknown_id_matches_original(
    byte_bpe_runner: NemotronVoiceChatTalkerModelRunner,
) -> None:
    token_ids = [max(byte_bpe_runner.tokenizer.get_vocab().values()) + 17]
    actual_batch = byte_bpe_runner.char_batch(token_ids)
    expected_batch = original_char_batch(byte_bpe_runner, token_ids)
    for actual_tensor, expected_tensor in zip(
        actual_batch, expected_batch, strict=True
    ):
        torch.testing.assert_close(actual_tensor, expected_tensor, rtol=0, atol=0)


def test_fast_tokenizer_negative_id_retains_error(
    byte_bpe_runner: NemotronVoiceChatTalkerModelRunner,
) -> None:
    with pytest.raises(OverflowError):
        original_char_batch(byte_bpe_runner, [-1])
    with pytest.raises(OverflowError):
        byte_bpe_runner.char_batch([-1])


def test_empty_batch_retains_error(
    sparse_runner: NemotronVoiceChatTalkerModelRunner,
) -> None:
    with pytest.raises(ValueError):
        original_char_batch(sparse_runner, [])
    with pytest.raises(ValueError):
        sparse_runner.char_batch([])
