# SPDX-License-Identifier: Apache-2.0
"""Tests for MLX-native Qwen3-TTS prompt construction and preprocessing."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

mx = pytest.importorskip("mlx.core")

from sglang_omni.models.qwen3_tts.mlx.config import (  # noqa: E402
    CodePredictorConfig,
    ModelConfig,
    TalkerConfig,
)
from sglang_omni.models.qwen3_tts.mlx.prompt import (  # noqa: E402
    ROLE_TOKENS,
    Qwen3TTSMlxPromptBuilder,
)
from sglang_omni.models.qwen3_tts.mlx.talker import (  # noqa: E402
    Qwen3TTSTalkerForConditionalGeneration,
)

HIDDEN = 8
GROUPS = 4


class _StubTokenizer:
    """Emits one id per character so slice offsets stay easy to reason about."""

    def encode(self, text: str) -> list[int]:
        return [(ord(ch) % 30) + 1 for ch in text]


def _model_config() -> ModelConfig:
    talker = TalkerConfig(
        code_predictor_config=CodePredictorConfig(
            vocab_size=32,
            hidden_size=HIDDEN,
            intermediate_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=4,
            num_code_groups=GROUPS,
        ),
        vocab_size=64,
        hidden_size=HIDDEN,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        text_hidden_size=6,
        text_vocab_size=64,
        num_code_groups=GROUPS,
        codec_language_id={"english": 40, "chinese": 41},
        spk_id={"serena": 50, "vivian": 51},
        spk_is_dialect={"vivian": "chinese"},
        # These default to ~2150, which would index past a 64-entry codec
        # embedding. MLX segfaults on an out-of-range lookup rather than
        # raising, so keep every id inside vocab_size.
        codec_eos_token_id=56,
        codec_think_id=57,
        codec_nothink_id=60,
        codec_think_bos_id=61,
        codec_think_eos_id=62,
        codec_pad_id=58,
        codec_bos_id=59,
    )
    # The TTS special ids default to ~151672, past this tiny text vocab.
    return ModelConfig(
        talker_config=talker,
        tts_bos_token_id=10,
        tts_eos_token_id=11,
        tts_pad_token_id=12,
    )


def _builder() -> Qwen3TTSMlxPromptBuilder:
    mx.random.seed(0)
    config = _model_config()
    talker = Qwen3TTSTalkerForConditionalGeneration(config.talker_config)
    mx.eval(talker.parameters())
    return Qwen3TTSMlxPromptBuilder(talker, config, _StubTokenizer())


# --------------------------------------------------------------------------
# Control prefix
# --------------------------------------------------------------------------


def test_explicit_language_adds_one_control_position() -> None:
    """A language id makes the think prefix 4 tokens instead of 3."""
    builder = _builder()
    auto = builder.build_custom_voice(text="hi", voice="serena", language="auto")
    english = builder.build_custom_voice(text="hi", voice="serena", language="english")
    mx.eval(auto.input_embeds, english.input_embeds)
    assert english.input_embeds.shape[1] == auto.input_embeds.shape[1] + 1


def test_dialect_voice_implies_its_language_on_auto() -> None:
    """spk_is_dialect resolves a language even when the caller said auto."""
    builder = _builder()
    plain = builder.build_custom_voice(text="hi", voice="serena", language="auto")
    dialect = builder.build_custom_voice(text="hi", voice="vivian", language="auto")
    mx.eval(plain.input_embeds, dialect.input_embeds)
    assert dialect.input_embeds.shape[1] == plain.input_embeds.shape[1] + 1


def test_unknown_language_is_rejected_with_the_supported_list() -> None:
    builder = _builder()
    with pytest.raises(ValueError, match="Supported: chinese, english"):
        builder.build_custom_voice(text="hi", voice="serena", language="klingon")


# --------------------------------------------------------------------------
# CustomVoice
# --------------------------------------------------------------------------


def test_unknown_voice_is_rejected() -> None:
    builder = _builder()
    with pytest.raises(ValueError, match="Supported speakers: serena, vivian"):
        builder.build_custom_voice(text="hi", voice="nobody")


def test_voice_lookup_is_case_insensitive() -> None:
    builder = _builder()
    lower = builder.build_custom_voice(text="hi", voice="serena")
    upper = builder.build_custom_voice(text="hi", voice="SERENA")
    mx.eval(lower.input_embeds, upper.input_embeds)
    assert float(mx.abs(lower.input_embeds - upper.input_embeds).max()) == 0.0


def test_different_voices_produce_different_prompts() -> None:
    builder = _builder()
    a = builder.build_custom_voice(text="hi", voice="serena", language="english")
    b = builder.build_custom_voice(text="hi", voice="vivian", language="english")
    mx.eval(a.input_embeds, b.input_embeds)
    assert a.input_embeds.shape == b.input_embeds.shape
    assert float(mx.abs(a.input_embeds - b.input_embeds).max()) > 0.0


def test_missing_speaker_table_is_rejected() -> None:
    config = _model_config()
    config.talker_config.spk_id = None
    talker = Qwen3TTSTalkerForConditionalGeneration(config.talker_config)
    mx.eval(talker.parameters())
    builder = Qwen3TTSMlxPromptBuilder(talker, config, _StubTokenizer())
    with pytest.raises(ValueError, match="configured spk_id"):
        builder.build_custom_voice(text="hi", voice="serena")


# --------------------------------------------------------------------------
# Streaming vs non-streaming text placement
# --------------------------------------------------------------------------


def test_streaming_leaves_the_text_for_the_decode_loop() -> None:
    builder = _builder()
    text = "hello world"
    streaming = builder.build_custom_voice(
        text=text, voice="serena", non_streaming_mode=False
    )
    non_streaming = builder.build_custom_voice(
        text=text, voice="serena", non_streaming_mode=True
    )
    mx.eval(
        streaming.input_embeds,
        streaming.trailing_text_embeds,
        non_streaming.input_embeds,
        non_streaming.trailing_text_embeds,
    )
    # Non-streaming puts the whole text in the prefill; streaming keeps one row
    # there and streams the rest.
    assert non_streaming.input_embeds.shape[1] > streaming.input_embeds.shape[1]
    assert non_streaming.trailing_text_embeds.shape[1] == 1
    assert streaming.trailing_text_embeds.shape[1] > 1


def test_streaming_trailing_length_tracks_the_text() -> None:
    builder = _builder()
    short = builder.build_custom_voice(text="ab", voice="serena")
    longer = builder.build_custom_voice(text="abcd", voice="serena")
    mx.eval(short.trailing_text_embeds, longer.trailing_text_embeds)
    assert (
        longer.trailing_text_embeds.shape[1] - short.trailing_text_embeds.shape[1] == 2
    )


# --------------------------------------------------------------------------
# Instructions and VoiceDesign
# --------------------------------------------------------------------------


def test_instructions_are_prepended() -> None:
    builder = _builder()
    plain = builder.build_custom_voice(text="hi", voice="serena")
    guided = builder.build_custom_voice(text="hi", voice="serena", instructions="calm")
    mx.eval(plain.input_embeds, guided.input_embeds)
    assert guided.input_embeds.shape[1] > plain.input_embeds.shape[1]
    # The original prompt is the suffix; instructions sit in front of it.
    tail = guided.input_embeds[:, -plain.input_embeds.shape[1] :, :]
    assert float(mx.abs(tail - plain.input_embeds).max()) == 0.0


def test_voice_design_requires_instructions() -> None:
    builder = _builder()
    with pytest.raises(ValueError, match="requires instructions"):
        builder.build_voice_design(text="hi", instructions="")


def test_voice_design_has_no_speaker_token() -> None:
    """VoiceDesign conditions on words, so its control prefix is shorter."""
    builder = _builder()
    design = builder.build_voice_design(text="hi", instructions="x", language="english")
    custom = builder.build_custom_voice(
        text="hi", voice="serena", language="english", instructions="x"
    )
    mx.eval(design.input_embeds, custom.input_embeds)
    assert design.input_embeds.shape[1] == custom.input_embeds.shape[1] - 1


# --------------------------------------------------------------------------
# Voice cloning
# --------------------------------------------------------------------------


def test_voice_clone_prompt_grows_with_the_reference() -> None:
    builder = _builder()
    codes_short = mx.zeros((1, GROUPS, 3), dtype=mx.int32)
    codes_long = mx.zeros((1, GROUPS, 7), dtype=mx.int32)
    short = builder.build_voice_clone(
        text="hi", ref_codes=codes_short, ref_text="ref", speaker_embed=None
    )
    long = builder.build_voice_clone(
        text="hi", ref_codes=codes_long, ref_text="ref", speaker_embed=None
    )
    mx.eval(short.input_embeds, long.input_embeds)
    assert long.input_embeds.shape[1] - short.input_embeds.shape[1] == 4


def test_voice_clone_streams_only_padding() -> None:
    builder = _builder()
    prompt = builder.build_voice_clone(
        text="hello",
        ref_codes=mx.zeros((1, GROUPS, 2), dtype=mx.int32),
        ref_text="ref",
        speaker_embed=None,
    )
    mx.eval(prompt.trailing_text_embeds, prompt.pad_embed)
    # ICL is non-streaming: the text is all in the prefill.
    assert prompt.trailing_text_embeds.shape[1] == 1
    assert float(mx.abs(prompt.trailing_text_embeds - prompt.pad_embed).max()) == 0.0


def test_float32_speaker_embedding_does_not_promote_the_prompt() -> None:
    """The speaker encoder runs in float32; the prompt must stay bf16."""
    builder = _builder()
    builder.talker.set_dtype(mx.bfloat16)
    mx.eval(builder.talker.parameters())
    speaker = mx.ones((1, HIDDEN), dtype=mx.float32)
    prompt = builder.build_voice_clone(
        text="hi",
        ref_codes=mx.zeros((1, GROUPS, 2), dtype=mx.int32),
        ref_text="ref",
        speaker_embed=speaker,
    )
    mx.eval(prompt.input_embeds)
    assert prompt.input_embeds.dtype == mx.bfloat16


def test_speaker_embedding_adds_one_position() -> None:
    builder = _builder()
    codes = mx.zeros((1, GROUPS, 2), dtype=mx.int32)
    without = builder.build_voice_clone(
        text="hi", ref_codes=codes, ref_text="ref", speaker_embed=None
    )
    with_speaker = builder.build_voice_clone(
        text="hi",
        ref_codes=codes,
        ref_text="ref",
        speaker_embed=mx.zeros((1, HIDDEN)),
    )
    mx.eval(without.input_embeds, with_speaker.input_embeds)
    assert with_speaker.input_embeds.shape[1] == without.input_embeds.shape[1] + 1


def test_role_tokens_open_every_prompt() -> None:
    """All three tasks start with the same role prefix."""
    builder = _builder()
    prompts = [
        builder.build_custom_voice(text="hi", voice="serena"),
        builder.build_voice_design(text="hi", instructions="x"),
        builder.build_voice_clone(
            text="hi",
            ref_codes=mx.zeros((1, GROUPS, 2), dtype=mx.int32),
            ref_text="r",
            speaker_embed=None,
        ),
    ]
    for prompt in prompts:
        mx.eval(prompt.input_embeds)
        assert prompt.input_embeds.shape[1] > ROLE_TOKENS
        assert prompt.input_embeds.shape[-1] == HIDDEN


# --------------------------------------------------------------------------
# Preprocessor dispatch
# --------------------------------------------------------------------------


def test_preprocessor_dispatches_on_task_type() -> None:
    from sglang_omni.models.qwen3_tts.mlx.preprocessing import Qwen3TTSMlxPreprocessor

    builder = _builder()
    pre = Qwen3TTSMlxPreprocessor(
        builder.talker, builder.model_config, _StubTokenizer(), checkpoint_dir="/nope"
    )

    custom = pre.prepare(
        SimpleNamespace(
            task_type="CustomVoice",
            text="hi",
            voice="serena",
            language="auto",
            non_streaming_mode=False,
            instructions=None,
        )
    )
    mx.eval(custom.input_embeds)
    assert custom.input_embeds.shape[1] > 0

    design = pre.prepare(
        SimpleNamespace(
            task_type="VoiceDesign",
            text="hi",
            voice=None,
            language="auto",
            non_streaming_mode=False,
            instructions="warm",
        )
    )
    mx.eval(design.input_embeds)
    assert design.input_embeds.shape[1] > 0

    with pytest.raises(ValueError, match="Unhandled Qwen3-TTS task type"):
        pre.prepare(
            SimpleNamespace(
                task_type="Nope",
                text="hi",
                voice=None,
                language="auto",
                non_streaming_mode=False,
                instructions=None,
            )
        )


def test_base_task_requires_reference_audio_and_text() -> None:
    from sglang_omni.models.qwen3_tts.mlx.preprocessing import Qwen3TTSMlxPreprocessor

    builder = _builder()
    pre = Qwen3TTSMlxPreprocessor(
        builder.talker, builder.model_config, _StubTokenizer(), checkpoint_dir="/nope"
    )
    with pytest.raises(ValueError, match="ref_audio and ref_text"):
        pre.prepare(
            SimpleNamespace(
                task_type="Base",
                text="hi",
                voice=None,
                language="auto",
                non_streaming_mode=False,
                instructions=None,
                ref_audio=None,
                ref_text=None,
            )
        )
