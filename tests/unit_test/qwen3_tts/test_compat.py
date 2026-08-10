# SPDX-License-Identifier: Apache-2.0
"""Qwen3-TTS upstream compatibility shims."""

from __future__ import annotations

import ast
import importlib.util
import re
from pathlib import Path

import pytest
import torch
from transformers import masking_utils
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.utils import generic

_REPO_ROOT = Path(__file__).resolve().parents[3]
_STAGES_PATH = _REPO_ROOT / "sglang_omni/models/qwen3_tts/stages.py"
_INSTALL_DOCS = (
    _REPO_ROOT / "docs/cookbook/qwen3_tts.md",
    _REPO_ROOT / "docs/basic_usage/tts.md",
)

# Note (Akazaakane): loaded by path so the module under test never imports sglang.
_COMPAT_PATH = _REPO_ROOT / "sglang_omni/models/qwen3_tts/compat.py"
_SPEC = importlib.util.spec_from_file_location("qwen3_tts_compat", _COMPAT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
compat = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(compat)

# Note (Akazaakane): must be captured before any test applies the patch.
_PRISTINE = {name: getattr(masking_utils, name) for name in compat._MASK_FACTORY_NAMES}


@pytest.fixture(autouse=True)
def restore_transformers_globals():
    """Undo every global ``apply_...()`` touches, so tests cannot leak into
    the rest of the pytest process."""
    saved_masks = {
        name: getattr(masking_utils, name) for name in compat._MASK_FACTORY_NAMES
    }
    saved_check_model_inputs = generic.check_model_inputs
    saved_rope = ROPE_INIT_FUNCTIONS.get("default")
    rope_was_set = "default" in ROPE_INIT_FUNCTIONS
    try:
        yield
    finally:
        for name, original in saved_masks.items():
            setattr(masking_utils, name, original)
        generic.check_model_inputs = saved_check_model_inputs
        if rope_was_set:
            ROPE_INIT_FUNCTIONS["default"] = saved_rope
        else:
            ROPE_INIT_FUNCTIONS.pop("default", None)


def _mask_config() -> LlamaConfig:
    config = LlamaConfig(
        hidden_size=8, num_attention_heads=2, num_hidden_layers=1, vocab_size=16
    )
    config._attn_implementation = "eager"
    config.sliding_window = 2
    return config


def _qwen_tts_kwargs(config: LlamaConfig) -> dict:
    return {
        "config": config,
        "input_embeds": torch.zeros(1, 4, 8),
        "attention_mask": torch.ones(1, 4, dtype=torch.long),
        "cache_position": torch.arange(4),
        "past_key_values": None,
    }


def _supported_kwargs(config: LlamaConfig) -> dict:
    return {
        "config": config,
        "inputs_embeds": torch.zeros(1, 4, 8),
        "attention_mask": torch.ones(1, 4, dtype=torch.long),
        "past_key_values": None,
    }


def _documented_install_commands(markdown: Path) -> list[str]:
    """The `uv pip install` lines from a page's qwen-tts prerequisites block."""
    for block in re.findall(r"```bash\n(.*?)```", markdown.read_text(), re.S):
        if "qwen-tts==" in block:
            return [
                line.strip()
                for line in block.splitlines()
                if line.strip().startswith("uv pip install")
            ]
    raise AssertionError(f"no qwen-tts install block found in {markdown}")


def _runtime_install_hint() -> str:
    """_QWEN_TTS_INSTALL_HINT, read without importing stages.py (needs sglang)."""
    for node in ast.parse(_STAGES_PATH.read_text()).body:
        if isinstance(node, ast.Assign) and any(
            getattr(target, "id", None) == "_QWEN_TTS_INSTALL_HINT"
            for target in node.targets
        ):
            return ast.literal_eval(node.value)
    raise AssertionError(f"_QWEN_TTS_INSTALL_HINT not found in {_STAGES_PATH}")


def test_transformers_globals_are_pristine_at_test_start() -> None:
    """Canary for patch leakage between tests in one pytest process."""
    for name, original in _PRISTINE.items():
        assert getattr(masking_utils, name) is original
    assert not getattr(generic.check_model_inputs, compat._PATCHED_FLAG, False)


@pytest.mark.parametrize("name", compat._MASK_FACTORY_NAMES)
def test_unpatched_transformers_rejects_the_qwen_tts_call_shape(name: str) -> None:
    with pytest.raises(TypeError, match="input_embeds"):
        getattr(masking_utils, name)(**_qwen_tts_kwargs(_mask_config()))


@pytest.mark.parametrize("name", compat._MASK_FACTORY_NAMES)
def test_shim_accepts_input_embeds_and_absorbs_cache_position(name: str) -> None:
    config = _mask_config()
    compat.apply_qwen_tts_transformers_compatibility_patches()

    shimmed = getattr(masking_utils, name)(**_qwen_tts_kwargs(config))
    expected = _PRISTINE[name](**_supported_kwargs(config))

    assert shimmed is not None
    torch.testing.assert_close(shimmed, expected)


@pytest.mark.parametrize("name", compat._MASK_FACTORY_NAMES)
def test_shim_passes_the_supported_call_shape_through(name: str) -> None:
    config = _mask_config()
    compat.apply_qwen_tts_transformers_compatibility_patches()

    torch.testing.assert_close(
        getattr(masking_utils, name)(**_supported_kwargs(config)),
        _PRISTINE[name](**_supported_kwargs(config)),
    )


@pytest.mark.parametrize("name", compat._MASK_FACTORY_NAMES)
def test_shim_passes_positional_arguments_through(name: str) -> None:
    """The patch is global, so Transformers' own positional callers must work."""
    config = _mask_config()
    compat.apply_qwen_tts_transformers_compatibility_patches()

    torch.testing.assert_close(
        getattr(masking_utils, name)(
            config, torch.zeros(1, 4, 8), torch.ones(1, 4, dtype=torch.long), None
        ),
        _PRISTINE[name](**_supported_kwargs(config)),
    )


def test_shim_prefers_inputs_embeds_when_both_spellings_are_given() -> None:
    config = _mask_config()
    compat.apply_qwen_tts_transformers_compatibility_patches()

    mask = masking_utils.create_causal_mask(
        config=config,
        input_embeds=torch.zeros(1, 4, 8),
        inputs_embeds=torch.zeros(1, 2, 8),
        attention_mask=torch.ones(1, 2, dtype=torch.long),
        past_key_values=None,
    )

    assert mask.shape == (1, 1, 2, 2)


def test_shim_is_idempotent() -> None:
    compat.apply_qwen_tts_transformers_compatibility_patches()
    patched = {
        name: getattr(masking_utils, name) for name in compat._MASK_FACTORY_NAMES
    }
    for name, fn in patched.items():
        assert fn is not _PRISTINE[name]

    compat.apply_qwen_tts_transformers_compatibility_patches()

    for name, fn in patched.items():
        assert getattr(masking_utils, name) is fn


def test_shim_is_a_no_op_when_transformers_already_accepts_input_embeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def already_compatible(config, input_embeds, cache_position=None):  # noqa: ARG001
        return None

    monkeypatch.setattr(masking_utils, "create_causal_mask", already_compatible)

    compat.apply_qwen_tts_transformers_compatibility_patches()

    assert masking_utils.create_causal_mask is already_compatible


def test_documented_install_commands_agree_across_pages() -> None:
    cookbook, basic_usage = (_documented_install_commands(p) for p in _INSTALL_DOCS)
    assert cookbook == basic_usage


def test_documented_install_commands_keep_no_deps() -> None:
    """Resolving sox or onnxruntime lifts numpy past the numba==0.65.1 ceiling,
    which breaks librosa and so `import qwen_tts`. Dropping --no-deps here has
    regressed twice."""
    for command in _documented_install_commands(_INSTALL_DOCS[0]):
        assert "--no-deps" in command, command
        assert "onnxruntime" not in command, command


def test_runtime_install_hint_matches_the_documented_commands() -> None:
    hint = _runtime_install_hint()
    for command in _documented_install_commands(_INSTALL_DOCS[0]):
        assert command in hint, f"{command!r} missing from _QWEN_TTS_INSTALL_HINT"
