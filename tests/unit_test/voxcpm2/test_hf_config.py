# SPDX-License-Identifier: Apache-2.0
"""VoxCPM2 checkpoint configuration and staging for SGLang."""

from __future__ import annotations

import json
from pathlib import Path

from sglang_omni.models.voxcpm2.hf_config import (
    VOXCPM2_MODEL_TYPE,
    VoxCPM2Config,
    load_voxcpm2_config,
    stage_checkpoint_for_autoconfig,
)

_LM_CONFIG = {
    "hidden_size": 2048,
    "intermediate_size": 6144,
    "max_position_embeddings": 32768,
    "num_attention_heads": 16,
    "num_hidden_layers": 28,
    "num_key_value_heads": 2,
    "rms_norm_eps": 1e-05,
    "rope_theta": 10000,
    "vocab_size": 73448,
    "use_mup": False,
    "scale_emb": 12,
    "dim_model_base": 256,
    "scale_depth": 1.4,
    "kv_channels": 128,
}


_CONFIG = {
    "architecture": "voxcpm2",
    "lm_config": _LM_CONFIG,
    "patch_size": 4,
    "feat_dim": 64,
    "scalar_quantization_latent_dim": 512,
    "scalar_quantization_scale": 9,
    "residual_lm_num_layers": 8,
    "residual_lm_no_rope": True,
    "encoder_config": {"hidden_dim": 1024, "ffn_dim": 4096, "num_heads": 16},
    "dit_config": {
        "hidden_dim": 1024,
        "ffn_dim": 4096,
        "num_heads": 16,
        "mean_mode": False,
        "cfm_config": {"solver": "euler", "inference_cfg_rate": 2.0},
    },
    "audio_vae_config": {
        "latent_dim": 64,
        "sample_rate": 16000,
        "out_sample_rate": 48000,
    },
    "max_length": 8192,
    "dtype": "bfloat16",
}


def _write_checkpoint(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps(_CONFIG))
    return str(tmp_path)


def test_runtime_config_reads_the_nested_sections(tmp_path):
    config = load_voxcpm2_config(_write_checkpoint(tmp_path))
    assert config.patch_size == 4
    assert config.feat_dim == 64
    assert config.residual_lm_num_layers == 8
    assert config.residual_lm_no_rope is True
    assert config.sample_rate == 16000
    assert config.out_sample_rate == 48000
    assert config.latent_dim == 64
    assert config.cfm["solver"] == "euler"
    assert config.dit_mean_mode is False


def test_sglang_view_reports_both_stacks_as_one_depth():
    """The KV pool is sized from this number, so it covers both stacks."""
    config = VoxCPM2Config(lm_config=_LM_CONFIG, voxcpm2_config=_CONFIG)
    assert config.num_hidden_layers == 28 + 8
    assert config.lm_config.num_hidden_layers == 28


def test_sglang_view_lifts_the_fields_sglang_reads():
    config = VoxCPM2Config(lm_config=_LM_CONFIG, voxcpm2_config=_CONFIG)
    assert config.hidden_size == 2048
    assert config.num_attention_heads == 16
    assert config.num_key_value_heads == 2
    assert config.vocab_size == 73448


def test_sub_config_name_stays_off_sglangs_text_config_lookup():
    """Renaming lm_config to one of these hands SGLang the 28-layer depth."""
    config = VoxCPM2Config(lm_config=_LM_CONFIG, voxcpm2_config=_CONFIG)
    for claimed in ("text_config", "llm_config", "language_config", "thinker_config"):
        assert not hasattr(config, claimed)


def _checkpoint(root: Path, *, model_type: str | None = None) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    config = {"architecture": "voxcpm2", "lm_config": {"num_hidden_layers": 2}}
    if model_type is not None:
        config["model_type"] = model_type
    (root / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (root / "model.safetensors").write_bytes(b"weights")
    (root / "audiovae.pth").write_bytes(b"vae")
    return root


def test_a_checkpoint_that_already_declares_its_type_is_used_as_is(tmp_path):
    root = _checkpoint(tmp_path / "snap", model_type=VOXCPM2_MODEL_TYPE)
    assert stage_checkpoint_for_autoconfig(str(root)) == str(root)


def test_staging_adds_the_model_type_without_touching_the_snapshot(tmp_path):
    root = _checkpoint(tmp_path / "snap")
    staged = Path(stage_checkpoint_for_autoconfig(str(root)))

    assert staged != root
    assert json.loads((staged / "config.json").read_text())["model_type"] == (
        VOXCPM2_MODEL_TYPE
    )
    assert "model_type" not in json.loads((root / "config.json").read_text())


def test_the_weights_are_linked_rather_than_copied(tmp_path):
    root = _checkpoint(tmp_path / "snap")
    staged = Path(stage_checkpoint_for_autoconfig(str(root)))

    assert (staged / "model.safetensors").is_symlink()
    assert (staged / "model.safetensors").read_bytes() == b"weights"
    assert (staged / "audiovae.pth").is_symlink()


def test_staging_twice_is_idempotent(tmp_path):
    root = _checkpoint(tmp_path / "snap")
    first = stage_checkpoint_for_autoconfig(str(root))
    assert stage_checkpoint_for_autoconfig(str(root)) == first


def test_a_dangling_link_left_behind_does_not_break_staging(tmp_path):
    """A returned checkpoint must expose readable weights, not a stale link."""
    root = _checkpoint(tmp_path / "snap")
    staged = tmp_path / "snap-sglang-omni"
    staged.mkdir()
    (staged / "model.safetensors").symlink_to(tmp_path / "gone")

    assert stage_checkpoint_for_autoconfig(str(root)) == str(staged)
    assert (staged / "model.safetensors").read_bytes() == b"weights"


def test_staging_a_relative_checkpoint_keeps_weight_links_readable(
    tmp_path, monkeypatch
):
    _checkpoint(tmp_path / "snap")
    monkeypatch.chdir(tmp_path)
    staged = Path(stage_checkpoint_for_autoconfig("snap"))
    assert (staged / "model.safetensors").read_bytes() == b"weights"


def test_staging_never_exposes_a_partially_written_config(tmp_path, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event

    root = _checkpoint(tmp_path / "snap")
    staged = Path(stage_checkpoint_for_autoconfig(str(root)))
    original_write = Path.write_text
    opened, release = Event(), Event()

    def paused_write(path, data, *args, **kwargs):
        if path.name == "config.json" and path != root / "config.json":
            # note (Xinhao Tan): hold the writer here so the reader reliably
            # checks the interval that used to expose an empty config.
            with path.open("w", encoding="utf-8"):
                opened.set()
                assert release.wait(5)
        return original_write(path, data, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", paused_write)
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(stage_checkpoint_for_autoconfig, str(root))
        try:
            assert opened.wait(5)
            assert json.loads((staged / "config.json").read_text())["model_type"] == (
                VOXCPM2_MODEL_TYPE
            )
        finally:
            release.set()
        assert future.result() == str(staged)
