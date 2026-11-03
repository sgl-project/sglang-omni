# SPDX-License-Identifier: Apache-2.0
"""CPU checks for how AuK artifact preparation publishes its output."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from omegaconf import DictConfig

from sglang_omni_mlx.native.tools.prepare_auk import ARTIFACT_LICENSES, prepare

MODULE = "sglang_omni_mlx.native.tools.prepare_auk"


def convert_diffusion(
    source: Path, directory: Path, architecture: DictConfig, latent_dimension: int
) -> None:
    (directory / "model.safetensors").write_bytes(source.read_bytes())


def convert_vae(source: Path, directory: Path, config: DictConfig) -> None:
    (directory / "model.safetensors").write_bytes(source.read_bytes())


def convert_conditioner(
    weight_files: list[Path], metadata: list[Path], directory: Path
) -> None:
    (directory / "model.safetensors").write_bytes(weight_files[0].read_bytes())


@pytest.fixture
def sources(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    """Official-looking AuK and Qwen directories, with the slow converters stubbed."""
    auk, conditioner = tmp_path / "auk", tmp_path / "conditioner"
    auk.mkdir()
    conditioner.mkdir()
    (auk / "config.json").write_text(
        json.dumps(
            {
                "name": "AuK",
                "arch": {},
                "vae": {
                    "model_init_kwargs": {},
                    "target_sample_rate": 24000,
                    "downsample_rate": 480,
                    "latent_dim": 2,
                },
            }
        )
    )
    for name in ("model.safetensors", "vae.safetensors"):
        (auk / name).write_text("{}")
    for name in (
        "config.json",
        "preprocessor_config.json",
        "tokenizer_config.json",
        "vocab.json",
        "merges.txt",
        "model.safetensors",
    ):
        (conditioner / name).write_text("{}")
    monkeypatch.setattr(f"{MODULE}.prepare_diffusion", convert_diffusion)
    monkeypatch.setattr(f"{MODULE}.prepare_vae", convert_vae)
    monkeypatch.setattr(f"{MODULE}.prepare_conditioner", convert_conditioner)
    return auk, conditioner


def test_preparation_publishes_components_with_their_weight_licenses(
    sources: tuple[Path, Path], tmp_path: Path
) -> None:
    output = tmp_path / "prepared"
    prepare(*sources, output)
    assert sorted(
        path.relative_to(output).as_posix()
        for path in output.rglob("*")
        if path.is_file()
    ) == sorted(
        [
            "config.json",
            "conditioner/model.safetensors",
            "dit/model.safetensors",
            "vae/model.safetensors",
            *(f"licenses/{name}" for name in ARTIFACT_LICENSES),
        ]
    )
    for name, source in ARTIFACT_LICENSES.items():
        assert (output / "licenses" / name).read_bytes() == source.read_bytes()
    assert sorted(path.name for path in tmp_path.iterdir()) == [
        "auk",
        "conditioner",
        "prepared",
    ]


def test_failed_conversion_leaves_no_output_or_staging_directory(
    sources: tuple[Path, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The server loads whatever sits at the output, so a partial conversion must never land there."""

    def fail(source: Path, directory: Path, config: DictConfig) -> None:
        raise RuntimeError("conversion failed")

    monkeypatch.setattr(f"{MODULE}.prepare_vae", fail)
    with pytest.raises(RuntimeError, match="conversion failed"):
        prepare(*sources, tmp_path / "prepared")
    assert sorted(path.name for path in tmp_path.iterdir()) == ["auk", "conditioner"]


def test_an_existing_output_is_left_untouched(
    sources: tuple[Path, Path], tmp_path: Path
) -> None:
    output = tmp_path / "prepared"
    output.mkdir()
    (output / "model.safetensors").write_bytes(b"previous checkpoint")
    with pytest.raises(FileExistsError):
        prepare(*sources, output)
    assert [path.name for path in output.iterdir()] == ["model.safetensors"]
    assert (output / "model.safetensors").read_bytes() == b"previous checkpoint"
