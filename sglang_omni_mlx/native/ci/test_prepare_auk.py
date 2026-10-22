# SPDX-License-Identifier: Apache-2.0
"""CPU checks for AuK artifact source selection and cache integrity."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import unittest
from importlib.metadata import version
from pathlib import Path
from unittest.mock import patch

from omegaconf import DictConfig

from sglang_omni_mlx.native.tools.prepare_auk import (
    ARTIFACT_FORMAT_VERSION,
    ArtifactManifest,
    conditioner_sources,
    file_digest,
    prepare,
    read_manifest,
    verify_reuse,
)


class TestPreparation(unittest.TestCase):
    def test_download_progress_does_not_enter_the_artifact(self) -> None:
        """Changing download progress previously invalidated prepared checkpoints."""
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            for name in (
                "config.json",
                "preprocessor_config.json",
                "tokenizer_config.json",
                "vocab.json",
                "merges.txt",
                "model.safetensors",
            ):
                (directory / name).write_text("{}")
            before = conditioner_sources(directory)
            progress = directory / "model.chunks.json"
            progress.write_text('{"downloaded": 1}')
            self.assertEqual(conditioner_sources(directory), before)
            progress.write_text('{"downloaded": 2}')
            self.assertEqual(conditioner_sources(directory), before)

    def test_reuse_rejects_modified_outputs(self) -> None:
        """A stale manifest must not authorize reuse after an output changes."""
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            configuration = directory / "config.json"
            configuration.write_text('{"model_type": "auk"}')
            manifest: ArtifactManifest = {
                "format_version": ARTIFACT_FORMAT_VERSION,
                "mlx_version": version("mlx"),
                "sources": {},
                "outputs": [file_digest(configuration, directory)],
            }
            (directory / "manifest.json").write_text(json.dumps(manifest))
            with patch(
                "sglang_omni_mlx.native.tools.prepare_auk.hashlib.sha256",
                wraps=hashlib.sha256,
            ) as sha256:
                verify_reuse(directory, {}, read_manifest(directory, {}))
                self.assertEqual(sha256.call_count, 0)
                metadata = configuration.stat()
                os.utime(
                    configuration,
                    ns=(metadata.st_atime_ns, metadata.st_mtime_ns + 1_000_000),
                )
                updated = verify_reuse(directory, {}, read_manifest(directory, {}))
                self.assertEqual(sha256.call_count, 1)
                self.assertEqual(
                    updated["outputs"][0]["sha256"], manifest["outputs"][0]["sha256"]
                )
                configuration.write_text('{"model_type": "changed"}')
                with self.assertRaisesRegex(ValueError, "output files changed"):
                    verify_reuse(directory, {}, read_manifest(directory, {}))
                self.assertEqual(sha256.call_count, 2)


class TestArtifactLifecycle(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.auk, self.conditioner, self.output = (
            self.directory / name for name in ("auk", "conditioner", "prepared")
        )
        self.auk.mkdir()
        self.conditioner.mkdir()
        (self.auk / "config.json").write_text(
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
            (self.auk / name).write_text("{}")
        for name in (
            "config.json",
            "preprocessor_config.json",
            "tokenizer_config.json",
            "vocab.json",
            "merges.txt",
            "model.safetensors",
        ):
            (self.conditioner / name).write_text("{}")
        self.converters = [
            self.enterContext(
                patch(
                    f"sglang_omni_mlx.native.tools.prepare_auk.{name}",
                    side_effect=convert,
                )
            )
            for name, convert in (
                ("prepare_diffusion", self.convert_diffusion),
                ("prepare_vae", self.convert_vae),
                ("prepare_conditioner", self.convert_conditioner),
            )
        ]

    @staticmethod
    def convert_diffusion(
        source: Path, directory: Path, architecture: DictConfig, latent_dimension: int
    ) -> None:
        (directory / "model.safetensors").write_bytes(source.read_bytes())

    @staticmethod
    def convert_vae(source: Path, directory: Path, config: DictConfig) -> None:
        (directory / "model.safetensors").write_bytes(source.read_bytes())

    @staticmethod
    def convert_conditioner(
        weight_files: list[Path], metadata: list[Path], directory: Path
    ) -> None:
        (directory / "model.safetensors").write_bytes(weight_files[0].read_bytes())

    def prepare(self) -> None:
        prepare(self.auk, self.conditioner, self.output, None, None, reuse=True)

    def test_failed_rebuild_keeps_existing_artifact(self) -> None:
        """A stale cache must survive a failed replacement without partial outputs."""
        self.output.mkdir()
        (self.output / "manifest.json").write_text('{"format_version": 0}')
        (self.output / "model.safetensors").write_bytes(b"previous checkpoint")
        original = {path.name: path.read_bytes() for path in self.output.iterdir()}
        self.converters[0].side_effect = RuntimeError("conversion failed")
        with self.assertRaisesRegex(RuntimeError, "conversion failed"):
            self.prepare()
        self.assertEqual(
            {path.name: path.read_bytes() for path in self.output.iterdir()}, original
        )
        self.assertEqual(list(self.directory.glob(".prepared.*")), [])

    def test_stale_replacement_removes_only_preparation_artifacts(self) -> None:
        """Successful replacement must publish new contents and reclaim abandoned conversions."""
        self.prepare()
        stale = self.directory / ".prepared.abcdefgh"
        shutil.copytree(self.output, stale)
        manifest = self.directory / ".prepared.manifest-abcdefgh"
        manifest.write_text("interrupted manifest")
        unrelated = self.directory / ".prepared.ijklmnop"
        unrelated.mkdir()
        (unrelated / "keep.txt").write_text("unrelated")
        symlink = self.directory / ".prepared.prepare-abcdefgh"
        symlink.symlink_to(unrelated, target_is_directory=True)
        (self.auk / "model.safetensors").write_bytes(b"replacement weights")
        self.prepare()
        self.assertEqual(
            (self.output / "dit/model.safetensors").read_bytes(), b"replacement weights"
        )
        recorded = json.loads((self.output / "manifest.json").read_text())
        verify_reuse(self.output, recorded["sources"], recorded)
        self.assertEqual(set(self.directory.glob(".prepared.*")), {unrelated, symlink})
        self.assertEqual((unrelated / "keep.txt").read_text(), "unrelated")
        self.assertTrue(symlink.is_symlink())

    def test_valid_reuse_skips_all_converters(self) -> None:
        """An unchanged artifact must bypass every expensive conversion stage."""
        self.prepare()
        for converter in self.converters:
            converter.reset_mock()
        configuration = self.output / "config.json"
        metadata = configuration.stat()
        os.utime(
            configuration, ns=(metadata.st_atime_ns, metadata.st_mtime_ns + 1_000_000)
        )
        self.prepare()
        for converter in self.converters:
            converter.assert_not_called()
        recorded = json.loads((self.output / "manifest.json").read_text())
        verify_reuse(self.output, recorded["sources"], recorded)
        self.assertEqual(list(self.directory.glob(".prepared.*")), [])

    def test_interrupted_replacement_restores_then_validates_backup(self) -> None:
        """A killed directory swap must restore the prior artifact before deciding whether to rebuild."""
        self.prepare()
        previous = self.directory / ".prepared.previous-abcdefgh"
        self.output.rename(previous)
        for converter in self.converters:
            converter.reset_mock()
        self.prepare()
        for converter in self.converters:
            converter.assert_not_called()
        self.output.rename(previous)
        (previous / "manifest.json").write_text('{"format_version": 0}')
        self.prepare()
        for converter in self.converters:
            converter.assert_called_once()
        self.assertEqual(list(self.directory.glob(".prepared.*")), [])

    def test_ambiguous_backups_remain_available_for_recovery(self) -> None:
        """Multiple backups without a published artifact must not be guessed at or deleted."""
        self.prepare()
        previous = self.directory / ".prepared.previous-abcdefgh"
        other = self.directory / ".prepared.previous-ijklmnop"
        self.output.rename(previous)
        shutil.copytree(previous, other)
        with self.assertRaisesRegex(RuntimeError, "Multiple artifact backups"):
            self.prepare()
        self.assertTrue((previous / "manifest.json").is_file())
        self.assertTrue((other / "manifest.json").is_file())
        self.assertFalse(self.output.exists())

    def test_cleanup_failure_does_not_fail_published_or_reused_artifact(self) -> None:
        """Failing to remove an old backup must not invalidate successfully published output."""
        self.prepare()
        (self.auk / "model.safetensors").write_bytes(b"replacement weights")
        with (
            patch(
                "sglang_omni_mlx.native.tools.prepare_auk.shutil.rmtree",
                side_effect=PermissionError("cleanup denied"),
            ),
            self.assertLogs(
                "sglang_omni_mlx.native.tools.prepare_auk", level="WARNING"
            ),
        ):
            self.prepare()
        previous = list(self.directory.glob(".prepared.previous-*"))
        self.assertEqual(len(previous), 1)
        shutil.copytree(previous[0], self.directory / ".prepared.previous-abcdefgh")
        for converter in self.converters:
            converter.reset_mock()
        with (
            patch(
                "sglang_omni_mlx.native.tools.prepare_auk.shutil.rmtree",
                side_effect=PermissionError("cleanup denied"),
            ),
            self.assertLogs(
                "sglang_omni_mlx.native.tools.prepare_auk", level="WARNING"
            ),
        ):
            self.prepare()
        for converter in self.converters:
            converter.assert_not_called()
        self.assertEqual(
            (self.output / "dit/model.safetensors").read_bytes(), b"replacement weights"
        )
        self.prepare()
        self.assertEqual(list(self.directory.glob(".prepared.*")), [])

    def test_concurrent_preparation_cannot_clean_active_conversion(self) -> None:
        """A second preparer must not remove the first preparer's staging directory."""

        def convert(
            source: Path,
            directory: Path,
            architecture: DictConfig,
            latent_dimension: int,
        ) -> None:
            with self.assertRaisesRegex(RuntimeError, "already running"):
                self.prepare()
            self.assertTrue(directory.is_dir())
            self.convert_diffusion(source, directory, architecture, latent_dimension)

        self.converters[0].side_effect = convert
        self.prepare()
        self.assertTrue((self.output / "manifest.json").is_file())

    def test_replaced_output_is_not_overwritten(self) -> None:
        """A noncooperating writer's new output must survive preparation."""
        self.prepare()
        (self.auk / "model.safetensors").write_bytes(b"replacement weights")

        def convert(
            source: Path,
            directory: Path,
            architecture: DictConfig,
            latent_dimension: int,
        ) -> None:
            self.output.rename(self.directory / "external-backup")
            self.output.mkdir()
            (self.output / "keep.txt").write_text("external output")
            self.convert_diffusion(source, directory, architecture, latent_dimension)

        self.converters[0].side_effect = convert
        with self.assertRaisesRegex(FileExistsError, "created during preparation"):
            self.prepare()
        self.assertEqual((self.output / "keep.txt").read_text(), "external output")
