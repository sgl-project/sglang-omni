# SPDX-License-Identifier: Apache-2.0
"""CPU checks for AuK artifact source selection and cache integrity."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import unittest
from importlib.metadata import version
from pathlib import Path
from unittest.mock import patch

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

    def test_failed_rebuild_keeps_existing_artifact(self) -> None:
        """A stale cache must survive a failed replacement without partial outputs."""
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            auk, conditioner, output = (
                directory / name for name in ("auk", "conditioner", "prepared")
            )
            for component in (auk, conditioner, output):
                component.mkdir()
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
            (output / "manifest.json").write_text('{"format_version": 0}')
            (output / "model.safetensors").write_bytes(b"previous checkpoint")
            original = {path.name: path.read_bytes() for path in output.iterdir()}
            with patch(
                "sglang_omni_mlx.native.tools.prepare_auk.prepare_diffusion",
                side_effect=RuntimeError("conversion failed"),
            ):
                with self.assertRaisesRegex(RuntimeError, "conversion failed"):
                    prepare(auk, conditioner, output, None, None, reuse=True)
            self.assertEqual(
                {path.name: path.read_bytes() for path in output.iterdir()}, original
            )
            self.assertEqual(
                sorted(path.name for path in directory.iterdir()),
                ["auk", "conditioner", "prepared"],
            )
