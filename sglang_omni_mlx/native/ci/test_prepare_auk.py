# SPDX-License-Identifier: Apache-2.0
"""CPU checks for AuK artifact source selection and cache integrity."""

from __future__ import annotations

import json
import tempfile
import unittest
from importlib.metadata import version
from pathlib import Path

from sglang_omni_mlx.native.tools.prepare_auk import (
    ArtifactManifest,
    conditioner_sources,
    file_digest,
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
                "format_version": 2,
                "mlx_version": version("mlx"),
                "sources": {},
                "outputs": [file_digest(configuration, directory)],
            }
            (directory / "manifest.json").write_text(json.dumps(manifest))
            verify_reuse(directory, {})
            configuration.write_text('{"model_type": "changed"}')
            with self.assertRaisesRegex(ValueError, "output files changed"):
                verify_reuse(directory, {})
