# SPDX-License-Identifier: Apache-2.0
"""Contract test for the shared ``resolve_checkpoint`` helper (RFC #661 quick win).

CPU-only, no network: the local-directory branch is exercised directly and the
download branch is checked with ``snapshot_download`` patched out.
"""

from __future__ import annotations

from unittest import mock

from sglang_omni.utils.hf import resolve_checkpoint


def test_existing_directory_is_returned_unchanged(tmp_path) -> None:
    assert resolve_checkpoint(str(tmp_path)) == str(tmp_path)


def test_repo_id_is_downloaded_via_snapshot_download() -> None:
    with mock.patch(
        "huggingface_hub.snapshot_download", return_value="/cache/snapshot"
    ) as download:
        resolved = resolve_checkpoint("org/some-model")

    assert resolved == "/cache/snapshot"
    download.assert_called_once_with("org/some-model")
