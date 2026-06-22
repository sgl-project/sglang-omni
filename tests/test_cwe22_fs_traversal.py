"""
PoC / regression test for CWE-22 path traversal in playground/qwen-omni/app.py.

The ``_fs_root()`` function returned ``Path("/")`` and the ``_fs_resolve()``
guard ``candidate.relative_to(root)`` always passes when root is ``/``,
allowing an unauthenticated caller to read arbitrary files and list arbitrary
directories via the ``/v1/fs/file`` and ``/v1/fs/list`` endpoints.

After the fix, _fs_root() must return a restricted directory, and requests
outside that directory must be rejected with HTTP 403.
"""
import argparse
import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Helpers to set up a confined sandbox and import the patched app
# ---------------------------------------------------------------------------

@pytest.fixture()
def sandbox(tmp_path: Path):
    """Create a small sandbox directory with some files and subdirs."""
    (tmp_path / "subdir").mkdir()
    (tmp_path / "allowed.txt").write_text("hello")
    (tmp_path / "subdir" / "nested.txt").write_text("nested content")
    return tmp_path


@pytest.fixture()
def client(sandbox: Path, monkeypatch):
    """
    Patch the environment so _fs_root() resolves to the sandbox, then
    import and build a TestClient around the app.
    """
    monkeypatch.setenv("SGLANG_OMNI_FS_ROOT", str(sandbox))

    # Re-import the app module so _fs_root() picks up the env var.
    import importlib
    import sys

    mod_name = "playground.qwen_omni_app_under_test"
    if mod_name in sys.modules:
        del sys.modules[mod_name]

    # Load the actual module from its file path
    import importlib.util

    worktree = os.environ.get(
        "WORKTREE",
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    )
    spec = importlib.util.spec_from_file_location(
        mod_name,
        os.path.join(worktree, "playground", "qwen-omni", "app.py"),
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod

    # Prevent uvicorn.run from actually starting a server at import time,
    # and prevent argparse from parsing pytest's CLI args.
    import unittest.mock
    with unittest.mock.patch("uvicorn.run"), \
         unittest.mock.patch(
             "argparse.ArgumentParser.parse_args",
             return_value=argparse.Namespace(port=7860),
         ):
        spec.loader.exec_module(mod)

    return TestClient(mod.app, raise_server_exceptions=False)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestPathTraversal:
    """Requests outside the configured root MUST be rejected."""

    def test_list_etc_rejected(self, client):
        """GET /v1/fs/list?path=/etc must return 403."""
        resp = client.get("/v1/fs/list", params={"path": "/etc"})
        assert resp.status_code == 403, (
            f"Expected 403 for /etc, got {resp.status_code}: {resp.text}"
        )

    def test_read_etc_passwd_rejected(self, client):
        """GET /v1/fs/file?path=/etc/passwd must return 403."""
        resp = client.get("/v1/fs/file", params={"path": "/etc/passwd"})
        assert resp.status_code == 403, (
            f"Expected 403 for /etc/passwd, got {resp.status_code}: {resp.text}"
        )

    def test_dotdot_traversal_rejected(self, client):
        """GET /v1/fs/list?path=../../etc must return 403."""
        resp = client.get("/v1/fs/list", params={"path": "../../etc"})
        assert resp.status_code == 403, (
            f"Expected 403 for ../../etc, got {resp.status_code}: {resp.text}"
        )

    def test_allowed_file_still_works(self, client, sandbox):
        """A file inside the root should still be readable."""
        resp = client.get("/v1/fs/file", params={"path": str(sandbox / "allowed.txt")})
        assert resp.status_code == 200, (
            f"Expected 200 for allowed file, got {resp.status_code}: {resp.text}"
        )

    def test_list_root_works(self, client, sandbox):
        """Listing the root itself should succeed."""
        resp = client.get("/v1/fs/list")
        assert resp.status_code == 200, (
            f"Expected 200 for root listing, got {resp.status_code}: {resp.text}"
        )

    def test_symlink_escape_rejected(self, client, sandbox):
        """A symlink pointing outside the root must be filtered out."""
        link = sandbox / "escape_link"
        try:
            link.symlink_to("/etc")
        except OSError:
            pytest.skip("Cannot create symlinks in this environment")

        resp = client.get("/v1/fs/list", params={"path": str(link)})
        assert resp.status_code == 403, (
            f"Expected 403 for symlink escape, got {resp.status_code}: {resp.text}"
        )
