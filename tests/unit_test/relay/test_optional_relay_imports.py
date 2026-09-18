# SPDX-License-Identifier: Apache-2.0
"""Optional dependency errors, isolated from cached or installed backends."""

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest


def _run(source: str) -> None:
    """Test import-time behavior with a fresh module cache for each scenario."""
    prefix = """
import importlib
import importlib.abc
import logging
import sys
from unittest.mock import patch
import pytest
import torch

records = []
class Capture(logging.Handler):
    def emit(self, record):
        records.append(record)
logging.getLogger().setLevel(logging.DEBUG)
logging.getLogger().addHandler(Capture())
"""
    result = subprocess.run(
        [sys.executable, "-c", prefix + textwrap.dedent(source)],
        cwd=Path(__file__).resolve().parents[3],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("backend", ["mooncake", "nixl"])
@pytest.mark.parametrize(
    "failure",
    ["top-level", "api-submodule", "transitive", "import-error"],
)
def test_import_failure_classification_and_selected_backend_errors(backend, failure):
    _run(
        f"""
        backend, failure = {backend!r}, {failure!r}
        api = backend + ('.engine' if backend == 'mooncake' else '._api')
        names = {{
            'top-level': backend,
            'api-submodule': api,
            'transitive': 'vendor_runtime',
            'import-error': backend,
        }}
        error_type = ImportError if failure == 'import-error' else ModuleNotFoundError
        original = error_type('injected import failure: ' + failure, name=names[failure])
        class FailedDependencies(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname.split('.')[0] in {{'mooncake', 'nixl'}}:
                    if fullname.split('.')[0] == backend:
                        raise original
                    raise ModuleNotFoundError(fullname, name=fullname.split('.')[0])
        sys.meta_path.insert(0, FailedDependencies())
        import sglang_omni.relay as package
        from sglang_omni.relay.base import create_relay
        assert package.MOONCAKE_AVAILABLE is False
        assert package.NIXL_AVAILABLE is False
        relevant = [r for r in records if r.name == 'sglang_omni.relay.' + backend]
        assert len(relevant) == 1, [r.getMessage() for r in relevant]
        assert relevant[0].levelno == (logging.DEBUG if failure == 'top-level' else logging.ERROR)
        module = importlib.import_module('sglang_omni.relay.' + backend)
        def forbidden(*args, **kwargs):
            raise AssertionError('unavailable backend initialized native code or a pool')
        if backend == 'mooncake':
            module.TransferEngine = forbidden
            connection = lambda: module.MooncakeConnection('direct', '127.0.0.1')
        else:
            module.nixl_agent_config = forbidden
            connection = lambda: package.Connection('direct')
        callers = [
            connection,
            lambda: create_relay(
                backend, engine_id='factory', device='cpu', slot_size_mb=1, credits=1
            ),
        ]
        with patch.object(torch, 'zeros', side_effect=forbidden):
            for call in callers:
                with pytest.raises(RuntimeError) as caught:
                    call()
                assert caught.value.__cause__ is original
                assert backend in str(caught.value).lower()
        """
    )


@pytest.mark.parametrize("backend", ["mooncake", "nixl"])
@pytest.mark.parametrize("error_type", ["OSError", "RuntimeError"])
def test_non_import_errors_propagate_eagerly(backend, error_type):
    _run(
        f"""
        original = {error_type}('injected native initialization failure')
        class BrokenDependency(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname.split('.')[0] == {backend!r}:
                    raise original
                if fullname.split('.')[0] in {{'mooncake', 'nixl'}}:
                    raise ModuleNotFoundError(fullname, name=fullname.split('.')[0])
        sys.meta_path.insert(0, BrokenDependency())
        with pytest.raises({error_type}) as caught:
            import sglang_omni.relay
        assert caught.value is original
        """
    )


def test_top_level_absence_allows_real_shm():
    _run(
        """
        import asyncio
        class MissingDependency(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname.split('.')[0] in {'mooncake', 'nixl'}:
                    raise ModuleNotFoundError(fullname, name=fullname.split('.')[0])
        sys.meta_path.insert(0, MissingDependency())
        from sglang_omni.relay.base import create_relay
        async def transfer():
            sender = create_relay('shm', engine_id='sender', device='cpu', credits=1)
            receiver = create_relay('shm', engine_id='receiver', device='cpu', credits=1)
            source = torch.arange(4096, dtype=torch.float32)
            destination = torch.empty_like(source)
            put = await sender.put_async(source)
            get = await receiver.get_async(put.metadata, destination)
            await get.wait_for_completion()
            assert torch.equal(source, destination)
            put.mark_receiver_done()
            await put.wait_for_completion()
            sender.close()
            receiver.close()
        asyncio.run(transfer())
        assert not [r for r in records if r.name.startswith('sglang_omni.relay') and r.levelno >= logging.WARNING]
        """
    )
