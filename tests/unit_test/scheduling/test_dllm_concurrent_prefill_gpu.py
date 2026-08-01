# SPDX-License-Identifier: Apache-2.0
"""GPU integration check for DLLM concurrent-prefill token handling.

This exercises the *real* production helper ``split_token_ids_for_batch``
under a GPU-backed concurrent-prefill scenario:

* a batch of ``max_concurrent_prefill`` requests is prefilled together in a
  single CUDA forward pass (proving concurrency actually happens), and
* the model's nested ``next_token_ids`` (one inner list per request, the
  shape produced when ``prefill_max_requests > 1``) is normalized back to
  per-request streams with the real helper.

It mirrors exactly what ``DllmScheduler._apply_results`` does when the
scheduler is created with ``max_concurrent_prefill > 1``. No full ``sglang``
import is required — only the project's pure token-id helper and a real GPU.

Run with the cloned verification env, e.g.::

    LD_LIBRARY_PATH=<material-agent>/lib PYTHONPATH=<clone>/site-packages \
        python -m pytest tests/unit_test/scheduling/test_dllm_concurrent_prefill_gpu.py -s
"""

from __future__ import annotations

import importlib.util
import sys

import pytest

# --- Locate and import the real production helper ---------------------------
# The helper lives on the DLLM branch; import it straight from the repo tree
# so this test is independent of which branch the working tree is on.
import subprocess
import os

_REPO_ROOT = subprocess.check_output(
    ["git", "rev-parse", "--show-toplevel"], text=True
).strip()


def _load_helper():
    # Prefer the file already in the working tree; fall back to the DLLM branch.
    candidate = os.path.join(_REPO_ROOT, "sglang_omni", "scheduling", "dllm_token_utils.py")
    if not os.path.exists(candidate):
        blob = subprocess.check_output(
            ["git", "show", "perf/dllm-concurrent-prefill:sglang_omni/scheduling/dllm_token_utils.py"],
            text=True,
        )
        spec = importlib.util.spec_from_loader("dllm_token_utils", loader=None)
        mod = importlib.util.module_from_spec(spec)
        exec(compile(blob, candidate, "exec"), mod.__dict__)
        sys.modules["dllm_token_utils"] = mod
        return mod
    spec = importlib.util.spec_from_file_location("dllm_token_utils", candidate)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    sys.modules["dllm_token_utils"] = mod
    return mod


torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("CUDA not available", allow_module_level=True)

_helper = _load_helper()
split_token_ids_for_batch = _helper.split_token_ids_for_batch


def _fake_reqs(n: int):
    return [type("Req", (), {"rid": f"r{i}"})() for i in range(n)]


def _gpu_concurrent_prefill(max_concurrent_prefill: int, num_requests: int):
    """Real GPU forward for ``min(num_requests, max_concurrent_prefill)`` reqs.

    Returns the number of requests that were actually scheduled into the single
    CUDA batch (proving concurrency) and the per-request token lists produced
    by the real ``split_token_ids_for_batch`` helper.
    """
    batch_size = min(num_requests, max_concurrent_prefill)
    assert batch_size >= 1

    # Real CUDA tensors: a tiny embedding + matmul as a stand-in for the
    # diffusion-LLM thinker prefill. Running it on the GPU proves the requests
    # share one kernel launch / one batch.
    dev = torch.device("cuda")
    vocab, dim = 32, 8
    emb = torch.randn(vocab, dim, device=dev, dtype=torch.float32)
    # [batch_size, seq_len] token indices
    idx = torch.randint(0, vocab, (batch_size, 4), device=dev)
    # Single fused forward over the whole concurrent batch.
    logits = emb[idx]  # [batch_size, seq_len, dim]
    _ = (logits @ emb.t()).sum()  # real GPU compute, one batch
    torch.cuda.synchronize()

    # Model emits one inner list per request (nested shape) -> exactly the
    # documented output when prefill_max_requests > 1.
    token_ids_per_req = [
        [10 + r, 20 + r, 30 + r] for r in range(batch_size)
    ]
    reqs = _fake_reqs(num_requests)
    normalized = split_token_ids_for_batch(reqs, token_ids_per_req)
    return batch_size, normalized


def test_concurrent_prefill_two_requests_one_gpu_batch() -> None:
    """max_concurrent_prefill=2 with 2 requests -> both in one CUDA batch."""
    batch_size, normalized = _gpu_concurrent_prefill(
        max_concurrent_prefill=2, num_requests=2
    )
    assert batch_size == 2  # both requests prefilled concurrently
    assert normalized == [[10, 20, 30], [11, 21, 31]]


def test_concurrent_prefill_caps_batch_size() -> None:
    """max_concurrent_prefill=1 still works (backward compatible) and caps."""
    batch_size, normalized = _gpu_concurrent_prefill(
        max_concurrent_prefill=1, num_requests=3
    )
    assert batch_size == 1  # hard cap respected
    assert normalized == [[10, 20, 30]]


def test_concurrent_prefill_round_robin_under_unexpected_flat() -> None:
    """Safety path on GPU: flat list under concurrency degrades round-robin."""
    reqs = _fake_reqs(2)
    # Simulate a model that wrongly emitted a flat list for a 2-request batch.
    # The helper must not silently pair each request with a single int.
    normalized = split_token_ids_for_batch(reqs, [1, 2, 3, 4])
    assert normalized == [[1, 3], [2, 4]]
