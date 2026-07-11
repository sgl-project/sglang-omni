# SPDX-License-Identifier: Apache-2.0
"""Tests for the MOSS-TD Whisper encoder CUDA-graph runner."""

from __future__ import annotations

import ast
import glob
import inspect
import os
import textwrap

import pytest
import torch

from sglang_omni.models.moss_transcribe_diarize.encoder_cuda_graph import (
    WhisperEncoderCudaGraphRunner,
)

pytestmark = pytest.mark.gpu

_HAS_CUDA = torch.cuda.is_available()

_CKPT_GLOB = (
    "/root/.cache/huggingface/hub/"
    "models--OpenMOSS-Team--MOSS-Transcribe-Diarize/snapshots/*/"
)

_INPUT_FEATURE_LEN = 3000
_CHUNK_BUCKETS = [1, 2, 4, 8, 16, 32]
# Chunk counts exercised by the bit-identity test: exact bucket hits plus counts
# that pad up to the next bucket (3->4, 5->8, 12->16, 20->32).
_TEST_CHUNKS = [1, 2, 3, 4, 5, 8, 12, 16, 20, 32]


def test_capture_uses_thread_local_error_mode():
    source = textwrap.dedent(
        inspect.getsource(WhisperEncoderCudaGraphRunner._capture_bucket)
    )
    tree = ast.parse(source)
    graph_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "graph"
        and isinstance(node.func.value, ast.Attribute)
        and node.func.value.attr == "cuda"
    ]
    assert graph_calls, "encoder CUDA graph capture call not found"
    assert any(
        kw.arg == "capture_error_mode"
        and isinstance(kw.value, ast.Constant)
        and kw.value.value == "thread_local"
        for call in graph_calls
        for kw in call.keywords
    ), "encoder CUDA graph capture must use thread-local error mode"


@pytest.fixture(scope="module")
def encoder_bundle():
    """sglang WhisperEncoder built from the MOSS-TD checkpoint. sglang's encoder
    uses TP-parallel layers, so a TP=1 group must exist before construction."""
    if not _HAS_CUDA:
        pytest.skip("needs CUDA")
    snaps = glob.glob(_CKPT_GLOB)
    if not snaps:
        pytest.skip("MOSS-Transcribe-Diarize checkpoint snapshot not found")

    from sglang.srt.distributed.parallel_state import (
        init_distributed_environment,
        initialize_model_parallel,
        model_parallel_is_initialized,
    )
    from sglang.srt.models.whisper import WhisperEncoder
    from transformers import AutoConfig

    torch.cuda.set_device(0)
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29551")
    if not torch.distributed.is_initialized():
        init_distributed_environment(
            world_size=1,
            rank=0,
            local_rank=0,
            distributed_init_method=f"tcp://127.0.0.1:{os.environ['MASTER_PORT']}",
            backend="nccl",
        )
    if not model_parallel_is_initialized():
        initialize_model_parallel(tensor_model_parallel_size=1)

    audio_config = AutoConfig.from_pretrained(
        snaps[0], trust_remote_code=True
    ).audio_config
    encoder = WhisperEncoder(audio_config).cuda().to(torch.bfloat16).eval()
    num_mel_bins = int(audio_config.num_mel_bins)
    runner = WhisperEncoderCudaGraphRunner(encoder, num_mel_bins, _INPUT_FEATURE_LEN)
    runner.capture(_CHUNK_BUCKETS)
    return encoder, num_mel_bins, runner


def _feat(num_mel_bins: int, n: int) -> torch.Tensor:
    return torch.randn(
        n, num_mel_bins, _INPUT_FEATURE_LEN, device="cuda", dtype=torch.bfloat16
    )


def _pos() -> torch.Tensor:
    encoder_len = (_INPUT_FEATURE_LEN - 1) // 2 + 1
    return torch.arange(encoder_len, device="cuda", dtype=torch.long)


def test_some_graphs_captured(encoder_bundle):
    _, _, runner = encoder_bundle
    assert runner._graphs, "no encoder CUDA graphs captured (all fell back to eager)"
    assert set(runner._graphs) == set(_CHUNK_BUCKETS)


@pytest.mark.parametrize("n", _TEST_CHUNKS)
def test_graph_bit_identical_to_eager(encoder_bundle, n):
    """Graphed replay must equal eager bit-for-bit at the SAME batch size. When n
    pads up to a larger bucket, we compare against eager run at that padded batch
    (not batch n) to avoid batch sizes affecting kernel selections."""
    encoder, num_mel_bins, runner = encoder_bundle
    pos = _pos()
    chunk_bucket = min(c for c in _CHUNK_BUCKETS if c >= n)
    torch.manual_seed(100 + n)
    feat = _feat(num_mel_bins, n)
    feat_padded = torch.zeros(
        chunk_bucket,
        num_mel_bins,
        _INPUT_FEATURE_LEN,
        device="cuda",
        dtype=torch.bfloat16,
    )
    feat_padded[:n] = feat
    with torch.no_grad():
        eager = encoder(feat_padded, pos, None)[:n]
        graphed = runner.run(feat, pos, None)
    assert torch.equal(eager, graphed), (
        f"graph replay not bit-identical to same-batch eager (n={n}, "
        f"bucket={chunk_bucket}): "
        f"max|delta|={(eager.float() - graphed.float()).abs().max().item():.3e}"
    )


def test_over_largest_bucket_falls_back_to_eager(encoder_bundle):
    """A chunk count above the largest captured bucket falls back to eager and
    still matches a direct eager call."""
    encoder, num_mel_bins, _ = encoder_bundle
    runner = WhisperEncoderCudaGraphRunner(encoder, num_mel_bins, _INPUT_FEATURE_LEN)
    runner.capture([1, 2])
    feat = _feat(num_mel_bins, 5)
    pos = _pos()
    with torch.no_grad():
        eager = encoder(feat, pos, None)
        out = runner.run(feat, pos, None)
    assert torch.equal(eager, out)


def test_vram_guard_skips_capture(encoder_bundle):
    """Below the configured VRAM headroom, capture is skipped (no graphs); forced
    via an absurd min_free_gb."""
    encoder, num_mel_bins, _ = encoder_bundle
    runner = WhisperEncoderCudaGraphRunner(
        encoder, num_mel_bins, _INPUT_FEATURE_LEN, min_free_gb=100000.0
    )
    runner.capture(_CHUNK_BUCKETS)
    assert runner._graphs == {}, "VRAM guard must skip all captures"


def test_capture_failure_falls_back_to_eager(encoder_bundle):
    """A capture exception is caught per-bucket, that bucket dropped; run() then
    falls back to eager, bit-identical to a direct eager call."""
    encoder, num_mel_bins, _ = encoder_bundle
    runner = WhisperEncoderCudaGraphRunner(encoder, num_mel_bins, _INPUT_FEATURE_LEN)

    def boom(*args, **kwargs):
        raise RuntimeError("simulated capture OOM")

    runner._capture_bucket = boom
    runner.capture(_CHUNK_BUCKETS)
    assert runner._graphs == {}, "capture failures must be caught -> no graphs"

    feat = _feat(num_mel_bins, 2)
    pos = _pos()
    with torch.no_grad():
        eager = encoder(feat, pos, None)
        out = runner.run(feat, pos, None)
    assert torch.equal(eager, out)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v", "-s"]))
