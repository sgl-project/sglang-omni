# SPDX-License-Identifier: Apache-2.0
"""Unit tests for pickle-free IPC wire codec, worker sessions, and trainer sync."""

from __future__ import annotations

import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import pytest
import torch
from torch.multiprocessing import reductions

from sglang_omni.lora.ipc_wire import (
    ENCODING_CUDA_IPC,
    ENCODING_INLINE_B64,
    WIRE_CODEC_VERSION,
    decode_wire_tensor,
    encode_cuda_ipc_handle,
    encode_inline_tensor,
    release_unpublished_ipc_handle,
)
from sglang_omni.lora.merged_manager import MergedLoRAManager
from sglang_omni.lora.weight_transfer import (
    WARNED_ALLOCATOR_PIDS,
    LoRAAdapterWeightSource,
    MergedLoRATrainerSync,
    MergedLoRAWorkerSession,
    warn_if_allocator_not_expandable,
)
from sglang_omni.models.auk.dit import AuKDit


def make_small_dit() -> AuKDit:
    torch.manual_seed(11)
    return AuKDit(
        dim=16,
        num_layers=1,
        num_single_layers=1,
        heads=2,
        dim_head=8,
        text_hidden_dim=8,
        latent_dim=4,
    )


def test_ipc_wire_inline_round_trip_and_validation() -> None:
    for dtype in (torch.float32, torch.bfloat16, torch.float16):
        tensor = torch.randn(4, 6, dtype=dtype).t()
        wire = encode_inline_tensor(tensor)
        assert wire["version"] == WIRE_CODEC_VERSION
        assert wire["encoding"] == ENCODING_INLINE_B64
        decoded = decode_wire_tensor(wire)
        assert decoded.dtype == dtype
        assert torch.equal(decoded, tensor.contiguous())

    with pytest.raises(ValueError, match="unsupported"):
        decode_wire_tensor(
            {
                "version": WIRE_CODEC_VERSION,
                "encoding": "pickle_v0",
                "dtype": "torch.float32",
                "shape": [2, 2],
            }
        )

    bad_dtype = encode_inline_tensor(torch.randn(2, 2))
    bad_dtype["dtype"] = "complex128"
    with pytest.raises(ValueError, match="dtype"):
        decode_wire_tensor(bad_dtype)

    bad_size = encode_inline_tensor(torch.randn(2, 2))
    bad_size["shape"] = [4, 4]
    with pytest.raises(ValueError, match="byte"):
        decode_wire_tensor(bad_size)


def test_ipc_wire_cuda_handle_14_and_15_args_and_release(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorded_calls: list[tuple[Any, ...]] = []
    expected = torch.arange(8, dtype=torch.float32).reshape(2, 4)

    def fake_rebuild(*args: Any) -> torch.Tensor:
        recorded_calls.append(args)
        return expected

    fake_rebuild.__module__ = "torch.multiprocessing.reductions"
    fake_rebuild.__name__ = "rebuild_cuda_tensor"
    monkeypatch.setattr(reductions, "rebuild_cuda_tensor", fake_rebuild)

    # Disallowed callable is rejected strictly.
    with pytest.raises(ValueError, match="rebuild_cuda_tensor"):
        encode_cuda_ipc_handle((eval, ()))  # noqa: S307

    args_14 = (
        torch.Tensor,
        (2, 4),
        (4, 1),
        0,
        torch.storage.UntypedStorage,
        torch.float32,
        0,
        b"handle-14",
        32,
        0,
        False,
        b"ref-14",
        0,
        b"event-14",
    )
    wire_14 = encode_cuda_ipc_handle((fake_rebuild, args_14))
    assert wire_14["encoding"] == ENCODING_CUDA_IPC
    assert wire_14["event_sync_required"] is False
    decoded_14 = decode_wire_tensor(wire_14)
    assert torch.equal(decoded_14, expected)
    assert len(recorded_calls[-1]) == 15

    args_15 = (*args_14, True)
    wire_15 = encode_cuda_ipc_handle((fake_rebuild, args_15))
    assert wire_15["event_sync_required"] is True
    decoded_15 = decode_wire_tensor(wire_15)
    assert torch.equal(decoded_15, expected)
    assert recorded_calls[-1][14] is True

    # Releasing an unpublished handle calls the IPC ref-counter releaser and returns True.
    released_counters: list[tuple[bytes, int]] = []
    assert (
        release_unpublished_ipc_handle(
            wire_14,
            release_fn=lambda h, off: released_counters.append((h, off)),
        )
        is True
    )
    assert released_counters == [(b"ref-14", 0)]

    # Inline payloads need no IPC release.
    inline_wire = encode_inline_tensor(expected)
    assert release_unpublished_ipc_handle(inline_wire) is False


def test_warn_if_allocator_not_expandable_once_per_pid(
    caplog: pytest.LogCaptureFixture,
) -> None:
    WARNED_ALLOCATOR_PIDS.clear()
    with caplog.at_level(logging.WARNING):
        assert warn_if_allocator_not_expandable(is_cuda=False, environ={}) is False
        assert (
            warn_if_allocator_not_expandable(
                is_cuda=True,
                environ={"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
            )
            is False
        )
        assert warn_if_allocator_not_expandable(is_cuda=True, environ={}) is True
        # Second call in the same PID is suppressed.
        assert warn_if_allocator_not_expandable(is_cuda=True, environ={}) is False
    assert sum("expandable_segments:True" in r.message for r in caplog.records) == 1
    WARNED_ALLOCATOR_PIDS.clear()


def test_adapter_weight_source_deterministic_sort_and_pair_validation() -> None:
    source = LoRAAdapterWeightSource(
        adapter_id="s1",
        target_modules=["to_q", "to_k"],
        named_tensors=[
            ("b.to_k.lora_B.weight", torch.zeros(4, 2)),
            ("a.to_q.lora_A.weight", torch.zeros(2, 4)),
            ("b.to_k.lora_A.weight", torch.zeros(2, 4)),
            ("a.to_q.lora_B.weight", torch.zeros(4, 2)),
        ],
    )
    sorted_names = [name for name, _ in source.sorted_items()]
    assert sorted_names == sorted(sorted_names)
    pairs = source.build_adapter_tensors()
    assert [p.module_name for p in pairs] == ["a.to_q", "b.to_k"]

    broken = LoRAAdapterWeightSource(
        adapter_id="bad",
        target_modules=["to_q"],
        named_tensors=[("a.to_q.lora_A.weight", torch.zeros(2, 4))],
    )
    with pytest.raises(ValueError, match="incomplete LoRA pair"):
        broken.build_adapter_tensors()


def test_worker_session_lifecycle_and_cleanup_logging(
    caplog: pytest.LogCaptureFixture,
) -> None:
    dit = make_small_dit()
    manager = MergedLoRAManager(
        dit, architecture="AuKForConditionalGeneration", prefer_pinned=False
    )
    cleanup_should_fail = False

    def on_cleanup() -> None:
        if cleanup_should_fail:
            raise RuntimeError("cleanup failure")

    session = MergedLoRAWorkerSession(manager, on_cleanup=on_cleanup, is_cuda=False)

    qkv_base = dit.transformer_blocks[0].attn.to_qkv.weight.detach().clone()
    lora_a = torch.randn(2, 16)
    lora_b = torch.randn(16, 2)

    start_res = session.start_update("u1")
    assert start_res["resumed"] is False

    # Concurrent update_id is rejected while u1 is active.
    with pytest.raises(RuntimeError, match="already active"):
        session.start_update("u2")

    stage_res = session.stage_update(
        "u1",
        {
            "adapter_id": "a1",
            "target_modules": ["to_q"],
            "scale": 0.5,
            "tensors": {
                "dit.transformer_blocks.0.attn.to_q.lora_A.weight": encode_inline_tensor(
                    lora_a
                ),
                "dit.transformer_blocks.0.attn.to_q.lora_B.weight": encode_inline_tensor(
                    lora_b
                ),
            },
        },
    )
    assert stage_res["matched_modules"] == 1

    # Live weights stay unchanged until finish.
    assert torch.equal(dit.transformer_blocks[0].attn.to_qkv.weight, qkv_base)

    finish_res = session.finish_update("u1")
    assert finish_res["updated_params"] == 1
    assert manager.active_adapter_id == "a1"
    assert torch.allclose(
        dit.transformer_blocks[0].attn.to_qkv.weight[:16],
        qkv_base[:16] + 0.5 * (lora_b @ lora_a),
        atol=1e-5,
    )

    # Restore base resets live weights.
    restore_res = session.restore_base()
    assert restore_res["restored_params"] == 1
    assert torch.equal(dit.transformer_blocks[0].attn.to_qkv.weight, qkv_base)

    # Verify cleanup failure logs with exc_info without failing abort_update.
    cleanup_should_fail = True
    session.start_update("u3")
    with caplog.at_level(logging.WARNING):
        abort_res = session.abort_update("u3")
    assert abort_res["aborted_update_id"] == "u3"
    assert abort_res["committed"] is False
    assert any(
        "cleanup failed" in rec.message and rec.exc_info is not None
        for rec in caplog.records
    )


def test_trainer_sync_e2e_and_two_rank_failure_modes() -> None:
    dit = make_small_dit()
    manager = MergedLoRAManager(
        dit, architecture="AuKForConditionalGeneration", prefer_pinned=False
    )
    worker = MergedLoRAWorkerSession(manager, is_cuda=False)
    qkv_base = dit.transformer_blocks[0].attn.to_qkv.weight.detach().clone()

    events: list[str] = []

    def start_rpc(uid: str) -> dict[str, Any]:
        events.append("start")
        return worker.start_update(uid)

    def stage_rpc(uid: str, payload: dict[str, Any]) -> dict[str, Any]:
        events.append("stage")
        return worker.stage_update(uid, payload)

    def finish_rpc(uid: str) -> dict[str, Any]:
        events.append("finish")
        return worker.finish_update(uid)

    def abort_rpc(uid: str) -> dict[str, Any]:
        events.append("abort")
        return worker.abort_update(uid)

    def on_ipc_collect() -> None:
        events.append("ipc_collect")

    lora_a = torch.randn(2, 16)
    lora_b = torch.randn(16, 2)
    source = LoRAAdapterWeightSource(
        adapter_id="sync_v1",
        target_modules=["to_q"],
        named_tensors=[
            ("dit.transformer_blocks.0.attn.to_q.lora_A.weight", lora_a),
            ("dit.transformer_blocks.0.attn.to_q.lora_B.weight", lora_b),
        ],
        scale=1.0,
    )

    trainer = MergedLoRATrainerSync(
        start_rpc=start_rpc,
        stage_rpc=stage_rpc,
        finish_rpc=finish_rpc,
        abort_rpc=abort_rpc,
        ipc_collect_fn=on_ipc_collect,
    )
    res = trainer.sync_adapter(source, update_id="u-sync-1")
    assert res.update_id == "u-sync-1"
    assert res.tensor_count == 2
    assert events == ["start", "stage", "finish", "ipc_collect"]
    assert torch.allclose(
        dit.transformer_blocks[0].attn.to_qkv.weight[:16],
        qkv_base[:16] + (lora_b @ lora_a),
        atol=1e-5,
    )


def test_trainer_sync_two_rank_collective_abort_and_handle_release() -> None:
    # Case 1: Phase 1a synchronize_fn fails on rank 1 -> neither rank exports or calls RPC.
    barrier_1 = threading.Barrier(2)
    gather_slots_1: list[object] = [None, None]

    def make_gather(barrier: threading.Barrier, slots: list[object], rank: int):
        def gather(val: object) -> list[object]:
            slots[rank] = val
            barrier.wait()
            snap = list(slots)
            barrier.wait()
            return snap

        return gather

    rpc_calls: list[str] = []
    export_calls: list[int] = []

    def run_phase1a_fail(rank: int) -> None:
        source = LoRAAdapterWeightSource(
            adapter_id="a",
            target_modules=["to_q"],
            named_tensors=[
                ("x.to_q.lora_A.weight", torch.zeros(2, 4)),
                ("x.to_q.lora_B.weight", torch.zeros(4, 2)),
            ],
        )

        def sync_fn() -> None:
            if rank == 1:
                raise RuntimeError("rank1 device sync error")

        def export_fn(t: torch.Tensor) -> dict[str, Any]:
            export_calls.append(rank)
            return encode_inline_tensor(t)

        trainer = MergedLoRATrainerSync(
            start_rpc=lambda uid: rpc_calls.append(f"start:{rank}")
            or {"success": True},
            stage_rpc=lambda uid, p: rpc_calls.append(f"stage:{rank}")
            or {"success": True},
            finish_rpc=lambda uid: rpc_calls.append(f"finish:{rank}")
            or {"success": True},
            abort_rpc=lambda uid: rpc_calls.append(f"abort:{rank}")
            or {"success": True},
            all_gather_fn=make_gather(barrier_1, gather_slots_1, rank),
            synchronize_fn=sync_fn,
            export_tensor_fn=export_fn,
        )
        trainer.sync_adapter(source)

    with ThreadPoolExecutor(max_workers=2) as pool:
        f0 = pool.submit(run_phase1a_fail, 0)
        f1 = pool.submit(run_phase1a_fail, 1)
        with pytest.raises(RuntimeError, match="rank1 device sync error"):
            f0.result()
        with pytest.raises(RuntimeError, match="rank1 device sync error"):
            f1.result()
    assert rpc_calls == []
    assert export_calls == []

    # Case 2: Phase 2 export fails on rank 1 -> rank 0 releases unpublished handles, no RPCs.
    barrier_2 = threading.Barrier(2)
    gather_slots_2: list[object] = [None, None]
    released: list[dict[str, Any]] = []
    rpc_calls_2: list[str] = []

    def run_phase2_fail(rank: int) -> None:
        source = LoRAAdapterWeightSource(
            adapter_id="a",
            target_modules=["to_q"],
            named_tensors=[
                ("x.to_q.lora_A.weight", torch.zeros(2, 4)),
                ("x.to_q.lora_B.weight", torch.zeros(4, 2)),
            ],
        )

        def export_fn(t: torch.Tensor) -> dict[str, Any]:
            if rank == 1:
                raise RuntimeError("rank1 ipc export failure")
            return {"encoding": ENCODING_CUDA_IPC, "rank": rank}

        trainer = MergedLoRATrainerSync(
            start_rpc=lambda uid: rpc_calls_2.append(f"start:{rank}")
            or {"success": True},
            stage_rpc=lambda uid, p: rpc_calls_2.append(f"stage:{rank}")
            or {"success": True},
            finish_rpc=lambda uid: rpc_calls_2.append(f"finish:{rank}")
            or {"success": True},
            abort_rpc=lambda uid: rpc_calls_2.append(f"abort:{rank}")
            or {"success": True},
            all_gather_fn=make_gather(barrier_2, gather_slots_2, rank),
            export_tensor_fn=export_fn,
            release_handle_fn=lambda h: released.append(dict(h)) or True,
        )
        trainer.sync_adapter(source)

    with ThreadPoolExecutor(max_workers=2) as pool:
        f0 = pool.submit(run_phase2_fail, 0)
        f1 = pool.submit(run_phase2_fail, 1)
        with pytest.raises(RuntimeError, match="rank1 ipc export failure"):
            f0.result()
        with pytest.raises(RuntimeError, match="rank1 ipc export failure"):
            f1.result()
    assert rpc_calls_2 == []
    assert len(released) == 2

    # Case 3: Lost-reply start_rpc aborts remote with caller-owned update_id and releases handles.
    aborted_ids: list[str] = []
    released_3: list[dict[str, Any]] = []
    source_3 = LoRAAdapterWeightSource(
        adapter_id="a",
        target_modules=["to_q"],
        named_tensors=[
            ("x.to_q.lora_A.weight", torch.zeros(2, 4)),
            ("x.to_q.lora_B.weight", torch.zeros(4, 2)),
        ],
    )

    def lost_reply_start(uid: str) -> dict[str, Any]:
        raise TimeoutError("start_rpc reply timed out")

    trainer_3 = MergedLoRATrainerSync(
        start_rpc=lost_reply_start,
        stage_rpc=lambda uid, p: {"success": True},
        finish_rpc=lambda uid: {"success": True},
        abort_rpc=lambda uid: aborted_ids.append(uid) or {"success": True},
        export_tensor_fn=lambda t: {"encoding": ENCODING_CUDA_IPC},
        release_handle_fn=lambda h: released_3.append(dict(h)) or True,
    )
    with pytest.raises(RuntimeError, match="start_rpc reply timed out"):
        trainer_3.sync_adapter(source_3, update_id="caller-owned-id")
    assert aborted_ids == ["caller-owned-id"]
    assert len(released_3) == 2
