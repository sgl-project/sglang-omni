# SPDX-License-Identifier: Apache-2.0
"""The startup channel that carries parameter handles between PD halves.

The halves are separate processes with no channel at load time. These pin the
three properties the exchange depends on: the run directory is derivable from
an endpoint the stage already has, a reader never sees a partial file, and an
unpublished peer reads as absent rather than blocking, because the reader holds
the GPU startup lock that peer needs.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from sglang_omni.model_runner.weight_rendezvous import (
    PublisherUnavailable,
    RendezvousUnavailable,
    publish_parameter_handles,
    read_parameter_handles,
    rendezvous_dir_from_endpoint,
    wait_for_parameter_handles,
)


def test_the_run_directory_comes_from_an_endpoint_the_stage_already_has() -> None:
    """allocate_endpoints puts every socket directly in the run directory."""
    endpoint = "ipc:///tmp/sglang_omni/qwen3-omni-ab12/stage_thinker_prefill.sock"

    assert rendezvous_dir_from_endpoint(endpoint) == Path(
        "/tmp/sglang_omni/qwen3-omni-ab12"
    )


def test_a_non_ipc_endpoint_is_rejected() -> None:
    """A TCP endpoint has no run directory, and guessing one would misplace it."""
    with pytest.raises(RendezvousUnavailable):
        rendezvous_dir_from_endpoint("tcp://127.0.0.1:5555")


def test_handles_survive_the_round_trip(tmp_path: Path) -> None:
    handles = {"model.layers.0.weight": ("rebuild", (1, 2, 3))}

    publish_parameter_handles(
        handles, rendezvous_dir=tmp_path, stage_name="prefill", gpu_id=0
    )

    manifest = read_parameter_handles(
        rendezvous_dir=tmp_path, stage_name="prefill", gpu_id=0
    )
    assert manifest.parameters == handles
    assert manifest.generation
    assert manifest.publisher_pid > 0


def test_an_unpublished_peer_reads_as_none(tmp_path: Path) -> None:
    """The caller publishes its own handles instead, so nothing waits."""
    assert (
        read_parameter_handles(rendezvous_dir=tmp_path, stage_name="prefill", gpu_id=0)
        is None
    )


def test_a_reader_never_observes_a_partial_file(tmp_path: Path) -> None:
    """Publishing stages under another name, so the final path appears whole."""
    directory = tmp_path / "pd-weights"
    directory.mkdir()
    big = {f"layer.{i}.weight": ("rebuild", tuple(range(64))) for i in range(400)}

    publish_parameter_handles(
        big, rendezvous_dir=tmp_path, stage_name="prefill", gpu_id=0
    )

    leftovers = [p.name for p in directory.iterdir() if p.name != "prefill.pkl"]
    assert leftovers == []
    assert (
        read_parameter_handles(
            rendezvous_dir=tmp_path, stage_name="prefill", gpu_id=0
        ).parameters
        == big
    )


def test_handles_published_for_another_gpu_are_declined(tmp_path: Path) -> None:
    """A CUDA IPC handle names memory on one device; adopting across is wrong."""
    publish_parameter_handles(
        {"w": ("rebuild", ())},
        rendezvous_dir=tmp_path,
        stage_name="prefill",
        gpu_id=0,
    )

    assert (
        read_parameter_handles(rendezvous_dir=tmp_path, stage_name="prefill", gpu_id=1)
        is None
    )


def test_a_peer_on_another_gpu_stops_the_wait_at_once(tmp_path: Path) -> None:
    """No amount of waiting turns another card's handles into usable ones."""
    import time

    publish_parameter_handles(
        {"w": ("rebuild", ())},
        rendezvous_dir=tmp_path,
        stage_name="prefill",
        gpu_id=0,
        weight_bytes=1,
    )
    started = time.monotonic()

    result = wait_for_parameter_handles(
        rendezvous_dir=tmp_path, stage_name="prefill", gpu_id=1, timeout_s=30.0
    )

    assert result is None
    assert time.monotonic() - started < 1.0


def test_dead_publisher_manifest_is_rejected(tmp_path: Path) -> None:
    publish_parameter_handles(
        {"w": ("rebuild", ())},
        rendezvous_dir=tmp_path,
        stage_name="prefill",
        gpu_id=0,
        publisher_pid=999_999_999,
    )

    with pytest.raises(PublisherUnavailable, match="publisher process"):
        read_parameter_handles(rendezvous_dir=tmp_path, stage_name="prefill", gpu_id=0)


def test_republishing_changes_the_publisher_generation(tmp_path: Path) -> None:
    kwargs = dict(rendezvous_dir=tmp_path, stage_name="prefill", gpu_id=0)
    publish_parameter_handles({"w": ("first", ())}, **kwargs)
    first = read_parameter_handles(**kwargs)
    publish_parameter_handles({"w": ("second", ())}, **kwargs)
    second = read_parameter_handles(**kwargs)

    assert first.generation != second.generation
