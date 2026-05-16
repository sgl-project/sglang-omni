# SPDX-License-Identifier: Apache-2.0
"""Shared fixtures and hooks for test_model tests."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

S2PRO_TTS_ALLOWED_CONCURRENCIES = (1, 2, 4, 8, 16)
S2PRO_STAGE_NONSTREAM = "s2pro-stage-1-nonstream"
S2PRO_STAGE_STREAM = "s2pro-stage-2-stream"
S2PRO_STAGE_CONSISTENCY = "s2pro-stage-3-consistency"
S2PRO_CI_STAGES = (
    S2PRO_STAGE_NONSTREAM,
    S2PRO_STAGE_STREAM,
    S2PRO_STAGE_CONSISTENCY,
)
S2PRO_TTS_FULL_SWEEP_VALUE = "all"
S2PRO_STAGE_ALL = "all"
S2PRO_TTS_CONCURRENCY_OPTION = "--concurrency"
SELECTED_S2PRO_TTS_CONCURRENCIES = pytest.StashKey[tuple[int, ...]]()
S2PRO_STAGE_OPTION = "--s2pro-stage"
SELECTED_S2PRO_CI_STAGE = pytest.StashKey[str]()
QWEN3_OMNI_MODEL_PATH = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
# Single source of truth for the model path used by Qwen3-Omni vision-encoder
# benchmarks and the SGLang state they bring up. Honors
# ``SGLANG_OMNI_TEST_QWEN3_MODEL=/local/path`` so an offline runner does not
# fall back to the HF hub name in ``ServerArgs.model_path``.
QWEN3_OMNI_TEST_MODEL_PATH = os.environ.get(
    "SGLANG_OMNI_TEST_QWEN3_MODEL", QWEN3_OMNI_MODEL_PATH
)
QWEN3_OMNI_STARTUP_TIMEOUT = 300


@pytest.fixture(scope="module")
def qwen3_omni_thinker_server(tmp_path_factory: pytest.TempPathFactory):
    """Start the text-only Qwen3-Omni server and wait until healthy."""
    from sglang_omni.utils import find_available_port
    from tests.utils import (
        ServerHandle,
        server_log_file,
        start_server_from_cmd,
        stop_server,
    )

    port = find_available_port()
    log_file = server_log_file(tmp_path_factory)
    cmd = [
        sys.executable,
        "examples/run_qwen3_omni_server.py",
        "--model-path",
        QWEN3_OMNI_MODEL_PATH,
        "--port",
        str(port),
        "--model-name",
        "qwen3-omni",
        "--thinker-max-seq-len",
        "32768",
        "--mem-fraction-static",
        "0.78",
    ]
    proc = start_server_from_cmd(
        cmd, log_file, port, timeout=QWEN3_OMNI_STARTUP_TIMEOUT
    )
    yield ServerHandle(proc=proc, port=port)
    stop_server(proc)


@pytest.fixture(scope="module")
def qwen3_omni_talker_server(tmp_path_factory: pytest.TempPathFactory):
    """Start the Qwen3-Omni speech server and wait until healthy."""
    from sglang_omni.utils import find_available_port
    from tests.utils import (
        ServerHandle,
        server_log_file,
        start_server_from_cmd,
        stop_server,
    )

    port = find_available_port()
    log_file = server_log_file(tmp_path_factory)
    cmd = [
        sys.executable,
        "examples/run_qwen3_omni_speech_server.py",
        "--model-path",
        QWEN3_OMNI_MODEL_PATH,
        "--gpu-thinker",
        "0",
        "--gpu-talker",
        "1",
        "--gpu-code2wav",
        "1",
        "--port",
        str(port),
        "--model-name",
        "qwen3-omni",
        "--thinker-max-seq-len",
        "32768",
        "--thinker-mem-fraction-static",
        "0.78",
    ]
    proc = start_server_from_cmd(
        cmd, log_file, port, timeout=QWEN3_OMNI_STARTUP_TIMEOUT
    )
    yield ServerHandle(proc=proc, port=port)
    stop_server(proc)


def _model_cache_present(model_path: str) -> bool:
    """Return True iff *model_path* is either a local directory or an
    already-resolvable HF snapshot. Avoids triggering a multi-GB download
    on a CI runner that did not opt in.
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        return False
    if Path(model_path).exists():
        return True
    try:
        snapshot_download(model_path, local_files_only=True)
    except Exception:
        return False
    return True


def resolve_qwen3_omni_model_dir(model_path: str) -> Path:
    """Return the model directory without triggering a download. Caller is
    responsible for confirming the cache is populated (see
    :func:`_model_cache_present`).
    """
    if Path(model_path).exists():
        return Path(model_path)
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(model_path, local_files_only=True))


@pytest.fixture(scope="module")
def cuda_device():
    """CUDA device for Qwen3-Omni benchmarks. Skips when CUDA is unavailable
    or the Qwen3-Omni checkpoint is not in the local HF cache."""
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    if not _model_cache_present(QWEN3_OMNI_TEST_MODEL_PATH):
        pytest.skip(
            f"{QWEN3_OMNI_TEST_MODEL_PATH} is not in the local HF cache; this "
            f"benchmark test refuses to auto-download a multi-GB checkpoint. "
            f"Pre-populate the cache or set SGLANG_OMNI_TEST_QWEN3_MODEL to a "
            f"local path."
        )
    torch.cuda.set_device(0)
    return torch.device("cuda:0")


@pytest.fixture(scope="session")
def qwen3_omni_vision_sglang_env():
    """Process-global SGLang dist + DP-attention bring-up shared by every
    Qwen3-Omni vision-encoder benchmark module in ``tests/test_model``.

    SGLang's TP group, DP-attention group, and global server-args slot are
    all process-global. Two benchmark modules that each owned an unguarded
    ``initialize_model_parallel`` call would trip an already-initialized
    assertion when the combined command ``pytest -m benchmark tests/test_model``
    runs them in the same process. Hoisting the bring-up to a session
    fixture means it executes at most once per session; the explicit
    ``model_parallel_is_initialized`` / ``torch.distributed.is_initialized``
    guards are belt-and-suspenders against an external initializer.
    """
    import torch
    import torch.distributed as torch_dist

    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    torch.cuda.set_device(0)

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29550")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("NCCL_SOCKET_IFNAME", "lo")
    os.environ.setdefault("NCCL_IB_DISABLE", "1")
    os.environ.setdefault("NCCL_P2P_DISABLE", "1")

    from sglang.srt.configs.model_config import ModelConfig
    from sglang.srt.distributed.parallel_state import (
        init_distributed_environment,
        initialize_model_parallel,
        model_parallel_is_initialized,
    )
    from sglang.srt.layers.dp_attention import initialize_dp_attention
    from sglang.srt.models.qwen3_omni_moe import (  # noqa: F401 -- lazy-import order
        Qwen3OmniMoeVisionEncoder,
    )
    from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler

    if not torch_dist.is_initialized():
        init_distributed_environment(
            world_size=1,
            rank=0,
            distributed_init_method=f"tcp://127.0.0.1:{os.environ['MASTER_PORT']}",
            local_rank=0,
            backend="nccl",
        )
    if not model_parallel_is_initialized():
        initialize_model_parallel(tensor_model_parallel_size=1)

    sa = ServerArgs(
        model_path=QWEN3_OMNI_TEST_MODEL_PATH,
        trust_remote_code=True,
        tp_size=1,
        dtype="bfloat16",
        disable_cuda_graph=True,
        random_seed=123,
    )
    set_global_server_args_for_scheduler(sa)
    initialize_dp_attention(sa, ModelConfig.from_server_args(sa))


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        S2PRO_TTS_CONCURRENCY_OPTION,
        action="store",
        default="1",
        help=(
            "Select the S2-Pro TTS benchmark concurrency. "
            "Use one of {1,2,4,8,16} or 'all' for the full sweep."
        ),
    )
    parser.addoption(
        S2PRO_STAGE_OPTION,
        action="store",
        default=S2PRO_STAGE_ALL,
        help=(
            "Select the S2-Pro CI stage. "
            f"Use one of {S2PRO_CI_STAGES} or '{S2PRO_STAGE_ALL}'."
        ),
    )


def pytest_configure(config: pytest.Config) -> None:
    option_value = config.getoption(S2PRO_TTS_CONCURRENCY_OPTION)
    config.stash[SELECTED_S2PRO_TTS_CONCURRENCIES] = _parse_s2pro_tts_concurrency(
        option_value
    )
    stage_value = config.getoption(S2PRO_STAGE_OPTION)
    config.stash[SELECTED_S2PRO_CI_STAGE] = _parse_s2pro_ci_stage(stage_value)


@pytest.fixture(scope="session")
def selected_s2pro_tts_concurrencies(
    pytestconfig: pytest.Config,
) -> tuple[int, ...]:
    return pytestconfig.stash[SELECTED_S2PRO_TTS_CONCURRENCIES]


@pytest.fixture(scope="session")
def selected_s2pro_ci_stage(pytestconfig: pytest.Config) -> str:
    return pytestconfig.stash[SELECTED_S2PRO_CI_STAGE]


def _parse_s2pro_tts_concurrency(option_value: str) -> tuple[int, ...]:
    normalized_value = option_value.strip().lower()
    if normalized_value == S2PRO_TTS_FULL_SWEEP_VALUE:
        return S2PRO_TTS_ALLOWED_CONCURRENCIES

    try:
        concurrency = int(normalized_value)
    except ValueError as exc:
        raise pytest.UsageError(
            "Invalid value for --concurrency. " "Use one of {1,2,4,8,16} or 'all'."
        ) from exc

    if concurrency not in S2PRO_TTS_ALLOWED_CONCURRENCIES:
        raise pytest.UsageError(
            f"Unsupported concurrency {concurrency}. "
            f"Use one of {S2PRO_TTS_ALLOWED_CONCURRENCIES} or 'all'."
        )
    return (concurrency,)


def _parse_s2pro_ci_stage(option_value: str) -> str:
    normalized_value = option_value.strip().lower()
    if normalized_value == S2PRO_STAGE_ALL:
        return S2PRO_STAGE_ALL
    if normalized_value not in S2PRO_CI_STAGES:
        raise pytest.UsageError(
            f"Unsupported value for {S2PRO_STAGE_OPTION}: {option_value!r}. "
            f"Use one of {S2PRO_CI_STAGES} or '{S2PRO_STAGE_ALL}'."
        )
    return normalized_value


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    for item in items:
        if item.path.name != "test_s2pro_tts_ci.py":
            continue

        stage_markers = tuple(item.iter_markers(name="s2pro_stage"))
        if len(stage_markers) != 1:
            raise pytest.UsageError(
                "Each test in tests/test_model/test_s2pro_tts_ci.py must have "
                "exactly one s2pro_stage marker."
            )

        stage_ids = tuple(str(arg) for arg in stage_markers[0].args)
        if len(stage_ids) != 1 or stage_ids[0] not in S2PRO_CI_STAGES:
            raise pytest.UsageError(
                "Each s2pro_stage marker in tests/test_model/test_s2pro_tts_ci.py "
                f"must provide exactly one valid stage ID from {S2PRO_CI_STAGES}."
            )

    selected_stage = config.stash.get(SELECTED_S2PRO_CI_STAGE, S2PRO_STAGE_ALL)
    if selected_stage == S2PRO_STAGE_ALL:
        return

    selected_items: list[pytest.Item] = []
    deselected_items: list[pytest.Item] = []
    for item in items:
        if item.path.name != "test_s2pro_tts_ci.py":
            selected_items.append(item)
            continue

        stage_marker = item.get_closest_marker("s2pro_stage")
        assert stage_marker is not None
        stage_ids = tuple(str(arg) for arg in stage_marker.args)
        if selected_stage in stage_ids:
            selected_items.append(item)
        else:
            deselected_items.append(item)

    if deselected_items:
        config.hook.pytest_deselected(items=deselected_items)
        items[:] = selected_items
