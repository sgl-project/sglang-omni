# SPDX-License-Identifier: Apache-2.0
"""Shared fixtures and hooks for test_model tests."""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

pytest_plugins = ["tests.utils"]

if TYPE_CHECKING:
    from typing import Generator

    from tests.utils import ServerHandle

TTS_ALLOWED_CONCURRENCIES = (1, 2, 4, 8, 16)
TTS_STAGE_NONSTREAM = "tts-stage-1-nonstream"
TTS_STAGE_STREAM = "tts-stage-2-stream"
TTS_STAGE_CONSISTENCY = "tts-stage-3-consistency"
TTS_CI_STAGES = (
    TTS_STAGE_NONSTREAM,
    TTS_STAGE_STREAM,
    TTS_STAGE_CONSISTENCY,
)
TTS_FULL_SWEEP_VALUE = "all"
TTS_STAGE_ALL = "all"
TTS_CONCURRENCY_OPTION = "--concurrency"
SELECTED_TTS_CONCURRENCIES = pytest.StashKey[tuple[int, ...]]()
TTS_STAGE_OPTION = "--tts-stage"
SELECTED_TTS_CI_STAGE = pytest.StashKey[str]()
TTS_CI_MODEL_OPTION = "--tts-ci-model"
QWEN3_OMNI_MODEL_PATH = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
# Single source of truth for the model path used by Qwen3-Omni vision-encoder
# benchmarks and the SGLang state they bring up. Honors
# ``SGLANG_OMNI_TEST_QWEN3_MODEL=/local/path`` so an offline runner does not
# fall back to the HF hub name in ``ServerArgs.model_path``.
QWEN3_OMNI_TEST_MODEL_PATH = os.environ.get(
    "SGLANG_OMNI_TEST_QWEN3_MODEL", QWEN3_OMNI_MODEL_PATH
)
QWEN3_OMNI_FP8_MODEL_PATH = "marksverdhei/Qwen3-Omni-30B-A3B-FP8"
QWEN3_OMNI_FP8_TEST_MODEL_PATH = os.environ.get(
    "SGLANG_OMNI_TEST_QWEN3_FP8_MODEL", QWEN3_OMNI_FP8_MODEL_PATH
)
QWEN3_OMNI_MODEL_NAME = "qwen3-omni"
QWEN3_OMNI_ROUTER_WAIT_TIMEOUT = 180
QWEN3_OMNI_COLOCATED_WORKER_ARGS = (
    "--config examples/configs/qwen3_omni_colocated_h20.yaml --colocate"
)
QWEN3_OMNI_MMMU_WORKER_ARGS = (
    "--config examples/configs/qwen3_omni_mmmu.yaml --text-only"
)
QWEN3_OMNI_MMSU_WORKER_ARGS = (
    "--config examples/configs/qwen3_omni_mmsu.yaml --text-only"
)
QWEN3_OMNI_VIDEO_WORKER_ARGS = (
    f"{QWEN3_OMNI_COLOCATED_WORKER_ARGS} "
    "--stages.0.factory-args.thinker-max-seq-len 32768 "
    "--stages.4.factory-args.thinker-max-seq-len 32768"
)


@pytest.fixture(scope="module")
def qwen3_omni_router_server(tmp_path_factory: pytest.TempPathFactory):
    """Start two colocated Qwen3-Omni workers behind the router."""
    with _launch_qwen3_omni_router(
        tmp_path_factory,
        worker_extra_args=QWEN3_OMNI_COLOCATED_WORKER_ARGS,
    ) as router:
        yield router


@pytest.fixture(scope="module")
def qwen3_omni_mmmu_server(tmp_path_factory: pytest.TempPathFactory):
    """Router-backed Qwen3-Omni endpoint tuned for MMMU text-output CI."""
    with _launch_qwen3_omni_router(
        tmp_path_factory,
        worker_extra_args=QWEN3_OMNI_MMMU_WORKER_ARGS,
    ) as router:
        yield router


@pytest.fixture(scope="module")
def qwen3_omni_mmsu_server(tmp_path_factory: pytest.TempPathFactory):
    """Router-backed Qwen3-Omni endpoint tuned for MMSU text-output CI."""
    with _launch_qwen3_omni_router(
        tmp_path_factory,
        worker_extra_args=QWEN3_OMNI_MMSU_WORKER_ARGS,
    ) as router:
        yield router


@pytest.fixture(scope="module")
def qwen3_omni_thinker_server(tmp_path_factory: pytest.TempPathFactory):
    """Router-backed Qwen3-Omni endpoint used by text-output benchmarks."""
    with _launch_qwen3_omni_router(
        tmp_path_factory,
        worker_extra_args=QWEN3_OMNI_VIDEO_WORKER_ARGS,
    ) as router:
        yield router


@pytest.fixture(scope="module")
def qwen3_omni_talker_server(tmp_path_factory: pytest.TempPathFactory):
    """Router-backed Qwen3-Omni endpoint used by audio-output benchmarks."""
    with _launch_qwen3_omni_router(
        tmp_path_factory,
        worker_extra_args=QWEN3_OMNI_VIDEO_WORKER_ARGS,
    ) as router:
        yield router


@pytest.fixture(scope="module")
def qwen3_omni_talker_server_tp2(tmp_path_factory: pytest.TempPathFactory):
    """Start Qwen3-Omni TP=2 thinker + disaggregated talker on the same two-card host."""
    yield from _start_qwen3_omni_speech_server(
        tmp_path_factory,
        extra_args=[
            "--thinker-tp-size",
            "2",
            "--gpu-thinker-tp",
            "0,1",
            "--gpu-talker",
            "1",
            "--gpu-code2wav",
            "1",
            "--thinker-mem-fraction-static",
            "0.55",
            "--talker-mem-fraction-static",
            "0.20",
        ],
        timeout=450,
        log_prefix="server_logs_tp2",
        force_log=True,
    )


@pytest.fixture(scope="module")
def qwen3_omni_fp8_talker_server_tp2(tmp_path_factory: pytest.TempPathFactory):
    """Start Qwen3-Omni FP8 with TP=2 thinker and TP=1 talker."""
    yield from _start_qwen3_omni_speech_server(
        tmp_path_factory,
        model_path=QWEN3_OMNI_FP8_TEST_MODEL_PATH,
        extra_args=[
            "--thinker-tp-size",
            "2",
            "--gpu-thinker-tp",
            "0,1",
            "--gpu-talker",
            "1",
            "--gpu-code2wav",
            "1",
            # note (Yue Yin): 0.40 not 0.55 — GPU 1 co-locates thinker rank-1, talker,
            # code2wav under TP=2; 0.55 OOMs on concurrency-16 video-prefill (issue #765).
            "--thinker-mem-fraction-static",
            "0.40",
            "--talker-mem-fraction-static",
            "0.20",
        ],
        timeout=450,
        log_prefix="server_logs_fp8_tp2",
        force_log=True,
    )


def _launch_qwen3_omni_router(
    tmp_path_factory: pytest.TempPathFactory,
    *,
    model_path: str = QWEN3_OMNI_TEST_MODEL_PATH,
    worker_extra_args: str,
):
    from tests.test_model.omni_router_utils import launch_managed_router

    return launch_managed_router(
        tmp_path_factory=tmp_path_factory,
        model_path=model_path,
        model_name=QWEN3_OMNI_MODEL_NAME,
        worker_extra_args=worker_extra_args,
        wait_timeout=QWEN3_OMNI_ROUTER_WAIT_TIMEOUT,
    )


def _start_qwen3_omni_speech_server(
    tmp_path_factory: pytest.TempPathFactory,
    *,
    model_path: str = QWEN3_OMNI_TEST_MODEL_PATH,
    extra_args: list[str],
    timeout: int,
    log_prefix: str,
    force_log: bool,
) -> Generator[ServerHandle, None, None]:
    """Shared bring-up for run_qwen3_omni_speech_server.py-based fixtures."""
    import sys

    from sglang_omni.utils import find_available_port
    from tests.utils import (
        ServerHandle,
        server_log_file,
        start_server_from_cmd,
        stop_server,
    )

    port = find_available_port()
    log_file = (
        tmp_path_factory.mktemp(log_prefix) / "server.log"
        if force_log
        else server_log_file(tmp_path_factory, prefix=log_prefix)
    )
    cmd = [
        sys.executable,
        "examples/run_qwen3_omni_speech_server.py",
        "--model-path",
        model_path,
        "--port",
        str(port),
        "--model-name",
        "qwen3-omni",
        "--thinker-max-seq-len",
        "32768",
        *extra_args,
    ]
    proc = start_server_from_cmd(cmd, log_file, port, timeout=timeout, tee=force_log)
    try:
        yield ServerHandle(proc=proc, port=port, log_file=log_file)
    finally:
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
        TTS_CONCURRENCY_OPTION,
        action="store",
        default="16",
        help=(
            "Select the TTS benchmark concurrency. "
            "Use one of {1,2,4,8,16} or 'all' for the full sweep."
        ),
    )
    parser.addoption(
        TTS_STAGE_OPTION,
        action="store",
        default=TTS_STAGE_ALL,
        help=(
            f"Select the TTS CI stage. Use one of {TTS_CI_STAGES} or '{TTS_STAGE_ALL}'."
        ),
    )
    parser.addoption(
        TTS_CI_MODEL_OPTION,
        action="store",
        default="",
        help=(
            "Select the TTS CI model preset. "
            "Use one of the presets in tests/test_model/tts_ci_config.py. "
            "If omitted, use TTS_CI_MODEL from the environment."
        ),
    )


def pytest_configure(config: pytest.Config) -> None:
    option_value = config.getoption(TTS_CONCURRENCY_OPTION)
    config.stash[SELECTED_TTS_CONCURRENCIES] = _parse_tts_concurrency(option_value)
    stage_value = config.getoption(TTS_STAGE_OPTION)
    config.stash[SELECTED_TTS_CI_STAGE] = _parse_tts_ci_stage(stage_value)
    model_value = config.getoption(TTS_CI_MODEL_OPTION)
    if model_value:
        os.environ["TTS_CI_MODEL"] = _parse_tts_ci_model(model_value)


@pytest.fixture(scope="session")
def selected_tts_concurrencies(
    pytestconfig: pytest.Config,
) -> tuple[int, ...]:
    return pytestconfig.stash[SELECTED_TTS_CONCURRENCIES]


@pytest.fixture(scope="session")
def selected_tts_ci_stage(pytestconfig: pytest.Config) -> str:
    return pytestconfig.stash[SELECTED_TTS_CI_STAGE]


def _parse_tts_concurrency(option_value: str) -> tuple[int, ...]:
    normalized_value = option_value.strip().lower()
    if normalized_value == TTS_FULL_SWEEP_VALUE:
        return TTS_ALLOWED_CONCURRENCIES

    try:
        concurrency = int(normalized_value)
    except ValueError as exc:
        raise pytest.UsageError(
            "Invalid value for --concurrency. Use one of {1,2,4,8,16} or 'all'."
        ) from exc

    if concurrency not in TTS_ALLOWED_CONCURRENCIES:
        raise pytest.UsageError(
            f"Unsupported concurrency {concurrency}. "
            f"Use one of {TTS_ALLOWED_CONCURRENCIES} or 'all'."
        )
    return (concurrency,)


def _parse_tts_ci_stage(option_value: str) -> str:
    normalized_value = option_value.strip().lower()
    if normalized_value == TTS_STAGE_ALL:
        return TTS_STAGE_ALL
    if normalized_value not in TTS_CI_STAGES:
        raise pytest.UsageError(
            f"Unsupported value for {TTS_STAGE_OPTION}: {option_value!r}. "
            f"Use one of {TTS_CI_STAGES} or '{TTS_STAGE_ALL}'."
        )
    return normalized_value


def _parse_tts_ci_model(option_value: str) -> str:
    from tests.test_model.tts_ci_config import TTS_CI_PRESETS

    normalized_value = option_value.strip().lower()
    if normalized_value not in TTS_CI_PRESETS:
        allowed = tuple(sorted(TTS_CI_PRESETS))
        raise pytest.UsageError(
            f"Unsupported value for {TTS_CI_MODEL_OPTION}: {option_value!r}. "
            f"Use one of {allowed}."
        )
    return normalized_value


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    for item in items:
        if item.path.name != "test_tts_ci.py":
            continue

        stage_markers = tuple(item.iter_markers(name="tts_stage"))
        if len(stage_markers) != 1:
            raise pytest.UsageError(
                "Each test in tests/test_model/test_tts_ci.py must have "
                "exactly one tts_stage marker."
            )

        stage_ids = tuple(str(arg) for arg in stage_markers[0].args)
        if len(stage_ids) != 1 or stage_ids[0] not in TTS_CI_STAGES:
            raise pytest.UsageError(
                "Each tts_stage marker in tests/test_model/test_tts_ci.py "
                f"must provide exactly one valid stage ID from {TTS_CI_STAGES}."
            )

    selected_stage = config.stash.get(SELECTED_TTS_CI_STAGE, TTS_STAGE_ALL)
    if selected_stage == TTS_STAGE_ALL:
        return

    selected_items: list[pytest.Item] = []
    deselected_items: list[pytest.Item] = []
    for item in items:
        if item.path.name != "test_tts_ci.py":
            selected_items.append(item)
            continue

        stage_marker = item.get_closest_marker("tts_stage")
        assert stage_marker is not None
        stage_ids = tuple(str(arg) for arg in stage_marker.args)
        if selected_stage in stage_ids:
            selected_items.append(item)
        else:
            deselected_items.append(item)

    if deselected_items:
        config.hook.pytest_deselected(items=deselected_items)
        items[:] = selected_items
