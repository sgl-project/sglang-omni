# SPDX-License-Identifier: Apache-2.0
# Adapted from vLLM-Omni diffusion profiler (Apache 2.0 licensed)
# Original files:
# - https://github.com/vllm-project/vllm-omni/blob/main/vllm_omni/diffusion/profiler/torch_profiler.py

import logging
import os
import subprocess
import threading
from contextlib import nullcontext
from typing import Any, Callable

from .base_profiler import ProfilerBase

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def _resolve_profiler_api(
    device_type: str,
) -> tuple[Callable[..., Any], list[Any], Callable[..., Any] | None]:
    if device_type == "npu":
        import torch_npu

        return (
            torch_npu.profiler.profile,
            [
                torch_npu.profiler.ProfilerActivity.CPU,
                torch_npu.profiler.ProfilerActivity.NPU,
            ],
            torch_npu.profiler.tensorboard_trace_handler,
        )

    from torch.profiler import ProfilerActivity, profile

    return profile, [ProfilerActivity.CPU, ProfilerActivity.CUDA], None


def _make_native_trace_handler(
    handler_factory: Callable[..., Any],
    trace_path_template: str,
    rank: int,
) -> tuple[str, Callable[..., Any]]:
    trace_dir = f"{trace_path_template}_rank{rank}"
    os.makedirs(trace_dir, mode=0o750, exist_ok=True)
    os.chmod(trace_dir, 0o750)
    handler = handler_factory(
        dir_name=trace_dir,
        worker_name=f"rank{rank}",
        analyse_flag=False,
        async_mode=False,
    )
    return trace_dir, handler


class TorchProfiler(ProfilerBase):
    """
    Torch-based profiler configured for End-to-End continuous recording.
    Uses 'on_trace_ready' to handle Trace export.
    Compression is offloaded to a background subprocess to avoid blocking the worker loop.
    """

    _profiler: Any | None = None
    _trace_template: str = ""
    _native_trace_dir: str | None = None

    _active_run_id: str | None = None
    _lock = threading.Lock()

    @classmethod
    def get_active_run_id(cls) -> str | None:
        return cls._active_run_id

    @classmethod
    def start(cls, trace_path_template: str, run_id: str | None = None) -> str:
        """
        Start the profiler with the given trace path template.
        """
        with cls._lock:
            rank = cls._get_rank()

            # 1. Cleanup any existing profiler
            if cls._profiler is not None:
                if run_id is not None and cls._active_run_id == run_id:
                    return cls._native_trace_dir or (
                        f"{cls._trace_template}_rank{rank}.trace.json.gz"
                    )

                logger.warning(
                    "[Rank %s] Torch profiler already active (run_id=%s), restarting for run_id=%s",
                    rank,
                    cls._active_run_id,
                    run_id,
                )
                try:
                    cls._profiler.stop()
                except Exception as e:
                    logger.warning(
                        "[Rank %s] Failed to stop existing profiler: %s", rank, e
                    )
                cls._profiler = None
                cls._active_run_id = None
                cls._trace_template = ""
                cls._native_trace_dir = None

            # 2. Make path absolute
            trace_path_template = os.path.abspath(trace_path_template)
            cls._trace_template = trace_path_template
            cls._active_run_id = run_id

            # Expected paths
            json_file = f"{trace_path_template}_rank{rank}.trace.json"

            os.makedirs(os.path.dirname(json_file), exist_ok=True)

            logger.info(
                "[Rank %s] Starting End-to-End Torch profiler (run_id=%s)", rank, run_id
            )

            # 3. Define the on_trace_ready handler
            def trace_handler(p):
                nonlocal json_file

                # A. Export JSON Trace
                try:
                    p.export_chrome_trace(json_file)
                    logger.info(f"[Rank {rank}] Trace exported to {json_file}")

                    try:
                        subprocess.Popen(["gzip", "-f", json_file])
                        logger.info(
                            f"[Rank {rank}] Triggered background compression for {json_file}"
                        )
                        # Update variable to point to the eventual file
                        json_file = f"{json_file}.gz"
                    except Exception as compress_err:
                        logger.warning(
                            f"[Rank {rank}] Background gzip failed to start: {compress_err}"
                        )

                except Exception as e:
                    logger.warning(f"[Rank {rank}] Failed to export trace: {e}")

            # No ``schedule``: record continuously between start/stop.
            # Expensive flags are env-var opt-in (default off keeps the
            # trace tens of MB; all on can hit multi-GB).
            from sglang_omni.platforms import current_platform

            profile_factory, activities, native_trace_handler = _resolve_profiler_api(
                current_platform.device_type
            )
            on_trace_ready = trace_handler
            if native_trace_handler is not None:
                cls._native_trace_dir, on_trace_ready = _make_native_trace_handler(
                    native_trace_handler,
                    trace_path_template,
                    rank,
                )
            else:
                cls._native_trace_dir = None
            cls._profiler = profile_factory(
                activities=activities,
                on_trace_ready=on_trace_ready,
                record_shapes=os.environ.get("SGLANG_TORCH_PROFILER_RECORD_SHAPES")
                == "1",
                profile_memory=os.environ.get("SGLANG_TORCH_PROFILER_PROFILE_MEMORY")
                == "1",
                with_stack=os.environ.get("SGLANG_TORCH_PROFILER_WITH_STACK") == "1",
                with_flops=os.environ.get("SGLANG_TORCH_PROFILER_WITH_FLOPS") == "1",
            )

            # 5. Start profiling
            cls._profiler.start()

            # Return the expected final path
            return (
                cls._native_trace_dir
                or f"{trace_path_template}_rank{rank}.trace.json.gz"
            )

    @classmethod
    def stop(cls, *, run_id: str | None = None) -> dict | None:
        """
        Stop the profiler.

        If run_id is provided:
          - only stop when active_run_id matches (otherwise ignore)
        """
        with cls._lock:
            if cls._profiler is None:
                return None

            rank = cls._get_rank()
            active = cls._active_run_id

            if run_id is not None and active is not None and active != run_id:
                logger.warning(
                    "[Rank %s] Ignoring profiler stop for run_id=%s because active_run_id=%s",
                    rank,
                    run_id,
                    active,
                )
                return None

            base_path = f"{cls._trace_template}_rank{rank}"
            json_path = f"{base_path}.trace.json"
            gz_path = f"{json_path}.gz"

            profiler = cls._profiler
            native_trace_dir = cls._native_trace_dir
            try:
                profiler.stop()
            except Exception as e:
                logger.warning("[Rank %s] Profiler stop failed: %s", rank, e)

            if native_trace_dir is None:
                # No schedule → on_trace_ready isn't fired on stop, so
                # export here.
                try:
                    os.makedirs(os.path.dirname(json_path), exist_ok=True)
                    profiler.export_chrome_trace(json_path)
                    logger.info("[Rank %s] Trace exported to %s", rank, json_path)
                    try:
                        subprocess.Popen(["gzip", "-f", json_path])
                        logger.info(
                            "[Rank %s] Triggered background compression for %s",
                            rank,
                            json_path,
                        )
                    except Exception as compress_err:
                        logger.warning(
                            "[Rank %s] Background gzip failed: %s",
                            rank,
                            compress_err,
                        )
                except Exception as e:
                    logger.warning("[Rank %s] Failed to export trace: %s", rank, e)

            cls._profiler = None
            cls._active_run_id = None
            cls._trace_template = ""
            cls._native_trace_dir = None

            return {"trace": native_trace_dir or gz_path, "table": None}

    @classmethod
    def step(cls):
        if cls._profiler is not None:
            cls._profiler.step()

    @classmethod
    def is_active(cls) -> bool:
        return cls._profiler is not None

    @classmethod
    def get_step_context(cls):
        return nullcontext()
