from __future__ import annotations

import logging
import os
import subprocess
import threading
from contextlib import nullcontext

from torch.profiler import ProfilerActivity, profile, supported_activities

from sglang_omni.platforms import current_platform

if current_platform.is_npu():
    import torch_npu

from .base_profiler import ProfilerBase

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)


def profiler_activities() -> list[ProfilerActivity]:
    """CPU plus whichever device activity this torch build supports."""
    device = sorted(
        (a for a in supported_activities() if a != ProfilerActivity.CPU),
        key=lambda a: a.name,
    )
    return [ProfilerActivity.CPU, *device]


class TorchProfiler(ProfilerBase):
    """
    Torch-based profiler configured for End-to-End continuous recording.
    Uses 'on_trace_ready' to handle Trace export.
    Compression is offloaded to a background subprocess to avoid blocking the worker loop.
    """

    profiler: profile | None = None
    trace_template: str = ""
    active_run_id: str | None = None
    lock = threading.Lock()

    @classmethod
    def get_active_run_id(cls) -> str | None:
        return cls.active_run_id

    @classmethod
    def start(cls, trace_path_template: str, run_id: str | None = None) -> str:
        """
        Start the profiler with the given trace path template.
        """
        with cls.lock:
            if cls.profiler is not None:
                if run_id is not None and cls.active_run_id == run_id:
                    return f"{cls.trace_template}_rank{rank}.trace.json.gz"
                logger.warning(
                    "[Rank %s] Torch profiler already active (run_id=%s), restarting for run_id=%s",
                    rank,
                    cls.active_run_id,
                    run_id,
                )
                try:
                    cls.profiler.stop()
                except Exception as e:
                    logger.warning(
                        "[Rank %s] Failed to stop existing profiler: %s", rank, e
                    )
                cls.profiler = None
                cls.active_run_id = None
                cls.trace_template = ""
            rank = cls.get_rank()
            trace_path_template = os.path.abspath(trace_path_template)
            cls.trace_template = trace_path_template
            cls.active_run_id = run_id
            json_file = f"{trace_path_template}_rank{rank}.trace.json"
            os.makedirs(os.path.dirname(json_file), exist_ok=True)
            logger.info(
                "[Rank %s] Starting End-to-End Torch profiler (run_id=%s)", rank, run_id
            )

            def trace_handler(p):
                nonlocal json_file
                try:
                    p.export_chrome_trace(json_file)
                    logger.info(f"[Rank {rank}] Trace exported to {json_file}")
                    try:
                        subprocess.Popen(["gzip", "-f", json_file])
                        logger.info(
                            f"[Rank {rank}] Triggered background compression for {json_file}"
                        )
                        json_file = f"{json_file}.gz"
                    except Exception as compress_err:
                        logger.warning(
                            f"[Rank {rank}] Background gzip failed to start: {compress_err}"
                        )
                except Exception as e:
                    logger.warning(f"[Rank {rank}] Failed to export trace: {e}")

            cls.profiler = profile(
                activities=profiler_activities(),
                on_trace_ready=trace_handler,
                record_shapes=os.environ.get("SGLANG_TORCH_PROFILER_RECORD_SHAPES")
                == "1",
                profile_memory=os.environ.get("SGLANG_TORCH_PROFILER_PROFILE_MEMORY")
                == "1",
                with_stack=os.environ.get("SGLANG_TORCH_PROFILER_WITH_STACK") == "1",
                with_flops=os.environ.get("SGLANG_TORCH_PROFILER_WITH_FLOPS") == "1",
            )
            cls.profiler.start()
            return f"{trace_path_template}_rank{rank}.trace.json.gz"

    @classmethod
    def stop(cls, *, run_id: str | None = None) -> dict | None:
        """
        Stop the profiler.

        If run_id is provided:
          - only stop when active_run_id matches (otherwise ignore)
        """
        with cls.lock:
            if cls.profiler is None:
                return None
            rank = cls.get_rank()
            active = cls.active_run_id
            if run_id is not None and active is not None and (active != run_id):
                logger.warning(
                    "[Rank %s] Ignoring profiler stop for run_id=%s because active_run_id=%s",
                    rank,
                    run_id,
                    active,
                )
                return None
            base_path = f"{cls.trace_template}_rank{rank}"
            json_path = f"{base_path}.trace.json"
            gz_path = f"{json_path}.gz"
            profiler = cls.profiler
            try:
                profiler.stop()
            except Exception as e:
                logger.warning("[Rank %s] Profiler stop failed: %s", rank, e)
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
                        "[Rank %s] Background gzip failed: %s", rank, compress_err
                    )
            except Exception as e:
                logger.warning("[Rank %s] Failed to export trace: %s", rank, e)
            cls.profiler = None
            cls.active_run_id = None
            cls.trace_template = ""
            return {"trace": gz_path, "table": None}

    @classmethod
    def step(cls):
        if cls.profiler is not None:
            cls.profiler.step()

    @classmethod
    def is_active(cls) -> bool:
        return cls.profiler is not None

    @classmethod
    def get_step_context(cls):
        return nullcontext()


class TorchNPUProfiler(TorchProfiler):

    @classmethod
    def start(cls, trace_path_template: str, run_id: str | None = None) -> str:
        with cls.lock:
            trace_path_template = os.path.abspath(trace_path_template)
            rank = cls.get_rank()
            if cls.profiler is not None:
                if run_id is not None and cls.active_run_id == run_id:
                    return trace_path_template
                rank = cls.get_rank()
                logger.warning(
                    "[Rank %s] Torch profiler already active (run_id=%s), restarting for run_id=%s",
                    rank,
                    cls.active_run_id,
                    run_id,
                )
                try:
                    cls.profiler.stop()
                except Exception as e:
                    logger.warning(
                        "[Rank %s] Failed to stop existing profiler: %s", rank, e
                    )
                cls.profiler = None
                cls.active_run_id = None
                cls.trace_template = ""
            cls.active_run_id = run_id
            cls.trace_template = trace_path_template
            os.makedirs(trace_path_template, exist_ok=True)
            logger.info(
                "[Rank %s] Starting End-to-End Torch profiler (run_id=%s)", rank, run_id
            )
            cls.profiler = torch_npu.profiler.profile(
                activities=[
                    torch_npu.profiler.ProfilerActivity.CPU,
                    torch_npu.profiler.ProfilerActivity.NPU,
                ],
                on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(
                    trace_path_template
                ),
                record_shapes=os.environ.get("SGLANG_TORCH_PROFILER_RECORD_SHAPES")
                == "1",
                profile_memory=os.environ.get("SGLANG_TORCH_PROFILER_PROFILE_MEMORY")
                == "1",
                with_stack=os.environ.get("SGLANG_TORCH_PROFILER_WITH_STACK") == "1",
                with_flops=os.environ.get("SGLANG_TORCH_PROFILER_WITH_FLOPS") == "1",
            )
            cls.profiler.start()
            return trace_path_template

    @classmethod
    def stop(cls, *, run_id: str | None = None) -> dict | None:
        with cls.lock:
            if cls.profiler is None:
                return None
            rank = cls.get_rank()
            active = cls.active_run_id
            trace_path = cls.trace_template
            if run_id is not None and active is not None and (active != run_id):
                logger.warning(
                    "[Rank %s] Ignoring profiler stop for run_id=%s because active_run_id=%s",
                    rank,
                    run_id,
                    active,
                )
                return None
            profiler = cls.profiler
            try:
                profiler.stop()
            except Exception as e:
                logger.warning("[Rank %s] Profiler stop failed: %s", rank, e)
            cls.profiler = None
            cls.active_run_id = None
            cls.trace_template = ""
            return {"trace": trace_path, "table": None}
