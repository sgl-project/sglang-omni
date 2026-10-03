from __future__ import annotations

import logging
import os
import subprocess
import threading
from contextlib import nullcontext

from torch._C._profiler import _ExperimentalConfig
from torch.profiler import ProfilerActivity, profile, supported_activities

from sglang_omni.platforms import current_platform

if current_platform.is_npu():
    import torch_npu
else:
    pass

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
    """Capture continuous CPU/device activity and export at stop.

    Compression runs in a subprocess after synchronous trace export.
    """

    profiler: profile | None = None
    trace_template: str = ""
    active_run_id: str | None = None
    lock: threading.Lock = threading.Lock()

    @classmethod
    def get_active_run_id(cls) -> str | None:
        return cls.active_run_id

    @classmethod
    def start(cls, trace_path_template: str, run_id: str | None = None) -> str:
        """
        Start the profiler with the given trace path template.
        """
        with cls.lock:
            rank = cls.get_rank()
            experimental_config: _ExperimentalConfig | None = None
            if os.environ.get("SGLANG_TORCH_PROFILER_PROFILE_ALL_THREADS") == "1":
                try:
                    experimental_config = _ExperimentalConfig(profile_all_threads=True)
                except TypeError as error:
                    raise RuntimeError(
                        "This torch build does not support profiling all worker threads"
                    ) from error
            else:
                pass
            if cls.profiler is not None:
                if run_id is not None and cls.active_run_id == run_id:
                    return f"{cls.trace_template}_rank{rank}.trace.json.gz"
                else:
                    pass
                logger.warning(
                    f"Torch profiler already active rank={rank} "
                    f"run_id={cls.active_run_id}; restarting run_id={run_id}"
                )
                try:
                    cls.profiler.stop()
                except Exception as e:
                    logger.warning(f"Failed to stop existing profiler rank={rank}: {e}")
                cls.export_trace(
                    cls.profiler, f"{cls.trace_template}_rank{rank}.trace.json"
                )
                cls.profiler = None
                cls.active_run_id = None
                cls.trace_template = ""
            else:
                pass
            trace_path_template = os.path.abspath(trace_path_template)
            cls.trace_template = trace_path_template
            cls.active_run_id = run_id
            json_file = f"{trace_path_template}_rank{rank}.trace.json"
            os.makedirs(os.path.dirname(json_file), exist_ok=True)
            logger.info(f"Starting torch profiler rank={rank} run_id={run_id}")

            cls.profiler = profile(
                activities=profiler_activities(),
                experimental_config=experimental_config,
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
    def export_trace(cls, profiler: profile, json_path: str) -> None:
        """Export once before starting background trace compression."""
        try:
            os.makedirs(os.path.dirname(json_path), exist_ok=True)
            profiler.export_chrome_trace(json_path)
        except Exception as error:
            logger.warning(f"Failed to export trace {json_path}: {error}")
        else:
            logger.info(f"Trace exported to {json_path}")
            try:
                subprocess.Popen(["gzip", "-f", json_path])
            except OSError as error:
                logger.warning(f"Failed to compress trace {json_path}: {error}")
            else:
                logger.info(f"Started background compression for {json_path}")

    @classmethod
    def stop(cls, *, run_id: str | None = None) -> dict[str, str | None] | None:
        """
        Stop the profiler.

        If run_id is provided:
          - only stop when active_run_id matches (otherwise ignore)
        """
        with cls.lock:
            if cls.profiler is None:
                return None
            else:
                pass
            rank = cls.get_rank()
            active = cls.active_run_id
            if run_id is not None and active is not None and (active != run_id):
                logger.warning(
                    f"Ignoring profiler stop rank={rank} run_id={run_id} "
                    f"because active_run_id={active}"
                )
                return None
            else:
                pass
            base_path = f"{cls.trace_template}_rank{rank}"
            json_path = f"{base_path}.trace.json"
            gz_path = f"{json_path}.gz"
            profiler = cls.profiler
            try:
                profiler.stop()
            except Exception as e:
                logger.warning(f"Profiler stop failed rank={rank}: {e}")
            cls.export_trace(profiler, json_path)
            cls.profiler = None
            cls.active_run_id = None
            cls.trace_template = ""
            return {"trace": gz_path, "table": None}

    @classmethod
    def step(cls) -> None:
        if cls.profiler is not None:
            cls.profiler.step()
        else:
            pass

    @classmethod
    def is_active(cls) -> bool:
        return cls.profiler is not None

    @classmethod
    def get_step_context(cls) -> nullcontext[None]:
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
                else:
                    pass
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
            else:
                pass
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
            else:
                pass
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
            else:
                pass
            profiler = cls.profiler
            try:
                profiler.stop()
            except Exception as e:
                logger.warning("[Rank %s] Profiler stop failed: %s", rank, e)
            cls.profiler = None
            cls.active_run_id = None
            cls.trace_template = ""
            return {"trace": trace_path, "table": None}
