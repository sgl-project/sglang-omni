# SPDX-License-Identifier: Apache-2.0
# Adapted from vLLM-Omni diffusion profiler (Apache 2.0 licensed)
# Original files:
# - https://github.com/vllm-project/vllm-omni/blob/main/vllm_omni/diffusion/profiler/torch_profiler.py

import logging
import os
import subprocess
import threading
from contextlib import nullcontext

import torch
from torch.profiler import ProfilerActivity, profile

from .base_profiler import ProfilerBase

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def _gpu_activity() -> ProfilerActivity | None:
    """The accelerator activity for the live backend, or None on a CPU build.

    ``ProfilerActivity.CUDA`` exists as an enum member on a ``+xpu`` torch build,
    so passing it unconditionally raises nothing and silently records a CPU-only
    trace -- every XPU kernel missing, with no error to explain it. Select by what
    the process can actually see instead. Mirrors the activity mapping in SGLang's
    ``SchedulerProfilerManager``.
    """
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return getattr(ProfilerActivity, "XPU", None)
    if torch.cuda.is_available():
        # CUDA activity covers NVIDIA and AMD (ROCm) builds alike.
        return getattr(ProfilerActivity, "CUDA", None)
    return None


class TorchProfiler(ProfilerBase):
    """
    Torch-based profiler configured for End-to-End continuous recording.
    Uses 'on_trace_ready' to handle Trace export.
    Compression is offloaded to a background subprocess to avoid blocking the worker loop.
    """

    _profiler: profile | None = None
    _trace_template: str = ""

    _active_run_id: str | None = None
    _lock = threading.Lock()
    _et_observer = None
    _trace_exported: bool = False

    @classmethod
    def get_active_run_id(cls) -> str | None:
        return cls._active_run_id

    @classmethod
    def _make_et_observer(cls, et_file: str, rank: int):
        """Register an ExecutionTraceObserver for ``et_file``, or None."""
        observer_cls = getattr(torch.profiler, "ExecutionTraceObserver", None)
        if observer_cls is None:
            logger.warning(
                "[Rank %s] SGLANG_TORCH_PROFILER_RECORD_ET=1 but this torch build "
                "has no torch.profiler.ExecutionTraceObserver; no .et.json written",
                rank,
            )
            return None
        try:
            os.makedirs(os.path.dirname(et_file), exist_ok=True)
            observer = observer_cls()
            observer.register_callback(et_file)
            return observer
        except Exception as e:
            logger.warning(
                "[Rank %s] Failed to register execution trace observer: %s", rank, e
            )
            return None

    @classmethod
    def _unregister_et_observer(cls, rank: int) -> None:
        """Close the ET file. Skipping this leaves a truncated .et.json."""
        if cls._et_observer is None:
            return
        try:
            cls._et_observer.unregister_callback()
        except Exception as e:
            logger.warning(
                "[Rank %s] Failed to unregister execution trace observer: %s", rank, e
            )
        finally:
            cls._et_observer = None

    @classmethod
    def start(cls, trace_path_template: str, run_id: str | None = None) -> str:
        """
        Start the profiler with the given trace path template.
        """
        with cls._lock:
            # Resolve the rank before the cleanup branch below: both the
            # already-active early return and its warning interpolate it, so
            # reading it afterwards made either path raise UnboundLocalError.
            rank = cls._get_rank()

            # 1. Cleanup any existing profiler
            if cls._profiler is not None:
                if run_id is not None and cls._active_run_id == run_id:
                    return f"{cls._trace_template}_rank{rank}.trace.json.gz"

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
                cls._unregister_et_observer(rank)
                cls._profiler = None
                cls._active_run_id = None
                cls._trace_template = ""

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
            cls._trace_exported = False

            def trace_handler(p):
                nonlocal json_file

                # A. Export JSON Trace
                try:
                    p.export_chrome_trace(json_file)
                    cls._trace_exported = True
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

            # Chakra Execution Trace (ET): the graph-level artifact PARAM replay
            # and MTSA consume, written alongside the chrome trace as
            # *_rank<N>.et.json. Opt-in like the other expensive flags, and armed
            # before profile() so the observer can be handed to it.
            et_kwargs = {}
            if os.environ.get("SGLANG_TORCH_PROFILER_RECORD_ET") == "1":
                et_file = f"{trace_path_template}_rank{rank}.et.json"
                observer = cls._make_et_observer(et_file, rank)
                if observer is not None:
                    cls._et_observer = observer
                    et_kwargs["execution_trace_observer"] = observer
                    logger.info(
                        "[Rank %s] Execution trace will be written to %s",
                        rank,
                        et_file,
                    )

            # No ``schedule``: record continuously between start/stop.
            # Expensive flags are env-var opt-in (default off keeps the
            # trace tens of MB; all on can hit multi-GB).
            activities = [ProfilerActivity.CPU]
            gpu_activity = _gpu_activity()
            if gpu_activity is not None:
                activities.append(gpu_activity)
            else:
                logger.warning(
                    "[Rank %s] No accelerator activity available; "
                    "recording a CPU-only trace",
                    rank,
                )
            cls._profiler = profile(
                activities=activities,
                on_trace_ready=trace_handler,
                record_shapes=os.environ.get("SGLANG_TORCH_PROFILER_RECORD_SHAPES")
                == "1",
                profile_memory=os.environ.get("SGLANG_TORCH_PROFILER_PROFILE_MEMORY")
                == "1",
                with_stack=os.environ.get("SGLANG_TORCH_PROFILER_WITH_STACK") == "1",
                with_flops=os.environ.get("SGLANG_TORCH_PROFILER_WITH_FLOPS") == "1",
                **et_kwargs,
            )

            # 5. Start profiling
            cls._profiler.start()

            # Return the expected final path
            return f"{trace_path_template}_rank{rank}.trace.json.gz"

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
            et_path = f"{base_path}.et.json" if cls._et_observer is not None else None
            try:
                profiler.stop()
            except Exception as e:
                logger.warning("[Rank %s] Profiler stop failed: %s", rank, e)

            # Unregister before export: the observer must be closed for the
            # .et.json to be complete, and it must happen even if stop() raised.
            cls._unregister_et_observer(rank)
            if et_path is not None:
                logger.info("[Rank %s] Execution trace written to %s", rank, et_path)

            # Without a ``schedule`` some torch versions fire on_trace_ready on
            # stop() and some do not, so export only if the handler did not
            # already. Exporting twice raises "Trace is already saved", which the
            # old unconditional path logged as a failed export on every stop.
            if cls._trace_exported:
                cls._profiler = None
                cls._active_run_id = None
                cls._trace_template = ""
                cls._trace_exported = False
                return {"trace": gz_path, "table": None, "execution_trace": et_path}

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

            return {"trace": gz_path, "table": None, "execution_trace": et_path}

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
