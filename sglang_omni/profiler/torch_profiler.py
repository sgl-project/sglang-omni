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

    @classmethod
    def get_active_run_id(cls) -> str | None:
        return cls._active_run_id

    @classmethod
    def start(cls, trace_path_template: str, run_id: str | None = None) -> str:
        """
        Start the profiler with the given trace path template.
        """
        with cls._lock:

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
                cls._profiler = None
                cls._active_run_id = None
                cls._trace_template = ""

            rank = cls._get_rank()

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

            # 4. Initialize profiler with long active period
            cls._profiler = profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                schedule=torch.profiler.schedule(
                    wait=0,
                    warmup=0,
                    active=100000,  # long capture window
                ),
                on_trace_ready=trace_handler,
                record_shapes=True,
                profile_memory=True,
                with_stack=True,
                with_flops=True,
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
            gz_path = f"{base_path}.trace.json.gz"

            try:
                cls._profiler.stop()
            except Exception as e:
                logger.warning("[Rank %s] Profiler stop failed: %s", rank, e)

            cls._profiler = None
            cls._active_run_id = None
            cls._trace_template = ""

            return {"trace": gz_path, "table": None}

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
