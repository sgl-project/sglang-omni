# SPDX-License-Identifier: Apache-2.0
# Adapted from vLLM-Omni diffusion profiler (Apache 2.0 licensed)
# Original files:
# - https://github.com/vllm-project/vllm-omni/blob/main/vllm_omni/diffusion/profiler/base.py

import logging
from abc import ABC, abstractmethod

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


class ProfilerBase(ABC):
    """
    Abstract base class for all diffusion profilers.
    """

    @abstractmethod
    def start(self, trace_path_template: str) -> str:
        """
        Start profiling.

        Args:
            trace_path_template: Base path (without rank or extension).
                                 e.g. "/tmp/profiles/omni_run"

        Returns:
            Full path of the trace file this rank will write.
        """

    @abstractmethod
    def stop(self) -> str | None:
        """
        Stop profiling and finalize/output the trace.

        Returns:
            Path to the saved trace file, or None if not active.
        """

    @abstractmethod
    def get_step_context(self):
        """
        Returns a context manager that advances one profiling step.
        Should be a no-op (nullcontext) when profiler is not active.
        """

    @abstractmethod
    def is_active(self) -> bool:
        """Return True if profiling is currently running."""

    @classmethod
    def _get_rank(cls) -> int:
        import os

        return int(os.getenv("RANK", "0"))
