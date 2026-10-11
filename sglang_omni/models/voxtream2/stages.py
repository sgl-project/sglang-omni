# SPDX-License-Identifier: Apache-2.0
"""VoXtream2 Stage factory; model imports are deferred to worker startup."""

import sys
from typing import Literal

from sglang_omni.models.voxtream2.remote import RemoteVoXtreamBackend
from sglang_omni.models.voxtream2.scheduler import VoXtreamScheduler


def create_executor(
    *,
    model_path: str,
    gpu_id: int | None,
    reference_audio: str,
    mimi_checkpoint: str,
    speaker_checkpoint: str,
    generator_config_path: str,
    speaking_rate_config_path: str,
    dtype: Literal["float16", "bfloat16", "float32"],
    use_cuda_graph: bool,
    max_pending: int,
    output_buffer_chunks: int,
    model_python: str | None,
    model_source_paths: list[str],
    worker_startup_timeout: float,
    worker_response_timeout: float,
    worker_cancel_timeout: float,
) -> VoXtreamScheduler:
    model_config = dict(
        model_path=model_path,
        mimi_checkpoint=mimi_checkpoint,
        speaker_checkpoint=speaker_checkpoint,
        generator_config_path=generator_config_path,
        speaking_rate_config_path=speaking_rate_config_path,
        device=f"cuda:{gpu_id}" if gpu_id is not None else "cpu",
        dtype=dtype,
        use_cuda_graph=use_cuda_graph,
    )
    backend = RemoteVoXtreamBackend(
        python_executable=sys.executable if model_python is None else model_python,
        model_config={**model_config, "reference_audio": reference_audio},
        source_paths=model_source_paths,
        startup_timeout=worker_startup_timeout,
        response_timeout=worker_response_timeout,
        cancel_timeout=worker_cancel_timeout,
    )
    return VoXtreamScheduler(
        backend,
        max_pending=max_pending,
        output_buffer_chunks=output_buffer_chunks,
    )
