# SPDX-License-Identifier: Apache-2.0
"""VoXtream2 Stage factory; model imports are deferred to worker startup."""

from sglang_omni.models.voxtream2.backend import VoXtreamBackend
from sglang_omni.models.voxtream2.loader import load_generator
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
    dtype: str = "float16",
    use_cuda_graph: bool = True,
    max_pending: int = 32,
    output_buffer_chunks: int = 64,
    model_python: str | None = None,
    model_source_paths: list[str] | None = None,
    worker_startup_timeout: float = 180.0,
    worker_response_timeout: float = 120.0,
    worker_cancel_timeout: float = 10.0,
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
    if model_python is not None:
        backend = RemoteVoXtreamBackend(
            python_executable=model_python,
            model_config={**model_config, "reference_audio": reference_audio},
            source_paths=model_source_paths or (),
            startup_timeout=worker_startup_timeout,
            response_timeout=worker_response_timeout,
            cancel_timeout=worker_cancel_timeout,
        )
    else:
        backend = VoXtreamBackend(load_generator(**model_config), reference_audio)
    return VoXtreamScheduler(
        backend,
        max_pending=max_pending,
        output_buffer_chunks=output_buffer_chunks,
    )
