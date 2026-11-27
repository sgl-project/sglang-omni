# SPDX-License-Identifier: Apache-2.0
"""Full MOSS decode through the production FlashInfer CUDA Graph runner."""

import multiprocessing
import os
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
from safetensors.torch import save_file
from sglang.srt.distributed.parallel_state import (
    destroy_distributed_environment,
    destroy_model_parallel,
)
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
)

from sglang_omni.models.moss_speech.fsm import (
    AUDIO_FORBIDDEN_FROM,
    AUDIO_PAD,
    AUDIO_VOCAB,
    MODALITY_PAD,
    MODE_AUDIO,
    MODE_TEXT,
    TEXT_VOCAB,
)
from sglang_omni.models.moss_speech.hf_config import MossSpeechConfig
from sglang_omni.models.moss_speech.request_builders import MossSpeechSGLangRequestData
from sglang_omni.models.moss_speech.stages import create_ar_engine_executor
from sglang_omni.scheduling.types import SchedulerRequest
from tests.unit_test.fixtures.accelerator import require_cuda
from tests.unit_test.moss_speech.model_helpers import checkpoint_weights


@pytest.mark.accelerator
def test_native_decode_graph_updates_kv_in_current_request_order(
    tmp_path: Path,
) -> None:
    require_cuda()
    if os.environ.get("SGLANG_USE_CPU_ENGINE") == "1":
        pytest.skip("Run the accelerator lane without the CPU engine override")
    process = multiprocessing.get_context("spawn").Process(
        target=run_native_decode_graph, args=(tmp_path,)
    )
    process.start()
    try:
        process.join(timeout=180)
        assert process.exitcode == 0
    finally:
        if process.is_alive():
            process.kill()
            process.join()
        process.close()


def run_native_decode_graph(tmp_path: Path) -> None:
    """The production AR engine requires an exclusive process runtime."""
    configuration = MossSpeechConfig(
        vocab_size=TEXT_VOCAB,
        audio_vocab_size=AUDIO_VOCAB,
        modality_pad_token_id=MODALITY_PAD,
        hidden_size=128,
        intermediate_size=256,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=64,
        num_shared_layers=2,
        num_modality_layers=1,
        num_hidden_layers=3,
        architectures=["MossSpeechForCausalLM"],
    )
    configuration.save_pretrained(tmp_path)
    generator = torch.Generator().manual_seed(184)
    weights = dict(checkpoint_weights(configuration))
    for name, tensor in weights.items():
        if "norm.weight" in name:
            tensor.fill_(1.0)
        else:
            tensor.normal_(std=0.02, generator=generator)
    save_file(weights, str(tmp_path / "model.safetensors"))
    scheduler = create_ar_engine_executor(
        str(tmp_path),
        device="cuda",
        gpu_id=0,
        server_args_overrides={
            "max_running_requests": 4,
            "max_total_tokens": 128,
            "skip_tokenizer_init": True,
            "attention_backend": "flashinfer",
            "disable_cuda_graph": False,
        },
    )
    try:
        runner = scheduler.model_runner
        backend_runner = runner.tp_worker.model_runner
        graph = backend_runner.decode_cuda_graph_runner
        assert graph is not None
        assert "flashinfer" in type(backend_runner.attn_backend).__module__
        model = backend_runner.model
        device = model.embed_tokens.weight.device
        kv_pool = backend_runner.token_to_kv_pool
        request_pool = backend_runner.req_to_token_pool
        request_pool.req_to_token[1:5, :8] = torch.arange(1, 33, device=device).reshape(
            4, 8
        )
        cache_generator = torch.Generator(device=device).manual_seed(185)
        for layer_index in model.attention_layer_ids():
            kv_pool.get_key_buffer(layer_index)[:33].normal_(
                std=0.1, generator=cache_generator
            )
            kv_pool.get_value_buffer(layer_index)[:33].normal_(
                std=0.1, generator=cache_generator
            )
        lengths = [2] * 4
        with torch.inference_mode():
            for step, order in enumerate(([0, 1, 2, 3], [3, 1, 0], [1], [0, 3])):
                requests = [
                    SchedulerRequest(
                        request_id=f"request-{index}",
                        data=MossSpeechSGLangRequestData(
                            prompt_rows=torch.tensor([[3, AUDIO_PAD]]),
                            mode=MODE_AUDIO if index % 2 else MODE_TEXT,
                            output_rows=[
                                (
                                    (MODALITY_PAD, (step + index) % 64)
                                    if index % 2
                                    else (step + index + 3, AUDIO_PAD)
                                )
                            ],
                        ),
                    )
                    for index in order
                ]
                positions = torch.tensor(
                    [lengths[index] for index in order], device=device
                )
                slots = torch.tensor(
                    [index * 8 + lengths[index] + 1 for index in order], device=device
                )
                batch = ForwardBatch(
                    capture_hidden_mode=CaptureHiddenMode.NULL,
                    forward_mode=ForwardMode.DECODE,
                    batch_size=len(order),
                    input_ids=torch.zeros(len(order), dtype=torch.long, device=device),
                    req_pool_indices=torch.tensor(
                        [index + 1 for index in order], device=device
                    ),
                    seq_lens=(positions + 1).to(torch.int32),
                    seq_lens_cpu=(positions + 1).to(device="cpu", dtype=torch.int32),
                    out_cache_loc=slots,
                    seq_lens_sum=sum(lengths[index] + 1 for index in order),
                    positions=positions,
                    rids=[request.request_id for request in requests],
                )
                runner.before_decode(batch, None, requests)
                before = [
                    (
                        kv_pool.get_key_buffer(layer_index)[:33].clone(),
                        kv_pool.get_value_buffer(layer_index)[:33].clone(),
                    )
                    for layer_index in model.attention_layer_ids()
                ]
                with patch.object(graph, "execute", wraps=graph.execute) as replay:
                    captured = backend_runner.forward(batch)
                    assert captured.can_run_graph
                    replay.assert_called_once()
                captured_logits = (
                    captured.logits_output.text_logits[: len(order)].clone(),
                    captured.logits_output.audio_logits[: len(order)].clone(),
                )
                unchanged = torch.ones(33, dtype=torch.bool, device=device)
                unchanged[0] = False
                unchanged[slots] = False
                written = []
                for layer_index, (old_keys, old_values) in zip(
                    model.attention_layer_ids(), before
                ):
                    keys = kv_pool.get_key_buffer(layer_index)[:33]
                    values = kv_pool.get_value_buffer(layer_index)[:33]
                    torch.testing.assert_close(
                        keys[unchanged], old_keys[unchanged], rtol=0, atol=0
                    )
                    torch.testing.assert_close(
                        values[unchanged], old_values[unchanged], rtol=0, atol=0
                    )
                    assert not torch.equal(keys[slots], old_keys[slots])
                    assert not torch.equal(values[slots], old_values[slots])
                    written.append((keys[slots].clone(), values[slots].clone()))
                with patch.object(backend_runner, "decode_cuda_graph_runner", None):
                    eager = backend_runner.forward(batch)
                    assert not eager.can_run_graph
                for actual, expected in zip(
                    captured_logits,
                    (eager.logits_output.text_logits, eager.logits_output.audio_logits),
                ):
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                for layer_index, (keys, values) in zip(
                    model.attention_layer_ids(), written
                ):
                    torch.testing.assert_close(
                        kv_pool.get_key_buffer(layer_index)[slots], keys, rtol=0, atol=0
                    )
                    torch.testing.assert_close(
                        kv_pool.get_value_buffer(layer_index)[slots],
                        values,
                        rtol=0,
                        atol=0,
                    )
                selected = runner.sample_next_token_ids(
                    captured.logits_output, batch, None, requests
                )
                assert selected.shape == (len(order),)
                for row_index, request in enumerate(requests):
                    text_token = int(captured_logits[0][row_index].argmax())
                    audio_token = int(
                        captured_logits[1][row_index, :AUDIO_FORBIDDEN_FROM].argmax()
                    )
                    expected_row = (
                        MODALITY_PAD if request.data.mode == MODE_AUDIO else text_token,
                        audio_token,
                    )
                    assert request.data.output_rows[-1] == expected_row
                    assert int(selected[row_index]) == (
                        audio_token if request.data.mode == MODE_AUDIO else text_token
                    )
                for index in order:
                    lengths[index] += 1
    finally:
        scheduler.stop()
        destroy_model_parallel()
        destroy_distributed_environment()
