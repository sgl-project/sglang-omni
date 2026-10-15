# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from sglang_omni.models.fishaudio_s2_pro import stages
from sglang_omni.models.fishaudio_s2_pro.fish_speech.models.dac import modded_dac
from sglang_omni.models.fishaudio_s2_pro.fish_speech.models.dac.modded_dac import DAC
from sglang_omni.models.fishaudio_s2_pro.payload_types import S2ProState
from sglang_omni.scheduling.message import IncomingMessage
from tests.unit_test.fixtures.fish_fakes import FakeFishCodec, make_s2pro_payload
from tests.unit_test.pipeline.helpers import run_scheduler


def test_fish_vocoder_batches_and_trims_audio_by_code_length(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Preserves batched vocoder decode and per-request trim by code length."""
    codec = FakeFishCodec(frame_length=4)
    monkeypatch.setattr(stages, "_resolve_checkpoint", lambda model_path: model_path)
    monkeypatch.setattr(stages, "load_codec", lambda checkpoint, device: codec)
    scheduler = stages.create_vocoder_executor(
        "unused",
        device="cpu",
        max_batch_size=4,
        max_batch_wait_ms=50,
    )
    assert scheduler.stream_stride == 40
    assert scheduler.stream_followup_stride == 45

    def payload(request_id: str, code_len: int) -> object:
        return make_s2pro_payload(
            S2ProState(
                output_codes=torch.arange(3 * code_len).reshape(3, code_len),
                prompt_tokens=4,
                completion_tokens=code_len,
                engine_time_s=0.5,
            ),
            request_id=request_id,
        )

    first, second = run_scheduler(
        scheduler,
        [
            IncomingMessage("req-short", "new_request", payload("req-short", 2)),
            IncomingMessage("req-long", "new_request", payload("req-long", 3)),
        ],
        output_count=2,
    )
    outputs = {first.request_id: first.data, second.request_id: second.data}

    assert codec.calls == [(2, 2, 3)]
    assert outputs["req-short"].data["audio_data"] == [1.0] * 8
    assert outputs["req-long"].data["audio_data"] == [2.0] * 12
    assert outputs["req-short"].data["usage"]["total_tokens"] == 6


CODEC_CONFIG_PATH = (
    Path(modded_dac.__file__).parents[2] / "configs" / "modded_dac_vq.yaml"
)
TINY_TRANSFORMER = {
    "input_dim": 64,
    "config": {"n_layer": 1, "n_head": 2, "dim": 64, "intermediate_size": 64},
}


def tiny_codec() -> DAC:
    OmegaConf.register_new_resolver("eval", eval, replace=True)
    config = OmegaConf.merge(
        OmegaConf.load(CODEC_CONFIG_PATH),
        OmegaConf.create(
            {
                "encoder_dim": 4,
                "decoder_dim": 32,
                "encoder_transformer_layers": [0, 0, 0, 1],
                "decoder_transformer_layers": [0, 0, 0, 0],
                "quantizer": {
                    "input_dim": 64,
                    "codebook_size": 16,
                    "semantic_codebook_size": 32,
                    "post_module": TINY_TRANSFORMER,
                    "pre_module": TINY_TRANSFORMER,
                },
            }
        ),
    )
    torch.manual_seed(0)
    codec = instantiate(config)
    assert isinstance(codec, DAC)
    return codec.eval()


def test_folding_weight_norm_keeps_what_the_codec_decodes_and_encodes() -> None:
    codec = tiny_codec()
    codes = torch.randint(0, 16, (1, 10, 12))
    audio = 0.1 * torch.randn(1, 4096)
    audio_lengths = torch.tensor([4096])
    with torch.no_grad():
        decoded = codec.from_indices(codes.clone())
        encoded, _ = codec.encode(audio, audio_lengths)
        codec.fold_weight_norm()
        decoded_after = codec.from_indices(codes.clone())
        encoded_after, _ = codec.encode(audio, audio_lengths)

    assert torch.equal(decoded_after, decoded)
    assert torch.equal(encoded_after, encoded)
    assert not [
        name
        for name, _ in codec.named_parameters()
        if name.endswith(("weight_g", "weight_v", "original0", "original1"))
    ]


@pytest.mark.accelerator
def test_a_codec_decode_can_be_captured_in_a_cuda_graph() -> None:
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA device")
    else:
        pass
    codec = tiny_codec().cuda()
    codec.fold_weight_norm()
    static_codes = torch.randint(0, 16, (1, 10, 12), device="cuda")
    graph = torch.cuda.CUDAGraph()
    with torch.no_grad():
        eager_audio = codec.from_indices(static_codes.clone())
        with torch.cuda.graph(graph):
            graph_audio = codec.from_indices(static_codes)
        graph.replay()

    torch.testing.assert_close(graph_audio, eager_audio)
