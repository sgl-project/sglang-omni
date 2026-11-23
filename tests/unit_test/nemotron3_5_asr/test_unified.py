# SPDX-License-Identifier: Apache-2.0
"""Nemotron Unified ASR: checkpoint routing and engine streaming sessions."""

from __future__ import annotations

import json
import threading
from pathlib import Path

import huggingface_hub
import numpy as np
import pytest
import torch
from huggingface_hub.errors import OfflineModeIsEnabled
from transformers import (
    Nemotron3_5AsrConfig,
    Nemotron3_5AsrForRNNT,
    ParakeetFeatureExtractor,
)

import sglang_omni.utils.hf
from sglang_omni.config.manager import resolve_config_cls_for_model_path
from sglang_omni.models.nemotron3_5_asr.batch_engine import NemotronBatchEngine
from sglang_omni.models.nemotron3_5_asr.config import Nemotron3_5ASRPipelineConfig
from sglang_omni.models.nemotron3_5_asr.unified import (
    NemotronUnifiedASRModelRunner,
    is_unified_checkpoint,
)
from sglang_omni.proto.request import OmniRequest, StagePayload
from sglang_omni.proto.session import SessionIdentity, TimedChunk
from sglang_omni.utils.hf import (
    NEMOTRON_UNIFIED_ASR_ARCHIVE,
    try_resolve_arch_from_nemotron_unified_asr_layout,
)


class TinyProcessor:
    prompt_dictionary = {"en": 0}
    feature_extractor = ParakeetFeatureExtractor(feature_size=16)

    def batch_decode(self, sequences, skip_special_tokens: bool) -> list[str]:
        return [" ".join(map(str, sequence.tolist())) for sequence in sequences]


class TinyRunner(NemotronUnifiedASRModelRunner):
    def load_model(self, checkpoint, dtype, num_lookahead_tokens):
        torch.manual_seed(0)
        encoder = {"model_type": "parakeet_encoder", "num_mel_bins": 16}
        encoder.update(hidden_size=16, intermediate_size=32, num_hidden_layers=2)
        encoder.update(num_attention_heads=2, num_key_value_heads=2)
        config = Nemotron3_5AsrConfig(
            encoder_config={**encoder, "subsampling_conv_channels": 8},
            vocab_size=12,
            blank_token_id=11,
            pad_token_id=11,
            decoder_hidden_size=16,
            num_decoder_layers=1,
            num_prompts=4,
            prompt_intermediate_size=16,
        )
        return TinyProcessor(), Nemotron3_5AsrForRNNT(config)


def test_unified_checkpoints_route_to_the_nemotron3_5_stage(tmp_path: Path) -> None:
    transformers_layout, nemo_layout = tmp_path / "hf", tmp_path / "nemo"
    transformers_layout.mkdir()
    nemo_layout.mkdir()
    config = {
        "architectures": ["Nemotron3_5AsrForRNNT"],
        "encoder_config": {"model_type": "parakeet_encoder"},
    }
    transformers_layout.joinpath("config.json").write_text(json.dumps(config))
    nemo_layout.joinpath("nemotron-labs-unified-asr-0.6b.nemo").touch()
    for layout in (transformers_layout, nemo_layout):
        assert is_unified_checkpoint(str(layout))
        assert (
            resolve_config_cls_for_model_path(str(layout))
            is Nemotron3_5ASRPipelineConfig
        )
    assert not is_unified_checkpoint(str(tmp_path))


def test_cached_hub_archive_routes_without_the_hub(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo_id, revision = "nvidia/nemotron-labs-unified-asr-0.6b", "0" * 40
    repo = tmp_path / "models--nvidia--nemotron-labs-unified-asr-0.6b"
    repo.joinpath("refs").mkdir(parents=True)
    repo.joinpath("refs", "main").write_text(revision)
    snapshot = repo / "snapshots" / revision
    snapshot.mkdir(parents=True)
    snapshot.joinpath(NEMOTRON_UNIFIED_ASR_ARCHIVE).touch()
    monkeypatch.setattr(huggingface_hub.constants, "HF_HUB_CACHE", str(tmp_path))

    def offline(*args, **kwargs):
        raise OfflineModeIsEnabled("offline")

    monkeypatch.setattr(sglang_omni.utils.hf, "file_exists", offline)
    resolve = try_resolve_arch_from_nemotron_unified_asr_layout
    assert resolve(repo_id) == "Nemotron3_5AsrForRNNT"
    assert resolve("nvidia/other-asr") is None


def test_engine_sessions_match_whole_file_hops(tmp_path: Path) -> None:
    """20 ms appends through the batched engine decode like whole-file hops."""
    generator = np.random.default_rng(0)
    audio = [
        (generator.standard_normal(length) * 3000).astype("<i2")
        for length in (19080, 32000, 81417)
    ]
    runner = TinyRunner(str(tmp_path), device="cpu", encoder_graph_max_batch_size=1)
    engine = NemotronBatchEngine(
        runner,
        max_batch_size=8,
        max_batch_wait_ms=0,
        max_pending_tasks=64,
        max_open_sessions=8,
        max_state_bytes=1 << 30,
        max_pcm_bytes=1 << 20,
        max_history_tokens=16384,
        max_text_bytes=1 << 20,
    )
    request = OmniRequest(inputs=None, params={"language": "en"})
    expected = []
    for pcm in audio:
        state = runner.stream_state_type(
            request_id="r",
            payload=StagePayload(request_id="r", request=request, data=None),
            language="en",
            spec=engine.spec,
            decode=runner.new_streaming_decode_state(),
        )
        state.append_bytes(pcm.tobytes())
        state.mark_done()
        while state.has_ready_window():
            hop = runner.prepare_stream_window(state, state.pop_ready_window())
            runner.run_streaming_batch(
                [state.decode], [hop], requested_languages=["en"]
            )
        expected.append(state.decode.tokens)

    identities = [SessionIdentity(f"s{index}") for index in range(len(audio))]
    finals = {}
    try:
        for identity in identities:
            engine.open(identity, request).result(timeout=30)
        for start in range(0, max(map(len, audio)) + 320, 320):
            futures = []
            for index, pcm in enumerate(audio):
                if start <= len(pcm):
                    payload = pcm[start : start + 320].tobytes()
                    chunk = TimedChunk(
                        "audio",
                        start / 16,
                        len(payload) / 32,
                        start // 320,
                        payload,
                        format="pcm16",
                        eos=start + 320 > len(pcm),
                    )
                    empty = StagePayload(
                        request_id=str(index), request=request, data=None
                    )
                    futures.append(
                        (
                            index,
                            engine.append(
                                identities[index], chunk, empty, threading.Event()
                            ),
                        )
                    )
                else:
                    pass
            for index, future in futures:
                result = future.result(timeout=30)
                if result.is_final:
                    finals[index] = result.final_payload.data["token_ids"]
                else:
                    pass
    finally:
        engine.shutdown()
    assert [finals[index] for index in range(len(audio))] == expected
