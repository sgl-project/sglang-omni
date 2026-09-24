# SPDX-License-Identifier: Apache-2.0
"""Public MiniCPM-o vocoder contracts: import, checkpoint decode, speaker ref."""

from __future__ import annotations

import base64
import math
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from sglang_omni.models.minicpm_o.components.code2wav import (
    SAMPLES_PER_CODEC_TOKEN,
    MiniCPMOCode2Wav,
)
from sglang_omni.models.minicpm_o.components.token2wav.dit import TimestepEmbedder
from sglang_omni.models.minicpm_o.components.token2wav.hift import HiFTGenerator
from sglang_omni.models.minicpm_o.components.token2wav.vocoder import (
    MiniCPMOReferenceEncodeHook,
    resolve_token2wav_assets,
)
from sglang_omni.models.minicpm_o.config import MiniCPMOSpeechPipelineConfig
from sglang_omni.models.minicpm_o.payload_types import (
    MiniCPMOPipelineState,
    SpeakerPromptInputs,
)
from sglang_omni.models.minicpm_o.routing import (
    project_preprocessing_to_thinker,
    project_talker_to_code2wav,
    project_thinker_to_decode,
    project_thinker_to_talker,
    speaker_reference_audio,
)
from sglang_omni.models.minicpm_o.stages import vocode_code2wav_payloads
from sglang_omni.proto import OmniRequest, StagePayload
from sglang_omni.scheduling.reference_encoder import ReferenceEncodeService

REPO_ROOT = Path(__file__).resolve().parents[3]


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("frequency_size", [255, 256])
def test_timestep_embedding_matches_reference(
    dtype: torch.dtype, frequency_size: int
) -> None:
    model = TimestepEmbedder(16, frequency_size).to(dtype).eval()
    t = torch.linspace(0, 1, 11, dtype=dtype)
    half = frequency_size // 2
    frequencies = torch.exp(-math.log(10000) * torch.arange(half) / half).to(t)
    angles = (t * 1000)[:, None] * frequencies[None]
    embedding = torch.cat([angles.cos(), angles.sin()], dim=-1)
    if frequency_size % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    torch.testing.assert_close(model(t), model.mlp(embedding), rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_timestep_embedding_autocast_preserves_frequencies(dtype: torch.dtype) -> None:
    model = TimestepEmbedder(16).to(device="cuda", dtype=dtype).eval()
    t = torch.linspace(0, 1, 11, device="cuda", dtype=torch.float32)
    frequencies = torch.exp(-math.log(10000) * torch.arange(128) / 128).to(t)
    angles = (t * 1000)[:, None] * frequencies[None]
    embedding = torch.cat([angles.cos(), angles.sin()], dim=-1)
    with torch.inference_mode(), torch.amp.autocast("cuda", dtype=dtype):
        torch.testing.assert_close(model(t), model.mlp(embedding), rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_timestep_embedding_cuda_graph_replays_new_inputs() -> None:
    model = TimestepEmbedder(16).cuda().eval()
    t = torch.zeros(2, device="cuda")
    with torch.inference_mode():
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                model(t)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = model(t)
        t.fill_(0.25)
        expected = model(t)
        graph.replay()
        torch.testing.assert_close(output, expected, rtol=0, atol=0)


def test_native_vocoder_import_does_not_require_legacy_packages() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import importlib.abc
import sys

class BlockLegacy(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {
            "stepaudio2", "s3tokenizer", "minicpmo", "hyperpyyaml"
        }:
            raise ImportError(f"Legacy dependency requested: {fullname}")

sys.meta_path.insert(0, BlockLegacy())
from sglang_omni.models.minicpm_o.components.token2wav.vocoder import Token2Wav
""",
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr


def _checkpoint_dir() -> Path | None:
    env = os.environ.get("MINICPMO_CHECKPOINT")
    hf_home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    candidates = [Path(env)] if env else []
    candidates += [REPO_ROOT / "MiniCPM-o-4_6", REPO_ROOT / "MiniCPM-o-4_5"]
    for hub in (
        hf_home / "hub" / "models--openbmb--MiniCPM-o-4_5" / "snapshots",
        hf_home / "models--openbmb--MiniCPM-o-4_5" / "snapshots",
    ):
        if hub.is_dir():
            candidates.extend(sorted(hub.iterdir(), reverse=True))
    for path in candidates:
        if path is not None and (path / "assets" / "token2wav").is_dir():
            return path
    return None


def _default_speaker_prompt(checkpoint: Path) -> SpeakerPromptInputs:
    asset_dir, default_reference = resolve_token2wav_assets(str(checkpoint))
    hook = MiniCPMOReferenceEncodeHook(
        asset_dir,
        device=torch.device("cuda:0"),
        default_reference=default_reference,
        onnx_intra_op_threads=16,
    )
    return hook.encode_one(None)


def _speaker_prompt(
    prompt_len: int, *, mel_frames: int | None = None
) -> SpeakerPromptInputs:
    return {
        "speech_tokens": torch.zeros(1, prompt_len, dtype=torch.int32),
        "speech_token_len": torch.tensor([prompt_len], dtype=torch.int32),
        "speaker_embedding": torch.zeros(1, 4),
        "prompt_mel": torch.zeros(1, mel_frames or prompt_len * 2, 80),
    }


@pytest.mark.accelerator
def test_native_vocoder_with_checkpoint() -> None:
    checkpoint = _checkpoint_dir()
    if checkpoint is None or not torch.cuda.is_available():
        pytest.skip("Set MINICPMO_CHECKPOINT and provide CUDA for vocoder validation")
    model = MiniCPMOCode2Wav(
        str(checkpoint), device="cuda:0", hift_max_padding_waste=1.5
    )
    tokens = [1498, 1734, 3732, 3726, 3645]
    output = model(
        codec_tokens=torch.tensor(tokens),
        speaker_prompt=_default_speaker_prompt(checkpoint),
    )
    waveform = output["waveform"]
    assert output["sample_rate"] == 24000
    assert waveform.dtype == np.float32
    assert waveform.shape == (len(tokens) * SAMPLES_PER_CODEC_TOKEN,)
    assert np.isfinite(waveform).all()
    assert np.max(np.abs(waveform)) > 1e-5
    assert np.max(np.abs(waveform)) <= 0.99


@pytest.mark.accelerator
def test_native_vocoder_batch_matches_single_request_shapes() -> None:
    checkpoint = _checkpoint_dir()
    if checkpoint is None or not torch.cuda.is_available():
        pytest.skip("Set MINICPMO_CHECKPOINT and provide CUDA for vocoder validation")
    model = MiniCPMOCode2Wav(
        str(checkpoint), device="cuda:0", hift_max_padding_waste=1.5
    )
    prompt = _default_speaker_prompt(checkpoint)
    tokens_a = [1498, 1734, 3732, 3726, 3645]
    tokens_b = tokens_a + [3645, 3726]
    batched = model.vocode([tokens_a, tokens_b], [prompt, prompt])
    single_a = model.vocode([tokens_a], [prompt])[0]
    single_b = model.vocode([tokens_b], [prompt])[0]
    assert (
        batched[0].shape == single_a.shape == (len(tokens_a) * SAMPLES_PER_CODEC_TOKEN,)
    )
    assert (
        batched[1].shape == single_b.shape == (len(tokens_b) * SAMPLES_PER_CODEC_TOKEN,)
    )
    assert all(np.isfinite(wave).all() for wave in (*batched, single_a, single_b))


def _data_uri(audio: bytes) -> str:
    return "data:audio/wav;base64," + base64.b64encode(audio).decode("ascii")


def _payload(
    *,
    request_id: str = "test",
    tokens: list[int] | None = None,
    params: dict[str, object] | None = None,
    metadata: dict[str, object] | None = None,
    speaker_prompt: SpeakerPromptInputs | None = None,
) -> StagePayload:
    return StagePayload(
        request_id=request_id,
        request=OmniRequest(inputs=None, params=params or {}, metadata=metadata or {}),
        data=MiniCPMOPipelineState(
            engine_outputs={"talker": {"codec_tokens": torch.tensor(tokens or [1, 2])}},
            speaker_prompt=speaker_prompt,
        ).to_dict(),
    )


def test_chat_api_forwards_reference_to_vocoder() -> None:
    from sglang_omni.client.client import build_params
    from sglang_omni.serve.openai_api import (
        ChatCompletionRequest,
        build_chat_generate_request,
    )

    reference = _data_uri(b"reference")
    request = ChatCompletionRequest(
        model="minicpm-o",
        messages=[{"role": "user", "content": "Hello"}],
        modalities=["text", "audio"],
        audio={"format": "wav", "ref_audio": reference},
    )
    generate_request = build_chat_generate_request(request)
    payload = _payload(
        params=build_params(generate_request), metadata=generate_request.metadata
    )
    assert speaker_reference_audio(payload) == b"reference"


def test_invalid_reference_does_not_silently_use_default() -> None:
    payload = _payload(params={"ref_audio": "/tmp/ref.wav"})
    with pytest.raises(ValueError, match="inline audio"):
        speaker_reference_audio(payload)


def test_speaker_prompt_reaches_code2wav_but_not_decode() -> None:
    prompt = _speaker_prompt(3)
    payload = StagePayload(
        request_id="speaker",
        request=OmniRequest(inputs=None, params={}, metadata={}),
        data=MiniCPMOPipelineState(
            prompt={"prompt_text": "", "input_ids": torch.tensor([1])},
            speaker_prompt=prompt,
        ).to_dict(),
    )
    thinker = project_preprocessing_to_thinker(payload)
    thinker_state = MiniCPMOPipelineState.from_dict(thinker.data)
    thinker_state.thinker_out = {"output_ids": [1]}
    thinker.data = thinker_state.to_dict()
    talker = project_thinker_to_talker(thinker)
    talker_state = MiniCPMOPipelineState.from_dict(talker.data)
    talker_state.engine_outputs["talker"] = {"codec_tokens": torch.tensor([1])}
    talker.data = talker_state.to_dict()
    code2wav = project_talker_to_code2wav(talker)
    assert MiniCPMOPipelineState.from_dict(code2wav.data).speaker_prompt is prompt
    decode = project_thinker_to_decode(thinker)
    assert "speaker_prompt" not in decode.data


def test_reference_service_reuses_references_across_calls() -> None:
    hook = object.__new__(MiniCPMOReferenceEncodeHook)
    hook.default_reference = None
    hook.model_revision = "test"
    hook.encoder_config_hash = "test"
    hook.extract = MagicMock(return_value=_speaker_prompt(2))
    service = ReferenceEncodeService(hook, max_items=2)
    first = service.get_or_encode(b"a")
    first["speech_tokens"].add_(7)
    assert service.get_or_encode(b"a")["speech_tokens"].eq(0).all()
    # "c" evicts the least recently used "a" from the two-entry budget.
    for reference in (b"b", b"c", b"a"):
        service.get_or_encode(reference)
    assert hook.extract.call_count == 4
    with pytest.raises(ValueError, match="No speaker-reference"):
        service.get_or_encode(None)


def test_speech_pipeline_enables_code2wav_batching_by_default() -> None:
    config = MiniCPMOSpeechPipelineConfig(model_path="unused")
    code2wav = next(stage for stage in config.stages if stage.name == "code2wav")
    assert code2wav.factory.max_batch_size == 8
    assert code2wav.factory.max_batch_wait_ms == 0.0
    assert code2wav.factory.batch_wait_when_idle is False
    assert code2wav.factory.hift_max_padding_waste == 1.5


def test_vocode_slices_waveforms_to_token_lengths() -> None:
    class FakeFlow:
        up_rate = 2

        def inference(
            self,
            speech_tokens: torch.Tensor,
            speech_tokens_lens: torch.Tensor,
            *args: object,
        ) -> torch.Tensor:
            frames = speech_tokens.shape[1] * self.up_rate
            return torch.zeros(speech_tokens.shape[0], 80, frames)

    class FakeHiFT:
        def __call__(
            self, speech_feat: torch.Tensor, mel_lengths: list[int]
        ) -> tuple[torch.Tensor, None]:
            samples = speech_feat.shape[-1] * (SAMPLES_PER_CODEC_TOKEN // 2)
            wav = speech_feat.new_ones(speech_feat.shape[0], 1, samples)
            return wav, None

    model = MiniCPMOCode2Wav.__new__(MiniCPMOCode2Wav)
    model.hift_max_padding_waste = 1.5
    model.token2wav = SimpleNamespace(
        device=torch.device("cpu"),
        dtype=torch.float32,
        n_timesteps=10,
        flow=FakeFlow(),
        hift=FakeHiFT(),
    )
    waveforms = model.vocode([[1, 2], [3, 4, 5]], [_speaker_prompt(1)] * 2)
    assert [wave.shape for wave in waveforms] == [
        (2 * SAMPLES_PER_CODEC_TOKEN,),
        (3 * SAMPLES_PER_CODEC_TOKEN,),
    ]


def test_vocode_rejects_mismatched_speaker_prompt_count() -> None:
    model = MiniCPMOCode2Wav.__new__(MiniCPMOCode2Wav)
    with pytest.raises(ValueError, match="does not match"):
        model.vocode([[1], [2]], [_speaker_prompt(1)])


def _batch_model() -> MiniCPMOCode2Wav:
    class FakeFlow:
        up_rate = 2

        def inference(
            self,
            speech_tokens: torch.Tensor,
            speech_tokens_lens: torch.Tensor,
            prompt_tokens: torch.Tensor,
            prompt_tokens_lens: torch.Tensor,
            prompt_mels: torch.Tensor,
            speaker_embedding: torch.Tensor,
            n_timesteps: int,
        ) -> torch.Tensor:
            frames = (speech_tokens_lens + prompt_tokens_lens).max() * self.up_rate
            return torch.zeros(speech_tokens.shape[0], 80, frames)

    class FakeHiFT:
        def __call__(
            self, speech_feat: torch.Tensor, mel_lengths: list[int]
        ) -> tuple[torch.Tensor, None]:
            samples = speech_feat.shape[-1] * (SAMPLES_PER_CODEC_TOKEN // 2)
            return speech_feat.new_ones(speech_feat.shape[0], 1, samples), None

    model = MiniCPMOCode2Wav.__new__(MiniCPMOCode2Wav)
    model.hift_max_padding_waste = 1.5
    model.token2wav = SimpleNamespace(
        device=torch.device("cpu"),
        dtype=torch.float32,
        n_timesteps=10,
        flow=FakeFlow(),
        hift=FakeHiFT(),
    )
    return model


def test_vocode_mixed_references_and_lengths_share_one_batch() -> None:
    model = _batch_model()
    waveforms = model.vocode(
        [[1, 2], [3, 4, 5], [6]],
        [_speaker_prompt(1), _speaker_prompt(3, mel_frames=5), _speaker_prompt(1)],
    )
    assert [wave.shape for wave in waveforms] == [
        (2 * SAMPLES_PER_CODEC_TOKEN,),
        (3 * SAMPLES_PER_CODEC_TOKEN,),
        (SAMPLES_PER_CODEC_TOKEN,),
    ]


def test_vocode_mixed_lengths_preserve_hift_boundaries() -> None:
    class BoundarySensitiveHiFT:
        def __init__(self) -> None:
            self.batch_sizes: list[int] = []

        def __call__(
            self, speech_feat: torch.Tensor, mel_lengths: list[int]
        ) -> tuple[torch.Tensor, None]:
            self.batch_sizes.append(speech_feat.shape[0])
            positions = torch.arange(speech_feat.shape[-1])
            mask = (positions < torch.tensor(mel_lengths)[:, None]).unsqueeze(1)
            kernel = speech_feat.new_ones(1, 1, 3)
            hidden = (
                torch.nn.functional.conv1d(speech_feat[:, :1], kernel, padding=1) + 1
            ) * mask
            samples = torch.nn.functional.conv1d(hidden, kernel, padding=1)
            waveform = samples.repeat_interleave(SAMPLES_PER_CODEC_TOKEN // 2, dim=-1)
            return waveform, None

    model = _batch_model()
    hift = BoundarySensitiveHiFT()
    model.token2wav.hift = hift
    sequences = [[1, 2], [3, 4, 5], [6, 7]]
    prompt = _speaker_prompt(1)
    batched = model.vocode(sequences, [prompt] * len(sequences))
    assert hift.batch_sizes == [3]
    for tokens, waveform in zip(sequences, batched, strict=True):
        reference = model.vocode([tokens], [prompt])[0]
        np.testing.assert_array_equal(waveform, reference)


def test_vocode_splits_hift_batch_past_padding_budget() -> None:
    model = _batch_model()
    model.hift_max_padding_waste = 1.0
    calls: list[list[int]] = []
    fake_hift = model.token2wav.hift

    def record(
        speech_feat: torch.Tensor, mel_lengths: list[int]
    ) -> tuple[torch.Tensor, None]:
        calls.append(mel_lengths)
        return fake_hift(speech_feat, mel_lengths)

    model.token2wav.hift = record
    model.vocode([[1, 2], [3, 4, 5], [6, 7]], [_speaker_prompt(1)] * 3)
    assert calls == [[4, 4], [6]]


def test_hift_padded_batch_matches_single_rows(monkeypatch) -> None:
    # Remove the random excitation phase and noise so rows are comparable.
    monkeypatch.setattr(torch, "rand", torch.zeros)
    monkeypatch.setattr(torch, "randn_like", torch.zeros_like)
    torch.manual_seed(0)
    hift = HiFTGenerator().eval()
    mel_lengths = [14, 9, 6]
    mel = torch.randn(len(mel_lengths), 80, max(mel_lengths))
    for row, length in enumerate(mel_lengths):
        mel[row, :, length:] = 0

    padded, _ = hift(speech_feat=mel, mel_lengths=mel_lengths)
    unmasked, _ = hift(speech_feat=mel)
    samples_per_frame = SAMPLES_PER_CODEC_TOKEN // 2
    short_row = len(mel_lengths) - 1
    for row, length in enumerate(mel_lengths):
        single, _ = hift(speech_feat=mel[row : row + 1, :, :length])
        torch.testing.assert_close(
            padded[row, : length * samples_per_frame],
            single[0],
            rtol=1e-4,
            atol=1e-5,
        )
        if row == short_row:
            tail_error = (unmasked[row, : length * samples_per_frame] - single[0]).abs()
            assert tail_error.max() > 1e-3


def test_vocode_rejects_empty_sequences() -> None:
    model = MiniCPMOCode2Wav.__new__(MiniCPMOCode2Wav)
    assert model.vocode([], []) == []
    with pytest.raises(ValueError, match="non-empty"):
        model.vocode([[1], []], [_speaker_prompt(1)] * 2)


def _fake_code2wav_model() -> MagicMock:
    fake = MagicMock()
    fake.sample_rate = 24000
    fake.vocode.side_effect = lambda sequences, speaker_prompts: [
        np.full(
            len(tokens) * SAMPLES_PER_CODEC_TOKEN,
            float(len(tokens)),
            dtype=np.float32,
        )
        for tokens in sequences
    ]
    return fake


def test_vocode_payloads_uses_preprocessed_speaker_prompts() -> None:
    fake = _fake_code2wav_model()
    prompts = [_speaker_prompt(1), _speaker_prompt(3), _speaker_prompt(2)]
    outputs = vocode_code2wav_payloads(
        fake,
        [
            _payload(request_id="a", tokens=[1, 2], speaker_prompt=prompts[0]),
            _payload(request_id="b", tokens=[3], speaker_prompt=prompts[1]),
            _payload(request_id="c", tokens=[4, 5, 6], speaker_prompt=prompts[2]),
        ],
    )
    sequences, speaker_prompts = fake.vocode.call_args.args
    assert sequences == [[1, 2], [3], [4, 5, 6]]
    assert all(a is b for a, b in zip(speaker_prompts, prompts, strict=True))
    assert [out.data["audio_waveform_shape"][0] for out in outputs] == [
        2 * SAMPLES_PER_CODEC_TOKEN,
        SAMPLES_PER_CODEC_TOKEN,
        3 * SAMPLES_PER_CODEC_TOKEN,
    ]
    assert outputs[0].data["sample_rate"] == 24000


def test_vocode_payloads_rejects_missing_speaker_prompt() -> None:
    with pytest.raises(RuntimeError, match="without speaker conditioning"):
        vocode_code2wav_payloads(_fake_code2wav_model(), [_payload(tokens=[1])])
