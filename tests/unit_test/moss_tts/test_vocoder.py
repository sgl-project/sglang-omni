# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

import sglang_omni.models.moss_tts.vocoder as vocoder_module
from sglang_omni.models.moss_tts.payload_types import MossTTSState
from sglang_omni.models.moss_tts.vocoder import (
    MossTTSVocoder,
    _copy_valid_waveforms_to_cpu,
)
from sglang_omni.models.moss_tts.vocoder_quantizer import (
    MossAudioTokenizerQuantizerDecoder,
)
from sglang_omni.proto import OmniRequest, StagePayload


def _make_payload(request_id: str) -> StagePayload:
    return StagePayload(
        request_id=request_id,
        request=OmniRequest(inputs=request_id, params={}),
        data={},
    )


class _AlwaysPackedVocoderDecoder(nn.Module):
    @classmethod
    def from_module(cls, source: nn.Module):
        return cls(source)

    def __init__(self, source: nn.Module) -> None:
        super().__init__()
        self.stages = source

    def __len__(self) -> int:
        return len(self.stages)

    def supports_packed_attention(self, device, dtype) -> bool:
        del device, dtype
        return True

    def output_lengths(self, input_lengths):
        output_lengths = list(input_lengths)
        for stage in self.stages:
            patch_size = int(getattr(stage, "patch_size", 1))
            if bool(getattr(stage, "is_downsample", False)):
                output_lengths = [length // patch_size for length in output_lengths]
            else:
                output_lengths = [length * patch_size for length in output_lengths]
        return output_lengths

    def forward(self, x, input_lengths, *, input_lengths_cpu=None):
        del input_lengths_cpu
        for stage in self.stages:
            x, input_lengths = stage(x, input_lengths)
        return x, input_lengths


class _NeverPackedVocoderDecoder(_AlwaysPackedVocoderDecoder):
    def supports_packed_attention(self, device, dtype) -> bool:
        del device, dtype
        return False


class _FakeCodebookQuantizer(nn.Module):
    def __init__(
        self,
        *,
        codebook_size: int,
        codebook_dim: int,
        output_dim: int,
    ) -> None:
        super().__init__()
        self.codebook = nn.Embedding(codebook_size, codebook_dim)
        self.out_proj = nn.Conv1d(codebook_dim, output_dim, kernel_size=1)

    def decode_code(self, codes: torch.Tensor) -> torch.Tensor:
        embedded = self.codebook(codes).transpose(1, 2).float()
        return self.out_proj(embedded).float()


class _FakeResidualQuantizer(nn.Module):
    def __init__(self, *, num_quantizers: int = 4) -> None:
        super().__init__()
        self.quantizers = nn.ModuleList(
            [
                _FakeCodebookQuantizer(
                    codebook_size=16,
                    codebook_dim=3,
                    output_dim=8,
                )
                for _ in range(num_quantizers)
            ]
        )
        self.output_proj = nn.Conv1d(8, 6, kernel_size=1)

    def decode_codes(self, codes: torch.Tensor) -> torch.Tensor:
        _, batch_size, frames = codes.shape
        decoded = torch.zeros(batch_size, 8, frames, dtype=torch.float32)
        for index, quantizer in enumerate(self.quantizers[: codes.shape[0]]):
            decoded += quantizer.decode_code(codes[index])
        return self.output_proj(decoded)


@pytest.mark.parametrize("num_quantizers", [1, 3, 4])
def test_cached_quantizer_matches_source_decode(num_quantizers: int) -> None:
    torch.manual_seed(0)
    source = _FakeResidualQuantizer()
    decoder = MossAudioTokenizerQuantizerDecoder(source)
    codes = torch.randint(0, 16, (num_quantizers, 2, 7))

    expected = source.decode_codes(codes)
    actual = decoder.decode_codes(codes)

    assert torch.equal(actual, expected)


def test_cached_quantizer_rejects_invalid_codebook_count() -> None:
    decoder = MossAudioTokenizerQuantizerDecoder(_FakeResidualQuantizer())

    with pytest.raises(ValueError, match="codebook count"):
        decoder.decode_codes(torch.empty((0, 2, 3), dtype=torch.long))
    with pytest.raises(ValueError, match="codebook count"):
        decoder.decode_codes(torch.zeros((5, 2, 3), dtype=torch.long))


def test_moss_tts_vocoder_copies_only_valid_waveforms() -> None:
    audio = torch.tensor(
        [
            [[1.0, 2.0, 99.0]],
            [[3.0, 4.0, 5.0]],
        ],
        dtype=torch.bfloat16,
    )

    waveforms = _copy_valid_waveforms_to_cpu(audio, [2, 3])

    assert [waveform.dtype for waveform in waveforms] == [
        torch.float32,
        torch.float32,
    ]
    assert [waveform.tolist() for waveform in waveforms] == [
        [[1.0, 2.0]],
        [[3.0, 4.0, 5.0]],
    ]


def test_moss_tts_vocoder_batches_mixed_length_segments_across_requests(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sglang_omni.models.moss_tts import stages

    class FakePatchStage(nn.Module):
        module_type = "PatchedPretransform"
        patch_size = 2
        is_downsample = False

        def __init__(self) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.ones((), dtype=torch.float32))
            self.autocast_enabled: list[bool] = []

        def forward(self, x, input_lengths):
            self.autocast_enabled.append(torch.is_autocast_enabled("cpu"))
            return x.repeat_interleave(2, dim=-1), input_lengths * 2

    class FakeQuantizer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.codebook = nn.Parameter(torch.zeros((), dtype=torch.bfloat16))
            self.decode_shapes: list[tuple[int, ...]] = []
            self.autocast_enabled: list[bool] = []

        def decode_codes(self, audio_codes):
            self.decode_shapes.append(tuple(audio_codes.shape))
            self.autocast_enabled.append(torch.is_autocast_enabled("cpu"))
            _, batch_size, frames = audio_codes.shape
            decoded = torch.zeros(batch_size, 1, frames, dtype=torch.float32)
            for index in range(batch_size):
                decoded[index].fill_(float(index + 1))
            return decoded

    class FakeCodec(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.dummy = nn.Parameter(torch.zeros(()))
            self.quantizer = FakeQuantizer()
            self.decoder = nn.ModuleList([FakePatchStage()])
            self.config = SimpleNamespace(sampling_rate=24000)

    class FakeAudioTokenizer:
        def __init__(self, model) -> None:
            self.model = model
            self.sample_rate = 24000
            self.decode_calls = 0

        def decode_codes(self, segments):
            self.decode_calls += 1
            return [torch.zeros(1) for _ in segments]

    codec = FakeCodec()
    audio_vocoder = FakeAudioTokenizer(codec)
    processor = SimpleNamespace(
        audio_tokenizer=None,
        model_config=SimpleNamespace(audio_pad_code=1024, sampling_rate=24000),
    )
    monkeypatch.setattr(stages, "_load_moss_processor", lambda *args: processor)
    monkeypatch.setattr(
        stages,
        "load_moss_audio_vocoder",
        lambda *args, **kwargs: audio_vocoder,
    )
    monkeypatch.setattr(
        vocoder_module,
        "MossAudioTokenizerVocoderDecoder",
        _AlwaysPackedVocoderDecoder,
    )

    scheduler = stages.create_vocoder_executor(
        "model",
        device="cpu",
        max_batch_size=4,
        compute_dtype="bfloat16",
    )
    first = _make_payload("first")
    first.data = MossTTSState(
        delayed_audio_codes=torch.tensor(
            [[1, 1024], [2, 3], [1024, 4], [1024, 1024]],
            dtype=torch.long,
        )
    ).to_dict()
    second = _make_payload("second")
    second.data = MossTTSState(
        delayed_audio_codes=torch.tensor(
            [[5, 1024], [6, 7], [9, 8], [1024, 10], [1024, 1024]],
            dtype=torch.long,
        )
    ).to_dict()

    results = asyncio.run(scheduler._batch_fn([first, second]))

    assert codec.quantizer.decode_shapes == [(2, 2, 3)]
    assert codec.quantizer.autocast_enabled == [False]
    assert codec.quantizer.codebook.dtype is torch.float32
    assert codec.decoder[0].weight.dtype is torch.float32
    assert codec.decoder[0].autocast_enabled == [True]
    assert audio_vocoder.decode_calls == 0
    assert np.frombuffer(
        results[0].data["audio_waveform"], dtype=np.float32
    ).tolist() == [1.0, 1.0, 1.0, 1.0]
    assert np.frombuffer(
        results[1].data["audio_waveform"], dtype=np.float32
    ).tolist() == [2.0, 2.0, 2.0, 2.0, 2.0, 2.0]
    assert results[0].data["sample_rate"] == 24000


def test_moss_tts_vocoder_uses_standalone_codec_without_packed_flash(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sglang_omni.models.moss_tts import stages

    class FakeQuantizer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.codebook = nn.Parameter(torch.zeros((), dtype=torch.bfloat16))

        def decode_codes(self, _audio_codes):
            pytest.fail("standalone fallback must not call quantizer directly")

    class FakeCodec(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.dummy = nn.Parameter(torch.zeros(()))
            self.quantizer = FakeQuantizer()
            self.decoder = nn.ModuleList([nn.Identity()])

    class FakeAudioTokenizer:
        sample_rate = 16000

        def __init__(self) -> None:
            self.model = FakeCodec()
            self.decode_calls = 0
            self.decode_shapes: list[list[tuple[int, ...]]] = []

        def decode_codes(self, segments):
            self.decode_calls += 1
            self.decode_shapes.append([tuple(segment.shape) for segment in segments])
            return [
                torch.full((2,), float(self.decode_calls), dtype=torch.float32)
                for _ in segments
            ]

    audio_vocoder = FakeAudioTokenizer()
    processor = SimpleNamespace(
        model_config=SimpleNamespace(audio_pad_code=1024, sampling_rate=16000)
    )
    monkeypatch.setattr(stages, "_load_moss_processor", lambda *args: processor)
    monkeypatch.setattr(
        stages,
        "load_moss_audio_vocoder",
        lambda *args, **kwargs: audio_vocoder,
    )
    monkeypatch.setattr(
        vocoder_module,
        "MossAudioTokenizerVocoderDecoder",
        _NeverPackedVocoderDecoder,
    )

    scheduler = stages.create_vocoder_executor(
        "model",
        device="cpu",
        max_batch_size=2,
        compute_dtype="bfloat16",
    )
    payloads = []
    for request_id, offset in (("first", 0), ("second", 4)):
        payload = _make_payload(request_id)
        payload.data = MossTTSState(
            delayed_audio_codes=torch.tensor(
                [
                    [1 + offset, 1024],
                    [2 + offset, 3 + offset],
                    [1024, 4 + offset],
                    [1024, 1024],
                ],
                dtype=torch.long,
            )
        ).to_dict()
        payloads.append(payload)

    results = asyncio.run(scheduler._batch_fn(payloads))

    assert scheduler._vocoder._nonstream_decoder is None
    assert audio_vocoder.model.quantizer.codebook.dtype is torch.bfloat16
    assert audio_vocoder.decode_calls == 2
    assert audio_vocoder.decode_shapes == [[(2, 2)], [(2, 2)]]
    assert np.frombuffer(
        results[0].data["audio_waveform"], dtype=np.float32
    ).tolist() == [1.0, 1.0]
    assert np.frombuffer(
        results[1].data["audio_waveform"], dtype=np.float32
    ).tolist() == [2.0, 2.0]


def test_moss_tts_vocoder_autocasts_low_precision_standalone_codec() -> None:
    class FakeCodec(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.decoder = nn.Linear(1, 1, bias=False).to(dtype=torch.bfloat16)

    class FakeAudioTokenizer:
        sample_rate = 16000

        def __init__(self) -> None:
            self.model = FakeCodec()
            self.autocast_enabled: list[bool] = []

        def decode_codes(self, _segments):
            self.autocast_enabled.append(torch.is_autocast_enabled("cpu"))
            hidden = torch.ones((1, 1), dtype=torch.float32)
            return [self.model.decoder(hidden).flatten().float()]

    audio_vocoder = FakeAudioTokenizer()
    processor = SimpleNamespace(
        model_config=SimpleNamespace(audio_pad_code=1024, sampling_rate=16000)
    )
    vocoder = MossTTSVocoder(processor, audio_vocoder, "cpu")
    state = MossTTSState()
    delayed_codes = torch.tensor(
        [[1, 1024], [2, 3], [1024, 4], [1024, 1024]],
        dtype=torch.long,
    )

    waveform, sample_rate = vocoder._decode_audio(state, delayed_codes)

    assert audio_vocoder.autocast_enabled == [True]
    assert waveform.dtype is torch.float32
    assert sample_rate == 16000


def test_moss_tts_vocoder_falls_back_after_packed_batch_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sglang_omni.models.moss_tts import stages

    released_markers: list[bool] = []

    class FailedBatchMarker:
        def __del__(self) -> None:
            released_markers.append(True)

    class FakeQuantizer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.codebook = nn.Parameter(torch.zeros((), dtype=torch.bfloat16))
            self.calls = 0
            self.autocast_enabled: list[bool] = []

        def decode_codes(self, audio_codes):
            self.calls += 1
            self.autocast_enabled.append(torch.is_autocast_enabled("cpu"))
            _, batch_size, frames = audio_codes.shape
            return torch.ones(batch_size, 1, frames, dtype=torch.float32)

    class FailingPackedDecoder(_AlwaysPackedVocoderDecoder):
        def __init__(self, source: nn.Module) -> None:
            super().__init__(source)
            self.calls = 0

        def forward(self, x, input_lengths, *, input_lengths_cpu=None):
            del input_lengths_cpu
            self.calls += 1
            marker = FailedBatchMarker()
            assert marker is not None
            raise RuntimeError(f"cannot batch shape={tuple(x.shape)}")

    class FakeCodec(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.dummy = nn.Parameter(torch.zeros(()))
            self.quantizer = FakeQuantizer()
            self.decoder = nn.ModuleList([nn.Identity()])

    class FakeAudioTokenizer:
        sample_rate = 16000

        def __init__(self) -> None:
            self.model = FakeCodec()
            self.decode_calls = 0
            self.decode_shapes: list[list[tuple[int, ...]]] = []

        def decode_codes(self, segments):
            assert released_markers == [True]
            self.decode_calls += 1
            self.decode_shapes.append([tuple(segment.shape) for segment in segments])
            return [
                torch.full((2,), float(self.decode_calls), dtype=torch.float32)
                for _ in segments
            ]

    audio_vocoder = FakeAudioTokenizer()
    processor = SimpleNamespace(
        model_config=SimpleNamespace(audio_pad_code=1024, sampling_rate=16000)
    )
    monkeypatch.setattr(stages, "_load_moss_processor", lambda *args: processor)
    monkeypatch.setattr(
        stages,
        "load_moss_audio_vocoder",
        lambda *args, **kwargs: audio_vocoder,
    )
    packed_decoders: list[FailingPackedDecoder] = []

    class FailingPackedDecoderFactory:
        @classmethod
        def from_module(cls, source: nn.Module):
            del cls
            decoder = FailingPackedDecoder(source)
            packed_decoders.append(decoder)
            return decoder

    monkeypatch.setattr(
        vocoder_module,
        "MossAudioTokenizerVocoderDecoder",
        FailingPackedDecoderFactory,
    )

    scheduler = stages.create_vocoder_executor(
        "model",
        device="cpu",
        max_batch_size=2,
        compute_dtype="bfloat16",
    )

    def make_vocoder_payload(request_id: str, offset: int) -> StagePayload:
        payload = _make_payload(request_id)
        payload.data = MossTTSState(
            delayed_audio_codes=torch.tensor(
                [
                    [1 + offset, 1024],
                    [2 + offset, 3 + offset],
                    [1024, 4 + offset],
                    [1024, 1024],
                ],
                dtype=torch.long,
            )
        ).to_dict()
        return payload

    first_results = asyncio.run(
        scheduler._batch_fn(
            [
                make_vocoder_payload("first", 0),
                make_vocoder_payload("second", 4),
            ]
        )
    )
    second_results = asyncio.run(
        scheduler._batch_fn([make_vocoder_payload("third", 8)])
    )

    quantizer = audio_vocoder.model.quantizer
    assert packed_decoders[0].calls == 1
    assert quantizer.calls == 1
    assert quantizer.autocast_enabled == [False]
    assert quantizer.codebook.dtype is torch.float32
    assert released_markers == [True]
    assert scheduler._vocoder._nonstream_decoder is None
    assert scheduler._vocoder._quantizer_decoder is None
    assert audio_vocoder.decode_calls == 3
    assert audio_vocoder.decode_shapes == [[(2, 2)], [(2, 2)], [(2, 2)]]
    assert np.frombuffer(
        first_results[0].data["audio_waveform"], dtype=np.float32
    ).tolist() == [1.0, 1.0]
    assert np.frombuffer(
        first_results[1].data["audio_waveform"], dtype=np.float32
    ).tolist() == [2.0, 2.0]
    assert np.frombuffer(
        second_results[0].data["audio_waveform"], dtype=np.float32
    ).tolist() == [3.0, 3.0]


@pytest.mark.parametrize("compute_dtype", [None, "float32"])
def test_moss_tts_vocoder_can_disable_batched_decode(
    monkeypatch: pytest.MonkeyPatch,
    compute_dtype: str | None,
) -> None:
    from sglang_omni.models.moss_tts import stages

    codec = nn.Module()
    codec.decoder = nn.ModuleList()
    audio_vocoder = SimpleNamespace(
        model=codec,
        sample_rate=16000,
        decode_codes=lambda segments: [torch.ones(2) for _ in segments],
    )
    processor = SimpleNamespace(
        model_config=SimpleNamespace(audio_pad_code=1024, sampling_rate=16000)
    )
    monkeypatch.setattr(stages, "_load_moss_processor", lambda *args: processor)
    monkeypatch.setattr(
        stages,
        "load_moss_audio_vocoder",
        lambda *args, **kwargs: audio_vocoder,
    )
    monkeypatch.setattr(
        vocoder_module,
        "MossAudioTokenizerVocoderDecoder",
        lambda *_args, **_kwargs: pytest.fail(
            "disabled batched decode must not construct the decoder wrapper"
        ),
    )

    scheduler = stages.create_vocoder_executor(
        "model",
        device="cpu",
        compute_dtype=compute_dtype,
    )
    payload = _make_payload("baseline")
    payload.data = MossTTSState(
        delayed_audio_codes=torch.tensor(
            [[1, 1024], [2, 3], [1024, 4], [1024, 1024]],
            dtype=torch.long,
        )
    ).to_dict()

    [result] = asyncio.run(scheduler._batch_fn([payload]))

    assert np.frombuffer(result.data["audio_waveform"], dtype=np.float32).tolist() == [
        1.0,
        1.0,
    ]
