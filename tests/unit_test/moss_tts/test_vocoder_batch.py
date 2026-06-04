# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import torch

from sglang_omni.models.moss_tts.payload_types import MossTTSState
from sglang_omni.proto import OmniRequest, StagePayload


def make_payload(request_id: str, delayed_codes: torch.Tensor) -> StagePayload:
    state = MossTTSState(
        delayed_audio_codes=delayed_codes,
        assistant_start_length=0,
    )
    return StagePayload(
        request_id=request_id,
        request=OmniRequest(inputs="hello", params={}, metadata={}),
        data=state.to_dict(),
    )


def make_mock_processor(waveforms: list[torch.Tensor]) -> MagicMock:
    processor = MagicMock()
    processor.model_config.audio_pad_code = 1024
    processor.model_config.sampling_rate = 24000

    mock_audio_tokenizer = MagicMock()
    mock_audio_tokenizer.parameters.return_value = iter([torch.zeros(1)])

    mock_dec = MagicMock()
    mock_dec.audio_lengths = torch.tensor([w.shape[0] for w in waveforms])
    mock_dec.audio = torch.zeros(len(waveforms), 1, max(w.shape[0] for w in waveforms))
    for i, w in enumerate(waveforms):
        mock_dec.audio[i, 0, : w.shape[0]] = w
    mock_audio_tokenizer.decode.return_value = mock_dec
    processor.audio_tokenizer = mock_audio_tokenizer
    return processor


def test_vocode_batch_calls_decode_once():
    """_vocode_batch should call decode_audio_codes exactly once for a batch."""
    from sglang_omni.models.moss_tts.stages import create_vocoder_executor

    codes1 = torch.tensor(
        [[1, 1024], [2, 3], [1024, 4], [1024, 1024]], dtype=torch.long
    )
    codes2 = torch.tensor(
        [[5, 1024], [6, 7], [1024, 8], [1024, 1024]], dtype=torch.long
    )

    payload1 = make_payload("req-1", codes1)
    payload2 = make_payload("req-2", codes2)

    wav1 = torch.ones(100)
    wav2 = torch.ones(80)
    mock_processor = make_mock_processor([wav1, wav2])

    with patch(
        "sglang_omni.models.moss_tts.stages._load_moss_processor",
        return_value=mock_processor,
    ):
        scheduler = create_vocoder_executor("fake/model", device="cpu")
        results = scheduler._batch_fn([payload1, payload2])

    assert mock_processor.audio_tokenizer.decode.call_count == 1
    assert len(results) == 2


def test_vocode_batch_distributes_waveforms_correctly():
    """Each payload should get its own waveform back."""
    from sglang_omni.models.moss_tts.stages import create_vocoder_executor

    codes1 = torch.tensor(
        [[1, 1024], [2, 3], [1024, 4], [1024, 1024]], dtype=torch.long
    )
    codes2 = torch.tensor(
        [[5, 1024], [6, 7], [1024, 8], [1024, 1024]], dtype=torch.long
    )

    payload1 = make_payload("req-1", codes1)
    payload2 = make_payload("req-2", codes2)

    wav1 = torch.full((100,), 1.0)
    wav2 = torch.full((80,), 2.0)
    mock_processor = make_mock_processor([wav1, wav2])

    with patch(
        "sglang_omni.models.moss_tts.stages._load_moss_processor",
        return_value=mock_processor,
    ):
        scheduler = create_vocoder_executor("fake/model", device="cpu")
        results = scheduler._batch_fn([payload1, payload2])

    audio1 = torch.from_numpy(
        np.frombuffer(results[0].data["audio_waveform"], dtype=np.float32).copy()
    )
    audio2 = torch.from_numpy(
        np.frombuffer(results[1].data["audio_waveform"], dtype=np.float32).copy()
    )

    assert torch.allclose(audio1, wav1)
    assert torch.allclose(audio2, wav2)


def test_vocode_batch_single_payload():
    """batch size 1 should still work correctly."""
    from sglang_omni.models.moss_tts.stages import create_vocoder_executor

    codes = torch.tensor([[1, 1024], [2, 3], [1024, 4], [1024, 1024]], dtype=torch.long)
    payload = make_payload("req-1", codes)
    wav = torch.ones(100)
    mock_processor = make_mock_processor([wav])

    with patch(
        "sglang_omni.models.moss_tts.stages._load_moss_processor",
        return_value=mock_processor,
    ):
        scheduler = create_vocoder_executor("fake/model", device="cpu")
        results = scheduler._batch_fn([payload])

    assert mock_processor.audio_tokenizer.decode.call_count == 1
    assert len(results) == 1


def test_vocode_batch_multi_segment_payload():
    """A payload with two audio segments should cat them into one waveform."""
    from sglang_omni.models.moss_tts.stages import create_vocoder_executor

    codes = torch.tensor([[1, 1024], [2, 3], [1024, 4], [1024, 1024]], dtype=torch.long)
    payload = make_payload("req-1", codes)

    seg0_wav = torch.full((100,), 1.0)
    seg1_wav = torch.full((80,), 2.0)
    mock_processor = make_mock_processor([seg0_wav, seg1_wav])

    seg0 = torch.zeros(10, 2, dtype=torch.long)
    seg1 = torch.zeros(8, 2, dtype=torch.long)

    with (
        patch(
            "sglang_omni.models.moss_tts.stages._load_moss_processor",
            return_value=mock_processor,
        ),
        patch(
            "sglang_omni.models.moss_tts.stages.split_moss_audio_segments",
            return_value=[seg0, seg1],
        ),
    ):
        scheduler = create_vocoder_executor("fake/model", device="cpu")
        results = scheduler._batch_fn([payload])

    assert mock_processor.audio_tokenizer.decode.call_count == 1
    call_args = mock_processor.audio_tokenizer.decode.call_args
    assert call_args[0][0].shape[1] == 2

    audio = torch.from_numpy(
        np.frombuffer(results[0].data["audio_waveform"], dtype=np.float32).copy()
    )
    expected = torch.cat([seg0_wav, seg1_wav])
    assert torch.allclose(audio, expected)
