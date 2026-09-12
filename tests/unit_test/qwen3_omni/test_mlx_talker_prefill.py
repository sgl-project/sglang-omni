# SPDX-License-Identifier: Apache-2.0
"""Parity tests for the native MLX Qwen3-Omni talker prefill builder."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
import torch

pytest.importorskip("mlx.core")

from sglang_omni.models.qwen3_omni.components.talker_prefill import (  # noqa: E402
    TalkerPrefillBuilder,
)
from sglang_omni.models.qwen3_omni.mlx import (  # noqa: E402
    talker_prefill as mlx_prefill,
)
from sglang_omni.models.qwen3_omni.mlx.runner import (  # noqa: E402
    load_qwen3_omni_mlx_talker,
)
from sglang_omni.models.qwen3_omni.mlx.talker_prefill import (  # noqa: E402
    Qwen3OmniMlxTalkerPrefillBuilder,
)
from sglang_omni.models.qwen3_omni.payload_types import (  # noqa: E402
    Qwen3OmniPipelineState,
)
from sglang_omni.models.qwen3_omni.pending_text_queue import (  # noqa: E402
    PendingTextTensorQueue,
)
from sglang_omni.proto.request import OmniRequest, StagePayload  # noqa: E402
from tests.utils.build_tiny_qwen3_omni_checkpoint import (  # noqa: E402
    OFFICIAL_SPEAKER_IDS,
    OFFICIAL_SPECIAL_TOKEN_IDS,
    TINY_DIMS,
    build_tiny_qwen3_omni_checkpoint,
)


@pytest.fixture(scope="module")
def tiny_checkpoint(tmp_path_factory: pytest.TempPathFactory):
    root = tmp_path_factory.mktemp("tiny_qwen3_omni_mlx_prefill")
    return build_tiny_qwen3_omni_checkpoint(root / "tiny")


def tiny_talker_token_ids() -> dict[str, int]:
    return {
        "audio_token_id": OFFICIAL_SPECIAL_TOKEN_IDS["thinker_config.audio_token_id"],
        "image_token_id": OFFICIAL_SPECIAL_TOKEN_IDS["thinker_config.image_token_id"],
        "video_token_id": OFFICIAL_SPECIAL_TOKEN_IDS["thinker_config.video_token_id"],
        "tts_bos_token_id": OFFICIAL_SPECIAL_TOKEN_IDS["tts_bos_token_id"],
        "tts_eos_token_id": OFFICIAL_SPECIAL_TOKEN_IDS["tts_eos_token_id"],
        "tts_pad_token_id": OFFICIAL_SPECIAL_TOKEN_IDS["tts_pad_token_id"],
        "im_start_token_id": OFFICIAL_SPECIAL_TOKEN_IDS["im_start_token_id"],
        "im_end_token_id": OFFICIAL_SPECIAL_TOKEN_IDS["im_end_token_id"],
        "system_token_id": OFFICIAL_SPECIAL_TOKEN_IDS["system_token_id"],
        "user_token_id": OFFICIAL_SPECIAL_TOKEN_IDS["user_token_id"],
        "assistant_token_id": OFFICIAL_SPECIAL_TOKEN_IDS["assistant_token_id"],
        "codec_bos_id": OFFICIAL_SPECIAL_TOKEN_IDS["talker_config.codec_bos_id"],
        "codec_nothink_id": OFFICIAL_SPECIAL_TOKEN_IDS[
            "talker_config.codec_nothink_id"
        ],
        "codec_think_bos_id": OFFICIAL_SPECIAL_TOKEN_IDS[
            "talker_config.codec_think_bos_id"
        ],
        "codec_think_eos_id": OFFICIAL_SPECIAL_TOKEN_IDS[
            "talker_config.codec_think_eos_id"
        ],
        "codec_pad_id": OFFICIAL_SPECIAL_TOKEN_IDS["talker_config.codec_pad_id"],
    }


def load_tiny_mlx_talker(tiny_checkpoint):
    return load_qwen3_omni_mlx_talker(str(tiny_checkpoint))["model"]


def make_torch_reference_prefill_builder(tiny_checkpoint):
    from transformers import Qwen3OmniMoeForConditionalGeneration

    reference = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
        str(tiny_checkpoint), dtype=torch.float32
    )
    reference.eval()
    talker = reference.talker
    model = SimpleNamespace(
        model=talker.model,
        text_projection=talker.text_projection,
        hidden_projection=talker.hidden_projection,
        get_input_embeddings=talker.get_input_embeddings,
        activation_dtype=torch.float32,
        config=talker.config,
    )
    return TalkerPrefillBuilder(
        model=model,
        model_path=str(tiny_checkpoint),
        speaker_map=OFFICIAL_SPEAKER_IDS,
        **tiny_talker_token_ids(),
    )


def make_prefill_payload_and_chunks(
    *,
    speaker: str = "Ethan",
) -> tuple[StagePayload, list[Any]]:
    ids = tiny_talker_token_ids()
    prompt_ids = torch.tensor(
        [
            ids["im_start_token_id"],
            ids["system_token_id"],
            100,
            ids["im_end_token_id"],
            ids["im_start_token_id"],
            ids["user_token_id"],
            101,
            ids["image_token_id"],
            ids["audio_token_id"],
            ids["video_token_id"],
            102,
            ids["im_end_token_id"],
            ids["im_start_token_id"],
            ids["assistant_token_id"],
        ],
        dtype=torch.long,
    )
    hidden = TINY_DIMS["thinker_hidden_size"]
    model_inputs = {
        "image_embeds": torch.full((1, hidden), 0.25, dtype=torch.float32),
        "audio_embeds": torch.full((1, hidden), -0.5, dtype=torch.float32),
        "video_embeds": torch.arange(hidden, dtype=torch.float32).reshape(1, -1)
        / hidden,
        "use_audio_in_video": False,
    }
    state = Qwen3OmniPipelineState(
        prompt={"input_ids": prompt_ids},
        thinker_inputs={
            "model_inputs": model_inputs,
            "capture_model_output_keys": [0, 24],
        },
    )
    payload = StagePayload(
        request_id="mlx-prefill",
        request=OmniRequest(inputs=[], params={"speaker": speaker}),
        data=state.to_dict(),
    )
    chunks = [
        SimpleNamespace(
            data=torch.full((hidden,), float(index), dtype=torch.float32),
            metadata={"token_id": token_id},
        )
        for index, token_id in enumerate((200, 201, 202, 203, 204, 205))
    ]
    chunks[0].data = (
        torch.arange(len(prompt_ids) * hidden, dtype=torch.float32).reshape(-1, hidden)
        / 100
    )
    chunks[0].metadata["prompt_hidden_layer"] = 24
    return payload, chunks


def make_multi_placeholder_prefill_payload_and_chunks(
    *,
    speaker: str = "Ethan",
) -> tuple[StagePayload, list[Any]]:
    ids = tiny_talker_token_ids()
    prompt_ids = torch.tensor(
        [
            ids["im_start_token_id"],
            ids["system_token_id"],
            100,
            ids["im_end_token_id"],
            ids["im_start_token_id"],
            ids["user_token_id"],
            101,
            ids["image_token_id"],
            ids["audio_token_id"],
            ids["video_token_id"],
            102,
            ids["video_token_id"],
            ids["image_token_id"],
            ids["audio_token_id"],
            103,
            ids["im_end_token_id"],
            ids["im_start_token_id"],
            ids["assistant_token_id"],
        ],
        dtype=torch.long,
    )
    hidden = TINY_DIMS["thinker_hidden_size"]

    def feature_rows(first: float, second: float) -> torch.Tensor:
        return torch.stack(
            [
                torch.arange(hidden, dtype=torch.float32) + first,
                torch.arange(hidden, dtype=torch.float32) + second,
            ]
        )

    model_inputs = {
        "image_embeds": feature_rows(100.0, 200.0),
        "audio_embeds": feature_rows(300.0, 400.0),
        "video_embeds": feature_rows(500.0, 600.0),
        "use_audio_in_video": False,
    }
    state = Qwen3OmniPipelineState(
        prompt={"input_ids": prompt_ids},
        thinker_inputs={
            "model_inputs": model_inputs,
            "capture_model_output_keys": [0, 24],
        },
    )
    payload = StagePayload(
        request_id="mlx-prefill-multi-placeholder",
        request=OmniRequest(inputs=[], params={"speaker": speaker}),
        data=state.to_dict(),
    )
    chunks = [
        SimpleNamespace(
            data=torch.full((hidden,), float(index), dtype=torch.float32),
            metadata={"token_id": token_id},
        )
        for index, token_id in enumerate((200, 201, 202, 203, 204, 205))
    ]
    chunks[0].data = (
        torch.arange(len(prompt_ids) * hidden, dtype=torch.float32).reshape(-1, hidden)
        / 100
    )
    chunks[0].metadata["prompt_hidden_layer"] = 24
    return payload, chunks


@pytest.fixture(scope="module")
def mlx_talker(tiny_checkpoint):
    return load_tiny_mlx_talker(tiny_checkpoint)


@pytest.fixture(scope="module")
def mlx_builder(tiny_checkpoint, mlx_talker):
    return Qwen3OmniMlxTalkerPrefillBuilder.from_talker(
        mlx_talker,
        model_path=str(tiny_checkpoint),
        special_token_ids=tiny_talker_token_ids(),
        speaker_map=OFFICIAL_SPEAKER_IDS,
    )


@pytest.fixture(scope="module")
def torch_builder(tiny_checkpoint):
    return make_torch_reference_prefill_builder(tiny_checkpoint)


def _stack_pending(queue: PendingTextTensorQueue) -> torch.Tensor:
    return torch.stack(list(queue), dim=0)


def _official_prompt_prefill(torch_builder, payload, chunks, *, thinker_done):
    from transformers import Qwen3OmniMoeForConditionalGeneration

    result = torch_builder.build_prompt_prefill(
        payload, chunks, thinker_done=thinker_done
    )
    state = Qwen3OmniPipelineState.from_dict(payload.data)
    ids, embed, _, _ = torch_builder._reconstruct_prompt_states(state)
    mask = torch_builder.build_multimodal_mask(ids)
    # The upstream local Torch builder still zeroes media rows. Derive the
    # golden user segment from Transformers, using the actual prompt capture.
    reference = SimpleNamespace(
        talker=torch_builder._model,
        config=SimpleNamespace(talker_config=torch_builder._model.config),
    )
    reference.talker.dtype = torch.float32
    user_end = len(ids) - 2
    result["input_embeds"][: user_end - 4] = (
        Qwen3OmniMoeForConditionalGeneration._get_talker_user_parts(
            reference, 4, user_end, mask[None], chunks[0].data[None], embed[None]
        )[0]
    )
    return result


@pytest.mark.parametrize("thinker_done", [False, True])
@pytest.mark.parametrize("speaker", ["Ethan", "Aiden"])
def test_mlx_talker_prefill_matches_torch_builder(
    mlx_builder, torch_builder, thinker_done: bool, speaker: str
) -> None:
    payload, chunks = make_prefill_payload_and_chunks(speaker=speaker)

    expected = _official_prompt_prefill(
        torch_builder, payload, chunks, thinker_done=thinker_done
    )
    actual = mlx_builder.build_prompt_prefill(
        payload, chunks, thinker_done=thinker_done
    )

    torch.testing.assert_close(actual["input_embeds"], expected["input_embeds"])
    torch.testing.assert_close(actual["input_ids"], expected["input_ids"])
    torch.testing.assert_close(
        _stack_pending(actual["pending_text_queue"]),
        _stack_pending(expected["pending_text_queue"]),
    )
    torch.testing.assert_close(actual["tts_pad_embed"], expected["tts_pad_embed"])
    torch.testing.assert_close(actual["tts_eos_embed"], expected["tts_eos_embed"])
    assert (
        actual["prompt_model_inputs"].keys() == expected["prompt_model_inputs"].keys()
    )
    for key, value in actual["prompt_model_inputs"].items():
        expected_value = expected["prompt_model_inputs"][key]
        if isinstance(value, torch.Tensor):
            torch.testing.assert_close(value, expected_value)
        else:
            assert value == expected_value


@pytest.mark.parametrize("thinker_done", [False, True])
def test_mlx_talker_prefill_scatter_places_multiple_rows_per_modality(
    mlx_builder, torch_builder, thinker_done: bool
) -> None:
    payload, chunks = make_multi_placeholder_prefill_payload_and_chunks()

    expected = _official_prompt_prefill(
        torch_builder, payload, chunks, thinker_done=thinker_done
    )
    actual = mlx_builder.build_prompt_prefill(
        payload, chunks, thinker_done=thinker_done
    )

    torch.testing.assert_close(actual["input_embeds"], expected["input_embeds"])
    torch.testing.assert_close(actual["input_ids"], expected["input_ids"])
    torch.testing.assert_close(
        _stack_pending(actual["pending_text_queue"]),
        _stack_pending(expected["pending_text_queue"]),
    )

    state = Qwen3OmniPipelineState.from_dict(payload.data)
    _, actual_embed, actual_hidden, model_inputs = (
        mlx_builder._reconstruct_prompt_states(state)
    )
    _, expected_embed, expected_hidden, _ = torch_builder._reconstruct_prompt_states(
        state
    )
    actual_embed = mlx_prefill._mlx_to_torch(actual_embed)
    actual_hidden = mlx_prefill._mlx_to_torch(actual_hidden)
    torch.testing.assert_close(actual_embed, expected_embed)
    torch.testing.assert_close(actual_hidden, expected_hidden)

    for modality, positions in {
        "image": [7, 12],
        "audio": [8, 13],
        "video": [9, 11],
    }.items():
        feature_rows = model_inputs[f"{modality}_embeds"]
        torch.testing.assert_close(actual_embed[positions], feature_rows)
        assert torch.count_nonzero(actual_hidden[positions]) == 0


@pytest.mark.parametrize("modality", ["image", "audio", "video"])
def test_mlx_talker_prefill_rejects_feature_placeholder_count_mismatch(
    mlx_builder, modality: str
) -> None:
    payload, _ = make_multi_placeholder_prefill_payload_and_chunks()
    state = Qwen3OmniPipelineState.from_dict(payload.data)
    state.thinker_inputs["model_inputs"][f"{modality}_embeds"] = state.thinker_inputs[
        "model_inputs"
    ][f"{modality}_embeds"][:1]

    with pytest.raises(
        ValueError,
        match=rf"{modality} feature rows \(1\) do not match prompt placeholders \(2\)",
    ):
        mlx_builder._reconstruct_prompt_states(state)


def test_mlx_talker_prefill_preserves_row_order_and_transport_types(
    mlx_builder,
) -> None:
    payload, chunks = make_prefill_payload_and_chunks()

    actual = mlx_builder.build_prompt_prefill(payload, chunks, thinker_done=True)

    ids = tiny_talker_token_ids()
    expected_user_ids = payload.data["prompt"]["input_ids"][4:12]
    expected_ids = torch.cat(
        [
            expected_user_ids,
            torch.full((9,), ids["tts_pad_token_id"], dtype=torch.long),
        ]
    )
    assert torch.equal(actual["input_ids"], expected_ids)
    assert actual["input_ids"].device == torch.device("cpu")
    assert actual["input_ids"].dtype == torch.long
    for key in ("input_embeds", "tts_pad_embed", "tts_eos_embed"):
        assert actual[key].device == torch.device("cpu")
        assert actual[key].dtype == torch.float32
    assert isinstance(actual["pending_text_queue"], PendingTextTensorQueue)
    assert len(actual["pending_text_queue"]) == 5
    assert all(
        row.device == torch.device("cpu") and row.dtype == torch.float32
        for row in actual["pending_text_queue"]
    )


def test_mlx_talker_prefill_includes_assistant_eos_only_when_done(mlx_builder) -> None:
    payload, chunks = make_prefill_payload_and_chunks()

    incomplete = mlx_builder.build_prompt_prefill(payload, chunks, thinker_done=False)
    completed = mlx_builder.build_prompt_prefill(payload, chunks, thinker_done=True)

    incomplete_rows = _stack_pending(incomplete["pending_text_queue"])
    completed_rows = _stack_pending(completed["pending_text_queue"])
    assert completed_rows.shape[0] == incomplete_rows.shape[0] + 1
    torch.testing.assert_close(completed_rows[:-1], incomplete_rows)
    torch.testing.assert_close(completed_rows[-1], completed["tts_eos_embed"])


def test_mlx_talker_prefill_incremental_queue_matches_torch(
    mlx_builder, torch_builder
) -> None:
    payload, chunks = make_prefill_payload_and_chunks()
    mlx_prefill_result = mlx_builder.build_prompt_prefill(
        payload, chunks, thinker_done=False
    )
    torch_prefill_result = torch_builder.build_prompt_prefill(
        payload, chunks, thinker_done=False
    )
    mlx_req = SimpleNamespace(
        thinker_chunks_done=False,
        pending_text_queue=mlx_prefill_result["pending_text_queue"],
        tts_eos_embed=mlx_prefill_result["tts_eos_embed"],
    )
    torch_req = SimpleNamespace(
        thinker_chunks_done=False,
        pending_text_queue=torch_prefill_result["pending_text_queue"],
        tts_eos_embed=torch_prefill_result["tts_eos_embed"],
    )
    next_chunk = SimpleNamespace(data=None, metadata={"token_id": 206})

    mlx_builder.append_text_chunk(mlx_req, next_chunk)
    torch_builder.append_text_chunk(torch_req, next_chunk)
    mlx_builder.append_text_chunk(
        mlx_req,
        SimpleNamespace(
            data=None,
            metadata={"token_id": tiny_talker_token_ids()["im_end_token_id"]},
        ),
    )
    torch_builder.append_text_chunk(
        torch_req,
        SimpleNamespace(
            data=None,
            metadata={"token_id": tiny_talker_token_ids()["im_end_token_id"]},
        ),
    )
    mlx_builder.mark_thinker_done(mlx_req)
    torch_builder.mark_thinker_done(torch_req)

    assert mlx_req.thinker_chunks_done is True
    torch.testing.assert_close(
        _stack_pending(mlx_req.pending_text_queue),
        _stack_pending(torch_req.pending_text_queue),
    )
    before = len(mlx_req.pending_text_queue)
    mlx_builder.mark_thinker_done(mlx_req)
    mlx_builder.append_text_chunk(mlx_req, next_chunk)
    assert len(mlx_req.pending_text_queue) == before


def test_mlx_talker_prefill_caches_thinker_embedding_rows(
    tiny_checkpoint, mlx_talker, monkeypatch
) -> None:
    calls: list[tuple[int, ...]] = []
    real_loader = mlx_prefill._load_mlx_embedding_rows

    def counting_loader(model_path: str, row_ids: list[int], **kwargs):
        calls.append(tuple(row_ids))
        return real_loader(model_path, row_ids, **kwargs)

    monkeypatch.setattr(mlx_prefill, "_load_mlx_embedding_rows", counting_loader)
    builder = Qwen3OmniMlxTalkerPrefillBuilder.from_talker(
        mlx_talker,
        model_path=str(tiny_checkpoint),
        special_token_ids=tiny_talker_token_ids(),
        speaker_map=OFFICIAL_SPEAKER_IDS,
    )
    payload, chunks = make_prefill_payload_and_chunks()

    builder.build_prompt_prefill(payload, chunks, thinker_done=True)
    first_call_count = len(calls)
    builder.build_prompt_prefill(payload, chunks, thinker_done=True)

    assert first_call_count > 0
    assert len(calls) == first_call_count
