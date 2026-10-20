# SPDX-License-Identifier: Apache-2.0
"""Native relay of unit images and audio into the thinker session, and the window over its retained history."""

import random
from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import Mock, call

import numpy as np
import pytest
import torch
from PIL import Image
from sglang.srt.layers.rotary_embedding.base import RotaryEmbedding
from sglang.srt.managers.schedule_batch import (
    FINISH_ABORT,
    FINISH_MATCHED_TOKEN,
    BaseFinishReason,
    Req,
)
from sglang.srt.mem_cache.allocator import TokenToKVPoolAllocator
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.common import release_kv_cache
from sglang.srt.mem_cache.memory_pool import (
    MHATokenToKVPool,
    NoOpMHATokenToKVPool,
    ReqToTokenPool,
)
from sglang.srt.mem_cache.radix_cache import RadixCache
from sglang.srt.session.session_controller import SessionController
from sglang.srt.session.streaming_session import StreamingSession

from sglang_omni.models.minicpm_o.components.streaming_perception import (
    MiniCPMOPerceptionState,
    PerceptionStepPlan,
)
from sglang_omni.models.minicpm_o.engine_builder import check_window_kv_cache
from sglang_omni.models.minicpm_o.native_config import (
    DEFAULT_WINDOW_SETTINGS,
    IMAGE_TOKENS,
    MiniCPMODuplexSampling,
    MiniCPMODuplexWindow,
)
from sglang_omni.models.minicpm_o.native_stages import PerceptionHooks
from sglang_omni.models.minicpm_o.session_adapters import ThinkerAdapter
from sglang_omni.models.minicpm_o.special_tokens import REQUIRED_SPECIAL_TOKENS
from sglang_omni.models.minicpm_o.thinker_state import (
    DuplexUnitRequestData,
    MiniCPMOThinkerSessionState,
)
from sglang_omni.proto.request import OmniRequest, StagePayload
from sglang_omni.proto.session import SessionIdentity, SessionOperation, TimedChunk
from sglang_omni.scheduling.session import SessionAppend
from sglang_omni.scheduling.sglang_backend.ar_session import ARSessionBridge
from sglang_omni.scheduling.sglang_backend.request_data import (
    EmbeddingSpan,
    splice_embedding_spans,
)
from sglang_omni.scheduling.sglang_backend.unit_scorers import UnitScorer
from tests.unit_test.fixtures.accelerator import require_cuda

IDENTITY = SessionIdentity("vision")
SPECIAL_TOKEN_IDS = dict(
    zip(REQUIRED_SPECIAL_TOKENS, range(1, len(REQUIRED_SPECIAL_TOKENS) + 1))
)
UNK_TOKEN_ID = 0
HEAD_DIM = 128
ROPE_THETA = 1_000_000
ROPE_POSITIONS = 8192
KV_HEAD_COUNT = 2
KV_LAYER_COUNT = 2
KV_POOL_TOKENS = 2048
WINDOW_CONTEXT_LENGTH = 1024
VOCAB_SIZE = 2048
AUDIO_ROWS = 10
EMBEDDING_WIDTH = 4
SYSTEM_TOKEN_IDS = [40, 41, 42]
IM_END_TOKEN_ID = 43
REFERENCE_ROWS = 8
REFERENCE_EMBEDDING = -1.0
PREFIX_SCHEMA = [
    ("token", len(SYSTEM_TOKEN_IDS) + 1),
    ("audio", REFERENCE_ROWS),
    ("token", 2),
]
PREFIX_TOKEN_IDS = [
    *SYSTEM_TOKEN_IDS,
    SPECIAL_TOKEN_IDS["<|audio_start|>"],
    *[UNK_TOKEN_ID] * REFERENCE_ROWS,
    SPECIAL_TOKEN_IDS["<|audio_end|>"],
    IM_END_TOKEN_ID,
]
REFERENCE_SPAN = (
    len(SYSTEM_TOKEN_IDS) + 1,
    len(SYSTEM_TOKEN_IDS) + 1 + REFERENCE_ROWS,
    REFERENCE_EMBEDDING,
)
TILE_TOKENS = IMAGE_TOKENS + 2
TEXT_TOKEN_ID = 1000


@pytest.fixture
def perception() -> MiniCPMOPerceptionState:
    tokenizer = Mock(unk_token_id=UNK_TOKEN_ID, bad_token_ids=[])
    tokenizer.convert_tokens_to_ids.side_effect = SPECIAL_TOKEN_IDS.__getitem__
    audio_encoder = Mock()
    audio_encoder.forward_streaming_batch.side_effect = lambda chunks: [
        (torch.full((10, 4), 9.0), None) for _ in chunks
    ]
    state = MiniCPMOPerceptionState(
        tokenizer=tokenizer,
        processor=Mock(),
        audio_encoder=audio_encoder,
        image_encoder=Mock(),
        max_slice_nums=1,
        mel_filter_bank=Mock(),
    )
    state.prepare_audio = Mock(return_value=np.zeros(4, dtype=np.float32))
    state.mel_chunk = Mock(return_value=Mock(batch_key=Mock(return_value=())))
    state.finish_audio = Mock()
    image = torch.full((64, 4), 6.0)
    state.prepare_image = Mock()
    state.encode_images = Mock(
        side_effect=lambda prepared_image_features: (
            (image,) if prepared_image_features else ()
        )
    )
    return state


@pytest.fixture
def hooks(perception: MiniCPMOPerceptionState) -> PerceptionHooks:
    hooks = PerceptionHooks(
        perception.tokenizer,
        Mock(),
        perception.audio_encoder,
        reference_audio=b"",
        image_encoder=Mock(),
        mel_filter_bank=Mock(
            log_mel=Mock(side_effect=lambda windows: torch.zeros(len(windows), 80, 1))
        ),
        reference_cache_capacity=1,
    )
    hooks.states[IDENTITY] = perception
    return hooks


def append_unit(
    hooks: PerceptionHooks, chunk: TimedChunk, payload: StagePayload
) -> StagePayload:
    [result] = hooks.append_batch(
        [
            SessionAppend(
                chunk=chunk,
                payload=payload,
                context=SimpleNamespace(session_identity=IDENTITY),
            )
        ]
    )
    return result


def unit_payload(
    data: PerceptionStepPlan | None = None, request_id: str = "unit"
) -> StagePayload:
    return StagePayload(
        request_id,
        OmniRequest(None, params=MiniCPMODuplexSampling().model_dump()),
        data,
    )


def make_thinker_adapter(
    tokenizer: Mock,
    vocab_size: int,
    *,
    window: MiniCPMODuplexWindow = DEFAULT_WINDOW_SETTINGS,
    device: str = "cpu",
) -> ThinkerAdapter:
    return ThinkerAdapter(
        tokenizer,
        vocab_size,
        window=window,
        rotary_embedding=RotaryEmbedding(
            HEAD_DIM, HEAD_DIM, ROPE_POSITIONS, ROPE_THETA, True, torch.float32
        ).to(device),
    )


@pytest.mark.parametrize(
    ("has_image", "first_unit"), [(False, False), (True, False), (True, True)]
)
def test_append_and_thinker_splice(
    perception: MiniCPMOPerceptionState,
    hooks: PerceptionHooks,
    has_image: bool,
    first_unit: bool,
) -> None:
    pcm = np.arange(16000, dtype="<i2").tobytes()
    chunk = TimedChunk(
        "audio", 0, 1000, 0, {"pcm": pcm, "images": [b"frame"]} if has_image else pcm
    )
    payload = unit_payload()
    result = append_unit(hooks, chunk, payload)
    assert result is payload
    np.testing.assert_array_equal(
        perception.prepare_audio.call_args.args[0],
        np.arange(16000, dtype=np.float32) / 32768,
    )
    if has_image:
        perception.prepare_image.assert_called_once_with(b"frame")
        perception.encode_images.assert_called_once_with(
            (perception.prepare_image.return_value,)
        )
    else:
        perception.prepare_image.assert_not_called()
        perception.encode_images.assert_called_once_with(())
    adapter = make_thinker_adapter(perception.tokenizer, 100)
    adapter.open(IDENTITY, payload.request)
    adapter.states[IDENTITY].is_prefix_pending = first_unit
    request = adapter.build(IDENTITY, chunk, payload)
    assert len(request.unit_embedding_spans) == (2 if has_image else 1)
    prefix_length = 0 if first_unit else 1
    for actual, planned in zip(
        request.unit_embedding_spans, payload.data["embedding_spans"]
    ):
        assert (actual.start, actual.end) == (
            planned["token_start"] + prefix_length,
            planned["token_end"] + prefix_length,
        )
    rows = splice_embedding_spans(
        torch.full((len(request.input_ids), 4), -1.0), 0, request.unit_embedding_spans
    )
    assert torch.equal(rows[-10:], torch.full((10, 4), 9.0))
    if has_image:
        assert torch.equal(
            rows[prefix_length + 2 : prefix_length + 66], torch.full((64, 4), 6.0)
        )
        assert torch.equal(rows[prefix_length + 66], torch.full((4,), -1.0))
    else:
        pass


def test_batched_sessions_keep_vision_and_audio_associated(
    perception: MiniCPMOPerceptionState, hooks: PerceptionHooks
) -> None:
    second_identity = SessionIdentity("second")
    second_state = MiniCPMOPerceptionState(
        tokenizer=perception.tokenizer,
        processor=perception.processor,
        audio_encoder=perception.audio_encoder,
        image_encoder=perception.image_encoder,
        max_slice_nums=1,
        mel_filter_bank=perception.mel_filter_bank,
    )
    second_state.prepare_audio = Mock(return_value=np.ones(4, dtype=np.float32))
    second_state.mel_chunk = Mock(return_value=Mock(batch_key=Mock(return_value=())))
    second_state.finish_audio = Mock()
    second_prepared_images = [Mock(), Mock()]
    second_state.prepare_image = Mock(side_effect=second_prepared_images)
    second_image_embeds = (
        torch.full((64, 4), 3.0),
        torch.full((64, 4), 4.0),
    )
    second_state.encode_images = Mock(return_value=second_image_embeds)
    hooks.states[second_identity] = second_state

    first_prepared_images = [Mock(), Mock()]
    first_image_embeds = (
        torch.full((64, 4), 1.0),
        torch.full((64, 4), 2.0),
    )
    perception.prepare_image.side_effect = first_prepared_images
    perception.encode_images.side_effect = None
    perception.encode_images.return_value = first_image_embeds
    perception.audio_encoder.forward_streaming_batch.side_effect = lambda chunks: [
        (torch.full((10, 4), 9.0 + index), None) for index, _ in enumerate(chunks)
    ]

    first_payload = unit_payload()
    second_payload = unit_payload()
    appends = [
        SessionAppend(
            chunk=TimedChunk(
                "audio",
                0,
                1000,
                0,
                {"pcm": b"\0\0", "images": [b"first-0", b"first-1"]},
            ),
            payload=first_payload,
            context=SimpleNamespace(session_identity=IDENTITY),
        ),
        SessionAppend(
            chunk=TimedChunk(
                "audio",
                0,
                1000,
                0,
                {"pcm": b"\0\0", "images": [b"second-0", b"second-1"]},
            ),
            payload=second_payload,
            context=SimpleNamespace(session_identity=second_identity),
        ),
    ]

    assert hooks.append_batch(appends) == [first_payload, second_payload]
    assert perception.prepare_image.call_args_list == [
        call(b"first-0"),
        call(b"first-1"),
    ]
    assert second_state.prepare_image.call_args_list == [
        call(b"second-0"),
        call(b"second-1"),
    ]
    perception.encode_images.assert_called_once_with(tuple(first_prepared_images))
    second_state.encode_images.assert_called_once_with(tuple(second_prepared_images))
    perception.audio_encoder.forward_streaming_batch.assert_called_once()
    assert len(perception.audio_encoder.forward_streaming_batch.call_args.args[0]) == 2
    perception.finish_audio.assert_called_once_with(None)
    second_state.finish_audio.assert_called_once_with(None)

    assert first_payload.data is not None
    assert second_payload.data is not None
    assert torch.equal(first_payload.data["input_embeds"][:64], first_image_embeds[0])
    assert torch.equal(
        first_payload.data["input_embeds"][64:128], first_image_embeds[1]
    )
    assert torch.equal(second_payload.data["input_embeds"][:64], second_image_embeds[0])
    assert torch.equal(
        second_payload.data["input_embeds"][64:128], second_image_embeds[1]
    )
    assert torch.equal(
        first_payload.data["input_embeds"][-10:], torch.full((10, 4), 9.0)
    )
    assert torch.equal(
        second_payload.data["input_embeds"][-10:], torch.full((10, 4), 10.0)
    )


@pytest.mark.parametrize("finish", ["complete", "abort"])
def test_image_audio_commit_atomically(
    perception: MiniCPMOPerceptionState, finish: str
) -> None:
    payload = unit_payload(
        perception.build_step_plan(
            torch.full((10, 4), 9.0), (torch.full((64, 4), 6.0),)
        )
    )
    adapter = make_thinker_adapter(perception.tokenizer, 100)
    adapter.open(IDENTITY, payload.request)
    request = adapter.build(IDENTITY, TimedChunk("audio", 0, 1000, 0, b""), payload)
    history = EmbeddingSpan(start=1, end=3, input_embeds=torch.ones(2, 4))
    unit = SimpleNamespace(
        session_identity=IDENTITY, session_request=None, embedding_spans=[]
    )
    session = SimpleNamespace(unit=unit, embedding_spans=[history])
    native = Mock(session_id=IDENTITY.id, req_nodes={})
    native.create_req.return_value = Mock(
        origin_input_ids=[7] * 12 + list(request.req.origin_input_ids), to_finish=None
    )
    bridge = ARSessionBridge.__new__(ARSessionBridge)
    bridge.adapter = adapter
    bridge.drain = Mock()
    bridge.adapter = adapter
    bridge.sessions = {IDENTITY.id: session}
    bridge.units_by_request_id = {"unit": unit}
    bridge.bridge_scheduler = Mock(max_req_len=WINDOW_CONTEXT_LENGTH - 1)
    bridge.bridge_scheduler.session_controller.get.return_value = native
    bridge.create_session_request(payload, request)
    assert session.embedding_spans == [history]
    assert [(span.start, span.end) for span in unit.embedding_spans] == [
        (14, 78),
        (79, 89),
    ]
    assert request.session_embedding_spans == [history, *unit.embedding_spans]
    assert request.req.skip_radix_cache_insert is True
    if finish == "complete":
        bridge.complete("unit")
        assert session.embedding_spans == [history, *unit.embedding_spans]
        native.abort_req.assert_not_called()
    else:
        bridge.release_append_unit("unit")
        assert session.embedding_spans == [history]
        native.abort_req.assert_called_once_with()
    assert session.unit is None


def test_empty_eos_does_not_encode(
    perception: MiniCPMOPerceptionState, hooks: PerceptionHooks
) -> None:
    payload = unit_payload()
    append_unit(hooks, TimedChunk("audio", 0, 0, 1, None, eos=True), payload)
    assert payload.data is None
    perception.prepare_audio.assert_not_called()
    perception.prepare_image.assert_not_called()
    perception.encode_images.assert_not_called()


def test_image_encoder_failure_propagates(
    perception: MiniCPMOPerceptionState, hooks: PerceptionHooks
) -> None:
    perception.encode_images.side_effect = ValueError("image encoder failed")
    payload = unit_payload()
    with pytest.raises(ValueError, match="image encoder failed"):
        append_unit(
            hooks,
            TimedChunk("audio", 0, 1000, 0, {"pcm": b"\0\0", "images": [b"frame"]}),
            payload,
        )
    perception.prepare_image.assert_called_once_with(b"frame")


@pytest.mark.parametrize(
    "error",
    [
        ValueError("bad frame"),
        Image.DecompressionBombError("big"),
    ],
)
def test_undecodable_frame_is_dropped_and_siblings_kept(
    perception: MiniCPMOPerceptionState, hooks: PerceptionHooks, error: Exception
) -> None:
    first = torch.full((64, 4), 3.0)
    last = torch.full((64, 4), 7.0)
    first_prepared = Mock()
    last_prepared = Mock()
    perception.prepare_image.side_effect = [first_prepared, error, last_prepared]
    perception.encode_images.side_effect = lambda _: (first, last)
    payload = unit_payload()
    append_unit(
        hooks,
        TimedChunk(
            "audio", 0, 1000, 0, {"pcm": b"\0\0", "images": [b"first", b"bad", b"last"]}
        ),
        payload,
    )
    perception.encode_images.assert_called_once_with((first_prepared, last_prepared))
    spans = payload.data["embedding_spans"]
    assert [span["modality"] for span in spans] == ["image", "image", "audio"]
    assert torch.equal(payload.data["input_embeds"][:128], torch.cat([first, last]))


@dataclass(frozen=True, kw_only=True)
class WindowUnit:
    tile_count: int
    generated_token_ids: list[int]


CUT_WINDOW = MiniCPMODuplexWindow(high_tokens=120, low_tokens=90)
CUT_UNITS = [
    WindowUnit(
        tile_count=0,
        generated_token_ids=[
            TEXT_TOKEN_ID,
            TEXT_TOKEN_ID + 1,
            SPECIAL_TOKEN_IDS["<|chunk_eos|>"],
        ],
    ),
    WindowUnit(tile_count=1, generated_token_ids=[SPECIAL_TOKEN_IDS["<|listen|>"]]),
    WindowUnit(
        tile_count=0,
        generated_token_ids=[
            TEXT_TOKEN_ID + 2,
            TEXT_TOKEN_ID + 3,
            TEXT_TOKEN_ID + 4,
            SPECIAL_TOKEN_IDS["<|chunk_eos|>"],
        ],
    ),
    WindowUnit(tile_count=0, generated_token_ids=[SPECIAL_TOKEN_IDS["<|listen|>"]]),
]
CUT_UNIT_INDEX = 3
CUT_START = len(PREFIX_TOKEN_IDS)
CUT_LENGTH = 94
FIT_MAX_REQUEST_LENGTH = 200
KEPT_KEY_COUNT = 64
KEPT_KEY_START = 300


class WindowScheduler:
    """The bridge's scheduler surface over real SGLang session, request-row and KV pools."""

    def __init__(
        self, device: str, max_req_len: int, kv_dtype: torch.dtype = torch.bfloat16
    ) -> None:
        self.req_to_token_pool = ReqToTokenPool(
            size=1,
            max_context_len=WINDOW_CONTEXT_LENGTH,
            device=device,
            enable_memory_saver=False,
        )
        self.kv_cache = MHATokenToKVPool(
            size=KV_POOL_TOKENS,
            page_size=1,
            dtype=kv_dtype,
            head_num=KV_HEAD_COUNT,
            head_dim=HEAD_DIM,
            layer_num=KV_LAYER_COUNT,
            device=device,
            enable_memory_saver=False,
        )
        self.token_to_kv_pool_allocator = TokenToKVPoolAllocator(
            size=KV_POOL_TOKENS,
            dtype=kv_dtype,
            device=device,
            kvcache=self.kv_cache,
            need_sort=False,
        )
        self.tree_cache = StreamingSession(
            RadixCache(
                CacheInitParams(
                    disable=False,
                    req_to_token_pool=self.req_to_token_pool,
                    token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
                    page_size=1,
                    enable_kv_cache_events=False,
                    eviction_policy="lru",
                )
            )
        )
        self.session_controller = SessionController(self.tree_cache)
        self.model_config = SimpleNamespace(vocab_size=VOCAB_SIZE)
        self.waiting_queue: list[Req] = []
        self.chunked_req: Req | None = None
        self.max_running_requests = 1
        self.max_req_len = max_req_len
        self.generator = torch.Generator().manual_seed(0)

    def synchronize_launched_decode(self) -> None:
        pass

    def resolve_pending_async(self) -> None:
        pass

    def run(
        self,
        request: Req,
        generated_token_ids: list[int],
        finished_reason: BaseFinishReason,
    ) -> int:
        """Prefill and decode one request on the pools as SGLang does, then release it; returns the KV length it reused."""
        request.init_next_round_input(self.tree_cache)
        if request.kv.holds_kv:
            pass
        else:
            self.req_to_token_pool.alloc([request])
        reused_length = len(request.prefix_indices)
        kv_length = (
            len(request.full_untruncated_fill_ids) + len(generated_token_ids) - 1
        )
        slots = self.token_to_kv_pool_allocator.alloc(kv_length - reused_length)
        self.req_to_token_pool.req_to_token[
            request.kv.req_pool_idx, reused_length:kv_length
        ] = slots.to(torch.int32)
        for layer_id in range(KV_LAYER_COUNT):
            for buffer in (
                self.kv_cache.get_key_buffer(layer_id),
                self.kv_cache.get_value_buffer(layer_id),
            ):
                storage_rows(buffer)[slots] = storage_rows(
                    torch.randn(
                        (len(slots), KV_HEAD_COUNT, HEAD_DIM), generator=self.generator
                    ).to(buffer)
                )
        request.kv.kv_allocated_len = kv_length
        request.kv.kv_committed_len = kv_length
        request.output_ids.extend(generated_token_ids)
        request.finished_reason = finished_reason
        release_kv_cache(request, self.tree_cache)
        return reused_length


def storage_rows(buffer: torch.Tensor) -> torch.Tensor:
    """The rows SGLang stores: uint8 behind an FP8 cache's float8 view, which indexing ops do not all take."""
    return buffer.view(torch.uint8) if buffer.dtype.itemsize == 1 else buffer


def open_window_session(
    perception: MiniCPMOPerceptionState,
    window: MiniCPMODuplexWindow,
    *,
    device: str = "cpu",
    max_req_len: int = WINDOW_CONTEXT_LENGTH - 1,
    kv_dtype: torch.dtype = torch.bfloat16,
) -> tuple[WindowScheduler, ARSessionBridge, ThinkerAdapter]:
    perception.prefix_token_ids = list(PREFIX_TOKEN_IDS)
    perception.prefix_schema = list(PREFIX_SCHEMA)
    perception.prefix_embeds = torch.full(
        (REFERENCE_ROWS, EMBEDDING_WIDTH), REFERENCE_EMBEDDING
    )
    scheduler = WindowScheduler(device, max_req_len, kv_dtype)
    adapter = make_thinker_adapter(
        perception.tokenizer, VOCAB_SIZE, window=window, device=device
    )
    bridge = ARSessionBridge(scheduler, adapter)
    bridge.apply_operation(
        unit_payload(), SessionOperation("open", IDENTITY, ("thinker",))
    )
    return scheduler, bridge, adapter


def append_window_unit(
    perception: MiniCPMOPerceptionState,
    bridge: ARSessionBridge,
    adapter: ThinkerAdapter,
    unit_index: int,
    unit: WindowUnit,
) -> DuplexUnitRequestData:
    perception.audio_chunk_index = 1 if unit_index == 0 else 2
    payload = unit_payload(
        perception.build_step_plan(
            torch.full((AUDIO_ROWS, EMBEDDING_WIDTH), float(unit_index)),
            tuple(
                torch.full((IMAGE_TOKENS, EMBEDDING_WIDTH), unit_index + 0.5)
                for _ in range(unit.tile_count)
            ),
        ),
        request_id=f"unit-{unit_index}",
    )
    chunk = TimedChunk("audio", 1000 * unit_index, 1000, unit_index, b"")
    bridge.accept(payload, SessionOperation("append", IDENTITY, ("thinker",), chunk))
    request_data = adapter.build(IDENTITY, chunk, payload)
    bridge.create_session_request(payload, request_data)
    return request_data


def finish_window_unit(
    scheduler: WindowScheduler,
    bridge: ARSessionBridge,
    adapter: ThinkerAdapter,
    request_data: DuplexUnitRequestData,
    unit: WindowUnit,
) -> int:
    reused_length = scheduler.run(
        request_data.req,
        unit.generated_token_ids,
        FINISH_MATCHED_TOKEN(matched=unit.generated_token_ids[-1]),
    )
    request_data.output_ids = list(request_data.req.output_ids)
    adapter.result(IDENTITY, request_data)
    bridge.complete(request_data.req.rid)
    return reused_length


def finish_window_units(
    perception: MiniCPMOPerceptionState,
    scheduler: WindowScheduler,
    bridge: ARSessionBridge,
    adapter: ThinkerAdapter,
    units: list[WindowUnit],
) -> None:
    for unit_index, unit in enumerate(units):
        finish_window_unit(
            scheduler,
            bridge,
            adapter,
            append_window_unit(perception, bridge, adapter, unit_index, unit),
            unit,
        )


def unit_token_ids(unit: WindowUnit) -> list[int]:
    """The ids a finished unit holds in the session, from its <unit> to its </unit>."""
    tile_token_ids = [
        SPECIAL_TOKEN_IDS["<image>"],
        *[UNK_TOKEN_ID] * IMAGE_TOKENS,
        SPECIAL_TOKEN_IDS["</image>"],
    ]
    return [
        SPECIAL_TOKEN_IDS["<unit>"],
        *tile_token_ids * unit.tile_count,
        *[UNK_TOKEN_ID] * AUDIO_ROWS,
        *unit.generated_token_ids,
        SPECIAL_TOKEN_IDS["</unit>"],
    ]


def expected_session_layout(
    kept_units: list[tuple[int, WindowUnit]], unit_index: int, unit: WindowUnit
) -> tuple[list[int], list[tuple[int, int, float]]]:
    """Ids and (start, end, embedding) spans of a unit request whose history holds kept_units."""
    token_ids = list(PREFIX_TOKEN_IDS)
    spans = [REFERENCE_SPAN]
    for index, kept_unit in [*kept_units, (unit_index, unit)]:
        unit_start = len(token_ids)
        spans.extend(
            (
                unit_start + 2 + tile * TILE_TOKENS,
                unit_start + 2 + tile * TILE_TOKENS + IMAGE_TOKENS,
                index + 0.5,
            )
            for tile in range(kept_unit.tile_count)
        )
        audio_start = unit_start + 1 + kept_unit.tile_count * TILE_TOKENS
        spans.append((audio_start, audio_start + AUDIO_ROWS, float(index)))
        token_ids.extend(unit_token_ids(kept_unit))
    # note (0xtoward): The new unit's request ends at its media; its outputs and </unit> come later.
    return token_ids[: audio_start + AUDIO_ROWS], spans


def observed_session_layout(
    request_data: DuplexUnitRequestData,
) -> tuple[list[int], list[tuple[int, int, float]]]:
    return list(request_data.req.origin_input_ids), [
        (span.start, span.end, float(span.input_embeds[0, 0]))
        for span in request_data.session_embedding_spans
    ]


def checkpoint_kept_units(
    preserved_length: int, unit_lengths: list[int], high_tokens: int, low_tokens: int
) -> list[list[int]]:
    """Units the checkpoint's basic window keeps after each unit, from its StreamDecoder.enforce_window."""
    history: list[tuple[int, int]] = []
    cache_length = preserved_length
    kept_after_each_unit = []
    for unit_index, unit_length in enumerate(unit_lengths):
        history.append((unit_index, unit_length))
        cache_length += unit_length
        if cache_length > high_tokens:
            while cache_length > low_tokens and history:
                _, dropped_length = history.pop(0)
                cache_length -= dropped_length
        else:
            pass
        kept_after_each_unit.append([index for index, _ in history])
    return kept_after_each_unit


def rotate_keys_exactly(keys: torch.Tensor, distance: int) -> torch.Tensor:
    """Neox RoPE by -distance positions in float64."""
    angles = distance * ROPE_THETA ** (
        -torch.arange(0, HEAD_DIM, 2, dtype=torch.float64) / HEAD_DIM
    )
    first_half, second_half = keys.to(torch.float64).chunk(2, dim=-1)
    return torch.cat(
        (
            first_half * angles.cos() + second_half * angles.sin(),
            second_half * angles.cos() - first_half * angles.sin(),
        ),
        dim=-1,
    )


def same_bits(values: torch.Tensor, reference: torch.Tensor) -> bool:
    return torch.equal(
        values.contiguous().view(torch.uint8), reference.contiguous().view(torch.uint8)
    )


def relative_error(values: torch.Tensor, reference: torch.Tensor) -> float:
    return float((values.to(torch.float64) - reference).norm() / reference.norm())


def read_session_kv(
    scheduler: WindowScheduler, length: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Keys and values of the session's first length positions, stacked over layers."""
    slots = scheduler.req_to_token_pool.req_to_token[
        scheduler.tree_cache.slots[IDENTITY.id].kv.req_pool_idx, :length
    ].to(torch.int64)
    return tuple(
        torch.stack(
            [
                storage_rows(buffer(layer_id))[slots].view(buffer(layer_id).dtype)
                for layer_id in range(KV_LAYER_COUNT)
            ]
        ).cpu()
        for buffer in (
            scheduler.kv_cache.get_key_buffer,
            scheduler.kv_cache.get_value_buffer,
        )
    )


def kv_buffer_addresses(scheduler: WindowScheduler) -> list[int]:
    return [
        buffer.data_ptr()
        for buffer in (
            *scheduler.kv_cache.k_buffer,
            *scheduler.kv_cache.v_buffer,
            scheduler.req_to_token_pool.req_to_token,
        )
    ]


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_window_keeps_the_units_the_checkpoint_window_keeps(
    perception: MiniCPMOPerceptionState, seed: int
) -> None:
    """Before every unit the session ids and spans hold exactly the units the checkpoint's basic window kept."""
    unit_count = 40
    window = MiniCPMODuplexWindow(high_tokens=256, low_tokens=192)
    random_source = random.Random(seed)
    units = [
        WindowUnit(
            tile_count=random_source.choice([0, 0, 0, 1, 2]),
            generated_token_ids=(
                [SPECIAL_TOKEN_IDS["<|listen|>"]]
                if random_source.random() < 0.5
                else [TEXT_TOKEN_ID + unit_index] * random_source.randint(1, 19)
                + [SPECIAL_TOKEN_IDS["<|chunk_eos|>"]]
            ),
        )
        for unit_index in range(unit_count)
    ]
    kept_after_each_unit = checkpoint_kept_units(
        len(PREFIX_TOKEN_IDS),
        [len(unit_token_ids(unit)) for unit in units],
        window.high_tokens,
        window.low_tokens,
    )
    assert (
        sum(
            later[0] > earlier[0]
            for earlier, later in zip(kept_after_each_unit, kept_after_each_unit[1:])
        )
        >= 3
    ), "the case drops units at least three times"
    scheduler, bridge, adapter = open_window_session(perception, window)

    layouts = []
    for unit_index, unit in enumerate(units):
        request_data = append_window_unit(perception, bridge, adapter, unit_index, unit)
        layouts.append(observed_session_layout(request_data))
        finish_window_unit(scheduler, bridge, adapter, request_data, unit)

    for unit_index in range(1, unit_count):
        kept_units = [
            (index, units[index]) for index in kept_after_each_unit[unit_index - 1]
        ]
        assert layouts[unit_index] == expected_session_layout(
            kept_units, unit_index, units[unit_index]
        ), f"unit {unit_index} history differs from the checkpoint's window"


@pytest.mark.parametrize(
    "unit_lengths",
    [[13] * 154, [13, 33, 296, 81, 25] * 13],
    ids=["listening", "mixed"],
)
def test_window_key_shift_rounds_the_exact_rotation_once(
    perception: MiniCPMOPerceptionState, unit_lengths: list[int]
) -> None:
    """One float32 rotation by the dropped length stays at bf16 storage precision and beats the checkpoint's per-unit bf16 realignment."""
    adapter = make_thinker_adapter(perception.tokenizer, VOCAB_SIZE)
    distance = sum(unit_lengths)
    keys = torch.randn(
        (KEPT_KEY_COUNT, KV_HEAD_COUNT, HEAD_DIM),
        generator=torch.Generator().manual_seed(0),
    ).to(torch.bfloat16)
    exact_keys = rotate_keys_exactly(keys, distance)
    inverse_frequencies = 1.0 / (
        ROPE_THETA ** (torch.arange(0, HEAD_DIM, 2, dtype=torch.float32) / HEAD_DIM)
    )
    checkpoint_keys = keys
    positions = torch.arange(
        KEPT_KEY_START + distance, KEPT_KEY_START + distance + KEPT_KEY_COUNT
    )
    for unit_length in unit_lengths:
        old_angles = torch.outer(positions.to(torch.float32), inverse_frequencies)
        new_angles = torch.outer(
            (positions - unit_length).to(torch.float32), inverse_frequencies
        )
        old_cosine, old_sine, new_cosine, new_sine = (
            torch.cat((angles, angles), dim=-1)[:, None].to(torch.bfloat16)
            for angles in (
                old_angles.cos(),
                old_angles.sin(),
                new_angles.cos(),
                new_angles.sin(),
            )
        )
        first_half, second_half = checkpoint_keys.chunk(2, dim=-1)
        unrotated = old_cosine * checkpoint_keys - old_sine * torch.cat(
            (-second_half, first_half), dim=-1
        )
        first_half, second_half = unrotated.chunk(2, dim=-1)
        checkpoint_keys = new_cosine * unrotated + new_sine * torch.cat(
            (-second_half, first_half), dim=-1
        )
        positions = positions - unit_length

    shifted_keys = adapter.shift_key_positions(keys, distance)

    shift_error = relative_error(shifted_keys, exact_keys)
    assert shifted_keys.dtype == keys.dtype, "the shift keeps the cache dtype"
    # note (0xtoward): One rounding errs by at most half a bf16 step and the floor by half a step over sqrt(3) on average, so one rounding stays under twice the floor.
    assert shift_error <= 2 * relative_error(
        exact_keys.to(torch.bfloat16), exact_keys
    ), f"shift error {shift_error:.2e} exceeds two bf16 roundings"
    assert shift_error <= relative_error(
        checkpoint_keys, exact_keys
    ), f"shift error {shift_error:.2e} exceeds the checkpoint's realignment error"


@pytest.mark.parametrize(
    "device", ["cpu", pytest.param("cuda", marks=pytest.mark.accelerator)]
)
@pytest.mark.parametrize(
    "kv_dtype", [torch.bfloat16, torch.float8_e4m3fn], ids=["bf16", "fp8"]
)
def test_window_cut_moves_retained_kv_ids_spans_and_unit_starts(
    perception: MiniCPMOPerceptionState, device: str, kv_dtype: torch.dtype
) -> None:
    """A cut drops the oldest units' slots in place and moves every later position down by the cut length."""
    if device == "cuda":
        require_cuda()
    else:
        pass
    scheduler, bridge, adapter = open_window_session(
        perception, CUT_WINDOW, device=device, kv_dtype=kv_dtype
    )
    finish_window_units(
        perception, scheduler, bridge, adapter, CUT_UNITS[:CUT_UNIT_INDEX]
    )
    kept_kv_end = scheduler.tree_cache.slots[IDENTITY.id].kv.kv_committed_len
    keys_before, values_before = read_session_kv(scheduler, kept_kv_end)
    free_slots_before = scheduler.token_to_kv_pool_allocator.available_size()
    buffer_addresses = kv_buffer_addresses(scheduler)
    history_length = (
        len(PREFIX_TOKEN_IDS)
        + sum(len(unit_token_ids(unit)) for unit in CUT_UNITS[:CUT_UNIT_INDEX])
        - 1
    )

    request_data = append_window_unit(
        perception, bridge, adapter, CUT_UNIT_INDEX, CUT_UNITS[CUT_UNIT_INDEX]
    )
    unit_starts_after_cut = list(adapter.states[IDENTITY].unit_starts)
    free_slots_after_cut = scheduler.token_to_kv_pool_allocator.available_size()
    keys_after, values_after = read_session_kv(scheduler, kept_kv_end - CUT_LENGTH)
    reused_length = finish_window_unit(
        scheduler, bridge, adapter, request_data, CUT_UNITS[CUT_UNIT_INDEX]
    )

    expected_ids, expected_spans = expected_session_layout(
        [(2, CUT_UNITS[2])], CUT_UNIT_INDEX, CUT_UNITS[CUT_UNIT_INDEX]
    )
    assert observed_session_layout(request_data) == (expected_ids, expected_spans)
    assert (
        list(request_data.req.full_untruncated_fill_ids) == expected_ids
    ), "the prefill ids are the cut history"
    assert request_data.session_history_length == history_length - CUT_LENGTH
    assert unit_starts_after_cut == [CUT_START]
    assert adapter.states[IDENTITY].unit_starts == [
        CUT_START,
        CUT_START + len(unit_token_ids(CUT_UNITS[2])),
    ], "the finished unit starts where the cut moved it"
    assert (
        free_slots_after_cut == free_slots_before + CUT_LENGTH
    ), "the cut returns one slot per dropped token"
    assert reused_length == kept_kv_end - CUT_LENGTH, "the prefill reuses all kept KV"
    assert same_bits(keys_after[:, :CUT_START], keys_before[:, :CUT_START])
    assert same_bits(values_after[:, :CUT_START], values_before[:, :CUT_START])
    assert same_bits(
        values_after[:, CUT_START:], values_before[:, CUT_START + CUT_LENGTH :]
    ), "kept values move down unchanged"
    exact_keys = rotate_keys_exactly(
        keys_before[:, CUT_START + CUT_LENGTH :], CUT_LENGTH
    )
    assert relative_error(keys_after[:, CUT_START:], exact_keys) <= 2 * relative_error(
        exact_keys.to(kv_dtype), exact_keys
    ), "kept keys move down rotated by the cut length"
    assert (
        kv_buffer_addresses(scheduler) == buffer_addresses
    ), "the cut keeps every buffer a captured graph reads"


def test_window_cut_history_rebuilds_after_an_aborted_unit(
    perception: MiniCPMOPerceptionState,
) -> None:
    """After a cut and an aborted unit, the next unit re-prefills the cut ids under shifted spans."""
    next_unit = CUT_UNITS[CUT_UNIT_INDEX]
    scheduler, bridge, adapter = open_window_session(perception, CUT_WINDOW)
    finish_window_units(
        perception, scheduler, bridge, adapter, CUT_UNITS[:CUT_UNIT_INDEX]
    )
    aborted_request_data = append_window_unit(
        perception, bridge, adapter, CUT_UNIT_INDEX, next_unit
    )
    scheduler.run(
        aborted_request_data.req, next_unit.generated_token_ids, FINISH_ABORT()
    )
    bridge.release_append_unit(aborted_request_data.req.rid)

    request_data = append_window_unit(
        perception, bridge, adapter, CUT_UNIT_INDEX + 1, next_unit
    )
    reused_length = finish_window_unit(
        scheduler, bridge, adapter, request_data, next_unit
    )

    assert reused_length == 0, "the abort released the session KV"
    assert observed_session_layout(request_data) == expected_session_layout(
        [(2, CUT_UNITS[2])], CUT_UNIT_INDEX + 1, next_unit
    )


@pytest.mark.parametrize(
    ("mode", "kept_unit_indices", "fits"),
    [("basic", [2], True), ("off", [0, 1, 2], False)],
)
def test_window_cuts_below_high_tokens_when_the_next_unit_would_not_fit(
    perception: MiniCPMOPerceptionState,
    mode: str,
    kept_unit_indices: list[int],
    fits: bool,
) -> None:
    window = MiniCPMODuplexWindow(mode=mode, high_tokens=1000, low_tokens=90)
    vision_unit = WindowUnit(
        tile_count=1, generated_token_ids=[SPECIAL_TOKEN_IDS["<|listen|>"]]
    )
    scheduler, bridge, adapter = open_window_session(
        perception, window, max_req_len=FIT_MAX_REQUEST_LENGTH
    )
    finish_window_units(
        perception, scheduler, bridge, adapter, CUT_UNITS[:CUT_UNIT_INDEX]
    )

    request_data = append_window_unit(
        perception, bridge, adapter, CUT_UNIT_INDEX, vision_unit
    )

    assert observed_session_layout(request_data) == expected_session_layout(
        [(index, CUT_UNITS[index]) for index in kept_unit_indices],
        CUT_UNIT_INDEX,
        vision_unit,
    )
    assert (
        len(request_data.req.origin_input_ids)
        + request_data.req.sampling_params.max_new_tokens
        <= FIT_MAX_REQUEST_LENGTH - 1
    ) is fits


class FixedUnitScorer(UnitScorer):
    """Importance given up front, oldest unit first, so a test chooses what a scored cut drops."""

    layer_stride = 1

    def __init__(self, importance: list[float]) -> None:
        self.importance = importance

    def score(self, keys: torch.Tensor, unit_lengths: list[int]) -> torch.Tensor:
        return torch.tensor(self.importance[: len(unit_lengths)], device=keys.device)


KEYDIFF_WINDOW_ABOVE_TEST_HISTORY = MiniCPMODuplexWindow(
    mode="keydiff", high_tokens=1000, low_tokens=900, protect_tokens=0
)
SCORED_UNITS = [CUT_UNITS[index % len(CUT_UNITS)] for index in range(8)]
SCORED_IMPORTANCE = [0.9, 0.1, 0.8, 0.2, 0.7, 0.6]
SCORED_DROPPED = (1, 3)


@pytest.mark.parametrize(
    ("importance", "protect_tokens", "dropped_ranges", "kept_starts"),
    [
        ([0.9, 0.1, 0.2, 0.8], 25, [(20, 40)], [10, 20, 30, 40]),
        ([0.1, 0.9, 0.2, 0.8], 25, [(10, 20), (30, 40)], [10, 20, 30, 40]),
        ([0.5, 0.4], 45, [(10, 40)], [10, 20, 30]),
    ],
    ids=["adjacent-units-merge", "apart", "protected-fallback"],
)
def test_scored_window_drops_the_least_important_units_first(
    importance: list[float],
    protect_tokens: int,
    dropped_ranges: list[tuple[int, int]],
    kept_starts: list[int],
) -> None:
    """Five finished 10-token units start at 10..50 and the newest at 60; the next unit starts at 70."""
    next_unit_start = 70
    state = MiniCPMOThinkerSessionState(sampling=MiniCPMODuplexSampling())
    state.unit_starts = [10, 20, 30, 40, 50, 60]
    target_start = next_unit_start - sum(end - start for start, end in dropped_ranges)

    scoreable_count = state.scoreable_unit_count(next_unit_start, protect_tokens)
    cut = state.drop_scored_units(next_unit_start, target_start, importance)

    assert scoreable_count == len(
        importance
    ), "units within protect_tokens are not scored"
    assert cut is not None and list(cut.ranges) == dropped_ranges, "dropped ranges"
    assert (
        state.unit_starts == kept_starts
    ), "kept units move down by what went before them"


@pytest.mark.parametrize(
    "device", ["cpu", pytest.param("cuda", marks=pytest.mark.accelerator)]
)
@pytest.mark.parametrize(
    "kv_dtype", [torch.bfloat16, torch.float8_e4m3fn], ids=["bf16", "fp8"]
)
def test_scored_window_cut_moves_each_kept_stretch_by_what_was_dropped_before_it(
    perception: MiniCPMOPerceptionState, device: str, kv_dtype: torch.dtype
) -> None:
    """Dropping units 1 and 3 moves unit 2 down by unit 1 and everything after unit 3 down by both."""
    if device == "cuda":
        require_cuda()
    else:
        pass
    scheduler, bridge, adapter = open_window_session(
        perception,
        KEYDIFF_WINDOW_ABOVE_TEST_HISTORY,
        device=device,
        kv_dtype=kv_dtype,
    )
    finish_window_units(perception, scheduler, bridge, adapter, SCORED_UNITS[:7])
    state = adapter.states[IDENTITY]
    starts = list(state.unit_starts)
    lengths = [len(unit_token_ids(unit)) for unit in SCORED_UNITS[:7]]
    first, second = SCORED_DROPPED
    cut_length = lengths[first] + lengths[second]
    next_unit_start = starts[6] + lengths[6]
    adapter.window = MiniCPMODuplexWindow(
        mode="keydiff",
        high_tokens=next_unit_start - 1,
        low_tokens=next_unit_start - cut_length,
        protect_tokens=0,
    )
    adapter.unit_scorer = FixedUnitScorer(SCORED_IMPORTANCE)
    kv_end = scheduler.tree_cache.slots[IDENTITY.id].kv.kv_committed_len
    keys_before, values_before = read_session_kv(scheduler, kv_end)
    free_slots_before = scheduler.token_to_kv_pool_allocator.available_size()
    buffer_addresses = kv_buffer_addresses(scheduler)

    request_data = append_window_unit(perception, bridge, adapter, 7, SCORED_UNITS[7])
    keys_after, values_after = read_session_kv(scheduler, kv_end - cut_length)

    kept_units = [
        (index, SCORED_UNITS[index])
        for index in range(7)
        if index not in SCORED_DROPPED
    ]
    assert observed_session_layout(request_data) == expected_session_layout(
        kept_units, 7, SCORED_UNITS[7]
    )
    assert state.unit_starts == [
        starts[0],
        starts[2] - lengths[first],
        *(starts[index] - cut_length for index in (4, 5, 6)),
    ]
    assert (
        scheduler.token_to_kv_pool_allocator.available_size()
        == free_slots_before + cut_length
    ), "the cut returns one slot per dropped token"
    kept_stretches = [
        (0, starts[first], 0),
        (starts[first + 1], starts[second], lengths[first]),
        (starts[second + 1], kv_end, cut_length),
    ]
    for source_start, source_end, distance in kept_stretches:
        source = slice(source_start, source_end)
        target = slice(source_start - distance, source_end - distance)
        assert same_bits(values_after[:, target], values_before[:, source])
        if distance == 0:
            assert same_bits(keys_after[:, target], keys_before[:, source])
        else:
            exact_keys = rotate_keys_exactly(keys_before[:, source], distance)
            assert relative_error(
                keys_after[:, target], exact_keys
            ) <= 2 * relative_error(
                exact_keys.to(kv_dtype), exact_keys
            ), f"keys after the cut move down by {distance}, rotated once"
    assert (
        kv_buffer_addresses(scheduler) == buffer_addresses
    ), "the cut keeps every buffer a captured graph reads"


@pytest.mark.parametrize(
    "kv_dtype", [torch.bfloat16, torch.float8_e4m3fn], ids=["bf16", "fp8"]
)
def test_keydiff_window_keeps_the_unit_whose_cached_keys_stand_out(
    perception: MiniCPMOPerceptionState, kv_dtype: torch.dtype
) -> None:
    """Five units whose position-free keys share one direction go first; unit 2 points elsewhere and stays."""
    rare_unit_index = 2
    scheduler, bridge, adapter = open_window_session(
        perception,
        KEYDIFF_WINDOW_ABOVE_TEST_HISTORY,
        kv_dtype=kv_dtype,
    )
    finish_window_units(perception, scheduler, bridge, adapter, SCORED_UNITS[:7])
    starts = list(adapter.states[IDENTITY].unit_starts)
    lengths = [len(unit_token_ids(unit)) for unit in SCORED_UNITS[:7]]
    common_direction, rare_direction = torch.linalg.qr(
        torch.randn((HEAD_DIM, 2), generator=torch.Generator().manual_seed(0))
    )[0].T
    slots = scheduler.req_to_token_pool.req_to_token[
        scheduler.tree_cache.slots[IDENTITY.id].kv.req_pool_idx
    ].to(torch.int64)
    for unit_index in range(6):
        direction = (
            rare_direction if unit_index == rare_unit_index else common_direction
        )
        for position in range(starts[unit_index], starts[unit_index + 1]):
            for layer_id in range(KV_LAYER_COUNT):
                key_buffer = scheduler.kv_cache.get_key_buffer(layer_id)
                storage_rows(key_buffer)[slots[position]] = storage_rows(
                    rotate_keys_exactly(
                        direction.expand(KV_HEAD_COUNT, HEAD_DIM), -position
                    ).to(key_buffer)
                )
    next_unit_start = starts[6] + lengths[6]
    common_length = sum(
        length for index, length in enumerate(lengths[:6]) if index != rare_unit_index
    )
    adapter.window = MiniCPMODuplexWindow(
        mode="keydiff",
        high_tokens=next_unit_start - 1,
        low_tokens=next_unit_start - common_length,
        protect_tokens=0,
    )

    request_data = append_window_unit(perception, bridge, adapter, 7, SCORED_UNITS[7])

    assert observed_session_layout(request_data) == expected_session_layout(
        [(index, SCORED_UNITS[index]) for index in (rare_unit_index, 6)],
        7,
        SCORED_UNITS[7],
    ), "the cut keeps the unit whose keys stand out and the newest unit"


def test_keydiff_window_drops_the_oldest_units_once_an_abort_released_the_session_kv(
    perception: MiniCPMOPerceptionState,
) -> None:
    """With no cached keys left to score, a keydiff cut drops the oldest units as the basic window does."""
    next_unit = CUT_UNITS[CUT_UNIT_INDEX]
    scheduler, bridge, adapter = open_window_session(
        perception, KEYDIFF_WINDOW_ABOVE_TEST_HISTORY
    )
    finish_window_units(
        perception, scheduler, bridge, adapter, CUT_UNITS[:CUT_UNIT_INDEX]
    )
    aborted_request_data = append_window_unit(
        perception, bridge, adapter, CUT_UNIT_INDEX, next_unit
    )
    scheduler.run(
        aborted_request_data.req, next_unit.generated_token_ids, FINISH_ABORT()
    )
    bridge.release_append_unit(aborted_request_data.req.rid)
    adapter.window = MiniCPMODuplexWindow(
        mode="keydiff",
        high_tokens=CUT_WINDOW.high_tokens,
        low_tokens=CUT_WINDOW.low_tokens,
        protect_tokens=0,
    )

    request_data = append_window_unit(
        perception, bridge, adapter, CUT_UNIT_INDEX + 1, next_unit
    )
    reused_length = finish_window_unit(
        scheduler, bridge, adapter, request_data, next_unit
    )

    assert reused_length == 0, "the abort released the session KV"
    assert observed_session_layout(request_data) == expected_session_layout(
        [(2, CUT_UNITS[2])], CUT_UNIT_INDEX + 1, next_unit
    ), "the oldest units go"


@pytest.mark.parametrize(
    ("window", "pool_type", "kv_dtype", "use_hnd", "error"),
    [
        (CUT_WINDOW, MHATokenToKVPool, torch.bfloat16, False, None),
        (CUT_WINDOW, MHATokenToKVPool, torch.float8_e4m3fn, False, None),
        (CUT_WINDOW, MHATokenToKVPool, torch.bfloat16, True, "NHD KV cache"),
        (CUT_WINDOW, NoOpMHATokenToKVPool, torch.bfloat16, False, "token-row KV cache"),
        (
            MiniCPMODuplexWindow(mode="off"),
            MHATokenToKVPool,
            torch.bfloat16,
            True,
            None,
        ),
    ],
    ids=["bf16", "fp8", "hnd", "subclass", "hnd-window-off"],
)
def test_window_refuses_a_kv_cache_whose_rows_it_cannot_move(
    monkeypatch: pytest.MonkeyPatch,
    window: MiniCPMODuplexWindow,
    pool_type: type[MHATokenToKVPool],
    kv_dtype: torch.dtype,
    use_hnd: bool,
    error: str | None,
) -> None:
    """A cut moves rows through get_key_buffer, so only the plain NHD pool qualifies; window off takes any cache."""
    monkeypatch.setenv("SGLANG_USE_HND_KVCACHE", "true" if use_hnd else "false")
    kv_cache = pool_type(
        size=KV_POOL_TOKENS,
        page_size=1,
        dtype=kv_dtype,
        head_num=KV_HEAD_COUNT,
        head_dim=HEAD_DIM,
        layer_num=KV_LAYER_COUNT,
        device="cpu",
        enable_memory_saver=False,
    )

    if error is None:
        check_window_kv_cache(window, kv_cache)
    else:
        with pytest.raises(ValueError, match=error):
            check_window_kv_cache(window, kv_cache)
