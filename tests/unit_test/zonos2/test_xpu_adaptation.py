# SPDX-License-Identifier: Apache-2.0
"""Intel XPU adaptation tests for ZONOS2.

The two platform-policy cases need no device. The accelerator cases run the
per-frame tail graph, the per-request sampler, the decode-state pool and the
one-shot codec decode on an Intel XPU, and skip elsewhere.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch
from torch import nn

from sglang_omni.model_runner.model_worker import ModelWorker
from sglang_omni.models.zonos2 import callbacks
from sglang_omni.models.zonos2 import engine_builder as engine_builder_module
from sglang_omni.models.zonos2 import model_runner as model_runner_module
from sglang_omni.models.zonos2.components import audio_codec
from sglang_omni.models.zonos2.components.audio_codec import (
    DAC_HOP_LENGTH,
    Zonos2DACVocoder,
)
from sglang_omni.models.zonos2.components.text_frontend import TTSSamplingParams, shear
from sglang_omni.models.zonos2.engine_builder import Zonos2EngineBuilder
from sglang_omni.models.zonos2.hf_config import Zonos2Config
from sglang_omni.models.zonos2.sampler import apply_repetition_penalty, sample_tts
from sglang_omni.models.zonos2.sglang_model import Zonos2SGLangModel
from sglang_omni.models.zonos2.state_pool import Zonos2DecodeStatePool
from sglang_omni.platforms import current_platform
from sglang_omni.platforms.cpu import CPUOmniPlatform
from sglang_omni.platforms.device_graph import ReplayableGraph
from sglang_omni.platforms.interface import OmniPlatform
from sglang_omni.platforms.xpu import XPUOmniPlatform
from sglang_omni.scheduling.sglang_backend.output_processor import SGLangOutputProcessor
from tests.unit_test.fixtures.accelerator import require_device_streams

# A tail-sized stand-in for the served backbone: the real codebook count, audio
# vocabulary, sentinel ids and softcap, with a narrow hidden width and head so
# capture stays cheap.
TAIL_HIDDEN_DIM = 64
TAIL_CONFIG = Zonos2Config(dim=TAIL_HIDDEN_DIM, head_dim=TAIL_HIDDEN_DIM)
TAIL_BUCKETS = [1, 2, 4]
MAX_VALID_CODE = TAIL_CONFIG.codebook_size - 1

# The per-row top_k case gives each row a best and a near-best code, so a top_k
# taken from either row and applied to the batch changes what that row can draw.
# The runner-up sits one tenth of a logit below the best, so it takes about 47
# percent of the draws once top_k admits it.
CANDIDATES_BY_ROW = ((10, 11), (20, 21))
BEST_SCORE = 5.0
RUNNER_UP_SCORE = 4.9
REJECTED_SCORE = -10.0
CANDIDATE_DRAWS = 200


class Zonos2TailHarness(nn.Module):
    """The per-frame tail of the served model: real head, sampler, embed and hash."""

    compute_logits = Zonos2SGLangModel.compute_logits
    embed_frames = Zonos2SGLangModel.embed_frames
    tail_compute = Zonos2SGLangModel.tail_compute
    capture_tail_graphs = Zonos2SGLangModel.capture_tail_graphs
    run_tail_graph = Zonos2SGLangModel.run_tail_graph

    def __init__(
        self, device: torch.device, dtype: torch.dtype, seed: int, decode_rows: int = 4
    ) -> None:
        super().__init__()
        self.config = TAIL_CONFIG
        self.n_codebooks = TAIL_CONFIG.n_codebooks
        self.audio_vocab = TAIL_CONFIG.audio_vocab
        generator = torch.Generator().manual_seed(seed)
        self.embedders = nn.ModuleList(
            [
                nn.Embedding(TAIL_CONFIG.audio_vocab, TAIL_CONFIG.dim)
                for _ in range(TAIL_CONFIG.n_codebooks)
            ]
            + [nn.Embedding(TAIL_CONFIG.text_vocab + 1, TAIL_CONFIG.dim)]
        )
        for embedder in self.embedders:
            embedder.weight.data.normal_(generator=generator)
        self.multi_output = nn.Parameter(
            torch.randn(
                (TAIL_CONFIG.audio_vocab * TAIL_CONFIG.n_codebooks, TAIL_CONFIG.dim),
                generator=generator,
            )
            * 0.1
        )
        self.decode_input_embedding = nn.Embedding(decode_rows, TAIL_CONFIG.dim)
        self.decode_input_embedding.weight.data.zero_()
        self.to(device=device, dtype=dtype).eval().requires_grad_(False)
        self.tail_buckets: list[int] = []
        self.tail_graphs: dict[int, ReplayableGraph] = {}
        self.tail_params: TTSSamplingParams | None = None
        self.tail_top_k_max = 0
        self.tail_any_top_p = False
        self.tail_any_min_p = False
        self.cg: dict[str, torch.Tensor] = {}

    @property
    def device(self) -> torch.device:
        return self.embedders[0].weight.device

    @property
    def dtype(self) -> torch.dtype:
        return self.embedders[0].weight.dtype


class StubQuantizer:
    def from_codes(self, codes: torch.Tensor) -> tuple[torch.Tensor]:
        return (codes.float(),)


class StubCodec:
    """The DAC forward, stood in by a sum over codebooks repeated per hop.

    The codec weights are a separate download the XPU image does not carry, and
    summing the codebooks makes the waveform a fingerprint of every de-sheared
    column, so a dropped or misaligned frame shows up in the samples.
    """

    def __init__(self) -> None:
        self.decoded_codes: list[torch.Tensor] = []
        self.quantizer = StubQuantizer()

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        self.decoded_codes.append(latents.clone())
        return latents.sum(dim=1, keepdim=True).repeat_interleave(
            DAC_HOP_LENGTH, dim=-1
        )


@dataclass(frozen=True, kw_only=True)
class RequestRow:
    """The one field the pool and the buffer writer read off a decode request."""

    request_id: str


@dataclass(kw_only=True)
class ForwardBatchStub:
    batch_size: int
    input_ids: torch.Tensor | None = None
    input_embeds: torch.Tensor | None = None


@dataclass(kw_only=True)
class ModelStub:
    decode_input_embedding: nn.Embedding
    decode_state_pool: Zonos2DecodeStatePool


@dataclass(kw_only=True)
class RunnerStub:
    model: ModelStub


def require_xpu() -> torch.device:
    device = require_device_streams()
    if device.type != "xpu":
        pytest.skip(f"needs an Intel XPU, found {device.type}")
    else:
        pass
    return device


@pytest.fixture
def xpu_device() -> torch.device:
    return require_xpu()


@pytest.fixture(scope="module")
def captured_tail() -> tuple[Zonos2TailHarness, TTSSamplingParams]:
    """One harness with the shipped tail graphs captured on this host's XPU.

    Greedy temperature keeps the captured kernels the shipped ones, top_k and
    min_p included, while making the draw reproducible, so a replay can be
    compared with an eager run of the same tail.
    """
    device = require_xpu()
    graph_backend = current_platform.get_device_graph_backend(device)
    harness = Zonos2TailHarness(device, torch.bfloat16, seed=7)
    params = TTSSamplingParams(temperature=0.0)
    harness.capture_tail_graphs(TAIL_BUCKETS, params, graph_backend)
    return harness, params


def tail_inputs(
    harness: Zonos2TailHarness,
    params: TTSSamplingParams,
    batch_rows: int,
    seed: int,
) -> dict[str, torch.Tensor]:
    """Staging tensors, keyed by the names run_tail_graph and the buffers share."""
    device = harness.device
    generator = torch.Generator(device=device).manual_seed(seed)
    return {
        "hidden": torch.randn(
            (batch_rows, TAIL_CONFIG.dim),
            generator=generator,
            device=device,
            dtype=harness.dtype,
        ),
        "temperature": torch.full((batch_rows,), params.temperature, device=device),
        "top_k": torch.full(
            (batch_rows,), params.top_k, device=device, dtype=torch.long
        ),
        "top_p": torch.full((batch_rows,), params.top_p, device=device),
        "min_p": torch.full((batch_rows,), params.min_p, device=device),
        "rep_pen": torch.full((batch_rows,), params.repetition_penalty, device=device),
        "rep_ids": torch.full(
            (batch_rows, TAIL_CONFIG.n_codebooks, int(params.repetition_window)),
            -1,
            device=device,
            dtype=torch.long,
        ),
        "break_mask": torch.zeros((batch_rows, TAIL_CONFIG.audio_vocab), device=device),
    }


def row_sampling_kwargs(
    device: torch.device, batch_rows: int, temperature: float
) -> dict[str, torch.Tensor]:
    """Every per-request knob neutral, so one override isolates one mask."""
    return {
        "temperature": torch.full((batch_rows,), temperature, device=device),
        "top_k": torch.zeros((batch_rows,), device=device, dtype=torch.long),
        "top_p": torch.zeros((batch_rows,), device=device),
        "min_p": torch.zeros((batch_rows,), device=device),
        "repetition_penalty": torch.ones((batch_rows,), device=device),
    }


def decode_with_stub_codec(
    monkeypatch: pytest.MonkeyPatch,
    device: str,
    delayed_codes: torch.Tensor,
    eos_frame: int | None,
) -> tuple[torch.Tensor, StubCodec]:
    codec = StubCodec()
    monkeypatch.setattr(audio_codec, "get_dac", lambda requested_device: codec)
    return (
        Zonos2DACVocoder(device=device).decode(delayed_codes, eos_frame=eos_frame),
        codec,
    )


@pytest.mark.parametrize(
    ("platform_class", "compiles"),
    [(XPUOmniPlatform, False), (CPUOmniPlatform, True)],
)
def test_zonos2_torch_compile_stays_off_on_xpu(
    monkeypatch: pytest.MonkeyPatch,
    platform_class: type[OmniPlatform],
    compiles: bool,
) -> None:
    """A compiled ZONOS2 engine on XPU returned audio for the first request and a
    header-only WAV for every request after it, so the platform has to refuse the
    compile the other accelerators take."""
    monkeypatch.setattr(engine_builder_module, "current_platform", platform_class())

    defaults = Zonos2EngineBuilder().generation_defaults(dtype="bfloat16")

    assert defaults["enable_torch_compile"] is compiles


@pytest.mark.parametrize(
    ("platform_class", "compiles"),
    [(XPUOmniPlatform, False), (CPUOmniPlatform, True)],
)
def test_compiled_sampler_follows_the_platform_compile_decision(
    monkeypatch: pytest.MonkeyPatch,
    platform_class: type[OmniPlatform],
    compiles: bool,
) -> None:
    """The sampler compile is a second entry into torch.compile, so an operator
    asking for it must not reinstate the compile the platform just refused."""
    requested: list[bool] = []

    class RecordingModelRunner:
        def __init__(
            self,
            model_worker: ModelWorker | None,
            output_processor: SGLangOutputProcessor | None,
            *,
            compile_sampler: bool,
            frame_graph: bool,
            async_decode: bool,
            stream_emit_chunk_frames: int,
            stream_emit_first_chunk_frames: int,
        ) -> None:
            del model_worker, output_processor, frame_graph, async_decode
            del stream_emit_chunk_frames, stream_emit_first_chunk_frames
            requested.append(compile_sampler)

    monkeypatch.setattr(engine_builder_module, "current_platform", platform_class())
    monkeypatch.setattr(model_runner_module, "Zonos2ModelRunner", RecordingModelRunner)

    Zonos2EngineBuilder(compile_sampler=True).make_model_runner(None, None)

    assert requested == [compiles]


@pytest.mark.accelerator
@pytest.mark.parametrize("batch_rows", [2, 3])
def test_tail_graph_replay_matches_an_eager_tail_on_xpu(
    captured_tail: tuple[Zonos2TailHarness, TTSSamplingParams],
    batch_rows: int,
) -> None:
    """Replay has to recompute from the newly staged buffers rather than reproduce
    the frame that was captured; batch_rows=3 also pads up to the 4-row bucket,
    whose extra rows carry stale state that must not reach the result."""
    harness, params = captured_tail

    replayed: list[list[torch.Tensor]] = []
    for seed in (101, 202):
        staged = tail_inputs(harness, params, batch_rows, seed)
        codes, keys, feedback = harness.run_tail_graph(**staged)
        replayed.append([codes.clone(), keys.clone(), feedback.clone()])

        for name, value in staged.items():
            harness.cg[name][:batch_rows].copy_(value)
        harness.tail_compute(batch_rows)
        eager = [
            harness.cg["codes"][:batch_rows],
            harness.cg["keys"][:batch_rows],
            harness.cg["feedback"][:batch_rows],
        ]
        for replay_tensor, eager_tensor in zip(replayed[-1], eager):
            assert replay_tensor.shape == eager_tensor.shape
            assert torch.equal(replay_tensor, eager_tensor)

    assert not torch.equal(replayed[0][0], replayed[1][0])


@pytest.mark.accelerator
@pytest.mark.parametrize(
    "narrowed",
    [{"top_k": 1}, {"top_p": 1e-6}, {"min_p": 1.0}, {"temperature": 0.0}],
    ids=["top_k", "top_p", "min_p", "greedy"],
)
def test_narrowing_one_sampler_mask_fixes_the_draw_on_xpu(
    xpu_device: torch.device, narrowed: dict[str, float]
) -> None:
    """Each mask left with a single survivor per row makes the draw that row's
    highest-scoring code, which pins the masks, the softmax and the multinomial
    on the device at the served codebook count and vocabulary."""
    params = TTSSamplingParams()
    batch_rows = 4
    generator = torch.Generator(device=xpu_device).manual_seed(5)
    logits = torch.randn(
        (batch_rows, TAIL_CONFIG.n_codebooks, TAIL_CONFIG.audio_vocab),
        generator=generator,
        device=xpu_device,
    )
    kwargs = row_sampling_kwargs(xpu_device, batch_rows, params.temperature)
    for name, value in narrowed.items():
        kwargs[name] = torch.full_like(kwargs[name], value)

    codes = sample_tts(
        logits,
        **kwargs,
        top_k_max=1 if "top_k" in narrowed else 0,
        any_top_p="top_p" in narrowed,
        any_min_p="min_p" in narrowed,
    )

    assert torch.equal(codes, logits.argmax(dim=-1))


@pytest.mark.accelerator
def test_each_row_keeps_its_own_top_k_on_xpu(xpu_device: torch.device) -> None:
    """One top_k reused for the whole batch would silently widen or narrow every
    other request: the row allowed one code must never reach its runner-up, and
    the row allowed two must reach both."""
    logits = torch.full(
        (2, 1, TAIL_CONFIG.audio_vocab), REJECTED_SCORE, device=xpu_device
    )
    for row, (best, runner_up) in enumerate(CANDIDATES_BY_ROW):
        logits[row, 0, best] = BEST_SCORE
        logits[row, 0, runner_up] = RUNNER_UP_SCORE
    kwargs = row_sampling_kwargs(xpu_device, batch_rows=2, temperature=1.0)
    kwargs["top_k"] = torch.tensor([1, 2], device=xpu_device)

    drawn = [
        sample_tts(logits, **kwargs, top_k_max=2, any_top_p=False, any_min_p=False)
        for _ in range(CANDIDATE_DRAWS)
    ]

    assert {int(codes[0, 0]) for codes in drawn} == {CANDIDATES_BY_ROW[0][0]}
    assert {int(codes[1, 0]) for codes in drawn} == set(CANDIDATES_BY_ROW[1])


@pytest.mark.accelerator
def test_repetition_penalty_on_xpu_matches_the_cpu_penalty(
    xpu_device: torch.device,
) -> None:
    """The penalty counts repeats with a scatter-add over the whole vocabulary per
    codebook, a shape at which XPU reductions have returned wrong values."""
    params = TTSSamplingParams()
    generator = torch.Generator(device=xpu_device).manual_seed(13)
    logits = torch.randn(
        (4, TAIL_CONFIG.n_codebooks, TAIL_CONFIG.audio_vocab),
        generator=generator,
        device=xpu_device,
    )
    repeated_ids = torch.randint(
        -1,
        TAIL_CONFIG.audio_vocab,
        (4, TAIL_CONFIG.n_codebooks, int(params.repetition_window)),
        generator=generator,
        device=xpu_device,
    )
    penalties = torch.full((4,), params.repetition_penalty, device=xpu_device)

    penalized = apply_repetition_penalty(logits, repeated_ids, penalties)

    assert not torch.equal(penalized, logits)
    assert torch.equal(
        penalized.cpu(),
        apply_repetition_penalty(logits.cpu(), repeated_ids.cpu(), penalties.cpu()),
    )


@pytest.mark.accelerator
def test_decode_state_pool_stages_each_row_on_the_xpu(
    xpu_device: torch.device,
) -> None:
    """The pool holds the per-request feedback the backbone decodes from, so its
    rows, the gather into the decode buffer and the release reset all have to land
    on the device the model runs on."""
    model = Zonos2TailHarness(xpu_device, torch.bfloat16, seed=3, decode_rows=4)
    pool = Zonos2DecodeStatePool(model)
    requests = [RequestRow(request_id="first"), RequestRow(request_id="second")]

    rows = pool.prepare_active_rows(requests)

    assert rows.device == xpu_device
    assert {
        tensor.device
        for tensor in (
            pool.feedback_embeds,
            pool.eos_frame_set,
            pool.eos_frame_val,
            pool.eos_countdown,
            pool.generation_step,
            pool.rep_hist,
            pool.rep_len,
        )
    } == {xpu_device}

    pool.feedback_embeds[rows[0]].fill_(3)
    pool.feedback_embeds[rows[1]].fill_(5)
    buffer = model.decode_input_embedding
    # A used buffer, so the padded tail has to be cleared rather than left as the
    # previous step's embedding.
    buffer.weight.data.fill_(7)
    forward_batch = ForwardBatchStub(batch_size=4)
    runner = RunnerStub(
        model=ModelStub(decode_input_embedding=buffer, decode_state_pool=pool)
    )

    callbacks.write_zonos2_buffers(runner, forward_batch, None, requests)

    assert torch.all(buffer.weight[0] == 3)
    assert torch.all(buffer.weight[1] == 5)
    assert torch.count_nonzero(buffer.weight[2:]) == 0
    assert forward_batch.input_ids.device == xpu_device
    assert forward_batch.input_ids.tolist() == [0, 1, 2, 3]
    assert forward_batch.input_embeds is None

    pool.release_row("first")

    assert torch.count_nonzero(pool.feedback_embeds[rows[0]]) == 0
    assert torch.all(pool.feedback_embeds[rows[1]] == 5)


@pytest.mark.accelerator
def test_staging_refuses_more_requests_than_the_decode_buffer_holds(
    xpu_device: torch.device,
) -> None:
    """Both bounds guard an in-place write into fixed device memory: a forward
    batch narrower than the request list, and a request past the pool's last row."""
    model = Zonos2TailHarness(xpu_device, torch.bfloat16, seed=3, decode_rows=2)
    pool = Zonos2DecodeStatePool(model)
    runner = RunnerStub(
        model=ModelStub(
            decode_input_embedding=model.decode_input_embedding,
            decode_state_pool=pool,
        )
    )

    with pytest.raises(ValueError, match="batch_size"):
        callbacks.write_zonos2_buffers(
            runner,
            ForwardBatchStub(batch_size=1),
            None,
            [RequestRow(request_id="first"), RequestRow(request_id="second")],
        )

    for index in range(pool.padding_row):
        pool.acquire_row(f"request-{index}")

    with pytest.raises(RuntimeError, match="pool exhausted"):
        pool.acquire_row("one-too-many")


@pytest.mark.accelerator
@pytest.mark.parametrize("eos_frame", [None, 10])
def test_delayed_frames_decode_to_aligned_audio_on_xpu(
    monkeypatch: pytest.MonkeyPatch, xpu_device: torch.device, eos_frame: int | None
) -> None:
    """De-shearing on the device has to invert the delay pattern the prompt
    applies and hand the codec one hop of audio per aligned frame, in order,
    capped at the EOS frame."""
    frames = 24
    generator = torch.Generator().manual_seed(3)
    aligned_codes = torch.randint(
        0,
        TAIL_CONFIG.codebook_size,
        (frames, TAIL_CONFIG.n_codebooks),
        generator=generator,
    )
    delayed_codes = shear(aligned_codes, TAIL_CONFIG.audio_pad_id)
    kept_frames = frames - (TAIL_CONFIG.n_codebooks - 1)
    if eos_frame is not None:
        kept_frames = min(kept_frames, eos_frame)
    else:
        pass

    waveform, _ = decode_with_stub_codec(
        monkeypatch, str(xpu_device), delayed_codes, eos_frame
    )

    assert waveform.dtype == torch.float32
    assert waveform.device == torch.device("cpu")
    assert torch.equal(
        waveform,
        aligned_codes[:kept_frames]
        .sum(dim=-1)
        .to(torch.float32)
        .repeat_interleave(DAC_HOP_LENGTH),
    )
    cpu_waveform, _ = decode_with_stub_codec(
        monkeypatch, "cpu", delayed_codes, eos_frame
    )
    assert torch.equal(waveform, cpu_waveform)


@pytest.mark.accelerator
def test_sentinel_codes_never_reach_the_codec_on_xpu(
    monkeypatch: pytest.MonkeyPatch, xpu_device: torch.device
) -> None:
    """End-of-audio and pad ids are frame markers, not codebook entries; left in
    place they index past the codebook and the request returns noise."""
    generator = torch.Generator().manual_seed(3)
    aligned_codes = torch.randint(
        0, TAIL_CONFIG.codebook_size, (24, TAIL_CONFIG.n_codebooks), generator=generator
    )
    aligned_codes[5, 4] = TAIL_CONFIG.eoa_id
    aligned_codes[6, 0] = TAIL_CONFIG.audio_pad_id

    _, codec = decode_with_stub_codec(
        monkeypatch,
        str(xpu_device),
        shear(aligned_codes, TAIL_CONFIG.audio_pad_id),
        None,
    )

    assert codec.decoded_codes
    for decoded in codec.decoded_codes:
        assert int(decoded.max()) <= MAX_VALID_CODE
