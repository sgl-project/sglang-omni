# SPDX-License-Identifier: Apache-2.0
"""The gesture stage's session contract on tiny random native models."""

import threading
from unittest.mock import patch

import numpy as np
import pytest
import torch

from sglang_omni.models.miburi.architecture import (
    SPEECH_AUDIO_CODEBOOKS,
    GestureLMSpec,
    MotionCodecSpec,
)
from sglang_omni.models.miburi.batching import GestureBatch
from sglang_omni.models.miburi.components.gesture_model import GestureModel
from sglang_omni.models.miburi.components.motion_codec import MotionCodec
from sglang_omni.models.miburi.formats import (
    EXPRESSION_FEATURES,
    FEATURES_PER_FRAME,
    MOTION,
    POSE_FEATURES,
    ROOT_FEATURES,
    SMPLX_MOTION,
)
from sglang_omni.models.miburi.loading import GestureRuntime, GestureSampling
from sglang_omni.models.miburi.motion import ROOT_START_POSITION_M, BodyPartMasks
from sglang_omni.models.miburi.stages import GestureHooks
from sglang_omni.proto.request import OmniRequest, StagePayload
from sglang_omni.proto.session import SessionIdentity, TimedChunk
from sglang_omni.scheduling.session import SessionAppend, SessionContext

CODEBOOK_SIZE = 16
TEXT_VOCAB = 32
AUDIO_VOCAB = 16
TINY_LM = GestureLMSpec(
    codebooks_per_part=(2, 2, 1),
    codebook_size=CODEBOOK_SIZE,
    dim=32,
    codebook_dim=8,
    condition_dim=16,
    text_vocab=TEXT_VOCAB,
    audio_vocab=AUDIO_VOCAB,
    ffn_dim=84,
    temporal_layers=1,
    depth_layers=1,
    num_speakers=4,
)
# Upper body and hands (43 joints), lower body (9 joints, root velocity, 4 foot contacts), jaw and 100 expressions.
CODEC_FEATURES = (43 * 6, 9 * 6 + 3 + 4, 6 + 100)
ROOT_TRANSLATION = POSE_FEATURES + EXPRESSION_FEATURES
LOUD = 1.0
SILENT = 0.0


def tiny_codec(features: int, codebooks: int) -> MotionCodec:
    codec = MotionCodec(
        MotionCodecSpec(
            features=features,
            codebooks=codebooks,
            num_layers=1,
            num_heads=2,
            dim=32,
            codebook_dim=8,
            # One row past the LM's codes: a random LM samples the pad code on codebook 0, which the released one leaves samplable.
            codebook_size=CODEBOOK_SIZE + 1,
            ffn_dim=64,
            n_filters=8,
            residual_blocks=1,
        )
    )
    for layer in codec.quantizer.vq.layers:
        # A fresh codebook is all zeros; random rows make codes move the decoded motion.
        layer.codebook.embedding_sum.normal_()
    return codec


@pytest.fixture(scope="module")
def gesture() -> GestureRuntime:
    torch.manual_seed(0)
    return GestureRuntime(
        model=GestureModel(TINY_LM),
        codecs=tuple(
            tiny_codec(features, codebooks)
            for features, codebooks in zip(
                CODEC_FEATURES, TINY_LM.codebooks_per_part, strict=True
            )
        ),
        sampling=GestureSampling(
            cfg_coef=1.3,
            temporal_temperature=0.9,
            temporal_top_p=0.8,
            depth_temperature=0.9,
            depth_top_p=0.95,
        ),
        speaker_id=1,
        idle_upper_codes=torch.randint(0, CODEBOOK_SIZE, (1, 2, 30)),
        idle_face_codes=torch.randint(0, CODEBOOK_SIZE, (1, 1, 5)),
        masks=BodyPartMasks.smplx(),
        motion_fps=25.0,
        device=torch.device("cpu"),
    )


def warmed_hooks(gesture: GestureRuntime, max_open_sessions: int) -> GestureHooks:
    hooks = GestureHooks(gesture, max_open_sessions)
    hooks.warm_up_serving_thread()
    return hooks


def agent_frames(seed: int, energies: list[float]) -> list[dict[str, list]]:
    """Units of two frames, then one: text ids, audio codes and per-frame audio energy."""
    generator = torch.Generator().manual_seed(seed)
    text_ids = torch.randint(0, TEXT_VOCAB, (len(energies),), generator=generator)
    codes = torch.randint(0, AUDIO_VOCAB, (len(energies), 8), generator=generator)
    bounds = [(0, 2)] + [(start, start + 1) for start in range(2, len(energies))]
    return [
        {
            "text_ids": text_ids[start:end].tolist(),
            "audio_codes": codes[start:end].tolist(),
            "audio_energy": energies[start:end],
        }
        for start, end in bounds
    ]


def append(
    hooks: GestureHooks,
    identity: SessionIdentity,
    seq: int,
    frames: dict[str, list],
    emitted: list[TimedChunk],
    *,
    eos: bool = False,
) -> None:
    hooks.append(
        TimedChunk(
            "speech_tokens", 0.0, 80.0 * len(frames["text_ids"]), seq, frames, eos=eos
        ),
        StagePayload(
            request_id=f"{identity.id}-{seq}", request=OmniRequest(inputs=None), data={}
        ),
        SessionContext(
            session_identity=identity, cancelled=threading.Event(), emit=emitted.append
        ),
    )


def run_session(
    hooks: GestureHooks, name: str, seed: int, units: list[dict[str, list]]
) -> list[TimedChunk]:
    identity = SessionIdentity(name)
    hooks.open(identity, OmniRequest(inputs=None, params={"seed": seed}))
    emitted: list[TimedChunk] = []
    for seq, frames in enumerate(units):
        append(hooks, identity, seq, frames, emitted)
    hooks.close(identity)
    return emitted


def rows(chunks: list[TimedChunk]) -> np.ndarray:
    return np.concatenate(
        [
            np.frombuffer(chunk.payload, "<f4").reshape(-1, FEATURES_PER_FRAME)
            for chunk in chunks
        ]
    )


def test_each_agent_frame_gives_one_80_ms_motion_chunk_on_the_frame_clock(gesture):
    hooks = warmed_hooks(gesture, 1)
    identity = SessionIdentity("call")
    hooks.open(identity, OmniRequest(inputs=None, params={"seed": 1}))
    emitted: list[TimedChunk] = []
    for seq, frames in enumerate(agent_frames(1, [LOUD] * 3)):
        append(hooks, identity, seq, frames, emitted)
    append(
        hooks,
        identity,
        2,
        {"text_ids": [], "audio_codes": [], "audio_energy": []},
        emitted,
        eos=True,
    )
    hooks.close(identity)

    assert [(chunk.modality, chunk.format) for chunk in emitted] == [
        (MOTION, SMPLX_MOTION)
    ] * 4
    assert [(chunk.t_start_ms, chunk.duration_ms) for chunk in emitted] == [
        (0.0, 80.0),
        (80.0, 80.0),
        (160.0, 80.0),
        (240.0, 0.0),
    ]
    assert [len(chunk.payload) for chunk in emitted] == [
        2 * FEATURES_PER_FRAME * 4
    ] * 3 + [0]
    assert [chunk.eos for chunk in emitted] == [False, False, False, True]
    first_root = rows(emitted[:1])[
        0, ROOT_TRANSLATION : ROOT_TRANSLATION + ROOT_FEATURES
    ]
    assert np.allclose(first_root, ROOT_START_POSITION_M)


def test_closing_a_session_frees_its_state_and_a_new_one_starts_fresh(gesture):
    hooks = warmed_hooks(gesture, 1)
    units = agent_frames(2, [LOUD] * 4)
    first = run_session(hooks, "first", 5, units)

    assert hooks.sessions == {}
    assert hooks.batch.free_slots == [0]
    assert np.array_equal(rows(run_session(hooks, "second", 5, units)), rows(first))


def test_serving_warmup_runs_the_models_without_changing_a_session(
    gesture: GestureRuntime,
) -> None:
    hooks = warmed_hooks(gesture, 1)
    units = agent_frames(2, [LOUD, SILENT, LOUD])
    expected = rows(run_session(hooks, "before", 5, units))
    random_state = torch.get_rng_state().clone()

    with patch.object(
        gesture.model, "temporal_step", wraps=gesture.model.temporal_step
    ) as step:
        hooks.warm_up_serving_thread()
        assert step.call_count > 0

    assert torch.equal(torch.get_rng_state(), random_state)
    assert hooks.sessions == {}
    assert hooks.batch.free_slots == [0]
    assert np.array_equal(rows(run_session(hooks, "after", 5, units)), expected)


def test_interleaved_sessions_move_as_they_would_alone(gesture):
    hooks = warmed_hooks(gesture, 2)
    scripts = {
        SessionIdentity("a"): (3, agent_frames(3, [LOUD] * 5)),
        SessionIdentity("b"): (4, agent_frames(4, [SILENT] * 2 + [LOUD] * 3)),
    }
    alone = {
        identity: rows(run_session(hooks, identity.id, seed, units))
        for identity, (seed, units) in scripts.items()
    }
    emitted: dict[SessionIdentity, list[TimedChunk]] = {}
    for identity, (seed, _) in scripts.items():
        hooks.open(identity, OmniRequest(inputs=None, params={"seed": seed}))
        emitted[identity] = []
    for seq in range(4):
        for identity, (_, units) in scripts.items():
            append(hooks, identity, seq, units[seq], emitted[identity])
    for identity in scripts:
        hooks.close(identity)

    for identity in scripts:
        assert np.array_equal(rows(emitted[identity]), alone[identity])


def test_silence_plays_the_idle_slices_on_upper_body_and_face_only(gesture):
    hooks = warmed_hooks(gesture, 1)
    lower = gesture.masks.lower.bool().numpy()
    upper_and_face = (gesture.masks.upper + gesture.masks.face).bool().numpy()
    loud = rows(run_session(hooks, "loud", 6, agent_frames(6, [LOUD] * 5)))
    silent = rows(run_session(hooks, "silent", 6, agent_frames(6, [SILENT] * 5)))
    other_words = rows(run_session(hooks, "other", 7, agent_frames(7, [SILENT] * 5)))

    assert np.array_equal(
        silent[:, :POSE_FEATURES][:, lower], loud[:, :POSE_FEATURES][:, lower]
    )
    assert not np.array_equal(
        silent[:, :POSE_FEATURES][:, upper_and_face],
        loud[:, :POSE_FEATURES][:, upper_and_face],
    )
    assert np.array_equal(
        silent[:, :POSE_FEATURES][:, upper_and_face],
        other_words[:, :POSE_FEATURES][:, upper_and_face],
    )
    assert np.array_equal(
        silent[:, POSE_FEATURES:ROOT_TRANSLATION],
        other_words[:, POSE_FEATURES:ROOT_TRANSLATION],
    )


@pytest.mark.parametrize("session_count", [1, 2, 3, 4])
def test_ready_batches_preserve_seeded_motion_through_pause_and_slot_reuse(
    gesture: GestureRuntime, session_count: int
) -> None:
    """A paused or replaced peer must not change another session's seeded motion beyond batched rounding."""
    units = [
        agent_frames(seed, [SILENT] * (TINY_LM.memory_context + seed))
        for seed in range(1, session_count + 1)
    ]
    reference = warmed_hooks(gesture, 1)
    expected = [
        rows(run_session(reference, f"reference-{index}", index, script))
        for index, script in enumerate(units)
    ]
    hooks = warmed_hooks(gesture, session_count)
    identities = [SessionIdentity(f"batch-{index}") for index in range(session_count)]
    emitted = [[] for _ in identities]
    requests = [
        OmniRequest(inputs=None, params={"seed": index})
        for index in range(session_count)
    ]
    for identity, request in zip(identities, requests, strict=True):
        hooks.open(identity, request)
    positions = [0] * session_count
    wave = 0
    while any(
        position < len(script)
        for position, script in zip(positions, units, strict=True)
    ):
        appends = []
        for index in reversed(range(session_count)):
            position = positions[index]
            if position < len(units[index]) and (wave + index) % 3 != 0:
                frames = units[index][position]
                appends.append(
                    SessionAppend(
                        chunk=TimedChunk(
                            "speech_tokens",
                            0.0,
                            80.0 * len(frames["text_ids"]),
                            position,
                            frames,
                        ),
                        payload=StagePayload(
                            request_id=f"{index}-{position}",
                            request=requests[index],
                            data={},
                        ),
                        context=SessionContext(
                            session_identity=identities[index],
                            cancelled=threading.Event(),
                            emit=emitted[index].append,
                        ),
                    )
                )
                positions[index] += 1
            else:
                pass
        if appends:
            assert hooks.append_batch(appends) == [append.payload for append in appends]
        else:
            pass
        wave += 1
    for index, identity in enumerate(identities):
        hooks.close(identity)
        # Batched CPU kernels round differently from a single row.
        torch.testing.assert_close(
            torch.from_numpy(rows(emitted[index])),
            torch.from_numpy(expected[index]),
            rtol=0,
            atol=1e-3,
        )
        assert [chunk.t_start_ms for chunk in emitted[index]] == [
            80.0 * frame for frame in range(len(emitted[index]))
        ]
    assert np.array_equal(
        rows(run_session(hooks, "replacement", 0, units[0])), expected[0]
    )


def test_vectorized_state_gather_matches_independent_eager_streams(
    gesture: GestureRuntime,
) -> None:
    """CFG histories and codec state follow slots even when a peer pauses across wrap."""
    batch = GestureBatch(gesture, 2)
    batch.warm_up()
    condition = torch.zeros(1, 1 + SPEECH_AUDIO_CODEBOOKS, 1, dtype=torch.long)
    idle_mask = torch.ones(1, 1, 1, dtype=torch.bool)
    idle_upper = gesture.idle_upper_codes[:, :, :1]
    idle_face = gesture.idle_face_codes[:, :, :1]
    expected = []
    for seed in (11, 29):
        slot = batch.open(torch.Generator().manual_seed(seed))
        expected.append(
            [
                tuple(
                    tensor.clone()
                    for tensor in batch.step(
                        [slot], condition, idle_mask, idle_upper, idle_face
                    )
                )
                for _ in range(TINY_LM.memory_context + 1)
            ]
        )
        batch.close(slot)
    slots = [batch.open(torch.Generator().manual_seed(seed)) for seed in (11, 29)]
    positions = [0, 0]
    while min(positions) <= TINY_LM.memory_context:
        indices = [
            index
            for index in (1, 0)
            if positions[index] <= TINY_LM.memory_context
            and (
                index == 0
                or positions[0] % 2 == 0
                or positions[0] == TINY_LM.memory_context + 1
            )
        ]
        outputs = batch.step(
            [slots[index] for index in indices],
            condition.expand(len(indices), -1, -1),
            idle_mask.expand(len(indices), -1, -1),
            idle_upper.expand(len(indices), -1, -1),
            idle_face.expand(len(indices), -1, -1),
        )
        for row, index in enumerate(indices):
            reference = expected[index][positions[index]]
            assert torch.equal(outputs[0][row : row + 1], reference[0])
            for actual, wanted in zip(outputs[1:], reference[1:], strict=True):
                torch.testing.assert_close(actual[row : row + 1], wanted)
            positions[index] += 1
    for slot in slots:
        batch.close(slot)


def test_empty_eos_and_cancelled_append_do_not_advance_a_slot(
    gesture: GestureRuntime,
) -> None:
    hooks = warmed_hooks(gesture, 2)
    request = OmniRequest(inputs=None, params={"seed": 3})
    identity = SessionIdentity("cancelled")
    emitted = []
    hooks.open(identity, request)
    cancelled = threading.Event()
    cancelled.set()
    frames = agent_frames(3, [LOUD] * 3)
    hooks.append_batch(
        [
            SessionAppend(
                chunk=TimedChunk("speech_tokens", 0.0, 160.0, 0, frames[0]),
                payload=StagePayload(request_id="cancelled", request=request, data={}),
                context=SessionContext(
                    session_identity=identity, cancelled=cancelled, emit=emitted.append
                ),
            )
        ]
    )
    assert emitted == []
    append(
        hooks,
        identity,
        1,
        {"text_ids": [], "audio_codes": [], "audio_energy": []},
        emitted,
        eos=True,
    )
    assert [(chunk.t_start_ms, chunk.duration_ms, chunk.eos) for chunk in emitted] == [
        (0.0, 0.0, True)
    ]
    emitted.clear()
    append(hooks, identity, 2, frames[0], emitted)
    hooks.close(identity)
    expected = run_session(hooks, "fresh", 3, frames[:1])
    assert np.array_equal(rows(emitted), rows(expected))
