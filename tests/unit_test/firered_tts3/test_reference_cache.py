# SPDX-License-Identifier: Apache-2.0
"""Reference encoding cache behavior."""

from __future__ import annotations

import base64
import concurrent.futures
import contextlib
import io
import threading
import weakref
from collections.abc import Iterator
from pathlib import Path
from typing import Literal
from unittest.mock import patch

import av
import numpy as np
import pytest
import torch

from sglang_omni.config.runtime import (
    resolve_stage_factory_kwargs,
    resolve_stage_typed_kwargs,
)
from sglang_omni.models.firered_tts3.config import FireRedTTS3PipelineConfig
from sglang_omni.models.firered_tts3.hf_config import FireRedTTS3Config
from sglang_omni.models.firered_tts3.payload_types import load_firered_tts3_state
from sglang_omni.models.firered_tts3.ref_encode import (
    FireRedReferenceBundle,
    FireRedReferenceEncoder,
    FireRedReferenceHook,
    ReferenceAudioInput,
    ReferenceConditioning,
)
from sglang_omni.models.firered_tts3.weights import (
    build_redae_encoder,
    drop_embedding_tables,
)
from sglang_omni.scheduling.reference_encoder import ReferenceEncodeService
from sglang_omni.utils.audio import check_torchcodec_ready, load_audio
from tests.unit_test.fixtures.firered_tts3 import make_speaker_tokenizer
from tests.unit_test.fixtures.firered_tts3_preprocessing import (
    make_preprocessing_payload,
    make_reference_encoder,
    preprocess_payload,
)


@pytest.mark.parametrize(
    "task_type,is_data_uri",
    [("Base", True), ("SemanticEdit", False), ("AcousticEdit", True)],
)
def test_compressed_reference_prompt_uses_encoded_audio_and_cache(
    tmp_path: Path,
    task_type: Literal["Base", "SemanticEdit", "AcousticEdit"],
    is_data_uri: bool,
) -> None:
    """AAC references must share the encoder's patch count without a second decode."""
    if not check_torchcodec_ready():
        pytest.skip("AAC decoding requires TorchCodec and FFmpeg")
    else:
        pass
    encoded_audio = io.BytesIO()
    with av.open(encoded_audio, mode="w", format="mp4") as container:
        audio_stream = container.add_stream("aac", rate=16000)
        audio_stream.layout = "stereo"
        frame = av.AudioFrame.from_ndarray(
            np.zeros((2, 4097), dtype=np.float32), format="fltp", layout="stereo"
        )
        frame.sample_rate = 16000
        for packet in audio_stream.encode(frame):
            container.mux(packet)
        for packet in audio_stream.encode():
            container.mux(packet)
    audio_bytes = encoded_audio.getvalue()
    if is_data_uri:
        reference_source = (
            f"data:audio/mp4;base64,{base64.b64encode(audio_bytes).decode('ascii')}"
        )
    else:
        reference_path = tmp_path / "reference.m4a"
        reference_path.write_bytes(audio_bytes)
        reference_source = str(reference_path)
    config = FireRedTTS3Config(use_lm_head=True)
    tokenizer = make_speaker_tokenizer()
    with make_reference_encoder(config) as reference_encoder:
        with patch(
            "sglang_omni.models.firered_tts3.ref_encode.load_audio", wraps=load_audio
        ) as decode_reference:
            for _ in range(2):
                payload = make_preprocessing_payload(
                    inputs={
                        "text": "hello" if task_type == "Base" else "",
                        "task_type": task_type,
                        "instructions": "Replace 'birthday' with 'anniversary'.",
                        "references": [
                            {"audio_path": reference_source, "text": "source text"}
                        ],
                    }
                )
                preprocess_payload(payload, config=config, tokenizer=tokenizer)
                state = load_firered_tts3_state(
                    reference_encoder.encode_payload(payload)
                )
                if task_type == "Base":
                    reference_latents = state.prompt_latents
                    pad_token = "<|video_pad|>"
                else:
                    reference_latents = state.audio_in_latents
                    pad_token = "<|image_pad|>"
                assert reference_latents is not None
                assert state.input_ids.count(
                    tokenizer.convert_tokens_to_ids(pad_token)
                ) == (reference_latents.shape[1] // config.patch_size)
            assert decode_reference.call_count == 1


class ReferenceEncoderStub:
    """Distinct artifacts make cache hits observable without loading an audio model."""

    sample_rate: int = 24000
    pad_multiple: int = 3840
    redae_scale: float = 1.0
    spk_extractor: torch.nn.Module | None = None

    def __init__(self) -> None:
        self.encode_calls: int = 0

    def encode(
        self, path: str, max_seconds: float | None = None
    ) -> dict[str, torch.Tensor]:
        self.encode_calls += 1
        return {"prompt_latents": torch.full((1, 2, 4), float(self.encode_calls))}


def make_reference_service(
    reference_encoder: ReferenceEncoderStub, *, cache_enabled: bool = True
) -> ReferenceEncodeService[
    ReferenceAudioInput,
    ReferenceConditioning,
    ReferenceConditioning,
    ReferenceAudioInput,
]:
    return ReferenceEncodeService(
        FireRedReferenceHook(
            reference_encoder, model_id="stub", cache_enabled=cache_enabled
        ),
        max_items=8,
        max_bytes=1 << 20,
    )


def test_reference_cache_hit_reuses_stored_artifact(tmp_path: Path) -> None:
    reference_encoder = ReferenceEncoderStub()
    service = make_reference_service(reference_encoder)
    reference_audio_path = tmp_path / "ref.wav"
    reference_audio_path.write_bytes(b"ref-audio-bytes")

    reference_audio = ReferenceAudioInput(str(reference_audio_path), None)
    first = service.get_or_encode(reference_audio)
    second = service.get_or_encode(reference_audio)
    assert reference_encoder.encode_calls == 1
    assert service.stats()["hits"] == 1
    assert torch.equal(first["prompt_latents"], second["prompt_latents"])

    second["prompt_latents"].fill_(-1.0)
    third = service.get_or_encode(reference_audio)
    assert torch.equal(third["prompt_latents"], first["prompt_latents"])
    service.close()


def test_reference_cache_key_separates_edit_cap(tmp_path: Path) -> None:
    reference_encoder = ReferenceEncoderStub()
    service = make_reference_service(reference_encoder)
    reference_audio_path = tmp_path / "ref.wav"
    reference_audio_path.write_bytes(b"ref-audio-bytes")

    uncapped_reference = ReferenceAudioInput(str(reference_audio_path), None)
    capped_reference = ReferenceAudioInput(str(reference_audio_path), 30.0)
    uncapped = service.get_or_encode(uncapped_reference)
    capped = service.get_or_encode(capped_reference)
    assert reference_encoder.encode_calls == 2
    assert service.stats()["entries"] == 2
    assert torch.equal(uncapped["prompt_latents"], torch.ones(1, 2, 4))
    assert torch.equal(capped["prompt_latents"], torch.full((1, 2, 4), 2.0))
    assert torch.equal(
        service.get_or_encode(uncapped_reference)["prompt_latents"],
        uncapped["prompt_latents"],
    )
    assert torch.equal(
        service.get_or_encode(capped_reference)["prompt_latents"],
        capped["prompt_latents"],
    )
    assert reference_encoder.encode_calls == 2
    service.close()


def test_reference_cache_disabled_encodes_every_call(tmp_path: Path) -> None:
    reference_encoder = ReferenceEncoderStub()
    service = make_reference_service(reference_encoder, cache_enabled=False)
    reference_audio_path = tmp_path / "ref.wav"
    reference_audio_path.write_bytes(b"ref-audio-bytes")

    reference_audio = ReferenceAudioInput(str(reference_audio_path), None)
    service.get_or_encode(reference_audio)
    service.get_or_encode(reference_audio)
    assert reference_encoder.encode_calls == 2
    stats = service.stats()
    assert stats["uncacheable"] == 2
    assert stats["hits"] == 0
    assert stats["entries"] == 0
    service.close()


class LengthSensitiveSpeaker(torch.nn.Module):
    """Stands in for CAM++, whose statistics pooling makes padded audio change the embedding."""

    def forward(self, audio: torch.Tensor, audio_sr: int) -> torch.Tensor:
        return torch.stack(
            [
                audio.mean(),
                audio.abs().max(),
                audio.new_tensor(float(audio.shape[-1])),
            ]
        ).reshape(1, 3)


@pytest.fixture
def reference_bundle() -> FireRedReferenceBundle:
    """A tiny random RedAE encoder whose sliding window is 24 patches."""
    torch.manual_seed(0)
    # note (yucheng): Only the 16-patch short clip is under the window, so alone it takes the mask-free attention path and inside a batch the explicit-mask path.
    config = FireRedTTS3Config(
        audio_patch_size=8,
        bottleneck_dim=6,
        enc_hidden_size=16,
        enc_intermediate_size=32,
        enc_num_hidden_layers=2,
        enc_num_attention_heads=2,
        enc_num_key_value_heads=1,
        enc_sliding_window=24,
        enc_downsample_num_hidden_layers=1,
        spk_in_dim=3,
    )
    encoder = build_redae_encoder(config)
    drop_embedding_tables(encoder)
    # Large weights make attention a visible share of the output, so a leak would not hide in rounding.
    for parameter in encoder.parameters():
        if parameter.dim() > 1:
            torch.nn.init.normal_(parameter, std=0.5)
        else:
            pass
    with (
        patch(
            "sglang_omni.models.firered_tts3.ref_encode.load_redae_encoder",
            return_value=encoder.eval(),
        ),
        patch(
            "sglang_omni.models.firered_tts3.ref_encode.load_campp",
            return_value=LengthSensitiveSpeaker(),
        ),
    ):
        return FireRedReferenceBundle("unused", config, "cpu")


@pytest.fixture
def fp32_encoder(monkeypatch: pytest.MonkeyPatch) -> None:
    """Run the encoder in fp32 instead of bf16 autocast, so comparisons can be tight."""
    monkeypatch.setattr(torch, "autocast", lambda **kwargs: contextlib.nullcontext())


@pytest.fixture
def batch_only(monkeypatch: pytest.MonkeyPatch) -> None:
    """Send every request through the batch queue, where the first arrival at an idle service would otherwise encode alone."""
    monkeypatch.setattr(FireRedReferenceHook, "can_encode_alone", lambda self: False)


def make_waveforms() -> dict[str, np.ndarray]:
    generator = np.random.default_rng(0)
    waveforms = {
        name: generator.standard_normal(sample_count).astype(np.float32)
        for name, sample_count in (
            ("long", 300),
            ("short", 70),
            ("capped", 600),
            ("mid", 190),
        )
    }
    waveforms["empty"] = np.zeros(0, dtype=np.float32)
    return waveforms


@pytest.fixture
def decoded_audio() -> Iterator[list[tuple[str, str]]]:
    """Serve synthetic clips by file name and yield the (source, thread name) of every decode."""
    waveforms = make_waveforms()
    decodes: list[tuple[str, str]] = []

    def decode(
        source: str,
        source_name: str = "audio",
        target_sample_rate: int = 16000,
        mono: bool = True,
    ) -> np.ndarray:
        decodes.append((source, threading.current_thread().name))
        if Path(source).name == "broken":
            raise ValueError("undecodable reference")
        else:
            pass
        return waveforms[Path(source).name]

    with patch("sglang_omni.models.firered_tts3.ref_encode.load_audio", decode):
        yield decodes


# Padded lengths are 320, 128, 384 (capped to 375 samples) and 192 samples: distinct frame counts expose any reorder.
MIXED_REFERENCES = [
    ReferenceAudioInput("long", None),
    ReferenceAudioInput("short", None),
    ReferenceAudioInput("capped", 1 / 64),
    ReferenceAudioInput("mid", None),
]


def encode_prepared_batch(
    hook: FireRedReferenceHook, references: list[ReferenceAudioInput]
) -> list[ReferenceConditioning | Exception]:
    return hook.encode_batch([hook.prepare_batch_input(item) for item in references])


def assert_matches_per_clip(
    batched: ReferenceConditioning | Exception, per_clip: ReferenceConditioning
) -> None:
    assert isinstance(batched, dict), batched
    assert batched.keys() == per_clip.keys()
    torch.testing.assert_close(
        batched["prompt_latents"], per_clip["prompt_latents"], atol=1e-5, rtol=0
    )
    assert torch.equal(batched["speaker_embedding"], per_clip["speaker_embedding"])


@pytest.mark.usefixtures("fp32_encoder", "decoded_audio")
def test_batched_reference_encode_matches_per_clip(
    reference_bundle: FireRedReferenceBundle,
) -> None:
    """One right-padded RedAE forward must give every clip its own encode, in input order."""
    hook = FireRedReferenceHook(reference_bundle, model_id="stub", cache_enabled=False)
    batched = encode_prepared_batch(hook, MIXED_REFERENCES)
    assert len(batched) == len(MIXED_REFERENCES)
    for batched_artifact, reference in zip(batched, MIXED_REFERENCES):
        assert_matches_per_clip(batched_artifact, hook.encode_one(reference))


@pytest.mark.usefixtures("fp32_encoder", "decoded_audio")
def test_batched_reference_encode_bounds_padded_samples_per_forward(
    reference_bundle: FireRedReferenceBundle,
) -> None:
    """A drained batch splits into forwards whose rows x longest clip stay under the budget."""
    budget = 640
    reference_bundle.batch_sample_budget = budget
    forward_shapes: list[tuple[int, ...]] = []
    reference_bundle.encoder.register_forward_pre_hook(
        lambda module, args: forward_shapes.append(tuple(args[0].shape))
    )
    hook = FireRedReferenceHook(reference_bundle, model_id="stub", cache_enabled=False)
    batched = encode_prepared_batch(hook, MIXED_REFERENCES)
    assert sum(rows for rows, _ in forward_shapes) == len(MIXED_REFERENCES)
    assert (
        max(rows for rows, _ in forward_shapes) > 1
    ), "clips must still share forwards"
    # A group of one is the only way to run a clip that alone exceeds the budget.
    assert all(
        rows == 1 or rows * samples <= budget for rows, samples in forward_shapes
    )
    assert len(forward_shapes) > 1
    for batched_artifact, reference in zip(batched, MIXED_REFERENCES):
        assert_matches_per_clip(batched_artifact, hook.encode_one(reference))


@pytest.mark.usefixtures("fp32_encoder", "decoded_audio")
def test_failed_forward_fails_only_its_own_group(
    reference_bundle: FireRedReferenceBundle,
) -> None:
    """An oversize clip that runs out of memory must not discard the clips batched with others, nor keep its activations alive while the larger groups run."""
    # Groups by ascending length: [short, mid] (24 patches), [long] (40), [capped] (48).
    reference_bundle.batch_sample_budget = 640
    failed_activations: list[weakref.ref[torch.Tensor]] = []
    failed_activation_alive_at_next_group: list[bool] = []

    def run_out_of_memory_on_long_group(
        module: torch.nn.Module, args: tuple[torch.Tensor]
    ) -> None:
        hidden_states = args[0]
        if hidden_states.shape[1] == 40:
            failed_activations.append(weakref.ref(hidden_states))
            raise RuntimeError("out of memory")
        elif hidden_states.shape[1] == 48:
            failed_activation_alive_at_next_group.append(
                failed_activations[0]() is not None
            )
        else:
            pass

    reference_bundle.encoder.qwen3.layers[0].register_forward_pre_hook(
        run_out_of_memory_on_long_group
    )
    hook = FireRedReferenceHook(reference_bundle, model_id="stub", cache_enabled=False)
    outcomes = encode_prepared_batch(hook, MIXED_REFERENCES)
    assert failed_activation_alive_at_next_group == [False]
    assert isinstance(outcomes[0], RuntimeError)
    for index in (1, 2, 3):
        assert_matches_per_clip(
            outcomes[index], hook.encode_one(MIXED_REFERENCES[index])
        )


@pytest.mark.usefixtures("fp32_encoder", "decoded_audio")
def test_a_group_that_fails_together_retries_its_clips_alone(
    reference_bundle: FireRedReferenceBundle,
) -> None:
    """A forward that fails only because clips share it, such as out of memory on the padded group, must not fail clips that encode alone."""
    # Groups by ascending length: [short, mid] (24 patches), [long] (40), [capped] (48).
    reference_bundle.batch_sample_budget = 640
    forward_rows: list[int] = []

    def run_out_of_memory_on_shared_forwards(
        module: torch.nn.Module, args: tuple[torch.Tensor]
    ) -> None:
        forward_rows.append(args[0].shape[0])
        if args[0].shape[0] > 1:
            raise RuntimeError("out of memory")
        else:
            pass

    reference_bundle.encoder.qwen3.layers[0].register_forward_pre_hook(
        run_out_of_memory_on_shared_forwards
    )
    hook = FireRedReferenceHook(reference_bundle, model_id="stub", cache_enabled=False)
    outcomes = encode_prepared_batch(hook, MIXED_REFERENCES)
    assert forward_rows == [2, 1, 1, 1, 1]
    for outcome, reference in zip(outcomes, MIXED_REFERENCES):
        assert_matches_per_clip(outcome, hook.encode_one(reference))


@pytest.mark.usefixtures("fp32_encoder", "batch_only")
def test_bad_reference_fails_alone_without_re_encoding_its_batch_mates(
    reference_bundle: FireRedReferenceBundle, decoded_audio: list[tuple[str, str]]
) -> None:
    """A clip that cannot be decoded, or is empty, must not send its batch-mates to serial encodes."""
    reference_encoder = FireRedReferenceEncoder(
        reference_bundle,
        model_id="stub",
        max_batch_size=2,
        max_batch_wait_ms=5000.0,
        cache_enabled=False,
    )
    try:
        service = reference_encoder.service
        names = ("long", "broken", "short", "empty")
        with patch.object(
            reference_bundle, "encode", wraps=reference_bundle.encode
        ) as per_clip_encode:
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
                futures = {
                    name: pool.submit(
                        service.get_or_encode, ReferenceAudioInput(name, None)
                    )
                    for name in names
                }
                with pytest.raises(ValueError, match="undecodable"):
                    futures["broken"].result(timeout=10)
                with pytest.raises(ValueError, match="no samples"):
                    futures["empty"].result(timeout=10)
                encoded = {
                    name: futures[name].result(timeout=10) for name in ("long", "short")
                }
            assert per_clip_encode.call_count == 0
        stats = service.stats()
        assert (stats["batches"], stats["batched_items"], stats["failed"]) == (1, 2, 2)
        for name, artifact in encoded.items():
            assert_matches_per_clip(artifact, reference_bundle.encode(name))
    finally:
        reference_encoder.close()


@pytest.mark.usefixtures("batch_only")
def test_normalize_runs_in_caller_threads_and_never_for_cache_hits(
    reference_bundle: FireRedReferenceBundle,
    decoded_audio: list[tuple[str, str]],
    tmp_path: Path,
) -> None:
    """Decoding must not queue behind the single batch worker, and a hit decodes nothing."""
    references = []
    for name in ("long", "short", "mid"):
        reference_audio_path = tmp_path / name
        reference_audio_path.write_bytes(name.encode())
        references.append(ReferenceAudioInput(str(reference_audio_path), None))
    reference_encoder = FireRedReferenceEncoder(
        reference_bundle,
        model_id="stub",
        max_batch_size=8,
        max_batch_wait_ms=4.0,
        cache_enabled=True,
    )
    try:
        service = reference_encoder.service
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=3, thread_name_prefix="caller"
        ) as pool:
            for _ in range(2):
                list(pool.map(service.get_or_encode, references))
        stats = service.stats()
        assert (stats["misses"], stats["hits"]) == (3, 3)
        assert stats["batched_items"] == 3
        assert len(decoded_audio) == 3
        assert all(thread.startswith("caller") for _, thread in decoded_audio)
    finally:
        reference_encoder.close()


@pytest.mark.parametrize(
    ("variant", "yaml_max_batch_size", "expected_max_batch_size"),
    [("base", None, 8), ("instruct", None, 1), ("base", 1, 1), ("instruct", 4, 4)],
)
def test_reference_encode_batching_follows_variant_default_and_yaml_override(
    reference_bundle: FireRedReferenceBundle,
    tmp_path: Path,
    variant: Literal["base", "instruct"],
    yaml_max_batch_size: int | None,
    expected_max_batch_size: int,
) -> None:
    """Base enables batching by default, Instruct stays per clip, a YAML value wins for both, and a lone request takes the per-clip encode in every row."""
    config_values = FireRedTTS3PipelineConfig(
        model_path=str(tmp_path), variant=variant
    ).model_dump()
    if yaml_max_batch_size is not None:
        stage = next(
            stage
            for stage in config_values["stages"]
            if stage["name"] == "reference_encode"
        )
        stage["factory"] = {"max_batch_size": yaml_max_batch_size}
    else:
        pass
    config = FireRedTTS3PipelineConfig(**config_values)
    stage = config.stage_named("reference_encode")
    factory_kwargs = {
        **resolve_stage_factory_kwargs(stage, config),
        **resolve_stage_typed_kwargs(stage),
    }
    assert factory_kwargs["max_batch_size"] == expected_max_batch_size
    # note (yucheng): Base keeps main's 8 request threads; raising it needs a c=1 TTFP check after c=16 load.
    assert factory_kwargs["max_concurrency"] == 8
    reference_audio_path = tmp_path / "ref.wav"
    reference_audio_path.write_bytes(b"ref-audio-bytes")
    reference_audio = ReferenceAudioInput(str(reference_audio_path), None)
    reference_encoder = FireRedReferenceEncoder(
        reference_bundle,
        model_id="stub",
        max_batch_size=factory_kwargs["max_batch_size"],
        max_batch_wait_ms=factory_kwargs["max_batch_wait_ms"],
        cache_enabled=factory_kwargs["reference_cache"],
    )
    try:
        service = reference_encoder.service
        assert service.batching_enabled is (expected_max_batch_size > 1)
        with patch(
            "sglang_omni.models.firered_tts3.ref_encode.load_audio",
            return_value=make_waveforms()["long"],
        ):
            first = service.get_or_encode(reference_audio)
            second = service.get_or_encode(reference_audio)
            direct = reference_bundle.encode(str(reference_audio_path))
        stats = service.stats()
        assert (stats["hits"], stats["batches"]) == (1, 0)
        assert stats["solo_encodes"] == int(expected_max_batch_size > 1)
        for name in direct:
            assert torch.equal(first[name], direct[name])
            assert torch.equal(second[name], direct[name])
    finally:
        reference_encoder.close()


def test_lone_reference_request_takes_the_per_clip_encode(
    reference_bundle: FireRedReferenceBundle, decoded_audio: list[tuple[str, str]]
) -> None:
    """With no other encode in flight, a request must run the per-clip encode on the service's solo thread, with no batch worker hop and no linger, so its conditioning is the per-clip one."""
    reference_encoder = FireRedReferenceEncoder(
        reference_bundle,
        model_id="stub",
        max_batch_size=8,
        # A request that reached the batch worker would sit out this whole linger.
        max_batch_wait_ms=5000.0,
        cache_enabled=False,
    )
    try:
        service = reference_encoder.service
        assert service.batching_enabled
        with (
            patch.object(
                reference_bundle, "encode", wraps=reference_bundle.encode
            ) as per_clip_encode,
            patch.object(
                reference_bundle, "encode_batch", wraps=reference_bundle.encode_batch
            ) as batched_encode,
        ):
            artifact = service.get_or_encode(ReferenceAudioInput("long", None))
        per_clip_encode.assert_called_once_with("long", max_seconds=None)
        batched_encode.assert_not_called()
        stats = service.stats()
        assert (stats["batches"], stats["solo_encodes"]) == (0, 1)
        [(source, decode_thread)] = decoded_audio
        assert source == "long"
        assert decode_thread.startswith("fireredtts3-ref-encode-batch-solo")
        per_clip_artifact = reference_bundle.encode("long")
        assert artifact.keys() == per_clip_artifact.keys()
        for name, tensor in per_clip_artifact.items():
            assert torch.equal(artifact[name], tensor)
    finally:
        reference_encoder.close()
