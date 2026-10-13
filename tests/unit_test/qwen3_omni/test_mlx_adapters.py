"""Small-model checks for MLX-VLM ownership and integrated request state."""

import json
from collections import deque
from queue import Queue
from types import SimpleNamespace

import numpy as np
import pytest
import torch

mx = pytest.importorskip("mlx.core")
nn = pytest.importorskip("mlx.nn")
pytest.importorskip("mlx_vlm")

from mlx.utils import tree_flatten
from mlx_vlm.models.qwen3_omni_moe.config import ModelConfig
from mlx_vlm.models.qwen3_omni_moe.language import LanguageModel
from sglang.srt.sampling.sampling_params import SamplingParams

from sglang_omni.models.qwen3_omni.bootstrap import create_talker_scheduler
from sglang_omni.models.qwen3_omni.mlx.encoders import (
    MlxAudioEncoder,
    MlxCode2Wav,
    MlxImageEncoder,
)
from sglang_omni.models.qwen3_omni.mlx.loading import load_component
from sglang_omni.models.qwen3_omni.mlx.policy import apply_mlx_policy
from sglang_omni.models.qwen3_omni.mlx.runner import (
    OmniMlxRunner,
    OmniMlxSchedulerModelRunner,
)
from sglang_omni.models.qwen3_omni.mlx.talker_runner import MlxTalkerModelRunner
from sglang_omni.models.qwen3_omni.request_builders import (
    make_thinker_stream_output_builder,
)
from sglang_omni.proto import OmniRequest, StagePayload


@pytest.fixture(autouse=True)
def cpu_execution(monkeypatch):
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    new_stream = mx.new_thread_local_stream
    monkeypatch.setattr(
        mx, "new_thread_local_stream", lambda device: new_stream(mx.cpu)
    )
    yield
    mx.set_default_device(previous)


@pytest.fixture
def checkpoint(tmp_path):
    text = dict(
        num_hidden_layers=2,
        hidden_size=64,
        intermediate_size=128,
        num_attention_heads=1,
        num_key_value_heads=1,
        head_dim=64,
        num_experts=2,
        num_experts_per_tok=1,
        decoder_sparse_step=1,
        mlp_only_layers=[],
        moe_intermediate_size=64,
        rms_norm_eps=1e-6,
        vocab_size=128,
        rope_theta=10000,
        max_position_embeddings=128,
        rope_scaling={"type": "default", "mrope_section": [16, 8, 8]},
    )
    raw = dict(
        model_type="qwen3_omni_moe",
        thinker_config=dict(
            text_config=text,
            vision_config={"deepstack_visual_indexes": [0]},
            audio_config={},
            image_token_id=120,
            video_token_id=121,
            audio_token_id=122,
        ),
        talker_config=dict(text_config=text, accept_hidden_layer=1),
        code2wav_config={},
        quantization={
            "bits": 4,
            "group_size": 64,
            "mode": "affine",
            "thinker.language_model.model.embed_tokens": {"bits": 8, "group_size": 64},
        },
    )
    config = ModelConfig.from_dict(raw)
    model = LanguageModel(config.thinker_config.text_config, config.thinker_config)
    nn.quantize(
        model,
        bits=4,
        group_size=64,
        class_predicate=lambda name, module: (
            {"bits": 8, "group_size": 64}
            if name == "model.embed_tokens"
            else hasattr(module, "to_quantized")
        ),
    )
    weights = {
        f"thinker.language_model.{name}": value
        for name, value in tree_flatten(model.parameters())
    }
    weights["code2wav.unowned.weight"] = mx.ones((1,))
    mx.save_safetensors(str(tmp_path / "weights.safetensors"), weights)
    (tmp_path / "config.json").write_text(json.dumps(raw))
    return tmp_path


def make_request(model_inputs=None, positions=None, delta=0):
    return SimpleNamespace(
        omni_model_inputs=model_inputs or {},
        multimodal_inputs=(
            None
            if positions is None
            else SimpleNamespace(
                mrope_positions=positions, mrope_position_delta=torch.tensor([delta])
            )
        ),
    )


def prefill(runner, request, tokens, request_id="first"):
    return runner.prefill_start(
        req_id=request_id,
        new_token_ids=tokens,
        full_token_ids=tokens,
        prefix_slot_ids=[],
        new_slot_ids=list(range(len(tokens))),
        req_pool_idx=0,
        req=request,
    )


def test_component_ownership_and_per_layer_quantization(checkpoint):
    model = load_component(str(checkpoint), "language")
    assert model.model.embed_tokens.bits == 8
    assert model.model.layers[0].self_attn.q_proj.bits == 4
    embedding = load_component(str(checkpoint), "embedding")
    assert embedding.bits == 8
    np.testing.assert_array_equal(
        np.array(embedding(mx.array([2, 3]))),
        np.array(model.model.embed_tokens(mx.array([2, 3]))),
    )


def test_strict_component_load_rejects_missing_weight(checkpoint):
    weights = mx.load(str(checkpoint / "weights.safetensors"))
    weights.pop("thinker.language_model.model.norm.weight")
    mx.save_safetensors(str(checkpoint / "weights.safetensors"), weights)
    with pytest.raises(ValueError):
        load_component(str(checkpoint), "language")


def test_explicit_positions_capture_decode_and_cleanup(checkpoint):
    runner = OmniMlxRunner(str(checkpoint), disable_radix_cache=True, pool_size=64)
    positions = torch.tensor([[0, 4, 5], [0, 2, 3], [0, 6, 7]])
    request = make_request(
        {
            "image_embeds": torch.ones(1, 64),
            "deepstack_visual_embeds": [torch.ones(1, 64)],
            "pad_values": {"image": 1000000},
        },
        positions=positions,
        delta=5,
    )
    pending = prefill(runner, request, [2, 1000000, 3])
    runner.prefill_finalize(pending)
    assert pending.full_token_ids == [2, 120, 3]
    assert request.omni_mlx_prompt_hidden.shape == (3, 64)
    assert request.omni_mlx_capture_layer == 1
    decoded = runner.decode_batch_start(["first"])
    runner.decode_batch_finalize(decoded)
    assert (
        runner._req_caches["first"][0].offset == 4
    )  # noqa: leading-underscore - native cache state
    scheduler_runner = OmniMlxSchedulerModelRunner.__new__(OmniMlxSchedulerModelRunner)
    scheduler_runner.tp_worker = SimpleNamespace(_mlx_runner=runner)
    scheduler_runner.abort_request("first")
    assert not runner.position_deltas
    assert not runner._req_caches  # noqa: leading-underscore - native cache state
    assert not runner._req_token_ids  # noqa: leading-underscore - native token state
    second = prefill(runner, make_request(), [2, 3], request_id="second")
    runner.prefill_finalize(second)
    runner.clear()
    assert not runner.position_deltas
    assert not runner._req_caches  # noqa: leading-underscore - native cache state
    assert (
        runner.model._position_ids is None
    )  # noqa: leading-underscore - library state
    assert runner.model._rope_deltas is None  # noqa: leading-underscore - library state


@pytest.mark.parametrize(
    "features,layers",
    [
        (torch.ones(2, 64), [torch.ones(2, 64)]),
        (torch.ones(1, 64), []),
        (torch.ones(1, 64), [torch.ones(2, 64)]),
    ],
)
def test_invalid_media_counts_raise(checkpoint, features, layers):
    runner = OmniMlxRunner(str(checkpoint), disable_radix_cache=True, pool_size=64)
    with pytest.raises(ValueError):
        prefill(
            runner,
            make_request(
                {
                    "image_embeds": features,
                    "deepstack_visual_embeds": layers,
                }
            ),
            [2, 120, 3],
        )
    assert not runner.position_deltas


@pytest.mark.parametrize(
    "option,value",
    [
        ("max_running_requests", 2),
        ("disable_radix_cache", False),
        ("chunked_prefill_size", 256),
        ("enable_mixed_chunk", True),
        ("mlx_enable_sampling", True),
        ("cuda_graph_backend_prefill", "breakable"),
        ("cuda_graph_backend_prefill", "piecewise"),
        ("disable_decode_cuda_graph", False),
        ("tp_size", 2),
    ],
)
def test_unsupported_scheduling_modes_fail(option, value):
    with pytest.raises(ValueError, match="requires"):
        apply_mlx_policy({"tp_size": 1}, {option: value})


@pytest.mark.parametrize(
    "option,value",
    [
        ("enable_partial_start", True),
        ("feedback_enabled", False),
        ("codec_coalesce_frames", 2),
    ],
)
def test_unsupported_talker_settings_fail_before_worker_construction(option, value):
    with pytest.raises(ValueError, match="MLX talker"):
        create_talker_scheduler(SimpleNamespace(), **{option: value})


def test_text_prefill_accepts_absent_multimodal_sidecar(checkpoint):
    runner = OmniMlxRunner(str(checkpoint), disable_radix_cache=True, pool_size=64)
    request = make_request()
    request.omni_model_inputs = None
    pending = prefill(runner, request, [2, 3])
    assert runner.prefill_finalize(pending) >= 0
    runner.remove_request("first")


def test_mixed_visual_prompt_requires_both_residuals(checkpoint):
    runner = OmniMlxRunner(str(checkpoint), disable_radix_cache=True, pool_size=64)
    request = make_request(
        {
            "image_embeds": torch.ones(1, 64),
            "video_embeds": torch.ones(1, 64),
            "image_deepstack_visual_embeds": [torch.ones(1, 64)],
        }
    )
    with pytest.raises(ValueError, match="video requires"):
        prefill(runner, request, [2, 120, 121])
    request.omni_model_inputs["video_deepstack_visual_embeds"] = [
        torch.full((1, 64), 2.0)
    ]
    pending = prefill(runner, request, [2, 121, 120])
    runner.prefill_finalize(pending)
    assert request.omni_mlx_prompt_hidden.shape == (3, 64)
    runner.remove_request("first")


def test_audio_adapter_rejects_holey_mask():
    encoder = MlxAudioEncoder.__new__(MlxAudioEncoder)
    with pytest.raises(ValueError, match="valid frame prefixes"):
        encoder(
            input_features=torch.ones(1, 128, 3),
            feature_attention_mask=torch.tensor([[1, 0, 1]]),
        )


def test_audio_adapter_preserves_shared_packing():
    from sglang_omni.models.qwen3_omni.mlx import encoders

    encoder = MlxAudioEncoder.__new__(MlxAudioEncoder)
    features = torch.arange(20).reshape(2, 2, 5).float()
    mask = torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 0, 0]])
    expected = encoders.pack_padded_audio_features(features, mask.bool(), mask.sum(-1))
    rows = int(
        encoders._get_feat_extract_output_lengths(mask.sum(-1)).sum()
    )  # noqa: leading-underscore - Transformers length contract

    def tower(packed, feature_lens):
        np.testing.assert_array_equal(np.array(packed), expected.numpy())
        assert feature_lens.tolist() == [5, 3]
        return mx.zeros((rows, 64))

    encoder.audio_tower = tower
    output = encoder(input_features=features, feature_attention_mask=mask)
    assert output["audio_embeds"].shape == (rows, 64)


def test_mlx_special_embeddings_use_shared_cache(monkeypatch):
    from sglang_omni.models.qwen3_omni.mlx.talker_prefill import MlxTalkerPrefillBuilder

    builder = MlxTalkerPrefillBuilder.__new__(MlxTalkerPrefillBuilder)
    builder.tts_special_cache = None
    builder.tts_bos_token_id, builder.tts_eos_token_id, builder.tts_pad_token_id = (
        1,
        2,
        3,
    )
    builder.model = SimpleNamespace(text_projection=lambda rows: rows + 1)
    calls = []

    def embeddings(token_ids):
        calls.append(token_ids.tolist())
        return torch.zeros(3, 64)

    monkeypatch.setattr(builder, "load_prompt_token_embeddings", embeddings)
    first = builder.get_tts_special_embeds()
    assert builder.get_tts_special_embeds() is first
    assert calls == [[1, 2, 3]]
    assert all(torch.equal(row, torch.ones(1, 64)) for row in first)


def test_vocoder_rejects_invalid_codes():
    vocoder = MlxCode2Wav.__new__(MlxCode2Wav)
    vocoder.config = SimpleNamespace(num_quantizers=16, codebook_size=2048)
    with pytest.raises(ValueError, match="invalid codec"):
        vocoder(torch.full((1, 16, 2), 2150))


def test_video_encoder_accepts_shared_timing_sidecar():
    encoder = MlxImageEncoder.__new__(MlxImageEncoder)
    encoder.spatial_merge_size = 2
    encoder.visual = lambda pixels, grid_thw: (mx.ones((1, 64)), [mx.ones((1, 64))])
    result = encoder(
        pixel_values_videos=torch.ones(4, 64),
        video_grid_thw=torch.tensor([[1, 2, 2]]),
        video_second_per_grid=torch.tensor([2.0]),
        use_audio_in_video=True,
    )
    assert result["video_token_counts"].tolist() == [1]
    assert result["video_embeds"].shape == (1, 64)
    assert result["deepstack_visual_embeds_video"][0].shape == (1, 64)


def test_talker_preserves_greedy_code_prediction_after_sampling_normalization():
    class FakeTalker:
        config = SimpleNamespace(text_config=SimpleNamespace(hidden_size=64))
        predictor_temperature = None

        def __call__(self, **kwargs):
            return mx.zeros((1, 1, 3072)), mx.zeros((1, 1, 64))

        def prepare_inputs_for_generation(self, tokens, hidden, **kwargs):
            self.predictor_temperature = kwargs["temperature"]
            return mx.zeros((1, 1, 64)), mx.zeros((1, 16), dtype=mx.int32)

    model = FakeTalker()
    native = SimpleNamespace(model=model, _req_caches={"speech": []})
    runner = MlxTalkerModelRunner.__new__(MlxTalkerModelRunner)
    runner.tp_worker = SimpleNamespace(_mlx_runner=native)
    runner.outbox = Queue()
    sampling = SamplingParams(temperature=0.0)
    assert sampling.temperature == 1.0
    assert sampling.top_k == 1
    data = SimpleNamespace(
        suppress_tokens=[2150],
        req=SimpleNamespace(sampling_params=sampling),
        output_ids=[],
        pending_feedback_queue=deque(),
        stage_payload=SimpleNamespace(request=SimpleNamespace(params={"stream": True})),
    )
    result = runner.speech_step(
        SimpleNamespace(request_id="speech", data=data),
        mx.zeros((1, 1, 64)),
        mx.zeros((3, 1, 1), dtype=mx.int32),
    )
    assert model.predictor_temperature == 0
    assert result.next_token_ids.shape == (1,)
    emitted = runner.outbox.get_nowait()
    assert emitted.target == "code2wav"
    assert emitted.metadata == {"stream": True}
    assert emitted.data.shape == (16,)
    assert len(data.pending_feedback_queue) == 1


def test_prompt_capture_uses_existing_stream_and_is_consumed_once():
    hidden = torch.ones(3, 64)
    request = SimpleNamespace(
        inflight_middle_chunks=0,
        omni_mlx_prompt_hidden=hidden,
        omni_mlx_capture_layer=24,
    )
    data = SimpleNamespace(
        req=request,
        stage_payload=StagePayload(
            request_id="speech",
            request=OmniRequest(
                inputs=[],
                params={"stream": True},
                metadata={"output_modalities": ["text", "audio"]},
            ),
            data={},
        ),
    )
    builder = make_thinker_stream_output_builder(speech_enabled=True)
    messages = builder("speech", data, SimpleNamespace(data=17))
    assert [message.target for message in messages] == ["decode", "talker_ar"]
    assert messages[0].data.tolist() == [17]
    assert messages[1].data is hidden
    assert messages[1].metadata == {"token_id": 17, "prompt_hidden_layer": 24}
    assert request.omni_mlx_prompt_hidden is None
    following = builder("speech", data, SimpleNamespace(data=18))
    assert following[1].data.tolist() == [18]


def run_sampled_talker_step(logits, sampling, *, suppress=(), history=(), steps=1):
    class FakeTalker:
        config = SimpleNamespace(text_config=SimpleNamespace(hidden_size=64))

        def __call__(self, **kwargs):
            return logits[None, None, :], mx.zeros((1, 1, 64))

        def prepare_inputs_for_generation(self, tokens, hidden, **kwargs):
            codes = mx.concatenate([tokens, mx.zeros((1, 15), dtype=mx.int32)], axis=-1)
            return mx.zeros((1, 1, 64)), codes

    runner = MlxTalkerModelRunner.__new__(MlxTalkerModelRunner)
    runner.tp_worker = SimpleNamespace(
        _mlx_runner=SimpleNamespace(model=FakeTalker(), _req_caches={"speech": []})
    )
    runner.outbox = Queue()
    data = SimpleNamespace(
        suppress_tokens=list(suppress),
        req=SimpleNamespace(sampling_params=sampling, output_ids=list(history)),
        output_ids=[],
        pending_feedback_queue=deque(),
        stage_payload=SimpleNamespace(request=SimpleNamespace(params={"stream": True})),
    )
    for _ in range(steps):
        result = runner.speech_step(
            SimpleNamespace(request_id="speech", data=data),
            mx.zeros((1, 1, 64)),
            mx.zeros((3, 1, 1), dtype=mx.int32),
        )
        data.req.output_ids.append(result.next_token_ids.item())
        assert data.output_ids == []
    return result


def test_talker_penalty_reads_live_history_before_terminalization():
    result = run_sampled_talker_step(
        mx.array([3.0, 2.0, 0.0]),
        SamplingParams(temperature=0.0, repetition_penalty=2.0),
        steps=2,
    )
    assert result.next_token_ids.item() == 1


@pytest.mark.parametrize("params", [{"seed": 17}, {"sampling_seed": 29}, {}])
def test_talker_prefill_uses_shared_resolved_seed(monkeypatch, params):
    from contextlib import nullcontext

    from sglang_omni.models.qwen3_omni.mlx import talker_prefill
    from sglang_omni.models.qwen3_omni.request_builders import (
        make_talker_scheduler_adapters,
    )
    from sglang_omni.pipeline.stage.stream_queue import StreamItem
    from sglang_omni.scheduling.pending_text_queue import PendingTextTensorQueue

    fake_builder = SimpleNamespace(
        build_prompt_prefill=lambda *args, **kwargs: {
            "input_embeds": torch.zeros(2, 64),
            "input_ids": [1, 2],
            "pending_text_queue": PendingTextTensorQueue(),
            "tts_pad_embed": torch.zeros(64),
            "tts_eos_embed": torch.zeros(64),
            "prompt_model_inputs": {},
        },
        append_text_chunk=lambda *args: None,
        mark_thinker_done=lambda *args: None,
    )
    monkeypatch.setattr(
        talker_prefill, "MlxTalkerPrefillBuilder", lambda **kwargs: fake_builder
    )
    builder, *_ = make_talker_scheduler_adapters(
        tokenizer=None,
        codec_vocab_size=3072,
        model=SimpleNamespace(config=SimpleNamespace(codec_eos_token_id=2150)),
        model_path="unused",
        thinker_config=None,
    )
    payload = StagePayload(
        request_id="seeded-speech",
        request=OmniRequest(inputs=[], params=params),
        data={},
    )
    payload.prefetched_chunks = [StreamItem(0, torch.tensor([3]), "thinker")]
    payload.prefetched_stream_done = True
    data = builder(payload)
    resolved = data.req.sampling_params.sampling_seed
    assert resolved == params.get("seed", params.get("sampling_seed", resolved))
    assert resolved is not None
    data.req.prefix_indices = []
    data.req.extend_range = SimpleNamespace(length=2)
    native = SimpleNamespace(
        position_deltas={}, _req_caches={}, _acquire_cache=lambda: []
    )
    runner = MlxTalkerModelRunner.__new__(MlxTalkerModelRunner)
    runner.tp_worker = SimpleNamespace(_mlx_runner=native)
    monkeypatch.setattr(runner, "mlx_stream_context", nullcontext)
    monkeypatch.setattr(
        "sglang_omni.models.qwen3_omni.talker_model_runner.QwenTalkerModelRunner.projected_prefill_slice",
        lambda **kwargs: torch.zeros(2, 64),
    )
    monkeypatch.setattr(
        runner,
        "speech_step",
        lambda *args: mx.random.uniform(shape=(4,)).tolist(),
    )
    mx.random.seed(resolved)
    expected = mx.random.uniform(shape=(4,)).tolist()
    mx.random.seed(999)
    actual = runner.custom_prefill_forward(
        None, None, [SimpleNamespace(request_id=payload.request_id, data=data)]
    )
    assert actual == expected


def test_talker_nucleus_is_normalized_and_shift_invariant(monkeypatch):
    from mlx_lm.sample_utils import apply_top_p, make_sampler

    from sglang_omni.models.qwen3_omni.mlx import talker_runner

    received = []

    def recording_sampler(**kwargs):
        actual_sampler = make_sampler(**kwargs)

        def sample(logprobs):
            received.append(np.array(logprobs))
            nucleus = np.array(apply_top_p(logprobs, kwargs["top_p"]))
            assert np.isfinite(nucleus).tolist() == [[True, True, False]]
            return actual_sampler(logprobs)

        return sample

    monkeypatch.setattr(talker_runner, "make_sampler", recording_sampler)
    for shift in (0.0, 100.0, -100.0):
        result = run_sampled_talker_step(
            mx.array([-10.0, -11.0, -12.0]) + shift,
            SamplingParams(temperature=0.8, top_p=0.9, top_k=2),
        )
        assert result.next_token_ids.item() in (0, 1)
    expected = torch.log_softmax(torch.tensor([[-10.0, -11.0, -12.0]]), dim=-1)
    for logprobs in received:
        np.testing.assert_allclose(logprobs, expected.numpy(), atol=1e-5)
        np.testing.assert_allclose(np.exp(logprobs).sum(), 1.0, atol=1e-5)


def test_talker_normalizes_after_suppression_and_repetition_penalty(monkeypatch):
    from mlx_lm.sample_utils import make_sampler

    from sglang_omni.models.qwen3_omni.mlx import talker_runner

    def recording_sampler(**kwargs):
        actual_sampler = make_sampler(**kwargs)

        def sample(logprobs):
            expected = torch.log_softmax(
                torch.tensor([[-float("inf"), -20.0, -11.0]]), dim=-1
            )
            np.testing.assert_allclose(np.array(logprobs), expected.numpy(), atol=1e-5)
            return actual_sampler(logprobs)

        return sample

    monkeypatch.setattr(talker_runner, "make_sampler", recording_sampler)
    result = run_sampled_talker_step(
        mx.array([100.0, -10.0, -11.0]),
        SamplingParams(temperature=0.8, top_p=0.9, top_k=2, repetition_penalty=2.0),
        suppress=(0,),
        history=(1,),
    )
    assert result.next_token_ids.item() == 2


@pytest.mark.parametrize(
    ("top_k", "expected_top_k"), [(-1, 0), (3072, 0), (4096, 0), (8, 8)]
)
def test_talker_translates_normalized_top_k_for_real_sampler(
    monkeypatch, top_k, expected_top_k
):
    from mlx_lm.sample_utils import make_sampler

    from sglang_omni.models.qwen3_omni.mlx import talker_runner

    observed = []

    def recording_sampler(**kwargs):
        observed.append(kwargs)
        return make_sampler(**kwargs)

    monkeypatch.setattr(talker_runner, "make_sampler", recording_sampler)
    sampling = SamplingParams(temperature=0.8, top_p=0.9, top_k=top_k)
    if top_k == -1:
        assert sampling.top_k == 1073741824
    else:
        pass
    logits = mx.full((3072,), -float("inf"))
    logits[:3] = mx.array([-10.0, -11.0, -12.0])
    result = run_sampled_talker_step(logits, sampling)
    assert observed == [{"temp": 0.8, "top_p": 0.9, "top_k": expected_top_k}]
    assert result.next_token_ids.item() in (0, 1)


@pytest.mark.parametrize("source", ["local", "hub"])
def test_runner_uses_resolved_checkpoint_for_component_and_config(
    checkpoint, monkeypatch, source
):
    from sglang_omni.models import weight_loader

    downloads = []
    hub_id = f"mlx-community/{checkpoint.name}"

    def cached_snapshot(model_path, *, local_files_only):
        assert model_path == hub_id
        downloads.append(model_path)
        return str(checkpoint)

    monkeypatch.setattr(weight_loader, "snapshot_download", cached_snapshot)
    model_path = str(checkpoint) if source == "local" else hub_id
    runner = OmniMlxRunner(
        model_path, disable_radix_cache=True, pool_size=64, trust_remote_code=False
    )
    embedding = load_component(model_path, "embedding")
    assert embedding(mx.array([2, 3])).shape == (2, 64)
    assert downloads == ([] if source == "local" else [model_path])
    pending = prefill(runner, make_request(), [2, 3])
    assert runner.prefill_finalize(pending) >= 0
    runner.remove_request("first")
