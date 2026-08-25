# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the ARK-ASR-3B adapter (CPU-only, no checkpoint download)."""

import inspect
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from sglang.srt.managers.schedule_batch import Modality, MultimodalDataItem
from transformers import WhisperConfig

import sglang_omni.model_runner.base as model_runner_base
import sglang_omni.models.arkasr.engine_builder as arkasr_builder
import sglang_omni.platforms as platforms
import sglang_omni.scheduling.bootstrap as bootstrap
import sglang_omni.scheduling.omni_scheduler as omni_scheduler
import sglang_omni.scheduling.sglang_backend as sglang_backend
from sglang_omni.models.arkasr import request_builders
from sglang_omni.models.arkasr.audio_lengths import (
    arkasr_audio_token_lengths,
    arkasr_num_audio_tokens,
)
from sglang_omni.models.arkasr.audio_tower import ArkAudioMLPAdapter, ArkAudioTower
from sglang_omni.models.arkasr.config import ArkasrPipelineConfig
from sglang_omni.models.arkasr.configuration_arkasr import ArkasrConfig
from sglang_omni.models.arkasr.request_builders import _build_suppressed_token_ids
from sglang_omni.models.arkasr.sglang_model import ArkasrForConditionalGeneration
from sglang_omni.models.arkasr.stages import create_sglang_arkasr_executor
from sglang_omni.models.registry import PIPELINE_CONFIG_REGISTRY
from tests.unit_test.fakes import FakeServerArgs


def _tiny_config():
    """Small ARK config for CPU shape tests (no checkpoint)."""
    whisper = WhisperConfig(
        d_model=32,
        encoder_layers=2,
        encoder_attention_heads=4,
        encoder_ffn_dim=64,
        num_mel_bins=8,
        max_source_positions=64,
    )
    return ArkasrConfig(
        whisper_config=whisper,
        merge_factor=4,
        hidden_size=48,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=64,
        vocab_size=256,
        audio_token_id=151663,
    )


def test_arkasr_config_registered():
    config = ArkasrPipelineConfig(model_path="AutoArk-AI/ARK-ASR-3B")
    assert config.entry_stage == "asr"
    assert config.stages[0].name == "asr"
    assert config.stages[0].terminal
    assert config.stages[0].factory.encoder_max_batch_size == 8
    assert config.stages[0].factory.prefill_coalesce_requests == 16
    assert config.stages[0].factory.prefill_coalesce_wait_ms == 32
    assert config.stages[0].factory.prefill_coalesce_when_idle is True
    assert config.stages[0].factory.prefill_coalesce_requires_pending_builds is True
    assert (
        PIPELINE_CONFIG_REGISTRY.get_config("ArkasrForConditionalGeneration")
        is ArkasrPipelineConfig
    )
    assert ArkasrPipelineConfig.mem_fraction_role_to_stage() == {"asr": "asr"}
    assert ArkasrPipelineConfig.generation_sglang_role_to_stage() == {
        "generation": "asr"
    }


def test_arkasr_stage_defaults():
    signature = inspect.signature(create_sglang_arkasr_executor)
    assert signature.parameters["max_running_requests"].default == 32
    assert signature.parameters["encoder_max_batch_size"].default == 8
    assert signature.parameters["request_build_max_workers"].default == 2
    assert signature.parameters["request_build_max_pending"].default == 16
    assert signature.parameters["prefill_coalesce_requests"].default == 16
    assert signature.parameters["prefill_coalesce_wait_ms"].default == 32.0
    assert signature.parameters["prefill_coalesce_when_idle"].default is True
    assert (
        signature.parameters["prefill_coalesce_requires_pending_builds"].default is True
    )
    assert signature.parameters["enable_pre_lm_encoder"].default is True
    assert signature.parameters["pre_lm_max_batch_size"].default == 8
    assert signature.parameters["pre_lm_max_batch_wait_ms"].default == 0
    assert signature.parameters["pre_lm_max_pending"].default == 32


def test_arkasr_pre_lm_group_matches_one_encoder_microbatch_by_default():
    """pre_lm_max_batch_size decides how many requests reach one
    get_audio_feature call; encoder_max_batch_size bounds how that call is
    executed. Equal defaults mean one drained group is one encoder microbatch,
    so the pre-LM service cannot silently widen a single encoder forward."""
    signature = inspect.signature(create_sglang_arkasr_executor)

    assert (
        signature.parameters["pre_lm_max_batch_size"].default
        == signature.parameters["encoder_max_batch_size"].default
    )


def test_arkasr_pre_lm_encoder_knobs_are_stage_configurable():
    factory = ArkasrPipelineConfig(model_path="AutoArk-AI/ARK-ASR-3B").stages[0].factory

    assert factory.request_build_max_workers == 2
    assert factory.request_build_max_pending == 16
    assert factory.enable_pre_lm_encoder is True
    assert factory.pre_lm_cache_max_entries == 4096
    assert factory.pre_lm_cache_size_bytes == 2 * 1024**3
    assert factory.pre_lm_max_batch_size == 8
    assert factory.pre_lm_max_batch_wait_ms == 0
    assert factory.pre_lm_max_pending == 32


def test_arkasr_rejects_invalid_pre_lm_batch_size():
    with pytest.raises(ValueError, match="pre_lm_max_batch_size"):
        arkasr_builder.ArkasrEngineBuilder(
            max_running_requests=32,
            encoder_max_batch_size=8,
            max_new_tokens=256,
            enable_async_decode=True,
            async_decode_min_batch_size=2,
            mem_fraction_static=None,
            mm_embedding_cache_size_bytes=0,
            enable_torch_compile=False,
            mm_attention_backend=None,
            request_build_max_workers=8,
            request_build_max_pending=32,
            prefill_coalesce_requests=16,
            prefill_coalesce_wait_ms=32.0,
            prefill_coalesce_when_idle=True,
            prefill_coalesce_requires_pending_builds=True,
            pre_lm_max_batch_size=0,
        )


def test_arkasr_rejects_invalid_pre_lm_pending_limit():
    with pytest.raises(ValueError, match="pre_lm_max_pending"):
        arkasr_builder.ArkasrEngineBuilder(
            max_running_requests=32,
            encoder_max_batch_size=8,
            max_new_tokens=256,
            enable_async_decode=True,
            async_decode_min_batch_size=2,
            mem_fraction_static=None,
            mm_embedding_cache_size_bytes=0,
            enable_torch_compile=False,
            mm_attention_backend=None,
            request_build_max_workers=2,
            request_build_max_pending=16,
            prefill_coalesce_requests=16,
            prefill_coalesce_wait_ms=32.0,
            prefill_coalesce_when_idle=True,
            prefill_coalesce_requires_pending_builds=True,
            pre_lm_max_pending=0,
        )


def test_arkasr_stage_default_enables_async_decode():
    signature = inspect.signature(create_sglang_arkasr_executor)

    assert signature.parameters["enable_async_decode"].default is True
    assert signature.parameters["async_decode_min_batch_size"].default == 2


def _stub_arkasr_engine_build(
    monkeypatch: pytest.MonkeyPatch,
    *,
    want_cuda_graph: bool,
    encoder_service: object,
) -> SimpleNamespace:
    """Stub every out-of-process dependency of ArkasrEngineBuilder.build().

    Returns the recorders the tests assert on. The worker stub stays minimal:
    helper internals are already covered in
    tests/unit_test/serve/test_sglang_bootstrap.py.
    """
    graph_init_workers: list[object] = []
    adapter_kwargs: dict[str, object] = {}
    encoder_service_kwargs: dict[str, object] = {}

    encoder_batch_sizes: list[int] = []
    model_worker = SimpleNamespace(
        gpu_id=0,
        model_runner=SimpleNamespace(
            model=SimpleNamespace(set_encoder_max_batch_size=encoder_batch_sizes.append)
        ),
    )
    infra = (want_cuda_graph, (model_worker, None, None, None, None, None, None))

    monkeypatch.setattr(
        platforms.current_platform, "get_device", lambda index: "cpu", raising=False
    )

    monkeypatch.setattr(
        arkasr_builder,
        "AutoTokenizer",
        SimpleNamespace(from_pretrained=lambda *a, **k: object()),
    )
    monkeypatch.setattr(
        arkasr_builder,
        "WhisperFeatureExtractor",
        SimpleNamespace(
            # note (jiannan-17): 2000 differs from the 3000 fallback, so the
            # context_length assertion below fails if the builder stops
            # reading nb_max_frames from the extractor.
            from_pretrained=lambda *a, **k: SimpleNamespace(nb_max_frames=2000)
        ),
    )
    monkeypatch.setattr(
        arkasr_builder,
        "AutoConfig",
        SimpleNamespace(
            # note (jiannan-17): sentinels differ from the builder fallbacks
            # (4 / 151663) so the assertions below prove the values were read
            # from the checkpoint config.
            from_pretrained=lambda *a, **k: SimpleNamespace(
                merge_factor=7, audio_token_id=4242
            )
        ),
    )
    monkeypatch.setattr(arkasr_builder, "init_mm_embedding_cache", lambda n: None)
    monkeypatch.setattr(
        arkasr_builder, "build_cache_namespace", lambda *a, **k: "testns"
    )
    monkeypatch.setattr(
        arkasr_builder,
        "ArkasrPreLMEncoderService",
        lambda *a, **k: (encoder_service_kwargs.update(k) or encoder_service),
    )
    monkeypatch.setattr(
        request_builders,
        "make_arkasr_scheduler_adapters",
        lambda **k: (adapter_kwargs.update(k) or object(), object()),
    )

    def _fake_server_args_builder(model_path, context_length, **overrides):
        server_args = FakeServerArgs(context_length=context_length, **overrides)
        server_args.cuda_graph_config = SimpleNamespace(
            decode=SimpleNamespace(
                max_bs=overrides["cuda_graph_max_bs"],
                bs=overrides["cuda_graph_bs"],
            ),
            prefill=SimpleNamespace(backend="disabled", bs=None, max_bs=None),
        )
        return server_args

    monkeypatch.setattr(
        sglang_backend, "build_sglang_server_args", _fake_server_args_builder
    )
    infra_kwargs_seen: dict[str, object] = {}

    def _fake_defer_infra(server_args, gpu_id, **kwargs):
        infra_kwargs_seen.update(kwargs)
        return infra

    monkeypatch.setattr(
        bootstrap,
        "create_sglang_infrastructure_defer_cuda_graph",
        _fake_defer_infra,
    )
    monkeypatch.setattr(
        bootstrap,
        "init_sglang_cuda_graphs",
        lambda worker: graph_init_workers.append(worker),
    )
    monkeypatch.setattr(
        model_runner_base,
        "ModelRunner",
        lambda *args, **kwargs: object(),
    )
    monkeypatch.setattr(sglang_backend, "SGLangOutputProcessor", lambda **k: object())
    monkeypatch.setattr(omni_scheduler, "OmniScheduler", SimpleNamespace)
    return SimpleNamespace(
        graph_init_workers=graph_init_workers,
        adapter_kwargs=adapter_kwargs,
        encoder_service_kwargs=encoder_service_kwargs,
        infra_kwargs_seen=infra_kwargs_seen,
        encoder_batch_sizes=encoder_batch_sizes,
        model_worker=model_worker,
    )


@pytest.mark.parametrize("want_cuda_graph", [True, False])
def test_arkasr_factory_triggers_deferred_cuda_graph_capture(
    monkeypatch: pytest.MonkeyPatch, want_cuda_graph: bool
) -> None:
    """The factory defers capture, then calls bootstrap.init_sglang_cuda_graphs
    exactly when the deferred-capture probe asked for graphs."""
    stub = _stub_arkasr_engine_build(
        monkeypatch,
        want_cuda_graph=want_cuda_graph,
        encoder_service=SimpleNamespace(close=lambda: None),
    )

    scheduler = create_sglang_arkasr_executor(
        "AutoArk-AI/ARK-ASR-3B",
        mm_attention_backend="triton_attn",
        enable_async_decode=False,
        async_decode_min_batch_size=4,
        prefill_coalesce_requests=8,
        prefill_coalesce_wait_ms=24.0,
    )

    assert stub.graph_init_workers == ([stub.model_worker] if want_cuda_graph else [])
    # note (jiannan-17): the scheduler fake records its kwargs, proving the
    # builder forwards the stage knobs instead of dropping them.
    assert scheduler.request_build_max_workers == 2
    assert scheduler.request_build_max_pending == 16
    assert scheduler.enable_async_decode is False
    assert scheduler.async_decode_min_batch_size == 4
    assert scheduler.prefill_coalesce_requests == 8
    assert scheduler.prefill_coalesce_wait_ms == 24.0
    assert scheduler.prefill_coalesce_when_idle is True
    assert scheduler.prefill_coalesce_requires_pending_builds is True
    assert stub.adapter_kwargs["merge_factor"] == 7
    assert stub.adapter_kwargs["audio_token_id"] == 4242
    assert stub.adapter_kwargs["max_new_tokens"] == 256
    # note (jiannan-17): context_length and model_arch_override moved from
    # stages.py into the builder. Keep both pinned so a missing value fails
    # here instead of at serve time.
    assert scheduler.server_args.context_length == 2000 // 2 + 256 + 8
    assert (
        stub.infra_kwargs_seen["model_arch_override"]
        == "ArkasrForConditionalGeneration"
    )
    assert stub.encoder_batch_sizes == [8]
    assert stub.encoder_service_kwargs["max_queue_size"] == 32


def test_arkasr_pre_lm_encoder_reaches_request_builder_and_shutdown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The encoder service is handed to the request builder (so encoding runs
    before LM admission) and its close() is registered as the scheduler's
    shutdown callback (so the worker thread cannot outlive the stage)."""
    encoder_service = SimpleNamespace(close=lambda: None)
    stub = _stub_arkasr_engine_build(
        monkeypatch, want_cuda_graph=False, encoder_service=encoder_service
    )

    scheduler = create_sglang_arkasr_executor("AutoArk-AI/ARK-ASR-3B")

    assert stub.adapter_kwargs["audio_encoder_service"] is encoder_service
    assert scheduler.shutdown_callback == encoder_service.close
    # Note (Akazaakane): The model-internal bound still applies behind the
    # pre-LM service because the two batch limits are independent.
    assert stub.encoder_batch_sizes == [8]


def test_arkasr_pre_lm_encoder_can_be_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With the service off, the request builder falls back to encoding inside
    the LM forward and no shutdown callback is registered."""
    stub = _stub_arkasr_engine_build(
        monkeypatch,
        want_cuda_graph=False,
        encoder_service=SimpleNamespace(close=lambda: None),
    )

    scheduler = create_sglang_arkasr_executor(
        "AutoArk-AI/ARK-ASR-3B", enable_pre_lm_encoder=False
    )

    assert stub.adapter_kwargs["audio_encoder_service"] is None
    assert not hasattr(scheduler, "shutdown_callback")
    assert stub.encoder_batch_sizes == [8]


def test_arkasr_audio_token_count():
    # matches checkpoint processor: (mel+1)//2 // merge_factor(4)
    assert arkasr_num_audio_tokens(400) == 50  # (401//2)=200 -> 50
    assert arkasr_num_audio_tokens(3000) == 375  # (3001//2)=1500 -> 375
    assert arkasr_num_audio_tokens(1) == 1  # floor at 1
    # list form
    assert arkasr_audio_token_lengths([400, 8]) == [50, 1]


def test_arkasr_config_keeps_lm_params_at_top_level():
    cfg = _tiny_config()
    assert cfg.num_attention_heads == 4
    assert cfg.num_key_value_heads == 2
    assert cfg.hidden_size == 48
    assert cfg.num_hidden_layers == 2


def test_arkasr_import_does_not_register_auto_config():
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING

    assert "arkasr" not in CONFIG_MAPPING._extra_content


def test_ark_audio_tower_forward_shape():
    torch.manual_seed(0)
    cfg = _tiny_config()
    adapter = ArkAudioMLPAdapter(cfg).eval()
    mel_frames = 40  # -> conv2 stride2 -> ~20 -> merge4 -> ~5 audio tokens
    mel = torch.randn(1, cfg.whisper_config.num_mel_bins, mel_frames)
    with torch.no_grad():
        out = adapter(mel)
    assert out.dim() == 3
    assert out.size(0) == 1
    assert out.size(-1) == cfg.hidden_size  # projected to LLM hidden
    assert out.size(1) >= 1


def _tiny_ark_audio_mm_model() -> ArkasrForConditionalGeneration:
    """Encoder-only ARK model used to test get_audio_feature without an LLM."""
    model = ArkasrForConditionalGeneration.__new__(ArkasrForConditionalGeneration)
    nn.Module.__init__(model)
    model.audio_encoder = ArkAudioMLPAdapter(_tiny_config()).eval()
    model.encoder_max_batch_size = model.DEFAULT_ENCODER_MAX_BATCH_SIZE
    return model


def _ark_audio_item(
    feature: torch.Tensor, valid_frames: int, *, hash_id: int
) -> MultimodalDataItem:
    return MultimodalDataItem(
        modality=Modality.AUDIO,
        hash=hash_id,
        feature=feature,
        model_specific_data={
            "feature_attention_mask": torch.cat(
                [
                    torch.ones((1, valid_frames), dtype=torch.long),
                    torch.zeros(
                        (1, feature.shape[-1] - valid_frames), dtype=torch.long
                    ),
                ],
                dim=1,
            )
        },
    )


def test_ark_get_audio_feature_batched_matches_serial_and_uses_one_encoder_call():
    torch.manual_seed(2)
    model = _tiny_ark_audio_mm_model()
    lengths = [9, 18, 25]
    items = [
        _ark_audio_item(torch.randn(1, 8, length), length, hash_id=index + 1)
        for index, length in enumerate(lengths)
    ]
    calls = []

    def record_call(_module, args, kwargs):
        calls.append((args[0].shape, kwargs.get("attention_mask")))

    handle = model.audio_encoder.register_forward_pre_hook(
        record_call, with_kwargs=True
    )
    try:
        with torch.no_grad():
            batched = model.get_audio_feature(items)
    finally:
        handle.remove()
    with torch.no_grad():
        serial = torch.cat([model.get_audio_feature([item]) for item in items], dim=0)

    assert len(calls) == 1
    assert calls[0][0] == (len(items), 8, max(lengths))
    assert calls[0][1] is not None
    assert calls[0][1].shape == (len(items), max(lengths))
    assert batched.shape == (
        sum(arkasr_num_audio_tokens(length) for length in lengths),
        48,
    )
    assert torch.allclose(batched, serial, atol=1e-5, rtol=1e-5)


def test_ark_get_audio_feature_splits_large_batches_in_order():
    torch.manual_seed(4)
    model = _tiny_ark_audio_mm_model()
    model.set_encoder_max_batch_size(2)
    lengths = [9, 18, 25, 12, 20]
    items = [
        _ark_audio_item(torch.randn(1, 8, length), length, hash_id=200 + index)
        for index, length in enumerate(lengths)
    ]
    calls = []

    def record_call(_module, args, kwargs):
        calls.append((args[0].shape, kwargs.get("attention_mask")))

    handle = model.audio_encoder.register_forward_pre_hook(
        record_call, with_kwargs=True
    )
    try:
        with torch.no_grad():
            batched = model.get_audio_feature(items)
            serial = torch.cat(
                [model.get_audio_feature([item]) for item in items], dim=0
            )
    finally:
        handle.remove()

    assert [shape[0][0] for shape in calls[:3]] == [2, 2, 1]
    assert torch.allclose(batched, serial, atol=1e-5, rtol=1e-5)


def test_ark_get_audio_feature_masks_pre_padded_garbage():
    torch.manual_seed(3)
    model = _tiny_ark_audio_mm_model()
    lengths = [9, 25]
    t_max = max(lengths)
    items = []
    for index, length in enumerate(lengths):
        feature = torch.zeros(1, 8, t_max)
        feature[:, :, :length] = torch.randn(1, 8, length)
        if length < t_max:
            feature[:, :, length:] = 50.0
        items.append(_ark_audio_item(feature, length, hash_id=100 + index))

    with torch.no_grad():
        batched = model.get_audio_feature(items)
        serial = torch.cat([model.get_audio_feature([item]) for item in items], dim=0)

    assert batched.shape == serial.shape
    assert torch.allclose(batched, serial, atol=1e-5, rtol=1e-5)


def test_ark_tower_rope_toggle():
    cfg = _tiny_config()
    tower = ArkAudioTower(cfg)
    assert tower.use_rope is True
    assert hasattr(tower, "rotary_embedding")


def test_ark_suppressed_token_ids():
    """The generation-suppression set must cover every ``<...>`` added/special
    token except EOS (mirrors the checkpoint's bad_words_ids), so markers like
    <|audio|> / <tool_call> cannot leak into transcripts."""

    class _FakeTok:
        eos_token_id = 100
        all_special_ids = [100, 101, 102]

        def get_added_vocab(self):
            return {
                "<|im_start|>": 101,
                "<|audio|>": 103,
                "<tool_call>": 104,
                "hello": 105,  # normal token: must NOT be suppressed
                "</tool_call>": 106,
            }

    ids = _build_suppressed_token_ids(_FakeTok())
    assert 100 not in ids  # EOS kept
    assert 101 in ids and 102 in ids  # special ids
    assert 103 in ids and 104 in ids and 106 in ids  # <...> added tokens
    assert 105 not in ids  # normal token untouched
    assert ids == sorted(ids)  # deterministic order


def test_ark_encoder_layer_fp16_clamp():
    """The encoder layer must clamp the final residual under fp16 so large
    activations stay finite (mirrors the checkpoint's modeling_audio.py). This
    guards the exposed dtype="float16" path; the bf16 path must be a no-op.
    """
    from sglang_omni.models.arkasr.audio_tower import WhisperSpecialEncoderLayer

    cfg = _tiny_config().whisper_config
    cfg._attn_implementation = "sdpa"

    # fp16: drive fc2 output large enough to exceed fp16 max (~65504) so an
    # unclamped residual would overflow to +/-inf. Clamp must keep it finite.
    torch.manual_seed(0)
    layer = WhisperSpecialEncoderLayer(cfg).eval().half()
    with torch.no_grad():
        layer.fc2.weight.fill_(50.0)
        layer.fc2.bias.fill_(65000.0)
        x = torch.full((1, 6, cfg.d_model), 300.0, dtype=torch.float16)
        out_fp16 = layer(x)[0]
    assert out_fp16.dtype == torch.float16
    assert torch.isfinite(out_fp16).all()  # clamp prevented inf/NaN
    # clamp keeps values strictly under the fp16 ceiling (target is max-1000;
    # fp16 rounding lands a few ulps above the exact target, still < max).
    fp16_max = torch.finfo(torch.float16).max
    assert out_fp16.abs().max().item() < fp16_max
    clamp_value = fp16_max - 1000

    # bf16: same weights do not overflow (fp32-range exponent) and the clamp
    # branch never fires, so the bf16 path is unchanged.
    torch.manual_seed(0)
    layer_bf16 = WhisperSpecialEncoderLayer(cfg).eval().to(torch.bfloat16)
    with torch.no_grad():
        layer_bf16.fc2.weight.fill_(50.0)
        layer_bf16.fc2.bias.fill_(65000.0)
        xb = torch.full((1, 6, cfg.d_model), 300.0, dtype=torch.bfloat16)
        out_bf16 = layer_bf16(xb)[0]
    assert out_bf16.dtype == torch.bfloat16
    assert torch.isfinite(out_bf16).all()
    # bf16 values exceed the fp16 clamp ceiling -> proves no clamp was applied.
    assert out_bf16.abs().max().item() > clamp_value
