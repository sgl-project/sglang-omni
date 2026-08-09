# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the ARK-ASR-3B adapter (CPU-only, no checkpoint download)."""

import inspect
from types import SimpleNamespace

import pytest
import torch
from transformers import WhisperConfig

import sglang_omni.models.arkasr.engine_builder as arkasr_builder
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
    assert (
        PIPELINE_CONFIG_REGISTRY.get_config("ArkasrForConditionalGeneration")
        is ArkasrPipelineConfig
    )


def test_arkasr_stage_defaults():
    signature = inspect.signature(create_sglang_arkasr_executor)
    assert signature.parameters["max_running_requests"].default == 32
    assert signature.parameters["request_build_max_workers"].default == 2
    assert signature.parameters["request_build_max_pending"].default == 16


@pytest.mark.parametrize("want_cuda_graph", [True, False])
def test_arkasr_factory_triggers_deferred_cuda_graph_capture(
    monkeypatch: pytest.MonkeyPatch, want_cuda_graph: bool
) -> None:
    """The factory defers capture, then calls bootstrap.init_sglang_cuda_graphs
    exactly when the deferred-capture probe asked for graphs. The worker stub
    stays minimal: helper internals are already covered in
    tests/unit_test/serve/test_sglang_bootstrap.py."""
    graph_init_workers: list[object] = []
    adapter_kwargs: dict[str, object] = {}

    model_worker = SimpleNamespace(
        gpu_id=0, model_runner=SimpleNamespace(model=object())
    )
    infra = (want_cuda_graph, (model_worker, None, None, None, None, None, None))

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
    monkeypatch.setattr(sglang_backend, "SGLangOutputProcessor", lambda **k: object())
    monkeypatch.setattr(
        omni_scheduler, "OmniScheduler", lambda **k: SimpleNamespace(**k)
    )

    scheduler = create_sglang_arkasr_executor(
        "AutoArk-AI/ARK-ASR-3B", mm_attention_backend="triton_attn"
    )

    assert graph_init_workers == ([model_worker] if want_cuda_graph else [])
    # note (jiannan-17): the scheduler fake records its kwargs, proving the
    # builder forwards the stage knobs instead of dropping them.
    assert scheduler.request_build_max_workers == 2
    assert scheduler.request_build_max_pending == 16
    assert adapter_kwargs["merge_factor"] == 7
    assert adapter_kwargs["audio_token_id"] == 4242
    assert adapter_kwargs["max_new_tokens"] == 256
    # note (jiannan-17): context_length and model_arch_override moved from
    # stages.py into the builder. Keep both pinned so a missing value fails
    # here instead of at serve time.
    assert scheduler.server_args.context_length == 2000 // 2 + 256 + 8
    assert infra_kwargs_seen["model_arch_override"] == "ArkasrForConditionalGeneration"


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
