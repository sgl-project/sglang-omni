"""Exercise reference conditioning and prefix memory admission on CPU."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

pytest.importorskip("sglang")

import torch
from PIL import Image
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages import base as stage_base
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit
from tqdm.auto import tqdm
from transformers import PreTrainedTokenizerFast
from transformers.cache_utils import DynamicCache

from sglang_omni.models.sensenova_u1.native.generation import SenseNovaU1GenerationStage
from sglang_omni.models.sensenova_u1.native.sampling import SenseNovaU1SamplingParams
from sglang_omni.native_generation import cancellation
from sglang_omni.vendor.sensenova_u1 import image_edit
from sglang_omni.vendor.sensenova_u1.configuration_neo_chat import NEOChatConfig
from sglang_omni.vendor.sensenova_u1.image_edit import build_it2i_queries, it2i_generate
from sglang_omni.vendor.sensenova_u1.image_edit_memory import (
    estimate_it2i_prefix_memory_bytes,
)
from sglang_omni.vendor.sensenova_u1.modeling_neo_chat import (
    NEOChatModel,
    prepare_flash_kv_cache,
)


@pytest.fixture
def edit_model_and_tokenizer() -> tuple[NEOChatModel, PreTrainedTokenizerFast]:
    torch.manual_seed(0)
    config = NEOChatConfig(
        vision_config={
            "hidden_size": 32,
            "llm_hidden_size": 32,
            "patch_size": 16,
            "max_position_embeddings_vision": 64,
        },
        llm_config={
            "architectures": ["Qwen3ForCausalLM"],
            "vocab_size": 16,
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 1,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 8,
            "max_position_embeddings": 2048,
            "max_position_embeddings_hw": 64,
        },
        template="neo1_0",
        use_pixel_head=False,
        fm_head_layers=2,
        concat_time_token_num=0,
        noise_scale=1.0,
        noise_scale_mode="constant",
        noise_scale_base_image_seq_len=1,
        add_noise_scale_embedding=True,
        noise_scale_max_value=10.0,
    )
    model = NEOChatModel(config).eval()
    vocabulary = {"[PAD]": 0, "[UNK]": 1, "<img>": 2, "</img>": 3, "<IMG_CONTEXT>": 4}
    tokenizer_backend = Tokenizer(WordLevel(vocabulary, unk_token="[UNK]"))
    tokenizer_backend.pre_tokenizer = WhitespaceSplit()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer_backend,
        unk_token="[UNK]",
        pad_token="[PAD]",
        additional_special_tokens=["<img>", "</img>", "<IMG_CONTEXT>"],
    )
    return model, tokenizer


@pytest.mark.parametrize(
    "cfg_scale,img_cfg_scale", [(1.0, 1.0), (4.0, 1.0), (2.0, 2.0), (4.0, 1.5)]
)
def test_edit_denoising_uses_references_and_is_repeatable(
    edit_model_and_tokenizer: tuple[NEOChatModel, PreTrainedTokenizerFast],
    cfg_scale: float,
    img_cfg_scale: float,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, tokenizer = edit_model_and_tokenizer
    extract_feature = Mock(wraps=model.extract_feature)
    monkeypatch.setattr(model, "extract_feature", extract_feature)
    timestep_forward = Mock(wraps=model.fm_modules["timestep_embedder"].forward)
    noise_forward = Mock(wraps=model.fm_modules["noise_scale_embedder"].forward)
    monkeypatch.setattr(
        model.fm_modules["timestep_embedder"], "forward", timestep_forward
    )
    monkeypatch.setattr(
        model.fm_modules["noise_scale_embedder"], "forward", noise_forward
    )
    references = [Image.new("RGB", (32, 32), color) for color in ["red", "blue"]]
    parameters = dict(
        image_size=(64, 64),
        num_steps=2,
        steps=range(2),
        cfg_scale=cfg_scale,
        img_cfg_scale=img_cfg_scale,
        seed=7,
        timestep_shift=3.0,
        t_eps=0.02,
        memory_budget_bytes=1024**3,
    )
    first = it2i_generate(model, tokenizer, "edit", references, **parameters)
    repeated = it2i_generate(model, tokenizer, "edit", references, **parameters)
    changed = it2i_generate(
        model, tokenizer, "edit", list(reversed(references)), **parameters
    )
    assert first.shape == (1, 3, 64, 64)
    # note (Wes): Native CUDA embeds four image tokens per step, rather than broadcasting a precomputed embedding.
    expected_steps = [0.0, 0.25] * 3
    assert timestep_forward.call_count == len(expected_steps)
    assert noise_forward.call_count == len(expected_steps)
    for call, timestep in zip(timestep_forward.call_args_list, expected_steps):
        torch.testing.assert_close(call.args[0], torch.full((4,), timestep))
    for call in noise_forward.call_args_list:
        torch.testing.assert_close(call.args[0], torch.full((4,), 0.1))
    assert torch.isfinite(first).all()
    assert torch.equal(first, repeated)
    assert not torch.equal(first, changed)
    changed_guidance = it2i_generate(
        model,
        tokenizer,
        "edit",
        references,
        **(parameters | {"img_cfg_scale": img_cfg_scale + 0.5}),
    )
    assert not torch.equal(first, changed_guidance)
    reference_encoding_calls = [
        call
        for call in extract_feature.call_args_list
        if call.kwargs["gen_model"] is False
    ]
    assert len(reference_encoding_calls) == 4


@pytest.mark.parametrize("budget_offset_bytes", [-1, 0])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_edit_prefix_memory_budget_boundary(
    edit_model_and_tokenizer: tuple[NEOChatModel, PreTrainedTokenizerFast],
    monkeypatch: pytest.MonkeyPatch,
    budget_offset_bytes: int,
    dtype: torch.dtype,
) -> None:
    model, tokenizer = edit_model_and_tokenizer
    model.to(dtype=dtype)
    references = [Image.new("RGB", (32, 32), "red")]
    queries = build_it2i_queries(model, "edit", [(32, 32)], 4.0, 1.0)
    required_memory_bytes = estimate_it2i_prefix_memory_bytes(
        model,
        [len(tokenizer.encode(query)) for query in queries],
        (64, 64),
        [(32, 32)],
    )
    normalize = Mock(wraps=image_edit.load_image_native)
    encode = Mock(wraps=model.extract_feature)
    prefill = Mock(wraps=model.language_model.model.forward)
    monkeypatch.setattr(image_edit, "load_image_native", normalize)
    monkeypatch.setattr(model, "extract_feature", encode)
    monkeypatch.setattr(model.language_model.model, "forward", prefill)
    parameters = dict(
        image_size=(64, 64),
        num_steps=2,
        steps=range(2),
        cfg_scale=4.0,
        img_cfg_scale=1.0,
        seed=7,
        timestep_shift=3.0,
        t_eps=0.02,
        memory_budget_bytes=required_memory_bytes + budget_offset_bytes,
    )
    if budget_offset_bytes < 0:
        with pytest.raises(ValueError, match="exceeding the available budget"):
            it2i_generate(model, tokenizer, "edit", references, **parameters)
        normalize.assert_not_called()
        encode.assert_not_called()
        prefill.assert_not_called()
    else:
        output = it2i_generate(model, tokenizer, "edit", references, **parameters)
        assert output.shape == (1, 3, 64, 64)
        assert torch.isfinite(output).all()


@pytest.mark.parametrize(
    "reference_count,reference_size_pixels,prompt,output_size_pixels",
    [
        (64, (512, 512), "edit", (64, 64)),
        (1, (32, 32), "edit " * 1024, (64, 64)),
        (1, (32, 32), "edit", (4096, 4096)),
    ],
)
def test_edit_prefix_budget_accounts_for_references_text_and_output(
    edit_model_and_tokenizer: tuple[NEOChatModel, PreTrainedTokenizerFast],
    monkeypatch: pytest.MonkeyPatch,
    reference_count: int,
    reference_size_pixels: tuple[int, int],
    prompt: str,
    output_size_pixels: tuple[int, int],
) -> None:
    model, tokenizer = edit_model_and_tokenizer
    queries = build_it2i_queries(model, "edit", [(32, 32)], 4.0, 1.0)
    memory_budget_bytes = estimate_it2i_prefix_memory_bytes(
        model,
        [len(tokenizer.encode(query)) for query in queries],
        (64, 64),
        [(32, 32)],
    )
    references = [Image.new("RGB", reference_size_pixels)] * reference_count
    normalize = Mock(wraps=image_edit.load_image_native)
    monkeypatch.setattr(image_edit, "load_image_native", normalize)
    with pytest.raises(ValueError, match="exceeding the available budget"):
        it2i_generate(
            model,
            tokenizer,
            prompt,
            references,
            image_size=output_size_pixels,
            num_steps=2,
            steps=range(2),
            cfg_scale=4.0,
            img_cfg_scale=1.0,
            seed=7,
            timestep_shift=3.0,
            t_eps=0.02,
            memory_budget_bytes=memory_budget_bytes,
        )
    normalize.assert_not_called()


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_prefix_estimate_covers_observed_softmax_buffers(
    edit_model_and_tokenizer: tuple[NEOChatModel, PreTrainedTokenizerFast],
    monkeypatch: pytest.MonkeyPatch,
    dtype: torch.dtype,
) -> None:
    model, tokenizer = edit_model_and_tokenizer
    model.to(dtype=dtype)
    prompt = "edit " * 256
    queries = build_it2i_queries(model, prompt, [(32, 32)], 4.0, 1.0)
    token_lengths = [len(tokenizer.encode(query)) for query in queries]
    live_attention_bytes = 0
    softmax_storage_bytes: dict[int, int] = {}
    softmax = torch.nn.functional.softmax
    dropout = torch.nn.functional.dropout

    def observe_softmax(
        scores: torch.Tensor, dim: int, dtype: torch.dtype
    ) -> torch.Tensor:
        nonlocal softmax_storage_bytes
        probabilities = softmax(scores, dim=dim, dtype=dtype)
        softmax_storage_bytes = {
            tensor.untyped_storage().data_ptr(): tensor.untyped_storage().nbytes()
            for tensor in (scores, probabilities)
        }
        return probabilities

    def observe_dropout(
        probabilities: torch.Tensor, p: float, training: bool
    ) -> torch.Tensor:
        nonlocal live_attention_bytes
        # note (Wes): Dropout receives the converted probabilities; FP32 conversion reuses storage.
        probability_storage = probabilities.untyped_storage()
        attention_storage_bytes = softmax_storage_bytes | {
            probability_storage.data_ptr(): probability_storage.nbytes()
        }
        live_attention_bytes = max(
            live_attention_bytes, sum(attention_storage_bytes.values())
        )
        return dropout(probabilities, p=p, training=training)

    monkeypatch.setattr(torch.nn.functional, "softmax", observe_softmax)
    monkeypatch.setattr(torch.nn.functional, "dropout", observe_dropout)
    it2i_generate(
        model,
        tokenizer,
        prompt,
        [Image.new("RGB", (32, 32))],
        image_size=(64, 64),
        num_steps=2,
        steps=range(2),
        cfg_scale=4.0,
        img_cfg_scale=1.0,
        seed=7,
        timestep_shift=3.0,
        t_eps=0.02,
        memory_budget_bytes=1024**3,
    )
    assert live_attention_bytes > 0
    assert (
        estimate_it2i_prefix_memory_bytes(model, token_lengths, (64, 64), [(32, 32)])
        >= live_attention_bytes
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("reference_sizes_pixels", [[(32, 32)], [(64, 32), (32, 64)]])
@torch.no_grad()
def test_prefix_estimate_includes_retained_reference_storage(
    edit_model_and_tokenizer: tuple[NEOChatModel, PreTrainedTokenizerFast],
    dtype: torch.dtype,
    reference_sizes_pixels: list[tuple[int, int]],
) -> None:
    """Reference storage must count during prefix forwards, but not after it is released."""
    model, _ = edit_model_and_tokenizer
    model.to(dtype=dtype)
    reference_patches = []
    reference_grids = []
    for size_pixels in reference_sizes_pixels:
        patches, grid = image_edit.load_image_native(
            Image.new("RGB", size_pixels), model.patch_size, model.downsample_ratio
        )
        reference_patches.append(patches.to(dtype=dtype))
        reference_grids.append(grid)
    pixel_values = torch.cat(reference_patches)
    reference_embeddings = model.extract_feature(
        pixel_values, grid_hw=torch.cat(reference_grids), gen_model=False
    )
    reference_storages = {
        tensor.untyped_storage().data_ptr(): tensor.untyped_storage().nbytes()
        for tensor in (*reference_patches, pixel_values, reference_embeddings)
    }
    without_references_bytes = estimate_it2i_prefix_memory_bytes(
        model, [512, 512], (64, 64), []
    )
    with_references_bytes = estimate_it2i_prefix_memory_bytes(
        model, [512, 512], (64, 64), reference_sizes_pixels
    )
    assert with_references_bytes - without_references_bytes == sum(
        reference_storages.values()
    )
    assert estimate_it2i_prefix_memory_bytes(
        model, [64, 64], (2048, 2048), reference_sizes_pixels
    ) == estimate_it2i_prefix_memory_bytes(model, [64, 64], (2048, 2048), [])


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_prefix_estimate_covers_allocated_caches(
    edit_model_and_tokenizer: tuple[NEOChatModel, PreTrainedTokenizerFast],
    dtype: torch.dtype,
) -> None:
    model, _ = edit_model_and_tokenizer
    model.to(dtype=dtype)
    caches = []
    for _ in range(2):
        cache = DynamicCache(config=model.config.llm_config)
        cache.update(
            torch.zeros(1, 2, 64, 8, dtype=dtype),
            torch.zeros(1, 2, 64, 8, dtype=dtype),
            layer_idx=0,
        )
        prepare_flash_kv_cache(cache, current_len=4096, batch_size=1)
        caches.append(cache)
    allocated_cache_bytes = sum(
        tensor.nbytes
        for cache in caches
        for layer in cache.layers
        for tensor in (
            layer.keys,
            layer.values,
            layer.flash_k_cache,
            layer.flash_v_cache,
        )
    )
    assert (
        estimate_it2i_prefix_memory_bytes(model, [64, 64], (2048, 2048), [(32, 32)])
        >= allocated_cache_bytes
    )


def test_prefix_attention_is_budgeted_sequentially(
    edit_model_and_tokenizer: tuple[NEOChatModel, PreTrainedTokenizerFast],
) -> None:
    model, _ = edit_model_and_tokenizer
    one_prefix_bytes = estimate_it2i_prefix_memory_bytes(
        model, [4096], (64, 64), [(32, 32)]
    )
    two_prefixes_bytes = estimate_it2i_prefix_memory_bytes(
        model, [4096, 4096], (64, 64), [(32, 32)]
    )
    assert one_prefix_bytes < two_prefixes_bytes < 2 * one_prefix_bytes


def test_workspace_reserve_rejects_before_reference_encoding(
    edit_model_and_tokenizer: tuple[NEOChatModel, PreTrainedTokenizerFast],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, tokenizer = edit_model_and_tokenizer
    monkeypatch.setattr(
        stage_base,
        "get_global_server_args",
        lambda: SimpleNamespace(comfyui_mode=False),
    )
    monkeypatch.setattr(
        torch.cuda, "mem_get_info", Mock(return_value=(16 * 1024**2, 0))
    )
    monkeypatch.setattr(torch.cuda, "memory_reserved", Mock(return_value=0))
    monkeypatch.setattr(torch.cuda, "memory_allocated", Mock(return_value=0))
    sampling = SenseNovaU1SamplingParams(
        prompt="edit",
        image_path="reference.png",
        width=64,
        height=64,
        num_inference_steps=2,
        seed=7,
        generator_device="cpu",
    )
    request = Req(sampling_params=sampling)
    sampling.apply_request_extra(request)
    request.condition_image = [Image.new("RGB", (32, 32))]
    encode = Mock(wraps=model.extract_feature)
    monkeypatch.setattr(model, "extract_feature", encode)
    with pytest.raises(ValueError, match="exceeding the available budget of 0 bytes"):
        SenseNovaU1GenerationStage(model, tokenizer).forward(request, None)
    encode.assert_not_called()


def test_edit_prefix_budget_includes_all_guidance_caches(
    edit_model_and_tokenizer: tuple[NEOChatModel, PreTrainedTokenizerFast],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, tokenizer = edit_model_and_tokenizer
    queries = build_it2i_queries(model, "edit", [(32, 32)], 1.0, 1.0)
    memory_budget_bytes = estimate_it2i_prefix_memory_bytes(
        model,
        [len(tokenizer.encode(query)) for query in queries],
        (4096, 4096),
        [(32, 32)],
    )
    normalize = Mock(wraps=image_edit.load_image_native)
    monkeypatch.setattr(image_edit, "load_image_native", normalize)
    with pytest.raises(ValueError, match="exceeding the available budget"):
        it2i_generate(
            model,
            tokenizer,
            "edit",
            [Image.new("RGB", (32, 32))],
            image_size=(4096, 4096),
            num_steps=2,
            steps=range(2),
            cfg_scale=4.0,
            img_cfg_scale=1.5,
            seed=7,
            timestep_shift=3.0,
            t_eps=0.02,
            memory_budget_bytes=memory_budget_bytes,
        )
    normalize.assert_not_called()


@pytest.mark.parametrize("armed", [False, True])
def test_edit_loop_checks_cancellation_between_steps(
    edit_model_and_tokenizer: tuple[NEOChatModel, PreTrainedTokenizerFast],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    armed: bool,
) -> None:
    """An edit must stop at the next step after cancellation and release its caches."""
    model, tokenizer = edit_model_and_tokenizer
    monkeypatch.setattr(
        stage_base,
        "get_global_server_args",
        lambda: SimpleNamespace(comfyui_mode=False),
    )
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: (4 * 1024**3, 0))
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda device: 0)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda device: 0)
    monkeypatch.setattr(cancellation.CancellationState, "enabled", True)
    original_progress_bar = stage_base.PipelineStage.progress_bar

    def progress_bar(
        stage: stage_base.PipelineStage, iterable: range, *, batch: Req
    ) -> tqdm:
        return cancellation.around_progress_bar(
            original_progress_bar, stage, iterable, batch=batch
        )

    monkeypatch.setattr(stage_base.PipelineStage, "progress_bar", progress_bar)
    sampling = SenseNovaU1SamplingParams(
        prompt="edit",
        image_path="reference.png",
        width=64,
        height=64,
        num_inference_steps=3,
        guidance_scale=4.0,
        seed=7,
        generator_device="cpu",
        output_path=str(tmp_path),
    )
    request = Req(sampling_params=sampling)
    sampling.apply_request_extra(request)
    request.condition_image = [Image.new("RGB", (32, 32))]
    stage = SenseNovaU1GenerationStage(model, tokenizer)
    baseline = stage.forward(request, None)
    if armed:
        (tmp_path / cancellation.ARMED_MARKER).touch()
    else:
        pass
    unpatchify = model.unpatchify

    def cancel_after_step(
        image_latents: torch.Tensor, patch_size: int, height: int, width: int
    ) -> torch.Tensor:
        result = unpatchify(image_latents, patch_size, height, width)
        (tmp_path / cancellation.CANCELLED_MARKER).touch()
        return result

    completed_steps = Mock(side_effect=cancel_after_step)
    clear_cache = Mock(wraps=image_edit.clear_flash_kv_cache)
    monkeypatch.setattr(model, "unpatchify", completed_steps)
    monkeypatch.setattr(image_edit, "clear_flash_kv_cache", clear_cache)
    if armed:
        with pytest.raises(cancellation.GenerationCancelled):
            stage.forward(request, None)
        assert completed_steps.call_count == 1
    else:
        output = stage.forward(request, None)
        assert completed_steps.call_count == 3
        assert torch.equal(output.output[0], baseline.output[0])
    assert clear_cache.call_count == 2
    for call in clear_cache.call_args_list:
        for layer in call.args[0].layers:
            assert not hasattr(layer, "flash_k_cache")
            assert not hasattr(layer, "flash_v_cache")
