"""Exercise SenseNova Cache-DiT switching, overrides and mount lifecycle on CPU."""

from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Protocol
from unittest.mock import Mock

import pytest

pytest.importorskip("sglang")
pytest.importorskip("cache_dit")

import torch
from cache_dit.caching.block_adapters import BlockAdapterRegister
from PIL import Image
from pydantic import JsonValue
from sglang.multimodal_gen.runtime.cache.cache_dit_integration import (
    _CUSTOM_BLOCK_ADAPTER_SPECS,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages import base as stage_base
from transformers.cache_utils import Cache
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs

from sglang_omni.models.sensenova_u1.native import cache_dit as sensenova_cache_dit
from sglang_omni.models.sensenova_u1.native import pipeline as sensenova_pipeline
from sglang_omni.models.sensenova_u1.native.generation import SenseNovaU1GenerationStage
from sglang_omni.models.sensenova_u1.native.pipeline import SenseNovaU1Pipeline
from sglang_omni.models.sensenova_u1.native.pipeline_config import (
    SenseNovaU1PipelineConfig,
)
from sglang_omni.models.sensenova_u1.native.sampling import SenseNovaU1SamplingParams
from tests.unit_test.fixtures.sensenova import (
    tiny_sensenova_config,
    tiny_sensenova_model,
    tiny_sensenova_tokenizer,
)

DECODER_LAYERS = 2
STEPS = 4
# Every step runs every layer, so a mounted run must equal an unmounted one.
COMPUTE_EVERY_STEP = {
    "Fn_compute_blocks": 1,
    "Bn_compute_blocks": 0,
    "max_warmup_steps": 0,
    "residual_diff_threshold": 0,
}
# Any step after the first may reuse the cached second layer.
CACHE_EVERY_STEP = {
    "Fn_compute_blocks": 1,
    "Bn_compute_blocks": 0,
    "max_warmup_steps": 0,
    "residual_diff_threshold": 1,
    "max_continuous_cached_steps": -1,
}
CACHE_DIT_VARIABLES = (
    "SGLANG_CACHE_DIT_ENABLED",
    "SGLANG_CACHE_DIT_FN",
    "SGLANG_CACHE_DIT_BN",
    "SGLANG_CACHE_DIT_WARMUP",
    "SGLANG_CACHE_DIT_RDT",
    "SGLANG_CACHE_DIT_MC",
    "SGLANG_CACHE_DIT_TAYLORSEER",
    "SGLANG_CACHE_DIT_TS_ORDER",
    "SGLANG_CACHE_DIT_DMD",
    "SGLANG_CACHE_DIT_SCM_PRESET",
    "SGLANG_CACHE_DIT_SCM_POLICY",
    "SGLANG_CACHE_DIT_SCM_COMPUTE_BINS",
    "SGLANG_CACHE_DIT_SCM_CACHE_BINS",
)


class StageFactory(Protocol):
    def __call__(self) -> SenseNovaU1GenerationStage: ...


class Unpatchify(Protocol):
    def __call__(
        self, latents: torch.Tensor, patch_size: int, height: int, width: int
    ) -> torch.Tensor: ...


@pytest.fixture(autouse=True)
def cpu_generation(monkeypatch: pytest.MonkeyPatch) -> None:
    for variable in CACHE_DIT_VARIABLES:
        monkeypatch.delenv(variable, raising=False)
    monkeypatch.setattr(
        stage_base,
        "get_global_server_args",
        lambda: SimpleNamespace(comfyui_mode=False),
    )
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: (4 * 1024**3, 0))
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda device: 0)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda device: 0)


@pytest.fixture
def checkpoint_path(tmp_path: Path) -> str:
    tiny_sensenova_config(DECODER_LAYERS).save_pretrained(tmp_path)
    return str(tmp_path)


def release(stage: SenseNovaU1GenerationStage) -> None:
    if stage.cache_dit.active_key is not None:
        stage.cache_dit.unmount(stage.model.language_model.model)
    else:
        pass


@pytest.fixture
def new_stage() -> Iterator[StageFactory]:
    """Build tiny stages and unmount them at teardown.

    cache-dit 1.3.0 allows one transformer-only mount per process at a time.
    """
    stages: list[SenseNovaU1GenerationStage] = []

    def build() -> SenseNovaU1GenerationStage:
        stage = SenseNovaU1GenerationStage(
            tiny_sensenova_model(DECODER_LAYERS), tiny_sensenova_tokenizer()
        )
        stages.append(stage)
        return stage

    yield build
    for stage in stages:
        release(stage)


def generation_request(
    *,
    enable_cache_dit: bool | None,
    cache_dit_params: dict[str, JsonValue] | None = None,
    num_inference_steps: int = STEPS,
    guidance_scale: float = 4.0,
    img_cfg_scale: float = 1.0,
    edit: bool = False,
    prompts: tuple[str, ...] = ("a red apple",),
) -> Req:
    sampling = SenseNovaU1SamplingParams(
        prompt=prompts[0],
        image_path="reference.png" if edit else None,
        width=64,
        height=64,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        img_cfg_scale=img_cfg_scale,
        seed=7,
        generator_device="cpu",
        enable_cache_dit=enable_cache_dit,
        cache_dit_params=cache_dit_params,
    )
    request = Req(sampling_params=sampling)
    sampling.apply_request_extra(request)
    if edit:
        request.condition_image = [Image.new("RGB", (32, 32), "red")]
    elif len(prompts) > 1:
        request.prompt = list(prompts)
        request.extra["dynamic_batch_seeds"] = list(range(7, 7 + len(prompts)))
    else:
        pass
    return request


def generate(stage: SenseNovaU1GenerationStage, request: Req) -> torch.Tensor:
    return torch.stack(stage.forward(request, None).output)


def view_calls(monkeypatch: pytest.MonkeyPatch) -> list[bool]:
    """Record each call of the denoise view, wrapped below the hooks cache-dit installs."""
    calls: list[bool] = []
    view_forward = sensenova_cache_dit.SenseNovaGenerationBlocks.forward

    def counted_forward(
        view: sensenova_cache_dit.SenseNovaGenerationBlocks,
        hidden_states: torch.Tensor,
        *,
        image_gen_indicators: torch.Tensor,
        exist_non_image_gen_tokens: bool,
        exist_image_gen_tokens: bool,
        indexes: torch.LongTensor | None,
        attention_mask: torch.Tensor | None,
        position_ids: torch.LongTensor | None,
        past_key_values: Cache | None,
        use_cache: bool | None,
        cache_position: torch.LongTensor | None,
        update_cache: bool,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        calls.append(update_cache)
        return view_forward(
            view,
            hidden_states,
            image_gen_indicators=image_gen_indicators,
            exist_non_image_gen_tokens=exist_non_image_gen_tokens,
            exist_image_gen_tokens=exist_image_gen_tokens,
            indexes=indexes,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            update_cache=update_cache,
            **kwargs,
        )

    monkeypatch.setattr(
        sensenova_cache_dit.SenseNovaGenerationBlocks, "forward", counted_forward
    )
    return calls


@pytest.mark.parametrize(
    ("request_options", "branch_count"),
    [
        ({"guidance_scale": 4.0}, 2),
        ({"guidance_scale": 1.0}, 1),
        ({"guidance_scale": 4.0, "prompts": ("a red apple", "a blue sky")}, 2),
        ({"guidance_scale": 4.0, "edit": True}, 2),
        ({"guidance_scale": 1.0, "edit": True}, 1),
    ],
)
def test_mounted_cache_runs_only_denoising_steps_and_keeps_output(
    monkeypatch: pytest.MonkeyPatch,
    new_stage: StageFactory,
    request_options: dict[str, float | bool | tuple[str, ...]],
    branch_count: int,
) -> None:
    """Prefix passes bypass the cache, and a cache that skips nothing changes no pixel."""
    calls = view_calls(monkeypatch)
    uncached = generate(
        new_stage(), generation_request(enable_cache_dit=False, **request_options)
    )
    assert calls == []
    stage = new_stage()
    cached = generate(
        stage,
        generation_request(
            enable_cache_dit=True,
            cache_dit_params=COMPUTE_EVERY_STEP,
            **request_options,
        ),
    )
    decoder = stage.model.language_model.model
    assert decoder.generation_blocks is stage.cache_dit.view
    assert "forward" not in decoder.__dict__
    assert calls == [False] * (STEPS * branch_count)
    assert torch.equal(cached, uncached)


def test_mount_follows_each_request(
    monkeypatch: pytest.MonkeyPatch,
    new_stage: StageFactory,
) -> None:
    """Same settings refresh; new settings or guidance branches remount; a disabled request unmounts."""
    mounts = Mock(wraps=sensenova_cache_dit.enable_cache_on_transformer)
    refreshes = Mock(wraps=sensenova_cache_dit.refresh_context_on_transformer)
    unmounts = Mock(wraps=sensenova_cache_dit.disable_cache_on_transformer)
    monkeypatch.setattr(sensenova_cache_dit, "enable_cache_on_transformer", mounts)
    monkeypatch.setattr(
        sensenova_cache_dit, "refresh_context_on_transformer", refreshes
    )
    monkeypatch.setattr(sensenova_cache_dit, "disable_cache_on_transformer", unmounts)
    stage = new_stage()
    decoder = stage.model.language_model.model
    sequence = [
        (
            generation_request(
                enable_cache_dit=True, cache_dit_params=COMPUTE_EVERY_STEP
            ),
            1,
            0,
            0,
        ),
        (
            generation_request(
                enable_cache_dit=True, cache_dit_params=COMPUTE_EVERY_STEP
            ),
            1,
            1,
            0,
        ),
        (
            generation_request(
                enable_cache_dit=True, cache_dit_params=CACHE_EVERY_STEP
            ),
            2,
            1,
            1,
        ),
        (
            generation_request(
                enable_cache_dit=True,
                cache_dit_params=CACHE_EVERY_STEP,
                guidance_scale=1.0,
            ),
            3,
            1,
            2,
        ),
        (generation_request(enable_cache_dit=False), 3, 1, 3),
    ]
    for request, mount_count, refresh_count, unmount_count in sequence:
        generate(stage, request)
        assert (
            mounts.call_count,
            refreshes.call_count,
            unmounts.call_count,
        ) == (mount_count, refresh_count, unmount_count)
    assert [call.kwargs["has_separate_cfg"] for call in mounts.call_args_list] == [
        True,
        True,
        False,
    ]
    assert decoder.generation_blocks is None
    warmup = generation_request(enable_cache_dit=True)
    warmup.set_as_warmup(warmup_steps=1)
    generate(stage, warmup)
    assert mounts.call_count == 3
    assert decoder.generation_blocks is None
    spec = _CUSTOM_BLOCK_ADAPTER_SPECS[  # noqa: leading-underscore  # upstream name
        "SenseNovaGenerationBlocks"
    ]
    assert spec.blocks_attr == "blocks"
    assert not BlockAdapterRegister.is_supported(stage.cache_dit.view)


def raise_after_unpatchify_calls(
    stage: SenseNovaU1GenerationStage, completed_calls: int
) -> Unpatchify:
    unpatchify = stage.model.unpatchify
    calls = [0]

    def failing_unpatchify(
        latents: torch.Tensor, patch_size: int, height: int, width: int
    ) -> torch.Tensor:
        if calls[0] == completed_calls:
            raise RuntimeError("generation failed mid-denoise")
        else:
            calls[0] += 1
            return unpatchify(latents, patch_size, height, width)

    return failing_unpatchify


@pytest.mark.parametrize(
    "earlier_requests",
    [
        pytest.param(
            [{"cache_dit_params": COMPUTE_EVERY_STEP}], id="different-settings"
        ),
        pytest.param([{"guidance_scale": 1.0}], id="one-branch-before"),
        pytest.param([{"edit": True}], id="edit-before"),
        pytest.param([{"fail_after_steps": 2}], id="failed-before"),
    ],
)
@pytest.mark.parametrize(
    "later_options",
    [
        pytest.param({}, id="later-t2i"),
        pytest.param({"edit": True}, id="later-edit"),
        pytest.param(
            {"prompts": ("a red apple", "a blue sky")}, id="later-two-prompts"
        ),
    ],
)
def test_cached_request_matches_a_fresh_stage(
    monkeypatch: pytest.MonkeyPatch,
    new_stage: StageFactory,
    earlier_requests: list[dict[str, JsonValue]],
    later_options: dict[str, bool | tuple[str, ...]],
) -> None:
    """Cache state from earlier, changed, or failed requests must not reach the next request.

    The cached run must differ from an uncached one, so a mount that silently stops caching fails here.
    """
    reference_stage = new_stage()
    fresh = generate(
        reference_stage,
        generation_request(
            enable_cache_dit=True, cache_dit_params=CACHE_EVERY_STEP, **later_options
        ),
    )
    release(reference_stage)
    uncached = generate(
        new_stage(), generation_request(enable_cache_dit=False, **later_options)
    )
    assert not torch.equal(fresh, uncached)
    stage = new_stage()
    for options in earlier_requests:
        options = {"cache_dit_params": CACHE_EVERY_STEP} | options
        fail_after_steps = options.pop("fail_after_steps", None)
        request = generation_request(enable_cache_dit=True, **options)
        if fail_after_steps is None:
            generate(stage, request)
        else:
            with monkeypatch.context() as patch:
                patch.setattr(
                    stage.model,
                    "unpatchify",
                    raise_after_unpatchify_calls(stage, fail_after_steps),
                )
                with pytest.raises(RuntimeError, match="mid-denoise"):
                    generate(stage, request)
    later = generate(
        stage,
        generation_request(
            enable_cache_dit=True, cache_dit_params=CACHE_EVERY_STEP, **later_options
        ),
    )
    assert torch.equal(later, fresh)


@pytest.mark.parametrize(
    ("server_switch", "request_switch", "expected"),
    [
        (None, None, False),
        ("true", None, True),
        ("true", False, False),
        ("false", True, True),
    ],
)
def test_request_switch_overrides_server_switch(
    monkeypatch: pytest.MonkeyPatch,
    checkpoint_path: str,
    server_switch: str | None,
    request_switch: bool | None,
    expected: bool,
) -> None:
    if server_switch is not None:
        monkeypatch.setenv("SGLANG_CACHE_DIT_ENABLED", server_switch)
    else:
        pass
    sampling = generation_request(
        enable_cache_dit=request_switch, cache_dit_params={"max_warmup_steps": 2}
    ).sampling_params
    sampling.resolve_cache_dit(checkpoint_path, None)
    assert sampling.enable_cache_dit is expected
    assert sampling.cache_dit_params == ({"max_warmup_steps": 2} if expected else None)


def test_warmup_beyond_the_last_step_is_clamped(
    monkeypatch: pytest.MonkeyPatch,
    new_stage: StageFactory,
) -> None:
    """cache-dit lists every warmup step on each step, so a huge warmup must not reach it."""
    mounts = Mock(wraps=sensenova_cache_dit.enable_cache_on_transformer)
    monkeypatch.setattr(sensenova_cache_dit, "enable_cache_on_transformer", mounts)
    images = []
    for warmup in (10**6, STEPS):
        stage = new_stage()
        images.append(
            generate(
                stage,
                generation_request(
                    enable_cache_dit=True,
                    cache_dit_params=CACHE_EVERY_STEP | {"max_warmup_steps": warmup},
                ),
            )
        )
        release(stage)
    assert [call.args[1].max_warmup_steps for call in mounts.call_args_list] == [
        STEPS,
        STEPS,
    ]
    assert torch.equal(images[0], images[1])


def test_mount_tracks_effective_warmup_across_requests(
    monkeypatch: pytest.MonkeyPatch,
    new_stage: StageFactory,
) -> None:
    """A short request must not shorten warmup for the next request."""
    mounts = Mock(wraps=sensenova_cache_dit.enable_cache_on_transformer)
    refreshes = Mock(wraps=sensenova_cache_dit.refresh_context_on_transformer)
    monkeypatch.setattr(sensenova_cache_dit, "enable_cache_on_transformer", mounts)
    monkeypatch.setattr(
        sensenova_cache_dit, "refresh_context_on_transformer", refreshes
    )
    stage = new_stage()
    for num_inference_steps in (2, 6, 4, 2):
        generate(
            stage,
            generation_request(
                enable_cache_dit=True,
                cache_dit_params={"max_warmup_steps": 4},
                num_inference_steps=num_inference_steps,
            ),
        )
    mounted_configs = [call.args[1] for call in mounts.call_args_list]
    assert [config.max_warmup_steps for config in mounted_configs] == [2, 4, 2]
    assert [call.args[1] for call in refreshes.call_args_list] == [4]


@pytest.mark.parametrize("request_switch", ["false", "true", 0, 1])
@pytest.mark.parametrize("server_switch", [None, "true"])
def test_non_boolean_request_switch_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
    checkpoint_path: str,
    request_switch: str | int,
    server_switch: str | None,
) -> None:
    """A JSON request can pass the switch as a string, which must not turn the cache on."""
    if server_switch is not None:
        monkeypatch.setenv("SGLANG_CACHE_DIT_ENABLED", server_switch)
    else:
        pass
    sampling = SenseNovaU1SamplingParams(
        prompt="a red apple",
        width=64,
        height=64,
        num_inference_steps=STEPS,
        enable_cache_dit=request_switch,
    )
    with pytest.raises(
        ValueError, match="enable_cache_dit must be true, false or omitted"
    ):
        sampling.resolve_cache_dit(checkpoint_path, None)


def test_request_settings_override_server_settings_per_key(
    monkeypatch: pytest.MonkeyPatch,
    new_stage: StageFactory,
) -> None:
    monkeypatch.setenv("SGLANG_CACHE_DIT_RDT", "0.3")
    monkeypatch.setenv("SGLANG_CACHE_DIT_WARMUP", "2")
    mounts = Mock(wraps=sensenova_cache_dit.enable_cache_on_transformer)
    monkeypatch.setattr(sensenova_cache_dit, "enable_cache_on_transformer", mounts)
    generate(
        new_stage(),
        generation_request(
            enable_cache_dit=True,
            cache_dit_params={
                "Fn_compute_blocks": 2,
                "max_continuous_cached_steps": -1,
            },
        ),
    )
    config = mounts.call_args.args[1]
    assert (
        config.Fn_compute_blocks,
        config.Bn_compute_blocks,
        config.max_warmup_steps,
        config.residual_diff_threshold,
        config.max_continuous_cached_steps,
        config.enable_taylorseer,
        config.steps_computation_mask,
        config.num_inference_steps,
    ) == (2, 0, 2, 0.3, -1, False, None, STEPS)


@pytest.mark.parametrize(
    ("cache_dit_params", "message"),
    [
        (
            {"unknown_knob": 1},
            r"Unknown cache_dit_params keys: \['unknown_knob'\]\. Valid keys: "
            r"\['Bn_compute_blocks', 'Fn_compute_blocks', 'max_continuous_cached_steps', "
            r"'max_warmup_steps', 'residual_diff_threshold'\]\.$",
        ),
        ({"enable_dmd": True}, "does not support"),
        ({"secondary": {}}, "does not support"),
        ({"enable_taylorseer": True}, "DBCache settings only"),
        ({"taylorseer_order": 2}, "DBCache settings only"),
        ({"scm_preset": "fast"}, "DBCache settings only"),
        ({"scm_policy": "static"}, "DBCache settings only"),
        (
            {"scm_compute_bins": [2], "scm_cache_bins": [2]},
            "DBCache settings only",
        ),
        ({"Fn_compute_blocks": 0}, "Fn_compute_blocks"),
        ({"Bn_compute_blocks": True}, "Bn_compute_blocks"),
        ({"residual_diff_threshold": 1.5}, "residual_diff_threshold"),
        ({"Fn_compute_blocks": 2, "Bn_compute_blocks": 1}, "decoder layers"),
    ],
)
@pytest.mark.parametrize("switch_source", ["request", "server"])
def test_unservable_cache_dit_requests_are_rejected(
    monkeypatch: pytest.MonkeyPatch,
    checkpoint_path: str,
    cache_dit_params: dict[str, JsonValue] | None,
    message: str,
    switch_source: str,
) -> None:
    if switch_source == "server":
        monkeypatch.setenv("SGLANG_CACHE_DIT_ENABLED", "true")
    else:
        pass

    def sampling_params(enable_cache_dit: bool | None) -> SenseNovaU1SamplingParams:
        return SenseNovaU1SamplingParams(
            prompt="a red apple",
            width=64,
            height=64,
            num_inference_steps=STEPS,
            enable_cache_dit=enable_cache_dit,
            cache_dit_params=cache_dit_params,
        )

    rejected = sampling_params(True if switch_source == "request" else None)
    with pytest.raises(ValueError, match=message):
        rejected.resolve_cache_dit(checkpoint_path, None)
    accepted = sampling_params(False)
    accepted.resolve_cache_dit(checkpoint_path, None)
    assert (accepted.enable_cache_dit, accepted.cache_dit_params) == (False, None)


THREE_BRANCH_EDIT = {
    "prompt": "edit",
    "image_path": "reference.png",
    "img_cfg_scale": 2.0,
}
PARTIAL_CFG_INTERVAL = {"prompt": "a red apple", "cfg_interval": (0.0, 0.5)}


@pytest.mark.parametrize(
    ("request_options", "message"),
    [
        pytest.param(
            THREE_BRANCH_EDIT,
            "editing with three guidance branches, where img_cfg_scale is neither 1 "
            "nor equal to guidance_scale",
            id="three-branch-edit",
        ),
        pytest.param(
            PARTIAL_CFG_INTERVAL,
            r"guidance on only part of the steps, with cfg_interval other than \(0\.0, 1\.0\)",
            id="partial-cfg-interval",
        ),
    ],
)
def test_unsupported_guidance_fails_only_an_explicit_enable(
    monkeypatch: pytest.MonkeyPatch,
    checkpoint_path: str,
    request_options: dict[str, JsonValue],
    message: str,
) -> None:
    """The server switch runs unsupported guidance uncached; only a request's own enable is refused."""
    monkeypatch.setenv("SGLANG_CACHE_DIT_ENABLED", "true")

    def sampling_params(enable_cache_dit: bool | None) -> SenseNovaU1SamplingParams:
        return SenseNovaU1SamplingParams(
            width=64,
            height=64,
            num_inference_steps=STEPS,
            enable_cache_dit=enable_cache_dit,
            cache_dit_params={"residual_diff_threshold": 0.5},
            **request_options,
        )

    explicit = sampling_params(True)
    with pytest.raises(
        ValueError, match=f"Cache-DiT does not support {message}; omit enable_cache_dit"
    ):
        explicit.resolve_cache_dit(checkpoint_path, None)
    for request_switch in (None, False):
        uncached = sampling_params(request_switch)
        uncached.resolve_cache_dit(checkpoint_path, None)
        assert (uncached.enable_cache_dit, uncached.cache_dit_params) == (False, None)


@pytest.mark.parametrize(
    ("cache_dit_params", "message"),
    [
        ({"unknown_key": 1}, r"Unknown cache_dit_params keys: \['unknown_key'\]"),
        ({"residual_diff_threshold": 1.5}, "residual_diff_threshold"),
        ({"Fn_compute_blocks": 2, "Bn_compute_blocks": 1}, "decoder layers"),
    ],
)
@pytest.mark.parametrize(
    "request_options",
    [THREE_BRANCH_EDIT, PARTIAL_CFG_INTERVAL],
    ids=["three-branch-edit", "partial-cfg-interval"],
)
def test_server_switch_still_rejects_invalid_settings_with_unsupported_guidance(
    monkeypatch: pytest.MonkeyPatch,
    checkpoint_path: str,
    cache_dit_params: dict[str, JsonValue],
    message: str,
    request_options: dict[str, JsonValue],
) -> None:
    """Falling back to an uncached run must not hide settings that would fail a cached one."""
    monkeypatch.setenv("SGLANG_CACHE_DIT_ENABLED", "true")

    def sampling_params(enable_cache_dit: bool | None) -> SenseNovaU1SamplingParams:
        return SenseNovaU1SamplingParams(
            width=64,
            height=64,
            num_inference_steps=STEPS,
            enable_cache_dit=enable_cache_dit,
            cache_dit_params=cache_dit_params,
            **request_options,
        )

    defaulted = sampling_params(None)
    with pytest.raises(ValueError, match=message):
        defaulted.resolve_cache_dit(checkpoint_path, None)
    disabled = sampling_params(False)
    disabled.resolve_cache_dit(checkpoint_path, None)
    assert (disabled.enable_cache_dit, disabled.cache_dit_params) == (False, None)


def test_server_switch_runs_unsupported_guidance_uncached(
    monkeypatch: pytest.MonkeyPatch,
    new_stage: StageFactory,
) -> None:
    """A three-branch edit on a stage whose server switch is on never mounts and matches an uncached run."""
    calls = view_calls(monkeypatch)
    uncached = generate(
        new_stage(),
        generation_request(enable_cache_dit=False, edit=True, img_cfg_scale=2.0),
    )
    monkeypatch.setenv("SGLANG_CACHE_DIT_ENABLED", "true")
    stage = new_stage()
    defaulted = generate(
        stage,
        generation_request(enable_cache_dit=None, edit=True, img_cfg_scale=2.0),
    )
    assert stage.cache_dit.active_key is None
    assert stage.model.language_model.model.generation_blocks is None
    assert calls == []
    assert torch.equal(defaulted, uncached)


def test_requests_with_the_same_resolved_switch_batch_together(
    monkeypatch: pytest.MonkeyPatch, checkpoint_path: str
) -> None:
    monkeypatch.setenv("SGLANG_CACHE_DIT_ENABLED", "true")
    resolved = []
    for request_switch in (None, True):
        sampling = generation_request(enable_cache_dit=request_switch).sampling_params
        sampling.resolve_cache_dit(checkpoint_path, None)
        resolved.append((sampling.enable_cache_dit, sampling.cache_dit_params))
    assert resolved == [(True, None), (True, None)]


@pytest.mark.parametrize(
    ("environment", "message"),
    [
        ({"SGLANG_CACHE_DIT_RDT": "2"}, "residual_diff_threshold"),
        ({"SGLANG_CACHE_DIT_FN": "0"}, "Fn_compute_blocks"),
        ({"SGLANG_CACHE_DIT_DMD": "true"}, "DMD"),
        ({"SGLANG_CACHE_DIT_TAYLORSEER": "true"}, "unset SGLANG_CACHE_DIT_TAYLORSEER"),
        ({"SGLANG_CACHE_DIT_SCM_PRESET": "fast"}, "unset SGLANG_CACHE_DIT_SCM_PRESET"),
        (
            {"SGLANG_CACHE_DIT_SCM_COMPUTE_BINS": "2,2"},
            "unset SGLANG_CACHE_DIT_SCM_COMPUTE_BINS",
        ),
    ],
)
def test_invalid_server_settings_fail_startup(
    monkeypatch: pytest.MonkeyPatch, environment: dict[str, str], message: str
) -> None:
    for variable, value in environment.items():
        monkeypatch.setenv(variable, value)
    server_args = SimpleNamespace(
        num_gpus=1,
        enable_torch_compile=False,
        lora_path=None,
        is_arg_explicitly_set=lambda option: False,
    )
    with pytest.raises(ValueError, match=message):
        SenseNovaU1PipelineConfig().validate_server_args(server_args)


def test_server_compute_blocks_beyond_the_decoder_fail_at_load(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SGLANG_CACHE_DIT_FN", "2")
    monkeypatch.setenv("SGLANG_CACHE_DIT_BN", "1")
    monkeypatch.setattr(
        torch._C,  # noqa: leading-underscore  # upstream spelling, or the public name is already taken
        "_accelerator_setAllocatorSettings",
        Mock(),
    )
    monkeypatch.setattr(
        sensenova_pipeline,
        "load_model_and_tokenizer",
        lambda model_path, server_args: {
            "model": tiny_sensenova_model(DECODER_LAYERS),
            "tokenizer": tiny_sensenova_tokenizer(),
        },
    )
    with pytest.raises(ValueError, match="decoder layers"):
        SenseNovaU1Pipeline.load_modules(
            SimpleNamespace(model_path="checkpoint"), Mock()
        )
