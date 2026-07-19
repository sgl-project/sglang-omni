# SPDX-License-Identifier: Apache-2.0
"""E2E speed test: production image generation via the thinker-fused pipeline.

Ported from the OLD PR #336 ``test_production_image_gen_e2e.py`` to the NEW
``feat/ming-image-generation-rewrite`` API.

WHY THE PORT IS A REWRITE, NOT A LINE-FOR-LINE EDIT
---------------------------------------------------
The OLD script drove the thinker stage in-process with a direct async
``executor.add_request(payload)`` / ``executor.get_result()`` contract, then
ran the SemanticConditioner projection and ZImage pipeline by hand.  In the
rewrite the thinker factory
(``create_sglang_thinker_executor_from_config``) returns an
``OmniScheduler`` that runs its own blocking event loop inside a dedicated
stage *process*; it has no ``add_request``/``get_result`` and cannot be poked
phase-by-phase from the test's event loop.  The supported way to run a real
thinker prefill is the declarative multi-process pipeline
(``MultiProcessPipelineRunner`` -> ``Coordinator.submit``), which is exactly
what the production server uses.

So the NEW test measures the REAL production path:

  Phase A  Preprocessor create + run        (direct, standalone — unchanged
                                              component, ``enable_image_gen``)
  Phase B  Pipeline startup                  (thinker TP4 load + image_gen
                                              stage load = SemanticConditioner
                                              + ZImage)
  Phase C  Per-request E2E via submit        (cold = image0, warm = image1)

Per-#336 sub-phases (thinker prefill, conditioner projection, DiT generate)
are emitted by the real executors as ``[E2E_TIMING] ...`` log lines when the
test instrumentation patch is applied (see the run wrapper / report).  They
are harvested from the captured stdout, not re-implemented here, because they
run inside the thinker / image_gen stage processes.

Run on the H100 box (all 5 GPUs free first)::

    CUDA_VISIBLE_DEVICES=0,1,2,3,4 PYTHONPATH=. python -u \
        tests/test_model/test_production_image_gen_e2e.py \
        --model-path inclusionAI/Ming-flash-omni-2.0 \
        --dit-model-path /root/.cache/huggingface/hub/models--Tongyi-MAI--Z-Image-Turbo/snapshots/<snap> \
        --tp-size 4 --thinker-gpu 0 --diffusion-gpu cuda:4
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import io
import logging
import os
import time
from typing import Any

import pytest
from PIL import Image, UnidentifiedImageError

logging.basicConfig(
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("e2e_speed")

MODEL_PATH = os.environ.get("MING_MODEL_PATH", "inclusionAI/Ming-flash-omni-2.0")
DIT_MODEL_PATH = os.environ.get("DIT_MODEL_PATH")
TP_SIZE = int(os.environ.get("TP_SIZE", "4"))
THINKER_GPU = int(os.environ.get("THINKER_GPU", "0"))
DIFFUSION_GPU = os.environ.get("DIFFUSION_GPU", "cuda:4")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "/tmp/production_image_gen_e2e")

PROMPTS = [
    "A cat sitting on a windowsill watching the sunset",
    "一幅水墨画，画中有竹子和远山",
]

IMAGE_GEN_PARAMS = {
    "width": 1024,
    "height": 1024,
    "num_inference_steps": 28,
    "guidance_scale": 2.0,
    "seed": 42,
    "semantic_source": "thinker",
    "enable_text_rendering": False,
}


def _diffusion_gpu_index(value: str) -> int:
    if value.startswith("cuda:"):
        return int(value.split(":", 1)[1])
    return int(value)


def _build_request(prompt: str) -> Any:
    """Build the OmniRequest exactly as the OpenAI image route would.

    Mirrors sglang_omni.client.client._extract_inputs: image-gen requests carry
    ``output_modalities`` and ``image_generation`` in ``metadata`` and a
    ``{messages, image_generation}`` inputs dict.
    """
    from sglang_omni.proto import OmniRequest

    return OmniRequest(
        inputs={
            "messages": [{"role": "user", "content": prompt}],
            "image_generation": IMAGE_GEN_PARAMS,
        },
        params={"max_new_tokens": 64, "temperature": 0.0},
        metadata={
            "output_modalities": ["image"],
            "image_generation": IMAGE_GEN_PARAMS,
        },
    )


async def _run_preprocessor_phase() -> None:
    """Phase A: create the preprocessor (enable_image_gen=True) and run it.

    API CHANGE vs #336: MingPreprocessor(model_path, *, enable_image_gen=True);
    the ``conditioner=`` param was removed (query tokens now come from the
    internal diffusion/query_info module).  Output state is read via
    MingOmniPipelineState.from_dict (was PipelineState.from_dict); image-gen
    fields live under mm_inputs["image_gen"].
    """
    from sglang_omni.models.ming_omni.components.preprocessor import MingPreprocessor
    from sglang_omni.models.ming_omni.io import MingOmniPipelineState
    from sglang_omni.proto import StagePayload

    logger.info(
        "=== Phase A: MingPreprocessor create + run (enable_image_gen=True) ==="
    )
    t0 = time.perf_counter()
    preprocessor = MingPreprocessor(model_path=MODEL_PATH, enable_image_gen=True)
    create_s = time.perf_counter() - t0
    logger.info("[E2E_TIMING] preprocessor_create %.3fs", create_s)

    for i, prompt in enumerate(PROMPTS):
        payload = StagePayload(
            request_id=f"prep-{i}",
            request=_build_request(prompt),
            data={"raw_inputs": {"messages": [{"role": "user", "content": prompt}]}},
        )
        t0 = time.perf_counter()
        result = await preprocessor(payload)
        prep_s = time.perf_counter() - t0

        state = MingOmniPipelineState.from_dict(result.data)
        image_gen = state.mm_inputs.get("image_gen", {})
        gen_mask = image_gen.get("gen_mask")
        prefill_only = image_gen.get("prefill_only")
        query_tokens = image_gen.get("query_tokens")
        input_ids = state.prompt["input_ids"]
        assert gen_mask is not None, "gen_mask not set by preprocessor"
        assert query_tokens is not None, "query_tokens not set by preprocessor"
        assert prefill_only is True, "prefill_only not set"
        logger.info(
            "[E2E_TIMING] preprocess[%d] %.3fs  input_ids=%s gen_mask_sum=%d "
            "prefill_only=%s",
            i,
            prep_s,
            list(input_ids.shape),
            int(sum(gen_mask)),
            prefill_only,
        )


async def _run_pipeline_phase() -> None:
    """Phase B/C: build the real production image pipeline and submit prompts.

    API CHANGE vs #336: there is no in-process add_request/get_result thinker
    driver anymore.  Build MingOmniImagePipelineConfig (thinker TP4 +
    image_gen DiT), bring it up with MultiProcessPipelineRunner, and drive it
    with Coordinator.submit — the supported production path.  Per-component
    sub-timings (thinker prefill, projection, DiT generate) are emitted by the
    instrumented executors as [E2E_TIMING] log lines.
    """
    from sglang_omni.models.ming_omni.config import MingOmniImagePipelineConfig
    from sglang_omni.pipeline.mp_runner import MultiProcessPipelineRunner

    diffusion_idx = _diffusion_gpu_index(DIFFUSION_GPU)

    logger.info(
        "=== Phase B: building image pipeline (thinker TP%d gpu%d-%d, "
        "image_gen gpu%d) ===",
        TP_SIZE,
        THINKER_GPU,
        THINKER_GPU + TP_SIZE - 1,
        diffusion_idx,
    )
    config = MingOmniImagePipelineConfig(
        model_path=MODEL_PATH,
        dit_type="zimage",
        dit_model_path=DIT_MODEL_PATH,
    )
    _set_thinker_tp(config, start_gpu=THINKER_GPU, tp_size=TP_SIZE)
    _set_stage_gpu(config, "image_gen", diffusion_idx)
    if TP_SIZE > 1:
        _merge_thinker_server_overrides(config, {"disable_custom_all_reduce": True})

    runner = MultiProcessPipelineRunner(config)
    t0 = time.perf_counter()
    startup_timeout = float(os.environ.get("SGLANG_OMNI_STARTUP_TIMEOUT", "900"))
    await runner.start(timeout=startup_timeout)
    startup_s = time.perf_counter() - t0
    logger.info(
        "[E2E_TIMING] pipeline_startup %.1fs (thinker TP%d load + image_gen "
        "stage load: SemanticConditioner + ZImage)",
        startup_s,
        TP_SIZE,
    )

    coordinator = runner.coordinator
    try:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        logger.info("=== Phase C: submitting %d image-gen requests ===", len(PROMPTS))
        for i, prompt in enumerate(PROMPTS):
            req = _build_request(prompt)
            t0 = time.perf_counter()
            result = await coordinator.submit(f"img-{i}", req)
            e2e_s = time.perf_counter() - t0

            # The image pipeline has TWO terminal stages (decode + image_gen),
            # so submit() returns a dict keyed by stage name with merged
            # partials.  The image payload lives under result["image_gen"].
            image_gen = _result_image_gen(result)
            out_path = _decode_and_write_first_image(
                request_index=i,
                image_gen=image_gen,
                output_dir=OUTPUT_DIR,
                expected_width=int(IMAGE_GEN_PARAMS["width"]),
                expected_height=int(IMAGE_GEN_PARAMS["height"]),
            )
            tag = "cold" if i == 0 else "warm"
            logger.info(
                "[E2E_TIMING] request[%d] %s e2e %.2fs  ok=True reason=%s -> %s",
                i,
                tag,
                e2e_s,
                (
                    image_gen.get("finish_reason")
                    if isinstance(image_gen, dict)
                    else None
                ),
                out_path,
            )
    finally:
        logger.info("Stopping pipeline …")
        await runner.stop()
        logger.info("Pipeline stopped.")


def test_decode_and_write_first_image_validates_png_dimensions(tmp_path) -> None:
    image = Image.new("RGB", (1024, 1024), color=(12, 34, 56))
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    payload = {
        "images": [{"b64_json": base64.b64encode(buf.getvalue()).decode("ascii")}]
    }

    out_path = _decode_and_write_first_image(
        request_index=0,
        image_gen=payload,
        output_dir=str(tmp_path),
        expected_width=1024,
        expected_height=1024,
    )

    assert out_path.endswith("prod_0.png")
    with Image.open(out_path) as saved:
        assert saved.size == (1024, 1024)
        assert saved.format == "PNG"


def test_decode_and_write_first_image_rejects_empty_payload(tmp_path) -> None:
    with pytest.raises(AssertionError, match=r"request\[1\] produced no images"):
        _decode_and_write_first_image(
            request_index=1,
            image_gen={"images": []},
            output_dir=str(tmp_path),
            expected_width=1024,
            expected_height=1024,
        )


def test_decode_and_write_first_image_rejects_truncated_png(tmp_path) -> None:
    image = Image.new("RGB", (1024, 1024), color=(12, 34, 56))
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    truncated = buf.getvalue()[:-12]
    payload = {"images": [{"b64_json": base64.b64encode(truncated).decode("ascii")}]}

    with pytest.raises(AssertionError, match=r"request\[2\].*(invalid|corrupt) PNG"):
        _decode_and_write_first_image(
            request_index=2,
            image_gen=payload,
            output_dir=str(tmp_path),
            expected_width=1024,
            expected_height=1024,
        )


def _decode_and_write_first_image(
    *,
    request_index: int,
    image_gen: dict[str, Any],
    output_dir: str,
    expected_width: int,
    expected_height: int,
) -> str:
    images = image_gen.get("images") if isinstance(image_gen, dict) else None
    assert images, f"request[{request_index}] produced no images"

    first = images[0]
    assert isinstance(
        first, dict
    ), f"request[{request_index}] image payload is not a dict"
    b64 = first.get("b64_json")
    assert (
        isinstance(b64, str) and b64
    ), f"request[{request_index}] image payload missing non-empty b64_json"

    raw = base64.b64decode(b64)
    assert raw.startswith(
        b"\x89PNG\r\n\x1a\n"
    ), f"request[{request_index}] decoded image is not a PNG"

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"prod_{request_index}.png")
    with open(out_path, "wb") as f:
        f.write(raw)

    try:
        with Image.open(out_path) as image:
            assert (
                image.format == "PNG"
            ), f"request[{request_index}] saved image is not PNG"
            assert image.size == (expected_width, expected_height), (
                f"request[{request_index}] image size {image.size} != "
                f"{expected_width}x{expected_height}"
            )
            image.verify()

        with Image.open(out_path) as image:
            image.load()
            assert (
                image.format == "PNG"
            ), f"request[{request_index}] decoded image is not PNG"
            assert image.size == (expected_width, expected_height), (
                f"request[{request_index}] image size {image.size} != "
                f"{expected_width}x{expected_height}"
            )
    except (UnidentifiedImageError, OSError, SyntaxError) as exc:
        raise AssertionError(
            f"request[{request_index}] invalid or corrupt PNG: {exc}"
        ) from exc

    return out_path


def _result_image_gen(result: Any) -> dict[str, Any]:
    """Pull the image_gen terminal payload out of a multi-terminal result.

    submit() on the image pipeline returns either the image_gen dict directly
    (single active terminal) or a stage-keyed dict {"decode": ..., "image_gen":
    {"images": [...]}} when multiple terminals complete.
    """
    if not isinstance(result, dict):
        return {}
    if "images" in result or result.get("modality") == "image":
        return result
    inner = result.get("image_gen")
    return inner if isinstance(inner, dict) else {}


def _set_thinker_tp(config: Any, *, start_gpu: int, tp_size: int) -> None:
    for stage in config.stages:
        if stage.name == "thinker":
            stage.tp_size = int(tp_size)
            stage.parallelism = stage.parallelism.model_copy(
                update={"tp": int(tp_size)}
            )
            if tp_size == 1:
                stage.gpu = int(start_gpu)
            else:
                stage.gpu = list(range(int(start_gpu), int(start_gpu) + int(tp_size)))
            return
    raise ValueError("Stage 'thinker' not found in config")


def _set_stage_gpu(config: Any, stage_name: str, gpu_id: int) -> None:
    for stage in config.stages:
        if stage.name == stage_name:
            stage.gpu = int(gpu_id)
            return
    raise ValueError(f"Stage {stage_name!r} not found in config")


def _merge_thinker_server_overrides(config: Any, updates: dict[str, Any]) -> None:
    for stage in config.stages:
        if stage.name == "thinker":
            factory_args = dict(stage.factory_args or {})
            overrides = dict(factory_args.get("server_args_overrides") or {})
            overrides.update(updates)
            factory_args["server_args_overrides"] = overrides
            stage.factory_args = factory_args
            return
    raise ValueError("Stage 'thinker' not found in config")


async def _main_async(skip_pipeline: bool) -> None:
    await _run_preprocessor_phase()
    if skip_pipeline:
        logger.info("--skip-pipeline set: stopping after preprocessor smoke phase")
        return
    await _run_pipeline_phase()


@pytest.mark.benchmark
def test_production_image_gen_e2e_benchmark() -> None:
    if os.environ.get("RUN_MING_IMAGE_E2E") != "1":
        pytest.skip("set RUN_MING_IMAGE_E2E=1 to run the Ming image full-model E2E")
    if not DIT_MODEL_PATH:
        pytest.skip("set DIT_MODEL_PATH to the local Z-Image-Turbo snapshot")

    asyncio.run(_main_async(skip_pipeline=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--dit-model-path", type=str, default=None)
    parser.add_argument("--tp-size", type=int, default=None)
    parser.add_argument("--thinker-gpu", type=int, default=None)
    parser.add_argument("--diffusion-gpu", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument(
        "--skip-pipeline",
        action="store_true",
        help="Run only Phase A (preprocessor) for a fast import/preprocess smoke.",
    )
    args = parser.parse_args()

    global MODEL_PATH, DIT_MODEL_PATH, TP_SIZE, THINKER_GPU, DIFFUSION_GPU, OUTPUT_DIR
    if args.model_path:
        MODEL_PATH = args.model_path
    if args.dit_model_path:
        DIT_MODEL_PATH = args.dit_model_path
    if args.tp_size:
        TP_SIZE = args.tp_size
    if args.thinker_gpu is not None:
        THINKER_GPU = args.thinker_gpu
    if args.diffusion_gpu:
        DIFFUSION_GPU = args.diffusion_gpu
    if args.output_dir:
        OUTPUT_DIR = args.output_dir

    if not args.skip_pipeline and not DIT_MODEL_PATH:
        raise SystemExit("--dit-model-path is required (or set DIT_MODEL_PATH)")

    asyncio.run(_main_async(skip_pipeline=args.skip_pipeline))


if __name__ == "__main__":
    main()
