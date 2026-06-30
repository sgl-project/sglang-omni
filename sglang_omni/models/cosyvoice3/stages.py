# SPDX-License-Identifier: Apache-2.0
"""Stage factories for the CosyVoice3 zero-shot TTS pipeline.

Three factories wire the already-proven CosyVoice components into sglang-omni:

  * ``create_preprocessing_executor`` builds the vendored ``CosyVoiceFrontEnd``
    once and returns a ``SimpleScheduler`` over ``preprocess_cosyvoice3_payload``.
  * ``create_sglang_tts_engine_executor`` boots the Qwen2 backbone from the
    ``CosyVoice-BlankEN`` subdir via ``model_arch_override="CosyVoice3ForCausalLM"``,
    OVERLAYS the real CosyVoice3 weights from ``llm.pt``, registers the
    preprocessing context, and returns an ``OmniScheduler`` (mirrors qwen3_tts).
  * ``create_vocoder_executor`` builds the vendored flow + hift from a *minimal*
    slice of ``cosyvoice3.yaml`` and returns a batch-1 ``SimpleScheduler`` that
    runs ``flow.inference -> hift.inference`` (mirrors fishaudio_s2_pro).
"""

from __future__ import annotations

import functools
import logging
import os
import random
import re
from typing import Any

import numpy as np
import torch

from sglang_omni.models.cosyvoice3.payload_types import CosyVoice3State
from sglang_omni.models.cosyvoice3.request_builders import (
    CV3_CONTEXT_LENGTH,
    cleanup_prepared_cosyvoice3_request,
    make_cosyvoice3_scheduler_adapters,
    preprocess_cosyvoice3_payload,
    set_cosyvoice3_preprocessing_context,
)
from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.generation_batch_policy import (
    build_generation_batch_overrides,
    validate_generation_batch_policy,
)
from sglang_omni.scheduling.simple_scheduler import SimpleScheduler
from sglang_omni.utils.audio_payload import audio_waveform_payload

logger = logging.getLogger(__name__)

CV3_SAMPLE_RATE = 24000


def _resolve_checkpoint(checkpoint: str) -> str:
    if os.path.isdir(checkpoint):
        return checkpoint
    from huggingface_hub import snapshot_download

    return snapshot_download(checkpoint)


def _build_usage(state: CosyVoice3State) -> dict[str, Any] | None:
    if not (state.prompt_tokens or state.completion_tokens or state.engine_time_s):
        return None
    usage = {
        "prompt_tokens": state.prompt_tokens,
        "completion_tokens": state.completion_tokens,
        "total_tokens": state.prompt_tokens + state.completion_tokens,
    }
    if state.engine_time_s:
        usage["engine_time_s"] = round(float(state.engine_time_s), 6)
    return usage


# ---------------------------------------------------------------------------
# Preprocessing — returns SimpleScheduler over the proven frontend
# ---------------------------------------------------------------------------


def create_preprocessing_executor(
    model_path: str,
    *,
    device: str | None = None,
    gpu_id: int | None = None,
) -> SimpleScheduler:
    """Build the vendored CosyVoice frontend once and wrap it in a scheduler.

    ``gpu_id`` is injected from stage placement (mirrors ``moss_tts_local``); the
    frontend's torch tensors and ORT speech-tokenizer session are pinned to that
    device instead of implicitly grabbing cuda:0.
    """
    if gpu_id is not None:
        device = f"cuda:{gpu_id}"
    checkpoint_dir = _resolve_checkpoint(model_path)

    from sglang_omni.models.cosyvoice3.cosyvoice.cli.frontend import CosyVoiceFrontEnd
    from sglang_omni.models.cosyvoice3.cosyvoice.tokenizer.tokenizer import (
        get_qwen_tokenizer,
    )
    from sglang_omni.models.cosyvoice3.matcha.utils.audio import mel_spectrogram

    qwen_dir = os.path.join(checkpoint_dir, "CosyVoice-BlankEN")
    get_tok = functools.partial(
        get_qwen_tokenizer,
        token_path=qwen_dir,
        skip_special_tokens=True,
        version="cosyvoice3",
    )
    feat = functools.partial(
        mel_spectrogram,
        n_fft=1920,
        num_mels=80,
        sampling_rate=CV3_SAMPLE_RATE,
        hop_size=480,
        win_size=1920,
        fmin=0,
        fmax=None,
        center=False,
    )
    frontend = CosyVoiceFrontEnd(
        get_tok,
        feat,
        os.path.join(checkpoint_dir, "campplus.onnx"),
        os.path.join(checkpoint_dir, "speech_tokenizer_v3.onnx"),
        "",
        "all",
        device=device,
    )
    logger.info(
        "CosyVoice3 frontend ready (checkpoint=%s, device=%s)",
        checkpoint_dir,
        frontend.device,
    )

    return SimpleScheduler(
        functools.partial(preprocess_cosyvoice3_payload, frontend=frontend),
        abort_callback=cleanup_prepared_cosyvoice3_request,
    )


# ---------------------------------------------------------------------------
# TTS engine — returns OmniScheduler (Qwen2 backbone + speech head)
# ---------------------------------------------------------------------------


def create_sglang_tts_engine_executor(
    model_path: str,
    *,
    device: str = "cuda:0",
    gpu_id: int | None = None,
    dtype: str = "bfloat16",
    server_args_overrides: dict[str, Any] | None = None,
) -> Any:
    """Boot the CosyVoice3 AR engine and return an OmniScheduler.

    Generation length is governed per-request (``max_len``, clamped to the AR context in
    preprocessing), so there is no engine-level ``max_new_tokens`` argument.
    """

    from sglang_omni.models.cosyvoice3.model_runner import CosyVoice3ModelRunner
    from sglang_omni.scheduling.bootstrap import (
        create_sglang_infrastructure_defer_cuda_graph,
    )
    from sglang_omni.scheduling.omni_scheduler import OmniScheduler
    from sglang_omni.scheduling.sglang_backend import (
        SGLangOutputProcessor,
        build_sglang_server_args,
    )

    checkpoint_dir = _resolve_checkpoint(model_path)
    if gpu_id is not None:
        device = f"cuda:{gpu_id}"
    gpu_id = int(device.split(":")[-1]) if ":" in device else 0

    # The Qwen2 backbone config.json lives in the BlankEN subdir.
    qwen_dir = os.path.join(checkpoint_dir, "CosyVoice-BlankEN")

    overrides = build_generation_batch_overrides(
        max_running_requests=16,
        server_args_overrides=server_args_overrides,
        dtype=dtype,
        disable_cuda_graph=False,
        disable_overlap_schedule=True,
        # The vocoder (flow + hift) colocates on the same GPU and needs headroom for its
        # per-request DiT/HiFi-GAN activations, so the AR engine must not claim the whole
        # device. 0.45 leaves room for vocoder + preprocessing on a single 24 GB card.
        mem_fraction_static=0.45,
        max_prefill_tokens=8192,
        sampling_backend="pytorch",
        trust_remote_code=True,
    )

    server_args = build_sglang_server_args(
        qwen_dir,
        context_length=CV3_CONTEXT_LENGTH,  # single source of truth (request_builders clamps to it)
        **overrides,
    )

    # CosyVoice3's speech LM is not tensor-parallel-safe (global attention split sizes in
    # sglang_model.py and direct speech_embedding weight indexing); fail loudly on TP>1
    # rather than silently producing corrupted audio.
    if int(getattr(server_args, "tp_size", 1) or 1) > 1:
        raise ValueError("CosyVoice3 does not support tensor parallelism (tp_size>1)")

    want_cuda_graph, (
        model_worker,
        tree_cache,
        req_to_token_pool,
        token_to_kv_pool_allocator,
        prefill_mgr,
        decode_mgr,
        model_config,
    ) = create_sglang_infrastructure_defer_cuda_graph(
        server_args,
        gpu_id,
        model_arch_override="CosyVoice3ForCausalLM",
    )

    validate_generation_batch_policy(
        model_name="CosyVoice3",
        server_args=server_args,
    )

    model = model_worker.model_runner.model

    # CosyVoice3's AR speech LM is built from a generic Qwen2 (BlankEN) checkpoint whose
    # config.json reports vocab_size=151936 (text vocab), but its real output vocab is the
    # speech-token vocab (SPEECH_VOCAB_SIZE=6761, the llm_decoder width — the speech LM weights
    # are overlaid from llm.pt below). The sglang sampler sizes its suppress / min-new-tokens
    # penalty tensors from model_config.vocab_size, so it must be realigned to 6761 or those
    # tensors mismatch the logits. This is done here as a local post-build override rather than
    # at model-config construction on purpose: the config is built inside the shared
    # `create_sglang_infrastructure` bootstrap, and correcting it there would mean modifying
    # core infrastructure for one model's quirk. The override runs before CUDA-graph capture
    # and before the first request, so the sampler always sees the corrected vocab. The loop
    # covers both handles in case the returned model_config and the runner's model_config are
    # distinct objects (setting the same value twice is harmless if they are the same object).
    from sglang_omni.models.cosyvoice3.request_builders import SPEECH_VOCAB_SIZE

    for _cfg in (
        model_config,
        getattr(model_worker.model_runner, "model_config", None),
    ):
        if _cfg is not None:
            _cfg.vocab_size = SPEECH_VOCAB_SIZE

    # CRITICAL two-checkpoint step: the engine built the model from BlankEN
    # (Qwen2 weights, entirely skipped by our load_weights -- BlankEN's `model.*` keys
    # don't match its llm.pt-oriented prefix handling); now overlay the real CosyVoice3
    # weights from llm.pt (maps llm.model.model.* / speech_embedding / llm_decoder onto
    # our modules).
    llm_state = torch.load(os.path.join(checkpoint_dir, "llm.pt"), map_location="cpu")
    loaded = model.load_weights(llm_state.items())

    # Verify the overlay actually populated the real CosyVoice3 weights — the BlankEN pass
    # loads nothing through our load_weights, so a silent miss here would leave random/stale
    # parameters. llm.pt is the full CosyVoice3 LM, so its overlay must cover the speech-specific
    # heads and every model parameter.
    required_heads = {"speech_embedding.weight", "llm_decoder.weight"}
    missing_heads = sorted(required_heads - loaded)
    if missing_heads:
        raise RuntimeError(
            f"CosyVoice3 llm.pt overlay did not load required heads: {missing_heads}"
        )
    unloaded = sorted({name for name, _ in model.named_parameters()} - loaded)
    if unloaded:
        raise RuntimeError(
            f"CosyVoice3 AR weights left uninitialized after llm.pt overlay "
            f"({len(unloaded)} params), e.g. {unloaded[:8]}"
        )

    # Share the loaded model with the preprocessing stage (same process).
    set_cosyvoice3_preprocessing_context(model=model)

    if want_cuda_graph:
        model_worker.model_runner.init_device_graphs()

    output_proc = SGLangOutputProcessor(
        capture_hidden=False,
        capture_hidden_layers=None,
        model=model,
    )
    request_builder, result_adapter = make_cosyvoice3_scheduler_adapters(model=model)

    return OmniScheduler(
        tp_worker=model_worker,
        tree_cache=tree_cache,
        req_to_token_pool=req_to_token_pool,
        token_to_kv_pool_allocator=token_to_kv_pool_allocator,
        server_args=server_args,
        model_config=model_config,
        prefill_manager=prefill_mgr,
        decode_manager=decode_mgr,
        model_runner=CosyVoice3ModelRunner(model_worker, output_proc),
        request_builder=request_builder,
        result_adapter=result_adapter,
        abort_callback=cleanup_prepared_cosyvoice3_request,
    )


def create_tts_engine_executor(*args, **kwargs) -> Any:
    return create_sglang_tts_engine_executor(*args, **kwargs)


# ---------------------------------------------------------------------------
# Vocoder — returns batch-1 SimpleScheduler (flow -> hift)
# ---------------------------------------------------------------------------

# Top-level yaml keys whose blocks must be dropped before instantiation: they
# reference un-vendored training/dataset/GAN modules (or already-dropped keys).
_VOCODER_DROP_BLOCK_KEYS = frozenset(
    {
        "hifigan",
        "data_pipeline",
        "data_pipeline_gan",
        "train_conf",
        "train_conf_gan",
    }
)
# Top-level `!new:` blocks to KEEP (everything else with !new:/!name: is dropped).
_VOCODER_KEEP_BLOCK_KEYS = frozenset({"flow", "hift"})

_TOP_LEVEL_HEADER_RE = re.compile(r"^([A-Za-z_][\w]*):")


def _slice_flow_hift_yaml(text: str) -> str:
    """Slice ``cosyvoice3.yaml`` down to scalar params + the flow & hift blocks.

    The full cosyvoice3.yaml is a TRAINING config: loading it whole would
    instantiate un-vendored dataset/discriminator modules. We keep the scalar
    params (referenced by ``!ref``) plus the ``flow`` and ``hift`` blocks, drop
    every ``!new:``/``!name:`` block that is not flow/hift, then rewrite the
    ``cosyvoice.`` / ``matcha.`` module prefixes to the vendored package path.
    """
    lines = text.splitlines()
    header_indices = [i for i, ln in enumerate(lines) if _TOP_LEVEL_HEADER_RE.match(ln)]
    kept: list[str] = []
    for pos, start in enumerate(header_indices):
        end = header_indices[pos + 1] if pos + 1 < len(header_indices) else len(lines)
        block = lines[start:end]
        key = _TOP_LEVEL_HEADER_RE.match(block[0]).group(1)
        header = block[0]
        is_module = ("!new:" in header) or ("!name:" in header)
        if key in _VOCODER_KEEP_BLOCK_KEYS:
            keep = True
        elif key in _VOCODER_DROP_BLOCK_KEYS:
            keep = False
        elif is_module:
            keep = False
        else:
            # Scalar params and __set_seed* apply lines (referenced via !ref).
            keep = True
        if keep:
            kept.extend(block)

    sliced = "\n".join(kept) + "\n"
    pkg = "sglang_omni.models.cosyvoice3."
    for tag in ("!new:", "!name:", "!apply:"):
        sliced = sliced.replace(f"{tag}cosyvoice.", f"{tag}{pkg}cosyvoice.")
        sliced = sliced.replace(f"{tag}matcha.", f"{tag}{pkg}matcha.")
    return sliced


def _build_cosyvoice3_flow_hift(checkpoint_dir: str, device: str):
    """Instantiate + load the vendored flow and hift from the sliced yaml."""
    from hyperpyyaml import load_hyperpyyaml

    yaml_path = os.path.join(checkpoint_dir, "cosyvoice3.yaml")
    with open(yaml_path, encoding="utf-8") as f:
        sliced = _slice_flow_hift_yaml(f.read())

    # The vendored CausalConditionalCFM constructor (and any seed directives the sliced yaml
    # still carries) reset Python/NumPy/CPU/CUDA global RNG. Save + restore RNG state around
    # construction so building the colocated vocoder does not perturb the AR sampler or other
    # stages sharing this process.
    _rng_state = (
        random.getstate(),
        np.random.get_state(),
        torch.get_rng_state(),
        torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    )
    try:
        configs = load_hyperpyyaml(sliced)
    finally:
        random.setstate(_rng_state[0])
        np.random.set_state(_rng_state[1])
        torch.set_rng_state(_rng_state[2])
        if _rng_state[3] is not None:
            torch.cuda.set_rng_state_all(_rng_state[3])

    flow = configs["flow"]
    hift = configs["hift"]

    flow.load_state_dict(
        torch.load(os.path.join(checkpoint_dir, "flow.pt"), map_location="cpu"),
        strict=True,
    )
    hift_state = torch.load(os.path.join(checkpoint_dir, "hift.pt"), map_location="cpu")
    # hift.pt stores the HiFi-GAN generator under a ``generator.`` prefix and also carries
    # extra (discriminator / f0) keys the inference generator does not use, so load
    # non-strict but assert no generator weight is left random (missing); unexpected extra
    # checkpoint keys are expected and ignored.
    hift_load = hift.load_state_dict(
        {k.replace("generator.", ""): v for k, v in hift_state.items()},
        strict=False,
    )
    if hift_load.missing_keys:
        raise RuntimeError(
            "CosyVoice3 HiFi-GAN generator is missing weights from hift.pt "
            f"({len(hift_load.missing_keys)}): {hift_load.missing_keys[:8]}"
        )

    flow.to(device).eval()
    hift.to(device).eval()
    return flow, hift


def create_vocoder_executor(
    model_path: str,
    *,
    device: str = "cuda:0",
    gpu_id: int | None = None,
    dtype: str = "float32",
) -> SimpleScheduler:
    """Build the flow + hift vocoder and return a batch-1 SimpleScheduler.

    ``dtype`` is accepted for config parity; flow/hift run in fp32 (the flow
    asserts batch==1 and the hift f0 predictor needs fp64), which matches the
    validated reference pipeline.
    """
    del dtype
    if gpu_id is not None:
        device = f"cuda:{gpu_id}"
    checkpoint_dir = _resolve_checkpoint(model_path)
    flow, hift = _build_cosyvoice3_flow_hift(checkpoint_dir, device)
    logger.info("CosyVoice3 vocoder ready (device=%s)", device)

    def _to_device_tensor(value: Any, dtype: torch.dtype) -> torch.Tensor:
        tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
        return tensor.to(device=device, dtype=dtype)

    def _vocode(payload: StagePayload) -> StagePayload:
        state = CosyVoice3State.from_dict(payload.data)
        if state.speech_tokens is None:
            raise RuntimeError(
                "CosyVoice3 vocoder requires speech_tokens from tts_engine"
            )

        token = _to_device_tensor(state.speech_tokens, torch.long).reshape(1, -1)
        token_len = torch.tensor([token.shape[1]], dtype=torch.int32, device=device)
        prompt_token = _to_device_tensor(state.prompt_speech_token, torch.long).reshape(
            1, -1
        )
        prompt_token_len = torch.tensor(
            [prompt_token.shape[1]], dtype=torch.int32, device=device
        )
        prompt_feat = _to_device_tensor(state.prompt_feat, torch.float32)
        if prompt_feat.ndim == 2:
            prompt_feat = prompt_feat.unsqueeze(0)  # [1, T_mel, 80]
        prompt_feat_len = torch.tensor(
            [prompt_feat.shape[1]], dtype=torch.int32, device=device
        )
        embedding = _to_device_tensor(state.flow_embedding, torch.float32)
        if embedding.ndim == 1:
            embedding = embedding.unsqueeze(0)  # [1, 192]

        with torch.no_grad():
            mel, _ = flow.inference(
                token=token,
                token_len=token_len,
                prompt_token=prompt_token,
                prompt_token_len=prompt_token_len,
                prompt_feat=prompt_feat,
                prompt_feat_len=prompt_feat_len,
                embedding=embedding,
                streaming=False,
                finalize=True,
            )
            # NOTE: OpenAI `speed` is deliberately NOT applied here. The serving
            # layer applies it generically to every model's waveform at response
            # encoding (openai_api.py `apply_speed`); a second, model-level mel
            # interpolation (upstream CosyVoice's mechanism) would compound to
            # speed**2. Verified e2e: service-layer alone halves/doubles duration.
            wav, _ = hift.inference(speech_feat=mel.float())

        audio_payload = audio_waveform_payload(
            wav,
            sample_rate=CV3_SAMPLE_RATE,
            modality="audio",
            source_hint="CosyVoice3",
        )

        # Drop the heavy reference tensors from the terminal payload.
        state.prompt_speech_token = None
        state.prompt_feat = None
        state.flow_embedding = None
        state.speech_tokens = None
        state.sample_rate = CV3_SAMPLE_RATE

        data = state.to_dict()
        data.update(audio_payload)
        data["sample_rate"] = CV3_SAMPLE_RATE
        data["modality"] = "audio"
        usage = _build_usage(state)
        if usage is not None:
            data["usage"] = usage
        return StagePayload(
            request_id=payload.request_id,
            request=payload.request,
            data=data,
        )

    # flow.inference asserts batch == 1, so no batch_compute_fn.
    return SimpleScheduler(_vocode)
