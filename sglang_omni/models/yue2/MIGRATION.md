# YuE2 in SGLang-Omni

Port of the SGLang-YuE2 music model (lyrics + style -> 48 kHz stereo song) into
the SGLang-Omni multi-stage runtime. Template: `models/minimax_music3/`.

## Layout

Model code ported verbatim from SGLang-YuE2 `python/sglang/srt/models/yue2/`
(+ `model_loader/yue2_loader.py` -> `runtime.py`), imports made package-relative:

    modeling_yue2.py modeling_vae.py tokenization_yue2.py protocol.py
    sampling.py runtime.py cuda_graph.py fastpath.py batched.py
    nar.py nar_fast.py song.py streaming.py generation.py storage.py fa3.py

Glue added for the SGLang-Omni contract:

    __init__.py         CAPABILITIES (cuda_graph=True, torch_compile=False)
    config.py           Yue2PipelineConfig(architecture="YuE2ForCausalLM"), stages, EntryClass
    constants.py        defaults
    payload_types.py    Yue2State (DeclarativeStateBase)
    request_builders.py /v1/audio/speech -> Yue2State
    synth.py            Yue2Synthesizer (AR -> NAR -> VAE)
    stages.py           create_preprocessing_executor, create_synth_executor

Registration is automatic (registry scans `models/*/config.py` for `EntryClass`).
`YuE2ForCausalLM` also added to `tests/unit_test/models/test_model_capabilities.py`.

## Two adaptations for portability
- `fastpath.py`: `_TorchTopKBackend` fallback when DeepSelect is absent
  (SGLang-Omni env has no `sglang.kernels`). DeepSelect still preferred.
- `fa3.py`: falls back to ATen when the in-tree FA3 varlen op is absent.

## Pipeline
`preprocessing` (SimpleScheduler) -> `yue2_synth` (terminal SimpleScheduler, one GPU).
`yue2_synth` (`synth.Yue2Synthesizer`) uses the ported SGLang-YuE2 optimized path:
one pooled `GraphAR` session with the fused decode+sample step graph for AR
(`song.generate_song_tokens`), fused NAR on the borrowed KV with the velocity
graph (`nar_fast.synthesize_from_session`), and tiled VAE decode. It falls back
to the reference `nar.synthesize` only if the fused NAR rejects the request,
then emits `audio_waveform_payload`.

## TODO
- Batched synthesis: `SimpleScheduler(batch_compute_fn=...)` over the ported
  `batched.run_batched_ar` + `nar_fast.synthesize_batched` + VAE same-frame
  `decode_tiled`, to recover the measured ~3.2x concurrency (`bench_yue2.md`).
  Today `yue2_synth` is `max_concurrency=1`, so c>=2 serializes.
- Optional: SGLang AR engine (`OmniScheduler`) if the AR should use paged KV.
- Streaming stages; `yue2_artifacts_dir` for WSB scoring.

## Lint
- Glue files pass `scripts/check_if_else.py` and `scripts/check_leading_underscore.py`.
- Ported model files: if-else normalized (`check_if_else.py --fix`); leading-underscore
  names kept in the source style (`check_leading_underscore.py` still flags them).
