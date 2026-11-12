# YuE2

YuE2 is an AR-NAR music model: lyrics + style/tags -> ABC plan -> semantic codec
tokens -> NAR acoustic latents -> 48 kHz stereo song.

## Endpoint

Served on `/v1/audio/speech`:

- `input`: lyrics
- `instructions`: style / tags (e.g. `soft pop, female vocal`)
- `params.max_new_tokens`: semantic codec budget (default 9000)
- `tts_params.seed`, `tts_params.cot` (`off|melody|full`), `tts_params.abc`
  (external ABC score), `tts_params.cfg_scale`

## Stages

- `preprocessing`: request -> `Yue2State`.
- `yue2_synth` (terminal, one GPU): AR codec -> NAR acoustic -> VAE waveform.

The AR-NAR math and the SGLang-YuE2 optimizations (fused decode CUDA graphs,
fast sampler, FA3 varlen, batched modules) are ported in this package; see
`MIGRATION.md`.

## Notes

- Requires `sgl_kernel` for the fused decode path; otherwise the reference NAR
  is used.
- DeepSelect (`sglang.kernels.ops.deep_select`) is preferred for sampling, with a
  `torch.topk` fallback.
