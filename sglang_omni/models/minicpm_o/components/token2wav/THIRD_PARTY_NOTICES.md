# MiniCPM-o Token2wav

This directory maintains the inference components used by MiniCPM-o.
It does not include model weights.

## Sources

- Behavioral reference: `minicpmo-utils==1.0.6`, source distribution SHA-256
  `3b35de45a46db449079d96fc96723df4151ab8db319888016edfd9475f9406c3`.
- Flow, Conformer, DiT, HiFT, and prompt processing:
  [Step-Audio2](https://github.com/stepfun-ai/Step-Audio2), inspected at commit
  `76e272b56c3917a8d7188f18bbb5a65dfc8a0845` (Apache-2.0).
- Speech tokenizer: the modified S3Tokenizer V2 implementation distributed in
  the reference package, derived from
  [S3Tokenizer](https://github.com/xingchensong/S3Tokenizer) (Apache-2.0).
  In particular, its FSQ quantization, LayerNorm epsilon, and rotary precision
  are preserved; the separately published tokenizer is not an equivalent
  replacement.

Original copyright notices remain in the source files. The Apache-2.0 license
is included in `licenses/Apache-2.0.txt`.

## Additional HiFT Attributions

The HiFT layers identify the following sources. Their MIT notices are included:

- [HiFi-GAN](https://github.com/jik876/hifi-gan): `licenses/HiFi-GAN.txt`.
- [ParallelWaveGAN](https://github.com/kan-bayashi/ParallelWaveGAN):
  `licenses/ParallelWaveGAN.txt`.
- [BigVGAN](https://github.com/NVIDIA/BigVGAN): `licenses/BigVGAN.txt`.
- [Snake](https://github.com/EdwardDixon/snake): `licenses/Snake.txt`.

## Additional Model Attributions

- The S3 tokenizer contains layers derived from
  [Whisper](https://github.com/openai/whisper). Its MIT notice is included in
  `licenses/Whisper.txt`.
- The DiT timestep embedding is derived from
  [GLIDE](https://github.com/openai/glide-text2im). Its MIT notice is included in
  `licenses/GLIDE.txt`.

## Local Changes

Only the full-utterance inference path is maintained here. Unused streaming
caches, CUDA graphs, training utilities, downloaders, and CLI entry points
were removed. Imports use the local package namespace. The checkpoint YAML
loader accepts only the four known flow component tags and does not import
arbitrary Python objects. Device placement is explicit.

Speaker-reference tokenization retains the 30-second window with 4-second
overlap. Tokenizer log-mel extraction uses the already-required Whisper
implementation; the vocoder's distinct 24 kHz mel configuration is retained.
Weights continue to load strictly from the original checkpoint files.
