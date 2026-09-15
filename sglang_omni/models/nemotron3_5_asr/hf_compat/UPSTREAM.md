# Transformers compatibility provenance

This package backports the Nemotron ASR implementation from Hugging Face
Transformers `v5.13.0`, commit
`6af945f436d85f2b0c5dff9b14feccd27b1d470b`, onto this repository's pinned
Transformers 5.12.1 runtime. The reused Parakeet base is from Transformers
`v5.12.1`, commit `ddb849abe009d1089e6c691bfc897f27211c663c`.

The vendored Python modules come from these upstream directories:

- `src/transformers/models/nemotron_asr_streaming/`
- `src/transformers/models/nemotron3_5_asr/`
- `src/transformers/models/parakeet/generation_parakeet.py` (only the 5.13
  cache-aware encoder-kwargs delta; the implementation otherwise reuses 5.12.1)

Their original Apache-2.0 copyright and license headers are retained. Local
changes replace package-relative imports with this compatibility package,
load the checkpoint's nested feature extractor without mutating global Auto
mappings, and reuse Transformers 5.12.1's Parakeet generation implementation
with only the 5.13 cache-aware encoder-kwargs delta.

The backport omits the older `NemotronAsrStreaming` RNN-T model, decoder,
joint network, output type, and top-level config. Nemotron 3.5 defines its
own RNN-T head and config; the shared encoder, encoder config, generation
mixin, and decoder cache remain. The encoder base config annotation is
narrowed to `NemotronAsrStreamingEncoderConfig`.

The processor retains plain-text `decode` and `batch_decode`, including
RNN-T repeated-token handling, but omits token-level timestamp post-processing.
Passing non-`None` `durations` to `decode` is unsupported and raises `ValueError`.

Regenerate this directory from the pinned upstream commit and reapply those
compatibility changes and scope reductions when updating it. Remove the
backport once the repository dependency moves to Transformers 5.13 or newer.
