# Capture-safe GPU radix hash (MOSS-TTS Local generated rows)

## Problem

Every decoded frame on the Local v1.5 path appends one radix-cache token id to
the request's KV chain; the radix tree keys on those ids. Because the text
channel alone is the same assistant-slot id for every continuing frame, the key
must hash the **full** 13-channel row (text + 12 RVQ codes) so that a radix
prefix match implies identical audio content (otherwise a re-prefill after
retraction could falsely prefix-match into another identical-prompt request's
cached generated region).

The previous generated-row key reused the prompt-path helper
`moss_tts.request_builders.build_row_cache_key_ids`, which moves the row tensor
to host (`row.numpy().tobytes()`) and hashes with blake2b. On the prompt path
that is fine -- it runs once, before decode, outside any CUDA-graph capture
region. On the **generated**-row path it runs every decode step on a tensor the
local-frame decode just produced on device, so it forces a
GPU->CPU sync + host hashing + CPU->GPU upload of `next_token_ids` **per frame**.
That host sync blocks CUDA-graph capture and the async-decode lookahead
(#734 / #736), which this PR is preparing the path for.

## Approach

`sglang_omni/models/moss_tts_local/radix_hash.py` fuses token-row construction
and radix hashing into one Triton kernel on CUDA. The hash algorithm is
unchanged; inputs unsupported by the Triton path use the original int64 Torch
implementation:

```
acc = 0
for channel in range(C):                       # C is fixed by the model configuration
    acc = (acc * BASE + row[:, channel]) % MOD
key = acc % HASH_SPACE                          # continuing frames
key = audio_end_id                              # EOS rows (torch.where)
```

* `MOD = 2**31 - 1` (Mersenne prime M31), `BASE = 1_000_000_007` (prime < MOD).
* `HASH_SPACE = 151643` -- `<|endoftext|>` opens the special/control id band.
  Continuing keys fold strictly below it; EOS rows keep the raw `audio_end` id
  (in the band) so `Req._check_vocab_boundary_finish` still fires eos.

The prompt path is **unchanged**: `build_row_cache_key_ids` (blake2b) stays for
prompt preprocessing. Generated rows now use `build_rows_and_radix_token_ids`
to fuse row construction and GPU hashing. This boundary is deliberate -- the prompt hash
is not on the decode hot path and never enters a capture region.

## Capture-safety argument

The Triton kernels compute the hash entirely on device, without host readback
or synchronization. Warm up the kernels before CUDA-graph capture. The Torch
fallback remains capture-safe on CUDA.

### No int64 overflow

With `acc < MOD` and each channel value reduced `< MOD` (both `< 2**31`) and
`BASE < MOD`, every Horner step `acc * BASE + v < 2**31 * 2**31 = 2**62`, well
inside signed int64 (`< 2**63`). So the arithmetic is exact and bit-reproducible
on CPU and GPU -- no implementation-defined wraparound. This is asserted in
`test_matches_python_reference` against a pure-Python bignum reference.

## Collision analysis

The key only drives radix cache hit/miss; it never changes sampled output (see
two-layer verification below). A clash matters only when two **distinct** audio
rows produce the same key **and** sit on top of an otherwise identical full
prefix -- the same risk profile the prior blake2b-then-fold key already carried.

* Raw polynomial space is `MOD ~ 2.1e9`. Horner is order-sensitive (a channel
  permutation changes the key) and spreads neighbours (a single-channel +/-1
  shifts the key by a power of `BASE mod MOD`), so structurally similar rows do
  not cluster. Raw-hash injectivity on a constructed adjacent/permuted/random
  batch is asserted in `test_no_collisions_on_adjacent_and_permuted_rows`.
* The fold to `HASH_SPACE = 151643` is the operative space (identical to the
  prior key's fold). Expected folded clashes for `n` co-resident generated rows
  on a shared prefix are `~ n^2 / (2 * 151643)` -- e.g. < 0.02 for n = 75. A fold
  clash degrades to a single-position radix false-match (~1/151643) that only
  takes effect atop an identical full prefix; it cannot corrupt output, only
  cache locality.

## Two-layer verification rubric

Replacing the key algorithm means the old "both sides run the same blake2b"
parity assumption no longer holds, so verification splits into two independent
layers:

1. **Output layer (bit-identity).** `audio_codes` / `output_ids` are produced by
   sampling, which does **not** depend on the radix key value. The fixed-seed S0
   gate (`tests/unit_test/moss_tts_local/test_s0_gate.py`, gpu-marked) and the
   graph-vs-eager sentinel assert bit-identical decoded output before vs after
   this change. **Status: PASS** (H100 at the review head: S0 3/3, the
   graph-vs-eager sentinel, and the on-GPU `moss_tts_local` suite).
2. **Key layer (algorithm properties).** Determinism, collision behaviour, EOS
   preservation, output domain, and dtype/device-follow are covered as CPU unit
   tests in `tests/unit_test/moss_tts_local/test_radix_hash.py` (green on CPU).

The output layer is the behaviour-neutrality guarantee; the key layer pins the
hash's own contract independent of any GPU run.
