# Fish Audio Drama 3 prefix cache

Drama 3 reuses hybrid-backbone and codebook-tower prefill across requests with shared prompt frames. It addresses [#2797](https://github.com/sgl-project/sglang-omni/issues/2797). Prompt-only prefix caching is enabled by default; reference-audio encoding is not cached.

## Configuration

Prepare the bundled runtime after installation, then start fresh Python workers:

```bash
sgl-omni-prepare-runtime
```

No cache options are needed for the default path. To disable prefix caching, add this option to the serving command:

```text
--tts_engine.engine.disable_radix_cache true
```

When caching is enabled, the builder selects the extra-buffer Mamba strategy. Prompt-only mode records reusable checkpoints during prefill and skips generated-tail checkpoint tracking during decode. Live KV and recurrent state still update normally. The policy is fixed before graph capture and cannot change while serving.

## Correctness constraints

- **Complete frame identity:** tower KV at position `p` also depends on frame `p + 1`. Cache identities hash both complete frames, including residual codebooks. A unique final identity keeps generated acoustic tails private.
- **Matching state boundaries:** a hit restores backbone/tower KV, the FP32 GDN recurrent checkpoint and its convolution window at the same boundary. Reuse stops at an available recurrent checkpoint, which may be shorter than the matching prompt.
- **Suffix-only prefill:** only uncached frames run through prefill; absolute positions and full repetition-aware sampling history remain intact. A one-token graph bucket handles hits leaving only the final prompt row.

Checkpoint buffers still consume GPU memory. Short prompts may have no reusable checkpoint. See [request identity](../../sglang_omni/models/fishaudio_drama3/request_builders.py) and [suffix staging](../../sglang_omni/models/fishaudio_drama3/decode_state.py) for the implementation.

## Runtime dependency

[Omni #3235](https://github.com/sgl-project/sglang-omni/pull/3235) is stacked on [#3234](https://github.com/sgl-project/sglang-omni/pull/3234), which must merge first. #3234 bundles the GDN graph/FP32 snapshot implementation from [SGLang #46619](https://github.com/sgl-project/sglang/pull/46619); #3235 adds the prefix policy from [SGLang #46683](https://github.com/sgl-project/sglang/pull/46683). Omni applies both patches to its pinned base without waiting for either SGLang PR to merge. Docker/CI prepare the runtime automatically; see [installation](../get_started/installation.md#prepare-the-bundled-sglang-runtime) and [patch maintenance](../../sglang_omni/vendor/sglang/patches/README.md).

## Validation

The synthetic GPU regression covers checkpoint reuse, changed residual input, one-token suffix replay and abort/reuse:

```bash
uv run python -m pytest tests/unit_test/fishaudio_drama3/test_graph_engine.py \
  -k prefix_cache_reuses_prompt
```

The H200 c16 prototype passed sampled WER/speaker-similarity gates; measurements and protocol are recorded in [#3235](https://github.com/sgl-project/sglang-omni/pull/3235). The current rebased stack still needs fresh GPU qualification, including retraction/eviction behavior. CPU checks and synthetic tests do not establish real-checkpoint quality or cross-schedule bitwise audio identity.
