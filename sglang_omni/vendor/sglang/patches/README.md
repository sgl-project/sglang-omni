# Bundled SGLang runtime changes

Omni pins SGLang to b74fd392e18bb9745e6a47698ea3998bc22614f8, the upstream merge of [#45474](https://github.com/sgl-project/sglang/pull/45474). The GDN prefill graph comes directly from that commit. runtime.json records the base commit and expected source/output hashes; the only bundled patch is the FP32 snapshot fix from [#48682](https://github.com/sgl-project/sglang/pull/48682). Prefix-cache policy belongs to Omni #3235.

The upstream graph defaults to off; Drama3 opts in through its tts_engine stage environment, preserving explicit overrides. FP32 snapshots are opt-in through enable_gdn_fp32_snapshots (default false); Drama3 enables the parameter explicitly. When enabled, tracking preserves FP32 snapshots for FP32 state pools with FP16/BF16 prefill activations. The selection is passed to graph metadata and the kernel through arguments. Other dtype combinations retain their existing snapshot path; the shared SGLang source upgrade requires model regression testing.

Build the pinned source with its Rust toolchain and build dependencies, then run sgl-omni-prepare-runtime after installing Omni. It copies the unmodified package into an environment-local directory, applies the patch and activates the result for new Python processes. Docker and CI run preparation automatically. The source installation and native extensions are preserved, including their relative Torch-library lookup.

When advancing the SGLang pin, regenerate the source/output hashes and verify the patch. Remove it once the pinned commit includes the fix. Earlier source-overlay GPU results do not qualify the complete pinned dependency or a newly prepared environment.
