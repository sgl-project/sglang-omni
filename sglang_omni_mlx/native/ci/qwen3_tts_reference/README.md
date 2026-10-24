# Qwen3-TTS reference qualification

The independent migration reference is the Python MLX implementation in
[PR #1960](https://github.com/sgl-project/sglang-omni/pull/1960), commit
`f4ed5942633e790f59e57b3f150bbffaa21e395d`. Its talker, speech tokenizer and
speaker encoder derive from mlx-audio; preserve their MIT notices.

Use the immutable official checkpoint revisions and file hashes in
`../model_pins.json`, and MLX 0.32.3 from `../../scripts/build-tools.lock`.
Run the reference in an isolated environment; serving uses the C++ executables.
Do not restore the legacy scheduler or worker into `sglang_omni/`.

`requests.json` fixes initial CustomVoice inputs, languages, voices, seeds and
frame caps. Before recording native per-chip regression outputs, independently
compare prompt embeddings, talker prefill and feedback logits, all code groups,
and fixed-code waveform decoding against that reference. Add a 96-frame fixed-code
case to cover the decoder's window-72 mask, and compare one-shot versus irregular
streaming chunks. For Base, compare mel features, speaker embeddings, encoded
reference codes and reference-audio trimming.

The native golden checker consumes `kind="qwen3_tts"` model goldens through the
existing `check_model_golden.py` entry point. Each case output contains the CLI's
`prompt_ids`, `control_ids`, `instruction_ids`, `reference_text_ids`, `codes`,
`finish_reason`, `sample_count`, `sample_rate`, plus `waveform_sha256` over its WAV.
No reference outputs have been generated or qualified in this change. Mac CI
must remain blocked on that missing qualification rather than accepting an empty
baseline or promoting native outputs to an independent reference.
