# MUSA Hardware / Backend / Model Support Matrix

This page summarizes the current MUSA validation surface for the models we
adapted in this branch. It is meant to sit next to the cookbook pages and the
MUSA installation guide, so readers can move from a matrix view to a runnable
recipe without searching through experiment logs.

Legend:

- `✅` = validated end-to-end on this backend in the current branch
- `🟡` = usable only with a documented fallback or conditional path
- `❌` = not validated on this backend in the current branch

## Matrix

| Model | CUDA | XPU | NPU | ROCm | MUSA | CPU |
| --- | --- | --- | --- | --- | --- | --- |
| Audar-TTS V1 Turbo | ✅ | 🟡 | 🟡 | 🟡 | ✅ | 🟡 |
| dots.tts | ✅ | 🟡 | 🟡 | 🟡 | ✅ | 🟡 |
| Fish Audio S2-Pro | ✅ | 🟡 | 🟡 | 🟡 | ✅ | ❌ |
| Fun-CosyVoice3 | ✅ | 🟡 | 🟡 | 🟡 | ✅ | ❌ |
| Higgs Audio v3 TTS | ✅ | 🟡 | 🟡 | 🟡 | ✅ | ❌ |
| Ming-Omni-TTS | ✅ | 🟡 | 🟡 | 🟡 | ✅ | ❌ |
| MOSS-TTS | ✅ | 🟡 | 🟡 | 🟡 | ✅ | ❌ |
| MOSS-TTS Local | ✅ | 🟡 | 🟡 | 🟡 | ✅ | ❌ |
| Qwen3-TTS | ✅ | 🟡 | 🟡 | 🟡 | ✅ | ❌ |
| Voxtral TTS | ✅ | 🟡 | 🟡 | 🟡 | ✅ | ❌ |
| ZONOS2 | ✅ | 🟡 | 🟡 | 🟡 | ✅ | ❌ |

## Bench Summary

The rows below summarize the validated MUSA evidence directly in this PR instead
of pointing at checkout-local markdown files. That keeps the page self-contained
when a local log note has not been checked into the branch.

| Model | Cookbook entry point | Bench / smoke summary |
| --- | --- | --- |
| Audar-TTS V1 Turbo | `docs/get_started/installation_musa_cloud.md` | MUSA smoke and adaptation recorded in the branch evidence notes |
| dots.tts | `docs/cookbook/dots_tts.md` | MUSA smoke and adaptation recorded in the branch evidence notes |
| Fish Audio S2-Pro | `docs/cookbook/fishaudio_s2_pro.md` | MUSA smoke and adaptation recorded in the branch evidence notes |
| Fun-CosyVoice3 | `docs/cookbook/fun_cosyvoice3.md` | MUSA smoke and adaptation recorded in the branch evidence notes |
| Higgs Audio v3 TTS | `docs/cookbook/higgs_tts.md` | MUSA smoke and adaptation recorded in the branch evidence notes |
| Ming-Omni-TTS | `docs/cookbook/ming_tts.md` | MUSA smoke and adaptation recorded in the branch evidence notes |
| MOSS-TTS | `docs/cookbook/moss_tts.md` | MUSA smoke and adaptation recorded in the branch evidence notes |
| MOSS-TTS Local | `docs/cookbook/moss_tts_local.md` | MUSA smoke and adaptation recorded in the branch evidence notes |
| Qwen3-TTS | `docs/cookbook/qwen3_tts.md` | MUSA smoke and adaptation recorded in the branch evidence notes |
| Voxtral TTS | `docs/cookbook/voxtral_tts.md` | MUSA smoke and adaptation recorded in the branch evidence notes |
| ZONOS2 | `docs/cookbook/zonos2.md` | MUSA smoke and adaptation recorded in the branch evidence notes |

## Notes

- The matrix is model-level. It does not try to enumerate every temporary
  fallback component inside a model path.
- The install guide covers the shared cloud workflow and points to the
  verified MUSA offline path.
- The cookbook pages remain the canonical launch and request examples.
- Keep the summary self-contained when the backing logs are not checked into
  the PR.
