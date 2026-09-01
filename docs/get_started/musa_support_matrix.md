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

## Cookbooks And Evidence

Each row below links the model to the primary cookbook or install page and to
the MUSA record that captured the smoke and bench evidence.

| Model | Cookbook / install guide | MUSA evidence |
| --- | --- | --- |
| Audar-TTS V1 Turbo | `docs/get_started/installation_musa_cloud.md` | `musa_cloud_offline_install/tts_models_musa_adaptation_20260827/README.md` |
| dots.tts | `docs/cookbook/dots_tts.md` | `musa_cloud_offline_install/tts_models_musa_adaptation_20260827/README.md` |
| Fish Audio S2-Pro | `docs/cookbook/fishaudio_s2_pro.md` | `musa_cloud_offline_install/tts_models_musa_adaptation_20260827/README.md` |
| Fun-CosyVoice3 | `docs/cookbook/fun_cosyvoice3.md` | `musa_cloud_offline_install/NEXT_TTS_MODELS_BENCH_LIST.md` |
| Higgs Audio v3 TTS | `docs/cookbook/higgs_tts.md` | `musa_cloud_offline_install/NEXT_TTS_MODELS_BENCH_LIST.md` |
| Ming-Omni-TTS | `docs/cookbook/ming_tts.md` | `musa_cloud_offline_install/NEXT_TTS_MODELS_BENCH_LIST.md` |
| MOSS-TTS | `docs/cookbook/moss_tts.md` | `musa_cloud_offline_install/NEXT_TTS_MODELS_BENCH_LIST.md` |
| MOSS-TTS Local | `docs/cookbook/moss_tts_local.md` | `musa_cloud_offline_install/NEXT_TTS_MODELS_BENCH_LIST.md` |
| Qwen3-TTS | `docs/cookbook/qwen3_tts.md` | `musa_cloud_offline_install/NEXT_TTS_MODELS_BENCH_LIST.md` |
| Voxtral TTS | `docs/cookbook/voxtral_tts.md` | `musa_cloud_offline_install/NEXT_TTS_MODELS_BENCH_LIST.md` |
| ZONOS2 | `docs/cookbook/zonos2.md` | `musa_cloud_offline_install/NEXT_TTS_MODELS_BENCH_LIST.md` |

## Notes

- The matrix is model-level. It does not try to enumerate every temporary
  fallback component inside a model path.
- The install guide covers the shared cloud workflow and points to the
  verified MUSA offline path.
- The cookbook pages remain the canonical launch and request examples.
