# ROCm model support

The authoritative declarations live in
`sglang_omni.models.accelerator_support`. This page distinguishes implemented
portability from executed hardware validation.

Status meanings:

- **supported**: the current revision passed its model-specific E2E contract on
  both gfx942 and gfx950.
- **preview**: a ROCm-safe framework path and basic functional E2E exist, but
  the complete CUDA feature contract or scheduled stability gates remain open.
- **unsupported**: startup is intentionally rejected.

| Wave | Architecture | gfx942 | gfx950 | Status |
| --- | --- | --- | --- | --- |
| 1 | `MiniMaxMusic3ForConditionalGeneration` | validated | validated | supported |
| 1 | `Qwen3OmniMoeForConditionalGeneration` | functional | functional | preview |
| 1 | `Qwen3TTSForConditionalGeneration` | functional | functional | preview |
| 1 | `Qwen3ASRForConditionalGeneration` | functional | functional | preview |
| 1 | `FunAsrNanoForConditionalGeneration` | functional | functional | preview |
| 1 | `HiggsMultimodalQwen3ForConditionalGeneration` | functional | functional | preview |
| 1 | `MossTTSDelayModel` | functional | functional | preview |
| 1 | `MossTranscribeDiarizeForConditionalGeneration` | functional | functional | preview |
| 2 | `WhisperForConditionalGeneration` | functional | functional | preview |
| 2 | `ArkasrForConditionalGeneration` | functional | functional | preview |
| 2 | `DotsTTSForConditionalGeneration` | functional | functional | preview |
| 2 | `AudarTTSForConditionalGeneration` | functional | functional | preview |
| 2 | `FishQwen3OmniForCausalLM` | functional | functional | preview |
| 2 | `MossTTSLocalModel` | functional | functional | preview |
| 2 | `BailingMMNativeForConditionalGeneration` | functional | functional | preview |
| 2 | `VoxtralTTSForConditionalGeneration` | functional | functional | preview |
| 2 | `Zonos2ForCausalLM` | functional | functional | preview |
| 3 | `BailingMM2NativeForConditionalGeneration` | functional | functional | preview |
| 3 | `LLaDA2MoeModelLM` | functional | functional | preview |

MiniMax Music 3 was validated with ROCm 7.2 and SGLang 0.5.16 on MI300X and
MI355X through `/v1/audio/speech`. Generation and RVQ graphs are disabled; AR
uses AITER while the compiled DIT/DAV acoustic stage uses Torch SDPA.

## Promotion gate

A model is promoted to `supported` only in the same runtime PR that records:

1. Exact source SHA and image digest for both architectures.
2. Server boot and health, a real API request, output correctness, and clean
   teardown.
3. Streaming, batching, TP, quantization, or modality checks matching the
   corresponding CUDA-supported feature set.
4. Raw logs and artifacts from both hardware runs.

General ROCm support is not declared until all 19 entries are supported, the
two required runner lanes pass, three consecutive scheduled full matrices
pass, and the two-node GPU-memory NIXL gate passes. Host-memory verbs alone do
not satisfy the distributed gate.
