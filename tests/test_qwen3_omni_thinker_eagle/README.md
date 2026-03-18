# Qwen3-Omni Thinker EAGLE3 Speculative Decoding

This directory contains test scripts and dummy configurations to verify the integration of EAGLE3 Speculative Decoding for the Qwen3-Omni Thinker model within the `sglang-omni` framework.

## Supported Features
1. **Target Model Adaptation**: Modified `Qwen3OmniMoeThinkerTextModel` (`sglang_omni/models/qwen3_omni/thinker.py`) to support autoregressive capabilities by integrating `ParallelLMHead` and `LogitsProcessor`. Added interface to capture the final layer's hidden states (`aux_hidden_states`).
2. **Draft Model Implementation**: Created a lightweight, dense Draft Model `Qwen3OmniEagle3DraftModel` (`sglang_omni/models/qwen3_omni/draft.py`) that can reuse the target model's embeddings and LM head.
3. **Engine Integration**: Updated `server_args_builder.py` and `stages.py` to seamlessly mount the draft model alongside the target model in a single execution stage (`THINKER_STAGE`) to minimize cross-stage communication overhead.
4. **Framework Compatibility**: Patched `sglang` core modules (e.g., `RMSNorm` monkey patches, `AttentionBackend` mocking) to ensure smooth execution of the multi-model forward pass on a single GPU.

## Modified/Added Files
- `sglang_omni/models/qwen3_omni/thinker.py`: Target model modifications.
- `sglang_omni/models/qwen3_omni/draft.py`: (New) Draft model definition.
- `sglang_omni/engines/ar/sglang_backend/server_args_builder.py`: Speculative decoding arguments support.
- `sglang_omni/models/qwen3_omni/pipeline/stages.py`: Pipeline stage adjustments.
- `sglang_omni/vendor/sglang/layers.py`: Framework level patches.

## How to Run the Tests

Since the true 30B model requires significant VRAM, we provide scripts to generate a tiny dummy model (<100MB) to verify the structural forward pass and generate loop logic on a single local GPU (e.g., RTX 4090).

### 1. Generate Dummy Weights
Run the generation script to create tiny safetensors weights that perfectly match the Hugging Face unfused formats required by the models.
```bash
python gen_dummy_qwen3_omni_eagle3.py
```
This will create `tiny-qwen3-omni` and `tiny-qwen3-omni-draft` directories (ignored by git).

### 2. Run the Forward & Generate Loop Test
Execute the test script to simulate a multi-step autoregressive generation process where the Target Model and Draft Model work synergistically.
```bash
python test_forward_qwen3_omni_eagle3.py
```
**Expected Output**: The script will initialize both models on CUDA, simulate a 4-step generate loop by incrementally updating `seq_lens` and `positions` in `ForwardBatch`, and print the tokens speculated/predicted by both models at each step without any framework crashes.
