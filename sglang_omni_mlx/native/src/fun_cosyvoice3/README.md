# CosyVoice3 native LLM probe

Work in progress for #2888. This is a numeric-ID LLM probe, not a complete TTS implementation or server. Flow/DiT, HiFT, reference-audio preprocessing, tokenization, sampling and streaming are not implemented here. Quantized checkpoint loading is implemented but not yet runtime-validated.

## Reference and scope

The reference is `sglang_omni/models/fun_cosyvoice3/mlx/model.py` at `28c3e35a33bef2f45b3315431f4a512d6ed463cf`, immediately before the legacy path was removed. The parity script verifies the reference file SHA-256 before executing it. Retrieve that file from Git history and supply it explicitly; the script does not download or execute an unpinned remote module.

The probe reuses the existing decoder with Q/K normalization disabled and half-split RoPE, loads the fine-tuned Qwen2 weights, builds `[SOS, text, TASK, reference speech]`, and projects the final normalized hidden state through the separate speech head. Text IDs must already include any reference transcript and tokenizer control tokens. Forced decode IDs exercise the KV cache without sampling.

Build the `cosyvoice3_llm_probe` CMake target using MLX 0.32.3. Run `cosyvoice3_llm_probe --help` for the standalone CLI. Its logits output is a float32 NPY array shaped `[1 + decode_token_count, 6761]`.

## Reproduce the comparison

From the repository root, with MLX 0.32.3, mlx-lm 0.32.2, PyTorch and NumPy available:

```sh
python sglang_omni_mlx/native/ci/cosyvoice3_llm_parity.py \
  --reference-file /path/to/pinned/model.py \
  --raw-model /path/to/official/model-bundle \
  --probe /path/to/build/cosyvoice3_llm_probe \
  --output-dir /path/to/local-validation-output
```

The official bundle must contain `llm.pt` and `CosyVoice-BlankEN/config.json`. The script exports an FP16 converted checkpoint using the legacy loader, records source and output SHA-256s and package versions, and compares prompt embeddings and prefill/decode logits. Use a local filesystem: the existing shared safetensors loader also attempts to load macOS AppleDouble files such as `._model.safetensors` on some network volumes.

## Initial measured results

With Python 3.12.15, MLX 0.32.3, mlx-lm 0.32.2, PyTorch 2.14.1 and NumPy 2.3.5 on Apple Silicon:

- Two fixed numeric-ID prompts, text-only and text plus reference speech, each with two forced decode steps: native/reference embeddings and every logits row matched exactly (maximum absolute difference 0).
- Six malformed/out-of-range input cases were rejected as expected.
- The full harness **fails** its separate cached-versus-full-prefix check for the text-only prompt at `atol=0.02`, `rtol=0.002`. Both Python and native have maximum absolute difference 0.0302734375, with matching argmax. The text-plus-speech case passes (maximum difference 0.01171875). Tolerances have not been relaxed.

These small fixtures establish initial implementation parity only, not speech quality, model-wide correctness, quantized correctness, throughput, or end-to-end TTS support. Investigate FP16 cache/full-prefix differences before treating the full harness as a passing regression gate. Mac CI goldens and full model pins are still pending.
