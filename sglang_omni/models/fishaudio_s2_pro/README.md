# FishAudio S2-Pro

Text-to-speech via the `FishQwen3OmniForCausalLM` architecture with DAC VQGAN
codec vocoding.

**This is a fundamentally different model from S1-Mini.** S2-Pro uses:

- `FishQwen3OmniForCausalLM` (HuggingFace `AutoModel`-compatible)
- Built-in audio decoder for codebook generation
- Qwen3 chat-format prompts (`<|im_start|>system/user/assistant<|im_end|>`)
- HuggingFace `PreTrainedTokenizerFast` tokenizer
- Repetition Aware Sampling (RAS)
- Constrained semantic decoding

## Checkpoint Preparation

The S2-Pro checkpoint ships in HuggingFace format. Only the DAC codec
symlink is needed:

```bash
# Symlink the DAC codec from S1-Mini (same codec)
ln -s /path/to/openaudio-s1-mini/codec.pth /path/to/s2-pro/codec.pth
```

## Quick Start

```bash
# Basic TTS
python examples/run_fishaudio_e2e.py \
    --checkpoint /path/to/s2-pro \
    --text "Hello, how are you?" \
    --output output.wav

# Voice cloning
python examples/run_fishaudio_e2e.py \
    --checkpoint /path/to/s2-pro \
    --text "Hello, how are you?" \
    --reference-audio ref.wav --reference-text "Transcript of ref audio." \
    --output output.wav
```

## Architecture

3-stage linear pipeline:
`preprocessing` (CPU) → `tts_engine` (GPU) → `vocoder` (GPU)

### Key differences from S1-Mini

| Aspect | S1-Mini | S2-Pro |
|--------|---------|--------|
| Model class | `DualARTransformer` | `FishQwen3OmniForCausalLM` |
| Loading | Custom `from_pretrained` | `AutoModel.from_pretrained` |
| Tokenizer | `FishTokenizer` (tiktoken) | `PreTrainedTokenizerFast` (HF) |
| Prompt format | `ContentSequence(modality="interleave")` | `Conversation` (Qwen3 chat) |
| Generation | Manual step loop (`inference.py`) | `qwen3.generate` (built-in) |
| Fast decoder | Manual `forward_generate_fast` | Built-in `audio_decoder` |
| Anti-repetition | Repetition penalty | RAS (Repetition Aware Sampling) |
| Constrained decode | Manual masking | `constrain_to_semantic=True` |

| Parameter | S1-Mini | S2-Pro |
|-----------|---------|--------|
| Total params | ~500M | ~4.5B |
| dim | 1024 | 2560 |
| Slow layers | 28 | 36 |
| Fast layers | 4 | 4 |
| Heads (slow) | 16 | 32 |
| KV heads | 8 | 8 |
| Codebooks | 10 | 10 |
| Codebook size | 4096 | 4096 |
| Config format | flat `dual_ar` | nested `fish_qwen3_omni` |
| Weight format | `model.pth` | sharded safetensors |

## Prompt Format

S2-Pro uses the Qwen3 chat format with `<|speaker:0|>` tags:

```
<|im_start|>system
convert the provided text to speech reference to the following:

Text:
<|speaker:0|>{reference_text}

Speech:
[VQ CODES]<|im_end|>
<|im_start|>user
<|speaker:0|>{target_text}<|im_end|>
<|im_start|>assistant
<|voice|>
```

## seed-tts-eval Evaluation

[seed-tts-eval](https://arxiv.org/abs/2406.02430) is the standard zero-shot
TTS benchmark from ByteDance. It measures two metrics:

- **WER** (Word Error Rate): ASR-transcribe the generated audio, compare with
  the target text. Lower is better. English uses Whisper-large-v3.
- **SIM** (Speaker Similarity): Cosine similarity of WavLM speaker embeddings
  between the generated audio and the reference audio. Higher is better.

The English test set contains **1088 samples** from Common Voice.

### Dependencies

```bash
pip install jiwer soundfile scipy transformers
# flash-attn is required by FishQwen3OmniForCausalLM
pip install flash-attn --no-build-isolation
pip install liger-kernel
```

### Running the evaluation via sglang-omni

The eval script runs the full sglang-omni 3-stage pipeline
(preprocessing → tts_engine → vocoder), then transcribes each output
with Whisper-large-v3 and computes WER via `jiwer`.

```bash
# Full 1088-sample evaluation
python examples/run_s2pro_seed_tts_eval.py \
    --checkpoint /path/to/s2-pro \
    --meta /path/to/seed-tts-eval/seedtts_testset/en/meta.lst \
    --output-dir ./s2pro_eval_outputs \
    --tts-device cuda:0 --asr-device cuda:1 \
    --temperature 1.0 --top-p 0.9 --top-k 30

# Quick 10-sample test
python examples/run_s2pro_seed_tts_eval.py --num-samples 10
```

The script writes per-sample WAV files and a JSON summary with WER for
each sample.

### Key implementation notes

1. **`FISH_BATCH_INVARIANT=true`** is required on some systems where
   `liger_kernel`'s Triton SwiGLU kernel has device compatibility issues.
   The eval script sets this automatically. This uses standard PyTorch
   `F.silu(x) * x` instead of the Triton kernel.

2. **Model loading** uses `qwen3.load_model()` which calls
   `AutoModel.from_pretrained()` internally. This requires the
   `fish_speech.models.text2semantic.modeling` module to be importable
   (it registers `fish_qwen3_omni` with `AutoConfig`/`AutoModel`).

3. **Prompt construction** must use the `Conversation` + `Message` classes
   from `fish_speech.conversation`. The system message contains reference
   text prefixed with `<|speaker:0|>` and VQ codes from the reference audio.
   The user message contains the target text with the same prefix. The
   assistant message has `modality="voice"` which inserts the `<|voice|>`
   token.

4. **VQ embedding** is critical: the `vq_parts` and `vq_mask_tokens` from
   the encoded conversation must be passed to `qwen3.generate()`. The model
   uses these to embed VQ codes at the correct positions during prefill.

5. **`constrain_to_semantic=True`** restricts the model to only generate
   semantic tokens (IDs 151678–155773) and `<|im_end|>`, preventing text
   token generation during the audio output phase.

6. **RAS (Repetition Aware Sampling)** is built into `qwen3.decode_one_token`.
   If a sampled semantic token appears in a recent window, it re-samples
   with higher temperature. This replaces the explicit `repetition_penalty`
   used by S1.

7. **VQGAN encode/decode** uses `fish_speech.models.dac.vqgan` (not
   `fish_speech.models.dac.inference`). `batch_encode` accepts raw audio
   bytes and returns codebook tensors. `decode` accepts a list of
   `[num_codebooks, seq_len]` tensors and returns waveforms.

### Results (full 1088-sample seed-tts-eval English)

Parameters: `temperature=1.0, top_p=0.9, top_k=30, max_new_tokens=2048`

Pipeline: sglang-omni (fused 3-stage), ASR: Whisper-large-v3

| Metric | Value |
|--------|-------|
| **Overall WER** | **4.75%** |
| Total samples | 1088 |
| WER = 0% | 694 (63.8%) |
| WER ≤ 10% | 886 (81.4%) |
| WER > 30% | 16 (1.5%) |

Most non-zero WER samples are ASR normalization artifacts (punctuation,
capitalization, numeral-vs-word like "twenty" → "20").

### Fish Audio official numbers (for reference)

From the S1 README (using gpt-4o-transcribe, not Whisper):

| Model | WER | CER | Speaker Distance |
|-------|-----|-----|-----------------|
| S1 | 0.8% | 0.4% | 0.332 |
| S1-mini | 1.1% | 0.5% | 0.380 |

Note: Official numbers use gpt-4o-transcribe which handles punctuation and
numeral normalization better than Whisper-large-v3. Our 4.75% WER with
Whisper is expected to be higher due to these normalization differences.
