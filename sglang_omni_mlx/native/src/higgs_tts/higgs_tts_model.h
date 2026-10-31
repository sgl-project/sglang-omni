// SPDX-License-Identifier: Apache-2.0
// Higgs Audio v3 TTS: a Qwen3 decoder over the text prompt that speaks eight
// codec codebooks a step in a delay pattern, read through the fused audio
// embedding table, and the codec that turns the codes into samples.
#pragma once

#include <atomic>
#include <filesystem>
#include <functional>
#include <string>
#include <vector>

#include "common/layers.h"
#include "common/tokenizer.h"
#include "higgs_tts/higgs_tts_codec.h"
#include "mlx/mlx.h"
#include "qwen3_asr/qwen3_asr_transcriber.h"
#include "qwen3_asr/qwen3_decoder.h"

namespace higgs_tts {

struct SamplingOptions {
  double temperature = 0.0;
  // Off at 1 or more.
  double top_p = 1.0;
  // Off at 0 or below.
  int top_k = 0;
};

// Note (Jiaxin Deng): the CUDA server's speech defaults, measured there on
// SeedTTS EN, and its engine's row cap.
constexpr SamplingOptions kSpeechSampling = {0.8, 0.8, 30};
constexpr int kMaxRows = 2048;
// Bits the decoder is quantized to as it loads, unless a flag says otherwise.
constexpr int kDefaultQuantizeBits = 8;

class HiggsTtsModel {
public:
  // quantize_bits 8 quantizes the decoder's linear layers and text
  // embeddings as they load (group size 64, affine); 0 keeps bfloat16.
  // Note (Jiaxin Deng): 4 bits is not offered: at 4 bits a quarter of seeded
  // requests never reach the end of audio, in the reference as well.
  HiggsTtsModel(const std::filesystem::path &model_directory,
                int quantize_bits);

  const AudioCodec &codec() const { return codec_; }

  // A zero-shot speech prompt: <|tts|> <|text|> text <|audio|>.
  std::vector<int> EncodePrompt(const std::string &text) const;
  // Generates up to max_rows delayed rows for prompt ids, passing each row's
  // evaluated int32 codes [codebooks] to on_row. Stops once the first
  // codebook's end of audio has passed through every codebook; throws
  // TranscriptionCancelled once cancel is set.
  qwen3_asr::FinishReason
  Generate(const std::vector<int> &prompt_ids, const SamplingOptions &sampling,
           int max_rows, const std::atomic<bool> &cancel,
           const std::function<void(const mlx::core::array &)> &on_row) const;

private:
  struct LoadedWeights {
    omni_common::WeightMap model;
    omni_common::WeightMap codec;
  };

  HiggsTtsModel(const std::filesystem::path &model_directory,
                const nlohmann::json &config, int quantize_bits,
                LoadedWeights weights);
  static LoadedWeights LoadWeights(const std::filesystem::path &directory,
                                   const nlohmann::json &config,
                                   int quantize_bits);
  static nlohmann::json CheckpointConfig(int quantize_bits);

  // The final-normed hidden state [1, hidden] after the embeddings.
  mlx::core::array LastHidden(const mlx::core::array &embeddings,
                              std::vector<qwen3_asr::KVCache> &caches) const;
  // One row's codes [codebooks] drawn from the hidden state's audio logits.
  mlx::core::array SampleRow(const mlx::core::array &hidden,
                             const SamplingOptions &sampling) const;
  // The sum of a row's codebook embeddings, [1, 1, hidden].
  mlx::core::array EmbedRow(const mlx::core::array &codes) const;

  omni_common::Checkpoint checkpoint_;
  AudioCodec codec_;
  omni_common::Tokenizer tokenizer_;
  qwen3_asr::Qwen3Decoder decoder_;
  float norm_eps_;
  int tts_token_id_;
  int text_token_id_;
  int audio_token_id_;
  // Each codebook's first row in the fused audio embedding table.
  mlx::core::array codebook_offsets_;
};

} // namespace higgs_tts
