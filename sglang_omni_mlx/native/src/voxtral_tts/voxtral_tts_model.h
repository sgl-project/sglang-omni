// SPDX-License-Identifier: Apache-2.0
// Voxtral TTS: a Mistral decoder over the prompt and each generated frame, a
// flow-matching acoustic head that predicts the frame's codes, and the codec
// that turns frames into samples.
#pragma once

#include <atomic>
#include <filesystem>
#include <functional>
#include <map>
#include <string>
#include <vector>

#include "common/layers.h"
#include "mlx/mlx.h"
#include "qwen3_asr/qwen3_asr_transcriber.h"
#include "qwen3_asr/qwen3_decoder.h"
#include "voxtral_tts/voxtral_tts_codec.h"
#include "voxtral_tts/voxtral_tts_tokenizer.h"

namespace voxtral_tts {

class VoxtralTtsModel {
public:
  explicit VoxtralTtsModel(const std::filesystem::path &model_directory);

  const SpeechTokenizer &tokenizer() const { return tokenizer_; }
  const AudioCodec &codec() const { return codec_; }

  // Generates up to max_frames frames for prompt ids from
  // SpeechTokenizer::EncodeSpeechRequest in voice, passing each frame's
  // evaluated int32 codes [1 + acoustic codebooks] to on_frame. Stops at the
  // end-of-audio code; throws TranscriptionCancelled once cancel is set.
  qwen3_asr::FinishReason
  Generate(const std::vector<int> &prompt_ids, const std::string &voice,
           int max_frames, const std::atomic<bool> &cancel,
           const std::function<void(const mlx::core::array &)> &on_frame) const;

private:
  struct LoadedWeights {
    qwen3_asr::WeightMap model;
    qwen3_asr::WeightMap codec;
  };

  VoxtralTtsModel(const std::filesystem::path &model_directory,
                  const nlohmann::json &config, LoadedWeights weights);
  static LoadedWeights LoadWeights(const std::filesystem::path &directory);

  // The final-normed hidden state [1, dim] after the embeddings.
  mlx::core::array LastHidden(const mlx::core::array &embeddings,
                              std::vector<qwen3_asr::KVCache> &caches) const;
  // A frame's codes [1 + acoustic codebooks]: the semantic code, then the
  // acoustic codes the flow reaches from noise.
  mlx::core::array PredictFrame(const mlx::core::array &hidden) const;
  mlx::core::array Velocity(const mlx::core::array &latents,
                            const mlx::core::array &time_features,
                            const mlx::core::array &language_features) const;
  mlx::core::array AcousticLayer(const mlx::core::array &x, int layer) const;
  // The sum of a frame's codebook embeddings, [1, 1, dim].
  mlx::core::array EmbedFrame(const mlx::core::array &codes) const;

  qwen3_asr::Checkpoint checkpoint_;
  AudioCodec codec_;
  SpeechTokenizer tokenizer_;
  qwen3_asr::Qwen3Decoder decoder_;
  float norm_eps_;
  int semantic_code_count_;
  int acoustic_codebook_count_;
  int acoustic_levels_;
  int acoustic_layer_count_;
  int acoustic_head_count_;
  int acoustic_key_value_head_count_;
  int acoustic_head_dim_;
  float acoustic_norm_eps_;
  // Each codebook's first row in the audio codebook embeddings.
  mlx::core::array codebook_offsets_;
  // Sinusoid frequencies of the flow's time embedding.
  mlx::core::array time_frequencies_;
  std::map<std::string, mlx::core::array> voices_;
};

} // namespace voxtral_tts
