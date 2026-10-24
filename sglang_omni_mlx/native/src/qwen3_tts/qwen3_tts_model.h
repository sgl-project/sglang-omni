// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>
#include <functional>
#include <map>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include "common/layers.h"
#include "common/qwen3_decoder.h"
#include "common/tokenizer.h"
#include "qwen3_tts/qwen3_tts_codec.h"

namespace qwen3_tts {

inline constexpr int kSampleRate = 24000;
inline constexpr int kCodeGroups = 16;
inline constexpr int kCodebookSize = 2048;
inline constexpr int kSamplesPerFrame = 1920;
inline constexpr int kMaximumFrames = 2048;
inline constexpr int kRepetitionContextFrames = 64;
inline constexpr int kCodecChunkFrames = 8;
inline constexpr int kMaximumInputBytes = 16384;
inline constexpr int kMaximumReferenceSamples = 30 * kSampleRate;

struct Sampling {
  float temperature = 0.9f;
  int top_k = 50;
  float top_p = 1.0f;
  float repetition_penalty = 1.0f;
};

struct Request {
  std::vector<int> text_ids;
  std::vector<int> instruction_ids;
  std::vector<int> reference_text_ids;
  std::vector<float> reference_samples;
  std::vector<int> control_ids;
  Sampling semantic;
  Sampling subtalker;
  uint64_t seed = 0;
  int max_frames = 256;
  bool pcm = false;
  bool stream = false;
  bool non_streaming_mode = false;
};

struct Result {
  std::vector<float> samples;
  std::vector<std::vector<int>> codes;
  std::string finish_reason;
};

std::string EncodeAudio(const std::vector<float> &samples, bool pcm);
std::vector<float> DecodeReferenceWav(const std::string &bytes);

using WeightShapes = std::map<std::string, mlx::core::Shape>;
void LinearShapes(WeightShapes &shapes, const std::string &prefix, int input,
                  int output, bool bias = false);
void TransformerShapes(WeightShapes &shapes, const std::string &prefix,
                       const nlohmann::json &config);
void SpeakerShapes(WeightShapes &shapes, const nlohmann::json &config);
void ValidateWeights(const qwen3_asr::WeightMap &weights,
                     const WeightShapes &shapes);
mlx::core::array TimeSlice(const mlx::core::array &values, int start, int stop);

class Model {
public:
  explicit Model(const std::filesystem::path &directory);
  Request ParseRequest(const nlohmann::json &body,
                       const std::string &model_name) const;
  Result Synthesize(
      const Request &request, const std::function<void()> &check_cancelled,
      const std::function<void(const std::vector<float> &)> &emit = {}) const;

private:
  struct Prompt {
    mlx::core::array embeddings;
    mlx::core::array trailing;
    mlx::core::array pad;
    std::optional<mlx::core::array> reference_codes;
  };
  mlx::core::array Text(const std::vector<int> &ids) const;
  mlx::core::array CodecEmbedding(const std::vector<int> &ids) const;
  Prompt BuildPrompt(const Request &request,
                     const std::function<void()> &check_cancelled) const;
  mlx::core::array SpeakerEmbedding(const std::vector<float> &samples) const;
  mlx::core::array Sample(mlx::core::array logits, const Sampling &sampling,
                          mlx::core::random::KeySequence &random_sequence,
                          const std::vector<int> &history, bool semantic) const;

  nlohmann::json config_;
  nlohmann::json talker_config_;
  qwen3_asr::Checkpoint checkpoint_;
  qwen3_asr::Qwen3Decoder talker_;
  qwen3_asr::Qwen3Decoder predictor_;
  qwen3_asr::Tokenizer tokenizer_;
  AudioCodec codec_;
};

} // namespace qwen3_tts
