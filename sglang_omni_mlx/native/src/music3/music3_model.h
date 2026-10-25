// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <atomic>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

#include "common/layers.h"
#include "common/tokenizer.h"
#include "qwen3_asr/qwen3_decoder.h"

namespace music3 {

inline constexpr int kSampleRate = 32000;
inline constexpr int kMaximumFrames = 9000;
inline constexpr int kMaximumPromptTokens = 5000;
inline constexpr int kChunkFrames = 200;
inline constexpr int kChunkHopFrames = 100;
inline constexpr int kLatentHopSamples = 512;
inline constexpr int kOverlapLatents = 172;

struct Request {
  std::vector<int> token_ids;
  uint64_t seed = 0;
  int max_frames = 750;
  int dit_steps = 30;
  float dit_guidance = 1.7f;
  bool pcm = false;
};

struct Result {
  std::vector<float> samples;
  int frames;
};

class Cancelled : public std::runtime_error {
public:
  Cancelled() : std::runtime_error("Music3 generation cancelled") {}
};

void CheckCancelled(const std::atomic<bool> &cancelled);
std::string BuildPrompt(const std::string &caption, const std::string &lyrics);
// An empty model_name accepts any model field, as the server does when no
// --model-name was chosen and clients cannot know the generated one.
Request ParseRequest(const nlohmann::json &body,
                     const omni_common::Tokenizer &tokenizer,
                     const std::string &model_name, int context_tokens);
std::string EncodeAudio(const std::vector<float> &samples, bool pcm);
uint64_t AcousticSeed(uint64_t seed, int chunk_index);
void ValidateSerialOffload(const std::string &offload_components);
std::vector<float> ResampleStereo(const mlx::core::array &waveform);

class Model {
public:
  explicit Model(const std::filesystem::path &directory,
                 bool is_serial_offload_enabled = false);
  Result Synthesize(const Request &request, const std::atomic<bool> &cancelled,
                    const std::function<void()> &on_ar_complete = {});
  const omni_common::Tokenizer &tokenizer() const { return tokenizer_; }
  int context_tokens() const {
    return config_.at("max_position_embeddings").get<int>();
  }

private:
  void ReleaseWeights();
  void LoadStageWeights(bool is_acoustic_stage);
  Result DecodeFrames(const mlx::core::array &frames, const Request &request,
                      const std::atomic<bool> &cancelled) const;
  mlx::core::array GenerateFrames(const Request &request,
                                  const std::atomic<bool> &cancelled) const;
  mlx::core::array Depth(const mlx::core::array &embeddings) const;
  mlx::core::array Condition(const mlx::core::array &hidden_states) const;
  mlx::core::array Velocity(const mlx::core::array &latents, float sigma,
                            const mlx::core::array &condition) const;
  mlx::core::array Vocoder(const mlx::core::array &latents) const;
  mlx::core::array Convolve(const mlx::core::array &hidden_states,
                            const std::string &prefix, int padding = 0,
                            int dilation = 1, int stride = 1,
                            bool transpose = false) const;
  mlx::core::array Snake(const mlx::core::array &hidden_states,
                         const std::string &prefix) const;

  std::filesystem::path checkpoint_directory_;
  bool is_serial_offload_enabled_;
  nlohmann::json config_;
  omni_common::Checkpoint checkpoint_;
  qwen3_asr::Qwen3Decoder decoder_;
  omni_common::Tokenizer tokenizer_;
};

} // namespace music3
