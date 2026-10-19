// SPDX-License-Identifier: Apache-2.0
// Silero VAD v6 on MLX, as Voxt's Swift MLXAudioVAD computes it.
#pragma once

#include <filesystem>
#include <string>
#include <unordered_map>
#include <vector>

#include "mlx/mlx.h"

namespace silero_vad {

inline constexpr int kSampleRate = 16000;
inline constexpr int kChunkSamples = 512;
inline constexpr int kContextSamples = 64;
inline constexpr int kLstmHidden = 128;

// One stream's state between 512-sample chunks.
struct StreamState {
  // LSTM (hidden, cell); zeros before the first chunk, which is exactly the
  // Swift port's stateless first step.
  std::vector<float> hidden = std::vector<float>(kLstmHidden, 0.0f);
  std::vector<float> cell = std::vector<float>(kLstmHidden, 0.0f);
  // The previous chunk's last 64 samples (zeros at the start).
  std::vector<float> context = std::vector<float>(kContextSamples, 0.0f);
};

struct Timestamp {
  long start = 0;
  long end = 0;
};

struct TimestampOptions {
  float threshold = 0.5f;
  int min_speech_ms = 250;
  int min_silence_ms = 100;
  int speech_pad_ms = 30;
};

class SileroVAD {
public:
  // Loads config.json and model.safetensors (16 kHz branch only).
  explicit SileroVAD(const std::filesystem::path &model_directory);

  // Speech probability of one 512-sample chunk; advances state.
  float Feed(const float *chunk, StreamState &state) const;
  // Speech probabilities of chunk_count consecutive 512-sample chunks, in
  // order; advances state as chunk_count Feed calls would.
  std::vector<float> FeedChunks(const float *samples, size_t chunk_count,
                                StreamState &state) const;
  // Per-chunk probabilities of a whole buffer: zero-padded to whole chunks,
  // fed in order from a fresh state (Swift predictProba).
  std::vector<float>
  PredictProbabilities(const std::vector<float> &samples) const;

  const TimestampOptions &defaults() const { return defaults_; }

private:
  // LSTM input gates (chunk_count, steps, 4 * hidden) of padded windows
  // (chunk_count, window_samples): everything before the recurrence.
  mlx::core::array InputGates(const std::vector<float> &windows,
                              int chunk_count) const;
  mlx::core::array Conv1d(const mlx::core::array &x, const std::string &name,
                          int stride, int padding) const;
  const mlx::core::array &Weight(const std::string &name) const;

  std::unordered_map<std::string, mlx::core::array> weights_;
  // Host copies for the recurrence: Wh (4 * hidden, hidden) row-major, the
  // final 1x1 convolution's weight (hidden) and bias.
  std::vector<float> recurrent_weight_;
  std::vector<float> output_weight_;
  float output_bias_ = 0.0f;
  int filter_length_ = 256;
  int hop_length_ = 128;
  int pad_ = 64;
  int cutoff_ = 129;
  TimestampOptions defaults_;
};

// Speech segments from per-chunk probabilities, as Swift probsToTimestamps.
std::vector<Timestamp>
ProbabilitiesToTimestamps(const std::vector<float> &probabilities,
                          long sample_count, const TimestampOptions &options);

} // namespace silero_vad
