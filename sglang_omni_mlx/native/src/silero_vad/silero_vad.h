// SPDX-License-Identifier: Apache-2.0
// Silero VAD v6 on MLX, as Voxt's Swift MLXAudioVAD computes it.
#pragma once

#include <filesystem>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "mlx/mlx.h"

namespace silero_vad {

inline constexpr int kSampleRate = 16000;
inline constexpr int kChunkSamples = 512;
inline constexpr int kContextSamples = 64;

// One stream's state between 512-sample chunks.
struct StreamState {
  // LSTM (hidden, cell); none before the first chunk, as in the Swift port.
  std::optional<mlx::core::array> hidden;
  std::optional<mlx::core::array> cell;
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
  // Per-chunk probabilities of a whole buffer: zero-padded to whole chunks,
  // fed in order from a fresh state (Swift predictProba).
  std::vector<float>
  PredictProbabilities(const std::vector<float> &samples) const;

  const TimestampOptions &defaults() const { return defaults_; }

private:
  mlx::core::array Conv1d(const mlx::core::array &x, const std::string &name,
                          int stride, int padding) const;
  const mlx::core::array &Weight(const std::string &name) const;

  std::unordered_map<std::string, mlx::core::array> weights_;
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
