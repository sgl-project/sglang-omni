// SPDX-License-Identifier: Apache-2.0
#include "silero_vad.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <stdexcept>

#include "layers.h"
#include "nlohmann/json.hpp"

namespace silero_vad {

namespace mx = mlx::core;

namespace {

constexpr int kLstmHidden = 128;
constexpr const char *kBranchPrefix = "vad_16k.";

mx::array Relu(const mx::array &x) {
  return mx::maximum(x, mx::array(0.0f, x.dtype()));
}

} // namespace

SileroVAD::SileroVAD(const std::filesystem::path &model_directory) {
  std::ifstream config_stream(model_directory / "config.json");
  if (!config_stream) {
    throw std::runtime_error("cannot read config.json in " +
                             model_directory.string());
  } else {
  }
  const nlohmann::json config = nlohmann::json::parse(config_stream);
  const nlohmann::json &branch = config.at("branch_16k");
  if (branch.at("sample_rate").get<int>() != kSampleRate ||
      branch.at("chunk_size").get<int>() != kChunkSamples ||
      branch.at("context_size").get<int>() != kContextSamples) {
    throw std::runtime_error("unsupported Silero VAD branch configuration");
  } else {
  }
  filter_length_ = branch.at("filter_length").get<int>();
  hop_length_ = branch.at("hop_length").get<int>();
  pad_ = branch.at("pad").get<int>();
  cutoff_ = branch.at("cutoff").get<int>();
  defaults_ = {config.value("threshold", 0.5f),
               config.value("min_speech_duration_ms", 250),
               config.value("min_silence_duration_ms", 100),
               config.value("speech_pad_ms", 30)};
  for (const auto &entry :
       std::filesystem::directory_iterator(model_directory)) {
    if (entry.path().extension() != ".safetensors") {
      continue;
    } else {
    }
    auto [loaded, metadata] = mx::load_safetensors(entry.path().string());
    for (auto &[name, array] : loaded) {
      // Note (Jiaxin Deng): val_* are conversion reference values, not weights.
      if (name.rfind(kBranchPrefix, 0) == 0) {
        weights_.insert_or_assign(
            name.substr(std::string(kBranchPrefix).size()), array);
      } else {
      }
    }
  }
  for (const char *name :
       {"stft_conv.weight", "conv1.weight", "conv1.bias", "conv2.weight",
        "conv2.bias", "conv3.weight", "conv3.bias", "conv4.weight",
        "conv4.bias", "lstm.Wx", "lstm.Wh", "lstm.bias", "final_conv.weight",
        "final_conv.bias"}) {
    Weight(name);
  }
  std::vector<mx::array> parameters;
  for (const auto &[name, array] : weights_)
    parameters.push_back(array);
  mx::eval(parameters);
}

const mx::array &SileroVAD::Weight(const std::string &name) const {
  const auto found = weights_.find(name);
  if (found == weights_.end()) {
    throw std::runtime_error("Silero VAD checkpoint is missing " + name);
  } else {
  }
  return found->second;
}

mx::array SileroVAD::Conv1d(const mx::array &x, const std::string &name,
                            int stride, int padding) const {
  mx::array y = mx::conv1d(x, Weight(name + ".weight"), stride, padding);
  const auto bias = weights_.find(name + ".bias");
  if (bias != weights_.end()) {
    return mx::add(y, bias->second);
  } else {
    return y;
  }
}

float SileroVAD::Feed(const float *chunk, StreamState &state) const {
  std::vector<float> window(state.context);
  window.insert(window.end(), chunk, chunk + kChunkSamples);
  const int window_length = static_cast<int>(window.size());
  mx::array x(window.data(), {1, window_length}, mx::float32);
  // Note (Jiaxin Deng): right-only reflect pad as torch: x[n-2], x[n-3], ...
  std::vector<int32_t> reflected_indices;
  for (int i = window_length - 2; i > window_length - pad_ - 2; --i) {
    reflected_indices.push_back(i);
  }
  const mx::array indices(reflected_indices.data(),
                          {static_cast<int>(reflected_indices.size())},
                          mx::int32);
  x = mx::concatenate({x, mx::take(x, indices, -1)}, -1);
  x = mx::conv1d(mx::expand_dims(x, -1), Weight("stft_conv.weight"),
                 hop_length_, 0);
  const int frames = x.shape(1);
  const mx::array real = mx::slice(x, {0, 0, 0}, {1, frames, cutoff_});
  const mx::array imaginary =
      mx::slice(x, {0, 0, cutoff_}, {1, frames, cutoff_ * 2});
  x = mx::sqrt(
      mx::add(mx::multiply(real, real), mx::multiply(imaginary, imaginary)));
  x = Relu(Conv1d(x, "conv1", 1, 1));
  x = Relu(Conv1d(x, "conv2", 2, 1));
  x = Relu(Conv1d(x, "conv3", 2, 1));
  x = Relu(Conv1d(x, "conv4", 1, 1));

  // Note (Jiaxin Deng): MLXNN's LSTM formulation, to match Swift (one timestep
  // per 512-sample chunk).
  x = mx::addmm(Weight("lstm.bias"), x, mx::transpose(Weight("lstm.Wx")));
  std::optional<qwen3_asr::LstmCell> lstm =
      state.hidden.has_value()
          ? std::optional<qwen3_asr::LstmCell>({*state.hidden, *state.cell})
          : std::nullopt;
  std::vector<mx::array> hidden_steps;
  for (int step = 0; step < x.shape(1); ++step) {
    const mx::array gates =
        mx::reshape(mx::slice(x, {0, step, 0}, {1, step + 1, 4 * kLstmHidden}),
                    {1, 4 * kLstmHidden});
    lstm = qwen3_asr::LstmStep(gates, lstm, Weight("lstm.Wh"));
    hidden_steps.push_back(lstm->hidden);
  }
  mx::array out = Relu(mx::stack(hidden_steps, -2));
  out = mx::sigmoid(Conv1d(out, "final_conv", 1, 0));
  const mx::array probability = mx::mean(mx::squeeze(out, -1), 1, true);
  mx::eval({probability, lstm->hidden, lstm->cell});
  state.hidden = lstm->hidden;
  state.cell = lstm->cell;
  state.context.assign(chunk + kChunkSamples - kContextSamples,
                       chunk + kChunkSamples);
  return probability.item<float>();
}

std::vector<float>
SileroVAD::PredictProbabilities(const std::vector<float> &samples) const {
  std::vector<float> padded(samples);
  padded.resize((samples.size() + kChunkSamples - 1) / kChunkSamples *
                    kChunkSamples,
                0.0f);
  StreamState state;
  std::vector<float> probabilities;
  for (size_t offset = 0; offset < padded.size(); offset += kChunkSamples) {
    probabilities.push_back(Feed(padded.data() + offset, state));
  }
  return probabilities;
}

std::vector<Timestamp>
ProbabilitiesToTimestamps(const std::vector<float> &probabilities,
                          long sample_count, const TimestampOptions &options) {
  const float min_speech_samples = static_cast<float>(kSampleRate) *
                                   static_cast<float>(options.min_speech_ms) /
                                   1000;
  const float min_silence_samples = static_cast<float>(kSampleRate) *
                                    static_cast<float>(options.min_silence_ms) /
                                    1000;
  const long pad_samples =
      static_cast<long>(static_cast<float>(kSampleRate) *
                        static_cast<float>(options.speech_pad_ms) / 1000);
  const float negative_threshold = std::max(options.threshold - 0.15f, 0.01f);
  std::vector<Timestamp> speeches;
  bool triggered = false;
  long current_start = 0;
  long temp_end = 0;
  for (size_t index = 0; index < probabilities.size(); ++index) {
    const float probability = probabilities[index];
    const long chunk_start = static_cast<long>(index) * kChunkSamples;
    if (probability >= options.threshold && !triggered) {
      triggered = true;
      current_start = chunk_start;
      temp_end = 0;
    } else if (triggered && probability >= options.threshold) {
      temp_end = 0;
    } else if (triggered && probability < negative_threshold) {
      if (temp_end == 0)
        temp_end = chunk_start;
      if (static_cast<float>(chunk_start - temp_end) >= min_silence_samples) {
        if (static_cast<float>(temp_end - current_start) >=
            min_speech_samples) {
          speeches.push_back({current_start, temp_end});
        } else {
        }
        triggered = false;
        temp_end = 0;
      } else {
      }
    } else {
    }
  }
  if (triggered) {
    const long end = std::min(
        sample_count, static_cast<long>(probabilities.size()) * kChunkSamples);
    if (static_cast<float>(end - current_start) >= min_speech_samples) {
      speeches.push_back({current_start, end});
    } else {
    }
  } else {
  }
  std::vector<Timestamp> padded;
  for (const Timestamp &speech : speeches) {
    const long start = std::max(0L, speech.start - pad_samples);
    const long end = std::min(sample_count, speech.end + pad_samples);
    if (!padded.empty() && start <= padded.back().end) {
      padded.back().end = std::max(padded.back().end, end);
    } else {
      padded.push_back({start, end});
    }
  }
  return padded;
}

} // namespace silero_vad
