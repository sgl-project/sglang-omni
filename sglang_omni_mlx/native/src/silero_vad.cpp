// SPDX-License-Identifier: Apache-2.0
#include "silero_vad.h"

#include <Accelerate/Accelerate.h>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <optional>
#include <stdexcept>

#include "nlohmann/json.hpp"

namespace silero_vad {

namespace mx = mlx::core;

namespace {

constexpr const char *kBranchPrefix = "vad_16k.";
// Note (Jiaxin Deng): bounds the front end's activations on long audio; the
// next block's graph runs on the GPU while the CPU runs this block's LSTM.
constexpr size_t kBlockChunks = 2048;

mx::array Relu(const mx::array &x) {
  return mx::maximum(x, mx::array(0.0f, x.dtype()));
}

float Sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }

std::vector<float> HostCopy(const mx::array &array) {
  const mx::array host = mx::flatten(mx::astype(array, mx::float32));
  mx::eval(host);
  return {host.data<float>(), host.data<float>() + host.size()};
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
  recurrent_weight_ = HostCopy(Weight("lstm.Wh"));
  output_weight_ = HostCopy(Weight("final_conv.weight"));
  output_bias_ = HostCopy(Weight("final_conv.bias")).at(0);
  if (recurrent_weight_.size() != 4 * kLstmHidden * kLstmHidden ||
      output_weight_.size() != kLstmHidden) {
    throw std::runtime_error("unsupported Silero VAD LSTM size");
  } else {
  }
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

mx::array SileroVAD::InputGates(const std::vector<float> &windows,
                                int chunk_count) const {
  const int window_length = static_cast<int>(windows.size()) / chunk_count;
  mx::array x(windows.data(), {chunk_count, window_length, 1}, mx::float32);
  x = mx::conv1d(x, Weight("stft_conv.weight"), hop_length_, 0);
  const int frames = x.shape(1);
  const mx::array real =
      mx::slice(x, {0, 0, 0}, {chunk_count, frames, cutoff_});
  const mx::array imaginary =
      mx::slice(x, {0, 0, cutoff_}, {chunk_count, frames, cutoff_ * 2});
  x = mx::sqrt(
      mx::add(mx::multiply(real, real), mx::multiply(imaginary, imaginary)));
  x = Relu(Conv1d(x, "conv1", 1, 1));
  x = Relu(Conv1d(x, "conv2", 2, 1));
  x = Relu(Conv1d(x, "conv3", 2, 1));
  x = Relu(Conv1d(x, "conv4", 1, 1));
  return mx::addmm(Weight("lstm.bias"), x, mx::transpose(Weight("lstm.Wx")));
}

std::vector<float> SileroVAD::FeedChunks(const float *samples,
                                         size_t chunk_count,
                                         StreamState &state) const {
  constexpr int kWindowSamples = kContextSamples + kChunkSamples;
  constexpr int kGateCount = 4 * kLstmHidden;
  const int padded_length = kWindowSamples + pad_;
  // Note (Jiaxin Deng): every window is known up front (its context is the
  // previous chunk's samples), so only the LSTM is sequential; it runs on the
  // CPU because a GPU dispatch per 512-sample step costs more than the step.
  const auto schedule = [&](size_t first, size_t count) {
    std::vector<float> windows(count * static_cast<size_t>(padded_length));
    for (size_t chunk = 0; chunk < count; ++chunk) {
      float *window = windows.data() + chunk * padded_length;
      const float *chunk_samples = samples + (first + chunk) * kChunkSamples;
      if (first + chunk == 0) {
        std::copy(state.context.begin(), state.context.end(), window);
      } else {
        std::copy(chunk_samples - kContextSamples, chunk_samples, window);
      }
      std::copy(chunk_samples, chunk_samples + kChunkSamples,
                window + kContextSamples);
      // Note (Jiaxin Deng): right-only reflect pad as torch: x[n-2], x[n-3].
      for (int i = 0; i < pad_; ++i) {
        window[kWindowSamples + i] = window[kWindowSamples - 2 - i];
      }
    }
    mx::array gates = InputGates(windows, static_cast<int>(count));
    mx::async_eval(gates);
    return gates;
  };

  std::vector<float> probabilities;
  probabilities.reserve(chunk_count);
  float *hidden = state.hidden.data();
  float *cell = state.cell.data();
  std::vector<float> step_gates(kGateCount);
  std::optional<mx::array> pending =
      chunk_count > 0 ? std::optional<mx::array>(
                            schedule(0, std::min(kBlockChunks, chunk_count)))
                      : std::nullopt;
  for (size_t first = 0; first < chunk_count; first += kBlockChunks) {
    const mx::array gates = *pending;
    const size_t next = first + kBlockChunks;
    pending = next < chunk_count
                  ? std::optional<mx::array>(schedule(
                        next, std::min(kBlockChunks, chunk_count - next)))
                  : std::nullopt;
    mx::eval(gates);
    const int steps = gates.shape(1);
    const float *chunk_gates = gates.data<float>();
    for (int chunk = 0; chunk < gates.shape(0); ++chunk) {
      float speech = 0.0f;
      for (int step = 0; step < steps; ++step) {
        std::copy(chunk_gates, chunk_gates + kGateCount, step_gates.begin());
        chunk_gates += kGateCount;
        cblas_sgemv(CblasRowMajor, CblasNoTrans, kGateCount, kLstmHidden, 1.0f,
                    recurrent_weight_.data(), kLstmHidden, hidden, 1, 1.0f,
                    step_gates.data(), 1);
        float output = output_bias_;
        for (int unit = 0; unit < kLstmHidden; ++unit) {
          const float input_gate = Sigmoid(step_gates[unit]);
          const float forget_gate = Sigmoid(step_gates[kLstmHidden + unit]);
          const float candidate = std::tanh(step_gates[2 * kLstmHidden + unit]);
          const float output_gate = Sigmoid(step_gates[3 * kLstmHidden + unit]);
          cell[unit] = forget_gate * cell[unit] + input_gate * candidate;
          hidden[unit] = output_gate * std::tanh(cell[unit]);
          output += std::max(hidden[unit], 0.0f) * output_weight_[unit];
        }
        speech += Sigmoid(output);
      }
      probabilities.push_back(speech / static_cast<float>(steps));
    }
  }
  if (chunk_count > 0) {
    const float *last = samples + chunk_count * kChunkSamples;
    state.context.assign(last - kContextSamples, last);
  } else {
  }
  return probabilities;
}

float SileroVAD::Feed(const float *chunk, StreamState &state) const {
  return FeedChunks(chunk, 1, state).front();
}

std::vector<float>
SileroVAD::PredictProbabilities(const std::vector<float> &samples) const {
  std::vector<float> padded(samples);
  padded.resize((samples.size() + kChunkSamples - 1) / kChunkSamples *
                    kChunkSamples,
                0.0f);
  StreamState state;
  return FeedChunks(padded.data(), padded.size() / kChunkSamples, state);
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
