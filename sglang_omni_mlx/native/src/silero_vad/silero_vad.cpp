// SPDX-License-Identifier: Apache-2.0
#include "silero_vad/silero_vad.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <stdexcept>

#include "common/layers.h"
#include "nlohmann/json.hpp"

namespace silero_vad {

namespace mx = mlx::core;

namespace {

constexpr int kLstmHidden = 128;
constexpr const char *kBranchPrefix = "vad_16k.";

mx::array Relu(const mx::array &x) {
  return mx::maximum(x, mx::array(0.0f, x.dtype()));
}

// The 16 kHz branch's weights, without its prefix.
omni_common::WeightMap
LoadBranchWeights(const std::filesystem::path &model_directory) {
  omni_common::WeightMap weights;
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
        weights.insert_or_assign(name.substr(std::string(kBranchPrefix).size()),
                                 array);
      } else {
      }
    }
  }
  return weights;
}

} // namespace

SileroVAD::SileroVAD(const std::filesystem::path &model_directory)
    : checkpoint_(LoadBranchWeights(model_directory),
                  nlohmann::json::object()) {
  const nlohmann::json config =
      omni_common::ReadJson(model_directory / "config.json");
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
  for (const char *name :
       {"stft_conv.weight", "conv1.weight", "conv1.bias", "conv2.weight",
        "conv2.bias", "conv3.weight", "conv3.bias", "conv4.weight",
        "conv4.bias", "lstm.Wx", "lstm.Wh", "lstm.bias", "final_conv.weight",
        "final_conv.bias"}) {
    checkpoint_.Weight(name);
  }
  // Prime every Metal kernel the feed uses with a throwaway stream: a
  // fresh process otherwise compiles them on the first real feed, which
  // lands on the user's first dictation. The warmup state and its
  // probabilities are discarded, so later streams are unaffected.
  StreamState warmup;
  const std::vector<float> quiet(8 * kChunkSamples, 0.0f);
  Feed(quiet.data(), 8, warmup);
  warmed_up_ = true;
}

std::vector<float> SileroVAD::Feed(const float *samples, int chunk_count,
                                   StreamState &state) const {
  if (chunk_count == 0) {
    return {};
  } else {
  }
  // Note (XXXXRT666): every chunk's window goes through the front end at once;
  // only the LSTM runs chunk by chunk, and the probabilities are read back
  // together.
  constexpr int window_length = kContextSamples + kChunkSamples;
  std::vector<float> windows(static_cast<size_t>(chunk_count) * window_length);
  for (int chunk = 0; chunk < chunk_count; ++chunk) {
    const float *context =
        chunk == 0 ? state.context.data()
                   : samples + chunk * kChunkSamples - kContextSamples;
    float *window = windows.data() + static_cast<size_t>(chunk) * window_length;
    std::copy(context, context + kContextSamples, window);
    std::copy(samples + chunk * kChunkSamples,
              samples + (chunk + 1) * kChunkSamples, window + kContextSamples);
  }
  mx::array x(windows.data(), {chunk_count, window_length}, mx::float32);
  // Note (Jiaxin Deng): right-only reflect pad as torch: x[n-2], x[n-3], ...
  std::vector<int32_t> reflected_indices;
  for (int i = window_length - 2; i > window_length - pad_ - 2; --i) {
    reflected_indices.push_back(i);
  }
  const mx::array indices(reflected_indices.data(),
                          {static_cast<int>(reflected_indices.size())},
                          mx::int32);
  x = mx::concatenate({x, mx::take(x, indices, -1)}, -1);
  x = mx::conv1d(mx::expand_dims(x, -1), checkpoint_.Weight("stft_conv.weight"),
                 hop_length_, 0);
  const int frames = x.shape(1);
  const mx::array real =
      mx::slice(x, {0, 0, 0}, {chunk_count, frames, cutoff_});
  const mx::array imaginary =
      mx::slice(x, {0, 0, cutoff_}, {chunk_count, frames, cutoff_ * 2});
  x = mx::sqrt(
      mx::add(mx::multiply(real, real), mx::multiply(imaginary, imaginary)));
  x = Relu(checkpoint_.Conv1d(x, "conv1", 1, 1));
  x = Relu(checkpoint_.Conv1d(x, "conv2", 2, 1));
  x = Relu(checkpoint_.Conv1d(x, "conv3", 2, 1));
  x = Relu(checkpoint_.Conv1d(x, "conv4", 1, 1));
  const int steps_per_chunk = x.shape(1);
  const int step_count = chunk_count * steps_per_chunk;

  // Note (Jiaxin Deng): MLXNN's LSTM formulation, to match Swift.
  const mx::array gates = mx::addmm(
      checkpoint_.Weight("lstm.bias"), mx::reshape(x, {step_count, x.shape(2)}),
      mx::transpose(checkpoint_.Weight("lstm.Wx")));
  std::optional<omni_common::LstmCell> lstm =
      state.hidden.has_value()
          ? std::optional<omni_common::LstmCell>({*state.hidden, *state.cell})
          : std::nullopt;
  std::vector<mx::array> hidden_steps;
  for (int step = 0; step < step_count; ++step) {
    lstm = omni_common::LstmStep(
        mx::slice(gates, {step, 0}, {step + 1, 4 * kLstmHidden}), lstm,
        checkpoint_.Weight("lstm.Wh"));
    hidden_steps.push_back(lstm->hidden);
  }
  mx::array out = mx::reshape(Relu(mx::concatenate(hidden_steps, 0)),
                              {chunk_count, steps_per_chunk, kLstmHidden});
  out = mx::sigmoid(checkpoint_.Conv1d(out, "final_conv", 1, 0));
  const mx::array probabilities = mx::mean(mx::squeeze(out, -1), 1);
  mx::eval({probabilities, lstm->hidden, lstm->cell});
  state.hidden = lstm->hidden;
  state.cell = lstm->cell;
  const float *last_chunk_end = samples + chunk_count * kChunkSamples;
  state.context.assign(last_chunk_end - kContextSamples, last_chunk_end);
  return {probabilities.data<float>(),
          probabilities.data<float>() + chunk_count};
}

std::vector<float>
SileroVAD::PredictProbabilities(const std::vector<float> &samples) const {
  std::vector<float> padded(samples);
  padded.resize((samples.size() + kChunkSamples - 1) / kChunkSamples *
                    kChunkSamples,
                0.0f);
  StreamState state;
  return Feed(padded.data(), static_cast<int>(padded.size() / kChunkSamples),
              state);
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

const SileroVAD &SileroVADCache::Load(const std::string &model_directory) {
  if (model_ == nullptr || model_directory_ != model_directory) {
    model_.reset();
    model_ = std::make_unique<SileroVAD>(model_directory);
    model_directory_ = model_directory;
  } else {
  }
  return *model_;
}

} // namespace silero_vad
