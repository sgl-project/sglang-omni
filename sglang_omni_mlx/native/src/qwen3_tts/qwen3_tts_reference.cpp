// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Prince Canuma and contributors.
#include "qwen3_tts/qwen3_tts_model.h"

#include <algorithm>
#include <cmath>
#include <numeric>

namespace qwen3_tts {
namespace mx = mlx::core;

namespace {

constexpr int kFftSamples = 1024;
constexpr int kHopSamples = 256;
constexpr int kMelBins = 128;
constexpr int kSpeakerChannels = 512;
constexpr int kRes2NetScale = 8;
constexpr int kAggregatedChannels = 1536;

void SpeakerConvolutionShapes(WeightShapes &shapes, const std::string &prefix,
                              int input, int output, int kernel) {
  shapes[prefix + ".weight"] = {output, kernel, input};
  shapes[prefix + ".bias"] = {output};
}

mx::array SpeakerMels(const std::vector<float> &samples) {
  const auto hertz_to_mel = [](double hertz) {
    return hertz < 1000.0
               ? hertz / (200.0 / 3.0)
               : 15.0 + std::log(hertz / 1000.0) / (std::log(6.4) / 27.0);
  };
  std::vector<double> edges(kMelBins + 2);
  const double maximum = hertz_to_mel(kSampleRate / 2.0);
  for (int index = 0; index < kMelBins + 2; ++index) {
    const double mel = maximum * index / (kMelBins + 1);
    edges[index] =
        mel < 15.0 ? (200.0 / 3.0) * mel
                   : 1000.0 * std::exp((std::log(6.4) / 27.0) * (mel - 15.0));
  }
  std::vector<float> filters((kFftSamples / 2 + 1) * kMelBins);
  for (int frequency = 0; frequency <= kFftSamples / 2; ++frequency) {
    const double hertz =
        static_cast<double>(frequency) * kSampleRate / kFftSamples;
    for (int mel = 0; mel < kMelBins; ++mel) {
      filters[frequency * kMelBins + mel] = static_cast<float>(
          std::max(0.0, std::min((hertz - edges[mel]) /
                                     (edges[mel + 1] - edges[mel]),
                                 (edges[mel + 2] - hertz) /
                                     (edges[mel + 2] - edges[mel + 1]))) *
          2.0 / (edges[mel + 2] - edges[mel]));
    }
  }
  std::vector<float> window(kFftSamples);
  for (int index = 0; index < kFftSamples; ++index) {
    window[index] = static_cast<float>(
        0.5 * (1.0 - std::cos(2.0 * M_PI * index / (kFftSamples - 1))));
  }
  const auto audio = mx::array(samples.data(),
                               {static_cast<int>(samples.size())}, mx::float32);
  const int padding = (kFftSamples - kHopSamples) / 2;
  const auto padded = mx::pad(audio, std::vector<int>{0}, mx::Shape{padding},
                              mx::Shape{padding}, mx::array(0), "reflect");
  const int frames = 1 + (padded.shape(0) - kFftSamples) / kHopSamples;
  const auto framed =
      mx::as_strided(padded, {frames, kFftSamples}, {kHopSamples, 1}, 0);
  const auto spectrum = mx::abs(mx::fft::rfft(
      framed * mx::array(window.data(), {kFftSamples}, mx::float32), 1));
  const auto magnitude = mx::sqrt(spectrum * spectrum + mx::array(1e-9f));
  const auto mels = mx::matmul(
      magnitude,
      mx::array(filters.data(), {kFftSamples / 2 + 1, kMelBins}, mx::float32));
  return mx::expand_dims(mx::log(mx::maximum(mels, mx::array(1e-5f))), 0);
}

} // namespace

void SpeakerShapes(WeightShapes &shapes, const nlohmann::json &configuration) {
  const auto &config = configuration.at("speaker_encoder_config");
  if (config.value("sample_rate", kSampleRate) != kSampleRate ||
      config.at("enc_dim") !=
          configuration.at("talker_config").at("hidden_size") ||
      config.value("mel_dim", kMelBins) != kMelBins ||
      config.value("enc_channels",
                   std::vector<int>{512, 512, 512, 512, 1536}) !=
          std::vector<int>{512, 512, 512, 512, 1536} ||
      config.value("enc_res2net_scale", kRes2NetScale) != kRes2NetScale ||
      config.value("enc_kernel_sizes", std::vector<int>{5, 3, 3, 3, 1}) !=
          std::vector<int>{5, 3, 3, 3, 1} ||
      config.value("enc_dilations", std::vector<int>{1, 2, 3, 4, 1}) !=
          std::vector<int>{1, 2, 3, 4, 1} ||
      config.value("enc_attention_channels", 128) != 128 ||
      config.value("enc_se_channels", 128) != 128) {
    throw std::invalid_argument(
        "unsupported Qwen3-TTS speaker encoder configuration");
  } else {
  }
  SpeakerConvolutionShapes(shapes, "speaker_encoder.blocks.0.conv", kMelBins,
                           kSpeakerChannels, 5);
  for (int block = 1; block < 4; ++block) {
    const auto stem = "speaker_encoder.blocks." + std::to_string(block);
    for (const auto &name : {"tdnn1", "tdnn2"}) {
      SpeakerConvolutionShapes(shapes, stem + "." + name + ".conv",
                               kSpeakerChannels, kSpeakerChannels, 1);
    }
    for (int group = 0; group < kRes2NetScale - 1; ++group) {
      SpeakerConvolutionShapes(shapes,
                               stem + ".res2net_block.blocks." +
                                   std::to_string(group) + ".conv",
                               kSpeakerChannels / kRes2NetScale,
                               kSpeakerChannels / kRes2NetScale, 3);
    }
    SpeakerConvolutionShapes(shapes, stem + ".se_block.conv1", kSpeakerChannels,
                             128, 1);
    SpeakerConvolutionShapes(shapes, stem + ".se_block.conv2", 128,
                             kSpeakerChannels, 1);
  }
  SpeakerConvolutionShapes(shapes, "speaker_encoder.mfa.conv",
                           kAggregatedChannels, kAggregatedChannels, 1);
  SpeakerConvolutionShapes(shapes, "speaker_encoder.asp.tdnn.conv",
                           kAggregatedChannels * 3, 128, 1);
  SpeakerConvolutionShapes(shapes, "speaker_encoder.asp.conv", 128,
                           kAggregatedChannels, 1);
  SpeakerConvolutionShapes(shapes, "speaker_encoder.fc",
                           kAggregatedChannels * 2, config.at("enc_dim"), 1);
}

mx::array Model::SpeakerEmbedding(const std::vector<float> &samples) const {
  const auto convolve = [&](mx::array hidden, const std::string &prefix,
                            int dilation, bool activation) {
    const auto &weight = checkpoint_.Weight(prefix + ".weight");
    const int padding = (weight.shape(1) - 1) * dilation / 2;
    if (padding > 0) {
      hidden = mx::pad(hidden, std::vector<int>{1}, mx::Shape{padding},
                       mx::Shape{padding}, mx::array(0), "reflect");
    } else {
    }
    auto output = mx::conv1d(hidden, weight, 1, 0, dilation) +
                  checkpoint_.Weight(prefix + ".bias");
    return activation ? mx::maximum(output, mx::array(0, output.dtype()))
                      : output;
  };
  auto hidden =
      convolve(SpeakerMels(samples), "speaker_encoder.blocks.0.conv", 1, true);
  std::vector<mx::array> aggregated;
  for (int block = 1; block < 4; ++block) {
    const auto stem = "speaker_encoder.blocks." + std::to_string(block);
    const auto input = hidden;
    const auto groups = mx::split(
        convolve(hidden, stem + ".tdnn1.conv", 1, true), kRes2NetScale, -1);
    std::vector<mx::array> outputs{groups.front()};
    for (int group = 1; group < kRes2NetScale; ++group) {
      const auto group_input =
          group == 1 ? groups[group] : groups[group] + outputs.back();
      outputs.push_back(convolve(group_input,
                                 stem + ".res2net_block.blocks." +
                                     std::to_string(group - 1) + ".conv",
                                 block + 1, true));
    }
    hidden =
        convolve(mx::concatenate(outputs, -1), stem + ".tdnn2.conv", 1, true);
    auto gate =
        convolve(mx::mean(hidden, 1, true), stem + ".se_block.conv1", 1, true);
    gate = mx::sigmoid(convolve(gate, stem + ".se_block.conv2", 1, false));
    hidden = input + hidden * gate;
    aggregated.push_back(hidden);
  }
  hidden = convolve(mx::concatenate(aggregated, -1), "speaker_encoder.mfa.conv",
                    1, true);
  const auto mean = mx::mean(hidden, 1, true);
  const auto standard_deviation =
      mx::sqrt(mx::var(hidden, 1, true) + mx::array(1e-12f));
  const auto context =
      mx::concatenate({hidden, mx::broadcast_to(mean, hidden.shape()),
                       mx::broadcast_to(standard_deviation, hidden.shape())},
                      -1);
  const auto attention = mx::softmax(
      convolve(
          mx::tanh(convolve(context, "speaker_encoder.asp.tdnn.conv", 1, true)),
          "speaker_encoder.asp.conv", 1, false),
      1);
  const auto weighted_mean = mx::sum(attention * hidden, 1, true);
  const auto centered = hidden - weighted_mean;
  const auto weighted_std = mx::sqrt(mx::maximum(
      mx::sum(attention * centered * centered, 1, true), mx::array(1e-12f)));
  return convolve(mx::concatenate({weighted_mean, weighted_std}, -1),
                  "speaker_encoder.fc", 1, false);
}

} // namespace qwen3_tts
