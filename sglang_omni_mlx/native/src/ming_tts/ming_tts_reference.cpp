// SPDX-License-Identifier: Apache-2.0
#include "ming_tts/ming_tts_reference.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <limits>
#include <numbers>

namespace ming_tts {
namespace mx = mlx::core;

namespace {

constexpr int kSpeakerSampleRate = 16000;
constexpr int kAudioSampleRate = 44100;
constexpr int kWindowSamples = 400;
constexpr int kHopSamples = 160;
constexpr int kFftSamples = 512;
constexpr int kMelBins = 80;
constexpr int kContextFrames = 100;
constexpr float kBatchNormEpsilon = 1e-5f;

mx::array Fbank(const std::vector<float> &samples) {
  const int frames =
      1 + (static_cast<int>(samples.size()) - kWindowSamples) / kHopSamples;
  const auto indices = mx::reshape(mx::arange(frames), {-1, 1}) * kHopSamples +
                       mx::arange(kWindowSamples);
  auto framed = mx::take(
      mx::array(samples.data(), {static_cast<int>(samples.size())}), indices);
  framed = framed - mx::mean(framed, -1, true);
  const auto previous =
      mx::concatenate({mx::slice(framed, {0, 0}, {frames, 1}),
                       mx::slice(framed, {0, 0}, {frames, kWindowSamples - 1})},
                      -1);
  const auto window =
      mx::power(0.5f - 0.5f * mx::cos(mx::arange(kWindowSamples, mx::float32) *
                                      (2.0f * std::numbers::pi_v<float> /
                                       (kWindowSamples - 1))),
                mx::array(0.85f));
  const auto spectrum = mx::abs(
      mx::fft::rfft((framed - 0.97f * previous) * window, kFftSamples, -1));
  const float low = 1127.0f * std::log1p(20.0f / 700.0f);
  const float high = 1127.0f * std::log1p(kSpeakerSampleRate / 2.0f / 700.0f);
  const float delta = (high - low) / (kMelBins + 1);
  std::vector<float> filters((kFftSamples / 2 + 1) * kMelBins, 0.0f);
  for (int i = 0; i < kFftSamples / 2; ++i) {
    const float mel =
        1127.0f * std::log1p(static_cast<float>(i * kSpeakerSampleRate) /
                             kFftSamples / 700.0f);
    for (int j = 0; j < kMelBins; ++j) {
      const float left = low + j * delta;
      filters[i * kMelBins + j] =
          std::max(0.0f, std::min((mel - left) / delta,
                                  (left + 2 * delta - mel) / delta));
    }
  }
  const auto features = mx::log(mx::maximum(
      mx::matmul(spectrum * spectrum,
                 mx::array(filters.data(), {kFftSamples / 2 + 1, kMelBins})),
      mx::array(std::numeric_limits<float>::epsilon())));
  return mx::expand_dims(features - mx::mean(features, 0, true), 0);
}

std::vector<float> ResampleForVae(const std::vector<float> &samples) {
  // Note (altale): Six-crossing Hann sinc with 0.99 rolloff matches the
  // reference resampler.
  constexpr double rolloff = 0.99;
  constexpr int width = 7;
  std::vector<float> output(
      (samples.size() * kAudioSampleRate + kSpeakerSampleRate - 1) /
      kSpeakerSampleRate);
  for (size_t i = 0; i < output.size(); ++i) {
    const double position =
        static_cast<double>(i) * kSpeakerSampleRate / kAudioSampleRate;
    const int center = static_cast<int>(position);
    for (int j = std::max(0, center - width);
         j < std::min(static_cast<int>(samples.size()), center + width + 1);
         ++j) {
      const double offset = (j - position) * rolloff;
      if (std::abs(offset) < 6.0) {
        const double angle = std::numbers::pi * offset;
        const double window = std::cos(angle / 12.0);
        const double sinc = angle == 0.0 ? 1.0 : std::sin(angle) / angle;
        output[i] +=
            samples[j] * static_cast<float>(sinc * window * window * rolloff);
      } else {
      }
    }
  }
  return output;
}

} // namespace

CampPlus::CampPlus(const std::filesystem::path &path)
    : speaker_(mx::load_safetensors(path.string()).first,
               nlohmann::json::object()) {}

mx::array CampPlus::BatchNorm(const mx::array &hidden,
                              const std::string &prefix) const {
  auto normalized =
      (hidden - speaker_.Weight(prefix + ".running_mean")) *
      mx::rsqrt(speaker_.Weight(prefix + ".running_var") + kBatchNormEpsilon);
  if (speaker_.Has(prefix + ".weight")) {
    return normalized * speaker_.Weight(prefix + ".weight") +
           speaker_.Weight(prefix + ".bias");
  } else {
    return normalized;
  }
}

mx::array CampPlus::Conv1d(const mx::array &hidden, const std::string &weight,
                           const std::string &bias, int stride,
                           int dilation) const {
  const auto kernel = mx::transpose(speaker_.Weight(weight), {0, 2, 1});
  auto output = mx::conv1d(hidden, kernel, stride,
                           (kernel.shape(1) - 1) / 2 * dilation, dilation);
  if (!bias.empty()) {
    return output + speaker_.Weight(bias);
  } else {
    return output;
  }
}

mx::array CampPlus::Encode(const std::vector<float> &samples) const {
  // Note (altale): The exported checkpoint retains numeric names for folded
  // convolutions.
  const auto name = [](int index) {
    return "onnx::Conv_" + std::to_string(index);
  };
  const auto relu = [](const mx::array &hidden) {
    return mx::maximum(hidden, mx::array(0.0f));
  };
  const auto conv2d = [&](const mx::array &hidden, int index, int stride) {
    const auto kernel =
        mx::transpose(speaker_.Weight(name(index)), {0, 2, 3, 1});
    const int pad = (kernel.shape(1) - 1) / 2;
    return mx::conv2d(hidden, kernel, {stride, 1}, {pad, pad}) +
           speaker_.Weight(name(index + 1));
  };
  auto hidden = relu(conv2d(
      mx::expand_dims(mx::transpose(Fbank(samples), {0, 2, 1}), -1), 4423, 1));
  for (int first : {4426, 4441}) {
    hidden = relu(conv2d(relu(conv2d(hidden, first, 2)), first + 3, 1) +
                  conv2d(hidden, first + 6, 2));
    hidden = relu(conv2d(relu(conv2d(hidden, first + 9, 1)), first + 12, 1) +
                  hidden);
  }
  hidden = relu(conv2d(hidden, 4456, 2));
  // Note (altale): Flatten channel before frequency, preserving the exported
  // NCHW order.
  hidden = mx::reshape(mx::transpose(hidden, {0, 2, 3, 1}),
                       {1, hidden.shape(2), -1});
  hidden = relu(Conv1d(hidden, name(4459), name(4460), 2));
  int convolution = 4462;
  constexpr std::array depths{12, 24, 16};
  for (int block = 0; block < 3; ++block) {
    for (int layer = 1; layer <= depths[block]; ++layer) {
      const std::string prefix = "xvector.block" + std::to_string(block + 1) +
                                 ".tdnnd" + std::to_string(layer);
      auto bottleneck =
          relu(Conv1d(relu(BatchNorm(hidden, prefix + ".nonlinear1.batchnorm")),
                      name(convolution), name(convolution + 1)));
      convolution += 3;
      const std::string attention = prefix + ".cam_layer";
      const auto local = Conv1d(bottleneck, attention + ".linear_local.weight",
                                "", 1, block == 0 ? 1 : 2);
      const int frames = bottleneck.shape(1);
      std::vector<mx::array> segments;
      for (int start = 0; start < frames; start += kContextFrames) {
        const int end = std::min(start + kContextFrames, frames);
        segments.push_back(
            mx::broadcast_to(mx::mean(mx::slice(bottleneck, {0, start, 0},
                                                {1, end, bottleneck.shape(2)}),
                                      1, true),
                             {1, end - start, bottleneck.shape(2)}));
      }
      const auto context =
          mx::concatenate(segments, 1) + mx::mean(bottleneck, 1, true);
      const auto gate = mx::sigmoid(
          Conv1d(relu(Conv1d(context, attention + ".linear1.weight",
                             attention + ".linear1.bias")),
                 attention + ".linear2.weight", attention + ".linear2.bias"));
      hidden = mx::concatenate({hidden, local * gate}, -1);
    }
    const std::string transit = "xvector.transit" + std::to_string(block + 1);
    hidden = relu(BatchNorm(hidden, transit + ".nonlinear.batchnorm"));
    hidden = block == 2 ? relu(Conv1d(hidden, name(4618), name(4619)))
                        : Conv1d(hidden, transit + ".linear.weight", "");
    mx::eval(hidden);
  }
  const auto mean = mx::mean(hidden, 1, true);
  const auto centered = hidden - mean;
  const auto deviation =
      mx::sqrt(mx::sum(centered * centered, 1, true) / (hidden.shape(1) - 1));
  const auto pooled = mx::concatenate({mean, deviation}, -1);
  return mx::squeeze(
      BatchNorm(Conv1d(pooled, "xvector.dense.linear.weight", ""),
                "xvector.dense.nonlinear.batchnorm"),
      1);
}

MingReferenceEncoder::MingReferenceEncoder(
    const std::filesystem::path &directory)
    : speaker_(directory / "campplus.safetensors"), audio_(directory) {
  const auto config = qwen3_asr::ReadJson(directory / "config.json");
  const auto &audio = config.at("audio_tokenizer_config");
  // Note (altale): Pad to complete acoustic patches after both encoder
  // reductions.
  alignment_samples_ = config.at("ditar_config").at("patch_size").get<int>() *
                       audio.at("enc_kwargs").value("hop_size", 320) *
                       std::max(1, audio.at("patch_size").get<int>());
}

ReferenceConditioning
MingReferenceEncoder::Encode(const std::vector<float> &samples) const {
  assert(samples.size() >= kWindowSamples);
  auto resampled = ResampleForVae(samples);
  resampled.resize((resampled.size() + alignment_samples_ - 1) /
                       alignment_samples_ * alignment_samples_,
                   0.0f);
  const auto latents = mx::astype(
      audio_.Encode(
          mx::array(resampled.data(), {1, static_cast<int>(resampled.size())})),
      mx::float32);
  const auto speaker = speaker_.Encode(samples);
  mx::eval(latents, speaker);
  return {latents, speaker};
}

} // namespace ming_tts
