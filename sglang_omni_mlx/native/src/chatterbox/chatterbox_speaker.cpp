// SPDX-License-Identifier: Apache-2.0
#include "chatterbox/chatterbox_speaker.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

#include "chatterbox/chatterbox_audio.h"

namespace chatterbox {

namespace mx = mlx::core;

namespace {

constexpr int kFbankSampleRate = 16000;
constexpr int kFbankFrameSamples = 400;
constexpr int kFbankShiftSamples = 160;
constexpr int kFbankFftSize = 512;
constexpr int kFbankBinCount = 80;
constexpr double kFbankLowHz = 20.0;
constexpr double kPreemphasis = 0.97;
constexpr double kPoveyExponent = 0.85;
constexpr float kBatchNormEps = 1e-5f;
constexpr int kContextSegmentFrames = 100;
// Layers per densely connected block and each block's dilation.
constexpr int kBlockLayerCounts[] = {12, 24, 16};
constexpr int kBlockDilations[] = {1, 2, 2};

// Kaldi's triangular mel filters on its 1127 ln(1 + f / 700) scale,
// [mel_bins, fft_size / 2 + 1] with the Nyquist bin zero.
std::vector<double> KaldiMelBanks() {
  constexpr int frequency_bin_count = kFbankFftSize / 2 + 1;
  const double nyquist_hz = 0.5 * kFbankSampleRate;
  const double bin_width_hz =
      static_cast<double>(kFbankSampleRate) / kFbankFftSize;
  const double low_mel = 1127.0 * std::log(1.0 + kFbankLowHz / 700.0);
  const double high_mel = 1127.0 * std::log(1.0 + nyquist_hz / 700.0);
  const double mel_step = (high_mel - low_mel) / (kFbankBinCount + 1);
  std::vector<double> banks(
      static_cast<size_t>(kFbankBinCount) * frequency_bin_count, 0.0);
  for (int mel_bin = 0; mel_bin < kFbankBinCount; ++mel_bin) {
    const double left_mel = low_mel + mel_bin * mel_step;
    const double center_mel = low_mel + (mel_bin + 1) * mel_step;
    const double right_mel = low_mel + (mel_bin + 2) * mel_step;
    for (int bin = 0; bin < kFbankFftSize / 2; ++bin) {
      const double mel = 1127.0 * std::log(1.0 + bin_width_hz * bin / 700.0);
      double weight = 0.0;
      if (mel > left_mel && mel <= center_mel) {
        weight = (mel - left_mel) / (center_mel - left_mel);
      } else if (mel > center_mel && mel < right_mel) {
        weight = (right_mel - mel) / (right_mel - center_mel);
      } else {
      }
      banks[mel_bin * frequency_bin_count + bin] = weight;
    }
  }
  return banks;
}

// Kaldi fbank of 16 kHz samples (snipped edges, per-frame DC removal,
// pre-emphasis, Povey window, power spectrum), float32 [frames, 80]; one zero
// frame when the audio is shorter than a frame.
std::vector<float> FbankFeatures(const std::vector<float> &samples,
                                 int &frame_count) {
  static const std::vector<double> banks = KaldiMelBanks();
  static const std::vector<double> window = [] {
    std::vector<double> values(kFbankFrameSamples);
    for (int i = 0; i < kFbankFrameSamples; ++i) {
      values[i] = std::pow(
          0.5 - 0.5 * std::cos(2 * M_PI * i / (kFbankFrameSamples - 1)),
          kPoveyExponent);
    }
    return values;
  }();
  constexpr int frequency_bin_count = kFbankFftSize / 2 + 1;
  const long sample_count = static_cast<long>(samples.size());
  frame_count = sample_count < kFbankFrameSamples
                    ? 0
                    : static_cast<int>(1 + (sample_count - kFbankFrameSamples) /
                                               kFbankShiftSamples);
  if (frame_count == 0) {
    frame_count = 1;
    return std::vector<float>(kFbankBinCount, 0.0f);
  } else {
  }
  const double floor = std::numeric_limits<float>::epsilon();
  std::vector<float> features(static_cast<size_t>(frame_count) *
                              kFbankBinCount);
  std::vector<double> frame(kFbankFrameSamples);
  std::vector<double> padded(kFbankFftSize, 0.0);
  std::vector<double> power(frequency_bin_count);
  for (int frame_index = 0; frame_index < frame_count; ++frame_index) {
    const float *start = samples.data() + frame_index * kFbankShiftSamples;
    for (int i = 0; i < kFbankFrameSamples; ++i) {
      frame[i] = static_cast<double>(start[i]);
    }
    const double mean = PairwiseSum(frame.data(), frame.size()) /
                        static_cast<double>(kFbankFrameSamples);
    for (double &value : frame) {
      value -= mean;
    }
    padded[0] = frame[0] * window[0];
    for (int i = 1; i < kFbankFrameSamples; ++i) {
      padded[i] = (frame[i] - kPreemphasis * frame[i - 1]) * window[i];
    }
    const std::vector<std::complex<double>> spectrum = RealFft(padded);
    for (int bin = 0; bin < frequency_bin_count; ++bin) {
      const double magnitude =
          std::hypot(spectrum[bin].real(), spectrum[bin].imag());
      power[bin] = magnitude * magnitude;
    }
    for (int mel_bin = 0; mel_bin < kFbankBinCount; ++mel_bin) {
      const double *row = banks.data() + mel_bin * frequency_bin_count;
      double energy = 0.0;
      for (int bin = 0; bin < frequency_bin_count; ++bin) {
        energy += power[bin] * row[bin];
      }
      features[frame_index * kFbankBinCount + mel_bin] =
          static_cast<float>(std::log(std::max(energy, floor)));
    }
  }
  return features;
}

mx::array Relu(const mx::array &x) { return mx::maximum(x, mx::array(0.0f)); }

// [batch, channels, time] <-> [batch, time, channels].
mx::array SwapTimeAndChannels(const mx::array &x) {
  return mx::transpose(x, {0, 2, 1});
}

// Mean over 100-frame segments, the last, partial one over its own frames as
// average pooling with ceil mode takes it, repeated back over each segment's
// frames; x is [batch, time, channels].
mx::array SegmentMeans(const mx::array &x) {
  const int batch = x.shape(0);
  const int frame_count = x.shape(1);
  const int channel_count = x.shape(2);
  const int segment_count =
      (frame_count + kContextSegmentFrames - 1) / kContextSegmentFrames;
  const int padding = segment_count * kContextSegmentFrames - frame_count;
  mx::array padded = x;
  if (padding > 0) {
    padded = mx::concatenate(
        {x, mx::zeros({batch, padding, channel_count}, mx::float32)}, 1);
  } else {
  }
  std::vector<float> segment_frames(segment_count, kContextSegmentFrames);
  segment_frames.back() = static_cast<float>(kContextSegmentFrames - padding);
  const mx::array segments = mx::divide(
      mx::sum(mx::reshape(padded, {batch, segment_count, kContextSegmentFrames,
                                   channel_count}),
              2),
      mx::array(segment_frames.begin(), {1, segment_count, 1}, mx::float32));
  const mx::array repeated = mx::reshape(
      mx::repeat(mx::expand_dims(segments, 2), kContextSegmentFrames, 2),
      {batch, -1, channel_count});
  return mx::slice(repeated, {0, 0, 0}, {batch, frame_count, channel_count});
}

} // namespace

SpeakerEncoder::SpeakerEncoder(const omni_common::Checkpoint &checkpoint,
                               const std::string &weight_prefix) {
  const auto copy_weight = [&](const std::string &name) {
    mx::array weight = checkpoint.Weight(weight_prefix + name);
    // Square 2-D kernels are stored [out, in, height, width]; MLX convolves
    // with [out, height, width, in].
    if (weight.ndim() == 4) {
      weight = mx::contiguous(mx::transpose(weight, {0, 2, 3, 1}));
    } else {
    }
    weights_.insert_or_assign(name, weight);
  };
  const auto copy_batch_norm = [&](const std::string &prefix, bool affine) {
    copy_weight(prefix + ".running_mean");
    copy_weight(prefix + ".running_var");
    if (affine) {
      copy_weight(prefix + ".weight");
      copy_weight(prefix + ".bias");
    } else {
    }
  };
  copy_weight("head.conv1.weight");
  copy_batch_norm("head.bn1", true);
  for (const std::string layer : {"head.layer1", "head.layer2"}) {
    for (const std::string block : {".0", ".1"}) {
      copy_weight(layer + block + ".conv1.weight");
      copy_batch_norm(layer + block + ".bn1", true);
      copy_weight(layer + block + ".conv2.weight");
      copy_batch_norm(layer + block + ".bn2", true);
    }
    copy_weight(layer + ".0.shortcut.0.weight");
    copy_batch_norm(layer + ".0.shortcut.1", true);
  }
  copy_weight("head.conv2.weight");
  copy_batch_norm("head.bn2", true);
  copy_weight("xvector.tdnn.linear.weight");
  copy_batch_norm("xvector.tdnn.nonlinear.batchnorm", true);
  for (int block = 0; block < 3; ++block) {
    const std::string block_prefix =
        "xvector.block" + std::to_string(block + 1) + ".tdnnd";
    for (int layer = 0; layer < kBlockLayerCounts[block]; ++layer) {
      const std::string prefix = block_prefix + std::to_string(layer + 1);
      copy_batch_norm(prefix + ".nonlinear1.batchnorm", true);
      copy_weight(prefix + ".linear1.weight");
      copy_batch_norm(prefix + ".nonlinear2.batchnorm", true);
      copy_weight(prefix + ".cam_layer.linear_local.weight");
      copy_weight(prefix + ".cam_layer.linear1.weight");
      copy_weight(prefix + ".cam_layer.linear1.bias");
      copy_weight(prefix + ".cam_layer.linear2.weight");
      copy_weight(prefix + ".cam_layer.linear2.bias");
    }
    const std::string transit = "xvector.transit" + std::to_string(block + 1);
    copy_batch_norm(transit + ".nonlinear.batchnorm", true);
    copy_weight(transit + ".linear.weight");
  }
  copy_batch_norm("xvector.out_nonlinear.batchnorm", true);
  copy_weight("xvector.dense.linear.weight");
  copy_batch_norm("xvector.dense.nonlinear.batchnorm", false);
}

const mx::array &SpeakerEncoder::Weight(const std::string &name) const {
  const auto found = weights_.find(name);
  if (found == weights_.end()) {
    throw std::runtime_error("missing speaker encoder weight " + name);
  } else {
  }
  return found->second;
}

mx::array SpeakerEncoder::BatchNorm(const mx::array &x,
                                    const std::string &prefix) const {
  const mx::array normalized =
      mx::multiply(mx::subtract(x, Weight(prefix + ".running_mean")),
                   mx::rsqrt(mx::add(Weight(prefix + ".running_var"),
                                     mx::array(kBatchNormEps))));
  if (weights_.count(prefix + ".weight") > 0) {
    return mx::add(mx::multiply(Weight(prefix + ".weight"), normalized),
                   Weight(prefix + ".bias"));
  } else {
    return normalized;
  }
}

// x is [batch, frequency, time, channels]; stride steps along frequency.
mx::array SpeakerEncoder::ResidualBlock(const mx::array &x,
                                        const std::string &prefix,
                                        int stride) const {
  mx::array out = Relu(BatchNorm(
      mx::conv2d(x, Weight(prefix + ".conv1.weight"), {stride, 1}, {1, 1}),
      prefix + ".bn1"));
  out = BatchNorm(
      mx::conv2d(out, Weight(prefix + ".conv2.weight"), {1, 1}, {1, 1}),
      prefix + ".bn2");
  const mx::array shortcut =
      stride != 1
          ? BatchNorm(mx::conv2d(x, Weight(prefix + ".shortcut.0.weight"),
                                 {stride, 1}, {0, 0}),
                      prefix + ".shortcut.1")
          : x;
  return Relu(mx::add(out, shortcut));
}

// One densely connected layer's new channels; x is [batch, channels, time].
mx::array SpeakerEncoder::DenseLayer(const mx::array &x,
                                     const std::string &prefix,
                                     int dilation) const {
  mx::array hidden = SwapTimeAndChannels(x);
  hidden = Relu(BatchNorm(hidden, prefix + ".nonlinear1.batchnorm"));
  hidden = mx::conv1d(hidden, Weight(prefix + ".linear1.weight"));
  hidden = Relu(BatchNorm(hidden, prefix + ".nonlinear2.batchnorm"));
  hidden = SwapTimeAndChannels(hidden);

  // Context-aware masking: local features gated by global and segment means.
  hidden = SwapTimeAndChannels(hidden);
  const mx::array local =
      mx::conv1d(hidden, Weight(prefix + ".cam_layer.linear_local.weight"), 1,
                 dilation, dilation);
  mx::array context = mx::add(mx::mean(hidden, 1, true), SegmentMeans(hidden));
  context = Relu(
      mx::add(mx::conv1d(context, Weight(prefix + ".cam_layer.linear1.weight")),
              Weight(prefix + ".cam_layer.linear1.bias")));
  const mx::array mask = mx::sigmoid(
      mx::add(mx::conv1d(context, Weight(prefix + ".cam_layer.linear2.weight")),
              Weight(prefix + ".cam_layer.linear2.bias")));
  return SwapTimeAndChannels(mx::multiply(local, mask));
}

mx::array SpeakerEncoder::Embed(const std::vector<float> &samples_16k) const {
  int frame_count = 0;
  const std::vector<float> fbank = FbankFeatures(samples_16k, frame_count);
  mx::array features(fbank.data(), {1, frame_count, kFbankBinCount},
                     mx::float32);
  features = mx::subtract(features, mx::mean(features, 1, true));

  // Convolutional head over [batch, frequency, time, 1].
  mx::array x = mx::expand_dims(SwapTimeAndChannels(features), 3);
  x = Relu(BatchNorm(mx::conv2d(x, Weight("head.conv1.weight"), {1, 1}, {1, 1}),
                     "head.bn1"));
  for (const std::string layer : {"head.layer1", "head.layer2"}) {
    x = ResidualBlock(x, layer + ".0", 2);
    x = ResidualBlock(x, layer + ".1", 1);
  }
  x = Relu(BatchNorm(mx::conv2d(x, Weight("head.conv2.weight"), {2, 1}, {1, 1}),
                     "head.bn2"));
  const int batch = x.shape(0);
  const int reduced_frequency = x.shape(1);
  const int frames = x.shape(2);
  const int channels = x.shape(3);
  x = mx::reshape(mx::transpose(x, {0, 3, 1, 2}),
                  {batch, channels * reduced_frequency, frames});

  // Time-delay network over [batch, channels, time].
  x = SwapTimeAndChannels(x);
  x = mx::conv1d(x, Weight("xvector.tdnn.linear.weight"), 2, 2);
  x = Relu(BatchNorm(x, "xvector.tdnn.nonlinear.batchnorm"));
  x = SwapTimeAndChannels(x);
  for (int block = 0; block < 3; ++block) {
    const std::string block_prefix =
        "xvector.block" + std::to_string(block + 1) + ".tdnnd";
    for (int layer = 0; layer < kBlockLayerCounts[block]; ++layer) {
      x = mx::concatenate(
          {x, DenseLayer(x, block_prefix + std::to_string(layer + 1),
                         kBlockDilations[block])},
          1);
    }
    const std::string transit = "xvector.transit" + std::to_string(block + 1);
    x = SwapTimeAndChannels(x);
    x = Relu(BatchNorm(x, transit + ".nonlinear.batchnorm"));
    x = mx::conv1d(x, Weight(transit + ".linear.weight"));
    x = SwapTimeAndChannels(x);
  }
  x = SwapTimeAndChannels(x);
  x = Relu(BatchNorm(x, "xvector.out_nonlinear.batchnorm"));
  x = SwapTimeAndChannels(x);

  // Mean and unbiased standard deviation over time, then the projection.
  x = mx::concatenate({mx::mean(x, 2), mx::std(x, 2, false, 1)}, 1);
  x = mx::squeeze(
      mx::conv1d(mx::expand_dims(x, 1), Weight("xvector.dense.linear.weight")),
      1);
  x = BatchNorm(x, "xvector.dense.nonlinear.batchnorm");
  mx::eval(x);
  return x;
}

} // namespace chatterbox
