// SPDX-License-Identifier: Apache-2.0
#include "dots_tts/dots_tts_speaker.h"

#include <cmath>
#include <cstdint>
#include <stdexcept>

#include "dots_tts/dots_tts_checkpoint.h"

namespace dots_tts {

namespace mx = mlx::core;

namespace {

constexpr float kSpeakerEpsilon = 1.1920928955078125e-07f;
constexpr float kPreEmphasis = 0.97f;
constexpr float kPoveyExp = 0.85f;
constexpr int kFftSize = 512;
constexpr int kWindowSize = 400;
constexpr int kHopSize = 160;

float MelScale(float freq) { return 1127.0f * std::log(1.0f + freq / 700.0f); }

} // namespace

std::vector<float> KaldiFbankFrameWindow() {
  std::vector<float> window(static_cast<size_t>(kWindowSize));
  for (int n = 0; n < kWindowSize; ++n) {
    const double angle = 2.0 * 3.141592653589793 * static_cast<double>(n) /
                         static_cast<double>(kWindowSize - 1);
    const double hann = 0.5 - 0.5 * std::cos(angle);
    window[n] = static_cast<float>(std::pow(hann, kPoveyExp));
  }
  return window;
}

std::vector<float> KaldiMelBanks(int num_bins, float sample_freq) {
  const int num_fft_bins = kFftSize / 2;
  const float nyquist = 0.5f * sample_freq;
  const float high_freq = nyquist; // note (guozhihao-224): default 0 -> Nyquist
  const float low_freq = 20.0f;
  const float fft_bin_width = sample_freq / static_cast<float>(kFftSize);
  const float mel_low = MelScale(low_freq);
  const float mel_high = MelScale(high_freq);
  const float mel_delta =
      (mel_high - mel_low) / static_cast<float>(num_bins + 1);

  std::vector<float> banks(static_cast<size_t>(num_bins) * (num_fft_bins + 1),
                           0.0f);
  const std::vector<float> mel_of_bin = [&]() {
    std::vector<float> out(static_cast<size_t>(num_fft_bins) + 1);
    for (int k = 0; k <= num_fft_bins; ++k) {
      out[k] = MelScale(fft_bin_width * static_cast<float>(k));
    }
    return out;
  }();

  for (int i = 0; i < num_bins; ++i) {
    const float left = mel_low + static_cast<float>(i) * mel_delta;
    const float center = mel_low + static_cast<float>(i + 1) * mel_delta;
    const float right = mel_low + static_cast<float>(i + 2) * mel_delta;
    for (int k = 0; k <= num_fft_bins; ++k) {
      float value = 0.0f;
      if (mel_of_bin[k] >= left && mel_of_bin[k] <= right) {
        const float up = (mel_of_bin[k] - left) / (center - left);
        const float down = (right - mel_of_bin[k]) / (right - center);
        value = std::max(0.0f, std::min(up, down));
      }
      banks[i * (num_fft_bins + 1) + k] = value;
    }
  }
  return banks;
}

KaldiFbank::KaldiFbank() {
  window_ = KaldiFbankFrameWindow();
  // note (guozhihao-224): fixed 512-point real DFT basis, k = 0..256.
  const int bins = kFftSize / 2 + 1;
  dft_cos_.assign(static_cast<size_t>(bins) * kFftSize, 0.0f);
  dft_sin_.assign(static_cast<size_t>(bins) * kFftSize, 0.0f);
  for (int k = 0; k < bins; ++k) {
    for (int n = 0; n < kFftSize; ++n) {
      const double angle = -2.0 * 3.141592653589793 *
                           static_cast<double>(k * n) /
                           static_cast<double>(kFftSize);
      dft_cos_[static_cast<size_t>(k) * kFftSize + n] =
          static_cast<float>(std::cos(angle));
      dft_sin_[static_cast<size_t>(k) * kFftSize + n] =
          static_cast<float>(std::sin(angle));
    }
  }
  mel_ = KaldiMelBanks(kSpeakerMelBins, 16000.0f);
}

std::vector<float> KaldiFbank::operator()(const std::vector<float> &wav) const {
  const int num_samples = static_cast<int>(wav.size());
  if (num_samples < kWindowSize) {
    return {};
  }
  const int m = 1 + (num_samples - kWindowSize) / kHopSize;
  std::vector<float> frames(static_cast<size_t>(m) * kWindowSize);
  for (int i = 0; i < m; ++i) {
    const int base = i * kHopSize;
    for (int n = 0; n < kWindowSize; ++n) {
      frames[static_cast<size_t>(i) * kWindowSize + n] = wav[base + n];
    }
  }

  for (int i = 0; i < m; ++i) {
    float sum = 0.0f;
    for (int n = 0; n < kWindowSize; ++n)
      sum += frames[static_cast<size_t>(i) * kWindowSize + n];
    const float mean = sum / static_cast<float>(kWindowSize);
    for (int n = 0; n < kWindowSize; ++n)
      frames[static_cast<size_t>(i) * kWindowSize + n] -= mean;
  }
  // note (guozhihao-224): kaldi pre-emphasis y[n] = x[n] - 0.97*x[n-1] over
  // the DC-removed samples, with the left edge y[0] = 0.03*x[0] matching
  // torchaudio's replicate-padded offset. A forward scan feeds the just
  // rewritten y[n-1] back in and becomes recursive, so scan backwards, where
  // sample n-1 is still untouched when sample n is written.
  for (int i = 0; i < m; ++i) {
    float *frame = &frames[static_cast<size_t>(i) * kWindowSize];
    for (int n = kWindowSize - 1; n >= 1; --n)
      frame[n] -= kPreEmphasis * frame[n - 1];
    frame[0] *= 1.0f - kPreEmphasis;
  }
  const int bins = kFftSize / 2 + 1;
  std::vector<float> spectrum(static_cast<size_t>(m) * bins, 0.0f);
  for (int i = 0; i < m; ++i) {
    std::vector<float> frame(kFftSize, 0.0f);
    for (int n = 0; n < kWindowSize; ++n) {
      frame[n] = frames[static_cast<size_t>(i) * kWindowSize + n] * window_[n];
    }
    for (int k = 0; k < bins; ++k) {
      float real = 0.0f;
      float imag = 0.0f;
      const float *cos_row = &dft_cos_[static_cast<size_t>(k) * kFftSize];
      const float *sin_row = &dft_sin_[static_cast<size_t>(k) * kFftSize];
      for (int n = 0; n < kFftSize; ++n) {
        real += frame[n] * cos_row[n];
        imag += frame[n] * sin_row[n];
      }
      spectrum[static_cast<size_t>(i) * bins + k] = real * real + imag * imag;
    }
  }

  std::vector<float> mel_energies(static_cast<size_t>(m) * kSpeakerMelBins,
                                  0.0f);
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < kSpeakerMelBins; ++j) {
      float energy = 0.0f;
      const float *spectrum_row = &spectrum[static_cast<size_t>(i) * bins];
      const float *mel_row = &mel_[static_cast<size_t>(j) * bins];
      for (int k = 0; k < bins; ++k) {
        energy += spectrum_row[k] * mel_row[k];
      }
      mel_energies[static_cast<size_t>(i) * kSpeakerMelBins + j] =
          std::log(std::max(energy, kSpeakerEpsilon));
    }
  }

  std::vector<float> means(static_cast<size_t>(kSpeakerMelBins), 0.0f);
  for (int j = 0; j < kSpeakerMelBins; ++j) {
    float sum = 0.0f;
    for (int i = 0; i < m; ++i)
      sum += mel_energies[static_cast<size_t>(i) * kSpeakerMelBins + j];
    means[j] = sum / static_cast<float>(m);
  }
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < kSpeakerMelBins; ++j) {
      mel_energies[static_cast<size_t>(i) * kSpeakerMelBins + j] -= means[j];
    }
  }
  return mel_energies;
}

mx::array SpeakerConv2d::operator()(const mx::array &x) const {
  mx::array y = mx::conv2d(x, weight, stride, padding);
  if (bias.has_value()) {
    y = mx::add(y, *bias);
  }
  return y;
}

mx::array SpeakerConv1d::operator()(const mx::array &x) const {
  mx::array y = mx::conv1d(x, weight, stride, padding, dilation);
  if (bias.has_value()) {
    y = mx::add(y, *bias);
  }
  return y;
}

namespace {
mx::array Relu(const mx::array &x) {
  return mx::maximum(x, mx::array(0.0f, x.dtype()));
}
} // namespace

mx::array BasicResBlock::operator()(const mx::array &x) const {
  mx::array out = Relu(bn1(conv1(x)));
  out = bn2(conv2(out));
  mx::array shortcut = x;
  if (shortcut_conv.has_value()) {
    shortcut = (*shortcut_bn)((*shortcut_conv)(x));
  }
  return Relu(mx::add(out, shortcut));
}

mx::array FCM::operator()(const mx::array &x) const {
  mx::array out = mx::expand_dims(x, -1);
  out = Relu(bn1(conv1(out)));
  for (const BasicResBlock &block : layer1)
    out = block(out);
  for (const BasicResBlock &block : layer2)
    out = block(out);
  out = Relu(bn2(conv2(out)));
  const int b = out.shape(0);
  const int h = out.shape(1);
  const int w = out.shape(2);
  const int c = out.shape(3);
  out = mx::transpose(out, {0, 3, 1, 2});
  return mx::reshape(out, {b, c * h, w});
}

mx::array TDNNLayer::operator()(const mx::array &x) const {
  return Relu(bn(conv(x)));
}

mx::array SegPooling(const mx::array &x, int seg_len) {
  const int b = x.shape(0);
  const int t = x.shape(1);
  const int c = x.shape(2);
  const int n_full = t / seg_len;
  std::vector<mx::array> pooled;
  if (n_full > 0) {
    const mx::array head =
        mx::reshape(mx::slice(x, {0, 0, 0}, {b, n_full * seg_len, c}),
                    {b, n_full, seg_len, c});
    pooled.push_back(mx::mean(head, 2));
  }
  const int rem = t - n_full * seg_len;
  if (rem > 0) {
    const mx::array tail =
        mx::mean(mx::slice(x, {0, n_full * seg_len, 0}, {b, t, c}), 1, true);
    pooled.push_back(tail);
  }
  const mx::array seg = mx::concatenate(pooled, 1);
  const int n_seg = seg.shape(1);
  const mx::array expanded =
      mx::broadcast_to(mx::expand_dims(seg, 2), {b, n_seg, seg_len, c});
  const mx::array flattened = mx::reshape(expanded, {b, n_seg * seg_len, c});
  return mx::slice(flattened, {0, 0, 0}, {b, t, c});
}

mx::array CAMLayer::operator()(const mx::array &x) const {
  const mx::array y = linear_local(x);
  const mx::array context = mx::add(mx::mean(x, 1, true), SegPooling(x));
  const mx::array m = mx::sigmoid(linear2(Relu(linear1(context))));
  return mx::multiply(y, m);
}

mx::array CamDenseTDNNLayer::operator()(const mx::array &x) const {
  mx::array out = linear1(Relu(bn1(x)));
  out = Relu(bn2(out));
  return cam_layer(out);
}

mx::array CamDenseTDNNBlock::operator()(const mx::array &x) const {
  mx::array out = x;
  for (const CamDenseTDNNLayer &layer : layers) {
    out = mx::concatenate({out, layer(out)}, 2);
  }
  return out;
}

mx::array TransitLayer::operator()(const mx::array &x) const {
  return linear(Relu(bn(x)));
}

mx::array StatsPooling(const mx::array &x) {
  const int t = x.shape(1);
  const mx::array mean = mx::mean(x, 1);
  const mx::array centered = mx::subtract(x, mx::expand_dims(mean, 1));
  const mx::array var = mx::divide(mx::sum(mx::multiply(centered, centered), 1),
                                   mx::array(static_cast<float>(t - 1)));
  const mx::array std = mx::sqrt(var);
  return mx::concatenate({mean, std}, 1);
}

mx::array CAMPPlus::operator()(const mx::array &x) const {
  mx::array out = mx::transpose(x, {0, 2, 1});
  out = head(out);
  out = mx::transpose(out, {0, 2, 1});
  out = tdnn(out);
  out = block1(out);
  out = transit1(out);
  out = block2(out);
  out = transit2(out);
  out = block3(out);
  out = transit3(out);
  out = Relu(out_nonlinear(out));
  out = StatsPooling(out);
  out = mx::expand_dims(out, 1);
  out = dense_linear(out);
  out = dense_bn(out);
  return mx::reshape(out, {out.shape(0), out.shape(2)});
}

CAMPPlus LoadSpeaker(const DotsCheckpoint &checkpoint) {
  const auto conv2d = [&](const std::string &name,
                          const std::pair<int, int> &stride,
                          const std::pair<int, int> &padding) {
    return SpeakerConv2d{
        Conv2dWeight(checkpoint.Weight(name + ".weight"), mx::float32),
        std::nullopt, stride, padding};
  };
  const auto bn = [&](const std::string &name) {
    return BN{checkpoint.Weight(name + "_bn_scale"),
              checkpoint.Weight(name + "_bn_shift")};
  };
  const auto res_block = [&](const std::string &name,
                             const std::pair<int, int> &stride) {
    std::optional<SpeakerConv2d> shortcut_conv;
    std::optional<BN> shortcut_bn;
    if (checkpoint.Has(name + ".shortcut.0.weight")) {
      shortcut_conv = conv2d(name + ".shortcut.0", {stride.first, 1}, {0, 0});
      shortcut_bn = bn(name + ".shortcut.1");
    }
    return BasicResBlock{conv2d(name + ".conv1", {stride.first, 1}, {1, 1}),
                         bn(name + ".bn1"),
                         conv2d(name + ".conv2", {1, 1}, {1, 1}),
                         bn(name + ".bn2"),
                         shortcut_conv,
                         shortcut_bn};
  };
  const auto conv1d = [&](const std::string &name, int stride, int padding,
                          int dilation) {
    return SpeakerConv1d{
        Conv1dWeight(checkpoint.Weight(name + ".weight"), mx::float32),
        std::nullopt, stride, padding, dilation};
  };
  const auto cam_layer = [&](const std::string &name, int dilation) {
    return CAMLayer{
        conv1d(name + ".cam_layer.linear_local", 1, dilation, dilation),
        SpeakerConv1d{
            Conv1dWeight(checkpoint.Weight(name + ".cam_layer.linear1.weight"),
                         mx::float32),
            checkpoint.Weight(name + ".cam_layer.linear1.bias"), 1, 0, 1},
        SpeakerConv1d{
            Conv1dWeight(checkpoint.Weight(name + ".cam_layer.linear2.weight"),
                         mx::float32),
            checkpoint.Weight(name + ".cam_layer.linear2.bias"), 1, 0, 1}};
  };
  const auto dense_block = [&](const std::string &block_name, int num_layers,
                               int dilation) {
    std::vector<CamDenseTDNNLayer> layers;
    for (int i = 1; i <= num_layers; ++i) {
      const std::string name =
          "xvector." + block_name + ".tdnnd" + std::to_string(i);
      layers.push_back(CamDenseTDNNLayer{
          bn(name + ".nonlinear1.batchnorm"),
          SpeakerConv1d{
              Conv1dWeight(checkpoint.Weight(name + ".linear1.weight"),
                           mx::float32),
              std::nullopt, 1, 0, 1},
          bn(name + ".nonlinear2.batchnorm"), cam_layer(name, dilation)});
    }
    return CamDenseTDNNBlock{std::move(layers)};
  };
  const auto transit = [&](const std::string &name) {
    return TransitLayer{
        bn("xvector." + name + ".nonlinear.batchnorm"),
        SpeakerConv1d{Conv1dWeight(checkpoint.Weight("xvector." + name +
                                                     ".linear.weight"),
                                   mx::float32),
                      std::nullopt, 1, 0, 1}};
  };

  return CAMPPlus{
      FCM{conv2d("head.conv1", {1, 1}, {1, 1}),
          bn("head.bn1"),
          {res_block("head.layer1.0", {2, 1}),
           res_block("head.layer1.1", {1, 1})},
          {res_block("head.layer2.0", {2, 1}),
           res_block("head.layer2.1", {1, 1})},
          conv2d("head.conv2", {2, 1}, {1, 1}),
          bn("head.bn2")},
      TDNNLayer{conv1d("xvector.tdnn.linear", 2, 2, 1),
                bn("xvector.tdnn.nonlinear.batchnorm")},
      dense_block("block1", 12, 1),
      transit("transit1"),
      dense_block("block2", 24, 2),
      transit("transit2"),
      dense_block("block3", 16, 2),
      transit("transit3"),
      bn("xvector.out_nonlinear.batchnorm"),
      SpeakerConv1d{
          Conv1dWeight(checkpoint.Weight("xvector.dense.linear.weight"),
                       mx::float32),
          std::nullopt, 1, 0, 1},
      BN{checkpoint.Weight("xvector.dense.nonlinear.batchnorm_bn_scale"),
         checkpoint.Weight("xvector.dense.nonlinear.batchnorm_bn_shift")}};
}

} // namespace dots_tts
