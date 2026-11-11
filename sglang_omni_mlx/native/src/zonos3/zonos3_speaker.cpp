// SPDX-License-Identifier: Apache-2.0
// A port of mlx-audio's zonos3/speaker.py. The frontend stays in float64 on
// the host as the reference's NumPy frontend; the network runs in MLX with
// NHWC activations [1, mels, frames, channels].
#include "zonos3/zonos3_speaker.h"

#include <cmath>
#include <numbers>
#include <stdexcept>

#include "common/npz.h"
#include "zonos3/zonos3_istft_stream.h"

namespace zonos3 {

namespace mx = mlx::core;

namespace {

constexpr const char *kArchitecture = "mect_spk";
constexpr int kHiddenSize = 64;
constexpr int kMaxSamples16k = 60 * 16000;
constexpr int kInputRate = 441;
constexpr int kOutputRate = 160;
constexpr int kResampleWidth = 17;
// Per stage, each block's residual blocks and its frequency and time strides.
struct StageBlock {
  int residual_count;
  int frequency_stride;
  int time_stride;
};
const std::vector<std::vector<StageBlock>> &MectB2() {
  static const std::vector<std::vector<StageBlock>> stages = {
      {{3, 1, 1}, {3, 1, 1}},
      {{3, 2, 1}, {3, 1, 2}, {3, 1, 2}, {3, 1, 2}},
      {{3, 2, 1}, {3, 1, 2}, {3, 1, 2}, {3, 1, 2}, {3, 1, 2}},
      {{3, 2, 1}, {2, 1, 2}, {2, 1, 2}}};
  return stages;
}

mx::array Scalar(float value) { return mx::array(value); }

// 0.5 * x * (1 + tanh(sqrt(2 / pi) * (x + 0.044715 * x^3))), in the
// reference's order.
mx::array NewGelu(const mx::array &x) {
  const mx::array cubic =
      mx::multiply(mx::multiply(mx::multiply(Scalar(0.044715f), x), x), x);
  return mx::multiply(
      mx::multiply(Scalar(0.5f), x),
      mx::add(Scalar(1.0f),
              mx::tanh(mx::multiply(
                  Scalar(static_cast<float>(std::sqrt(2.0 / std::numbers::pi))),
                  mx::add(x, cubic)))));
}

mx::array ErfGelu(const mx::array &x) {
  return mx::multiply(
      mx::multiply(Scalar(0.5f), x),
      mx::add(Scalar(1.0f),
              mx::erf(mx::multiply(
                  x, Scalar(static_cast<float>(1.0 / std::sqrt(2.0)))))));
}

// [1, F, T, C] -> [T, F * C] with k = f * C + c.
mx::array Flatten(const mx::array &x) {
  const int frequencies = x.shape(1);
  const int frames = x.shape(2);
  const int channels = x.shape(3);
  return mx::reshape(mx::transpose(mx::squeeze(x, 0), {1, 0, 2}),
                     {frames, frequencies * channels});
}

mx::array Unflatten(const mx::array &x, int frequencies, int channels) {
  return mx::expand_dims(
      mx::transpose(mx::reshape(x, {x.shape(0), frequencies, channels}),
                    {1, 0, 2}),
      0);
}

// NumPy's pairwise float64 sum over a contiguous run, as np.mean uses it.
double PairwiseSum(const double *values, long long count) {
  constexpr long long kBlock = 128;
  if (count < 8) {
    double sum = 0.0;
    for (long long i = 0; i < count; ++i) {
      sum += values[i];
    }
    return sum;
  } else if (count <= kBlock) {
    double partial[8];
    for (int j = 0; j < 8; ++j) {
      partial[j] = values[j];
    }
    long long i = 8;
    for (; i < count - (count % 8); i += 8) {
      for (int j = 0; j < 8; ++j) {
        partial[j] += values[i + j];
      }
    }
    double sum = ((partial[0] + partial[1]) + (partial[2] + partial[3])) +
                 ((partial[4] + partial[5]) + (partial[6] + partial[7]));
    for (; i < count; ++i) {
      sum += values[i];
    }
    return sum;
  } else {
    long long half = count / 2;
    half -= half % 8;
    return PairwiseSum(values, half) + PairwiseSum(values + half, count - half);
  }
}

std::vector<double> ToDouble(const mx::array &array) {
  const mx::array as_float = mx::astype(array, mx::float32);
  mx::eval(as_float);
  const float *data = as_float.data<float>();
  return std::vector<double>(data, data + as_float.size());
}

} // namespace

SpeakerEncoder::SpeakerEncoder(const std::filesystem::path &encoder_directory)
    : weights_(encoder_directory, ""), resample_kernel_(mx::array(0.0f)) {
  if (weights_.architecture() != kArchitecture ||
      weights_.Meta("backbone") != "MECT_B2") {
    throw std::runtime_error(encoder_directory.string() +
                             ": expected a MECT-B2 speaker encoder checkpoint");
  } else {
  }
  embedding_size_ = weights_.Meta("embedding_length").get<int>();
  mel_count_ = weights_.Meta("n_mels").get<int>();
  frame_length_ = weights_.Meta("frame_length").get<int>();
  frame_shift_ = weights_.Meta("frame_shift").get<int>();
  fft_size_ = weights_.Meta("n_fft").get<int>();
  head_count_ = weights_.Meta("n_heads").get<int>();
  position_kernel_ = weights_.Meta("pos_kernel").get<int>();
  preemphasis_ = weights_.Meta("preemphasis").get<double>();
  waveform_scale_ = weights_.Meta("waveform_scale").get<double>();
  log_floor_ = weights_.Meta("log_floor").get<double>();
  layer_norm_epsilon_ = weights_.Meta("ln_eps").get<float>();
  variance_floor_ = weights_.Meta("pool_var_floor").get<float>();
  const auto frontend = npz::Load(encoder_directory / "frontend.npz");
  window_ = ToDouble(frontend.at("frontend.window"));
  const mx::array &mel_banks = frontend.at("frontend.mel_banks");
  const std::vector<double> mel_flat = ToDouble(mel_banks);
  const int bins = mel_banks.shape(1);
  if (mel_banks.shape(0) != mel_count_ || bins != fft_size_ / 2 + 1 ||
      static_cast<int>(window_.size()) != frame_length_) {
    throw std::runtime_error(encoder_directory.string() +
                             ": frontend tables do not fit the encoder");
  } else {
  }
  for (int mel = 0; mel < mel_count_; ++mel) {
    mel_banks_.emplace_back(mel_flat.begin() + mel * bins,
                            mel_flat.begin() + (mel + 1) * bins);
  }
  resample_kernel_ = mx::astype(frontend.at("frontend.resample_44100"),
                                mx::float64, mx::Device::cpu);
  mx::eval(resample_kernel_);
  // Grouped 3x3 convolutions: slots w0..wn [out, 1, 3, 3] -> [out, 3, 3, n].
  for (size_t stage = 0; stage < MectB2().size(); ++stage) {
    for (size_t block = 0; block < MectB2()[stage].size(); ++block) {
      for (int residual = 0; residual < MectB2()[stage][block].residual_count;
           ++residual) {
        for (const char *convolution : {"conv1", "conv2"}) {
          const std::string prefix =
              "fe.b" + std::to_string(stage + 1) + "." + std::to_string(block) +
              ".cnn." + std::to_string(residual) + "." + convolution;
          std::vector<mx::array> slots;
          while (
              weights_.HasTable(prefix + ".w" + std::to_string(slots.size()))) {
            slots.push_back(
                weights_.Table(prefix + ".w" + std::to_string(slots.size())));
          }
          if (slots.empty()) {
            throw std::runtime_error("speaker encoder is missing " + prefix);
          } else {
          }
          grouped_.emplace(
              prefix, mx::transpose(mx::concatenate(slots, 1), {0, 2, 3, 1}));
        }
      }
    }
  }
  std::vector<mx::array> grouped;
  for (const auto &[prefix, weight] : grouped_) {
    grouped.push_back(weight);
  }
  mx::eval(grouped);
}

std::vector<float> SpeakerEncoder::Fbank(const std::vector<float> &samples,
                                         int &frame_count) const {
  const long long length = std::min<long long>(samples.size(), kMaxSamples16k);
  if (length < frame_length_) {
    throw std::invalid_argument("speaker audio is shorter than 25 ms");
  } else {
  }
  frame_count = static_cast<int>(1 + (length - frame_length_) / frame_shift_);
  const int bins = fft_size_ / 2 + 1;
  std::vector<float> features(static_cast<size_t>(frame_count) * mel_count_);
  std::vector<double> frame(frame_length_);
  for (int t = 0; t < frame_count; ++t) {
    for (int i = 0; i < frame_length_; ++i) {
      frame[i] =
          static_cast<double>(samples[t * frame_shift_ + i]) * waveform_scale_;
    }
    const double mean =
        PairwiseSum(frame.data(), frame_length_) / frame_length_;
    for (double &value : frame) {
      value -= mean;
    }
    std::vector<double> emphasized(frame_length_);
    for (int i = 0; i < frame_length_; ++i) {
      const double previous = frame[i == 0 ? 0 : i - 1];
      emphasized[i] = (frame[i] - preemphasis_ * previous) * window_[i];
    }
    const std::vector<std::complex<double>> spectrum =
        RealFft(emphasized, fft_size_);
    std::vector<double> power(bins);
    for (int bin = 0; bin < bins; ++bin) {
      power[bin] = spectrum[bin].real() * spectrum[bin].real() +
                   spectrum[bin].imag() * spectrum[bin].imag();
    }
    for (int mel = 0; mel < mel_count_; ++mel) {
      double energy = 0;
      for (int bin = 0; bin < bins; ++bin) {
        energy += power[bin] * mel_banks_[mel][bin];
      }
      features[static_cast<size_t>(t) * mel_count_ + mel] =
          static_cast<float>(std::log(std::max(energy, log_floor_)));
    }
  }
  return features;
}

mx::array SpeakerEncoder::BatchNorm(const mx::array &x,
                                    const std::string &prefix) const {
  return mx::add(mx::multiply(x, weights_.Table(prefix + ".scale")),
                 weights_.Table(prefix + ".shift"));
}

mx::array SpeakerEncoder::Pointwise(const mx::array &x,
                                    const std::string &prefix,
                                    bool has_bias) const {
  const mx::array &weight = weights_.Table(prefix + ".weight");
  const mx::array y = mx::matmul(
      x,
      mx::transpose(mx::reshape(weight, {weight.shape(0), weight.shape(1)})));
  return has_bias ? mx::add(y, weights_.Table(prefix + ".bias")) : y;
}

mx::array SpeakerEncoder::Linear(const mx::array &x,
                                 const std::string &prefix) const {
  return mx::add(
      mx::matmul(x, mx::transpose(weights_.Table(prefix + ".weight"))),
      weights_.Table(prefix + ".bias"));
}

mx::array SpeakerEncoder::LayerNorm(const mx::array &x,
                                    const std::string &prefix) const {
  return mx::fast::layer_norm(x, weights_.Table(prefix + ".weight"),
                              weights_.Table(prefix + ".bias"),
                              layer_norm_epsilon_);
}

mx::array SpeakerEncoder::Grouped(const mx::array &x, const std::string &prefix,
                                  int frequency_stride, int time_stride) const {
  const mx::array &weight = grouped_.at(prefix);
  const int groups = x.shape(3) / weight.shape(3);
  return mx::conv2d(x, weight, {frequency_stride, time_stride}, {1, 1}, {1, 1},
                    groups);
}

mx::array SpeakerEncoder::ResidualBlock(const mx::array &x,
                                        const std::string &prefix,
                                        int frequency_stride,
                                        int time_stride) const {
  mx::array t = Grouped(x, prefix + "conv1", frequency_stride, time_stride);
  t = mx::maximum(BatchNorm(Pointwise(t, prefix + "pw1", true), prefix + "bn1"),
                  Scalar(0.0f));
  t = Grouped(t, prefix + "conv2", 1, 1);
  t = mx::maximum(BatchNorm(Pointwise(t, prefix + "pw2", true), prefix + "bn2"),
                  Scalar(0.0f));
  mx::array shortcut = x;
  if (weights_.HasTable(prefix + "sc.weight")) {
    const mx::array strided = mx::slice(x, {0, 0, 0, 0}, x.shape(),
                                        {1, frequency_stride, time_stride, 1});
    shortcut =
        BatchNorm(Pointwise(strided, prefix + "sc", false), prefix + "sc.bn");
  } else {
  }
  return mx::add(t, shortcut);
}

mx::array SpeakerEncoder::Attention(const mx::array &hidden,
                                    const std::string &prefix) const {
  const int frames = hidden.shape(0);
  const int head_size = kHiddenSize / head_count_;
  const auto heads = [&](const mx::array &z) {
    return mx::expand_dims(
        mx::transpose(mx::reshape(z, {frames, head_count_, head_size}),
                      {1, 0, 2}),
        0);
  };
  const mx::array queries = heads(
      mx::multiply(Linear(hidden, prefix + "attn.q"),
                   Scalar(static_cast<float>(
                       1.0 / std::sqrt(static_cast<double>(head_size))))));
  const mx::array keys = heads(Linear(hidden, prefix + "attn.k"));
  const mx::array values = heads(Linear(hidden, prefix + "attn.v"));
  const mx::array attended = mx::reshape(
      mx::transpose(mx::squeeze(mx::fast::scaled_dot_product_attention(
                                    queries, keys, values, 1.0f),
                                0),
                    {1, 0, 2}),
      {frames, kHiddenSize});
  return Linear(attended, prefix + "attn.o");
}

mx::array SpeakerEncoder::Experts(const mx::array &hidden,
                                  const std::string &prefix) const {
  const mx::array routes = mx::softmax(
      mx::matmul(hidden,
                 mx::transpose(weights_.Table(prefix + "moe.gate.weight"))),
      std::vector<int>{-1});
  const mx::array experts = mx::reshape(
      NewGelu(mx::matmul(hidden, mx::transpose(weights_.Table(
                                     prefix + "moe.experts.weight")))),
      {hidden.shape(0), -1, kHiddenSize});
  const mx::array mixed =
      mx::sum(mx::multiply(experts, mx::expand_dims(routes, -1)), 1);
  return Linear(NewGelu(mixed), prefix + "ffn_out");
}

mx::array SpeakerEncoder::PositionConvolution(const mx::array &hidden,
                                              const std::string &prefix) const {
  const int pad = position_kernel_ / 2;
  const mx::array &weight = weights_.Table(prefix + "pos.dw.weight");
  const mx::array padded = mx::concatenate(
      {mx::zeros({pad, kHiddenSize}), hidden, mx::zeros({pad, kHiddenSize})},
      0);
  mx::array accumulated =
      mx::squeeze(mx::conv1d(mx::expand_dims(padded, 0),
                             mx::expand_dims(weight, -1), 1, 0, 1, kHiddenSize),
                  0);
  accumulated = mx::add(accumulated, weights_.Table(prefix + "pos.dw.bias"));
  accumulated = mx::add(
      mx::multiply(accumulated, weights_.Table(prefix + "pos.bn.scale")),
      weights_.Table(prefix + "pos.bn.shift"));
  return Linear(ErfGelu(accumulated), prefix + "pos.pw");
}

std::pair<mx::array, mx::array>
SpeakerEncoder::Block(const mx::array &x, const mx::array &flat,
                      const BlockShape &shape,
                      const std::string &prefix) const {
  const int frames = x.shape(2);
  mx::array y = x;
  for (int residual = 0; residual < shape.residual_count; ++residual) {
    y = ResidualBlock(y, prefix + "cnn." + std::to_string(residual) + ".",
                      residual == 0 ? shape.frequency_stride : 1,
                      residual == 0 ? shape.time_stride : 1);
  }
  if (weights_.HasTable(prefix + "reduce.weight")) {
    y = BatchNorm(Pointwise(y, prefix + "reduce", true), prefix + "reduce.bn");
  } else {
  }
  const int frequencies = y.shape(1);
  const int channels = y.shape(3);
  const mx::array reduced = Flatten(y);
  mx::array h =
      LayerNorm(Linear(reduced, prefix + "rproj"), prefix + "rproj.ln");
  h = mx::add(h, PositionConvolution(h, prefix));
  h = LayerNorm(mx::add(h, Attention(h, prefix)), prefix + "ln1");
  h = LayerNorm(mx::add(h, Experts(h, prefix)), prefix + "ln2");
  mx::array expanded = mx::add(Linear(h, prefix + "eproj"), reduced);
  if (shape.time_stride > 1) {
    expanded = mx::repeat(expanded, shape.time_stride, 0);
    expanded = mx::slice(expanded, {0, 0}, {frames, expanded.shape(1)});
  } else {
  }
  const mx::array output = mx::add(expanded, flat);
  return {Unflatten(output, frequencies, channels), output};
}

std::vector<float>
SpeakerEncoder::Embed(const std::vector<float> &samples) const {
  if (samples.empty()) {
    throw std::invalid_argument("reference audio must not be empty");
  } else {
  }
  const long long length = samples.size();
  const long long total = (kOutputRate * length + kInputRate - 1) / kInputRate;
  const long long blocks = (total + kOutputRate - 1) / kOutputRate;
  const int taps = 2 * kResampleWidth + kInputRate;
  const long long pad_right = std::max<long long>(
      0, (blocks - 1) * kInputRate + taps - (kResampleWidth + length));
  const mx::StreamOrDevice cpu = mx::Device::cpu;
  const mx::array input = mx::astype(
      mx::array(samples.data(), {static_cast<int>(length)}), mx::float64, cpu);
  const mx::array padded =
      mx::pad(input, {{kResampleWidth, static_cast<int>(pad_right)}},
              mx::array(0.0), "constant", cpu);
  const mx::array windows = mx::as_strided(
      padded, {static_cast<int>(blocks), taps}, {kInputRate, 1}, 0, cpu);
  mx::array resampled = mx::astype(
      mx::slice(
          mx::reshape(
              mx::matmul(windows, mx::transpose(resample_kernel_, cpu), cpu),
              {-1}, cpu),
          {0}, {static_cast<int>(total)}, cpu),
      mx::float32, cpu);
  mx::eval(resampled);
  const float *resampled_data = resampled.data<float>();
  const std::vector<float> samples16k(resampled_data,
                                      resampled_data + resampled.size());

  int frame_count = 0;
  const std::vector<float> features = Fbank(samples16k, frame_count);
  // Cepstral mean normalization in float64, summed over frames in order.
  std::vector<double> means(mel_count_, 0.0);
  for (int t = 0; t < frame_count; ++t) {
    for (int mel = 0; mel < mel_count_; ++mel) {
      means[mel] += static_cast<double>(features[t * mel_count_ + mel]);
    }
  }
  for (double &mean : means) {
    mean /= frame_count;
  }
  std::vector<float> normalized(features.size());
  for (int t = 0; t < frame_count; ++t) {
    for (int mel = 0; mel < mel_count_; ++mel) {
      normalized[t * mel_count_ + mel] = static_cast<float>(
          static_cast<double>(features[t * mel_count_ + mel]) - means[mel]);
    }
  }
  // [1, mels, frames, 1].
  mx::array x = mx::reshape(
      mx::transpose(mx::array(normalized.data(), {frame_count, mel_count_}),
                    {1, 0}),
      {1, mel_count_, frame_count, 1});
  const mx::array head_weight =
      mx::transpose(weights_.Table("fe.head.conv.weight"), {0, 2, 3, 1});
  x = BatchNorm(mx::add(mx::conv2d(x, head_weight, {1, 1}, {1, 1}),
                        weights_.Table("fe.head.conv.bias")),
                "fe.head.bn");
  mx::array flat = Flatten(x);
  std::vector<mx::array> stage_features;
  for (size_t stage = 0; stage < MectB2().size(); ++stage) {
    for (size_t block = 0; block < MectB2()[stage].size(); ++block) {
      const StageBlock &shape = MectB2()[stage][block];
      auto [next, next_flat] = Block(
          x, flat,
          {shape.residual_count, shape.frequency_stride, shape.time_stride},
          "fe.b" + std::to_string(stage + 1) + "." + std::to_string(block) +
              ".");
      x = next;
      flat = next_flat;
      mx::eval(x, flat);
    }
    stage_features.push_back(flat);
  }
  const mx::array &dense = weights_.Table("fe.dense_w");
  mx::array fused = mx::zeros_like(stage_features[0]);
  for (size_t stage = 0; stage < stage_features.size(); ++stage) {
    const mx::array weighted = mx::multiply(
        stage_features[stage],
        mx::squeeze(mx::slice(dense, {static_cast<int>(stage), 0},
                              {static_cast<int>(stage) + 1, dense.shape(1)}),
                    0));
    fused = stage == 0 ? weighted : mx::add(fused, weighted);
  }
  const mx::array attention =
      Linear(mx::tanh(Linear(fused, "pool.l1")), "pool.l2");
  const mx::array alpha = mx::softmax(attention, std::vector<int>{0});
  const mx::array mean = mx::sum(mx::multiply(alpha, fused), 0);
  const mx::array variance =
      mx::subtract(mx::sum(mx::multiply(mx::multiply(alpha, fused), fused), 0),
                   mx::multiply(mean, mean));
  const mx::array deviation =
      mx::sqrt(mx::maximum(variance, Scalar(variance_floor_)));
  const mx::array pooled = mx::concatenate({mean, deviation}, 0);
  mx::array embedding =
      mx::matmul(mx::add(mx::multiply(pooled, weights_.Table("bn.scale")),
                         weights_.Table("bn.shift")),
                 mx::transpose(weights_.Table("linear.weight")));
  embedding = mx::astype(embedding, mx::float32);
  mx::eval(embedding);
  const float *data = embedding.data<float>();
  std::vector<float> out(data, data + embedding.size());
  for (const float value : out) {
    if (!std::isfinite(value)) {
      throw std::invalid_argument(
          "speaker encoder produced a non-finite embedding");
    } else {
    }
  }
  return out;
}

} // namespace zonos3
