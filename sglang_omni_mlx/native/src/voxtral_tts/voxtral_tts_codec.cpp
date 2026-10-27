// SPDX-License-Identifier: Apache-2.0
#include "voxtral_tts/voxtral_tts_codec.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace voxtral_tts {

namespace mx = mlx::core;

namespace {

constexpr const char *kCodecPrefix = "audio_tokenizer";
// Note (Minyi Tang): floors from the reference codec: a codeword's usage
// count, and a weight-normed kernel's norm.
constexpr float kClusterUsageFloor = 1e-5f;
constexpr float kWeightNormFloor = 1e-12f;
// Note (Minyi Tang): the CUDA vocoder's schedule (models/voxtral_tts/
// vocoder.py): the first frame repeated twice before the codes, then windows.
constexpr int kWarmupFrames = 2;
constexpr int kInitialChunkFrames = 5;
constexpr int kChunkFrames = 25;
constexpr int kLeftContextFrames = 25;
constexpr int kFadeInMilliseconds = 10;
constexpr float kPcm16FullScale = 32767.0f;

std::vector<int> ParseIntegers(const std::string &text) {
  std::vector<int> values;
  std::stringstream stream(text);
  std::string item;
  while (std::getline(stream, item, ',')) {
    values.push_back(std::stoi(item));
  }
  return values;
}

// A causal convolution: the input padded on the left by kernel minus stride
// steps, in mode, and on the right up to a whole number of strides.
mx::array CausalConvolution(const mx::array &x, const mx::array &weight,
                            int stride, const std::string &mode) {
  const int length = x.shape(1);
  const int left = weight.shape(1) - stride;
  const int right = (length + stride - 1) / stride * stride - length;
  return mx::conv1d(mx::pad(x, std::vector<int>{1}, mx::Shape{left},
                            mx::Shape{right}, mx::array(0), mode),
                    weight, stride);
}

} // namespace

AudioCodec::AudioCodec(omni_common::WeightMap weights,
                       const nlohmann::json &arguments)
    : blocks_(BuildBlocks(arguments)),
      checkpoint_(PrepareWeights(std::move(weights), blocks_),
                  nlohmann::json::object()),
      dimension_(arguments.at("dim").get<int>()),
      head_count_(arguments.at("n_heads").get<int>()),
      head_dim_(arguments.at("head_dim").get<int>()),
      norm_eps_(arguments.at("norm_eps").get<float>()),
      query_key_norm_eps_(arguments.at("qk_norm_eps").get<float>()),
      acoustic_levels_(arguments.at("acoustic_codebook_size").get<int>()),
      samples_per_frame_(arguments.at("pretransform_patch_size").get<int>()),
      sample_rate_(arguments.at("sampling_rate").get<int>()),
      alibi_slopes_(0.0f) {
  if (head_count_ != arguments.at("n_kv_heads").get<int>() ||
      (head_count_ & (head_count_ - 1)) != 0) {
    throw std::runtime_error("the Voxtral codec needs one key/value head per "
                             "query head and a power-of-two head count");
  } else {
  }
  for (const Block &block : blocks_) {
    samples_per_frame_ *= block.stride;
  }
  // ALiBi's slopes for a power-of-two head count: 2^(-8/heads) to the power
  // of each head's index.
  const double base = std::pow(2.0, -8.0 / head_count_);
  std::vector<float> slopes;
  for (int head = 0; head < head_count_; ++head) {
    slopes.push_back(static_cast<float>(std::pow(base, head)));
  }
  alibi_slopes_ = mx::array(slopes.begin(), {head_count_}, mx::float32);
}

std::vector<AudioCodec::Block>
AudioCodec::BuildBlocks(const nlohmann::json &arguments) {
  const std::vector<int> kernels =
      ParseIntegers(arguments.at("decoder_convs_kernels_str"));
  const std::vector<int> strides =
      ParseIntegers(arguments.at("decoder_convs_strides_str"));
  const std::vector<int> lengths =
      ParseIntegers(arguments.at("decoder_transformer_lengths_str"));
  if (kernels.size() != lengths.size() || strides.size() != lengths.size() ||
      lengths.empty()) {
    throw std::runtime_error("the Voxtral codec's decoder lists disagree");
  } else {
  }
  const int window_frames = arguments.at("attn_sliding_window_size").get<int>();
  const bool halves_window =
      arguments.at("half_attn_window_upon_downsampling").get<bool>();
  std::vector<Block> blocks = {
      {BlockKind::kConvolution, kernels[0], strides[0], 0, 0}};
  for (size_t stage = 0; stage < lengths.size(); ++stage) {
    // The window is set at the waveform's rate and halves with each
    // upsampling still ahead.
    const int upsamplings_ahead = static_cast<int>(
        std::count_if(strides.begin() + stage + 1, strides.end(),
                      [](int stride) { return stride > 1; }));
    blocks.push_back(
        {BlockKind::kTransformer, 0, 1, lengths[stage],
         halves_window ? window_frames >> upsamplings_ahead : window_frames});
    if (stage + 1 < lengths.size()) {
      blocks.push_back({BlockKind::kTransposedConvolution, kernels[stage + 1],
                        strides[stage + 1], 0, 0});
    } else {
    }
  }
  return blocks;
}

omni_common::WeightMap
AudioCodec::PrepareWeights(omni_common::WeightMap weights,
                           const std::vector<Block> &blocks) {
  const std::string prefix = kCodecPrefix;
  const std::string codebook = prefix + ".quantizer.semantic_codebook.";
  const mx::array usage = weights.at(codebook + "cluster_usage");
  weights.insert_or_assign(
      prefix + ".semantic_embedding",
      mx::divide(weights.at(codebook + "embedding_sum"),
                 mx::maximum(mx::expand_dims(usage, 1),
                             mx::array(kClusterUsageFloor, usage.dtype()))));
  weights.erase(codebook + "cluster_usage");
  weights.erase(codebook + "embedding_sum");
  // Folds a weight-normed kernel (out, in, kernel), or (in, out, kernel) when
  // transposed, into MLX's (out, kernel, in).
  const auto fold = [&](const std::string &module, bool transposed) {
    const std::string stem = module + ".conv.parametrizations.weight.";
    const mx::array direction = weights.at(stem + "original1");
    const mx::array direction32 = mx::astype(direction, mx::float32);
    const mx::array norm =
        mx::sqrt(mx::sum(mx::multiply(direction32, direction32), {1, 2}, true));
    const mx::array weight = mx::astype(
        mx::divide(
            mx::multiply(direction32, mx::astype(weights.at(stem + "original0"),
                                                 mx::float32)),
            mx::maximum(norm, mx::array(kWeightNormFloor))),
        direction.dtype());
    weights.insert_or_assign(
        module + ".conv.weight",
        mx::transpose(weight, transposed ? std::vector<int>{1, 2, 0}
                                         : std::vector<int>{0, 2, 1}));
    weights.erase(stem + "original0");
    weights.erase(stem + "original1");
  };
  for (size_t index = 0; index < blocks.size(); ++index) {
    if (blocks[index].kind != BlockKind::kTransformer) {
      fold(prefix + ".decoder_blocks." + std::to_string(index),
           blocks[index].kind == BlockKind::kTransposedConvolution);
    } else {
    }
  }
  fold(prefix + ".output_proj", false);
  return weights;
}

mx::array AudioCodec::Decode(const mx::array &codes) const {
  const std::string prefix = kCodecPrefix;
  const int frame_count = codes.shape(0);
  const int codebook_count = codes.shape(1);
  const mx::array shifted =
      mx::subtract(codes, mx::array(kSpecialAudioCodeCount));
  const mx::array &output_weight =
      checkpoint_.Weight(prefix + ".output_proj.conv.weight");
  const mx::Dtype dtype = output_weight.dtype();
  const mx::array semantic = mx::astype(
      mx::take(checkpoint_.Weight(prefix + ".semantic_embedding"),
               mx::reshape(mx::slice(shifted, {0, 0}, {frame_count, 1}),
                           {frame_count}),
               0),
      dtype);
  // Acoustic codes are levels evenly spaced over [-1, 1].
  const mx::array acoustic = mx::astype(
      mx::subtract(
          mx::multiply(
              mx::astype(
                  mx::slice(shifted, {0, 1}, {frame_count, codebook_count}),
                  mx::float32),
              mx::array(static_cast<float>(2.0 / (acoustic_levels_ - 1)))),
          mx::array(1.0f)),
      dtype);
  mx::array hidden =
      mx::expand_dims(mx::concatenate({semantic, acoustic}, 1), 0);
  for (size_t index = 0; index < blocks_.size(); ++index) {
    const Block &block = blocks_[index];
    const std::string block_prefix =
        prefix + ".decoder_blocks." + std::to_string(index);
    if (block.kind == BlockKind::kConvolution) {
      hidden = CausalConvolution(
          hidden, checkpoint_.Weight(block_prefix + ".conv.weight"),
          block.stride, "edge");
    } else if (block.kind == BlockKind::kTransposedConvolution) {
      hidden = mx::conv_transpose1d(
          hidden, checkpoint_.Weight(block_prefix + ".conv.weight"),
          block.stride);
      // Causal: drops the trailing kernel minus stride steps.
      hidden =
          mx::slice(hidden, {0, 0, 0},
                    {1, hidden.shape(1) - (block.kernel_size - block.stride),
                     dimension_});
    } else {
      for (int layer = 0; layer < block.layer_count; ++layer) {
        hidden = TransformerLayer(
            hidden, block_prefix + ".layers." + std::to_string(layer),
            block.window_frames);
      }
    }
  }
  return mx::reshape(CausalConvolution(hidden, output_weight, 1, "reflect"),
                     {-1});
}

mx::array AudioCodec::TransformerLayer(const mx::array &x,
                                       const std::string &prefix,
                                       int window_frames) const {
  const int length = x.shape(1);
  const mx::Dtype dtype = x.dtype();
  const auto split_heads = [&](const mx::array &projected) {
    return mx::transpose(
        mx::reshape(projected, {1, length, head_count_, head_dim_}),
        {0, 2, 1, 3});
  };
  const mx::array normed =
      checkpoint_.RmsNorm(x, prefix + ".attention_norm", norm_eps_);
  const mx::array queries = split_heads(
      checkpoint_.RmsNorm(checkpoint_.Linear(normed, prefix + ".attention.wq"),
                          prefix + ".attention.q_norm", query_key_norm_eps_));
  const mx::array keys = split_heads(
      checkpoint_.RmsNorm(checkpoint_.Linear(normed, prefix + ".attention.wk"),
                          prefix + ".attention.k_norm", query_key_norm_eps_));
  const mx::array values =
      split_heads(checkpoint_.Linear(normed, prefix + ".attention.wv"));
  // ALiBi: each head's slope times the key's offset from the query, over
  // the causal window only.
  const mx::array positions = mx::arange(length);
  const mx::array offsets = mx::subtract(mx::expand_dims(positions, 0),
                                         mx::expand_dims(positions, 1));
  const mx::array bias =
      mx::where(mx::logical_or(mx::greater(offsets, mx::array(0)),
                               mx::less(offsets, mx::array(-window_frames))),
                mx::array(-std::numeric_limits<float>::infinity(), dtype),
                mx::multiply(mx::reshape(mx::astype(alibi_slopes_, dtype),
                                         {head_count_, 1, 1}),
                             mx::astype(offsets, dtype)));
  const mx::array attended = mx::fast::scaled_dot_product_attention(
      queries, keys, values,
      static_cast<float>(std::pow(static_cast<double>(head_dim_), -0.5)),
      "array", mx::expand_dims(bias, 0));
  const mx::array hidden = mx::add(
      x, mx::multiply(checkpoint_.Weight(prefix + ".attention_scale"),
                      checkpoint_.Linear(
                          mx::reshape(mx::transpose(attended, {0, 2, 1, 3}),
                                      {1, length, head_count_ * head_dim_}),
                          prefix + ".attention.wo")));
  const mx::array feed_forward_input =
      checkpoint_.RmsNorm(hidden, prefix + ".ffn_norm", norm_eps_);
  return mx::add(
      hidden, mx::multiply(
                  checkpoint_.Weight(prefix + ".ffn_scale"),
                  checkpoint_.Linear(
                      mx::multiply(
                          omni_common::Silu(checkpoint_.Linear(
                              feed_forward_input, prefix + ".feed_forward.w1")),
                          checkpoint_.Linear(feed_forward_input,
                                             prefix + ".feed_forward.w3")),
                      prefix + ".feed_forward.w2")));
}

std::vector<float> CodecStream::Push(const mx::array &frame_codes) {
  frames_.push_back(frame_codes);
  const int new_frames = static_cast<int>(frames_.size()) - emitted_frames_;
  if (new_frames >=
      (emitted_frames_ == 0 ? kInitialChunkFrames : kChunkFrames)) {
    return DecodeWindow();
  } else {
    return {};
  }
}

std::vector<float> CodecStream::Finish() {
  std::vector<float> samples;
  while (emitted_frames_ < static_cast<int>(frames_.size())) {
    const std::vector<float> window = DecodeWindow();
    samples.insert(samples.end(), window.begin(), window.end());
  }
  return samples;
}

std::vector<float> CodecStream::DecodeWindow() {
  const int total_frames = std::min(static_cast<int>(frames_.size()),
                                    emitted_frames_ + 2 * kChunkFrames);
  // Positions in the warmup frames followed by the frames.
  const int start =
      std::max(0, emitted_frames_ + kWarmupFrames - kLeftContextFrames);
  std::vector<mx::array> window;
  for (int position = start; position < total_frames + kWarmupFrames;
       ++position) {
    window.push_back(frames_[std::max(0, position - kWarmupFrames)]);
  }
  const int samples_per_frame = codec_.samples_per_frame();
  const int first_sample =
      (emitted_frames_ + kWarmupFrames - start) * samples_per_frame;
  const int new_frames = total_frames - emitted_frames_;
  const int sample_count = new_frames * samples_per_frame;
  // Note (Minyi Tang): the fade runs on float32 samples, as the CUDA
  // vocoder's does; a bfloat16 fade rounds the first samples differently.
  mx::array audio =
      mx::astype(mx::slice(codec_.Decode(mx::stack(window, 0)), {first_sample},
                           {first_sample + sample_count}),
                 mx::float32);
  if (emitted_frames_ == 0) {
    const int fade_samples = std::min(
        codec_.sample_rate() * kFadeInMilliseconds / 1000, sample_count);
    audio = mx::concatenate(
        {mx::multiply(mx::slice(audio, {0}, {fade_samples}),
                      mx::linspace(0.0, 1.0, fade_samples, mx::float32)),
         mx::slice(audio, {fade_samples}, {sample_count})},
        0);
  } else {
  }
  mx::eval(audio);
  emitted_frames_ = total_frames;
  return std::vector<float>(audio.data<float>(),
                            audio.data<float>() + audio.size());
}

std::string Pcm16Bytes(const std::vector<float> &samples) {
  std::string bytes;
  bytes.reserve(samples.size() * 2);
  for (const float sample : samples) {
    // Note (Minyi Tang): NaN becomes silence, since std::clamp passes it
    // through and casting NaN to an integer is undefined.
    const float bounded =
        std::isnan(sample) ? 0.0f : std::clamp(sample, -1.0f, 1.0f);
    const auto value = static_cast<int16_t>(bounded * kPcm16FullScale);
    bytes.push_back(static_cast<char>(value & 0xff));
    bytes.push_back(static_cast<char>((value >> 8) & 0xff));
  }
  return bytes;
}

std::string WavBytes(const std::string &pcm, int sample_rate) {
  std::string bytes;
  const auto append = [&](uint32_t value, int size) {
    for (int index = 0; index < size; ++index) {
      bytes.push_back(static_cast<char>((value >> (8 * index)) & 0xff));
    }
  };
  const auto data_size = static_cast<uint32_t>(pcm.size());
  bytes += "RIFF";
  append(36 + data_size, 4);
  bytes += "WAVEfmt ";
  append(16, 4);              // fmt chunk size
  append(1, 2);               // PCM
  append(1, 2);               // mono
  append(sample_rate, 4);     // samples per second
  append(sample_rate * 2, 4); // bytes per second
  append(2, 2);               // bytes per sample frame
  append(16, 2);              // bits per sample
  bytes += "data";
  append(data_size, 4);
  return bytes + pcm;
}

} // namespace voxtral_tts
