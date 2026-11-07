// SPDX-License-Identifier: Apache-2.0
#include "qwen3_tts/qwen3_tts_codec.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <optional>
#include <stdexcept>

namespace qwen3_tts {

namespace mx = mlx::core;

namespace {

constexpr const char *kPrefix = "decoder";
constexpr float kCodebookUsageEps = 1e-5f;
constexpr float kSnakeEps = 1e-9f;
constexpr float kConvNextNormEps = 1e-6f;
constexpr std::array<int, 3> kResidualDilations = {1, 3, 9};
// The other half of each head, the first half negated.
mx::array RotateHalf(const mx::array &x) {
  const int half = x.shape(-1) / 2;
  const mx::Shape &shape = x.shape();
  mx::Shape stop = shape;
  stop.back() = half;
  mx::Shape start(shape.size(), 0);
  const mx::array first = mx::slice(x, start, stop);
  start.back() = half;
  const mx::array second = mx::slice(x, start, shape);
  return mx::concatenate({mx::negative(second), first}, -1);
}

bool EndsWith(const std::string &text, const std::string &suffix) {
  return text.size() >= suffix.size() &&
         text.compare(text.size() - suffix.size(), suffix.size(), suffix) == 0;
}

} // namespace

omni_common::WeightMap
SpeechDecoder::PrepareWeights(omni_common::WeightMap weights) {
  omni_common::WeightMap prepared;
  const std::string usage_suffix = "._codebook.cluster_usage";
  for (auto &[name, array] : weights) {
    if (name.rfind(std::string(kPrefix) + ".", 0) != 0) {
      continue;
    } else if (EndsWith(name, usage_suffix)) {
      const std::string base =
          name.substr(0, name.size() - usage_suffix.size());
      const mx::array &sums = weights.at(base + "._codebook.embedding_sum");
      prepared.insert_or_assign(
          base + ".codebook.embed.weight",
          mx::divide(sums,
                     mx::expand_dims(
                         mx::maximum(array, mx::array(kCodebookUsageEps)), 1)));
      continue;
    } else if (EndsWith(name, "._codebook.embedding_sum")) {
      continue;
    } else {
    }
    // Checkpoint kernels are [out, in, kernel], and a transposed
    // convolution's [in, out, kernel]; MLX wants [out, kernel, in].
    const bool transposed =
        array.ndim() == 3 && ((name.find(".upsample.") != std::string::npos &&
                               EndsWith(name, ".0.conv.weight")) ||
                              EndsWith(name, ".block.1.conv.weight"));
    if (transposed) {
      prepared.insert_or_assign(name, mx::transpose(array, {1, 2, 0}));
    } else if (array.ndim() == 3) {
      prepared.insert_or_assign(name, mx::transpose(array, {0, 2, 1}));
    } else {
      prepared.insert_or_assign(name, array);
    }
  }
  return prepared;
}

SpeechDecoder::SpeechDecoder(
    const std::filesystem::path &speech_tokenizer_directory,
    bool incremental_streaming)
    : SpeechDecoder(
          omni_common::ReadJson(speech_tokenizer_directory / "config.json"),
          omni_common::LoadSafetensors(speech_tokenizer_directory),
          incremental_streaming) {}

SpeechDecoder::SpeechDecoder(const nlohmann::json &config,
                             omni_common::WeightMap weights,
                             bool incremental_streaming)
    : checkpoint_(PrepareWeights(std::move(weights)), nlohmann::json::object()),
      incremental_streaming_(incremental_streaming),
      inverse_frequencies_(0.0f) {
  const nlohmann::json &decoder = config.at("decoder_config");
  codebook_count_ = decoder.at("num_quantizers").get<int>();
  semantic_codebook_count_ = decoder.at("num_semantic_quantizers").get<int>();
  head_count_ = decoder.at("num_attention_heads").get<int>();
  head_dim_ = decoder.at("head_dim").get<int>();
  transformer_layer_count_ = decoder.at("num_hidden_layers").get<int>();
  rms_norm_eps_ = decoder.at("rms_norm_eps").get<float>();
  sliding_window_frames_ = decoder.at("sliding_window").get<int>();
  upsampling_ratios_ = decoder.at("upsampling_ratios").get<std::vector<int>>();
  upsample_rates_ = decoder.at("upsample_rates").get<std::vector<int>>();
  samples_per_frame_ = config.at("decode_upsample_rate").get<int>();
  sample_rate_ = config.at("output_sample_rate").get<int>();
  if (decoder.at("num_key_value_heads").get<int>() != head_count_) {
    throw std::runtime_error("the speech decoder's attention must have as "
                             "many key/value heads as query heads");
  } else {
  }
  const float rope_theta = decoder.at("rope_theta").get<float>();
  inverse_frequencies_ = mx::divide(
      mx::array(1.0f),
      mx::power(mx::array(rope_theta),
                mx::divide(mx::arange(0.0, static_cast<double>(head_dim_), 2.0,
                                      mx::float32),
                           mx::array(static_cast<float>(head_dim_)))));
  mx::eval(inverse_frequencies_);
}

std::optional<SpeechDecoderState> SpeechDecoder::NewStreamState() const {
  if (incremental_streaming_) {
    SpeechDecoderState state;
    state.attention.resize(transformer_layer_count_);
    return state;
  } else {
    return std::nullopt;
  }
}

mx::array SpeechDecoder::RmsNorm(const mx::array &x,
                                 const std::string &prefix) const {
  const mx::array x_float = mx::astype(x, mx::float32);
  const mx::array variance =
      mx::mean(mx::power(x_float, mx::array(2.0f)), -1, true);
  const mx::array normed = mx::multiply(
      x_float, mx::rsqrt(mx::add(variance, mx::array(rms_norm_eps_))));
  return mx::astype(
      mx::multiply(checkpoint_.Weight(prefix + ".weight"), normed), x.dtype());
}

mx::array SpeechDecoder::Snake(const mx::array &x,
                               const std::string &prefix) const {
  const mx::array alpha = mx::exp(checkpoint_.Weight(prefix + ".alpha"));
  const mx::array beta = mx::exp(checkpoint_.Weight(prefix + ".beta"));
  return mx::add(
      x, mx::multiply(
             mx::divide(mx::array(1.0f), mx::add(beta, mx::array(kSnakeEps))),
             mx::power(mx::sin(mx::multiply(x, alpha)), mx::array(2.0f))));
}

mx::array SpeechDecoder::CausalConvolution(const mx::array &x,
                                           const std::string &prefix,
                                           int dilation, int groups,
                                           SpeechDecoderState *state) const {
  const mx::array &weight = checkpoint_.Weight(prefix + ".weight");
  const int padding = (weight.shape(1) - 1) * dilation;
  mx::array padded = x;
  if (padding > 0) {
    padded =
        state && state->convolution_history.contains(prefix)
            ? mx::concatenate({state->convolution_history.at(prefix), x}, 1)
            : mx::pad(x, {{0, 0}, {padding, 0}, {0, 0}});
    if (state) {
      state->convolution_history.insert_or_assign(
          prefix, mx::copy(mx::slice(padded, {0, padded.shape(1) - padding, 0},
                                     padded.shape())));
    } else {
    }
  } else {
  }
  const mx::array y = mx::conv1d(padded, weight, 1, 0, dilation, groups);
  if (checkpoint_.Has(prefix + ".bias")) {
    return mx::add(y, checkpoint_.Weight(prefix + ".bias"));
  } else {
    return y;
  }
}

mx::array
SpeechDecoder::TransposedConvolution(const mx::array &x,
                                     const std::string &prefix, int stride,
                                     SpeechDecoderState *state) const {
  const mx::array &weight = checkpoint_.Weight(prefix + ".weight");
  const mx::array &bias = checkpoint_.Weight(prefix + ".bias");
  const int trim = weight.shape(1) - stride;
  mx::array input = x;
  int context_samples = 0;
  const bool resumes_block =
      state && state->decoded_frames % kStreamedChunkFrames != 0;
  const std::string input_key = prefix + ".input_tail";
  if (trim > 0 && state) {
    // Within a codec block, recompute the small overlap from its input.
    // Adding separately rounded output tails changes the PCM at a substep.
    if (resumes_block) {
      const mx::array &tail = state->convolution_history.at(input_key);
      input = mx::concatenate({tail, x}, 1);
      context_samples = tail.shape(1) * stride;
    } else {
    }
    const int keep = (trim + stride - 1) / stride;
    state->convolution_history.insert_or_assign(
        input_key, mx::copy(mx::slice(input, {0, input.shape(1) - keep, 0},
                                      input.shape())));
  } else {
  }
  mx::array y = mx::add(mx::conv_transpose1d(input, weight, stride), bias);
  if (context_samples > 0) {
    y = mx::slice(y, {0, context_samples, 0}, y.shape());
  } else {
  }
  if (trim > 0) {
    if (state) {
      auto &history = state->convolution_history;
      if (history.contains(prefix) && !resumes_block) {
        // Match the reference's overlap at its complete codec boundaries.
        const mx::Shape head = {y.shape(0), trim, y.shape(2)};
        y = mx::concatenate(
            {mx::add(mx::slice(y, {0, 0, 0}, head), history.at(prefix)),
             mx::slice(y, {0, trim, 0}, y.shape())},
            1);
      } else {
      }
      history.insert_or_assign(
          prefix, mx::copy(mx::slice(y, {0, y.shape(1) - trim, 0}, y.shape())));
    } else {
    }
    return mx::slice(y, {0, 0, 0}, {y.shape(0), y.shape(1) - trim, y.shape(2)});
  } else {
    return y;
  }
}

mx::array SpeechDecoder::TransformerLayer(const mx::array &x, int layer,
                                          const mx::array &cosines,
                                          const mx::array &sines,
                                          const mx::array &mask,
                                          SpeechDecoderState *state) const {
  const std::string prefix =
      std::string(kPrefix) + ".pre_transformer.layers." + std::to_string(layer);
  const int batch = x.shape(0);
  const int length = x.shape(1);
  const mx::array normed = RmsNorm(x, prefix + ".input_layernorm");
  const auto heads = [&](const std::string &projection) {
    return mx::transpose(
        mx::reshape(
            checkpoint_.Linear(normed, prefix + ".self_attn." + projection),
            {batch, length, head_count_, head_dim_}),
        {0, 2, 1, 3});
  };
  const auto rotate = [&](const mx::array &projected) {
    return mx::add(mx::multiply(projected, cosines),
                   mx::multiply(RotateHalf(projected), sines));
  };
  const mx::array queries = rotate(heads("q_proj"));
  mx::array keys = rotate(heads("k_proj"));
  mx::array values = heads("v_proj");
  if (state) {
    auto &history = state->attention[layer];
    if (history) {
      keys = mx::concatenate({history->first, keys}, 2);
      values = mx::concatenate({history->second, values}, 2);
    } else {
    }
    const int start = std::max(0, keys.shape(2) - (sliding_window_frames_ - 1));
    history.emplace(
        mx::contiguous(mx::slice(keys, {0, 0, start, 0}, keys.shape())),
        mx::contiguous(mx::slice(values, {0, 0, start, 0}, values.shape())));
  } else {
  }
  const mx::array attended = mx::fast::scaled_dot_product_attention(
      queries, keys, values,
      static_cast<float>(std::pow(static_cast<double>(head_dim_), -0.5)), "",
      mask);
  const mx::array attention_output = checkpoint_.Linear(
      mx::reshape(mx::transpose(attended, {0, 2, 1, 3}), {batch, length, -1}),
      prefix + ".self_attn.o_proj");
  const mx::array hidden = mx::add(
      x,
      mx::multiply(checkpoint_.Weight(prefix + ".self_attn_layer_scale.scale"),
                   attention_output));
  const mx::array mlp_input =
      RmsNorm(hidden, prefix + ".post_attention_layernorm");
  const mx::array mlp_output = checkpoint_.Linear(
      mx::multiply(omni_common::Silu(checkpoint_.Linear(
                       mlp_input, prefix + ".mlp.gate_proj")),
                   checkpoint_.Linear(mlp_input, prefix + ".mlp.up_proj")),
      prefix + ".mlp.down_proj");
  return mx::add(hidden, mx::multiply(checkpoint_.Weight(
                                          prefix + ".mlp_layer_scale.scale"),
                                      mlp_output));
}

mx::array SpeechDecoder::DecodeFeatures(const mx::array &codes,
                                        SpeechDecoderState *state) const {
  const std::string prefix = kPrefix;
  const int frame_count = codes.shape(0);
  // [1, frames] per codebook; each quantizer sums its codebooks' rows and
  // projects them back to the codec's dimension.
  const auto quantized = [&](const std::string &quantizer, int first_codebook,
                             int codebook_count) {
    const std::string quantizer_prefix = prefix + ".quantizer." + quantizer;
    std::optional<mx::array> sum;
    for (int codebook = 0; codebook < codebook_count; ++codebook) {
      const mx::array ids =
          mx::reshape(mx::slice(codes, {0, first_codebook + codebook},
                                {frame_count, first_codebook + codebook + 1}),
                      {1, frame_count});
      const mx::array rows = checkpoint_.Embed(
          ids, quantizer_prefix + ".vq.layers." + std::to_string(codebook) +
                   ".codebook.embed");
      sum = sum.has_value()
                ? mx::add(*sum, rows)
                : mx::add(mx::zeros(rows.shape(), mx::float32), rows);
    }
    return mx::conv1d(
        *sum, checkpoint_.Weight(quantizer_prefix + ".output_proj.weight"));
  };
  mx::array hidden = quantized("rvq_first", 0, semantic_codebook_count_);
  if (codebook_count_ > semantic_codebook_count_) {
    hidden =
        mx::add(hidden, quantized("rvq_rest", semantic_codebook_count_,
                                  codebook_count_ - semantic_codebook_count_));
  } else {
  }
  hidden = CausalConvolution(hidden, prefix + ".pre_conv.conv", 1, 1, state);

  const std::string transformer = prefix + ".pre_transformer";
  hidden = checkpoint_.Linear(hidden, transformer + ".input_proj");
  const int offset = state ? state->decoded_frames : 0;
  const mx::array positions =
      mx::astype(mx::arange(offset, offset + frame_count), mx::float32);
  const mx::array frequencies =
      mx::transpose(mx::multiply(mx::reshape(inverse_frequencies_, {1, -1, 1}),
                                 mx::reshape(positions, {1, 1, -1})),
                    {0, 2, 1});
  const mx::array angles = mx::concatenate({frequencies, frequencies}, -1);
  const mx::array cosines = mx::expand_dims(mx::cos(angles), 1);
  const mx::array sines = mx::expand_dims(mx::sin(angles), 1);
  // Note (Jiaxin Deng): the decoder's attention is causal within its
  // sliding window, as the CUDA vocoder's; a full causal mask changes the
  // samples of chunks longer than the window.
  const int cached_frames = std::min(offset, sliding_window_frames_ - 1);
  const mx::array queries =
      mx::expand_dims(mx::arange(offset, offset + frame_count), 1);
  const mx::array keys = mx::expand_dims(
      mx::arange(offset - cached_frames, offset + frame_count), 0);
  const mx::array masked = mx::logical_or(
      mx::less(queries, keys),
      mx::less_equal(keys,
                     mx::subtract(queries, mx::array(sliding_window_frames_))));
  const mx::array mask =
      mx::multiply(mx::astype(masked, mx::float32),
                   mx::array(std::numeric_limits<float>::lowest()));
  for (int layer = 0; layer < transformer_layer_count_; ++layer) {
    hidden = TransformerLayer(hidden, layer, cosines, sines, mask, state);
  }
  hidden = checkpoint_.Linear(RmsNorm(hidden, transformer + ".norm"),
                              transformer + ".output_proj");

  for (size_t stage = 0; stage < upsampling_ratios_.size(); ++stage) {
    const std::string stage_prefix =
        prefix + ".upsample." + std::to_string(stage);
    hidden = TransposedConvolution(hidden, stage_prefix + ".0.conv",
                                   upsampling_ratios_[stage], state);
    const std::string block = stage_prefix + ".1";
    const mx::array residual = hidden;
    mx::array x = CausalConvolution(hidden, block + ".dwconv.conv", 1,
                                    hidden.shape(2), state);
    x = checkpoint_.LayerNorm(x, block + ".norm", kConvNextNormEps);
    x = omni_common::Gelu(checkpoint_.Linear(x, block + ".pwconv1"));
    x = checkpoint_.Linear(x, block + ".pwconv2");
    hidden = mx::add(residual,
                     mx::multiply(checkpoint_.Weight(block + ".gamma"), x));
  }

  if (state) {
    state->decoded_frames += frame_count;
  } else {
  }
  return hidden;
}

mx::array SpeechDecoder::Decode(const mx::array &codes) const {
  return DecodeSamples(DecodeFeatures(codes, nullptr), codes.shape(0), nullptr);
}

mx::array SpeechDecoder::DecodeSamples(const mx::array &hidden, int frame_count,
                                       SpeechDecoderState *state) const {
  const std::string prefix = kPrefix;
  const std::string stack = prefix + ".decoder";
  mx::array wave = CausalConvolution(hidden, stack + ".0.conv", 1, 1, state);
  for (size_t stage = 0; stage < upsample_rates_.size(); ++stage) {
    const std::string block =
        stack + "." + std::to_string(stage + 1) + ".block";
    wave = Snake(wave, block + ".0");
    wave = TransposedConvolution(wave, block + ".1.conv",
                                 upsample_rates_[stage], state);
    for (size_t unit = 0; unit < kResidualDilations.size(); ++unit) {
      const std::string unit_prefix = block + "." + std::to_string(unit + 2);
      mx::array x = Snake(wave, unit_prefix + ".act1");
      x = CausalConvolution(x, unit_prefix + ".conv1.conv",
                            kResidualDilations[unit], 1, state);
      x = Snake(x, unit_prefix + ".act2");
      x = CausalConvolution(x, unit_prefix + ".conv2.conv", 1, 1, state);
      wave = mx::add(x, wave);
    }
  }
  const int output_stage = static_cast<int>(upsample_rates_.size()) + 1;
  wave = Snake(wave, stack + "." + std::to_string(output_stage));
  wave = CausalConvolution(
      wave, stack + "." + std::to_string(output_stage + 1) + ".conv", 1, 1,
      state);
  if (state) {
    state->decoded_frames += frame_count;
  } else {
  }
  return mx::reshape(mx::clip(wave, mx::array(-1.0f), mx::array(1.0f)), {-1});
}

} // namespace qwen3_tts
