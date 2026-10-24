// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Prince Canuma and contributors.
#include "qwen3_tts/qwen3_tts_model.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <set>

namespace qwen3_tts {
namespace mx = mlx::core;
using qwen3_asr::WeightMap;

namespace {

mx::array Elu(const mx::array &hidden) {
  return mx::where(hidden > mx::array(0, hidden.dtype()), hidden,
                   mx::exp(hidden) - mx::array(1, hidden.dtype()));
}

} // namespace

AudioCodec::AudioCodec(const std::filesystem::path &directory, bool encoder)
    : config_(qwen3_asr::ReadJson(directory / "config.json")),
      checkpoint_(LoadWeights(directory, config_, encoder),
                  nlohmann::json::object()) {}

mx::array AudioCodec::Convolve(const mx::array &hidden,
                               const std::string &prefix, State *state,
                               int stride, int dilation, int groups,
                               bool transpose,
                               const std::string &pad_mode) const {
  const auto &weight = checkpoint_.Weight(prefix + ".weight");
  const int padding = (weight.shape(1) - 1) * dilation + 1 - stride;
  mx::array output = hidden;
  if (transpose) {
    output = mx::conv_transpose1d(output, weight, stride, 0, dilation);
    if (state != nullptr && state->tails.contains(prefix)) {
      const auto &tail = state->tails.at(prefix);
      output = mx::concatenate({TimeSlice(output, 0, padding) + tail,
                                TimeSlice(output, padding, output.shape(1))},
                               1);
    } else {
    }
    if (padding > 0) {
      if (state != nullptr) {
        state->tails.insert_or_assign(
            prefix, mx::copy(TimeSlice(output, output.shape(1) - padding,
                                       output.shape(1))));
      } else {
      }
      output = TimeSlice(output, 0, output.shape(1) - padding);
    } else {
    }
  } else {
    if (padding > 0) {
      if (state != nullptr && state->tails.contains(prefix)) {
        output = mx::concatenate({state->tails.at(prefix), output}, 1);
      } else {
        output = mx::pad(output, std::vector<int>{1}, mx::Shape{padding},
                         mx::Shape{0}, mx::array(0, hidden.dtype()), pad_mode);
      }
      if (state != nullptr) {
        state->tails.insert_or_assign(
            prefix, mx::copy(TimeSlice(output, output.shape(1) - padding,
                                       output.shape(1))));
      } else {
      }
    } else {
    }
    if (stride > 1) {
      const int right =
          (hidden.shape(1) + stride - 1) / stride * stride - hidden.shape(1);
      output =
          mx::pad(output, std::vector<int>{1}, mx::Shape{0}, mx::Shape{right},
                  mx::array(0, hidden.dtype()), pad_mode);
    } else {
    }
    output = mx::conv1d(output, weight, stride, 0, dilation, groups);
  }
  return checkpoint_.Has(prefix + ".bias")
             ? output + checkpoint_.Weight(prefix + ".bias")
             : output;
}

mx::array
AudioCodec::Transformer(mx::array hidden, const std::string &prefix,
                        const nlohmann::json &config, State *state,
                        const std::function<void()> &check_cancelled) const {
  const bool encoder = prefix.starts_with("encoder.");
  const int length = hidden.shape(1);
  const int heads = config.at("num_attention_heads");
  const int kv_heads = config.at("num_key_value_heads");
  const int head_dim = config.at("head_dim");
  // Note (Akazaakane): Older tokenizer JSON omits the decoder's computed
  // window-72 field.
  const int window = config.value("sliding_window", encoder ? 250 : 72);
  const int offset = state == nullptr ? 0 : state->offset;
  const float epsilon = encoder ? config.at("norm_eps").get<float>()
                                : config.at("rms_norm_eps").get<float>();
  for (int layer = 0; layer < config.at("num_hidden_layers").get<int>();
       ++layer) {
    check_cancelled();
    const auto stem = prefix + ".layers." + std::to_string(layer);
    const auto normalize = [&](const mx::array &values,
                               const std::string &name) {
      if (encoder) {
        return checkpoint_.LayerNorm(values, stem + "." + name, epsilon);
      } else {
        const auto promoted = mx::astype(values, mx::float32);
        const auto normalized =
            promoted *
            mx::rsqrt(mx::mean(promoted * promoted, -1, true) + epsilon);
        return mx::astype(checkpoint_.Weight(stem + "." + name + ".weight") *
                              normalized,
                          values.dtype());
      }
    };
    const auto normalized = normalize(hidden, "input_layernorm");
    const auto project = [&](const std::string &name, int count) {
      return mx::transpose(
          mx::reshape(
              checkpoint_.Linear(normalized, stem + ".self_attn." + name),
              {1, length, count, head_dim}),
          {0, 2, 1, 3});
    };
    const auto queries =
        mx::fast::rope(project("q_proj", heads), head_dim, false,
                       config.at("rope_theta").get<float>(), 1.0f, offset);
    auto keys =
        mx::fast::rope(project("k_proj", kv_heads), head_dim, false,
                       config.at("rope_theta").get<float>(), 1.0f, offset);
    auto values = project("v_proj", kv_heads);
    if (state != nullptr && state->keys.contains(stem)) {
      keys = mx::concatenate({state->keys.at(stem), keys}, 2);
      values = mx::concatenate({state->values.at(stem), values}, 2);
    } else {
    }
    const int key_count = keys.shape(2);
    const auto query_positions =
        mx::reshape(mx::arange(offset, offset + length), {length, 1});
    const auto key_positions =
        mx::reshape(mx::arange(offset + length - key_count, offset + length),
                    {1, key_count});
    const auto mask = mx::where(
        mx::logical_and(key_positions <= query_positions,
                        key_positions > query_positions - window),
        mx::array(0.0f, hidden.dtype()),
        mx::array(-std::numeric_limits<float>::infinity(), hidden.dtype()));
    const auto attention = mx::fast::scaled_dot_product_attention(
        queries, keys, values, 1.0f / std::sqrt(static_cast<float>(head_dim)),
        "array", mask);
    hidden =
        hidden +
        checkpoint_.Linear(mx::reshape(mx::transpose(attention, {0, 2, 1, 3}),
                                       {1, length, heads * head_dim}),
                           stem + ".self_attn.o_proj") *
            checkpoint_.Weight(stem + ".self_attn_layer_scale.scale");
    const auto mlp_input = normalize(hidden, "post_attention_layernorm");
    const auto mlp =
        encoder ? checkpoint_.Linear(qwen3_asr::Gelu(checkpoint_.Linear(
                                         mlp_input, stem + ".mlp.fc1")),
                                     stem + ".mlp.fc2")
                : checkpoint_.Linear(
                      qwen3_asr::Silu(checkpoint_.Linear(
                          mlp_input, stem + ".mlp.gate_proj")) *
                          checkpoint_.Linear(mlp_input, stem + ".mlp.up_proj"),
                      stem + ".mlp.down_proj");
    hidden = hidden + mlp * checkpoint_.Weight(stem + ".mlp_layer_scale.scale");
    if (state != nullptr) {
      // Note (Akazaakane): Retain only prior keys visible to the next query;
      // RoPE still uses absolute positions.
      const mx::Shape start{0, 0, std::max(0, key_count - window + 1), 0};
      const mx::Shape stop{1, kv_heads, key_count, head_dim};
      state->keys.insert_or_assign(stem,
                                   mx::copy(mx::slice(keys, start, stop)));
      state->values.insert_or_assign(stem,
                                     mx::copy(mx::slice(values, start, stop)));
    } else {
    }
  }
  if (state != nullptr) {
    state->offset += length;
  } else {
  }
  return hidden;
}

mx::array AudioCodec::Snake(const mx::array &hidden,
                            const std::string &prefix) const {
  const auto alpha = mx::exp(checkpoint_.Weight(prefix + ".alpha"));
  const auto beta = mx::exp(checkpoint_.Weight(prefix + ".beta"));
  const auto sine = mx::sin(hidden * alpha);
  return hidden +
         mx::reciprocal(beta + mx::array(1e-9f, beta.dtype())) * (sine * sine);
}

mx::array
AudioCodec::Decode(const mx::array &codes, State &state,
                   const std::function<void()> &check_cancelled) const {
  if (codes.ndim() != 2 || codes.shape(0) == 0 ||
      codes.shape(1) != kCodeGroups) {
    throw std::invalid_argument("codec codes must have shape [frames,16]");
  } else {
  }
  const auto &config = config_.at("decoder_config");
  std::vector<mx::array> partitions;
  for (int partition = 0; partition < 2; ++partition) {
    const auto stem = std::string("decoder.quantizer.") +
                      (partition == 0 ? "rvq_first" : "rvq_rest");
    std::optional<mx::array> sum;
    const int first = partition == 0 ? 0 : 1;
    const int count = partition == 0 ? 1 : kCodeGroups - 1;
    for (int group = 0; group < count; ++group) {
      const auto indices =
          mx::reshape(mx::slice(codes, {0, first + group},
                                {codes.shape(0), first + group + 1}),
                      {1, codes.shape(0)});
      const auto embedding = checkpoint_.Embed(
          indices,
          stem + ".vq.layers." + std::to_string(group) + ".codebook.embed");
      sum = sum.has_value() ? *sum + embedding : embedding;
    }
    partitions.push_back(Convolve(*sum, stem + ".output_proj", nullptr));
  }
  auto hidden =
      Convolve(partitions[0] + partitions[1], "decoder.pre_conv.conv", &state);
  hidden = checkpoint_.Linear(hidden, "decoder.pre_transformer.input_proj");
  hidden = Transformer(hidden, "decoder.pre_transformer", config, &state,
                       check_cancelled);
  const auto promoted = mx::astype(hidden, mx::float32);
  const auto normalized =
      promoted * mx::rsqrt(mx::mean(promoted * promoted, -1, true) +
                           config.at("rms_norm_eps").get<float>());
  hidden = mx::astype(
      checkpoint_.Weight("decoder.pre_transformer.norm.weight") * normalized,
      hidden.dtype());
  hidden = checkpoint_.Linear(hidden, "decoder.pre_transformer.output_proj");
  const auto ratios = config.at("upsampling_ratios").get<std::vector<int>>();
  for (size_t index = 0; index < ratios.size(); ++index) {
    check_cancelled();
    const auto stem = "decoder.upsample." + std::to_string(index);
    hidden =
        Convolve(hidden, stem + ".0.conv", nullptr, ratios[index], 1, 1, true);
    auto transformed = Convolve(hidden, stem + ".1.dwconv.conv", &state, 1, 1,
                                hidden.shape(2));
    transformed = checkpoint_.LayerNorm(transformed, stem + ".1.norm", 1e-6f);
    transformed = checkpoint_.Linear(
        qwen3_asr::Gelu(checkpoint_.Linear(transformed, stem + ".1.pwconv1")),
        stem + ".1.pwconv2");
    hidden = hidden + transformed * checkpoint_.Weight(stem + ".1.gamma");
  }
  hidden = Convolve(hidden, "decoder.decoder.0.conv", &state);
  const auto rates = config.at("upsample_rates").get<std::vector<int>>();
  for (size_t index = 0; index < rates.size(); ++index) {
    check_cancelled();
    const auto stem = "decoder.decoder." + std::to_string(index + 1) + ".block";
    hidden = Convolve(Snake(hidden, stem + ".0"), stem + ".1.conv", &state,
                      rates[index], 1, 1, true);
    for (int residual = 0; residual < 3; ++residual) {
      const auto unit = stem + "." + std::to_string(residual + 2);
      const int dilation = residual == 0 ? 1 : residual == 1 ? 3 : 9;
      auto transformed = Convolve(Snake(hidden, unit + ".act1"),
                                  unit + ".conv1.conv", &state, 1, dilation);
      hidden = hidden + Convolve(Snake(transformed, unit + ".act2"),
                                 unit + ".conv2.conv", &state);
    }
  }
  hidden = Snake(hidden, "decoder.decoder." + std::to_string(rates.size() + 1));
  hidden = Convolve(
      hidden, "decoder.decoder." + std::to_string(rates.size() + 2) + ".conv",
      &state);
  std::vector<mx::array> buffers;
  for (const auto *collection : {&state.tails, &state.keys, &state.values}) {
    for (const auto &[name, buffer] : *collection) {
      buffers.push_back(buffer);
    }
  }
  buffers.push_back(hidden);
  mx::eval(buffers);
  return mx::reshape(mx::clip(hidden, mx::array(-1.0f, hidden.dtype()),
                              mx::array(1.0f, hidden.dtype())),
                     {-1});
}

mx::array
AudioCodec::Encode(const std::vector<float> &samples,
                   const std::function<void()> &check_cancelled) const {
  const auto &config = config_.at("encoder_config");
  auto hidden = mx::array(
      samples.data(), {1, static_cast<int>(samples.size()), 1}, mx::float32);
  hidden = Convolve(hidden, "encoder.encoder.layers.0.conv", nullptr);
  const auto ratios = config.at("upsampling_ratios").get<std::vector<int>>();
  int layer = 1;
  for (auto ratio = ratios.rbegin(); ratio != ratios.rend(); ++ratio) {
    check_cancelled();
    const auto stem = "encoder.encoder.layers." + std::to_string(layer);
    auto transformed = Convolve(Elu(hidden), stem + ".block.1.conv", nullptr);
    hidden =
        hidden + Convolve(Elu(transformed), stem + ".block.3.conv", nullptr);
    hidden = Convolve(Elu(hidden),
                      "encoder.encoder.layers." + std::to_string(layer + 2) +
                          ".conv",
                      nullptr, *ratio);
    layer += 3;
  }
  hidden = Convolve(
      Elu(hidden),
      "encoder.encoder.layers." + std::to_string(layer + 1) + ".conv", nullptr);
  hidden = Transformer(hidden, "encoder.encoder_transformer", config, nullptr,
                       check_cancelled);
  hidden = Convolve(hidden, "encoder.downsample.conv", nullptr, 2, 1, 1, false,
                    "edge");
  std::vector<mx::array> codes;
  for (int partition = 0; partition < 2; ++partition) {
    const auto stem = std::string("encoder.quantizer.") +
                      (partition == 0 ? "semantic_residual_vector_quantizer"
                                      : "acoustic_residual_vector_quantizer");
    auto residual = Convolve(hidden, stem + ".input_proj", nullptr);
    const int count = partition == 0 ? 1 : kCodeGroups - 1;
    for (int group = 0; group < count; ++group) {
      check_cancelled();
      const auto &table = checkpoint_.Weight(
          stem + ".layers." + std::to_string(group) + ".codebook.embed.weight");
      const auto table32 = mx::astype(table, mx::float32);
      const auto distance =
          mx::sum(table32 * table32, -1) / 2.0f -
          mx::matmul(mx::astype(residual, mx::float32), mx::transpose(table32));
      const auto indices = mx::astype(mx::argmin(distance, -1), mx::int32);
      codes.push_back(mx::reshape(indices, {-1}));
      residual =
          mx::astype(mx::astype(residual, mx::float32) -
                         mx::astype(mx::take(table, indices, 0), mx::float32),
                     residual.dtype());
    }
  }
  auto output = mx::stack(codes, 1);
  mx::eval(output);
  return output;
}

} // namespace qwen3_tts
