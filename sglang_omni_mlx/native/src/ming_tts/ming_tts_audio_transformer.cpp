// SPDX-License-Identifier: Apache-2.0
#include "ming_tts/ming_tts_audio.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <stdexcept>

#include "ming_tts/ming_tts_model.h"

namespace ming_tts {
namespace mx = mlx::core;

AudioTransformer::AudioTransformer(const omni_common::Checkpoint &checkpoint,
                                   std::string prefix,
                                   const nlohmann::json &config)
    : checkpoint_(checkpoint), prefix_(std::move(prefix)),
      hidden_size_(config.at("hidden_size")),
      layers_(config.at("num_hidden_layers")),
      heads_(config.at("num_attention_heads")),
      kv_heads_(config.value("num_key_value_heads", nlohmann::json()).is_null()
                    ? heads_
                    : config.at("num_key_value_heads").get<int>()),
      epsilon_(config.value("rms_norm_eps", 1e-6f)),
      rope_theta_(config.value("rope_theta", 1000000.0f)) {
  const int intermediate_size = config.at("intermediate_size");
  if (std::min({hidden_size_, layers_, heads_, kv_heads_, intermediate_size}) <=
          0 ||
      hidden_size_ % heads_ || (hidden_size_ / heads_) % 2 ||
      heads_ % kv_heads_ ||
      config.value("hidden_act", std::string("silu")) != "silu") {
    throw std::invalid_argument(
        "unsupported AudioVAE Qwen2 dimensions or activation");
  } else {
  }
  const auto rope = config.value(
      "rope_parameters", config.value("rope_scaling", nlohmann::json()));
  if (!rope.is_null()) {
    if (rope.value("rope_type", rope.value("type", std::string("default"))) !=
        "default") {
      throw std::invalid_argument("AudioVAE native supports default RoPE only");
    } else {
      rope_theta_ = rope.value("rope_theta", rope_theta_);
    }
  } else {
  }
  if (!(epsilon_ > 0) || !(rope_theta_ > 0)) {
    throw std::invalid_argument(
        "AudioVAE epsilon and RoPE base must be positive");
  } else {
  }
  const int window =
      config.value("use_sliding_window", false) &&
              !config.value("sliding_window", nlohmann::json(4096)).is_null()
          ? config.value("sliding_window", 4096)
          : 0;
  const auto layer_types = config.value("layer_types", nlohmann::json());
  if (!layer_types.is_null() &&
      layer_types.size() != static_cast<size_t>(layers_)) {
    throw std::invalid_argument(
        "AudioVAE layer_types must match the layer count");
  } else {
  }
  RequireShape(checkpoint_, prefix_ + ".norm.weight", {hidden_size_});
  for (int layer = 0; layer < layers_; ++layer) {
    const std::string kind =
        layer_types.is_null()
            ? (window > 0 && layer >= config.value("max_window_layers", 28)
                   ? "sliding_attention"
                   : "full_attention")
            : layer_types.at(layer).get<std::string>();
    if (kind == "full_attention") {
      windows_.push_back(0);
    } else if (kind == "sliding_attention" && window > 0) {
      windows_.push_back(window);
    } else {
      throw std::invalid_argument("unsupported AudioVAE attention window");
    }
    const std::string prefix = prefix_ + ".layers." + std::to_string(layer);
    for (const auto *norm : {"input_layernorm", "post_attention_layernorm"}) {
      RequireShape(checkpoint_, prefix + "." + norm + ".weight",
                   {hidden_size_});
    }
    for (const auto *projection : {"q_proj", "k_proj", "v_proj", "o_proj"}) {
      const std::string name = projection;
      const int output_size = name == "q_proj" || name == "o_proj"
                                  ? hidden_size_
                                  : kv_heads_ * (hidden_size_ / heads_);
      RequireShape(checkpoint_, prefix + ".self_attn." + name + ".weight",
                   {output_size, hidden_size_});
      if (name != "o_proj") {
        RequireShape(checkpoint_, prefix + ".self_attn." + name + ".bias",
                     {output_size});
      } else {
      }
    }
    for (const auto *projection : {"gate_proj", "up_proj", "down_proj"}) {
      const bool down = std::string(projection) == "down_proj";
      RequireShape(checkpoint_, prefix + ".mlp." + projection + ".weight",
                   {down ? hidden_size_ : intermediate_size,
                    down ? intermediate_size : hidden_size_});
    }
  }
}

mx::array AudioTransformer::Forward(mx::array hidden,
                                    std::vector<AudioCache> &caches) const {
  assert(caches.size() == static_cast<size_t>(layers_));
  const int batch = hidden.shape(0), length = hidden.shape(1),
            head_dim = hidden_size_ / heads_;
  for (int layer = 0; layer < layers_; ++layer) {
    auto &cache = caches[layer];
    const int past = cache.keys ? cache.keys->shape(2) : 0;
    const mx::array query =
        mx::reshape(mx::arange(past, past + length), {-1, 1});
    const mx::array key = mx::reshape(mx::arange(past + length), {1, -1});
    mx::array mask = mx::greater_equal(query, key);
    if (windows_[layer] > 0) {
      mask = mx::logical_and(
          mask, mx::less(mx::subtract(query, key), mx::array(windows_[layer])));
    } else {
    }
    const std::string prefix = prefix_ + ".layers." + std::to_string(layer);
    const mx::array normed =
        checkpoint_.RmsNorm(hidden, prefix + ".input_layernorm", epsilon_);
    const auto project = [&](const std::string &name, int heads) {
      return mx::transpose(
          mx::reshape(checkpoint_.Linear(normed, prefix + ".self_attn." + name),
                      {batch, length, heads, head_dim}),
          {0, 2, 1, 3});
    };
    const mx::array queries =
        mx::fast::rope(project("q_proj", heads_), head_dim, false, rope_theta_,
                       1.0f, cache.offset);
    mx::array keys = mx::fast::rope(project("k_proj", kv_heads_), head_dim,
                                    false, rope_theta_, 1.0f, cache.offset);
    mx::array values = project("v_proj", kv_heads_);
    if (cache.keys) {
      keys = mx::concatenate({*cache.keys, keys}, 2);
      values = mx::concatenate({*cache.values, values}, 2);
    } else {
    }
    const mx::array attended = mx::fast::scaled_dot_product_attention(
        queries, keys, values,
        static_cast<float>(std::pow(static_cast<double>(head_dim), -0.5)), "",
        mask);
    cache.offset += length;
    const int trim =
        windows_[layer] > 0 ? std::max(0, keys.shape(2) - windows_[layer]) : 0;
    cache.keys = mx::slice(keys, {0, 0, trim, 0}, keys.shape());
    cache.values = mx::slice(values, {0, 0, trim, 0}, values.shape());
    hidden = mx::add(
        hidden,
        checkpoint_.Linear(mx::reshape(mx::transpose(attended, {0, 2, 1, 3}),
                                       {batch, length, hidden_size_}),
                           prefix + ".self_attn.o_proj"));
    const mx::array mlp_input = checkpoint_.RmsNorm(
        hidden, prefix + ".post_attention_layernorm", epsilon_);
    hidden = mx::add(
        hidden, checkpoint_.Linear(
                    mx::multiply(
                        omni_common::Silu(checkpoint_.Linear(
                            mlp_input, prefix + ".mlp.gate_proj")),
                        checkpoint_.Linear(mlp_input, prefix + ".mlp.up_proj")),
                    prefix + ".mlp.down_proj"));
  }
  return checkpoint_.RmsNorm(hidden, prefix_ + ".norm", epsilon_);
}

std::vector<AudioCache> AudioTransformer::NewCaches() const {
  return std::vector<AudioCache>(layers_);
}
} // namespace ming_tts
