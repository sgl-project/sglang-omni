// SPDX-License-Identifier: Apache-2.0
#include "ming_tts/ming_tts_model.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <numeric>
#include <optional>
#include <stdexcept>

namespace ming_tts {

namespace mx = mlx::core;

omni_common::WeightMap
LoadComponentWeights(const std::filesystem::path &directory,
                     const std::vector<std::string> &prefixes) {
  auto source = omni_common::LoadSafetensors(directory);
  omni_common::WeightMap weights;
  for (auto &[name, tensor] : source) {
    // Note (altale): Audio transformers consume hidden states, not token IDs.
    if (name.starts_with("audio.decoder.decoder.embed_tokens.") ||
        name.starts_with("audio.encoder.encoder.embed_tokens.") ||
        name.starts_with("audio.encoder.aggregator.embed_tokens.") ||
        name.starts_with("audio.encoder.norm.")) {
      continue;
    } else {
    }
    for (const auto &prefix : prefixes) {
      if (name.starts_with(prefix)) {
        weights.emplace(name, std::move(tensor));
        break;
      } else {
      }
    }
  }
  return weights;
}

void RequireShape(const omni_common::Checkpoint &checkpoint,
                  const std::string &name, const mx::Shape &shape) {
  const auto &tensor = checkpoint.Weight(name);
  const std::string prefix =
      name.ends_with(".weight") ? name.substr(0, name.size() - 7) : name;
  if (name.ends_with(".weight") && checkpoint.Has(prefix + ".scales")) {
    const auto &quantization = checkpoint.quantization();
    if (shape.size() < 2 || quantization.group_size <= 0 ||
        quantization.bits <= 0 || quantization.mode != "affine" ||
        shape.back() % quantization.group_size != 0 ||
        shape.back() * quantization.bits % 32 != 0) {
      throw std::invalid_argument("invalid quantized tensor configuration: " +
                                  name);
    } else {
    }
    auto packed_shape = shape;
    packed_shape.back() = shape.back() * quantization.bits / 32;
    auto scale_shape = shape;
    scale_shape.back() /= quantization.group_size;
    if (tensor.dtype() != mx::uint32 || tensor.shape() != packed_shape) {
      throw std::invalid_argument("invalid packed tensor shape or dtype: " +
                                  name);
    } else {
    }
    RequireShape(checkpoint, prefix + ".scales", scale_shape);
    RequireShape(checkpoint, prefix + ".biases", scale_shape);
  } else if (tensor.shape() != shape ||
             !mx::issubdtype(tensor.dtype(), mx::floating)) {
    throw std::invalid_argument("invalid tensor shape or dtype: " + name);
  } else {
  }
}

mx::Dtype WeightDtype(const omni_common::Checkpoint &checkpoint,
                      const std::string &prefix) {
  if (checkpoint.Has(prefix + ".scales")) {
    return checkpoint.Weight(prefix + ".scales").dtype();
  } else {
    return checkpoint.Weight(prefix + ".weight").dtype();
  }
}

namespace {

// Note (altale): Sort larger token batches by expert before grouped matmul.
constexpr int kExpertSortThreshold = 64;

omni_common::WeightMap
BackboneWeights(const std::filesystem::path &model_directory,
                const BackboneConfig &config) {
  auto source = omni_common::LoadSafetensors(model_directory);
  omni_common::WeightMap weights;
  for (auto &[name, tensor] : source) {
    if (!name.starts_with("model.model.") ||
        name.ends_with(".rotary_emb.inv_freq")) {
      continue;
    } else {
    }
    std::string target = name.substr(std::string("model.model.").size());
    for (const auto &[from, to] :
         {std::pair{std::string(".mlp.gate.gate_proj."),
                    std::string(".mlp.gate.")},
          std::pair{std::string(".mlp.switch_mlp."),
                    std::string(".mlp.experts.")}}) {
      const auto position = target.find(from);
      if (position != std::string::npos) {
        target.replace(position, from.size(), to);
      } else {
      }
    }
    weights.emplace(target, std::move(tensor));
  }
  for (int layer = config.first_k_dense_replace;
       layer < config.num_hidden_layers; ++layer) {
    const std::string prefix =
        "layers." + std::to_string(layer) + ".mlp.experts.";
    for (const auto *projection : {"gate_proj", "up_proj", "down_proj"}) {
      if (!weights.contains(prefix + projection + ".weight")) {
        std::vector<mx::array> experts;
        for (int expert = 0; expert < config.num_experts; ++expert) {
          const auto name =
              prefix + std::to_string(expert) + "." + projection + ".weight";
          experts.push_back(weights.at(name));
          weights.erase(name);
        }
        weights.emplace(prefix + projection + ".weight", mx::stack(experts));
      } else {
      }
    }
  }
  return weights;
}

} // namespace

BackboneConfig BackboneConfig::FromJson(const nlohmann::json &config) {
  if (config.value("model_type", std::string("bailingmm")) != "bailingmm") {
    throw std::invalid_argument(
        "Ming backbone requires a bailingmm checkpoint");
  } else {
  }
  const auto &text = config.at("llm_config");
  BackboneConfig result{text.at("vocab_size"),
                        text.at("hidden_size"),
                        text.at("intermediate_size"),
                        text.at("moe_intermediate_size"),
                        text.at("num_hidden_layers"),
                        text.at("num_attention_heads"),
                        text.at("num_key_value_heads"),
                        text.at("head_dim"),
                        text.at("num_experts"),
                        text.at("num_experts_per_tok"),
                        text.at("num_shared_experts"),
                        text.value("first_k_dense_replace", 0),
                        text.value("max_position_embeddings", 32768),
                        text.value("rms_norm_eps", 1e-5f),
                        text.value("rope_theta", 600000.0f),
                        text.value("routed_scaling_factor", 1.0f),
                        text.value("norm_topk_prob", true),
                        text.value("use_qkv_bias", false),
                        text.value("use_bias", false),
                        {}};
  if (text.value("model_type", std::string("bailing_moe")) != "bailing_moe" ||
      std::min({result.vocab_size, result.hidden_size, result.intermediate_size,
                result.moe_intermediate_size, result.num_hidden_layers,
                result.num_attention_heads, result.num_key_value_heads,
                result.head_dim, result.num_experts, result.num_experts_per_tok,
                result.max_position_embeddings}) <= 0 ||
      result.num_experts_per_tok > result.num_experts ||
      result.num_shared_experts < 0 || result.first_k_dense_replace < 0 ||
      result.first_k_dense_replace > result.num_hidden_layers ||
      result.head_dim % 2 ||
      result.num_attention_heads % result.num_key_value_heads ||
      !(result.rope_theta > 0) || !(result.rms_norm_eps > 0)) {
    throw std::invalid_argument(
        "unsupported Ming backbone dimensions or model type");
  } else {
  }
  for (const auto *name :
       {"use_qk_norm", "use_sliding_window", "moe_router_enable_expert_bias"}) {
    if (text.value(name, false)) {
      throw std::invalid_argument(std::string("unsupported Ming option: ") +
                                  name);
    } else {
    }
  }
  for (const auto *name :
       {"n_group", "num_expert_group", "topk_group", "router_dtype"}) {
    const auto value = text.value(name, nlohmann::json());
    if (!value.is_null() && value != 0) {
      throw std::invalid_argument(std::string("unsupported Ming option: ") +
                                  name);
    } else {
    }
  }
  const auto score = text.value("score_function", nlohmann::json());
  const auto shared_size =
      text.value("moe_shared_expert_intermediate_size", nlohmann::json());
  if (text.value("hidden_act", std::string("silu")) != "silu" ||
      (!score.is_null() && score != "softmax") ||
      (!shared_size.is_null() && shared_size != result.moe_intermediate_size)) {
    throw std::invalid_argument(
        "Ming requires SiLU, softmax routing and matching expert widths");
  } else {
  }
  const auto rope = text.value("rope_scaling", nlohmann::json());
  if (rope.is_null()) {
    result.mrope_section = {result.head_dim / 2, 0, 0};
  } else {
    if (rope.value("type", std::string()) != "3D" ||
        !rope.value("factor", nlohmann::json()).is_null()) {
      throw std::invalid_argument("Ming requires unscaled 3D RoPE");
    } else {
    }
    const auto sections =
        rope.value("mrope_section", std::vector<int>{16, 24, 24});
    if (sections.size() != 3) {
      throw std::invalid_argument("mrope_section must have three axes");
    } else {
      std::copy(sections.begin(), sections.end(), result.mrope_section.begin());
    }
  }
  if (*std::min_element(result.mrope_section.begin(),
                        result.mrope_section.end()) < 0 ||
      std::accumulate(result.mrope_section.begin(), result.mrope_section.end(),
                      0) != result.head_dim / 2) {
    throw std::invalid_argument(
        "mrope_section must partition half the head dimension");
  } else {
  }
  return result;
}

MingBackbone::MingBackbone(const std::filesystem::path &model_directory)
    : MingBackbone(omni_common::ReadJson(model_directory / "config.json"),
                   model_directory) {}

MingBackbone::MingBackbone(const nlohmann::json &config,
                           const std::filesystem::path &model_directory)
    : config_(BackboneConfig::FromJson(config)),
      checkpoint_(BackboneWeights(model_directory, config_), config),
      inverse_frequencies_(mx::divide(
          mx::array(1.0f),
          mx::power(mx::array(config_.rope_theta),
                    mx::divide(mx::arange(0, config_.head_dim, 2, mx::float32),
                               mx::array(config_.head_dim))))) {
  RequireShape(checkpoint_, "word_embeddings.weight",
               {config_.vocab_size, config_.hidden_size});
  RequireShape(checkpoint_, "norm.weight", {config_.hidden_size});
  const auto linear = [&](const std::string &prefix, int output_size,
                          int input_size, bool bias) {
    RequireShape(checkpoint_, prefix + ".weight", {output_size, input_size});
    if (bias) {
      RequireShape(checkpoint_, prefix + ".bias", {output_size});
    } else {
    }
  };
  for (int layer = 0; layer < config_.num_hidden_layers; ++layer) {
    const std::string prefix = "layers." + std::to_string(layer);
    for (const auto *norm : {"input_layernorm", "post_attention_layernorm"}) {
      RequireShape(checkpoint_, prefix + "." + norm + ".weight",
                   {config_.hidden_size});
    }
    linear(prefix + ".attention.query_key_value",
           (config_.num_attention_heads + 2 * config_.num_key_value_heads) *
               config_.head_dim,
           config_.hidden_size, config_.use_qkv_bias);
    linear(prefix + ".attention.dense", config_.hidden_size,
           config_.num_attention_heads * config_.head_dim, config_.use_bias);
    if (layer >= config_.first_k_dense_replace) {
      linear(prefix + ".mlp.gate", config_.num_experts, config_.hidden_size,
             false);
    } else {
    }
    for (const auto *projection : {"gate_proj", "up_proj", "down_proj"}) {
      const bool down = std::string(projection) == "down_proj";
      if (layer < config_.first_k_dense_replace) {
        linear(prefix + ".mlp." + projection,
               down ? config_.hidden_size : config_.intermediate_size,
               down ? config_.intermediate_size : config_.hidden_size, false);
      } else {
        RequireShape(
            checkpoint_, prefix + ".mlp.experts." + projection + ".weight",
            {config_.num_experts,
             down ? config_.hidden_size : config_.moe_intermediate_size,
             down ? config_.moe_intermediate_size : config_.hidden_size});
        if (config_.num_shared_experts > 0) {
          const int intermediate_size =
              config_.moe_intermediate_size * config_.num_shared_experts;
          linear(prefix + ".mlp.shared_experts." + projection,
                 down ? config_.hidden_size : intermediate_size,
                 down ? intermediate_size : config_.hidden_size, false);
        } else {
        }
      }
    }
  }
  mx::eval(inverse_frequencies_);
}

mx::array MingBackbone::EmbedTokens(const mx::array &input_ids) const {
  return checkpoint_.Embed(input_ids, "word_embeddings");
}

mx::array MingBackbone::Rotary(const mx::array &x,
                               const mx::array &positions) const {
  const mx::array phases =
      mx::multiply(mx::expand_dims(mx::astype(positions, mx::float32), -1),
                   inverse_frequencies_);
  std::vector<mx::array> parts;
  int start = 0;
  for (int axis = 0; axis < 3; ++axis) {
    const int stop = start + config_.mrope_section[axis];
    parts.push_back(mx::squeeze(
        mx::slice(phases, {axis, 0, 0, start},
                  {axis + 1, positions.shape(1), positions.shape(2), stop}),
        0));
    start = stop;
  }
  const mx::array phase = mx::expand_dims(mx::concatenate(parts, -1), 1);
  const auto halves = mx::split(mx::astype(x, mx::float32), 2, -1);
  const mx::array cosine = mx::cos(phase), sine = mx::sin(phase);
  return mx::astype(
      mx::concatenate({mx::subtract(mx::multiply(halves[0], cosine),
                                    mx::multiply(halves[1], sine)),
                       mx::add(mx::multiply(halves[0], sine),
                               mx::multiply(halves[1], cosine))},
                      -1),
      x.dtype());
}

mx::array MingBackbone::Mlp(const mx::array &x,
                            const std::string &prefix) const {
  return checkpoint_.Linear(
      mx::multiply(
          omni_common::Silu(checkpoint_.Linear(x, prefix + ".gate_proj")),
          checkpoint_.Linear(x, prefix + ".up_proj")),
      prefix + ".down_proj");
}

mx::array MingBackbone::Moe(const mx::array &x,
                            const std::string &prefix) const {
  const mx::array logits =
      mx::astype(checkpoint_.Linear(
                     mx::astype(x, WeightDtype(checkpoint_, prefix + ".gate")),
                     prefix + ".gate"),
                 x.dtype());
  const mx::array scores = mx::softmax(mx::astype(logits, mx::float32), -1);
  const mx::array indices = mx::slice(
      mx::argpartition(mx::negative(scores), config_.num_experts_per_tok - 1,
                       -1),
      {0, 0, 0}, {x.shape(0), x.shape(1), config_.num_experts_per_tok});
  mx::array routing_weights = mx::take_along_axis(scores, indices, -1);
  if (config_.norm_topk_prob) {
    routing_weights =
        mx::divide(routing_weights, mx::sum(routing_weights, -1, true));
  } else {
  }
  routing_weights =
      mx::multiply(routing_weights, mx::array(config_.routed_scaling_factor));
  mx::array expanded =
      mx::reshape(x, {x.shape(0), x.shape(1), 1, 1, x.shape(2)});
  mx::array expert_indices = indices;
  std::optional<mx::array> inverse_order;
  const bool sorted_indices = indices.size() >= kExpertSortThreshold;
  if (sorted_indices) {
    const mx::array flat_indices = mx::reshape(indices, {-1});
    const mx::array order = mx::argsort(flat_indices);
    inverse_order = mx::argsort(order);
    expanded = mx::take(
        mx::reshape(expanded, {-1, 1, x.shape(2)}),
        mx::floor_divide(order, mx::array(config_.num_experts_per_tok)), 0);
    expert_indices = mx::take(flat_indices, order, 0);
  } else {
  }
  const auto project = [&](const mx::array &input,
                           const std::string &projection) {
    const auto name = prefix + ".experts." + projection;
    if (checkpoint_.Has(name + ".scales")) {
      const auto &quantization = checkpoint_.quantization();
      return mx::gather_qmm(input, checkpoint_.Weight(name + ".weight"),
                            checkpoint_.Weight(name + ".scales"),
                            checkpoint_.Weight(name + ".biases"), std::nullopt,
                            expert_indices, true, quantization.group_size,
                            quantization.bits, quantization.mode, std::nullopt,
                            sorted_indices);
    } else {
      return mx::gather_mm(
          input, mx::swapaxes(checkpoint_.Weight(name + ".weight"), -1, -2),
          std::nullopt, expert_indices, sorted_indices);
    }
  };
  mx::array routed =
      project(mx::multiply(omni_common::Silu(project(expanded, "gate_proj")),
                           project(expanded, "up_proj")),
              "down_proj");
  if (sorted_indices) {
    routed = mx::reshape(
        mx::take(routed, *inverse_order, 0),
        {x.shape(0), x.shape(1), config_.num_experts_per_tok, 1, x.shape(2)});
  } else {
  }
  routed = mx::squeeze(routed, -2);
  mx::array output =
      mx::astype(mx::sum(mx::multiply(mx::astype(routed, mx::float32),
                                      mx::expand_dims(routing_weights, -1)),
                         -2),
                 x.dtype());
  if (config_.num_shared_experts > 0) {
    return mx::add(output, Mlp(x, prefix + ".shared_experts"));
  } else {
    return output;
  }
}

mx::array MingBackbone::Forward(const mx::array &embeddings,
                                const mx::array &positions,
                                std::vector<qwen3_asr::KVCache> &caches) const {
  assert(embeddings.ndim() == 3 && embeddings.shape(2) == config_.hidden_size);
  assert(positions.shape() ==
         mx::Shape({3, embeddings.shape(0), embeddings.shape(1)}));
  assert(caches.size() == static_cast<size_t>(config_.num_hidden_layers));
  const int batch = embeddings.shape(0), length = embeddings.shape(1);
  const int heads = config_.num_attention_heads,
            kv_heads = config_.num_key_value_heads;
  const int head_dim = config_.head_dim;
  mx::array hidden = embeddings;
  for (int layer = 0; layer < config_.num_hidden_layers; ++layer) {
    const std::string prefix = "layers." + std::to_string(layer);
    const mx::array normed = checkpoint_.RmsNorm(
        hidden, prefix + ".input_layernorm", config_.rms_norm_eps);
    const auto qkv = mx::split(
        checkpoint_.Linear(normed, prefix + ".attention.query_key_value"),
        {heads * head_dim, (heads + kv_heads) * head_dim}, -1);
    const mx::array queries = Rotary(
        mx::transpose(mx::reshape(qkv[0], {batch, length, heads, head_dim}),
                      {0, 2, 1, 3}),
        positions);
    const mx::array keys = Rotary(
        mx::transpose(mx::reshape(qkv[1], {batch, length, kv_heads, head_dim}),
                      {0, 2, 1, 3}),
        positions);
    const mx::array values = mx::transpose(
        mx::reshape(qkv[2], {batch, length, kv_heads, head_dim}), {0, 2, 1, 3});
    auto [cached_keys, cached_values] =
        caches[layer].UpdateAndFetch(keys, values);
    const mx::array attended = mx::fast::scaled_dot_product_attention(
        queries, cached_keys, cached_values,
        static_cast<float>(std::pow(static_cast<double>(head_dim), -0.5)),
        length > 1 ? "causal" : "");
    hidden = mx::add(
        hidden,
        checkpoint_.Linear(mx::reshape(mx::transpose(attended, {0, 2, 1, 3}),
                                       {batch, length, heads * head_dim}),
                           prefix + ".attention.dense"));
    const mx::array mlp_input = checkpoint_.RmsNorm(
        hidden, prefix + ".post_attention_layernorm", config_.rms_norm_eps);
    hidden = mx::add(hidden, layer < config_.first_k_dense_replace
                                 ? Mlp(mlp_input, prefix + ".mlp")
                                 : Moe(mlp_input, prefix + ".mlp"));
  }
  return checkpoint_.RmsNorm(hidden, "norm", config_.rms_norm_eps);
}

std::vector<qwen3_asr::KVCache> MingBackbone::NewCaches() const {
  return std::vector<qwen3_asr::KVCache>(config_.num_hidden_layers);
}

} // namespace ming_tts
