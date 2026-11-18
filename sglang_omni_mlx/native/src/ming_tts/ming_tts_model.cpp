// SPDX-License-Identifier: Apache-2.0
#include "ming_tts/ming_tts_model.h"

#include <cassert>
#include <cmath>
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

mx::Dtype WeightDtype(const omni_common::Checkpoint &checkpoint,
                      const std::string &prefix) {
  if (checkpoint.Has(prefix + ".scales")) {
    return checkpoint.Weight(prefix + ".scales").dtype();
  } else {
    return checkpoint.Weight(prefix + ".weight").dtype();
  }
}

namespace {

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
                        text.value("use_bias", false)};
  if (text.value("model_type", std::string("bailing_moe")) != "bailing_moe") {
    throw std::invalid_argument("Ming backbone requires bailing_moe");
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
  if (!rope.is_null() && (rope.value("type", std::string()) != "3D" ||
                          !rope.value("factor", nlohmann::json()).is_null())) {
    throw std::invalid_argument("Ming requires unscaled 3D RoPE");
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
      checkpoint_(BackboneWeights(model_directory, config_), config) {}

mx::array MingBackbone::EmbedTokens(const mx::array &input_ids) const {
  return checkpoint_.Embed(input_ids, "word_embeddings");
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
  const mx::array routed = omni_common::RouteToExperts(
      mx::reshape(x, {-1, x.shape(2)}),
      mx::reshape(indices, {-1, config_.num_experts_per_tok}),
      [&](const mx::array &rows, const mx::array &row_experts, bool sorted) {
        const auto project = [&](const mx::array &input,
                                 const std::string &projection) {
          const auto name = prefix + ".experts." + projection;
          if (checkpoint_.Has(name + ".scales")) {
            const auto &quantization = checkpoint_.Quantization(name);
            return mx::gather_qmm(input, checkpoint_.Weight(name + ".weight"),
                                  checkpoint_.Weight(name + ".scales"),
                                  checkpoint_.Weight(name + ".biases"),
                                  std::nullopt, row_experts, true,
                                  quantization.group_size, quantization.bits,
                                  quantization.mode, std::nullopt, sorted);
          } else {
            return mx::gather_mm(
                input,
                mx::swapaxes(checkpoint_.Weight(name + ".weight"), -1, -2),
                std::nullopt, row_experts, sorted);
          }
        };
        return project(
            mx::multiply(omni_common::Silu(project(rows, "gate_proj")),
                         project(rows, "up_proj")),
            "down_proj");
      });
  const mx::array output = mx::astype(
      mx::reshape(
          mx::sum(
              mx::multiply(mx::astype(routed, mx::float32),
                           mx::expand_dims(
                               mx::reshape(routing_weights,
                                           {-1, config_.num_experts_per_tok}),
                               -1)),
              -2),
          x.shape()),
      x.dtype());
  if (config_.num_shared_experts > 0) {
    return mx::add(output, Mlp(x, prefix + ".shared_experts"));
  } else {
    return output;
  }
}

mx::array MingBackbone::Forward(const mx::array &embeddings,
                                std::vector<qwen3_asr::KVCache> &caches) const {
  assert(embeddings.ndim() == 3 && embeddings.shape(2) == config_.hidden_size);
  assert(caches.size() == static_cast<size_t>(config_.num_hidden_layers));
  const int batch = embeddings.shape(0), length = embeddings.shape(1);
  const int heads = config_.num_attention_heads,
            kv_heads = config_.num_key_value_heads;
  const int head_dim = config_.head_dim;
  // Note (XXXXRT666): a speech prompt has no image or video, so the three
  // axes of Ming's 3D RoPE share each token's position and its sections
  // rotate as one plain RoPE.
  const int offset = caches.front().offset();
  const auto rotate = [&](const mx::array &projected, int head_count) {
    return mx::fast::rope(
        mx::transpose(
            mx::reshape(projected, {batch, length, head_count, head_dim}),
            {0, 2, 1, 3}),
        head_dim, false, config_.rope_theta, 1.0f, offset);
  };
  mx::array hidden = embeddings;
  for (int layer = 0; layer < config_.num_hidden_layers; ++layer) {
    const std::string prefix = "layers." + std::to_string(layer);
    const mx::array normed = checkpoint_.RmsNorm(
        hidden, prefix + ".input_layernorm", config_.rms_norm_eps);
    const auto qkv = mx::split(
        checkpoint_.Linear(normed, prefix + ".attention.query_key_value"),
        {heads * head_dim, (heads + kv_heads) * head_dim}, -1);
    const mx::array queries = rotate(qkv[0], heads);
    const mx::array keys = rotate(qkv[1], kv_heads);
    const mx::array values = mx::transpose(
        mx::reshape(qkv[2], {batch, length, kv_heads, head_dim}), {0, 2, 1, 3});
    auto [cached_keys, cached_values] =
        caches[layer].UpdateAndFetch(keys, values);
    const mx::array attended = mx::fast::scaled_dot_product_attention(
        queries, cached_keys, cached_values,
        static_cast<float>(std::pow(static_cast<double>(head_dim), -0.5)),
        /*window_size=*/-1, length > 1 ? "causal" : "");
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
