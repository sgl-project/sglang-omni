// SPDX-License-Identifier: Apache-2.0
// Copyright 2025 Antgroup and The HuggingFace Inc. team. All rights reserved.
#include "ming3_audio/ming3_audio_model.h"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <stdexcept>
#include <utility>

#include "common/audio.h"

namespace ming3_audio {
namespace mx = mlx::core;

int ParseContextLength(const std::string &value) {
  int tokens = 0;
  const auto [end, error] =
      std::from_chars(value.data(), value.data() + value.size(), tokens);
  if (error != std::errc{} || end != value.data() + value.size() ||
      tokens < 1) {
    throw std::invalid_argument("--context-length must be a positive integer");
  } else {
    return tokens;
  }
}

TextConfig TextConfig::FromJson(const nlohmann::json &config) {
  return {config.at("num_hidden_layers"),
          config.at("num_attention_heads"),
          config.at("head_dim"),
          config.at("layer_group_size"),
          config.at("qk_rope_head_dim"),
          config.at("qk_nope_head_dim"),
          config.at("v_head_dim"),
          config.at("kv_lora_rank"),
          config.at("short_conv_kernel_size"),
          config.at("first_k_dense_replace"),
          config.at("num_experts"),
          config.at("num_experts_per_tok"),
          config.at("n_group"),
          config.at("topk_group"),
          config.at("max_position_embeddings"),
          config.at("rms_norm_eps"),
          config.at("rope_theta"),
          config.at("kda_lower_bound"),
          config.at("routed_scaling_factor")};
}

AudioConfig AudioConfig::FromJson(const nlohmann::json &config) {
  return {config.at("n_audio_state"),
          config.at("n_audio_head"),
          config.at("n_audio_layer"),
          config.at("downsampler_kernel_size_list").get<std::vector<int>>(),
          config.at("downsampler_stride_list").get<std::vector<int>>(),
          config.value("whisper_chunk_size", 3000)};
}

namespace {

omni_common::WeightMap LoadWeights(const std::filesystem::path &directory,
                                   const nlohmann::json &config,
                                   const TextConfig &text_config,
                                   const AudioConfig &audio_config) {
  const auto &text = config.at("text_config");
  if (config.at("model_type") != "bailing_moe_v3_audio" ||
      config.contains("quantization") || !text.at("no_kda_lora").get<bool>() ||
      !text.at("kda_safe_gate").get<bool>() ||
      !text.at("rope_scaling").is_null() ||
      !text.at("rope_interleave").get<bool>() ||
      text.at("hidden_act") != "silu" || text.at("num_shared_experts") != 1 ||
      text.at("q_lora_rank").is_null() ||
      text_config.num_hidden_layers % text_config.layer_group_size != 0 ||
      text.at("gated_attention_proj_granularity_type") != "head_wise" ||
      text.at("scoring_func") != "sigmoid" ||
      config.at("audio_config").at("n_mels") != 128) {
    throw std::invalid_argument(
        "expected an unquantized Ming-3.0-audio-tiny checkpoint");
  } else {
  }
  auto weights = omni_common::LoadSafetensors(directory);
  const auto prediction_prefix =
      "model.layers." + std::to_string(text_config.num_hidden_layers) + ".";
  std::erase_if(weights, [&](const auto &parameter) {
    return parameter.first.starts_with("linear_proj.") ||
           parameter.first.starts_with(prediction_prefix);
  });
  for (int layer = text_config.first_k_dense_replace;
       layer < text_config.num_hidden_layers; ++layer) {
    const std::string prefix =
        "model.layers." + std::to_string(layer) + ".mlp.experts.";
    for (const auto *projection : {"gate_proj", "up_proj", "down_proj"}) {
      std::vector<mx::array> experts;
      for (int expert = 0; expert < text_config.num_experts; ++expert) {
        const auto name =
            prefix + std::to_string(expert) + "." + projection + ".weight";
        experts.push_back(weights.at(name));
        weights.erase(name);
      }
      // note (altale): gather_mm needs [expert, in, out], unlike checkpoint
      // linear weights.
      auto stacked = mx::swapaxes(mx::stack(experts), -1, -2);
      mx::async_eval({stacked});
      weights.emplace(prefix + projection + ".weight", std::move(stacked));
    }
  }
  // note (altale): Prepare fixed layouts and FP32 parameters once for the
  // inference kernels.
  for (auto &[name, weight] : weights) {
    if (name.ends_with("_conv1d.weight") ||
        name == "model.audio.conv1.weight" ||
        name == "model.audio.conv2.weight" ||
        (name.starts_with("linear_proj_audio.") && weight.ndim() == 3)) {
      // note (altale): MLX conv1d expects [out, kernel, in], not [out, in,
      // kernel].
      weight = mx::contiguous(
          mx::transpose(mx::astype(weight, mx::float32), {0, 2, 1}));
    } else if (name.ends_with(".gate.weight")) {
      weight = mx::transpose(mx::astype(weight, mx::float32));
    } else if (name.ends_with(".dt_bias")) {
      weight =
          mx::reshape(mx::astype(weight, mx::float32),
                      {text_config.num_attention_heads, text_config.head_dim});
    } else if (name.ends_with(".o_norm.weight") ||
               name.ends_with(".gate.expert_bias") ||
               name.ends_with(".attn_ln.weight") ||
               name.ends_with(".attn_ln.bias") ||
               name.ends_with(".mlp_ln.weight") ||
               name.ends_with(".mlp_ln.bias") ||
               name.starts_with("model.audio.ln_post.") ||
               name == "model.audio.conv1.bias" ||
               name == "model.audio.conv2.bias") {
      weight = mx::astype(weight, mx::float32);
    } else {
    }
  }
  for (size_t i = 0; i < audio_config.downsampler_stride_list.size(); ++i) {
    auto &bias =
        weights.at("linear_proj_audio." + std::to_string(2 * i) + ".bias");
    bias = mx::astype(bias, mx::float32);
  }
  for (int layer = 0; layer < text_config.num_hidden_layers; ++layer) {
    if ((layer + 1) % text_config.layer_group_size != 0) {
      const auto prefix =
          "model.layers." + std::to_string(layer) + ".attention";
      // note (altale): exp(A_log) is independent of input; only the gate varies
      // per token.
      const auto decay_scale = mx::exp(
          mx::reshape(mx::astype(weights.at(prefix + ".A_log"), mx::float32),
                      {text_config.num_attention_heads, 1}));
      weights.erase(prefix + ".A_log");
      weights.emplace(prefix + ".decay_scale", decay_scale);
    } else {
    }
  }
  return weights;
}

mx::array Gelu(const mx::array &hidden) {
  return mx::astype(omni_common::Gelu(mx::astype(hidden, mx::float32)),
                    hidden.dtype());
}

std::string ChatPrompt(const ChatRequest &request, int audio_tokens) {
  std::string prompt = "<role>SYSTEM</role>" + request.system_prompt;
  const bool has_thinking_instruction =
      request.system_prompt.find("detailed thinking on") != std::string::npos ||
      request.system_prompt.find("detailed thinking off") != std::string::npos;
  if (!has_thinking_instruction) {
    if (!request.system_prompt.empty() &&
        request.system_prompt.back() != '\n') {
      prompt += '\n';
    } else {
    }
    prompt += "detailed thinking off";
  } else {
  }
  prompt += "<|role_end|><role>HUMAN</role>" + request.before_audio +
            "<|audio_start|>";
  for (int i = 0; i < audio_tokens; ++i) {
    prompt += "<|audio_pad|>";
  }
  return prompt + "<|audio_end|>" + request.after_audio +
         "<|role_end|><role>ASSISTANT</role>\n<think></think>";
}

} // namespace

Model::Model(const std::filesystem::path &directory, int context_tokens)
    : Model(directory, omni_common::ReadJson(directory / "config.json"),
            context_tokens) {}

Model::Model(const std::filesystem::path &directory,
             const nlohmann::json &config, int context_tokens)
    : text_(TextConfig::FromJson(config.at("text_config"))),
      audio_(AudioConfig::FromJson(config.at("audio_config"))),
      context_tokens_(std::min(context_tokens, text_.max_position_embeddings)),
      checkpoint_(LoadWeights(directory, config, text_, audio_), config),
      tokenizer_(directory),
      stop_tokens_(omni_common::ReadJson(directory / "generation_config.json")
                       .at("eos_token_id")
                       .get<std::vector<int>>()) {}

int Model::AudioTokenCount(int mel_frames) const {
  int total = 0;
  for (int offset = 0; offset < mel_frames;
       offset += audio_.whisper_chunk_size) {
    int tokens =
        (std::min(audio_.whisper_chunk_size, mel_frames - offset) + 1) / 2;
    for (size_t i = 0; i < audio_.downsampler_stride_list.size(); ++i) {
      const int kernel = audio_.downsampler_kernel_size_list[i];
      tokens = (tokens + 2 * (kernel / 2) - kernel) /
                   audio_.downsampler_stride_list[i] +
               1;
    }
    total += tokens;
  }
  return total;
}

mx::array Model::Encode(const mx::array &mel) const {
  const int width = audio_.n_audio_state;
  const int heads = audio_.n_audio_head;
  const auto convolution = [&](const mx::array &hidden,
                               const std::string &prefix, int stride) {
    const auto &weight = checkpoint_.Weight(prefix + ".weight");
    return mx::astype(mx::add(mx::conv1d(mx::astype(hidden, mx::float32),
                                         weight, stride, weight.shape(1) / 2),
                              checkpoint_.Weight(prefix + ".bias")),
                      hidden.dtype());
  };
  const auto norm = [&](const mx::array &hidden, const std::string &prefix) {
    return mx::astype(
        checkpoint_.LayerNorm(mx::astype(hidden, mx::float32), prefix, 1e-5f),
        hidden.dtype());
  };
  std::vector<mx::array> chunks;
  const int chunk_frames = audio_.whisper_chunk_size;
  for (int offset = 0; offset < mel.shape(0); offset += chunk_frames) {
    const int frames = std::min(chunk_frames, mel.shape(0) - offset);
    auto hidden = mx::expand_dims(
        mx::astype(mx::slice(mel, {offset, 0}, {offset + frames, mel.shape(1)}),
                   mx::bfloat16),
        0);
    hidden = Gelu(convolution(hidden, "model.audio.conv1", 1));
    hidden = Gelu(convolution(hidden, "model.audio.conv2", 2));
    hidden = mx::add(hidden, mx::slice(checkpoint_.Weight(
                                           "model.audio.positional_embedding"),
                                       {0, 0}, {hidden.shape(1), width}));
    for (int layer = 0; layer < audio_.n_audio_layer; ++layer) {
      const auto prefix = "model.audio.blocks." + std::to_string(layer);
      const auto normalized = norm(hidden, prefix + ".attn_ln");
      const auto project = [&](const std::string &name) {
        return mx::transpose(
            mx::reshape(
                checkpoint_.Linear(normalized, prefix + ".attn." + name),
                {1, hidden.shape(1), heads, width / heads}),
            {0, 2, 1, 3});
      };
      const auto attended = mx::fast::scaled_dot_product_attention(
          project("query"), project("key"), project("value"),
          1.0f / std::sqrt(width / heads));
      hidden = mx::add(
          hidden,
          checkpoint_.Linear(mx::reshape(mx::transpose(attended, {0, 2, 1, 3}),
                                         {1, hidden.shape(1), width}),
                             prefix + ".attn.out"));
      hidden = mx::add(
          hidden, checkpoint_.Linear(
                      Gelu(checkpoint_.Linear(norm(hidden, prefix + ".mlp_ln"),
                                              prefix + ".mlp.0")),
                      prefix + ".mlp.2"));
    }
    hidden = norm(hidden, "model.audio.ln_post");
    const auto &strides = audio_.downsampler_stride_list;
    for (size_t i = 0; i < strides.size(); ++i) {
      hidden = Gelu(convolution(
          hidden, "linear_proj_audio." + std::to_string(2 * i), strides[i]));
    }
    hidden = checkpoint_.Linear(
        hidden, "linear_proj_audio." + std::to_string(2 * strides.size() + 1));
    const auto lengths =
        mx::astype(mx::sqrt(mx::sum(mx::square(mx::astype(hidden, mx::float32)),
                                    -1, true)),
                   hidden.dtype());
    hidden = mx::divide(
        hidden, mx::maximum(lengths, mx::array(1e-12f, hidden.dtype())));
    chunks.push_back(mx::squeeze(hidden, 0));
    mx::async_eval({chunks.back()});
  }
  return mx::concatenate(chunks, 0);
}

mx::array Model::Embed(const mx::array &token_ids) const {
  return checkpoint_.Embed(token_ids, "model.word_embeddings");
}

mx::array Model::RmsNorm(const mx::array &hidden,
                         const std::string &prefix) const {
  const auto normalized = mx::fast::rms_norm(mx::astype(hidden, mx::float32),
                                             std::nullopt, text_.rms_norm_eps);
  return mx::multiply(mx::astype(normalized, hidden.dtype()),
                      checkpoint_.Weight(prefix + ".weight"));
}

mx::array Model::Mlp(const mx::array &hidden, const std::string &prefix) const {
  const auto gate = mx::astype(
      checkpoint_.Linear(hidden, prefix + ".gate_proj"), mx::float32);
  const auto up = checkpoint_.Linear(hidden, prefix + ".up_proj");
  return checkpoint_.Linear(
      mx::multiply(mx::astype(omni_common::Silu(gate), hidden.dtype()), up),
      prefix + ".down_proj");
}

mx::array Model::Moe(const mx::array &hidden, const std::string &prefix) const {
  const int tokens = hidden.shape(1), width = hidden.shape(2);
  const int groups = text_.n_group, experts = text_.num_experts;
  const int selected_groups = text_.topk_group,
            selected_experts = text_.num_experts_per_tok;
  const auto scores = mx::sigmoid(
      mx::matmul(mx::astype(mx::reshape(hidden, {tokens, width}), mx::float32),
                 checkpoint_.Weight(prefix + ".gate.weight")));
  const auto biased =
      mx::add(scores, checkpoint_.Weight(prefix + ".gate.expert_bias"));
  const auto grouped = mx::reshape(biased, {tokens, groups, experts / groups});
  const auto group_scores = mx::sum(mx::topk(grouped, 2, -1), -1);
  const auto group_indices = mx::slice(
      mx::argpartition(mx::negative(group_scores), selected_groups - 1, -1),
      {0, 0}, {tokens, selected_groups});
  const auto selected =
      mx::put_along_axis(mx::zeros({tokens, groups}, mx::bool_), group_indices,
                         mx::array(true), -1);
  const auto group_mask =
      mx::reshape(mx::broadcast_to(mx::expand_dims(selected, -1),
                                   {tokens, groups, experts / groups}),
                  scores.shape());
  const auto selected_scores =
      mx::where(group_mask, biased, mx::array(-INFINITY));
  const auto indices = mx::slice(
      mx::argpartition(mx::negative(selected_scores), selected_experts - 1, -1),
      {0, 0}, {tokens, selected_experts});
  auto routing = mx::take_along_axis(scores, indices, -1);
  routing = mx::multiply(mx::divide(routing, mx::add(mx::sum(routing, -1, true),
                                                     mx::array(1e-20f))),
                         mx::array(text_.routed_scaling_factor));

  const auto run_experts = [&](const mx::array &rows,
                               const mx::array &row_experts, bool sorted) {
    const auto project = [&](const mx::array &input,
                             const std::string &projection) {
      return mx::gather_mm(
          input,
          checkpoint_.Weight(prefix + ".experts." + projection + ".weight"),
          std::nullopt, row_experts, sorted);
    };
    const auto gate = mx::astype(project(rows, "gate_proj"), mx::float32);
    const auto up = project(rows, "up_proj");
    return project(
        mx::multiply(mx::astype(omni_common::Silu(gate), hidden.dtype()), up),
        "down_proj");
  };
  const auto restored = omni_common::RouteToExperts(
      mx::reshape(hidden, {tokens, width}), indices, run_experts);
  const auto output =
      mx::astype(mx::sum(mx::multiply(mx::astype(restored, mx::float32),
                                      mx::expand_dims(routing, -1)),
                         -2),
                 hidden.dtype());
  return mx::add(mx::reshape(output, hidden.shape()),
                 Mlp(hidden, prefix + ".shared_experts"));
}

mx::array Model::Forward(const mx::array &embeddings,
                         std::vector<LayerCache> &caches) const {
  auto hidden = embeddings;
  for (int layer = 0; layer < text_.num_hidden_layers; ++layer) {
    const auto prefix = "model.layers." + std::to_string(layer);
    const auto normalized = RmsNorm(hidden, prefix + ".input_layernorm");
    const auto attention =
        (layer + 1) % text_.layer_group_size == 0
            ? LatentAttention(normalized, prefix + ".attention",
                              caches[layer].attention)
            : DeltaAttention(normalized, prefix + ".attention", caches[layer]);
    hidden = mx::add(hidden, attention);
    const auto mlp_input =
        RmsNorm(hidden, prefix + ".post_attention_layernorm");
    hidden = mx::add(hidden, layer < text_.first_k_dense_replace
                                 ? Mlp(mlp_input, prefix + ".mlp")
                                 : Moe(mlp_input, prefix + ".mlp"));
  }
  hidden = RmsNorm(hidden, "model.norm");
  return mx::reshape(
      checkpoint_.Linear(
          mx::slice(hidden, {0, hidden.shape(1) - 1, 0}, hidden.shape()),
          "lm_head"),
      {-1});
}

PreparedChatRequest Model::Prepare(ChatRequest request) const {
  if (request.samples.empty() ||
      !std::all_of(request.samples.begin(), request.samples.end(),
                   [](float value) { return std::isfinite(value); })) {
    throw std::invalid_argument(
        "audio must contain at least one finite sample");
  } else if (request.max_new_tokens < 1) {
    throw std::invalid_argument("max_tokens must be a positive integer");
  } else {
  }
  const int mel_frames =
      std::max(request.samples.size(), size_t(omni_common::kFftSize)) /
      omni_common::kHopLength;
  const int audio_tokens = AudioTokenCount(mel_frames);
  auto prompt = tokenizer_.Encode(ChatPrompt(request, audio_tokens));
  const int patch_token = tokenizer_.AddedTokenId("<|audio_pad|>");
  const int audio_start =
      std::find(prompt.begin(), prompt.end(), patch_token) - prompt.begin();
  if (prompt.size() + request.max_new_tokens > size_t(context_tokens_)) {
    throw std::invalid_argument(
        "prompt and generation exceed the context limit");
  } else if (std::count(prompt.begin(), prompt.end(), patch_token) !=
             audio_tokens) {
    throw std::invalid_argument("text must not contain audio placeholders");
  } else {
    return {std::move(request), std::move(prompt), audio_start, audio_tokens};
  }
}

ChatResult Model::Generate(const PreparedChatRequest &prepared,
                           const Cancellation &cancel) const {
  if (cancel()) {
    throw Cancelled();
  } else {
  }
  const auto &request = prepared.request;
  const auto mel = mx::transpose(
      omni_common::LogMel(request.samples, omni_common::FinalStftFrame::kDropped));
  const auto audio = Encode(mel);
  const auto prompt_ids =
      mx::array(prepared.prompt_ids.data(),
                {1, int(prepared.prompt_ids.size())}, mx::int32);
  auto embeddings = Embed(prompt_ids);
  embeddings = mx::slice_update(
      embeddings, mx::expand_dims(audio, 0), {0, prepared.audio_start, 0},
      {1, prepared.audio_start + prepared.audio_tokens, embeddings.shape(2)});
  std::vector<LayerCache> caches(text_.num_hidden_layers);
  ChatResult result;
  result.prompt_tokens = prepared.prompt_ids.size();
  std::vector<int> generated;
  if (cancel()) {
    throw Cancelled();
  } else {
  }
  auto next_token = mx::argmax(Forward(embeddings, caches));
  mx::async_eval({next_token});
  for (int step = 0; step < request.max_new_tokens; ++step) {
    if (cancel()) {
      throw Cancelled();
    } else {
    }
    const auto token = next_token;
    if (step + 1 < request.max_new_tokens) {
      next_token =
          mx::argmax(Forward(Embed(mx::reshape(token, {1, 1})), caches));
      mx::async_eval({next_token});
    } else {
    }
    const int token_id = token.item<int>();
    generated.push_back(token_id);
    if (std::find(stop_tokens_.begin(), stop_tokens_.end(), token_id) !=
        stop_tokens_.end()) {
      result.finish_reason = "stop";
      break;
    } else {
    }
  }
  result.text = tokenizer_.Decode(generated, true);
  result.completion_tokens = generated.size();
  return result;
}

} // namespace ming3_audio
