// SPDX-License-Identifier: Apache-2.0
// Copyright 2025 Antgroup and The HuggingFace Inc. team. All rights reserved.
#include "ming3_audio/ming3_audio_model.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "common/audio.h"
#include "qwen3_asr/qwen3_audio.h"

namespace ming3_audio {
namespace mx = mlx::core;

namespace {

omni_common::WeightMap LoadWeights(const std::filesystem::path &directory,
                                   const nlohmann::json &config) {
  const auto &text = config.at("text_config");
  if (config.at("model_type") != "bailing_moe_v3_audio" ||
      config.contains("quantization") || !text.at("no_kda_lora").get<bool>() ||
      !text.at("kda_safe_gate").get<bool>() ||
      !text.at("rope_scaling").is_null() ||
      !text.at("rope_interleave").get<bool>() ||
      text.at("hidden_act") != "silu" || text.at("num_shared_experts") != 1 ||
      text.at("q_lora_rank").is_null() ||
      text.at("num_hidden_layers").get<int>() %
              text.at("layer_group_size").get<int>() !=
          0 ||
      text.at("gated_attention_proj_granularity_type") != "head_wise" ||
      text.at("scoring_func") != "sigmoid" ||
      config.at("audio_config").at("n_mels") != 128) {
    throw std::invalid_argument(
        "expected an unquantized Ming-3.0-audio-tiny checkpoint");
  } else {
  }
  auto weights = omni_common::LoadSafetensors(directory);
  const auto prediction_prefix =
      "model.layers." +
      std::to_string(text.at("num_hidden_layers").get<int>()) + ".";
  std::erase_if(weights, [&](const auto &parameter) {
    return parameter.first.starts_with("linear_proj.") ||
           parameter.first.starts_with(prediction_prefix);
  });
  for (int layer = text.at("first_k_dense_replace");
       layer < text.at("num_hidden_layers").get<int>(); ++layer) {
    const std::string prefix =
        "model.layers." + std::to_string(layer) + ".mlp.experts.";
    for (const auto *projection : {"gate_proj", "up_proj", "down_proj"}) {
      std::vector<mx::array> experts;
      for (int expert = 0; expert < text.at("num_experts").get<int>();
           ++expert) {
        const auto name =
            prefix + std::to_string(expert) + "." + projection + ".weight";
        experts.push_back(weights.at(name));
        weights.erase(name);
      }
      auto stacked = mx::stack(experts);
      mx::eval(stacked);
      weights.emplace(prefix + projection + ".weight", std::move(stacked));
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

Model::Model(const std::filesystem::path &directory)
    : Model(directory, omni_common::ReadJson(directory / "config.json")) {}

Model::Model(const std::filesystem::path &directory,
             const nlohmann::json &config)
    : text_(config.at("text_config")), audio_(config.at("audio_config")),
      checkpoint_(LoadWeights(directory, config), config),
      tokenizer_(directory),
      stop_tokens_(omni_common::ReadJson(directory / "generation_config.json")
                       .at("eos_token_id")
                       .get<std::vector<int>>()) {}

int Model::AudioTokenCount(int mel_frames) const {
  int tokens = (mel_frames + 1) / 2;
  for (const int stride : audio_.at("downsampler_stride_list")) {
    tokens = (tokens + stride - 1) / stride;
  }
  return tokens;
}

mx::array Model::Encode(const mx::array &mel,
                        omni_common::WeightMap *trace) const {
  const int width = audio_.at("n_audio_state");
  const int heads = audio_.at("n_audio_head");
  const auto convolution = [&](const mx::array &hidden,
                               const std::string &prefix, int stride) {
    const auto &weight = checkpoint_.Weight(prefix + ".weight");
    return mx::astype(
        mx::add(mx::conv1d(
                    mx::astype(hidden, mx::float32),
                    mx::transpose(mx::astype(weight, mx::float32), {0, 2, 1}),
                    stride, weight.shape(2) / 2),
                mx::astype(checkpoint_.Weight(prefix + ".bias"), mx::float32)),
        hidden.dtype());
  };
  const auto norm = [&](const mx::array &hidden, const std::string &prefix) {
    return mx::astype(
        checkpoint_.LayerNorm(mx::astype(hidden, mx::float32), prefix, 1e-5f),
        hidden.dtype());
  };
  std::vector<mx::array> chunks;
  constexpr int chunk_frames = 3000;
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
    if (trace != nullptr) {
      trace->emplace("encoder_input", hidden);
    } else {
    }
    for (int layer = 0; layer < audio_.at("n_audio_layer").get<int>();
         ++layer) {
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
      mx::eval(hidden);
      if (trace != nullptr) {
        trace->emplace("encoder_layer_" + std::to_string(layer), hidden);
      } else {
      }
    }
    hidden = norm(hidden, "model.audio.ln_post");
    const auto strides =
        audio_.at("downsampler_stride_list").get<std::vector<int>>();
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
  }
  return mx::concatenate(chunks, 0);
}

mx::array Model::Embed(const mx::array &token_ids) const {
  return checkpoint_.Embed(token_ids, "model.word_embeddings");
}

mx::array Model::RmsNorm(const mx::array &hidden,
                         const std::string &prefix) const {
  const auto normalized =
      mx::fast::rms_norm(mx::astype(hidden, mx::float32), std::nullopt,
                         text_.at("rms_norm_eps").get<float>());
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
  const int groups = text_.at("n_group"), experts = text_.at("num_experts");
  const int selected_groups = text_.at("topk_group"),
            selected_experts = text_.at("num_experts_per_tok");
  const auto scores = mx::sigmoid(mx::matmul(
      mx::astype(hidden, mx::float32),
      mx::transpose(mx::astype(checkpoint_.Weight(prefix + ".gate.weight"),
                               mx::float32))));
  const auto biased =
      mx::add(scores, checkpoint_.Weight(prefix + ".gate.expert_bias"));
  const auto grouped =
      mx::reshape(biased, {hidden.shape(1), groups, experts / groups});
  const auto sorted = mx::sort(grouped, -1);
  const auto group_scores = mx::sum(
      mx::slice(sorted, {0, 0, experts / groups - 2}, sorted.shape()), -1);
  const auto group_order =
      mx::argsort(mx::argsort(mx::negative(group_scores), -1), -1);
  const auto group_mask = mx::reshape(
      mx::broadcast_to(
          mx::expand_dims(mx::less(group_order, mx::array(selected_groups)),
                          -1),
          {hidden.shape(1), groups, experts / groups}),
      scores.shape());
  const auto selected_scores =
      mx::where(group_mask, biased, mx::array(-INFINITY));
  const auto indices =
      mx::slice(mx::argsort(mx::negative(selected_scores), -1), {0, 0, 0},
                {1, hidden.shape(1), selected_experts});
  auto routing = mx::take_along_axis(scores, indices, -1);
  routing =
      mx::multiply(mx::divide(routing, mx::add(mx::sum(routing, -1, true),
                                               mx::array(1e-20f))),
                   mx::array(text_.at("routed_scaling_factor").get<float>()));
  const auto project = [&](const mx::array &input,
                           const std::string &projection) {
    return mx::gather_mm(
        input,
        mx::swapaxes(
            checkpoint_.Weight(prefix + ".experts." + projection + ".weight"),
            -1, -2),
        std::nullopt, indices);
  };
  const auto expanded =
      mx::reshape(hidden, {1, hidden.shape(1), 1, 1, hidden.shape(2)});
  const auto gate = mx::astype(project(expanded, "gate_proj"), mx::float32);
  const auto up = project(expanded, "up_proj");
  const auto routed = mx::squeeze(
      project(
          mx::multiply(mx::astype(omni_common::Silu(gate), hidden.dtype()), up),
          "down_proj"),
      -2);
  const auto output =
      mx::astype(mx::sum(mx::multiply(mx::astype(routed, mx::float32),
                                      mx::expand_dims(routing, -1)),
                         -2),
                 hidden.dtype());
  return mx::add(output, Mlp(hidden, prefix + ".shared_experts"));
}

mx::array Model::Forward(const mx::array &embeddings,
                         std::vector<LayerCache> &caches,
                         omni_common::WeightMap *trace) const {
  auto hidden = embeddings;
  for (int layer = 0; layer < text_.at("num_hidden_layers").get<int>();
       ++layer) {
    const auto prefix = "model.layers." + std::to_string(layer);
    const auto normalized = RmsNorm(hidden, prefix + ".input_layernorm");
    const auto attention =
        (layer + 1) % text_.at("layer_group_size").get<int>() == 0
            ? LatentAttention(normalized, prefix + ".attention",
                              caches[layer].attention)
            : DeltaAttention(normalized, prefix + ".attention", caches[layer]);
    hidden = mx::add(hidden, attention);
    const auto mlp_input =
        RmsNorm(hidden, prefix + ".post_attention_layernorm");
    hidden =
        mx::add(hidden, layer < text_.at("first_k_dense_replace").get<int>()
                            ? Mlp(mlp_input, prefix + ".mlp")
                            : Moe(mlp_input, prefix + ".mlp"));
    mx::eval(hidden);
    if (trace != nullptr) {
      trace->emplace("layer_" + std::to_string(layer), hidden);
    } else {
    }
  }
  hidden = RmsNorm(hidden, "model.norm");
  return mx::reshape(
      checkpoint_.Linear(
          mx::slice(hidden, {0, hidden.shape(1) - 1, 0}, hidden.shape()),
          "lm_head"),
      {-1});
}

std::vector<LayerCache> Model::NewCaches() const {
  return std::vector<LayerCache>(text_.at("num_hidden_layers").get<int>());
}

ChatResult Model::Generate(const ChatRequest &request,
                           const chat_service::Cancellation &cancel,
                           omni_common::WeightMap *trace) const {
  if (request.samples.size() < omni_common::kFftSize ||
      request.samples.size() > omni_common::kWhisperWindowSampleCount ||
      !std::all_of(request.samples.begin(), request.samples.end(),
                   [](float value) { return std::isfinite(value); })) {
    throw std::invalid_argument(
        "audio must contain 400 to 480000 finite samples");
  } else if (request.max_new_tokens < 1 || request.max_new_tokens > 1024) {
    throw std::invalid_argument("max_tokens must be between 1 and 1024");
  } else {
  }
  const int mel_frames = request.samples.size() / omni_common::kHopLength;
  const int audio_tokens = AudioTokenCount(mel_frames);
  const auto prompt = tokenizer_.Encode(ChatPrompt(request, audio_tokens));
  const int patch_token = tokenizer_.AddedTokenId("<|audio_pad|>");
  const int audio_start =
      std::find(prompt.begin(), prompt.end(), patch_token) - prompt.begin();
  if (prompt.size() + request.max_new_tokens >
      size_t(std::min(8192, max_context_tokens()))) {
    throw std::invalid_argument(
        "prompt and generation exceed the context limit");
  } else if (std::count(prompt.begin(), prompt.end(), patch_token) !=
             audio_tokens) {
    throw std::invalid_argument("text must not contain audio placeholders");
  } else if (cancel()) {
    throw chat_service::Cancelled();
  } else {
  }
  const auto mel = mx::transpose(
      qwen3_asr::LogMel(request.samples, qwen3_asr::AudioLayout::kReference));
  const auto audio = Encode(mel, trace);
  const auto prompt_ids =
      mx::array(prompt.data(), {1, int(prompt.size())}, mx::int32);
  auto embeddings = Embed(prompt_ids);
  embeddings = mx::slice_update(
      embeddings, mx::expand_dims(audio, 0), {0, audio_start, 0},
      {1, audio_start + audio_tokens, embeddings.shape(2)});
  auto caches = NewCaches();
  ChatResult result;
  result.prompt_tokens = prompt.size();
  std::vector<int> generated;
  if (trace != nullptr) {
    trace->emplace("mel", mel);
    trace->emplace("audio_embeddings", audio);
    trace->emplace("prompt_ids", prompt_ids);
  } else {
  }
  for (int step = 0; step < request.max_new_tokens; ++step) {
    if (cancel()) {
      throw chat_service::Cancelled();
    } else {
    }
    const auto logits =
        Forward(embeddings, caches, step == 0 ? trace : nullptr);
    if (step == 0 && trace != nullptr) {
      trace->emplace("logits", logits);
    } else {
    }
    const int token = mx::argmax(logits).item<int>();
    generated.push_back(token);
    if (std::find(stop_tokens_.begin(), stop_tokens_.end(), token) !=
        stop_tokens_.end()) {
      result.finish_reason = "stop";
      break;
    } else {
      embeddings = Embed(mx::reshape(mx::array(token, mx::int32), {1, 1}));
    }
  }
  result.text = tokenizer_.Decode(generated, true);
  result.completion_tokens = generated.size();
  if (trace != nullptr) {
    trace->emplace(
        "generated_ids",
        mx::array(generated.data(), {int(generated.size())}, mx::int32));
  } else {
  }
  return result;
}

} // namespace ming3_audio
