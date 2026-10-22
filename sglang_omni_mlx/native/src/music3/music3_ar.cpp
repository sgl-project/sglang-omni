// SPDX-License-Identifier: MIT
// Copyright (c) 2024 Prince Canuma and contributors.
#include "music3/music3_model.h"

#include <algorithm>
#include <cmath>

namespace music3 {
namespace mx = mlx::core;

namespace {

mx::array TopK(const mx::array &logits) {
  const mx::array values = mx::where(mx::isnan(logits), mx::array(-1e9f),
                                     mx::astype(logits, mx::float32));
  const int count = std::min(50, values.shape(-1));
  const mx::array threshold = mx::min(mx::topk(values, count, -1), -1, true);
  return mx::where(mx::less(values, threshold), mx::array(-1e9f), values);
}

mx::array Sample(const mx::array &logits, mx::array &key) {
  const mx::array code = mx::random::categorical(TopK(logits), -1, key);
  key = mx::random::split(key).second;
  return code;
}

mx::array Guided(const mx::array &logits) {
  const auto parts = mx::split(mx::astype(logits, mx::float32), 2, 0);
  return parts[1] + (parts[0] - parts[1]) * 1.5f;
}

} // namespace

void CheckCancelled(const std::atomic<bool> &cancelled) {
  if (cancelled.load()) {
    throw Cancelled();
  } else {
  }
}

mx::array Model::Depth(const mx::array &embeddings) const {
  const int length = embeddings.shape(1);
  const int hidden_size = embeddings.shape(2);
  const int heads = config_.at("depth_num_heads");
  const int head_dim = hidden_size / heads;
  const float epsilon = config_.at("rms_norm_eps");
  mx::array hidden =
      embeddings +
      checkpoint_.Embed(mx::arange(length), "rvq_depth_decoder.pos_embedding");
  for (int layer = 0; layer < config_.at("depth_num_layers").get<int>();
       ++layer) {
    const std::string prefix =
        "rvq_depth_decoder.layers." + std::to_string(layer);
    const mx::array normalized =
        checkpoint_.RmsNorm(hidden, prefix + ".input_layernorm", epsilon);
    const auto project = [&](const std::string &name) {
      return mx::transpose(
          mx::reshape(checkpoint_.Linear(normalized, prefix + ".attn." + name),
                      {2, length, heads, head_dim}),
          {0, 2, 1, 3});
    };
    const mx::array attended = mx::fast::scaled_dot_product_attention(
        project("to_q"), project("to_k"), project("to_v"),
        static_cast<float>(std::pow(static_cast<double>(head_dim), -0.5)),
        "causal");
    hidden = hidden + checkpoint_.Linear(
                          mx::reshape(mx::transpose(attended, {0, 2, 1, 3}),
                                      {2, length, hidden_size}),
                          prefix + ".attn.to_out");
    const mx::array mlp = checkpoint_.RmsNorm(
        hidden, prefix + ".post_attention_layernorm", epsilon);
    hidden =
        hidden +
        checkpoint_.Linear(
            qwen3_asr::Silu(checkpoint_.Linear(mlp, prefix + ".gate_proj")) *
                checkpoint_.Linear(mlp, prefix + ".up_proj"),
            prefix + ".down_proj");
  }
  return checkpoint_.RmsNorm(hidden, "rvq_depth_decoder.norm", epsilon);
}

mx::array Model::GenerateFrames(const Request &request,
                                const std::atomic<bool> &cancelled) const {
  std::vector<int> text_pair = request.token_ids;
  std::vector<int> unconditional = request.token_ids;
  if (unconditional.size() > 3) {
    std::fill(unconditional.begin() + 1, unconditional.end() - 2,
              config_.at("audio_cfg_token_id").get<int>());
  } else {
  }
  text_pair.insert(text_pair.end(), unconditional.begin(), unconditional.end());
  const int prompt_tokens = static_cast<int>(request.token_ids.size());
  const mx::array token_ids(text_pair.data(), {2, prompt_tokens}, mx::int32);
  auto caches = decoder_.NewCaches();
  const float epsilon = config_.at("rms_norm_eps");
  const auto forward = [&](const mx::array &embeddings) {
    mx::array hidden =
        checkpoint_.RmsNorm(decoder_.Forward(embeddings, caches),
                            "language_model.model.norm", epsilon);
    return mx::squeeze(
        mx::slice(hidden, {0, hidden.shape(1) - 1, 0}, hidden.shape()), 1);
  };
  mx::array last_hidden = forward(decoder_.EmbedTokens(token_ids));
  mx::array key = mx::random::key(request.seed);
  const int offset = config_.at("audio_code_offset");
  const int end_token = config_.at("audio_end_token_id");
  const int semantic_vocab = config_.at("semantic_vocab_size");
  const int audio_vocab = config_.at("audio_vocab_size");
  const mx::array vocabulary = mx::arange(config_.at("vocab_size").get<int>());
  const mx::array allowed = mx::logical_or(
      mx::logical_and(mx::greater_equal(vocabulary, mx::array(offset)),
                      mx::less(vocabulary, mx::array(offset + semantic_vocab))),
      mx::equal(vocabulary, mx::array(end_token)));
  std::vector<mx::array> frames;
  for (int frame = 0; frame <= request.max_frames; ++frame) {
    CheckCancelled(cancelled);
    const auto keys = mx::random::split(key);
    key = keys.first;
    mx::array frame_key = keys.second;
    mx::array logits =
        config_.at("tie_word_embeddings").get<bool>()
            ? checkpoint_.TiedProjection(last_hidden,
                                         "language_model.model.embed_tokens")
            : checkpoint_.Linear(last_hidden, "language_model.lm_head");
    logits =
        mx::where(allowed, mx::astype(logits, mx::float32), mx::array(-1e9f));
    const mx::array conditional = mx::split(logits, 2, 0)[0];
    const mx::array threshold =
        mx::min(mx::topk(conditional, 50, -1), -1, true);
    logits = mx::where(mx::less(conditional, threshold), mx::array(-1e9f),
                       Guided(logits));
    const mx::array sampled =
        Sample(mx::where(allowed, logits, mx::array(-1e9f)), frame_key);
    mx::eval(sampled);
    if (sampled.item<int>() == end_token) {
      break;
    } else {
    }
    const mx::array semantic = mx::concatenate({sampled, sampled}, 0) - offset;
    std::vector<mx::array> sequence = {
        mx::expand_dims(
            checkpoint_.Linear(last_hidden, "rvq_depth_decoder.projection"), 1),
        mx::expand_dims(
            checkpoint_.Linear(decoder_.EmbedTokens(semantic + offset),
                               "rvq_depth_decoder.projection"),
            1)};
    std::vector<mx::array> hidden_parts = {mx::split(last_hidden, 2, 0)[0]};
    std::vector<mx::array> residual_embeddings;
    for (int index = 1; index < 8; ++index) {
      const mx::array hidden = Depth(mx::concatenate(sequence, 1));
      const mx::array last = mx::squeeze(
          mx::slice(hidden, {0, hidden.shape(1) - 1, 0}, hidden.shape()), 1);
      hidden_parts.push_back(mx::split(last, 2, 0)[0]);
      const mx::array code = Sample(
          Guided(checkpoint_.Linear(last, "rvq_depth_decoder.audio_heads." +
                                              std::to_string(index - 1))),
          frame_key);
      const mx::array duplicated = mx::concatenate({code, code}, 0);
      const mx::array embedding =
          checkpoint_.Embed(duplicated + (index - 1) * audio_vocab,
                            "rvq_depth_decoder.audio_embeddings");
      residual_embeddings.push_back(embedding);
      if (index < 7) {
        sequence.push_back(mx::expand_dims(
            checkpoint_.Linear(embedding, "rvq_depth_decoder.projection"), 1));
      } else {
      }
    }
    // Note (Akazaakane): the first prediction primes feedback but supplies no
    // acoustic frame.
    if (frame > 0) {
      frames.push_back(mx::concatenate(hidden_parts, -1));
      mx::eval(frames.back());
    } else {
    }
    const mx::array feedback = mx::expand_dims(
        (decoder_.EmbedTokens(semantic + offset) +
         mx::astype(mx::sum(mx::stack(residual_embeddings, 1), 1),
                    last_hidden.dtype())) *
            mx::array(static_cast<float>(std::pow(8.0, -0.5)),
                      last_hidden.dtype()),
        1);
    last_hidden = forward(feedback);
    mx::eval(last_hidden);
    if (static_cast<int>(frames.size()) >= request.max_frames) {
      break;
    } else {
    }
  }
  if (frames.empty()) {
    throw std::runtime_error("Music3 generated zero audio frames");
  } else {
    // Note (Akazaakane): preserve the reference's FP16 handoff to the acoustic
    // stage.
    return mx::astype(mx::stack(frames, 1), mx::float16);
  }
}

} // namespace music3
