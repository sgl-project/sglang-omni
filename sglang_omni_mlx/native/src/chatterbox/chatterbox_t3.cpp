// SPDX-License-Identifier: Apache-2.0
// Chatterbox-Turbo's T3 decoder and speech-token sampler.
#include "chatterbox/chatterbox_t3.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>

namespace chatterbox {

namespace mx = mlx::core;

namespace {

constexpr int kLayerCount = 24;
constexpr int kHeadCount = 16;
constexpr int kHiddenSize = 1024;
constexpr int kHeadDim = kHiddenSize / kHeadCount;
constexpr int kSpeakerEmbeddingSize = 256;
constexpr float kLayerNormEps = 1e-5f;
constexpr int kSpeechVocabularySize = kStopSpeechToken + 1;

std::string LayerPrefix(int layer) {
  return "t3.tfmr.h." + std::to_string(layer);
}

} // namespace

T3::T3(const omni_common::Checkpoint &checkpoint)
    : checkpoint_(checkpoint),
      gelu_scale_(mx::sqrt(mx::array(static_cast<float>(2.0 / M_PI)))) {}

mx::array T3::Block(const mx::array &x, int layer,
                    qwen3_asr::KVCache &cache) const {
  const std::string prefix = LayerPrefix(layer);
  const int positions = x.shape(1);
  const mx::array normed =
      checkpoint_.LayerNorm(x, prefix + ".ln_1", kLayerNormEps);
  const std::vector<mx::array> qkv =
      mx::split(checkpoint_.Linear(normed, prefix + ".attn.c_attn"), 3, -1);
  const auto heads = [&](const mx::array &projection) {
    return mx::transpose(
        mx::reshape(projection, {1, positions, kHeadCount, kHeadDim}),
        {0, 2, 1, 3});
  };
  const mx::array q = heads(qkv[0]);
  const auto [k, v] = cache.UpdateAndFetch(heads(qkv[1]), heads(qkv[2]));
  const int key_count = k.shape(2);
  mx::array scores =
      mx::multiply(mx::matmul(q, mx::transpose(k, {0, 1, 3, 2})),
                   mx::array(1.0f / std::sqrt(static_cast<float>(kHeadDim))));
  scores = mx::add(scores,
                   mx::triu(mx::full({positions, key_count},
                                     -std::numeric_limits<float>::infinity()),
                            key_count - positions + 1));
  const mx::array attention = mx::reshape(
      mx::transpose(mx::matmul(mx::softmax(scores, -1), v), {0, 2, 1, 3}),
      {1, positions, kHiddenSize});
  const mx::array hidden =
      mx::add(x, checkpoint_.Linear(attention, prefix + ".attn.c_proj"));

  const mx::array inner = checkpoint_.Linear(
      checkpoint_.LayerNorm(hidden, prefix + ".ln_2", kLayerNormEps),
      prefix + ".mlp.c_fc");
  // Note (Jiaxin Deng): the tanh GELU written out, in this order, so it
  // rounds as the checkpoint's model does.
  const mx::array cubic =
      mx::multiply(mx::array(0.044715f), mx::power(inner, mx::array(3.0f)));
  const mx::array activated = mx::multiply(
      mx::multiply(mx::array(0.5f), inner),
      mx::add(mx::array(1.0f),
              mx::tanh(mx::multiply(gelu_scale_, mx::add(inner, cubic)))));
  return mx::add(hidden, checkpoint_.Linear(activated, prefix + ".mlp.c_proj"));
}

mx::array T3::LastLogits(const mx::array &embeddings,
                         std::vector<qwen3_asr::KVCache> &caches) const {
  const int positions = embeddings.shape(1);
  const int offset = caches.front().offset();
  mx::array hidden = mx::add(
      embeddings,
      checkpoint_.Embed(mx::arange(offset, offset + positions, mx::int32),
                        "t3.tfmr.wpe"));
  for (int layer = 0; layer < kLayerCount; ++layer) {
    hidden = Block(hidden, layer, caches[layer]);
  }
  hidden = checkpoint_.LayerNorm(hidden, "t3.tfmr.ln_f", kLayerNormEps);
  const mx::array last =
      mx::slice(hidden, {0, positions - 1, 0}, {1, positions, kHiddenSize});
  return mx::reshape(checkpoint_.Linear(last, "t3.speech_head"),
                     {1, kSpeechVocabularySize});
}

mx::array T3::Sample(mx::array logits, const std::vector<float> *seen_tokens,
                     const SamplingOptions &options) const {
  const float infinity = std::numeric_limits<float>::infinity();
  if (seen_tokens != nullptr && options.repetition_penalty != 1.0f) {
    const mx::array penalty(options.repetition_penalty);
    const mx::array penalized =
        mx::where(mx::less(logits, mx::array(0.0f)),
                  mx::multiply(logits, penalty), mx::divide(logits, penalty));
    const mx::array seen(seen_tokens->begin(), {1, kSpeechVocabularySize},
                         mx::float32);
    logits = mx::where(mx::greater(seen, mx::array(0.0f)), penalized, logits);
  } else {
  }
  if (options.temperature > 0.0f && options.temperature != 1.0f) {
    logits = mx::divide(logits, mx::array(options.temperature));
  } else {
  }
  if (options.top_k > 0) {
    const int top_k = std::min(options.top_k, kSpeechVocabularySize);
    const mx::array kth_index =
        mx::slice(mx::argpartition(logits, -top_k, -1),
                  {0, kSpeechVocabularySize - top_k},
                  {1, kSpeechVocabularySize - top_k + 1});
    logits = mx::where(
        mx::greater_equal(logits, mx::take_along_axis(logits, kth_index, -1)),
        logits, mx::array(-infinity));
  } else {
  }
  if (options.top_p < 1.0f) {
    const mx::array sorted_indices = mx::argsort(mx::negative(logits), -1);
    const mx::array sorted_logits =
        mx::take_along_axis(logits, sorted_indices, -1);
    const mx::array above =
        mx::greater(mx::cumsum(mx::softmax(sorted_logits, -1), -1),
                    mx::array(options.top_p));
    // The first token past top_p stays, so at least one always does.
    const mx::array removed = mx::concatenate(
        {mx::zeros({1, 1}, mx::bool_),
         mx::slice(above, {0, 0}, {1, kSpeechVocabularySize - 1})},
        -1);
    logits = mx::take_along_axis(
        mx::where(removed, mx::array(-infinity), sorted_logits),
        mx::argsort(sorted_indices, -1), -1);
  } else {
  }
  return mx::random::categorical(logits);
}

qwen3_asr::FinishReason
T3::Generate(const T3Voice &voice, const std::vector<int> &text_ids,
             const SamplingOptions &options, int max_tokens,
             const std::atomic<bool> &cancel,
             const std::function<void(int)> &on_token) const {
  const mx::array speaker = mx::expand_dims(
      checkpoint_.Linear(
          mx::reshape(voice.speaker_embedding, {-1, kSpeakerEmbeddingSize}),
          "t3.cond_enc.spkr_enc"),
      1);
  const mx::array text = mx::array(
      text_ids.begin(), {1, static_cast<int>(text_ids.size())}, mx::int32);
  const mx::array prompt = mx::concatenate(
      {speaker, checkpoint_.Embed(voice.prompt_speech_tokens, "t3.speech_emb"),
       checkpoint_.Embed(text, "t3.text_emb"),
       checkpoint_.Embed(mx::full({1, 1}, kStartSpeechToken, mx::int32),
                         "t3.speech_emb")},
      1);

  std::vector<qwen3_asr::KVCache> caches(kLayerCount);
  std::vector<float> seen_tokens(kSpeechVocabularySize, 0.0f);
  mx::array token = Sample(LastLogits(prompt, caches), nullptr, options);
  for (int sampled = 1;; ++sampled) {
    if (cancel.load()) {
      throw qwen3_asr::TranscriptionCancelled();
    } else {
    }
    std::optional<mx::array> next_logits;
    if (sampled < max_tokens) {
      // Note (Jiaxin Deng): the next step runs on the GPU while this token
      // is read; it draws no random key until the token is known not to stop.
      next_logits = LastLogits(
          checkpoint_.Embed(mx::reshape(token, {1, 1}), "t3.speech_emb"),
          caches);
      mx::async_eval(*next_logits);
    } else {
    }
    mx::eval(token);
    const int token_id = static_cast<int>(token.item<uint32_t>());
    if (token_id == kStopSpeechToken) {
      return qwen3_asr::FinishReason::kStop;
    } else {
    }
    on_token(token_id);
    if (!next_logits.has_value()) {
      return qwen3_asr::FinishReason::kLength;
    } else {
    }
    seen_tokens[token_id] = 1.0f;
    token = Sample(*next_logits, &seen_tokens, options);
  }
}

} // namespace chatterbox
