// SPDX-License-Identifier: Apache-2.0
#include "fish_s2_pro/fish_s2_pro_model.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace fish_s2_pro {

namespace mx = mlx::core;

namespace {

constexpr const char *kSlowPrefix = "model";
constexpr const char *kFastPrefix = "model.fast";
constexpr const char *kImEndToken = "<|im_end|>";
constexpr const char *kFirstSemanticToken = "<|semantic:0|>";
// Note (Jiaxin Deng): the CUDA sampler's constants (sglang_model.py): a fixed
// 30-candidate top-k, a 16-token repetition history, and the hotter sampling
// it switches to when the last 4 semantic tokens repeat one.
constexpr int kSamplerCandidates = 30;
constexpr size_t kRepetitionHistory = 16;
constexpr size_t kRepeatWindow = 4;
constexpr float kRepeatTemperature = 1.0f;
constexpr float kRepeatTopP = 0.9f;
constexpr float kMinTemperature = 1e-5f;

// SGLang's MurmurHash3 x86_32 of a 64-bit seed, a step and a column.
uint32_t MurmurHash(uint64_t seed, uint32_t step, uint32_t column) {
  uint32_t hash = 0;
  const auto mix = [&hash](uint32_t block) {
    block *= 0xCC9E2D51u;
    block = (block << 15) | (block >> 17);
    block *= 0x1B873593u;
    hash ^= block;
    hash = (hash << 13) | (hash >> 19);
    hash = hash * 5 + 0xE6546B64u;
  };
  mix(static_cast<uint32_t>(seed & 0xFFFFFFFFu));
  mix(static_cast<uint32_t>(seed >> 32));
  mix(step);
  mix(column);
  hash ^= 16;
  hash ^= hash >> 16;
  hash *= 0x85EBCA6Bu;
  hash ^= hash >> 13;
  hash *= 0xC2B2AE35u;
  hash ^= hash >> 16;
  return hash;
}

// Rotates adjacent pairs of heads [batch, heads, length, head_dim] in float32
// by the bfloat16 phases cosines and sines [1, 1, length, head_dim / 2].
std::vector<mx::array> RotateGraph(const std::vector<mx::array> &inputs) {
  const mx::array &heads = inputs[0];
  const mx::array &cosines = inputs[1];
  const mx::array &sines = inputs[2];
  const int length = heads.shape(2);
  const int half = heads.shape(3) / 2;
  const mx::array pairs =
      mx::reshape(mx::astype(heads, mx::float32),
                  {heads.shape(0), heads.shape(1), length, half, 2});
  const mx::Shape stop = {pairs.shape(0), pairs.shape(1), length, half};
  const mx::array even = mx::squeeze(
      mx::slice(pairs, {0, 0, 0, 0, 0}, {stop[0], stop[1], length, half, 1}),
      -1);
  const mx::array odd = mx::squeeze(
      mx::slice(pairs, {0, 0, 0, 0, 1}, {stop[0], stop[1], length, half, 2}),
      -1);
  return {mx::astype(
      mx::reshape(mx::stack({mx::subtract(mx::multiply(even, cosines),
                                          mx::multiply(odd, sines)),
                             mx::add(mx::multiply(odd, cosines),
                                     mx::multiply(even, sines))},
                            -1),
                  heads.shape()),
      heads.dtype())};
}

// float32 softmax of values, in place; each exponential is rounded from
// float64, so the result does not depend on the platform's expf.
void Softmax(std::vector<float> &values) {
  const float largest = *std::max_element(values.begin(), values.end());
  float total = 0.0f;
  for (float &value : values) {
    value = static_cast<float>(std::exp(static_cast<double>(value - largest)));
    total += value;
  }
  for (float &value : values) {
    value /= total;
  }
}

} // namespace

FishS2ProModel::FishS2ProModel(const std::filesystem::path &model_directory)
    : FishS2ProModel(model_directory,
                     omni_common::ReadJson(model_directory / "config.json")) {}

FishS2ProModel::FishS2ProModel(const std::filesystem::path &model_directory,
                               const nlohmann::json &config)
    : checkpoint_(
          [&] {
            auto [weights, metadata] = mx::load_safetensors(
                (model_directory / "model.safetensors").string());
            return omni_common::WeightMap(weights.begin(), weights.end());
          }(),
          config),
      codec_([&] {
        auto [weights, metadata] = mx::load_safetensors(
            (model_directory / "codec.safetensors").string());
        return omni_common::WeightMap(weights.begin(), weights.end());
      }()),
      tokenizer_(model_directory),
      slow_rotary_{mx::array(0.0f), mx::array(0.0f)},
      fast_rotary_{mx::array(0.0f), mx::array(0.0f)} {
  const nlohmann::json &text = config.at("text_config");
  const nlohmann::json &audio = config.at("audio_decoder_config");
  if (text.value("use_moe", false) || audio.value("use_moe", false) ||
      !text.value("tie_word_embeddings", true) ||
      text.at("dim") != audio.at("dim")) {
    throw std::runtime_error("unsupported Fish S2-Pro checkpoint layout");
  } else {
  }
  const auto transformer = [](const std::string &prefix,
                              const nlohmann::json &settings) {
    return TransformerConfig{prefix,
                             settings.at("n_layer").get<int>(),
                             settings.at("n_head").get<int>(),
                             settings.at("n_local_heads").get<int>(),
                             settings.at("head_dim").get<int>(),
                             settings.at("norm_eps").get<float>(),
                             settings.at("attention_qk_norm").get<bool>()};
  };
  slow_ = transformer(kSlowPrefix, text);
  fast_ = transformer(kFastPrefix, audio);
  codebook_count_ = audio.at("num_codebooks").get<int>();
  codebook_size_ = audio.at("vocab_size").get<int>();
  max_positions_ = text.at("max_seq_len").get<int>();
  slow_rotary_ = BuildRotaryTable(max_positions_, slow_.head_dim,
                                  text.at("rope_base").get<float>());
  fast_rotary_ = BuildRotaryTable(codebook_count_, fast_.head_dim,
                                  audio.at("rope_base").get<float>());
  semantic_begin_id_ = tokenizer_.AddedTokenId(kFirstSemanticToken);
  semantic_end_id_ = semantic_begin_id_ + codebook_size_ - 1;
  im_end_id_ = tokenizer_.AddedTokenId(kImEndToken);
  if (semantic_begin_id_ != config.at("semantic_start_token_id").get<int>() ||
      semantic_end_id_ != config.at("semantic_end_token_id").get<int>() ||
      tokenizer_.AddedTokenId(
          "<|semantic:" + std::to_string(codebook_size_ - 1) + "|>") !=
          semantic_end_id_ ||
      im_end_id_ < 0) {
    throw std::runtime_error(
        "the tokenizer's semantic tokens do not match config.json");
  } else {
  }
  mx::eval({slow_rotary_.cosines, slow_rotary_.sines, fast_rotary_.cosines,
            fast_rotary_.sines});
  // Note (Jiaxin Deng): the codec's raw weight-norm tensors, freed once
  // folded, would otherwise stay in MLX's buffer cache for the process's life.
  mx::clear_cache();
}

FishS2ProModel::RotaryTable
FishS2ProModel::BuildRotaryTable(int positions, int head_dim, float base) {
  const mx::array frequencies = mx::divide(
      mx::array(1.0f),
      mx::power(mx::array(base),
                mx::divide(mx::arange(0, head_dim, 2, mx::float32),
                           mx::array(static_cast<float>(head_dim)))));
  const mx::array angles =
      mx::multiply(mx::reshape(mx::arange(positions, mx::float32), {-1, 1}),
                   mx::reshape(frequencies, {1, -1}));
  // Note (Jiaxin Deng): the checkpoint's reference caches the rotary phases in
  // bfloat16, and the model was trained with them.
  return {mx::astype(mx::cos(angles), mx::bfloat16),
          mx::astype(mx::sin(angles), mx::bfloat16)};
}

SpeechPrompt FishS2ProModel::BuildPrompt(
    const std::string &text,
    const std::vector<VoiceReference> &references) const {
  SpeechPrompt prompt;
  // Each part is tokenized on its own, as the CUDA server's prompt encoder
  // does.
  const auto append_text = [&](const std::string &part) {
    for (const int id : tokenizer_.Encode(part)) {
      prompt.ids.push_back(id);
      prompt.is_audio.push_back(false);
    }
  };
  if (!references.empty()) {
    append_text("<|im_start|>system\n");
    append_text("convert the provided text to speech reference to the "
                "following:\n\nText:\n");
    for (const VoiceReference &reference : references) {
      if (!reference.text.empty()) {
        append_text("<|speaker:0|>" + reference.text);
      } else {
      }
    }
    append_text("\n\nSpeech:\n");
    for (const VoiceReference &reference : references) {
      if (reference.codes.ndim() != 2 ||
          reference.codes.shape(0) != codebook_count_) {
        throw std::invalid_argument("reference codes must have shape (" +
                                    std::to_string(codebook_count_) + ", T)");
      } else {
      }
      const mx::array codes = mx::astype(reference.codes, mx::int32);
      mx::eval(codes);
      const int32_t *semantic = codes.data<int32_t>();
      for (int frame = 0; frame < codes.shape(1); ++frame) {
        prompt.ids.push_back(semantic_begin_id_ + semantic[frame]);
        prompt.is_audio.push_back(true);
      }
      prompt.audio_codes.push_back(codes);
    }
    append_text(std::string(kImEndToken) + "\n");
  } else {
  }
  append_text("<|im_start|>user\n");
  append_text("<|speaker:0|>" + text);
  append_text(std::string(kImEndToken) + "\n");
  append_text("<|im_start|>assistant\n<|voice|>");
  return prompt;
}

mx::array FishS2ProModel::Embed(const mx::array &ids,
                                const mx::array &codebook_rows,
                                const mx::array &is_audio) const {
  std::vector<mx::array> codebook_embeddings;
  for (int codebook = 0; codebook < codebook_count_; ++codebook) {
    const mx::array row = mx::slice(codebook_rows, {codebook, 0},
                                    {codebook + 1, codebook_rows.shape(1)});
    codebook_embeddings.push_back(
        checkpoint_.Embed(mx::add(row, mx::array(codebook * codebook_size_)),
                          "model.codebook_embeddings"));
  }
  const mx::array audio_mask = mx::reshape(is_audio, {1, -1, 1});
  const mx::array token_embeddings = checkpoint_.Embed(ids, "model.embeddings");
  const mx::array mixed = mx::add(
      token_embeddings,
      mx::where(audio_mask, mx::sum(mx::stack(codebook_embeddings, 0), 0),
                mx::array(0, token_embeddings.dtype())));
  // Note (Jiaxin Deng): a mixed row is scaled as the reference divides it,
  // by sqrt(codebooks + 1) in the embeddings' dtype.
  return mx::where(
      audio_mask,
      mx::divide(mixed,
                 mx::array(static_cast<float>(std::sqrt(codebook_count_ + 1.0)),
                           mixed.dtype())),
      mixed);
}

mx::array
FishS2ProModel::Forward(const mx::array &x, const TransformerConfig &config,
                        const RotaryTable &rotary,
                        std::vector<qwen3_asr::KVCache> &caches) const {
  const int batch = x.shape(0);
  const int length = x.shape(1);
  const int half = config.head_dim / 2;
  const int query_size = config.head_count * config.head_dim;
  const int key_value_size = config.key_value_head_count * config.head_dim;
  const auto layer_prefix = [&](int layer) {
    // The fast transformer's weights are fast_layers, fast_norm.
    return config.prefix +
           (config.prefix == kSlowPrefix ? ".layers." : "_layers.") +
           std::to_string(layer);
  };
  const int offset = caches[0].offset();
  const mx::Shape table_stop = {offset + length, half};
  const mx::array cosines = mx::reshape(
      mx::slice(rotary.cosines, {offset, 0}, table_stop), {1, 1, length, half});
  const mx::array sines = mx::reshape(
      mx::slice(rotary.sines, {offset, 0}, table_stop), {1, 1, length, half});
  const auto rotate = [&](const mx::array &heads) {
    static const auto compiled = mx::compile(RotateGraph);
    return compiled({heads, cosines, sines})[0];
  };
  const auto split_heads = [&](const mx::array &projection, int start, int size,
                               int head_count) {
    return mx::transpose(
        mx::reshape(
            mx::slice(projection, {0, 0, start}, {batch, length, start + size}),
            {batch, length, head_count, config.head_dim}),
        {0, 2, 1, 3});
  };
  const float scale =
      static_cast<float>(std::pow(static_cast<double>(config.head_dim), -0.5));

  mx::array hidden = x;
  for (int layer = 0; layer < config.layer_count; ++layer) {
    const std::string prefix = layer_prefix(layer);
    const mx::array qkv = checkpoint_.Linear(
        checkpoint_.RmsNorm(hidden, prefix + ".attention_norm",
                            config.norm_eps),
        prefix + ".attention.wqkv");
    mx::array queries = split_heads(qkv, 0, query_size, config.head_count);
    mx::array keys = split_heads(qkv, query_size, key_value_size,
                                 config.key_value_head_count);
    const mx::array values =
        split_heads(qkv, query_size + key_value_size, key_value_size,
                    config.key_value_head_count);
    if (config.has_query_key_norm) {
      queries = checkpoint_.RmsNorm(queries, prefix + ".attention.q_norm",
                                    config.norm_eps);
      keys = checkpoint_.RmsNorm(keys, prefix + ".attention.k_norm",
                                 config.norm_eps);
    } else {
    }
    auto [cached_keys, cached_values] =
        caches[layer].UpdateAndFetch(rotate(keys), values);
    const mx::array attended = mx::fast::scaled_dot_product_attention(
        rotate(queries), cached_keys, cached_values, scale,
        length > 1 ? "causal" : "");
    hidden = mx::add(
        hidden,
        checkpoint_.Linear(mx::reshape(mx::transpose(attended, {0, 2, 1, 3}),
                                       {batch, length, -1}),
                           prefix + ".attention.wo"));
    const mx::array feed_forward_input =
        checkpoint_.RmsNorm(hidden, prefix + ".ffn_norm", config.norm_eps);
    hidden = mx::add(
        hidden,
        checkpoint_.Linear(
            mx::multiply(omni_common::Silu(checkpoint_.Linear(
                             feed_forward_input, prefix + ".feed_forward.w1")),
                         checkpoint_.Linear(feed_forward_input,
                                            prefix + ".feed_forward.w3")),
            prefix + ".feed_forward.w2"));
  }
  const int dimension = hidden.shape(2);
  return checkpoint_.RmsNorm(
      mx::reshape(mx::slice(hidden, {0, length - 1, 0}, {1, length, dimension}),
                  {1, dimension}),
      config.prefix + (config.prefix == kSlowPrefix ? ".norm" : "_norm"),
      config.norm_eps);
}

std::vector<mx::array>
FishS2ProModel::ResidualCodes(const mx::array &hidden,
                              const mx::array &semantic_code) const {
  std::vector<qwen3_asr::KVCache> caches(fast_.layer_count);
  // The slow hidden state fills the first position; its logits are unused.
  Forward(mx::expand_dims(hidden, 1), fast_, fast_rotary_, caches);
  std::vector<mx::array> codes;
  mx::array code = semantic_code;
  for (int codebook = 1; codebook < codebook_count_; ++codebook) {
    const mx::array fast_hidden = Forward(
        mx::expand_dims(checkpoint_.Embed(code, "model.fast_embeddings"), 1),
        fast_, fast_rotary_, caches);
    code = mx::astype(
        mx::argmax(checkpoint_.Linear(fast_hidden, "model.fast_output"), -1),
        mx::int32);
    codes.push_back(code);
  }
  return codes;
}

int FishS2ProModel::SampleSemanticToken(const float *candidate_logits,
                                        const std::vector<int> &history,
                                        int step,
                                        const SamplingOptions &sampling) const {
  const int candidate_count = codebook_size_ + 1;
  std::vector<float> logits(candidate_logits,
                            candidate_logits + candidate_count);
  // Each penalty is taken from the unpenalized logit, so a repeated token is
  // penalized once.
  for (const int token : history) {
    const float logit = candidate_logits[token - semantic_begin_id_];
    logits[token - semantic_begin_id_] =
        logit < 0 ? logit * sampling.repetition_penalty
                  : logit / sampling.repetition_penalty;
  }
  bool repeats = false;
  if (history.size() >= kRepeatWindow) {
    std::vector<int> recent(history.end() - kRepeatWindow, history.end());
    std::sort(recent.begin(), recent.end());
    repeats = std::adjacent_find(recent.begin(), recent.end()) != recent.end();
  } else {
  }
  const float temperature = repeats ? kRepeatTemperature : sampling.temperature;
  const float top_p = repeats ? kRepeatTopP : sampling.top_p;

  std::vector<int> order(candidate_count);
  for (int index = 0; index < candidate_count; ++index) {
    order[index] = index;
  }
  std::partial_sort(order.begin(), order.begin() + kSamplerCandidates,
                    order.end(), [&](int left, int right) {
                      return logits[left] > logits[right] ||
                             (logits[left] == logits[right] && left < right);
                    });
  const int kept = sampling.top_k > 0
                       ? std::min(sampling.top_k, kSamplerCandidates)
                       : kSamplerCandidates;
  const float masked = -std::numeric_limits<float>::infinity();
  std::vector<float> top(kSamplerCandidates);
  for (int rank = 0; rank < kSamplerCandidates; ++rank) {
    top[rank] = rank < kept ? logits[order[rank]] : masked;
  }
  std::vector<float> probabilities = top;
  Softmax(probabilities);
  float cumulative = 0.0f;
  for (int rank = 0; rank < kSamplerCandidates; ++rank) {
    cumulative += probabilities[rank];
    if (rank > 0 && cumulative > top_p) {
      top[rank] = masked;
    } else {
    }
  }
  const float divisor = std::max(temperature, kMinTemperature);
  for (float &value : top) {
    value /= divisor;
  }
  Softmax(top);
  // Note (Jiaxin Deng): SGLang's seeded draw (multinomial_with_seed): Gumbel
  // noise from a hash of seed, step and candidate, in float64.
  int choice = 0;
  double best = -std::numeric_limits<double>::infinity();
  for (int rank = 0; rank < kSamplerCandidates; ++rank) {
    const double uniform =
        static_cast<double>(MurmurHash(sampling.seed,
                                       static_cast<uint32_t>(step),
                                       static_cast<uint32_t>(rank))) /
        static_cast<double>(std::numeric_limits<uint32_t>::max());
    const double log_uniform =
        std::clamp(std::log(uniform), std::numeric_limits<double>::lowest(),
                   -std::pow(2.0, -32));
    const double score =
        -std::log(-log_uniform) + std::log(static_cast<double>(top[rank]));
    if (score > best) {
      best = score;
      choice = rank;
    } else {
    }
  }
  const int candidate = order[choice];
  return candidate == codebook_size_ ? im_end_id_
                                     : semantic_begin_id_ + candidate;
}

qwen3_asr::FinishReason FishS2ProModel::Generate(
    const SpeechPrompt &prompt, int max_frames, const SamplingOptions &sampling,
    const std::atomic<bool> &cancel,
    const std::function<void(const mx::array &)> &on_frame) const {
  const int length = static_cast<int>(prompt.ids.size());
  if (length + max_frames > max_positions_) {
    throw PromptTooLong("the prompt and its audio exceed the model's " +
                        std::to_string(max_positions_) + " positions");
  } else {
  }
  std::vector<int32_t> rows(static_cast<size_t>(codebook_count_) * length, 0);
  std::vector<mx::array> audio_codes = prompt.audio_codes;
  if (!audio_codes.empty()) {
    const mx::array codes = mx::concatenate(audio_codes, 1);
    mx::eval(codes);
    const int32_t *values = codes.data<int32_t>();
    const int frames = codes.shape(1);
    int frame = 0;
    for (int position = 0; position < length; ++position) {
      if (prompt.is_audio[position]) {
        for (int codebook = 0; codebook < codebook_count_; ++codebook) {
          rows[static_cast<size_t>(codebook) * length + position] =
              values[static_cast<size_t>(codebook) * frames + frame];
        }
        ++frame;
      } else {
      }
    }
  } else {
  }
  std::vector<uint8_t> audio_flags(prompt.is_audio.begin(),
                                   prompt.is_audio.end());
  std::vector<qwen3_asr::KVCache> caches(slow_.layer_count);
  mx::array hidden = Forward(
      Embed(mx::array(prompt.ids.begin(), {1, length}, mx::int32),
            mx::array(rows.begin(), {codebook_count_, length}, mx::int32),
            mx::astype(mx::array(audio_flags.begin(), {length}, mx::uint8),
                       mx::bool_)),
      slow_, slow_rotary_, caches);
  const auto candidates = [&](const mx::array &last_hidden) {
    const mx::array logits =
        checkpoint_.TiedProjection(last_hidden, "model.embeddings");
    return mx::reshape(
        mx::astype(mx::concatenate({mx::slice(logits, {0, semantic_begin_id_},
                                              {1, semantic_end_id_ + 1}),
                                    mx::slice(logits, {0, im_end_id_},
                                              {1, im_end_id_ + 1})},
                                   1),
                   mx::float32),
        {-1});
  };
  mx::array candidate_logits = candidates(hidden);
  std::vector<int> history;
  for (int frame = 0; frame < max_frames; ++frame) {
    if (cancel.load()) {
      throw qwen3_asr::TranscriptionCancelled();
    } else {
    }
    mx::eval(candidate_logits);
    const int token = SampleSemanticToken(candidate_logits.data<float>(),
                                          history, frame, sampling);
    if (token == im_end_id_) {
      return qwen3_asr::FinishReason::kStop;
    } else {
    }
    history.push_back(token);
    if (history.size() > kRepetitionHistory) {
      history.erase(history.begin());
    } else {
    }
    std::vector<mx::array> codes = {
        mx::array({token - semantic_begin_id_}, mx::int32)};
    for (const mx::array &code : ResidualCodes(hidden, codes.front())) {
      codes.push_back(code);
    }
    const mx::array frame_codes = mx::concatenate(codes, 0);
    if (frame + 1 < max_frames) {
      hidden = Forward(Embed(mx::array({token}, {1, 1}, mx::int32),
                             mx::reshape(frame_codes, {codebook_count_, 1}),
                             mx::array({true})),
                       slow_, slow_rotary_, caches);
      candidate_logits = candidates(hidden);
      // Note (Minyi Tang): the decoder steps on the GPU while on_frame
      // handles this frame.
      mx::async_eval({frame_codes, candidate_logits});
    } else {
    }
    mx::eval(frame_codes);
    on_frame(frame_codes);
  }
  return qwen3_asr::FinishReason::kLength;
}

} // namespace fish_s2_pro
