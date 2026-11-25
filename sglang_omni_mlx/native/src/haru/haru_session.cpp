// SPDX-License-Identifier: Apache-2.0
#include "haru/haru_session.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <utility>

namespace haru {

namespace mx = mlx::core;

namespace {

constexpr int kReferencePartFrames = 64;

struct DryConfig {
  float multiplier = 0.8f;
  float base = 1.75f;
  int allowed_length = 1;
  int lookback = 1024;
  int max_match_length = 32;
  float max_penalty = 32.0f;
};

std::vector<std::pair<int, float>> DryPenalties(const std::vector<int> &history,
                                                const DryConfig &config) {
  std::vector<std::pair<int, float>> out;
  const int n = static_cast<int>(history.size());
  if (n < std::max(1, config.allowed_length)) {
    return out;
  } else {
  }
  const int last = history[n - 1];
  std::vector<std::pair<int, int>> matches; // (candidate, length)
  for (int j = n - 2; j >= std::max(0, n - 1 - config.lookback); --j) {
    if (history[j] != last) {
      continue;
    } else {
    }
    int length = 1;
    while (length < config.max_match_length && j - length >= 0 &&
           history[j - length] == history[n - 1 - length]) {
      ++length;
    }
    if (length >= config.allowed_length) {
      matches.push_back({history[j + 1], length});
    } else {
    }
  }
  std::sort(matches.begin(), matches.end(), [](const auto &a, const auto &b) {
    return a.first != b.first ? a.first < b.first : a.second > b.second;
  });
  for (size_t i = 0; i < matches.size(); ++i) {
    if (i > 0 && matches[i].first == matches[i - 1].first) {
      continue;
    } else {
    }
    const float penalty = std::min(
        config.max_penalty,
        static_cast<float>(
            config.multiplier *
            std::pow(config.base, matches[i].second - config.allowed_length)));
    out.push_back({matches[i].first, penalty});
  }
  return out;
}

} // namespace

DuplexSession::DuplexSession(const HaruModel &model,
                             const omni_common::Tokenizer &tokenizer,
                             const DuplexConfig &config)
    : model_(model), codec_(model.codec()), head_(model.head()),
      tokenizer_(tokenizer), config_(config),
      pad_id_(model.config().pad_token_id),
      eop_id_(model.config().eop_token_id),
      eos_id_(model.config().eos_token_id),
      silent_ids_(
          mx::array({model.config().pad_token_id, model.config().eop_token_id},
                    mx::int32)),
      token_(mx::array(0, mx::int32)),
      latent_(mx::zeros(mx::Shape{1, model.config().latent_dim}, mx::float32)),
      key_(mx::random::key(config.seed >= 0 ? static_cast<uint64_t>(config.seed)
                                            : std::random_device{}())) {}

mx::array DuplexSession::NextKey() {
  auto [key1, key2] = mx::random::split(key_);
  key_ = key1; // keep the first output, use the second
  return key2; // use the second
}

Frame DuplexSession::Start(const std::string &system_prompt,
                           const std::vector<float> &reference) {
  const HaruConfig &config = model_.config();
  const int dim = config.latent_dim;

  std::vector<int> card;
  if (!system_prompt.empty()) {
    card = tokenizer_.Encode("<|im_start|>system\n" + system_prompt +
                             "<|im_end|>\n");
  } else {
  }

  mx::array ref_latents =
      mx::zeros(mx::Shape{1, 1, dim}, mx::float32); // [1, n_ref, dim]
  int n_ref = 0;
  if (!reference.empty()) {
    // Encoded in parts with the codec's state carried over, so a long
    // reference does not hold every layer's output at once.
    VibeVoiceCodec::State state = codec_.InitEncoderState();
    const size_t part_samples =
        static_cast<size_t>(kReferencePartFrames) * config.frame_samples;
    std::vector<mx::array> parts;
    for (size_t begin = 0; begin < reference.size(); begin += part_samples) {
      const size_t count = std::min(part_samples, reference.size() - begin);
      parts.push_back(
          codec_.Encode(mx::array(reference.data() + begin,
                                  {static_cast<int>(count)}, mx::float32),
                        state));
      mx::eval(parts.back());
    }
    const mx::array means = mx::concatenate(parts, 0);
    ref_latents =
        codec_.Sample(mx::reshape(means, {1, means.shape(0), dim}), NextKey());
    mx::eval(ref_latents);
    n_ref = ref_latents.shape(1);
  } else {
  }

  std::vector<int> grid;
  if (config_.speak_first) {
    grid = {pad_id_, eop_id_};
  } else {
  }
  if (card.empty() && n_ref == 0 && grid.empty()) {
    grid = {pad_id_};
  } else {
  }
  std::vector<int> tokens = card;
  for (int i = 0; i < n_ref; ++i) {
    tokens.push_back(i == n_ref - 1 ? eop_id_ : pad_id_);
  }
  for (int t : grid) {
    tokens.push_back(t);
  }
  const int n = static_cast<int>(tokens.size());
  const int n_card = static_cast<int>(card.size());
  const int n_grid = static_cast<int>(grid.size());

  const mx::array &silence = model_.silence_latent();
  mx::array room_tone = silence;
  if (n_ref > 0) {
    const int frame = config.frame_samples;
    const int whole = static_cast<int>(reference.size()) / frame;
    if (whole > 0) {
      int best = 0;
      double best_energy = 1e30;
      for (int f = 0; f < whole; ++f) {
        double energy = 0.0;
        for (int s = 0; s < frame; ++s) {
          energy += reference[f * frame + s] * reference[f * frame + s];
        }
        energy /= frame;
        if (energy < best_energy) {
          best_energy = energy;
          best = f;
        } else {
        }
      }
      room_tone = mx::reshape(mx::slice(ref_latents, mx::Shape{0, best, 0},
                                        mx::Shape{1, best + 1, dim}),
                              mx::Shape{dim});
      mx::eval(room_tone);
    } else {
    }
  } else {
  }

  std::vector<float> model_flat(static_cast<size_t>(n) * dim, 0.0f);
  std::vector<float> user_flat(static_cast<size_t>(n) * dim, 0.0f);
  if (n_ref > 0) {
    const float *src = ref_latents.data<float>();
    for (int i = 0; i < n_ref * dim; ++i) {
      model_flat[static_cast<size_t>(n_card + i / dim) * dim + i % dim] =
          src[i];
    }
  } else {
  }
  if (n_grid > 0) {
    const float *room_tone_values = room_tone.data<float>();
    const float *silence_values = silence.data<float>();
    for (int g = 0; g < n_grid; ++g) {
      const int pos = n - n_grid + g;
      for (int d = 0; d < dim; ++d) {
        model_flat[static_cast<size_t>(pos) * dim + d] = room_tone_values[d];
        user_flat[static_cast<size_t>(pos) * dim + d] = silence_values[d];
      }
    }
  } else {
  }

  std::vector<float> model_mask(static_cast<size_t>(n), 0.0f);
  std::vector<float> user_mask(static_cast<size_t>(n), 0.0f);
  for (int p = n_card; p < n; ++p) {
    model_mask[p] = 1.0f;
  }
  for (int p = n - n_grid; p < n; ++p) {
    user_mask[p] = 1.0f;
  }

  caches_ = model_.NewCaches();
  encoder_state_ = codec_.InitEncoderState();
  decoder_state_ = codec_.InitDecoderState();
  index_ = 0;
  force_text_ = config_.speak_first ? 1 : 0;
  history_.clear();
  pending_ids_.clear();
  in_gap_ = false;

  const mx::array embeddings = model_.Embed(
      mx::reshape(mx::array(tokens.data(), {n}, mx::int32), {1, n}),
      mx::reshape(mx::array(model_flat.data(), {n * dim}, mx::float32),
                  {1, n, dim}),
      mx::reshape(mx::array(user_flat.data(), {n * dim}, mx::float32),
                  {1, n, dim}),
      mx::reshape(mx::array(model_mask.data(), {n}, mx::float32), {1, n}),
      mx::reshape(mx::array(user_mask.data(), {n}, mx::float32), {1, n}));
  return Emit(model_.Backbone(embeddings, caches_));
}

Frame DuplexSession::Step(const std::vector<float> &user_frame) {
  const HaruConfig &config = model_.config();
  const mx::array user_audio = mx::array(
      user_frame.data(), {static_cast<int>(user_frame.size())}, mx::float32);
  const mx::array user_mean = codec_.Encode(user_audio, encoder_state_);
  const mx::array user_latent = codec_.Sample(
      mx::reshape(user_mean, {1, 1, config.latent_dim}), NextKey());

  const mx::array embeddings =
      model_.Embed(mx::reshape(token_, {1, 1}),
                   mx::reshape(latent_, {1, 1, config.latent_dim}), user_latent,
                   mx::full(mx::Shape{1, 1}, 1.0f, mx::float32),
                   mx::full(mx::Shape{1, 1}, 1.0f, mx::float32));
  mx::array hidden = model_.Backbone(embeddings, caches_);
  mx::eval(hidden);
  return Emit(hidden);
}

mx::array DuplexSession::PreparedLogits(const mx::array &last_hidden) {
  mx::array logits =
      mx::reshape(model_.Logits(last_hidden), {model_.config().lm.vocab_size});
  std::vector<int32_t> repeated_tokens;
  std::vector<float> penalties;
  for (const auto &[token, penalty] : DryPenalties(history_, DryConfig{})) {
    repeated_tokens.push_back(token);
    penalties.push_back(penalty);
  }
  if (!repeated_tokens.empty()) {
    const int count = static_cast<int>(repeated_tokens.size());
    const mx::array indices(repeated_tokens.data(), {count}, mx::int32);
    logits = mx::put_along_axis(
        logits, indices,
        mx::subtract(mx::take(logits, indices, 0),
                     mx::array(penalties.data(), {count}, mx::float32)),
        0);
  } else {
  }
  // The PAD and EOP logits, recomputed in float32.
  logits = mx::put_along_axis(
      logits, silent_ids_,
      mx::reshape(model_.LogitsFp32(last_hidden, silent_ids_), {2}), 0);
  return force_text_ > 0 ? mx::add(logits, model_.speech_token_mask()) : logits;
}

mx::array DuplexSession::SampleToken(const mx::array &logits) {
  if (config_.temperature <= 0.0f) {
    return mx::argmax(logits, -1);
  } else {
  }
  mx::array scaled_logits =
      mx::divide(logits, mx::array(config_.temperature, mx::float32));
  const int vocab = scaled_logits.shape(0);
  std::optional<mx::array> indices;
  if (0 < config_.top_k && config_.top_k < vocab) {
    indices =
        mx::argpartition(mx::negative(scaled_logits), config_.top_k - 1, -1);
    indices = mx::take(*indices, mx::arange(0, config_.top_k, mx::int32), 0);
    scaled_logits = mx::take_along_axis(scaled_logits, *indices, -1);
  } else {
  }
  if (0.0f < config_.top_p && config_.top_p < 1.0f) {
    const mx::array order = mx::argsort(mx::negative(scaled_logits), -1);
    scaled_logits = mx::take_along_axis(scaled_logits, order, -1);
    if (indices.has_value()) {
      indices = mx::take_along_axis(*indices, order, -1);
    } else {
      // Without a top-k cut the sampled position indexes the sorted order.
      indices = order;
    }
    const mx::array probs = mx::softmax(scaled_logits, -1);
    const mx::array keep = mx::less(mx::subtract(mx::cumsum(probs, -1), probs),
                                    mx::array(config_.top_p, mx::float32));
    scaled_logits = mx::where(keep, scaled_logits,
                              mx::full(scaled_logits.shape(),
                                       -std::numeric_limits<float>::infinity(),
                                       mx::float32));
  } else {
  }
  const mx::array choice =
      mx::random::categorical(scaled_logits, -1, NextKey());
  if (indices.has_value()) {
    return mx::squeeze(
        mx::take_along_axis(*indices, mx::expand_dims(choice, -1), -1), -1);
  } else {
  }
  return choice;
}

std::string DuplexSession::Detokenize(int token) {
  if (token == pad_id_ || token == eop_id_) {
    if (in_gap_) {
      return "";
    } else {
    }
    in_gap_ = true;
    return " ";
  } else {
  }
  in_gap_ = false;
  pending_ids_.push_back(token);
  std::string text = tokenizer_.Decode(pending_ids_, true);
  const std::string replacement = "\xEF\xBF\xBD";
  if (text.size() >= replacement.size() &&
      text.compare(text.size() - replacement.size(), replacement.size(),
                   replacement) == 0) {
    return "";
  } else {
  }
  pending_ids_.clear();
  return text;
}

Frame DuplexSession::Emit(const mx::array &hidden) {
  const HaruConfig &config = model_.config();
  const mx::array last_hidden = mx::take(hidden, hidden.shape(1) - 1, 1);
  const mx::array token = SampleToken(PreparedLogits(last_hidden));
  const mx::array noise =
      mx::random::normal({1, config.latent_dim}, mx::float32, NextKey());
  const mx::array latent = model_.UnscaleLatents(
      head_.Sample(last_hidden, noise, config_.guidance_scale));
  const mx::array audio = codec_.Decode(
      mx::reshape(latent, {1, config.latent_dim}), decoder_state_);
  mx::eval({token, latent, audio});

  token_ = token;
  latent_ = latent;
  const int token_id = token.item<int>();
  if (force_text_ > 0) {
    --force_text_;
  } else {
  }
  if (token_id != pad_id_ && token_id != eop_id_ && token_id != eos_id_) {
    history_.push_back(token_id);
  } else {
  }
  Frame frame;
  const float *audio_data = audio.data<float>();
  frame.audio.assign(audio_data, audio_data + audio.size());
  frame.token = token_id;
  frame.text = Detokenize(token_id);
  frame.index = index_;
  ++index_;
  frame.done = index_ >= config_.max_frames;
  return frame;
}

} // namespace haru
