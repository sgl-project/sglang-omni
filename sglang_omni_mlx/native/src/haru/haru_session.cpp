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
  }
  const int last = history[n - 1];
  std::vector<std::pair<int, int>> matches; // (candidate, length)
  for (int j = n - 2; j >= std::max(0, n - 1 - config.lookback); --j) {
    if (history[j] != last) {
      continue;
    }
    int length = 1;
    while (length < config.max_match_length && j - length >= 0 &&
           history[j - length] == history[n - 1 - length]) {
      ++length;
    }
    if (length >= config.allowed_length) {
      matches.push_back({history[j + 1], length});
    }
  }
  std::sort(matches.begin(), matches.end(), [](const auto &a, const auto &b) {
    return a.first != b.first ? a.first < b.first : a.second > b.second;
  });
  for (size_t i = 0; i < matches.size(); ++i) {
    if (i > 0 && matches[i].first == matches[i - 1].first) {
      continue;
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

// The latent of digital silence: the last latent of 16 frames of zeros.
mx::array SilenceLatent(VibeVoiceCodec &codec, int frame_samples,
                        int latent_dim) {
  auto state = codec.InitEncoderState();
  const mx::array silence =
      mx::zeros(mx::Shape{16 * frame_samples}, mx::float32);
  const mx::array means = codec.Encode(silence, state);
  mx::eval(means);
  mx::array result =
      mx::reshape(mx::slice(means, mx::Shape{15, 0}, mx::Shape{16, latent_dim}),
                  mx::Shape{latent_dim});
  mx::eval(result);
  return result;
}

} // namespace

mx::array ForceTextMask(const HaruConfig &cfg) {
  const mx::array vocab = mx::arange(0, cfg.lm.vocab_size, mx::int32);
  const mx::array blocked =
      mx::full(mx::Shape{cfg.lm.vocab_size},
               mx::array(-std::numeric_limits<float>::infinity(), mx::float32));
  mx::array mask =
      mx::full(mx::Shape{cfg.lm.vocab_size}, mx::array(0.0f, mx::float32));
  for (int banned : {cfg.pad_token_id, cfg.eop_token_id, cfg.eos_token_id}) {
    mask = mx::where(mx::equal(vocab, mx::array(banned, mx::int32)), blocked,
                     mask);
  }
  mx::eval(mask);
  return mask;
}

DuplexSession::DuplexSession(HaruModel &model,
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
      force_text_mask_(ForceTextMask(model.config())),
      head_dtype_(model.checkpoint().Weight("head.cond_proj.weight").dtype()),
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
  // MLX prints generated Metal source on stdout while compiling a custom
  // kernel; keep a supervised server's JSON stream and the tools clean.
  const auto &cfg = model_.config();
  const int dim = cfg.latent_dim;

  std::vector<int> card;
  if (!system_prompt.empty()) {
    card = tokenizer_.Encode("<|im_start|>system\n" + system_prompt +
                             "<|im_end|>\n");
  }

  mx::array ref_latents =
      mx::zeros(mx::Shape{1, 1, dim}, mx::float32); // [1, n_ref, dim]
  int n_ref = 0;
  if (!reference.empty()) {
    auto state = codec_.InitEncoderState();
    const mx::array audio = mx::array(
        reference.data(), {static_cast<int>(reference.size())}, mx::float32);
    const mx::array means = codec_.Encode(audio, state);
    mx::eval(means);
    ref_latents =
        codec_.Sample(mx::reshape(means, {1, means.shape(0), dim}), NextKey());
    mx::eval(ref_latents);
    n_ref = ref_latents.shape(1);
  }

  std::vector<int> grid;
  if (config_.speak_first) {
    grid = {pad_id_, eop_id_};
  }
  if (card.empty() && n_ref == 0 && grid.empty()) {
    grid = {pad_id_};
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

  mx::array room_tone =
      SilenceLatent(codec_, cfg.frame_samples, cfg.latent_dim);
  if (n_ref > 0) {
    const int frame = cfg.frame_samples;
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
        }
      }
      room_tone = mx::reshape(mx::slice(ref_latents, mx::Shape{0, best, 0},
                                        mx::Shape{1, best + 1, dim}),
                              mx::Shape{dim});
      mx::eval(room_tone);
    }
  }
  const mx::array silence =
      SilenceLatent(codec_, cfg.frame_samples, cfg.latent_dim);

  std::vector<float> model_flat(static_cast<size_t>(n) * dim, 0.0f);
  std::vector<float> user_flat(static_cast<size_t>(n) * dim, 0.0f);
  if (n_ref > 0) {
    const float *src = ref_latents.data<float>();
    for (int i = 0; i < n_ref * dim; ++i) {
      model_flat[static_cast<size_t>(n_card + i / dim) * dim + i % dim] =
          src[i];
    }
  }
  if (n_grid > 0) {
    const float *rt = room_tone.data<float>();
    const float *sl = silence.data<float>();
    for (int g = 0; g < n_grid; ++g) {
      const int pos = n - n_grid + g;
      for (int d = 0; d < dim; ++d) {
        model_flat[static_cast<size_t>(pos) * dim + d] = rt[d];
        user_flat[static_cast<size_t>(pos) * dim + d] = sl[d];
      }
    }
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
  mx::array hidden = model_.Backbone(embeddings, caches_);
  mx::eval(hidden);
  return Emit(hidden);
}

Frame DuplexSession::Step(const std::vector<float> &user_frame) {
  // The codec compiles a kernel per frame shape on first use; see Start().
  const auto &cfg = model_.config();
  const mx::array user_audio = mx::array(
      user_frame.data(), {static_cast<int>(user_frame.size())}, mx::float32);
  const mx::array user_mean = codec_.Encode(user_audio, encoder_state_);
  mx::eval(user_mean);
  const mx::array user_latent =
      codec_.Sample(mx::reshape(user_mean, {1, 1, cfg.latent_dim}), NextKey());

  const mx::array embeddings = model_.Embed(
      mx::reshape(token_, {1, 1}), mx::reshape(latent_, {1, 1, cfg.latent_dim}),
      user_latent, mx::full(mx::Shape{1, 1}, 1.0f, mx::float32),
      mx::full(mx::Shape{1, 1}, 1.0f, mx::float32));
  mx::array hidden = model_.Backbone(embeddings, caches_);
  mx::eval(hidden);
  return Emit(hidden);
}

mx::array DuplexSession::PreparedLogits(const mx::array &hlast) {
  const auto &cfg = model_.config();
  mx::array logits = mx::reshape(model_.Logits(hlast), {cfg.lm.vocab_size});
  const mx::array vocab = mx::arange(0, cfg.lm.vocab_size, mx::int32);

  if (config_.dry) {
    const auto penalties = DryPenalties(history_, DryConfig{});
    if (!penalties.empty()) {
      // One scatter instead of a vocab-wide selection per penalized token.
      std::vector<int32_t> tokens;
      std::vector<float> amounts;
      tokens.reserve(penalties.size());
      amounts.reserve(penalties.size());
      for (const auto &[token, penalty] : penalties) {
        tokens.push_back(token);
        amounts.push_back(-penalty);
      }
      const mx::array index = mx::array(
          tokens.data(), mx::Shape{static_cast<int>(tokens.size())}, mx::int32);
      const mx::array delta =
          mx::array(amounts.data(), mx::Shape{static_cast<int>(amounts.size())},
                    mx::float32);
      logits = mx::put_along_axis(
          logits, index, mx::add(mx::take(logits, index, 0), delta), 0);
    }
  }
  // pad and eop get their values recomputed in float32, in place.
  const mx::array fp32 =
      mx::reshape(model_.LogitsFp32(hlast, silent_ids_), {2});
  const mx::array specials =
      mx::array({static_cast<int32_t>(pad_id_), static_cast<int32_t>(eop_id_)},
                mx::Shape{2}, mx::int32);
  logits = mx::put_along_axis(logits, specials, fp32, 0);
  if (force_text_ > 0) {
    // One masked add instead of three vocab-wide selections.
    logits = mx::add(logits, force_text_mask_);
  }
  return logits;
}

mx::array DuplexSession::SampleToken(const mx::array &logits) {
  if (config_.temperature <= 0.0f) {
    return mx::argmax(logits, -1);
  }
  mx::array l = mx::divide(logits, mx::array(config_.temperature, mx::float32));
  const int vocab = l.shape(0);
  std::optional<mx::array> indices;
  if (0 < config_.top_k && config_.top_k < vocab) {
    indices = mx::argpartition(mx::negative(l), config_.top_k - 1, -1);
    indices = mx::take(*indices, mx::arange(0, config_.top_k, mx::int32), 0);
    l = mx::take_along_axis(l, *indices, -1);
  }
  if (0.0f < config_.top_p && config_.top_p < 1.0f) {
    const mx::array order = mx::argsort(mx::negative(l), -1);
    l = mx::take_along_axis(l, order, -1);
    if (indices.has_value()) {
      indices = mx::take_along_axis(*indices, order, -1);
    } else {
      // Without a top-k cut the sampled position indexes the sorted order.
      indices = order;
    }
    const mx::array probs = mx::softmax(l, -1);
    const mx::array keep = mx::less(mx::subtract(mx::cumsum(probs, -1), probs),
                                    mx::array(config_.top_p, mx::float32));
    l = mx::where(keep, l,
                  mx::full(l.shape(), -std::numeric_limits<float>::infinity(),
                           mx::float32));
  }
  const mx::array choice = mx::random::categorical(l, -1, NextKey());
  if (indices.has_value()) {
    return mx::squeeze(
        mx::take_along_axis(*indices, mx::expand_dims(choice, -1), -1), -1);
  }
  return choice;
}

std::string DuplexSession::Detokenize(int token) {
  if (token == pad_id_ || token == eop_id_) {
    if (in_gap_) {
      return "";
    }
    in_gap_ = true;
    return " ";
  }
  in_gap_ = false;
  pending_ids_.push_back(token);
  std::string text = tokenizer_.Decode(pending_ids_, true);
  const std::string replacement = "\xEF\xBF\xBD";
  if (text.size() >= replacement.size() &&
      text.compare(text.size() - replacement.size(), replacement.size(),
                   replacement) == 0) {
    return "";
  }
  pending_ids_.clear();
  return text;
}

Frame DuplexSession::Emit(const mx::array &hidden) {
  const auto &cfg = model_.config();
  const int time = hidden.shape(1);
  const mx::array hlast =
      mx::reshape(mx::slice(hidden, mx::Shape{0, time - 1, 0},
                            mx::Shape{1, time, cfg.hidden_size()}),
                  mx::Shape{1, cfg.hidden_size()});

  const mx::array logits = PreparedLogits(hlast);
  const mx::array token = SampleToken(logits);

  // The head's weights are half precision; a float32 noise would promote every
  // matmul of the velocity network to float32 and cost an order of magnitude.
  const mx::array noise = mx::astype(
      mx::random::normal(mx::Shape{1, cfg.latent_dim}, mx::float32, NextKey()),
      head_dtype_);
  const mx::array latent =
      model_.UnscaleLatents(head_.Sample(hlast, noise, config_.guidance_scale));
  const mx::array audio =
      codec_.Decode(mx::reshape(latent, {1, cfg.latent_dim}), decoder_state_);
  // One synchronisation per frame: the token feeds the next step, the audio is
  // what the caller sends, and the decode overlaps the eval of the token.
  mx::eval({token, latent, audio});

  token_ = token;
  latent_ = latent;
  const int token_id = token.item<int>();
  if (force_text_ > 0) {
    --force_text_;
  }
  if (config_.dry && token_id != pad_id_ && token_id != eop_id_ &&
      token_id != eos_id_) {
    history_.push_back(token_id);
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
