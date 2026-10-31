// SPDX-License-Identifier: Apache-2.0
#include "qwen3_omni/generation.h"

#include "mlx/random.h"
#include <chrono>
#include <iostream>
#include <limits>

namespace qwen3_omni {
namespace {
mx::array last_hidden(const mx::array &hidden_states) {
  return mx::squeeze(rows(hidden_states, hidden_states.shape(1) - 1,
                          hidden_states.shape(1), 1),
                     1);
}
std::vector<mx::array> cache_arrays(const std::vector<KVCache> &cache,
                                    std::vector<mx::array> tensors) {
  for (const auto &state : cache) {
    if (state.keys) {
      tensors.push_back(*state.keys);
      tensors.push_back(*state.values);
    } else {
    }
  }
  return tensors;
}
struct SynchronizeOnExit {
  ~SynchronizeOnExit() { mx::synchronize(); }
};
} // namespace
void OmniModel::generate(const ChatRequest &request, const ChunkEmitter &emit,
                         const Cancellation &cancelled) {
  struct Cleanup {
    ~Cleanup() {
      mx::synchronize();
      mx::clear_cache();
    }
  } cleanup;
  if (request.seed) {
    mx::random::seed(*request.seed);
  } else {
  }
  auto prepared = prepare(request, tokenizer, cancelled);
  auto input_tokens =
      mx::array(prepared.tokens.data(), {1, int(prepared.tokens.size())});
  auto embeddings = weights.embedding(
      "thinker.language_model.model.embed_tokens", input_tokens);
  int captures = configuration.at("thinker_config")
                     .at("vision_config")
                     .at("deepstack_visual_indexes")
                     .size();
  std::vector<mx::array> deepstack(
      captures, mx::zeros(embeddings.shape(), embeddings.dtype()));
  std::map<int, int> offsets;
  bool has_visual = false;
  auto replace_features = [&](mx::array &target, const mx::array &features,
                              int token, int offset) {
    std::vector<int> positions;
    for (int i = 0; i < int(prepared.tokens.size()); ++i) {
      if (prepared.tokens[i] == token) {
        positions.push_back(i);
      } else {
      }
    }
    int count = features.shape(0);
    if (offset + count > int(positions.size())) {
      throw std::invalid_argument("media feature/token count mismatch");
    } else {
    }
    auto indices = mx::array(positions.data() + offset, {count});
    auto updates = mx::reshape(mx::astype(features, target.dtype()),
                               {count, 1, 1, target.shape(-1)});
    target = mx::scatter(target, indices, updates, 1);
  };
  for (const auto &entry : prepared.entries) {
    check_cancelled(cancelled);
    if (entry.pixels) {
      has_visual = true;
      auto [features, layers] =
          vision.forward(*entry.pixels, *entry.grid, cancelled);
      int offset = offsets[entry.visual_token];
      replace_features(embeddings, features, entry.visual_token, offset);
      for (int i = 0; i < int(layers.size()); ++i) {
        replace_features(deepstack[i], layers[i], entry.visual_token, offset);
      }
      offsets[entry.visual_token] += features.shape(0);
    } else {
    }
    if (entry.mel) {
      auto features = audio.forward(*entry.mel, cancelled);
      replace_features(embeddings, features, 151675, offsets[151675]);
      offsets[151675] += features.shape(0);
    } else {
    }
  }
  if (!has_visual) {
    deepstack.clear();
  } else {
  }
  auto cache = thinker.make_cache();
  auto [hidden_states, prompt_hidden] = thinker.forward(
      embeddings, prepared.positions, cache, cancelled, deepstack,
      request.speech ? configuration.at("talker_config")
                           .at("accept_hidden_layer")
                           .get<int>()
                     : -1);
  std::string text;
  auto generation = generate_text_tokens(
      weights.linear("thinker.language_model.lm_head",
                     last_hidden(hidden_states)),
      request.max_tokens, {configuration.at("im_end_token_id").get<int>()},
      request.sampling, cancelled,
      [&](const mx::array &token) {
        float position = cache.front().offset + prepared.delta;
        auto input =
            weights.embedding("thinker.language_model.model.embed_tokens",
                              mx::reshape(token, {1, 1}));
        auto next =
            thinker
                .forward(input, mx::full({3, 1, 1}, position), cache, cancelled)
                .first;
        return cache_arrays(cache,
                            {weights.linear("thinker.language_model.lm_head",
                                            last_hidden(next))});
      },
      [&](const std::vector<int> &tokens) {
        auto decoded = tokenizer.Decode(tokens, true);
        if (!decoded.ends_with("\xef\xbf\xbd")) {
          emit({decoded.substr(text.size()), {}, "", 0, 0});
          text = decoded;
        } else {
        }
      });
  cache.clear();
  if (request.speech) {
    if (generation.tokens.empty() || !prompt_hidden) {
      throw std::invalid_argument("thinker produced no speech conditioning");
    } else {
      generate_speech(
          request, prepared,
          mx::squeeze(
              weights.embedding("thinker.language_model.model.embed_tokens",
                                input_tokens),
              0),
          mx::squeeze(*prompt_hidden, 0), generation.tokens, emit, cancelled);
    }
  } else {
  }
  emit({"",
        {},
        generation.stopped ? "stop" : "length",
        int(prepared.tokens.size()),
        generation.completion_tokens});
}
mx::array OmniModel::projection(const std::string &name,
                                const mx::array &input) const {
  return weights.linear("talker." + name + ".linear_fc2",
                        omni_common::Silu(weights.linear(
                            "talker." + name + ".linear_fc1", input)));
}
void OmniModel::generate_speech(const ChatRequest &request,
                                const PreparedInput &prepared,
                                const mx::array &prompt_embeddings,
                                const mx::array &prompt_hidden,
                                const std::vector<int> &tokens,
                                const ChunkEmitter &emit,
                                const Cancellation &cancelled) {
  SynchronizeOnExit synchronize;
  const auto &config = configuration.at("talker_config");
  std::vector<int> all_tokens = prepared.tokens;
  all_tokens.insert(all_tokens.end(), tokens.begin(), tokens.end());
  auto generated = mx::astype(
      weights.embedding("thinker.language_model.model.embed_tokens",
                        mx::array(tokens.data(), {int(tokens.size())})),
      mx::float32);
  auto embeddings = mx::concatenate(
      {mx::astype(prompt_embeddings, mx::float32), generated}, 0);
  auto hidden_states =
      mx::concatenate({mx::astype(prompt_hidden, mx::float32), generated}, 0);
  auto special_tokens =
      mx::array({configuration.at("tts_bos_token_id").get<int>(),
                 configuration.at("tts_eos_token_id").get<int>(),
                 configuration.at("tts_pad_token_id").get<int>()});
  auto special = projection(
      "text_projection",
      mx::astype(weights.embedding("thinker.language_model.model.embed_tokens",
                                   special_tokens),
                 mx::float32));
  std::vector<int> starts;
  for (int i = 0; i < int(all_tokens.size()); ++i) {
    if (all_tokens[i] == configuration.at("im_start_token_id")) {
      starts.push_back(i);
    } else {
    }
  }
  std::vector<mx::array> speech_rows;
  std::vector<int> talker_tokens;
  std::optional<mx::array> trailing;
  for (int segment = 0; segment < int(starts.size()); ++segment) {
    int start = starts[segment];
    int end = segment + 1 < int(starts.size()) ? starts[segment + 1]
                                               : all_tokens.size();
    int role = all_tokens.at(start + 1);
    if (role == configuration.at("user_token_id")) {
      std::vector<int> mask(end - start);
      for (int i = start; i < end; ++i) {
        mask[i - start] = all_tokens[i] == 151655 || all_tokens[i] == 151656 ||
                          all_tokens[i] == 151675;
      }
      speech_rows.push_back(mx::where(
          mx::array(mask.data(), {end - start, 1}),
          projection("hidden_projection", rows(hidden_states, start, end)),
          projection("text_projection", rows(embeddings, start, end))));
      talker_tokens.insert(talker_tokens.end(), all_tokens.begin() + start,
                           all_tokens.begin() + end);
    } else if (role == configuration.at("assistant_token_id") &&
               segment == int(starts.size()) - 1) {
      auto stop =
          std::find(all_tokens.begin() + start, all_tokens.begin() + end,
                    configuration.at("im_end_token_id").get<int>());
      end = int(stop - all_tokens.begin());
      auto projected =
          projection("text_projection", rows(embeddings, start, end));
      auto text_rows = mx::concatenate(
          {rows(projected, 0, 3),
           mx::broadcast_to(rows(special, 2, 3), {4, special.shape(-1)}),
           rows(special, 0, 1),
           projected.shape(0) > 3 ? rows(projected, 3, 4)
                                  : mx::zeros({1, special.shape(-1)})},
          0);
      auto codec_tokens =
          mx::array({config.at("codec_nothink_id").get<int>(),
                     config.at("codec_think_bos_id").get<int>(),
                     config.at("codec_think_eos_id").get<int>(),
                     config.at("speaker_id").at(request.voice).get<int>(),
                     config.at("codec_pad_id").get<int>(),
                     config.at("codec_bos_id").get<int>()});
      auto codec_rows = mx::concatenate(
          {mx::zeros({3, special.shape(-1)}, text_rows.dtype()),
           mx::astype(
               weights.embedding("talker.model.codec_embedding", codec_tokens),
               mx::float32)},
          0);
      speech_rows.push_back(text_rows + codec_rows);
      talker_tokens.insert(talker_tokens.end(), text_rows.shape(0),
                           configuration.at("tts_pad_token_id").get<int>());
      trailing = mx::concatenate(
          {rows(projected, std::min(4, projected.shape(0)), projected.shape(0)),
           rows(special, 1, 2)},
          0);
    } else {
    }
  }
  if (!trailing) {
    throw std::invalid_argument("missing assistant speech prefix");
  } else {
  }
  std::vector<MediaEntry> user_entries;
  for (const auto &entry : prepared.entries) {
    if (entry.user_owned) {
      user_entries.push_back(entry);
    } else {
    }
  }
  auto [positions, delta] = position_ids(talker_tokens, user_entries);
  auto speech_embeddings = mx::expand_dims(mx::concatenate(speech_rows, 0), 0);
  auto cache = talker.make_cache();
  if (request.seed) {
    mx::random::seed(*request.seed);
  } else {
  }
  auto speech_hidden =
      talker.forward(speech_embeddings, positions, cache, cancelled).first;
  auto codec_logits = [&](const mx::array &hidden_states) {
    auto logits =
        weights.linear("talker.codec_head", last_hidden(hidden_states));
    int eos = config.at("codec_eos_token_id");
    auto vocabulary = mx::arange(logits.shape(-1), mx::int32);
    auto allowed =
        (vocabulary <
         configuration.at("code2wav_config").at("codebook_size").get<int>()) |
        (vocabulary == eos);
    return mx::where(
        allowed, logits,
        mx::array(-std::numeric_limits<float>::infinity(), logits.dtype()));
  };
  auto speech_logits = codec_logits(speech_hidden);
  std::vector<int> history;
  std::vector<mx::array> codes;
  int context_frames = 0, new_frames = 0;
  for (int i = 0; i < request.max_speech_frames; ++i) {
    check_cancelled(cancelled);
    auto device_token = sample(speech_logits, history, request.talker_sampling);
    int token = mx::astype(device_token, mx::int32).item<int>();
    check_cancelled(cancelled);
    if (token == config.at("codec_eos_token_id")) {
      break;
    } else {
      history.push_back(token);
    }
    auto first_embedding = weights.embedding("talker.model.codec_embedding",
                                             mx::reshape(device_token, {1, 1}));
    auto predictor_input =
        mx::concatenate({rows(speech_hidden, speech_hidden.shape(1) - 1,
                              speech_hidden.shape(1), 1),
                         first_embedding},
                        1);
    std::vector<mx::array> feedback_embeddings{first_embedding},
        frame_codes{device_token};
    auto predictor_cache = predictor.make_cache();
    auto residual_options = request.talker_sampling;
    residual_options.temperature =
        residual_options.top_k == 1 ? 0 : residual_options.temperature;
    residual_options.top_k = -1;
    residual_options.repetition_penalty = 1;
    std::optional<mx::array> previous_residual;
    for (int group = 0; group < config.at("num_code_groups").get<int>() - 1;
         ++group) {
      check_cancelled(cancelled);
      int offset = predictor_cache.front().offset;
      auto residual_hidden =
          predictor
              .forward(predictor_input,
                       mx::arange(offset, offset + predictor_input.shape(1),
                                  mx::int32),
                       predictor_cache, cancelled)
              .first;
      auto residual_logits = weights.linear("talker.code_predictor.lm_head." +
                                                std::to_string(group),
                                            last_hidden(residual_hidden));
      auto residual_token = sample(residual_logits, {}, residual_options);
      mx::async_eval(cache_arrays(predictor_cache, {residual_token}));
      if (previous_residual) {
        mx::eval(*previous_residual);
      } else {
      }
      previous_residual = residual_token;
      frame_codes.push_back(residual_token);
      predictor_input =
          weights.embedding("talker.code_predictor.model.codec_embedding." +
                                std::to_string(group),
                            mx::reshape(residual_token, {1, 1}));
      feedback_embeddings.push_back(predictor_input);
    }
    auto feedback = mx::sum(mx::concatenate(feedback_embeddings, 1), 1, true);
    speech_embeddings = feedback + mx::reshape(i < trailing->shape(0)
                                                   ? rows(*trailing, i, i + 1)
                                                   : rows(special, 2, 3),
                                               {1, 1, -1});
    auto frame = mx::concatenate(frame_codes, 0);
    mx::async_eval({speech_embeddings, frame});
    codes.push_back(frame);
    ++new_frames;
    std::optional<mx::array> waveform;
    if (new_frames == 10) {
      auto decoded =
          vocoder.forward(mx::expand_dims(mx::stack(codes, 1), 0), cancelled);
      waveform =
          mx::astype(rows(decoded, context_frames * vocoder.samples_per_frame(),
                          decoded.shape(0)),
                     mx::float32);
      mx::async_eval({*waveform});
    } else {
    }
    if (i + 1 < request.max_speech_frames) {
      speech_hidden =
          talker
              .forward(speech_embeddings,
                       mx::full({3, 1, 1}, cache.front().offset + delta), cache,
                       cancelled)
              .first;
      speech_logits = codec_logits(speech_hidden);
      mx::async_eval(cache_arrays(cache, {speech_logits, speech_hidden}));
    } else {
    }
    if (waveform) {
      mx::eval(*waveform);
      emit({"",
            encode_wav({waveform->data<float>(),
                        waveform->data<float>() + waveform->size()}),
            "", 0, 0});
      if (codes.size() > 25) {
        codes.erase(codes.begin(), codes.end() - 25);
      } else {
      }
      context_frames = codes.size();
      new_frames = 0;
    } else {
    }
  }
  if (history.empty()) {
    throw std::invalid_argument("talker generated no audio");
  } else if (new_frames > 0) {
    check_cancelled(cancelled);
    auto decoded =
        vocoder.forward(mx::expand_dims(mx::stack(codes, 1), 0), cancelled);
    auto waveform =
        mx::astype(rows(decoded, context_frames * vocoder.samples_per_frame(),
                        decoded.shape(0)),
                   mx::float32);
    mx::eval(waveform);
    emit({"",
          encode_wav({waveform.data<float>(),
                      waveform.data<float>() + waveform.size()}),
          "", 0, 0});
  } else {
  }
}
} // namespace qwen3_omni
