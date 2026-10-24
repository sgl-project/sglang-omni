// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Prince Canuma and contributors.
#include "qwen3_tts/qwen3_tts_model.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace qwen3_tts {
namespace mx = mlx::core;
using qwen3_asr::WeightMap;

void LinearShapes(WeightShapes &shapes, const std::string &prefix, int input,
                  int output, bool bias) {
  shapes[prefix + ".weight"] = {output, input};
  if (bias) {
    shapes[prefix + ".bias"] = {output};
  } else {
  }
}

void TransformerShapes(WeightShapes &shapes, const std::string &prefix,
                       const nlohmann::json &config) {
  const int hidden = config.at("hidden_size");
  const int intermediate = config.at("intermediate_size");
  const int head_dim = config.at("head_dim");
  const int heads = config.at("num_attention_heads");
  const int kv_heads = config.at("num_key_value_heads");
  if (hidden <= 0 || intermediate <= 0 || head_dim <= 0 || head_dim % 2 != 0 ||
      heads <= 0 || kv_heads <= 0 || heads % kv_heads != 0 ||
      config.at("num_hidden_layers").get<int>() <= 0 ||
      config.value("attention_bias", false) ||
      config.value("hidden_act", std::string("silu")) != "silu") {
    throw std::invalid_argument(
        "unsupported Qwen3-TTS transformer configuration");
  } else {
  }
  for (int layer = 0; layer < config.at("num_hidden_layers").get<int>();
       ++layer) {
    const auto stem = prefix + ".layers." + std::to_string(layer);
    shapes[stem + ".input_layernorm.weight"] = {hidden};
    shapes[stem + ".post_attention_layernorm.weight"] = {hidden};
    shapes[stem + ".self_attn.q_norm.weight"] = {head_dim};
    shapes[stem + ".self_attn.k_norm.weight"] = {head_dim};
    LinearShapes(shapes, stem + ".self_attn.q_proj", hidden, heads * head_dim);
    LinearShapes(shapes, stem + ".self_attn.k_proj", hidden,
                 kv_heads * head_dim);
    LinearShapes(shapes, stem + ".self_attn.v_proj", hidden,
                 kv_heads * head_dim);
    LinearShapes(shapes, stem + ".self_attn.o_proj", heads * head_dim, hidden);
    LinearShapes(shapes, stem + ".mlp.gate_proj", hidden, intermediate);
    LinearShapes(shapes, stem + ".mlp.up_proj", hidden, intermediate);
    LinearShapes(shapes, stem + ".mlp.down_proj", intermediate, hidden);
  }
  shapes[prefix + ".norm.weight"] = {hidden};
}

void ValidateWeights(const WeightMap &weights, const WeightShapes &shapes) {
  for (const auto &[name, shape] : shapes) {
    const auto found = weights.find(name);
    if (found == weights.end() || found->second.shape() != shape) {
      throw std::runtime_error("missing or incorrectly shaped weight: " + name);
    } else {
    }
  }
  for (const auto &[name, weight] : weights) {
    if (!shapes.contains(name)) {
      throw std::runtime_error("unexpected inference weight: " + name);
    } else {
    }
  }
}

namespace {

WeightMap LoadModel(const std::filesystem::path &directory,
                    const nlohmann::json &config) {
  if (config.at("model_type") != "qwen3_tts" ||
      config.at("tokenizer_type") != "qwen3_tts_tokenizer_12hz" ||
      config.contains("quantization")) {
    throw std::invalid_argument(
        "expected an unquantized Qwen3-TTS 12Hz checkpoint");
  } else {
  }
  const auto &talker = config.at("talker_config");
  const auto &predictor = talker.at("code_predictor_config");
  const int hidden = talker.at("hidden_size");
  const int predictor_hidden = predictor.at("hidden_size");
  const int text_hidden = talker.at("text_hidden_size");
  if (talker.at("num_code_groups") != kCodeGroups ||
      predictor.at("num_code_groups") != kCodeGroups ||
      predictor.at("vocab_size") != kCodebookSize ||
      talker.at("vocab_size") != kCodebookSize + 1024 ||
      (config.at("tts_model_type") != "custom_voice" &&
       config.at("tts_model_type") != "voice_design" &&
       config.at("tts_model_type") != "base")) {
    throw std::invalid_argument(
        "unsupported Qwen3-TTS model kind or codec groups");
  } else {
  }
  WeightShapes shapes;
  TransformerShapes(shapes, "talker.model", talker);
  TransformerShapes(shapes, "talker.code_predictor.model", predictor);
  shapes["talker.model.text_embedding.weight"] = {talker.at("text_vocab_size"),
                                                  text_hidden};
  shapes["talker.model.codec_embedding.weight"] = {talker.at("vocab_size"),
                                                   hidden};
  LinearShapes(shapes, "talker.text_projection.linear_fc1", text_hidden,
               text_hidden, true);
  LinearShapes(shapes, "talker.text_projection.linear_fc2", text_hidden, hidden,
               true);
  LinearShapes(shapes, "talker.codec_head", hidden, talker.at("vocab_size"));
  if (hidden != predictor_hidden) {
    LinearShapes(shapes, "talker.code_predictor.small_to_mtp_projection",
                 hidden, predictor_hidden, true);
  } else {
  }
  for (int group = 0; group < kCodeGroups - 1; ++group) {
    const auto index = std::to_string(group);
    shapes["talker.code_predictor.model.codec_embedding." + index + ".weight"] =
        {kCodebookSize, hidden};
    LinearShapes(shapes, "talker.code_predictor.lm_head." + index,
                 predictor_hidden, kCodebookSize);
  }
  WeightMap weights = qwen3_asr::LoadSafetensors(directory);
  WeightMap talker_weights;
  for (const auto &[name, weight] : weights) {
    if (name.starts_with("talker.")) {
      talker_weights.emplace(name, weight);
    } else if (name.starts_with("speaker_encoder.") &&
               config.at("tts_model_type") == "base") {
    } else {
      throw std::runtime_error("unexpected model weight: " + name);
    }
  }
  ValidateWeights(talker_weights, shapes);
  for (const auto &name :
       {"codec_eos_token_id", "codec_think_id", "codec_nothink_id",
        "codec_think_bos_id", "codec_think_eos_id", "codec_pad_id",
        "codec_bos_id"}) {
    const int token = talker.at(name);
    if (token < kCodebookSize || token >= talker.at("vocab_size").get<int>()) {
      throw std::invalid_argument(std::string("invalid codec control token: ") +
                                  name);
    } else {
    }
  }
  for (const auto &name :
       {"tts_pad_token_id", "tts_bos_token_id", "tts_eos_token_id"}) {
    const int token = config.at(name);
    if (token < 0 || token >= talker.at("text_vocab_size").get<int>()) {
      throw std::invalid_argument(std::string("invalid text control token: ") +
                                  name);
    } else {
    }
  }
  // Note (Akazaakane): Official speaker convolutions are OIK; serving uses OKI.
  for (auto &[name, weight] : weights) {
    if (name.starts_with("speaker_encoder.") && weight.ndim() == 3) {
      weight = mx::transpose(weight, {0, 2, 1});
    } else {
    }
  }
  if (config.at("tts_model_type") == "base") {
    SpeakerShapes(shapes, config);
  } else {
  }
  ValidateWeights(weights, shapes);
  return weights;
}

} // namespace

Model::Model(const std::filesystem::path &directory)
    : config_(qwen3_asr::ReadJson(directory / "config.json")),
      talker_config_(config_.at("talker_config")),
      checkpoint_(LoadModel(directory, config_), nlohmann::json::object()),
      talker_(checkpoint_, "talker.model",
              qwen3_asr::Qwen3DecoderConfig::FromJson(talker_config_)),
      predictor_(checkpoint_, "talker.code_predictor.model",
                 qwen3_asr::Qwen3DecoderConfig::FromJson(
                     talker_config_.at("code_predictor_config"))),
      tokenizer_(directory), codec_(directory / "speech_tokenizer",
                                    config_.at("tts_model_type") == "base") {}

mx::array TimeSlice(const mx::array &values, int start, int stop) {
  return mx::slice(values, {0, start, 0},
                   {values.shape(0), stop, values.shape(2)});
}

mx::array Model::Text(const std::vector<int> &ids) const {
  const mx::array indices(ids.data(), {1, static_cast<int>(ids.size())},
                          mx::int32);
  const auto embeddings =
      checkpoint_.Embed(indices, "talker.model.text_embedding");
  return checkpoint_.Linear(
      qwen3_asr::Silu(
          checkpoint_.Linear(embeddings, "talker.text_projection.linear_fc1")),
      "talker.text_projection.linear_fc2");
}

mx::array Model::CodecEmbedding(const std::vector<int> &ids) const {
  return checkpoint_.Embed(
      mx::array(ids.data(), {1, static_cast<int>(ids.size())}, mx::int32),
      "talker.model.codec_embedding");
}

mx::array Model::Sample(mx::array logits, const Sampling &sampling,
                        mx::random::KeySequence &random_sequence,
                        const std::vector<int> &history, bool semantic) const {
  logits = mx::reshape(logits, {-1});
  const auto negative_infinity =
      mx::array(-std::numeric_limits<float>::infinity(), logits.dtype());
  if (semantic) {
    const auto indices = mx::arange(logits.shape(0));
    const auto valid =
        (indices < mx::array(kCodebookSize)) |
        (indices ==
         mx::array(talker_config_.at("codec_eos_token_id").get<int>()));
    logits = mx::where(valid, logits, negative_infinity);
  } else {
  }
  if (!history.empty() && sampling.repetition_penalty != 1.0f) {
    const std::set<int> recent(
        history.begin() + std::max(0, static_cast<int>(history.size()) -
                                          kRepetitionContextFrames),
        history.end());
    const std::vector<int> tokens(recent.begin(), recent.end());
    const mx::array indices(tokens.data(), {static_cast<int>(tokens.size())},
                            mx::int32);
    const auto selected = mx::take(logits, indices, 0);
    logits =
        mx::put_along_axis(logits, indices,
                           mx::where(selected < mx::array(0),
                                     selected * sampling.repetition_penalty,
                                     selected / sampling.repetition_penalty),
                           0);
  } else {
  }
  if (sampling.temperature == 0.0f) {
    return mx::reshape(mx::astype(mx::argmax(logits), mx::int32), {1, 1});
  } else {
  }
  logits = logits / sampling.temperature;
  if (sampling.top_k > 0 && sampling.top_k < logits.shape(0)) {
    logits = mx::where(logits < mx::min(mx::topk(logits, sampling.top_k)),
                       negative_infinity, logits);
  } else {
  }
  if (sampling.top_p < 1.0f) {
    const auto probabilities = mx::softmax(mx::astype(logits, mx::float32));
    const auto order = mx::argsort(probabilities);
    const auto keep = mx::cumsum(mx::take(probabilities, order, 0)) >
                      mx::array(1.0f - sampling.top_p);
    logits = mx::where(mx::put_along_axis(mx::zeros_like(keep), order, keep, 0),
                       logits, negative_infinity);
  } else {
  }
  return mx::reshape(
      mx::astype(mx::random::categorical(mx::astype(logits, mx::float32), -1,
                                         random_sequence.next()),
                 mx::int32),
      {1, 1});
}

Result Model::Synthesize(
    const Request &request, const std::function<void()> &check_cancelled,
    const std::function<void(const std::vector<float> &)> &emit) const {
  check_cancelled();
  const auto prompt = BuildPrompt(request, check_cancelled);
  if (prompt.embeddings.shape(1) + request.max_frames >
      talker_config_.at("max_position_embeddings").get<int>()) {
    throw std::invalid_argument(
        "prompt and generated frames exceed model context");
  } else {
  }
  auto caches = talker_.NewCaches();
  mx::array embeddings = prompt.embeddings;
  mx::random::KeySequence random_sequence(request.seed);
  AudioCodec::State codec_state;
  std::vector<int> history;
  std::vector<mx::array> pending;
  Result result{{}, {}, "length"};
  const auto decode = [&] {
    if (pending.empty()) {
      return;
    } else {
    }
    auto waveform = mx::astype(codec_.Decode(mx::concatenate(pending, 0),
                                             codec_state, check_cancelled),
                               mx::float32);
    mx::eval(waveform);
    check_cancelled();
    const std::vector<float> samples(waveform.data<float>(),
                                     waveform.data<float>() + waveform.size());
    if (emit) {
      emit(samples);
    } else {
      result.samples.insert(result.samples.end(), samples.begin(),
                            samples.end());
    }
    pending.clear();
  };
  if (prompt.reference_codes.has_value()) {
    auto warmup =
        codec_.Decode(*prompt.reference_codes, codec_state, check_cancelled);
    mx::eval(warmup);
  } else {
  }
  for (int frame = 0; frame < request.max_frames; ++frame) {
    check_cancelled();
    const auto hidden = checkpoint_.RmsNorm(
        talker_.Forward(embeddings, caches), "talker.model.norm",
        talker_config_.at("rms_norm_eps").get<float>());
    const auto last = TimeSlice(hidden, hidden.shape(1) - 1, hidden.shape(1));
    const auto semantic =
        Sample(checkpoint_.Linear(last, "talker.codec_head"), request.semantic,
               random_sequence, history, true);
    mx::eval(semantic);
    const int token = semantic.item<int>();
    if (token == talker_config_.at("codec_eos_token_id").get<int>()) {
      result.finish_reason = "stop";
      break;
    } else {
      history.push_back(token);
    }
    std::vector<mx::array> codes{semantic};
    auto predictor_caches = predictor_.NewCaches();
    auto summed = checkpoint_.Embed(semantic, "talker.model.codec_embedding");
    auto predictor_input = mx::concatenate({last, summed}, 1);
    for (int group = 0; group < kCodeGroups - 1; ++group) {
      check_cancelled();
      if (checkpoint_.Has(
              "talker.code_predictor.small_to_mtp_projection.weight")) {
        predictor_input = checkpoint_.Linear(
            predictor_input, "talker.code_predictor.small_to_mtp_projection");
      } else {
      }
      const auto &predictor_config = talker_config_.at("code_predictor_config");
      auto predictor_hidden = checkpoint_.RmsNorm(
          predictor_.Forward(predictor_input, predictor_caches),
          "talker.code_predictor.model.norm",
          predictor_config.at("rms_norm_eps").get<float>());
      const auto code =
          Sample(checkpoint_.Linear(
                     TimeSlice(predictor_hidden, predictor_hidden.shape(1) - 1,
                               predictor_hidden.shape(1)),
                     "talker.code_predictor.lm_head." + std::to_string(group)),
                 request.subtalker, random_sequence, {}, false);
      codes.push_back(code);
      predictor_input = checkpoint_.Embed(
          code, "talker.code_predictor.model.codec_embedding." +
                    std::to_string(group));
      summed = summed + predictor_input;
    }
    embeddings = summed + (frame < prompt.trailing.shape(1)
                               ? TimeSlice(prompt.trailing, frame, frame + 1)
                               : prompt.pad);
    const auto complete_frame = mx::concatenate(codes, 1);
    mx::eval({complete_frame, embeddings});
    check_cancelled();
    result.codes.emplace_back(complete_frame.data<int>(),
                              complete_frame.data<int>() + kCodeGroups);
    pending.push_back(complete_frame);
    if (pending.size() >= kCodecChunkFrames) {
      decode();
    } else {
    }
  }
  decode();
  if (result.codes.empty()) {
    throw std::runtime_error("Qwen3-TTS generated no audio frames");
  } else {
  }
  return result;
}

} // namespace qwen3_tts
