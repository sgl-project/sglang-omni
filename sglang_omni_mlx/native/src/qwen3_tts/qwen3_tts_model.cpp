// SPDX-License-Identifier: Apache-2.0
#include "qwen3_tts/qwen3_tts_model.h"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <limits>
#include <optional>
#include <random>
#include <stdexcept>

#include "common/generation.h"
#include "common/kv_cache.h"
#include "common/text.h"

namespace qwen3_tts {

namespace mx = mlx::core;

namespace {

constexpr const char *kTalkerPrefix = "talker.model";
constexpr const char *kCodePredictorPrefix = "talker.code_predictor.model";
// Note (Jiaxin Deng): the codec vocabulary's last 1024 ids are control codes,
// which the talker never samples except the end of speech.
constexpr int kControlCodeCount = 1024;
// The chat template's assistant turn: <|im_start|>assistant\n before the text
// and <|im_end|>\n<|im_start|>assistant\n after it.
constexpr int kRoleTokenCount = 3;
constexpr int kClosingTokenCount = 5;
// Note (XXXXRT666): the layers are quantized at load, the talker's to 8 bits
// and the code predictor's, which a frame runs 15 times, to 4; the speech
// keeps its words, which a 4-bit talker did not. The embeddings and heads stay
// in the checkpoint dtype.
constexpr int kTalkerQuantizationBits = 8;
constexpr int kCodePredictorQuantizationBits = 4;
constexpr int kQuantizationGroupSize = 64;

int LayerBits(const std::string &name) {
  if (name.starts_with("talker.model.layers.")) {
    return kTalkerQuantizationBits;
  } else if (name.starts_with("talker.code_predictor.model.layers.")) {
    return kCodePredictorQuantizationBits;
  } else {
    return 0;
  }
}

// Rows [start, stop) of x [1, length, hidden].
mx::array Rows(const mx::array &x, int start, int stop) {
  return mx::slice(x, {0, start, 0}, {1, stop, x.shape(2)});
}

// A token [1, 1] from the last logits [1, vocabulary] and Gumbel noise of
// their shape: temperature, top-k, top-p, then a Gumbel-max draw, which is a
// categorical draw, in the reference sampler's order.
mx::array Sample(const mx::array &last_logits, const SamplerConfig &sampler,
                 const mx::array &noise) {
  if (sampler.temperature <= 0.0f) {
    return mx::astype(mx::argmax(last_logits, -1, true), mx::int32);
  } else {
  }
  const mx::array negative_infinity =
      mx::array(-std::numeric_limits<float>::infinity());
  mx::array logits = mx::astype(last_logits, mx::float32);
  if (sampler.temperature != 1.0f) {
    logits = mx::multiply(logits, mx::array(1.0f / sampler.temperature));
  } else {
  }
  const int vocabulary_size = logits.shape(-1);
  if (sampler.top_k > 0 && sampler.top_k < vocabulary_size) {
    const int kth = vocabulary_size - sampler.top_k;
    const mx::array threshold =
        mx::slice(mx::partition(logits, kth, -1), {0, kth}, {1, kth + 1});
    logits = mx::where(mx::less(logits, threshold), negative_infinity, logits);
  } else {
  }
  if (sampler.top_p > 0.0f && sampler.top_p < 1.0f) {
    const mx::array log_probabilities =
        mx::subtract(logits, mx::logsumexp(logits, -1, true));
    const mx::array sorted_ids = mx::argsort(log_probabilities, -1);
    const mx::array cumulative = mx::cumsum(
        mx::take_along_axis(mx::exp(log_probabilities), sorted_ids, -1), -1);
    const mx::array inverse_ids = mx::put_along_axis(
        mx::zeros_like(sorted_ids), sorted_ids,
        mx::reshape(mx::arange(vocabulary_size, sorted_ids.dtype()),
                    {1, vocabulary_size}),
        -1);
    // Note (Jiaxin Deng): the likeliest token always stays, as CUDA's sampler
    // keeps it, even where 1 - top_p rounds to 1.
    const mx::array kept = mx::logical_or(
        mx::greater(mx::take_along_axis(cumulative, inverse_ids, -1),
                    mx::array(1.0f - sampler.top_p)),
        mx::equal(log_probabilities, mx::max(log_probabilities, -1, true)));
    logits = mx::where(kept, logits, negative_infinity);
  } else {
  }
  return mx::astype(mx::argmax(mx::add(logits, noise), -1, true), mx::int32);
}

} // namespace

Qwen3TtsModel::Qwen3TtsModel(const std::filesystem::path &model_directory)
    : Qwen3TtsModel(model_directory,
                    omni_common::ReadJson(model_directory / "config.json")) {}

Qwen3TtsModel::Qwen3TtsModel(const std::filesystem::path &model_directory,
                             const nlohmann::json &config)
    : checkpoint_(omni_common::QuantizedCheckpoint(
          omni_common::LoadSafetensors(model_directory), config,
          kQuantizationGroupSize, LayerBits)),
      tokenizer_(model_directory),
      speech_decoder_(model_directory / "speech_tokenizer"),
      // Note (Jiaxin Deng): the talker's multimodal rotary positions are all
      // the text position for speech, which is plain rotary embedding.
      talker_(checkpoint_, kTalkerPrefix,
              omni_common::Qwen3DecoderConfig::FromJson(
                  config.at("talker_config"))),
      code_predictor_(
          checkpoint_, kCodePredictorPrefix,
          omni_common::Qwen3DecoderConfig::FromJson(
              config.at("talker_config").at("code_predictor_config"))) {
  if (config.value("tts_model_type", std::string()) != "custom_voice") {
    throw std::runtime_error("only Qwen3-TTS CustomVoice checkpoints are "
                             "supported");
  } else {
  }
  const nlohmann::json &talker = config.at("talker_config");
  const nlohmann::json &code_predictor = talker.at("code_predictor_config");
  if (code_predictor.at("hidden_size") != talker.at("hidden_size")) {
    throw std::runtime_error("the code predictor must share the talker's "
                             "hidden size");
  } else {
  }
  talker_norm_eps_ = talker.at("rms_norm_eps").get<float>();
  code_predictor_norm_eps_ = code_predictor.at("rms_norm_eps").get<float>();
  codebook_count_ = talker.at("num_code_groups").get<int>();
  codec_vocabulary_size_ = talker.at("vocab_size").get<int>();
  tts_bos_token_id_ = config.at("tts_bos_token_id").get<int>();
  tts_eos_token_id_ = config.at("tts_eos_token_id").get<int>();
  tts_pad_token_id_ = config.at("tts_pad_token_id").get<int>();
  codec_bos_id_ = talker.at("codec_bos_id").get<int>();
  codec_eos_id_ = talker.at("codec_eos_token_id").get<int>();
  codec_pad_id_ = talker.at("codec_pad_id").get<int>();
  codec_think_id_ = talker.at("codec_think_id").get<int>();
  codec_nothink_id_ = talker.at("codec_nothink_id").get<int>();
  codec_think_bos_id_ = talker.at("codec_think_bos_id").get<int>();
  codec_think_eos_id_ = talker.at("codec_think_eos_id").get<int>();
  for (const auto &[name, id] : talker.at("spk_id").items()) {
    speaker_ids_.emplace(omni_common::AsciiLowercase(name), id.get<int>());
  }
  // The JSON object above sorts its keys; the speakers' order is the file's.
  std::ifstream config_file(model_directory / "config.json");
  const nlohmann::ordered_json ordered_config =
      nlohmann::ordered_json::parse(config_file);
  for (const auto &[name, id] :
       ordered_config.at("talker_config").at("spk_id").items()) {
    speaker_names_.push_back(omni_common::AsciiLowercase(name));
  }
  for (const auto &[name, dialect] : talker.at("spk_is_dialect").items()) {
    if (dialect.is_string()) {
      speaker_dialects_.emplace(
          omni_common::AsciiLowercase(name),
          omni_common::AsciiLowercase(dialect.get<std::string>()));
    } else {
    }
  }
  for (const auto &[name, id] : talker.at("codec_language_id").items()) {
    language_ids_.emplace(omni_common::AsciiLowercase(name), id.get<int>());
  }
  if (speech_decoder_.codebook_count() != codebook_count_) {
    throw std::runtime_error("the speech decoder and the talker disagree on "
                             "the codebook count");
  } else {
  }
}

SpeechPrompt Qwen3TtsModel::BuildPrompt(const std::string &text,
                                        const std::string &speaker,
                                        const std::string &language,
                                        const std::string &instructions) const {
  SpeechPrompt prompt;
  const std::string speaker_name = omni_common::AsciiLowercase(speaker);
  const auto found_speaker = speaker_ids_.find(speaker_name);
  if (found_speaker == speaker_ids_.end()) {
    throw std::invalid_argument("unknown speaker " + speaker);
  } else {
  }
  prompt.speaker_id = found_speaker->second;
  const std::string language_name = omni_common::AsciiLowercase(language);
  if (language_name != "auto") {
    const auto found_language = language_ids_.find(language_name);
    if (found_language == language_ids_.end()) {
      throw std::invalid_argument("unknown language " + language);
    } else {
    }
    prompt.language_id = found_language->second;
  } else {
  }
  // Note (Jiaxin Deng): as the CUDA server, a dialect speaker speaks its
  // dialect when the language is left to the model, and an explicit Chinese
  // stays Chinese.
  const auto dialect = speaker_dialects_.find(speaker_name);
  if (language_name == "auto" && dialect != speaker_dialects_.end() &&
      language_ids_.count(dialect->second) > 0) {
    prompt.language_id = language_ids_.at(dialect->second);
  } else {
  }
  prompt.text_ids = tokenizer_.Encode("<|im_start|>assistant\n" + text +
                                      "<|im_end|>\n<|im_start|>assistant\n");
  if (static_cast<int>(prompt.text_ids.size()) <
      kRoleTokenCount + 1 + kClosingTokenCount) {
    throw std::invalid_argument("the text encodes to no tokens");
  } else {
  }
  if (!instructions.empty()) {
    prompt.instruction_ids =
        tokenizer_.Encode("<|im_start|>user\n" + instructions + "<|im_end|>\n");
  } else {
  }
  return prompt;
}

std::vector<int>
Qwen3TtsModel::CodecPrefixIds(const SpeechPrompt &prompt) const {
  std::vector<int> ids =
      prompt.language_id.has_value()
          ? std::vector<int>{codec_think_id_, codec_think_bos_id_,
                             *prompt.language_id, codec_think_eos_id_}
          : std::vector<int>{codec_nothink_id_, codec_think_bos_id_,
                             codec_think_eos_id_};
  ids.insert(ids.end(), {prompt.speaker_id, codec_pad_id_, codec_bos_id_});
  return ids;
}

std::vector<int> Qwen3TtsModel::PromptIds(const SpeechPrompt &prompt) const {
  std::vector<int> ids = prompt.instruction_ids;
  const std::vector<int> codec_ids = CodecPrefixIds(prompt);
  ids.insert(ids.end(), codec_ids.begin(), codec_ids.end());
  ids.insert(ids.end(), prompt.text_ids.begin(), prompt.text_ids.end());
  return ids;
}

int Qwen3TtsModel::PromptLength(const SpeechPrompt &prompt) const {
  // The codec ids follow the role tokens; the last rides on the first text
  // row.
  const int codec_length = static_cast<int>(CodecPrefixIds(prompt).size());
  const int text_length = static_cast<int>(prompt.text_ids.size());
  const int text_rows = prompt.text_up_front ? text_length - kRoleTokenCount -
                                                   kClosingTokenCount + 2
                                             : 1;
  return static_cast<int>(prompt.instruction_ids.size()) + kRoleTokenCount +
         codec_length - 1 + text_rows;
}

mx::array Qwen3TtsModel::EmbedText(const std::vector<int> &ids) const {
  const mx::array embedded = checkpoint_.Embed(
      mx::array(ids.begin(), {1, static_cast<int>(ids.size())}, mx::int32),
      std::string(kTalkerPrefix) + ".text_embedding");
  return checkpoint_.Linear(omni_common::Silu(checkpoint_.Linear(
                                embedded, "talker.text_projection.linear_fc1")),
                            "talker.text_projection.linear_fc2");
}

mx::array Qwen3TtsModel::EmbedCodec(const std::vector<int> &ids) const {
  return checkpoint_.Embed(
      mx::array(ids.begin(), {1, static_cast<int>(ids.size())}, mx::int32),
      std::string(kTalkerPrefix) + ".codec_embedding");
}

mx::array Qwen3TtsModel::PredictCodes(const mx::array &hidden,
                                      const mx::array &first_code,
                                      const SamplerConfig &sampler,
                                      mx::random::KeySequence &keys) const {
  std::vector<omni_common::KVCache> caches = code_predictor_.NewCaches();
  const int vocabulary_size =
      checkpoint_.Weight("talker.code_predictor.lm_head.0.weight").shape(0);
  const mx::array noise = mx::random::gumbel(
      {codebook_count_ - 1, vocabulary_size}, mx::float32, keys.next());
  std::vector<mx::array> codes = {first_code};
  mx::array input = mx::concatenate(
      {hidden, checkpoint_.Embed(first_code, std::string(kTalkerPrefix) +
                                                 ".codec_embedding")},
      1);
  for (int codebook = 1; codebook < codebook_count_; ++codebook) {
    if (codebook > 1) {
      input = checkpoint_.Embed(
          codes.back(), std::string(kCodePredictorPrefix) +
                            ".codec_embedding." + std::to_string(codebook - 2));
    } else {
    }
    const mx::array normed = checkpoint_.RmsNorm(
        code_predictor_.Forward(input, caches),
        std::string(kCodePredictorPrefix) + ".norm", code_predictor_norm_eps_);
    const int length = normed.shape(1);
    const mx::array logits = checkpoint_.Linear(
        Rows(normed, length - 1, length),
        "talker.code_predictor.lm_head." + std::to_string(codebook - 1));
    codes.push_back(Sample(
        mx::reshape(logits, {1, logits.shape(2)}), sampler,
        mx::slice(noise, {codebook - 1, 0}, {codebook, vocabulary_size})));
  }
  return mx::concatenate(codes, 1);
}

mx::array Qwen3TtsModel::EmbedFrame(const mx::array &codes) const {
  mx::array embedding =
      checkpoint_.Embed(mx::slice(codes, {0, 0}, {1, 1}),
                        std::string(kTalkerPrefix) + ".codec_embedding");
  for (int codebook = 1; codebook < codebook_count_; ++codebook) {
    embedding = mx::add(
        embedding, checkpoint_.Embed(
                       mx::slice(codes, {0, codebook}, {1, codebook + 1}),
                       std::string(kCodePredictorPrefix) + ".codec_embedding." +
                           std::to_string(codebook - 1)));
  }
  return embedding;
}

omni_common::FinishReason Qwen3TtsModel::Generate(
    const SpeechPrompt &prompt, const SamplingConfig &sampling, int max_frames,
    const std::atomic<bool> &cancel,
    const std::function<void(const mx::array &)> &on_frame) const {
  const mx::array text = EmbedText(prompt.text_ids);
  const mx::array special =
      EmbedText({tts_bos_token_id_, tts_eos_token_id_, tts_pad_token_id_});
  const mx::array tts_bos = Rows(special, 0, 1);
  const mx::array tts_eos = Rows(special, 1, 2);
  const mx::array tts_pad = Rows(special, 2, 3);

  // Note (Jiaxin Deng): the thinking and language ids, the speaker id, and
  // the pad and start ids are looked up apart, as the reference does.
  const std::vector<int> codec_ids = CodecPrefixIds(prompt);
  const auto speaker_position = codec_ids.end() - 3;
  const mx::array codec = mx::concatenate(
      {EmbedCodec(std::vector<int>(codec_ids.begin(), speaker_position)),
       EmbedCodec({prompt.speaker_id}),
       EmbedCodec({codec_pad_id_, codec_bos_id_})},
      1);
  const int codec_length = codec.shape(1);
  // Text pads over every codec position but the last two, then the
  // text-to-speech start; the first text token rides on the codec start.
  const mx::array overlay = mx::add(
      mx::concatenate(
          {mx::broadcast_to(tts_pad, {1, codec_length - 2, tts_pad.shape(2)}),
           tts_bos},
          1),
      Rows(codec, 0, codec_length - 1));
  const int text_length = text.shape(1);
  const mx::array conditioned =
      mx::concatenate({Rows(text, 0, kRoleTokenCount), overlay}, 1);
  mx::array embeddings = conditioned;
  // What the talker reads beside each frame's codes: the rest of the text,
  // then its end, then pads.
  mx::array trailing = tts_pad;
  if (prompt.text_up_front) {
    const mx::array text_rows = mx::concatenate(
        {Rows(text, kRoleTokenCount, text_length - kClosingTokenCount),
         tts_eos},
        1);
    const mx::array codec_pad = EmbedCodec({codec_pad_id_});
    embeddings = mx::concatenate(
        {conditioned,
         mx::add(text_rows, mx::broadcast_to(codec_pad, {1, text_rows.shape(1),
                                                         codec_pad.shape(2)})),
         mx::add(tts_pad, Rows(codec, codec_length - 1, codec_length))},
        1);
  } else {
    embeddings = mx::concatenate(
        {conditioned, mx::add(Rows(text, kRoleTokenCount, kRoleTokenCount + 1),
                              Rows(codec, codec_length - 1, codec_length))},
        1);
    trailing = mx::concatenate(
        {Rows(text, kRoleTokenCount + 1, text_length - kClosingTokenCount),
         tts_eos},
        1);
  }
  if (!prompt.instruction_ids.empty()) {
    embeddings =
        mx::concatenate({EmbedText(prompt.instruction_ids), embeddings}, 1);
  } else {
  }
  const int trailing_length = prompt.text_up_front ? 0 : trailing.shape(1);
  mx::random::KeySequence keys(sampling.seed.value_or(std::random_device()()));

  // Note (XXXXRT666): control codes are masked and recent first codes
  // penalized on the GPU, so a frame needs nothing from the host and the next
  // one is queued before this one is read.
  const mx::array control_codes = mx::logical_and(
      mx::greater_equal(mx::arange(codec_vocabulary_size_, mx::int32),
                        mx::array(codec_vocabulary_size_ - kControlCodeCount)),
      mx::not_equal(mx::arange(codec_vocabulary_size_, mx::int32),
                    mx::array(codec_eos_id_)));
  const mx::array negative_infinity =
      mx::array(-std::numeric_limits<float>::infinity());
  // First codes [1, frames] inside the repetition window.
  std::optional<mx::array> recent_codes;
  std::vector<omni_common::KVCache> caches = talker_.NewCaches();
  struct QueuedFrame {
    mx::array first_code;
    mx::array codes;
  };
  const auto queue_frame = [&](const mx::array &frame_embeddings) {
    const mx::array hidden = checkpoint_.RmsNorm(
        talker_.Forward(frame_embeddings, caches),
        std::string(kTalkerPrefix) + ".norm", talker_norm_eps_);
    const int length = hidden.shape(1);
    const mx::array last_hidden = Rows(hidden, length - 1, length);
    mx::array logits =
        mx::reshape(checkpoint_.Linear(last_hidden, "talker.codec_head"),
                    {1, codec_vocabulary_size_});
    logits = mx::where(control_codes, negative_infinity, logits);
    if (recent_codes.has_value() && sampling.repetition_penalty != 1.0f) {
      const mx::array is_recent =
          mx::put_along_axis(mx::zeros({1, codec_vocabulary_size_}, mx::bool_),
                             *recent_codes, mx::array(true), -1);
      logits = mx::where(
          is_recent,
          mx::where(
              mx::less(logits, mx::array(0.0f)),
              mx::multiply(logits, mx::array(sampling.repetition_penalty)),
              mx::divide(logits, mx::array(sampling.repetition_penalty))),
          logits);
    } else {
    }
    const mx::array first_code =
        Sample(logits, sampling.talker,
               mx::random::gumbel(logits.shape(), mx::float32, keys.next()));
    recent_codes = recent_codes.has_value()
                       ? mx::concatenate({*recent_codes, first_code}, 1)
                       : first_code;
    const int window = sampling.repetition_window_frames;
    if (window > 0 && recent_codes->shape(1) > window) {
      recent_codes =
          mx::slice(*recent_codes, {0, recent_codes->shape(1) - window},
                    recent_codes->shape());
    } else {
    }
    const mx::array codes =
        PredictCodes(last_hidden, first_code, sampling.code_predictor, keys);
    mx::async_eval({first_code, codes});
    return QueuedFrame{first_code, codes};
  };
  QueuedFrame current = queue_frame(embeddings);
  for (int frame = 0; frame < max_frames; ++frame) {
    if (cancel.load()) {
      throw omni_common::GenerationCancelled();
    } else {
    }
    std::optional<QueuedFrame> next;
    if (frame + 1 < max_frames) {
      // Note (Minyi Tang): the talker steps on the GPU while on_frame
      // decodes audio.
      next = queue_frame(mx::add(
          frame < trailing_length ? Rows(trailing, frame, frame + 1) : tts_pad,
          EmbedFrame(current.codes)));
    } else {
    }
    if (current.first_code.item<int32_t>() == codec_eos_id_) {
      return omni_common::FinishReason::kStop;
    } else {
    }
    const mx::array frame_codes = mx::reshape(current.codes, {-1});
    mx::eval(frame_codes);
    on_frame(frame_codes);
    if (next.has_value()) {
      current = std::move(*next);
    } else {
    }
  }
  return omni_common::FinishReason::kLength;
}

} // namespace qwen3_tts
