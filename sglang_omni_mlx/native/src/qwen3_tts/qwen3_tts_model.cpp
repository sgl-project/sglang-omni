// SPDX-License-Identifier: Apache-2.0
#include "qwen3_tts/qwen3_tts_model.h"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <limits>
#include <random>
#include <set>
#include <stdexcept>

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

std::string Lower(const std::string &text) {
  std::string lowered;
  for (const char character : text) {
    lowered.push_back(
        static_cast<char>(std::tolower(static_cast<unsigned char>(character))));
  }
  return lowered;
}

// Rows [start, stop) of x [1, length, hidden].
mx::array Rows(const mx::array &x, int start, int stop) {
  return mx::slice(x, {0, start, 0}, {1, stop, x.shape(2)});
}

// logits [1, vocabulary] with entries at ids set to value.
mx::array PutValues(const mx::array &logits, const std::vector<int> &ids,
                    const mx::array &values) {
  const mx::array indices =
      mx::array(ids.begin(), {1, static_cast<int>(ids.size())}, mx::int32);
  return mx::put_along_axis(logits, indices, values, -1);
}

// A token [1, 1] from the last logits [1, vocabulary]: control codes
// suppressed, recent tokens penalized, then temperature, top-k, top-p and a
// categorical draw, in the reference sampler's order.
mx::array Sample(mx::array logits, const SamplerConfig &sampler,
                 const std::vector<int> &suppressed_ids,
                 float repetition_penalty, int repetition_window_frames,
                 const std::vector<int> &history,
                 mx::random::KeySequence &keys) {
  const mx::Dtype dtype = logits.dtype();
  const mx::array negative_infinity =
      mx::array(-std::numeric_limits<float>::infinity(), dtype);
  if (!suppressed_ids.empty()) {
    logits = PutValues(logits, suppressed_ids, negative_infinity);
  } else {
  }
  if (!history.empty() && repetition_penalty != 1.0f) {
    const size_t window = static_cast<size_t>(repetition_window_frames);
    const size_t window_start =
        window > 0 && history.size() > window ? history.size() - window : 0;
    const std::set<int> recent(history.begin() + window_start, history.end());
    const std::vector<int> ids(recent.begin(), recent.end());
    const mx::array indices =
        mx::array(ids.begin(), {static_cast<int>(ids.size())}, mx::int32);
    const mx::array selected = mx::take(logits, indices, -1);
    const mx::array penalty = mx::array(repetition_penalty, dtype);
    logits = PutValues(logits, ids,
                       mx::where(mx::less(selected, mx::array(0.0f, dtype)),
                                 mx::multiply(selected, penalty),
                                 mx::divide(selected, penalty)));
  } else {
  }
  if (sampler.temperature <= 0.0f) {
    return mx::astype(mx::argmax(logits, -1, true), mx::int32);
  } else {
  }
  if (sampler.temperature != 1.0f) {
    logits = mx::divide(logits, mx::array(sampler.temperature, dtype));
  } else {
  }
  const int vocabulary_size = logits.shape(-1);
  if (sampler.top_k > 0 && sampler.top_k < vocabulary_size) {
    const mx::array masked_ids =
        mx::slice(mx::argpartition(mx::negative(logits), sampler.top_k - 1, -1),
                  {0, sampler.top_k}, {1, vocabulary_size});
    logits = mx::put_along_axis(logits, masked_ids, negative_infinity, -1);
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
                    mx::array(1.0f - sampler.top_p, dtype)),
        mx::equal(log_probabilities, mx::max(log_probabilities, -1, true)));
    logits = mx::where(
        mx::equal(mx::where(kept, log_probabilities, negative_infinity),
                  negative_infinity),
        negative_infinity, logits);
  } else {
  }
  return mx::astype(
      mx::expand_dims(
          mx::random::categorical(mx::multiply(logits, mx::array(1.0f, dtype)),
                                  -1, keys.next()),
          1),
      mx::int32);
}

// Note (Jiaxin Deng): the talker's multimodal rotary positions are all the
// text position for speech, which is plain rotary embedding; tables rounded
// to bfloat16 keep the reference's codes.
qwen3_asr::Qwen3DecoderConfig DecoderConfig(const nlohmann::json &config) {
  qwen3_asr::Qwen3DecoderConfig decoder =
      qwen3_asr::Qwen3DecoderConfig::FromJson(config);
  decoder.rotary_tables = true;
  return decoder;
}

} // namespace

Qwen3TtsModel::Qwen3TtsModel(const std::filesystem::path &model_directory)
    : Qwen3TtsModel(model_directory,
                    omni_common::ReadJson(model_directory / "config.json")) {}

Qwen3TtsModel::Qwen3TtsModel(const std::filesystem::path &model_directory,
                             const nlohmann::json &config)
    : checkpoint_(omni_common::LoadSafetensors(model_directory), config),
      tokenizer_(model_directory),
      speech_decoder_(model_directory / "speech_tokenizer"),
      talker_(checkpoint_, kTalkerPrefix,
              DecoderConfig(config.at("talker_config"))),
      code_predictor_(
          checkpoint_, kCodePredictorPrefix,
          DecoderConfig(
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
    speaker_ids_.emplace(Lower(name), id.get<int>());
  }
  // The JSON object above sorts its keys; the speakers' order is the file's.
  std::ifstream config_file(model_directory / "config.json");
  const nlohmann::ordered_json ordered_config =
      nlohmann::ordered_json::parse(config_file);
  for (const auto &[name, id] :
       ordered_config.at("talker_config").at("spk_id").items()) {
    speaker_names_.push_back(Lower(name));
  }
  for (const auto &[name, dialect] : talker.at("spk_is_dialect").items()) {
    if (dialect.is_string()) {
      speaker_dialects_.emplace(Lower(name), Lower(dialect.get<std::string>()));
    } else {
    }
  }
  for (const auto &[name, id] : talker.at("codec_language_id").items()) {
    language_ids_.emplace(Lower(name), id.get<int>());
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
  const std::string speaker_name = Lower(speaker);
  const auto found_speaker = speaker_ids_.find(speaker_name);
  if (found_speaker == speaker_ids_.end()) {
    throw std::invalid_argument("unknown speaker " + speaker);
  } else {
  }
  prompt.speaker_id = found_speaker->second;
  const std::string language_name = Lower(language);
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
  std::vector<qwen3_asr::KVCache> caches = code_predictor_.NewCaches();
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
    const mx::array logits =
        checkpoint_.Linear(normed, "talker.code_predictor.lm_head." +
                                       std::to_string(codebook - 1));
    const int length = logits.shape(1);
    codes.push_back(Sample(
        mx::reshape(Rows(logits, length - 1, length), {1, logits.shape(2)}),
        sampler, {}, 1.0f, 0, {}, keys));
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

qwen3_asr::FinishReason Qwen3TtsModel::Generate(
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

  std::vector<int> suppressed_ids;
  for (int id = codec_vocabulary_size_ - kControlCodeCount;
       id < codec_vocabulary_size_; ++id) {
    if (id != codec_eos_id_) {
      suppressed_ids.push_back(id);
    } else {
    }
  }
  std::vector<int> history;
  std::vector<qwen3_asr::KVCache> caches = talker_.NewCaches();
  for (int frame = 0; frame < max_frames; ++frame) {
    if (cancel.load()) {
      throw qwen3_asr::TranscriptionCancelled();
    } else {
    }
    const mx::array hidden = checkpoint_.RmsNorm(
        talker_.Forward(embeddings, caches),
        std::string(kTalkerPrefix) + ".norm", talker_norm_eps_);
    const mx::array logits = checkpoint_.Linear(hidden, "talker.codec_head");
    const int length = logits.shape(1);
    const mx::array first_code = Sample(
        mx::reshape(Rows(logits, length - 1, length), {1, logits.shape(2)}),
        sampling.talker, suppressed_ids, sampling.repetition_penalty,
        sampling.repetition_window_frames, history, keys);
    mx::eval(first_code);
    const int first_code_id = static_cast<int>(first_code.item<int32_t>());
    if (first_code_id == codec_eos_id_) {
      return qwen3_asr::FinishReason::kStop;
    } else {
    }
    history.push_back(first_code_id);
    const mx::array codes =
        PredictCodes(Rows(hidden, length - 1, length), first_code,
                     sampling.code_predictor, keys);
    const mx::array frame_codes = mx::reshape(codes, {-1});
    mx::eval(frame_codes);
    if (frame + 1 < max_frames) {
      embeddings = mx::add(
          frame < trailing_length ? Rows(trailing, frame, frame + 1) : tts_pad,
          EmbedFrame(codes));
      // Note (Minyi Tang): the talker steps on the GPU while on_frame
      // decodes audio.
      mx::async_eval(embeddings);
    } else {
    }
    on_frame(frame_codes);
  }
  return qwen3_asr::FinishReason::kLength;
}

} // namespace qwen3_tts
