#include "ming3_asr/ming3_asr_transcriber.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <limits>
#include <map>
#include <stdexcept>
#include <string_view>

#include "common/audio.h"
#include "common/layers.h"
#include "common/text.h"

namespace ming3_asr {

namespace mx = mlx::core;

namespace {

constexpr int kPrefillChunkSize = 256;
constexpr const char *kInstruction =
    "Please recognize the language of this speech and transcribe it, "
    "placing a `\t` between the identified language and the transcribe.";

} // namespace

MingTranscriber::AudioTokenLayout
MingTranscriber::ResolveAudioTokens(const omni_common::Tokenizer &tokenizer,
                                    int audio_token_id) {
  const std::array<std::array<std::string_view, 3>, 2> layouts = {{
      {"<audio>", "<audioPatch>", "</audio>"},
      {"<|audio_start|>", "<|audio_pad|>", "<|audio_end|>"},
  }};
  for (const auto &layout : layouts) {
    if (tokenizer.AddedTokenId(std::string(layout[1])) != audio_token_id) {
      continue;
    } else {
    }
    const int start_id = tokenizer.AddedTokenId(std::string(layout[0]));
    const int end_id = tokenizer.AddedTokenId(std::string(layout[2]));
    if (start_id < 0 || end_id < 0) {
      throw std::invalid_argument(
          "Ming checkpoint is missing a complete audio token layout");
    } else {
    }
    return {std::string(layout[0]), end_id};
  }
  throw std::invalid_argument(
      "Ming checkpoint audio token does not match a supported layout");
}

namespace {

std::vector<int>
ResolveStopTokenIds(const std::filesystem::path &model_directory,
                    const omni_common::Tokenizer &tokenizer,
                    const MingConfig &config) {
  std::vector<int> token_ids = config.eos_token_ids;
  const int role_end_id = tokenizer.TokenId("<|role_end|>");
  if (role_end_id < 0) {
    throw std::invalid_argument("Ming checkpoint is missing <|role_end|>");
  } else {
  }
  token_ids.push_back(role_end_id);
  const auto tokenizer_config_path = model_directory / "tokenizer_config.json";
  if (std::filesystem::exists(tokenizer_config_path)) {
    const auto tokenizer_config = omni_common::ReadJson(tokenizer_config_path);
    if (tokenizer_config.contains("eos_token")) {
      const auto &eos_token = tokenizer_config.at("eos_token");
      if (!eos_token.is_string() &&
          !(eos_token.is_object() && eos_token.contains("content") &&
            eos_token.at("content").is_string())) {
        throw std::invalid_argument(
            "Ming tokenizer eos_token must be a string or have string content");
      } else {
      }
      const std::string eos_text =
          eos_token.is_string() ? eos_token.get<std::string>()
                                : eos_token.at("content").get<std::string>();
      const int tokenizer_eos_id = tokenizer.TokenId(eos_text);
      if (tokenizer_eos_id < 0) {
        throw std::invalid_argument(
            "Ming tokenizer eos_token is not in the vocabulary");
      } else {
      }
      token_ids.push_back(tokenizer_eos_id);
    } else {
      throw std::invalid_argument("Ming tokenizer config is missing eos_token");
    }
  } else {
    throw std::invalid_argument(
        "Ming checkpoint is missing tokenizer_config.json");
  }
  if (std::any_of(token_ids.begin(), token_ids.end(), [&](int token_id) {
        return token_id < 0 || token_id >= config.vocabulary_size;
      })) {
    throw std::invalid_argument(
        "Ming stop token IDs must be within vocab_size");
  } else {
  }
  std::sort(token_ids.begin(), token_ids.end());
  token_ids.erase(std::unique(token_ids.begin(), token_ids.end()),
                  token_ids.end());
  return token_ids;
}

std::optional<std::string>
LanguageCode(const std::optional<std::string> &language) {
  if (!language.has_value() || language->empty()) {
    return std::nullopt;
  } else {
    std::string hint = omni_common::StripUnicodeWhitespace(*language);
    std::transform(hint.begin(), hint.end(), hint.begin(),
                   [](unsigned char character) {
                     return static_cast<char>(std::tolower(character));
                   });
    if (hint == "en" || hint == "english") {
      return "en";
    } else if (hint == "zh" || hint == "chinese") {
      return "zh";
    } else {
      throw std::invalid_argument("Ming ASR language must be en or zh");
    }
  }
}

} // namespace

MingTranscriber::MingTranscriber(const std::filesystem::path &model_directory)
    : model_(model_directory), tokenizer_(model_directory),
      audio_tokens_(
          ResolveAudioTokens(tokenizer_, model_.config().audio_token_id)),
      stop_token_ids_(
          ResolveStopTokenIds(model_directory, tokenizer_, model_.config())) {}

MingRequest MingTranscriber::Prepare(std::vector<float> samples,
                                     const MingOptions &options) const {
  if (options.max_new_tokens < 1 || options.no_repeat_ngram_size < 0) {
    throw std::invalid_argument(
        "max_new_tokens must be positive and no_repeat_ngram_size nonnegative");
  } else {
  }
  ValidateAudio(samples);
  const auto language = LanguageCode(options.language);
  std::string instruction = kInstruction;
  if (options.prompt.has_value()) {
    if (!options.prompt->starts_with(kInstruction) ||
        options.prompt->find_first_of("\n\r") != std::string::npos) {
      throw std::invalid_argument(
          "Ming ASR prompt must start with the ASR instruction and contain no "
          "newlines");
    } else {
    }
    instruction = *options.prompt;
  } else {
  }
  if (options.hotwords.has_value() && !options.hotwords->empty()) {
    if (options.hotwords->find_first_of("\n\r") != std::string::npos) {
      throw std::invalid_argument(
          "Ming ASR hotwords must not contain newlines");
    } else {
    }
    instruction += " Hotwords: " + *options.hotwords + ".";
  } else {
  }
  const std::vector<int> prefix_token_ids =
      tokenizer_.Encode(instruction + audio_tokens_.start_token);
  std::vector<int> suffix_token_ids{audio_tokens_.end_token_id};
  if (language.has_value()) {
    const auto language_hint_token_ids =
        tokenizer_.Encode(*language == "en" ? "English\t" : "Chinese\t");
    suffix_token_ids.insert(suffix_token_ids.end(),
                            language_hint_token_ids.begin(),
                            language_hint_token_ids.end());
  } else {
  }
  const auto &config = model_.config();
  const size_t frame_count = samples.size() / omni_common::kHopLength;
  if (frame_count > static_cast<size_t>(config.max_context_token_count) * 4) {
    throw std::invalid_argument("audio exceeds the Ming ASR context budget");
  } else {
  }
  const int audio_token_count =
      config.AudioTokenCount(static_cast<int>(frame_count));
  const int prompt_token_count =
      static_cast<int>(prefix_token_ids.size() + suffix_token_ids.size()) +
      audio_token_count;
  const int available_token_count =
      config.max_context_token_count - prompt_token_count;
  if (available_token_count < 1) {
    throw std::invalid_argument(
        "audio prompt exceeds the Ming ASR context budget");
  } else {
  }
  return {std::move(samples),
          std::move(prefix_token_ids),
          std::move(suffix_token_ids),
          language,
          std::min(options.max_new_tokens, available_token_count),
          options.no_repeat_ngram_size};
}

qwen3_asr::TranscriptionResult
MingTranscriber::Transcribe(const MingRequest &request,
                            const std::atomic<bool> &cancel) const {
  if (cancel.load()) {
    throw qwen3_asr::TranscriptionCancelled();
  } else {
  }
  const auto &config = model_.config();
  const auto prefix_embeddings = model_.EmbedTokens(request.prefix_token_ids);
  const auto audio_embeddings = mx::astype(
      model_.EncodeAudio(request.samples, cancel), prefix_embeddings.dtype());
  const auto embeddings =
      mx::concatenate({prefix_embeddings, audio_embeddings,
                       model_.EmbedTokens(request.suffix_token_ids)},
                      1);
  auto cache = model_.NewCache();
  std::optional<mx::array> hidden_states;
  for (int start = 0; start < embeddings.shape(1); start += kPrefillChunkSize) {
    hidden_states = model_.Forward(
        mx::slice(embeddings, {0, start, 0},
                  {1, std::min(start + kPrefillChunkSize, embeddings.shape(1)),
                   config.hidden_size}),
        cache, cancel);
  }
  std::vector<int> generated_token_ids;
  qwen3_asr::TranscriptionResult result;
  for (int step = 0; step < request.max_new_tokens; ++step) {
    if (cancel.load()) {
      throw qwen3_asr::TranscriptionCancelled();
    } else {
    }
    auto next_token_logits =
        mx::reshape(model_.LastTokenLogits(hidden_states.value()), {-1});
    const size_t ngram_token_count =
        static_cast<size_t>(request.no_repeat_ngram_size);
    if (ngram_token_count > 0 &&
        generated_token_ids.size() >= ngram_token_count) {
      std::vector<int> banned_token_ids;
      for (size_t start = 0;
           start + ngram_token_count <= generated_token_ids.size(); ++start) {
        if (std::equal(generated_token_ids.begin() + start,
                       generated_token_ids.begin() + start + ngram_token_count -
                           1,
                       generated_token_ids.end() - (ngram_token_count - 1))) {
          banned_token_ids.push_back(
              generated_token_ids[start + ngram_token_count - 1]);
        } else {
        }
      }
      if (!banned_token_ids.empty()) {
        next_token_logits = mx::put_along_axis(
            next_token_logits,
            mx::array(banned_token_ids.data(),
                      {static_cast<int>(banned_token_ids.size())}),
            mx::array(-std::numeric_limits<float>::infinity()), 0);
      } else {
      }
    } else {
    }
    const int token_id = mx::argmax(next_token_logits).item<int>();
    if (std::find(stop_token_ids_.begin(), stop_token_ids_.end(), token_id) !=
        stop_token_ids_.end()) {
      result.finish_reason = qwen3_asr::FinishReason::kStop;
      break;
    } else {
      generated_token_ids.push_back(token_id);
    }
    if (step + 1 < request.max_new_tokens) {
      hidden_states =
          model_.Forward(model_.EmbedTokens({token_id}), cache, cancel);
    } else {
    }
  }
  result.text = omni_common::StripUnicodeWhitespace(
      tokenizer_.Decode(generated_token_ids, true));
  result.language = request.language;
  static const std::map<std::string, std::string> language_codes = {
      {"English", "en"},
      {"Chinese", "zh"},
      {"Canton", "yue"},
      {"Japanese", "ja"},
      {"Korean", "ko"}};
  const size_t separator = result.text.find('\t');
  if (separator != std::string::npos) {
    const std::string label =
        omni_common::StripUnicodeWhitespace(result.text.substr(0, separator));
    const auto found = language_codes.find(label);
    result.language = found == language_codes.end() ? label : found->second;
    result.text =
        omni_common::StripUnicodeWhitespace(result.text.substr(separator + 1));
  } else {
    for (const auto &[label, code] : language_codes) {
      if (result.text.starts_with(label)) {
        result.language = code;
        result.text = omni_common::StripUnicodeWhitespace(
            result.text.substr(label.size()));
        break;
      } else {
      }
    }
  }
  result.generated_token_count = static_cast<int>(generated_token_ids.size());
  return result;
}

} // namespace ming3_asr
