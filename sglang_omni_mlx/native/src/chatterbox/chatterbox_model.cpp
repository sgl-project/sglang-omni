// SPDX-License-Identifier: Apache-2.0
// Chatterbox-Turbo's voices, text clean-up and token-to-audio path.
#include "chatterbox/chatterbox_model.h"

#include <CoreFoundation/CoreFoundation.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
#include <utility>

#include "chatterbox/chatterbox_audio.h"
#include "common/text.h"
#include "qwen3_asr/qwen3_asr_transcriber.h"

namespace chatterbox {

namespace mx = mlx::core;

namespace {

// The voice encoder and the S3 speech tokenizer read 16 kHz audio.
constexpr int kTokenizerSampleRate = 16000;
constexpr double kReferenceTargetLufs = -27.0;
constexpr double kMinReferenceSeconds = 5.0;
// Only the first 15 s are read, but loudness is measured over the whole
// recording, so a longer one is refused rather than cut.
constexpr double kMaxReferenceSeconds = 60.0;
// T3's prompt reads the first 15 s of a reference, the flow's the first 10 s.
constexpr int kT3ReferenceSeconds = 15;
constexpr int kFlowReferenceSeconds = 10;
// The speech token the flow reads as silence, three of which end every
// utterance.
constexpr int kSilenceToken = 4299;
constexpr int kTrailingSilenceTokens = 3;
// 20 ms of silence, then a 20 ms raised-cosine fade-in.
constexpr int kFadeSamples = kOutputSampleRate / 50;

// Keys T3 and S3Gen never read: T3's text head and token embeddings (T3
// embeds every input itself) and BatchNorm's batch counters.
bool IsUnusedWeight(const std::string &name) {
  return name == "t3.text_head.weight" || name == "t3.tfmr.wte.weight" ||
         name.ends_with(".num_batches_tracked");
}

// The weights of a checkpoint directory whose config.json names model_type,
// so another checkpoint fails at load rather than at its first request.
omni_common::WeightMap LoadWeights(const std::filesystem::path &directory,
                                   const std::string &model_type) {
  const nlohmann::json config =
      omni_common::ReadJson(directory / "config.json");
  if (config.value("model_type", std::string()) != model_type) {
    throw std::runtime_error(directory.string() + " is not a " + model_type +
                             " checkpoint");
  } else {
  }
  omni_common::WeightMap weights;
  for (auto &[name, array] :
       mx::load_safetensors((directory / "model.safetensors").string()).first) {
    if (!IsUnusedWeight(name)) {
      weights.emplace(name, std::move(array));
    } else {
    }
  }
  return weights;
}

std::string ReplaceAll(std::string text, std::string_view from,
                       std::string_view to) {
  for (size_t position = text.find(from); position != std::string::npos;
       position = text.find(from, position + to.size())) {
    text.replace(position, from.size(), to);
  }
  return text;
}

std::string CaseMapped(const std::string &text, bool upper) {
  CFStringRef source = CFStringCreateWithBytes(
      kCFAllocatorDefault, reinterpret_cast<const UInt8 *>(text.data()),
      static_cast<CFIndex>(text.size()), kCFStringEncodingUTF8, false);
  if (source == nullptr) {
    return text;
  } else {
  }
  CFMutableStringRef mapped =
      CFStringCreateMutableCopy(kCFAllocatorDefault, 0, source);
  CFRelease(source);
  if (upper) {
    CFStringUppercase(mapped, nullptr);
  } else {
    CFStringLowercase(mapped, nullptr);
  }
  const CFIndex length = CFStringGetLength(mapped);
  CFIndex byte_count = 0;
  CFStringGetBytes(mapped, CFRangeMake(0, length), kCFStringEncodingUTF8, 0,
                   false, nullptr, 0, &byte_count);
  std::string bytes(static_cast<size_t>(byte_count), '\0');
  CFStringGetBytes(mapped, CFRangeMake(0, length), kCFStringEncodingUTF8, 0,
                   false, reinterpret_cast<UInt8 *>(bytes.data()), byte_count,
                   nullptr);
  CFRelease(mapped);
  return bytes;
}

size_t Utf8Length(unsigned char lead) {
  return lead >= 0xF0 ? 4 : lead >= 0xE0 ? 3 : lead >= 0xC0 ? 2 : 1;
}

mx::array FadeIn() {
  std::vector<float> fade(2 * kFadeSamples, 0.0f);
  for (int index = 0; index < kFadeSamples; ++index) {
    // Note (Jiaxin Deng): the angle steps as an evenly spaced float64 grid
    // from pi to 0 does, ending exactly at 0.
    const double angle =
        index == kFadeSamples - 1
            ? 0.0
            : M_PI + index * (-M_PI / static_cast<double>(kFadeSamples - 1));
    fade[kFadeSamples + index] =
        static_cast<float>((std::cos(angle) + 1.0) / 2.0);
  }
  return mx::array(fade.begin(), {2 * kFadeSamples}, mx::float32);
}

std::vector<float> Prefix(const std::vector<float> &samples,
                          size_t sample_count) {
  return std::vector<float>(samples.begin(),
                            samples.begin() +
                                std::min(sample_count, samples.size()));
}

} // namespace

std::string NormalizePunctuation(const std::string &text) {
  if (text.empty()) {
    return "You need to add some text for me to talk.";
  } else {
  }
  std::string normalized = text;
  const size_t first_length =
      std::min(Utf8Length(static_cast<unsigned char>(text[0])), text.size());
  const std::string first = text.substr(0, first_length);
  const std::string upper = CaseMapped(first, true);
  // A lowercase letter is one that uppercasing changes and lowercasing keeps.
  if (upper != first && CaseMapped(first, false) == first) {
    normalized = upper + text.substr(first_length);
  } else {
  }

  std::string collapsed;
  bool in_word = false;
  for (size_t byte = 0; byte < normalized.size();) {
    const size_t length =
        std::min(Utf8Length(static_cast<unsigned char>(normalized[byte])),
                 normalized.size() - byte);
    const std::vector<uint32_t> code_point =
        omni_common::CodePoints(normalized.substr(byte, length));
    if (omni_common::IsUnicodeSpace(code_point.front())) {
      in_word = false;
    } else {
      if (!in_word && !collapsed.empty()) {
        collapsed.push_back(' ');
      } else {
      }
      in_word = true;
      collapsed.append(normalized, byte, length);
    }
    byte += length;
  }

  static constexpr std::array<std::pair<std::string_view, std::string_view>, 9>
      kReplacements = {{{"…", ", "},
                        {":", ","},
                        {"—", "-"},
                        {"–", "-"},
                        {" ,", ","},
                        {"“", "\""},
                        {"”", "\""},
                        {"‘", "'"},
                        {"’", "'"}}};
  for (const auto &[from, to] : kReplacements) {
    collapsed = ReplaceAll(std::move(collapsed), from, to);
  }
  while (!collapsed.empty() && collapsed.back() == ' ') {
    collapsed.pop_back();
  }
  if (collapsed.empty() || std::string_view(".!?-,").find(collapsed.back()) ==
                               std::string_view::npos) {
    collapsed.push_back('.');
  } else {
  }
  return collapsed;
}

ChatterboxModel::ChatterboxModel(const std::filesystem::path &model_directory)
    : checkpoint_(LoadWeights(model_directory, "chatterbox_turbo"),
                  nlohmann::json::object()),
      speech_tokenizer_checkpoint_(
          LoadWeights(model_directory / "speech_tokenizer", "s3_tokenizer_v2"),
          nlohmann::json::object()),
      tokenizer_(model_directory, omni_common::SplitPattern::kGpt2),
      t3_(checkpoint_), flow_(checkpoint_), vocoder_(checkpoint_),
      voice_encoder_(checkpoint_),
      speech_tokenizer_(speech_tokenizer_checkpoint_),
      speaker_encoder_(checkpoint_, "s3gen.speaker_encoder."),
      builtin_voice_([&] {
        const auto conditionals =
            mx::load_safetensors(
                (model_directory / "conds.safetensors").string())
                .first;
        return Voice{{conditionals.at("t3.speaker_emb"),
                      conditionals.at("t3.cond_prompt_speech_tokens")},
                     {conditionals.at("gen.prompt_token"),
                      conditionals.at("gen.prompt_feat"),
                      conditionals.at("gen.embedding")}};
      }()),
      fade_in_(FadeIn()) {
  mx::eval({builtin_voice_.t3.speaker_embedding,
            builtin_voice_.t3.prompt_speech_tokens,
            builtin_voice_.flow.prompt_tokens, builtin_voice_.flow.prompt_mel,
            builtin_voice_.flow.speaker_embedding, fade_in_});
}

void ChatterboxModel::CheckReference(std::string_view wav_bytes) {
  const double seconds = WavDurationSeconds(wav_bytes, kOutputSampleRate);
  if (!(seconds > kMinReferenceSeconds)) {
    throw std::invalid_argument(
        "reference audio must be longer than 5 seconds");
  } else if (seconds > kMaxReferenceSeconds) {
    throw std::invalid_argument("reference audio must be at most 60 seconds");
  } else {
  }
}

Voice ChatterboxModel::CloneVoice(std::string_view wav_bytes) const {
  const std::vector<float> reference =
      NormalizeLoudness(ReadAudio(wav_bytes, kOutputSampleRate),
                        kOutputSampleRate, kReferenceTargetLufs);
  const std::vector<float> reference_16k =
      Resample(reference, kOutputSampleRate, kTokenizerSampleRate);
  const std::vector<float> flow_reference =
      Prefix(reference, kFlowReferenceSeconds * kOutputSampleRate);
  const std::vector<float> t3_reference_16k =
      Prefix(reference_16k, kT3ReferenceSeconds * kTokenizerSampleRate);

  mx::array prompt_tokens = speech_tokenizer_.Encode(
      Prefix(reference_16k, kFlowReferenceSeconds * kTokenizerSampleRate));
  mx::array prompt_mel = S3GenMel(flow_reference);
  // Two mel frames per token: the longer of the two is cut to the other.
  const int token_count = prompt_tokens.shape(1);
  const int paired_token_count = prompt_mel.shape(1) / 2;
  if (token_count < paired_token_count) {
    prompt_mel = mx::slice(prompt_mel, {0, 0, 0},
                           {1, 2 * token_count, prompt_mel.shape(2)});
  } else if (token_count > paired_token_count) {
    prompt_tokens = mx::slice(prompt_tokens, {0, 0}, {1, paired_token_count});
  } else {
  }
  const mx::array speaker_embedding = speaker_encoder_.Embed(
      Resample(flow_reference, kOutputSampleRate, kTokenizerSampleRate));

  const mx::array t3_tokens = speech_tokenizer_.Encode(t3_reference_16k);
  Voice voice{
      {voice_encoder_.Embed(t3_reference_16k),
       mx::slice(t3_tokens, {0, 0},
                 {1, std::min(t3_tokens.shape(1), kPromptSpeechTokenLimit)})},
      {prompt_tokens, prompt_mel, speaker_embedding}};
  mx::eval({voice.t3.speaker_embedding, voice.t3.prompt_speech_tokens,
            voice.flow.prompt_tokens, voice.flow.prompt_mel,
            voice.flow.speaker_embedding});
  return voice;
}

std::vector<int> ChatterboxModel::EncodeText(const std::string &text) const {
  std::vector<int> ids = tokenizer_.Encode(NormalizePunctuation(text));
  if (ids.size() > static_cast<size_t>(kMaxTextTokens)) {
    throw std::invalid_argument(
        "input is " + std::to_string(ids.size()) + " text tokens; at most " +
        std::to_string(kMaxTextTokens) + " are supported");
  } else {
  }
  return ids;
}

std::vector<float>
ChatterboxModel::Synthesize(const std::vector<int> &speech_tokens,
                            const Voice &voice) const {
  std::vector<int32_t> tokens;
  for (const int token : speech_tokens) {
    if (token < kSpeechTokenCount) {
      tokens.push_back(token);
    } else {
    }
  }
  tokens.insert(tokens.end(), kTrailingSilenceTokens, kSilenceToken);
  const mx::array mel =
      flow_.Mel(mx::array(tokens.begin(), {1, static_cast<int>(tokens.size())},
                          mx::int32),
                voice.flow);
  mx::array waveform = vocoder_.Waveform(mel);
  const int sample_count = waveform.shape(1);
  if (sample_count >= fade_in_.shape(0)) {
    waveform = mx::concatenate(
        {mx::multiply(mx::slice(waveform, {0, 0}, {1, fade_in_.shape(0)}),
                      fade_in_),
         mx::slice(waveform, {0, fade_in_.shape(0)}, {1, sample_count})},
        1);
  } else {
  }
  waveform = mx::astype(mx::reshape(waveform, {-1}), mx::float32);
  mx::eval(waveform);
  return std::vector<float>(waveform.data<float>(),
                            waveform.data<float>() + waveform.size());
}

} // namespace chatterbox
