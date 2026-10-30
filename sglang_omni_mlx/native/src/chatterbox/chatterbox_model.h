// SPDX-License-Identifier: Apache-2.0
// Chatterbox-Turbo: T3 samples S3 speech tokens for text in a voice, the flow
// turns them into a mel and HiFT into 24 kHz audio. A voice is the
// checkpoint's built-in one or is cloned from a reference recording.
#pragma once

#include <atomic>
#include <filesystem>
#include <functional>
#include <string>
#include <string_view>
#include <vector>

#include "chatterbox/chatterbox_flow.h"
#include "chatterbox/chatterbox_speaker.h"
#include "chatterbox/chatterbox_speech_tokenizer.h"
#include "chatterbox/chatterbox_t3.h"
#include "chatterbox/chatterbox_vocoder.h"
#include "chatterbox/chatterbox_voice_encoder.h"
#include "common/layers.h"
#include "common/tokenizer.h"
#include "mlx/mlx.h"

namespace chatterbox {

inline constexpr int kOutputSampleRate = 24000;
// The most text tokens T3 reads, the text tokenizer's model_max_length.
inline constexpr int kMaxTextTokens = 1024;

struct Voice {
  T3Voice t3;
  FlowPrompt flow;
};

// Chatterbox's clean-up of text before tokenizing: first letter capitalized,
// whitespace runs collapsed, uncommon punctuation replaced and a full stop
// added when the text ends without one.
std::string NormalizePunctuation(const std::string &text);

class ChatterboxModel {
public:
  // Reads model.safetensors, conds.safetensors (the built-in voice), the text
  // tokenizer, and the S3 speech tokenizer in speech_tokenizer/.
  explicit ChatterboxModel(const std::filesystem::path &model_directory);

  const Voice &builtin_voice() const { return builtin_voice_; }
  // Throws std::invalid_argument unless wav_bytes is a WAV file whose
  // header gives more than 5 s and at most 60 s of audio; reads no samples.
  static void CheckReference(std::string_view wav_bytes);
  // A voice cloned from a recording CheckReference accepts.
  Voice CloneVoice(std::string_view wav_bytes) const;
  // T3's text ids for text after NormalizePunctuation; throws
  // std::invalid_argument past kMaxTextTokens.
  std::vector<int> EncodeText(const std::string &text) const;

  qwen3_asr::FinishReason
  GenerateSpeechTokens(const Voice &voice, const std::vector<int> &text_ids,
                       const SamplingOptions &options, int max_tokens,
                       const std::atomic<bool> &cancel,
                       const std::function<void(int)> &on_token) const {
    return t3_.Generate(voice.t3, text_ids, options, max_tokens, cancel,
                        on_token);
  }
  // 24 kHz samples for T3's speech tokens in voice.
  std::vector<float> Synthesize(const std::vector<int> &speech_tokens,
                                const Voice &voice) const;

private:
  omni_common::Checkpoint checkpoint_;
  omni_common::Checkpoint speech_tokenizer_checkpoint_;
  omni_common::Tokenizer tokenizer_;
  T3 t3_;
  FlowDecoder flow_;
  Vocoder vocoder_;
  VoiceEncoder voice_encoder_;
  SpeechTokenizer speech_tokenizer_;
  SpeakerEncoder speaker_encoder_;
  Voice builtin_voice_;
  // The fade-in over the first samples, which hides the vocoder's start-up.
  mlx::core::array fade_in_;
};

} // namespace chatterbox
