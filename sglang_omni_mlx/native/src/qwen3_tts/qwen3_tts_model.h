// SPDX-License-Identifier: Apache-2.0
// Qwen3-TTS CustomVoice: a Qwen3 talker that predicts each frame's first
// codebook from the text and preset speaker, a small Qwen3 code predictor
// for the other codebooks, and the speech decoder that turns frames into
// samples.
#pragma once

#include <atomic>
#include <filesystem>
#include <functional>
#include <map>
#include <optional>
#include <string>
#include <vector>

#include "common/layers.h"
#include "common/tokenizer.h"
#include "mlx/mlx.h"
#include "qwen3_asr/qwen3_asr_transcriber.h"
#include "qwen3_asr/qwen3_decoder.h"
#include "qwen3_tts/qwen3_tts_codec.h"

namespace qwen3_tts {

// One sampler's settings; temperature 0 or below is greedy.
struct SamplerConfig {
  float temperature = 0.9f;
  int top_k = 50;
  float top_p = 1.0f;
};

struct SamplingConfig {
  SamplerConfig talker;
  SamplerConfig code_predictor;
  // On the talker's first codes, over the last repetition_window_frames
  // frames, or every frame when 0.
  float repetition_penalty = 1.05f;
  int repetition_window_frames = 0;
  // The draws' key sequence starts here; unseeded requests draw a seed.
  std::optional<uint64_t> seed;
};

// The talker's prompt: chat-template token ids around the text and around
// the instructions, if any, the preset speaker's codec id, and the
// language's codec id, if any.
struct SpeechPrompt {
  std::vector<int> text_ids;
  std::vector<int> instruction_ids;
  int speaker_id = 0;
  std::optional<int> language_id;
  // Note (Jiaxin Deng): the CUDA server's CustomVoice prompt holds the whole
  // text before the first frame; the other layout feeds one text token per
  // frame, as the upstream streaming mode and mlx-audio do.
  bool text_up_front = true;
};

class Qwen3TtsModel {
public:
  explicit Qwen3TtsModel(const std::filesystem::path &model_directory);

  const SpeechDecoder &speech_decoder() const { return speech_decoder_; }
  // Preset speakers by lower-case name, and their names in checkpoint order.
  const std::map<std::string, int> &speaker_ids() const { return speaker_ids_; }
  const std::vector<std::string> &speaker_names() const {
    return speaker_names_;
  }
  // Languages by lower-case name; "auto" leaves the language to the model.
  const std::map<std::string, int> &language_ids() const {
    return language_ids_;
  }

  // The prompt for text in a preset speaker's voice; speaker and language are
  // case-insensitive, and empty instructions add none. Throws
  // std::invalid_argument for an unknown speaker or language.
  SpeechPrompt BuildPrompt(const std::string &text, const std::string &speaker,
                           const std::string &language,
                           const std::string &instructions) const;
  // Every id prompt embeds: the instructions' and text's token ids around the
  // codec's thinking, language, speaker, pad and start ids.
  std::vector<int> PromptIds(const SpeechPrompt &prompt) const;
  // The talker's prompt positions for prompt, before the first frame.
  int PromptLength(const SpeechPrompt &prompt) const;
  // Generates up to max_frames frames, passing each frame's evaluated int32
  // codes [codebooks] to on_frame. Stops at the end-of-speech code; throws
  // TranscriptionCancelled once cancel is set.
  qwen3_asr::FinishReason
  Generate(const SpeechPrompt &prompt, const SamplingConfig &sampling,
           int max_frames, const std::atomic<bool> &cancel,
           const std::function<void(const mlx::core::array &)> &on_frame) const;

private:
  Qwen3TtsModel(const std::filesystem::path &model_directory,
                const nlohmann::json &config);

  // Text token ids [1, length] to talker embeddings [1, length, hidden].
  mlx::core::array EmbedText(const std::vector<int> &ids) const;
  mlx::core::array EmbedCodec(const std::vector<int> &ids) const;
  // A frame's codes [1, codebooks] from the talker's normed hidden state
  // [1, 1, hidden] and its first code [1, 1].
  mlx::core::array PredictCodes(const mlx::core::array &hidden,
                                const mlx::core::array &first_code,
                                const SamplerConfig &sampler,
                                mlx::core::random::KeySequence &keys) const;
  // The talker's next input: the sum of a frame's codebook embeddings.
  mlx::core::array EmbedFrame(const mlx::core::array &codes) const;
  // The codec ids before the text: thinking, language, speaker, pad, start.
  std::vector<int> CodecPrefixIds(const SpeechPrompt &prompt) const;

  omni_common::Checkpoint checkpoint_;
  omni_common::Tokenizer tokenizer_;
  SpeechDecoder speech_decoder_;
  qwen3_asr::Qwen3Decoder talker_;
  qwen3_asr::Qwen3Decoder code_predictor_;
  float talker_norm_eps_;
  float code_predictor_norm_eps_;
  int codebook_count_;
  int codec_vocabulary_size_;
  int tts_bos_token_id_;
  int tts_eos_token_id_;
  int tts_pad_token_id_;
  int codec_bos_id_;
  int codec_eos_id_;
  int codec_pad_id_;
  int codec_think_id_;
  int codec_nothink_id_;
  int codec_think_bos_id_;
  int codec_think_eos_id_;
  std::map<std::string, int> speaker_ids_;
  std::vector<std::string> speaker_names_;
  std::map<std::string, std::string> speaker_dialects_;
  std::map<std::string, int> language_ids_;
};

} // namespace qwen3_tts
