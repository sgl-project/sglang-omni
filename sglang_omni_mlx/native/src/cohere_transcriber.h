// SPDX-License-Identifier: Apache-2.0
// Cohere Transcribe transcription as Voxt's Swift port runs it: the task
// prompt, greedy decoding, energy-cut chunks for long audio and the
// SentencePiece decoding of the generated ids.
#pragma once

#include <atomic>
#include <filesystem>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <vector>

#include "cohere_model.h"
#include "speech_segments.h"
#include "swift_port.h"
#include "transcriber.h"

namespace cohere_transcribe {

struct CohereOptions {
  // An ISO code or an English name; unknown is English.
  std::string language = "en";
  // Punctuation and casing.
  bool use_punctuation = true;
  // One budget shared by every chunk; zero is the decoder's context.
  int max_new_tokens = 0;
  // Zero decodes greedily.
  float temperature = 0.0f;
  // The energy-cut chunk length for long audio and the shortest chunk.
  float chunk_duration_seconds = 1200.0f;
  float min_chunk_duration_seconds = 1.0f;
  // Long audio cut at speech with this Silero VAD instead of at the quietest
  // point, as Voxt's voice-activity strategy does; no speech is no text.
  const silero_vad::SileroVAD *voice_activity_detector = nullptr;
  silero_vad::SpeechSegmentConfig speech_segments;
};

class CohereTranscriber {
public:
  // Reads the checkpoint, tokenizer.model and tokenizer_config.json.
  explicit CohereTranscriber(const std::filesystem::path &model_directory);

  qwen3_asr::TranscriptionResult
  Transcribe(const std::vector<float> &samples, const CohereOptions &options,
             const std::atomic<bool> &cancel) const;

private:
  struct ChunkResult {
    std::string text;
    int generated_token_count = 0;
    bool reached_token_limit = false;
  };

  // Text of generated ids: special ids dropped, SentencePiece pieces joined.
  std::string DecodeText(const std::vector<int> &token_ids) const;
  ChunkResult TranscribeChunk(const std::vector<float> &samples,
                              const std::vector<int> &prompt_ids,
                              int max_new_tokens, float temperature,
                              const std::atomic<bool> &cancel) const;

  CohereModel model_;
  swift_port::SentencePieceVocabulary vocabulary_;
  std::map<std::string, int> special_token_ids_;
  // Added tokens, which decoding drops beside control and unused pieces.
  std::set<int> special_ids_;
  int end_of_text_id_ = 0;
};

} // namespace cohere_transcribe
