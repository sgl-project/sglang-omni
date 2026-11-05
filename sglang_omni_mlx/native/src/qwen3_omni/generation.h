// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "qwen3_omni/preprocessing.h"
#include "qwen3_omni/vocoder.h"

namespace qwen3_omni {
struct GenerationChunk {
  std::string text;
  std::vector<uint8_t> audio;
  std::string finish_reason;
  int prompt_tokens = 0;
  int completion_tokens = 0;
};
using ChunkEmitter = std::function<void(const GenerationChunk &)>;
mx::array sample(const mx::array &input, const std::vector<int> &history,
                 const Sampling &options);
struct TextGeneration {
  std::vector<int> tokens;
  int completion_tokens;
  bool stopped;
};
TextGeneration generate_text_tokens(
    mx::array logits, int maximum, const std::set<int> &eos_tokens,
    const Sampling &options, const Cancellation &cancelled,
    const std::function<std::vector<mx::array>(const mx::array &)> &advance,
    const std::function<void(const std::vector<int> &)> &consume);

class OmniModel {
public:
  explicit OmniModel(const std::filesystem::path &directory);
  void generate(const ChatRequest &request, const ChunkEmitter &emit,
                const Cancellation &cancelled);

private:
  mx::array projection(const std::string &name, const mx::array &input) const;
  void generate_speech(const ChatRequest &request,
                       const PreparedInput &prepared,
                       const mx::array &prompt_embeddings,
                       const mx::array &prompt_hidden,
                       const std::vector<int> &tokens, const ChunkEmitter &emit,
                       const Cancellation &cancelled);
  Json configuration;
  omni_common::Tokenizer tokenizer;
  Weights weights;
  TextModel thinker;
  TextModel talker;
  TextModel predictor;
  VisionEncoder vision;
  AudioEncoder audio;
  Vocoder vocoder;
};
struct Completion {
  std::string id;
  int64_t created;
  std::string model;
  std::string text;
  std::vector<uint8_t> audio_pcm;
  std::string finish_reason;
  int prompt_tokens = 0;
  int completion_tokens = 0;
  void append(const GenerationChunk &chunk);
  Json json() const;
};
} // namespace qwen3_omni
