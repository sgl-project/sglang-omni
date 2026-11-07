// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "common/tokenizer.h"
#include "ming_tts/ming_tts_acoustic.h"
#include "ming_tts/ming_tts_model.h"
#include "ming_tts/ming_tts_reference.h"

namespace ming_tts {

struct Prompt {
  std::vector<int> input_ids;
  int speaker_position = -1;
  int reference_start = -1;
};

class MingFrontend {
public:
  explicit MingFrontend(const std::filesystem::path &directory);
  // The prompt of stripped, nonempty text and, when reference_tokens is
  // not 0, its reference's stripped, nonempty transcript.
  Prompt BuildPrompt(const std::string &text, const std::string &reference_text,
                     int reference_tokens) const;

private:
  omni_common::Tokenizer tokenizer_;
};

struct GenerationState {
  std::vector<qwen3_asr::KVCache> caches;
  mlx::core::array hidden, history;
  std::optional<mlx::core::array> feedback;
  int max_steps;
  int steps = 0;
  // Whether the stop head ended it, rather than max_steps.
  bool stopped = false;
  bool finished = false;
};

class MingGenerator {
public:
  explicit MingGenerator(const std::filesystem::path &directory);
  // Throws std::invalid_argument unless Start can run prompt for max_steps.
  void CheckLength(const Prompt &prompt, int max_steps) const;
  GenerationState
  Start(const Prompt &prompt, int max_steps,
        const std::optional<ReferenceConditioning> &reference) const;
  mlx::core::array Step(GenerationState &state, const mlx::core::array &noise,
                        const mlx::core::array &timesteps,
                        const mlx::core::array &sde_random, float cfg,
                        float sigma, float temperature) const;
  const MingAcoustic &acoustic() const { return acoustic_; }

private:
  MingBackbone backbone_;
  MingAcoustic acoustic_;
};
} // namespace ming_tts
