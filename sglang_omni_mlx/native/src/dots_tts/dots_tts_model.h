// SPDX-License-Identifier: Apache-2.0
// The dots.tts AR latent engine: the schedule-driven decode that ties the
// backbone, patch encoder, flow-matching DiT, AudioVAE and speaker x-vector
// into generate().
#pragma once

#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "dots_tts/dots_tts_acoustics.h"
#include "dots_tts/dots_tts_checkpoint.h"
#include "dots_tts/dots_tts_speaker.h"
#include "dots_tts/dots_tts_tokenizer.h"
#include "mlx/mlx.h"
#include "qwen3_asr/qwen3_decoder.h"

namespace dots_tts {

class DotsLLM {
public:
  explicit DotsLLM(const DotsCheckpoint &checkpoint);

  mlx::core::array EmbedTokens(const mlx::core::array &ids) const;
  mlx::core::array Forward(const mlx::core::array &embeddings,
                           std::vector<qwen3_asr::KVCache> &caches) const;
  std::vector<qwen3_asr::KVCache> NewCaches() const;
  mlx::core::array EosLogits(const mlx::core::array &hidden) const;

private:
  const DotsCheckpoint &checkpoint_;
  qwen3_asr::Qwen3Decoder decoder_;
};

struct SynthesisResult {
  std::vector<float> audio;
  int sample_rate = 48000;
  int num_patches = 0;
};

// Requests longer than 500 latent tokens are rejected at the boundary.
inline constexpr int kMaxGenerateLength = 500;

struct GenerationOptions {
  int num_steps = 10;
  float guidance_scale = 1.2f;
  float speaker_scale = 1.5f;
  std::optional<std::string> language;
  int seed = 42;
  int max_generate_length = kMaxGenerateLength;
  float eos_threshold = 0.8f;
  bool trim_onset = true;
};

class DotsTtsModel {
public:
  explicit DotsTtsModel(const std::filesystem::path &model_directory);

  SynthesisResult Generate(const std::string &text,
                           const std::optional<std::string> &prompt_wav,
                           const std::optional<std::string> &prompt_text,
                           const GenerationOptions &options) const;

  int sample_rate() const { return sample_rate_; }
  int patch_size() const { return patch_size_; }
  bool is_meanflow() const { return meanflow_; }

private:
  struct FlowState;

  struct PromptConditioning {
    std::optional<mlx::core::array> speaker_conditioning;
    std::optional<mlx::core::array> prompt_patches;
    std::optional<mlx::core::array> prompt_denorm_latents;
  };
  struct PromptConditioningResult {
    PromptConditioning conditioning;
    mlx::core::array next_rng_key;
  };
  PromptConditioningResult
  PreparePromptConditioning(const std::vector<float> &prompt_audio48k,
                            float speaker_scale, bool use_prompt_prefill,
                            mlx::core::array rng_key) const;

  mlx::core::array
  XvectorFromAudio48k(const std::vector<float> &audio48k) const;
  void AppendHiddenChunk(FlowState *state,
                         const mlx::core::array &hidden) const;
  void AppendHistoryChunk(FlowState *state,
                          const mlx::core::array &latent) const;
  std::vector<mlx::core::array> BuildFmAttnMask(int flow_sequence_length) const;
  int Prefill(const std::vector<int> &schedule_ids,
              const mlx::core::array &schedule, FlowState *state,
              const std::vector<int> &span_positions,
              const std::optional<mlx::core::array> &prompt_patches,
              const std::optional<mlx::core::array> &prompt_latents) const;
  mlx::core::array ReencodeAndStep(FlowState *state,
                                   const mlx::core::array &denorm_patch) const;
  bool ShouldStopAfterCurrentAudio(const mlx::core::array &llm_hiddens,
                                   float eos_threshold) const;

  std::unique_ptr<DotsCheckpoint> checkpoint_;
  std::unique_ptr<DotsTokenizer> tokenizer_;
  std::unique_ptr<DotsLLM> llm_;
  std::unique_ptr<AudioVAE> vae_;
  std::unique_ptr<CAMPPlus> speaker_;
  std::unique_ptr<KaldiFbank> fbank_;
  std::unique_ptr<DiT> dit_;
  std::unique_ptr<FlowSolver> flow_solver_;
  std::unique_ptr<VAESemanticEncoder> patch_encoder_;
  std::unique_ptr<nn::Linear> hidden_proj_;
  std::unique_ptr<nn::Linear> latent_proj_;
  std::unique_ptr<nn::Linear> xvec_linear_;
  mx::array xvec_norm_weight_{0.0f};
  mx::array xvec_norm_bias_{0.0f};
  std::unique_ptr<nn::Linear> coordinate_proj_;

  int patch_size_ = 4;
  int latent_dim_ = 128;
  int flow_hidden_size_ = 1024;
  int hop_size_ = 1920;
  int sample_rate_ = 48000;
  int max_sequence_length_ = 8192;
  bool meanflow_ = false;
};

} // namespace dots_tts
