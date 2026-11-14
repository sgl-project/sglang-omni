// SPDX-License-Identifier: Apache-2.0
// A live full-duplex conversation: one user frame (3200 samples) in, one model
// frame out.
#pragma once

#include <string>
#include <vector>

#include "common/tokenizer.h"
#include "haru/haru_codec.h"
#include "haru/haru_config.h"
#include "haru/haru_head.h"
#include "haru/haru_model.h"
#include "mlx/mlx.h"

namespace haru {

namespace mx = mlx::core;

struct DuplexConfig {
  float guidance_scale = 1.5f;
  float temperature = 0.7f;
  int top_k = 25;
  float top_p = 1.0f;
  float eop_bias = 0.0f;
  bool speak_first = false;
  int max_frames = 27000;
  bool dry = true;
  int seed = -1;  // -1 draws a fresh seed
};

struct Frame {
  std::vector<float> audio;  // float32 [3200] at 24 kHz
  int token = 0;
  std::string text;
  int index = 0;
  bool done = false;
};

class DuplexSession {
public:
  DuplexSession(HaruModel &model, VibeVoiceCodec &codec, MeanFlowHead &head,
                const omni_common::Tokenizer &tokenizer,
                const DuplexConfig &config);

  // Runs the prompt (system card + voice reference) and returns model frame 0.
  Frame Start(const std::string &system_prompt,
              const std::vector<float> &reference);
  // Consumes one user frame (3200 samples) and returns the next model frame.
  Frame Step(const std::vector<float> &user_frame);

private:
  Frame Emit(const mx::array &hidden);
  mx::array PreparedLogits(const mx::array &hidden);
  mx::array SampleToken(const mx::array &logits);
  std::string Detokenize(int token);
  // A fresh random key: split the session key, keep the first half and use the
  // second. The session never touches MLX's global random state.
  mx::array NextKey();

  HaruModel &model_;
  VibeVoiceCodec &codec_;
  MeanFlowHead &head_;
  const omni_common::Tokenizer &tokenizer_;
  DuplexConfig config_;

  int pad_id_, eop_id_, eos_id_;
  mx::array silent_ids_;  // [pad, eop]

  std::vector<LayerCache> caches_;
  VibeVoiceCodec::State encoder_state_;
  VibeVoiceCodec::State decoder_state_;
  mx::array token_, latent_;
  mx::array key_;
  int index_ = 0;
  int force_text_ = 0;
  std::vector<int> history_;       // DRY text-stream history
  std::vector<int> pending_ids_;   // detokenizer's pending multi-byte
  bool in_gap_ = false;
};

} // namespace haru
