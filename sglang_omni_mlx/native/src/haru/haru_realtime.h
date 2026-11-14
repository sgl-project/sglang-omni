// SPDX-License-Identifier: Apache-2.0
// The Haru full-duplex realtime session: one socket runs one DuplexSession,
// feeding user frames and emitting audio and transcript deltas. The
// DuplexSession is created and stepped on the worker thread (MLX streams are
// per-thread), which the server constructs the session against.
#pragma once

#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "common/tokenizer.h"
#include "haru/haru_codec.h"
#include "haru/haru_head.h"
#include "haru/haru_model.h"
#include "haru/haru_session.h"
#include "nlohmann/json.hpp"
#include "qwen3_asr/realtime.h"
#include "qwen3_asr/worker.h"

namespace haru {

// One /v1/realtime socket's full-duplex conversation.
class HaruRealtimeSession : public qwen3_asr::RealtimeConnection {
public:
  HaruRealtimeSession(HaruModel &model, VibeVoiceCodec &codec,
                      MeanFlowHead &head,
                      const omni_common::Tokenizer &tokenizer,
                      const DuplexConfig &config,
                      qwen3_asr::TranscriptionWorker &worker,
                      std::string system_prompt, std::vector<float> reference,
                      Sender sender);

  bool Handle(const nlohmann::json &message) override;

private:
  // Runs fn on the worker thread, the thread that loaded the model.
  void RunOnWorker(std::function<void()> fn);
  std::vector<float> AppendSamples(const nlohmann::json &message);
  void EmitFrame(const Frame &frame);
  void FinishResponse(const std::string &reason);
  void SendEvent(const char *type,
                    const nlohmann::json &fields = nlohmann::json::object());

  HaruModel &model_;
  VibeVoiceCodec &codec_;
  MeanFlowHead &head_;
  const omni_common::Tokenizer &tokenizer_;
  DuplexConfig config_;
  qwen3_asr::TranscriptionWorker &worker_;
  std::string system_prompt_;
  std::vector<float> reference_;
  std::unique_ptr<DuplexSession> session_;
  std::optional<Frame> first_frame_;
  std::vector<float> pending_;
  std::string transcript_;
  int response_index_ = 0;
  bool input_ended_ = false;
  bool response_open_ = false;
  bool started_ = false;
};

} // namespace haru
