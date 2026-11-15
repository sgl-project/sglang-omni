// SPDX-License-Identifier: Apache-2.0
// The Haru full-duplex realtime session: one socket runs one DuplexSession,
// feeding user frames and emitting audio and transcript deltas. The
// DuplexSession is created and stepped on the worker thread (MLX streams are
// per-thread), which the server constructs the session against.
#pragma once

#include <cstdint>
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

// The capabilities object reported in session.updated and, as JSON, by the
// /v1/realtime/capabilities endpoint. input_sample_rate is the rate clients
// send audio at.
nlohmann::json Capabilities(int input_sample_rate);

// Converts the client's 16 kHz audio to the 24 kHz the codec consumes. Filter
// state is kept across appends so the input is filtered as one signal, and the
// sample count follows the input exactly (3 outputs per 2 inputs) so unit
// receipts line up with what the client sent.
class StreamResampler {
public:
  explicit StreamResampler(int input_rate);

  std::vector<float> Process(const std::vector<float> &input);
  // Emits the samples still inside the filter window once the input ended.
  std::vector<float> Flush();

private:
  std::vector<float> Convert();

  int input_rate_;
  std::vector<float> buffer_;
  int64_t origin_ = 0;
  int64_t input_count_ = 0;
  int64_t output_count_ = 0;
};

// One /v1/realtime socket's full-duplex conversation.
class HaruRealtimeSession : public qwen3_asr::RealtimeConnection {
public:
  HaruRealtimeSession(HaruModel &model, VibeVoiceCodec &codec,
                      MeanFlowHead &head,
                      const omni_common::Tokenizer &tokenizer,
                      const DuplexConfig &config,
                      qwen3_asr::TranscriptionWorker &worker,
                      std::string system_prompt, std::vector<float> reference,
                      int input_sample_rate, Sender sender);

  bool Handle(const nlohmann::json &message) override;

private:
  // Runs fn on the worker thread, the thread that loaded the model.
  void RunOnWorker(std::function<void()> fn);
  // Applies sglang.sampling from a session.update to this session's config.
  void ApplySampling(const nlohmann::json &sampling);
  std::vector<float> AppendSamples(const nlohmann::json &message);
  // Steps one 24 kHz frame and records the unit receipt it completes.
  void StepFrame(const std::vector<float> &frame, bool is_padding,
                 std::vector<Frame> &frames);
  void SendUnitDone(double duration_ms);
  void EmitPendingUnits();
  double AcceptedEndMs() const;
  double PaddingMs() const;
  void EmitFrame(const Frame &frame);
  void FinishResponse(const std::string &status, const std::string &reason);
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
  int input_sample_rate_;
  StreamResampler resampler_;
  int64_t accepted_input_samples_ = 0;
  int64_t stepped_frames_ = 0;
  int64_t unit_count_ = 0;
  std::vector<double> pending_units_;
};

} // namespace haru
