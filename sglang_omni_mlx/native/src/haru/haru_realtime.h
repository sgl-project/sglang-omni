// SPDX-License-Identifier: Apache-2.0
// The Haru full-duplex realtime session: one socket runs one DuplexSession,
// feeding user frames and emitting audio and transcript deltas. The
// DuplexSession is created and stepped on the worker thread (MLX streams are
// per-thread), which the server constructs the session against.
#pragma once

#include <array>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "common/realtime_connection.h"
#include "common/tokenizer.h"
#include "common/worker.h"
#include "haru/haru_model.h"
#include "haru/haru_session.h"
#include "nlohmann/json.hpp"

namespace haru {

// The capabilities object reported in session.updated and, as JSON, by the
// /v1/realtime/capabilities endpoint. input_sample_rate is the rate clients
// send audio at.
nlohmann::json Capabilities(int input_sample_rate, int output_sample_rate);

// Converts the client's 16 kHz audio to the 24 kHz the codec consumes. Filter
// state is kept across appends so the input is filtered as one signal, and the
// sample count follows the input exactly (3 outputs per 2 inputs) so unit
// receipts line up with what the client sent.
class StreamResampler {
public:
  StreamResampler(int input_rate, int output_rate);

  std::vector<float> Process(const std::vector<float> &input);
  // Emits the samples still inside the filter window once the input ended.
  std::vector<float> Flush();

private:
  std::vector<float> Convert();

  int input_rate_;
  int output_rate_;
  // Per output phase, one weight per input offset around its sample.
  std::array<std::vector<double>, 3> phase_taps_;
  std::vector<float> buffer_;
  int64_t origin_ = 0;
  int64_t input_count_ = 0;
  int64_t output_count_ = 0;
};

// One /v1/realtime socket's full-duplex conversation.
class HaruRealtimeSession : public omni_server::RealtimeConnection {
public:
  HaruRealtimeSession(const HaruModel &model,
                      const omni_common::Tokenizer &tokenizer,
                      const DuplexConfig &config,
                      omni_server::TranscriptionWorker &worker,
                      std::string system_prompt, std::vector<float> reference,
                      int input_sample_rate, Sender sender);

  bool Handle(const nlohmann::json &message) override;

private:
  // Runs fn on the worker thread, the thread that loaded the model.
  void RunOnWorker(std::function<void()> fn);
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

  const HaruModel &model_;
  const omni_common::Tokenizer &tokenizer_;
  DuplexConfig config_;
  omni_server::TranscriptionWorker &worker_;
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
  bool finished_ = false; // The session reached its frame limit.
  int input_sample_rate_;
  int model_sample_rate_;
  int frame_samples_;
  StreamResampler resampler_;
  int64_t accepted_input_samples_ = 0;
  int64_t stepped_frames_ = 0;
  int64_t unit_count_ = 0;
  std::vector<double> pending_units_;
};

} // namespace haru
