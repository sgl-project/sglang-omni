// SPDX-License-Identifier: Apache-2.0
#include "haru/haru_realtime.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>

#include "civetweb.h"
#include "common/audio.h"
#include "common/realtime_connection.h"
#include "common/text.h"
#include "haru/haru_audio.h"

namespace haru {

namespace {

constexpr int kNativeUnitMs = 400; // three 3200-sample frames
constexpr int kNativeUnitFrames = 3;
// Modified Bessel function of the first kind, order zero, from its power
// series in Abramowitz & Stegun 9.8.1:
// I0(x) = sum_k (x^2/4)^k / (k!)^2,  term_{k+1} = term_k * (x/2k)^2.
// The Kaiser window needs it and libc++ does not ship std::cyl_bessel_i.
double BesselI0(double x) {
  double term = 1.0;
  double sum = 1.0;
  for (int order = 1; order < 32; ++order) {
    const double half = x / (2.0 * order);
    term *= half * half;
    sum += term;
    if (term < sum * 1e-17) {
      break;
    }
  }
  return sum;
}

// Half the length of the 16 kHz resampling filter, in input samples. Ten taps
// per side, the width scipy.signal.resample_poly designs for a 3:2 ratio.
constexpr int kResampleHalfTaps = 10;
constexpr double kResampleKaiserBeta = 5.0;
const double kBesselI0Beta = BesselI0(kResampleKaiserBeta);

// Windowed sinc for the 3:2 upsampler. The kernel keeps the amplitude of the
// input: a constant input comes out with the same constant, which is what the
// reference resampler does and what the model was tuned on.
double ResampleKernel(double t) {
  if (std::abs(t) >= kResampleHalfTaps) {
    return 0.0;
  }
  const double sinc = t == 0.0 ? 1.0 : std::sin(M_PI * t) / (M_PI * t);
  const double ratio = t / static_cast<double>(kResampleHalfTaps);
  const double window =
      BesselI0(kResampleKaiserBeta *
               std::sqrt(std::max(0.0, 1.0 - ratio * ratio))) /
      kBesselI0Beta;
  return sinc * window;
}

} // namespace

nlohmann::json Capabilities(int input_sample_rate, int output_sample_rate) {
  return {
      {"interaction", "native"},
      {"native_full_duplex", true},
      {"proactive_output", false},
      {"turn_control", {nullptr}},
      {"client_commit", false},
      {"input_modalities", {"audio"}},
      {"output_modalities", {"audio"}},
      {"input_audio_format",
       {{"type", "audio/pcm"}, {"rate", input_sample_rate}}},
      {"output_audio_format",
       {{"type", "audio/pcm"}, {"rate", output_sample_rate}}},
      {"native_unit_ms", kNativeUnitMs},
      {"first_unit_ms", kNativeUnitMs},
      {"microturn_ms", nullptr},
      {"tail_policy", "pad"},
      {"supports_server_interrupt", false},
      {"supports_truncate", false},
      {"supports_cancel", true},
      {"supports_resume", false},
      {"partial_style", "append_only"},
      {"pressure_policy", "reject"},
      {"strict_order", true},
      {"sampling_parameters",
       {"temperature", "top_k", "top_p", "guidance_scale", "seed"}},
      {"supports_reference_audio", true},
      {"supports_instruction_updates", false},
      {"supports_sampling_updates", false},
  };
}

StreamResampler::StreamResampler(int input_rate, int output_rate)
    : input_rate_(input_rate), output_rate_(output_rate) {
  // The 3:2 ratio visits three sub-sample positions, so the kernel is evaluated
  // once per phase instead of once per output sample.
  for (int phase = 0; phase < 3; ++phase) {
    const double fraction = static_cast<double>(phase) / 3.0;
    std::vector<double> &taps = phases_[phase].taps;
    taps.assign(2 * kResampleHalfTaps + 1, 0.0);
    double sum = 0.0;
    for (int offset = -kResampleHalfTaps; offset <= kResampleHalfTaps;
         ++offset) {
      const double weight =
          ResampleKernel(fraction - static_cast<double>(offset));
      taps[static_cast<size_t>(offset + kResampleHalfTaps)] = weight;
      sum += weight;
    }
    for (double &weight : taps) {
      weight /= sum;
    }
  }
}

std::vector<float> StreamResampler::Process(const std::vector<float> &input) {
  if (input_rate_ == output_rate_) {
    return input;
  }
  buffer_.insert(buffer_.end(), input.begin(), input.end());
  input_count_ += static_cast<int64_t>(input.size());
  return Convert();
}

std::vector<float> StreamResampler::Flush() {
  if (input_rate_ == output_rate_) {
    return {};
  }
  // Zeros let the filter finish the samples still inside its window without
  // inventing audio the client did not send.
  buffer_.insert(buffer_.end(), 2 * kResampleHalfTaps, 0.0f);
  return Convert();
}

std::vector<float> StreamResampler::Convert() {
  // Exactly three outputs per two inputs, so the duration the model sees
  // matches the duration the client sent.
  const int64_t target = input_count_ * 3 / 2;
  std::vector<float> output;
  while (output_count_ < target) {
    const int64_t doubled = output_count_ * 2;
    const int phase = static_cast<int>(doubled % 3);
    const int64_t center = doubled / 3;
    const int64_t last = center + kResampleHalfTaps;
    if (last >= origin_ + static_cast<int64_t>(buffer_.size())) {
      break;
    }
    const std::vector<double> &taps = phases_[phase].taps;
    double value = 0.0;
    for (int offset = -kResampleHalfTaps; offset <= kResampleHalfTaps;
         ++offset) {
      const int64_t index = center + offset;
      if (index < origin_) {
        continue;
      }
      value += buffer_[static_cast<size_t>(index - origin_)] *
               taps[static_cast<size_t>(offset + kResampleHalfTaps)];
    }
    output.push_back(static_cast<float>(value));
    ++output_count_;
  }
  const int64_t keep_from =
      static_cast<int64_t>(
          std::floor(static_cast<double>(output_count_) * 2.0 / 3.0)) -
      kResampleHalfTaps;
  if (keep_from > origin_ + 4096) {
    buffer_.erase(buffer_.begin(),
                  buffer_.begin() +
                      static_cast<std::ptrdiff_t>(keep_from - origin_));
    origin_ = keep_from;
  }
  return output;
}

HaruRealtimeSession::HaruRealtimeSession(
    HaruModel &model, const omni_common::Tokenizer &tokenizer,
    const DuplexConfig &config, omni_server::TranscriptionWorker &worker,
    std::string system_prompt, std::vector<float> reference,
    int input_sample_rate, Sender sender)
    : RealtimeConnection(std::move(sender)), model_(model),
      tokenizer_(tokenizer), config_(config), worker_(worker),
      system_prompt_(std::move(system_prompt)),
      reference_(std::move(reference)), input_sample_rate_(input_sample_rate),
      model_sample_rate_(model.config().sample_rate),
      frame_samples_(model.config().frame_samples),
      resampler_(input_sample_rate, model.config().sample_rate) {
  // The OpenAI Realtime protocol opens with session.created; the playground
  // waits for it before sending session.update.
  SendEvent("session.created", {{"session", {{"id", "sess"}}}});
}

void HaruRealtimeSession::RunOnWorker(std::function<void()> fn) {
  worker_.Transcribe(
      [fn = std::move(fn)](
          const std::atomic<bool> &) -> omni_common::TranscriptionResult {
        fn();
        return {};
      },
      cancel_);
}

void HaruRealtimeSession::SendEvent(const char *type,
                                    const nlohmann::json &fields) {
  nlohmann::json event = {{"type", type}};
  event.update(fields);
  Send(event);
}

double HaruRealtimeSession::AcceptedEndMs() const {
  return static_cast<double>(accepted_input_samples_) * 1000.0 /
         static_cast<double>(input_sample_rate_);
}

double HaruRealtimeSession::PaddingMs() const {
  const double accepted_ms = AcceptedEndMs();
  return std::ceil(accepted_ms / kNativeUnitMs) * kNativeUnitMs - accepted_ms;
}

void HaruRealtimeSession::StepFrame(const std::vector<float> &frame,
                                    bool is_padding,
                                    std::vector<Frame> &frames) {
  frames.push_back(session_->Step(frame));
  ++stepped_frames_;
  if (!is_padding && stepped_frames_ % kNativeUnitFrames == 0) {
    pending_units_.push_back(static_cast<double>(kNativeUnitMs));
  }
}

void HaruRealtimeSession::SendUnitDone(double duration_ms) {
  const std::string unit_id = "unit_" + std::to_string(unit_count_);
  const double start_ms = static_cast<double>(unit_count_) * kNativeUnitMs;
  ++unit_count_;
  SendEvent("sglang.unit.done",
            {{"unit_id", unit_id},
             {"sglang",
              {{"media_time",
                {{"t_start_ms", start_ms}, {"duration_ms", duration_ms}}}}}});
}

void HaruRealtimeSession::EmitPendingUnits() {
  for (const double duration_ms : pending_units_) {
    SendUnitDone(duration_ms);
  }
  pending_units_.clear();
}

void HaruRealtimeSession::EmitFrame(const Frame &frame) {
  if (finished_) {
    return;
  }
  if (!response_open_) {
    response_open_ = true;
    SendEvent("response.created",
              {{"response",
                {{"id", "resp_" + std::to_string(response_index_)},
                 {"object", "realtime.response"},
                 {"status", "in_progress"},
                 {"output", nlohmann::json::array()}}}});
  }
  const nlohmann::json position = {
      {"response_id", "resp_" + std::to_string(response_index_)},
      {"item_id", "item_" + std::to_string(response_index_)},
      {"output_index", 0},
      {"content_index", 0}};
  const std::string pcm = omni_common::Pcm16Bytes(frame.audio);
  std::string encoded(4 * ((pcm.size() + 2) / 3) + 1, '\0');
  size_t encoded_length = encoded.size();
  mg_base64_encode(reinterpret_cast<const unsigned char *>(pcm.data()),
                   static_cast<int>(pcm.size()), encoded.data(),
                   &encoded_length);
  // civetweb counts the terminating NUL it writes.
  encoded.resize(encoded_length - 1);
  nlohmann::json audio_event = position;
  audio_event["delta"] = encoded;
  SendEvent("response.output_audio.delta", audio_event);
  if (!frame.text.empty()) {
    transcript_ += frame.text;
    nlohmann::json text_event = position;
    text_event["delta"] = frame.text;
    SendEvent("response.output_audio_transcript.delta", text_event);
  }
  if (frame.done) {
    finished_ = true;
    FinishResponse("incomplete", "max_output_tokens");
  }
}

void HaruRealtimeSession::FinishResponse(const std::string &status,
                                         const std::string &reason) {
  if (!response_open_) {
    return;
  }
  response_open_ = false;
  const std::string response_id = "resp_" + std::to_string(response_index_);
  const std::string item_id = "item_" + std::to_string(response_index_);
  const nlohmann::json position = {{"response_id", response_id},
                                   {"item_id", item_id},
                                   {"output_index", 0},
                                   {"content_index", 0}};
  nlohmann::json transcript_done = position;
  transcript_done["transcript"] = transcript_;
  SendEvent("response.output_audio_transcript.done", transcript_done);
  SendEvent("response.output_audio.done", position);
  nlohmann::json content = nlohmann::json::array();
  content.push_back({{"type", "output_audio"}, {"transcript", transcript_}});
  nlohmann::json output = nlohmann::json::array();
  output.push_back({{"id", item_id},
                    {"object", "realtime.item"},
                    {"type", "message"},
                    {"role", "assistant"},
                    {"content", content}});
  SendEvent("response.done", {{"response",
                               {{"id", response_id},
                                {"object", "realtime.response"},
                                {"status", status},
                                {"status_details", {{"reason", reason}}},
                                {"output", output},
                                {"usage", nullptr}}}});
  ++response_index_;
  transcript_.clear();
}

void HaruRealtimeSession::ApplySampling(const nlohmann::json &sampling) {
  if (sampling.contains("temperature")) {
    const double value = sampling.at("temperature").get<double>();
    if (!(value >= 0.0 && value <= 2.0)) {
      throw std::invalid_argument("temperature must be in [0, 2]");
    }
    config_.temperature = static_cast<float>(value);
  }
  if (sampling.contains("top_k")) {
    const long long value = sampling.at("top_k").get<long long>();
    if (value < 1 || value > static_cast<long long>(INT32_MAX)) {
      throw std::invalid_argument("top_k must be >= 1");
    }
    config_.top_k = static_cast<int>(value);
  }
  if (sampling.contains("top_p")) {
    const double value = sampling.at("top_p").get<double>();
    if (!(value > 0.0 && value <= 1.0)) {
      throw std::invalid_argument("top_p must be in (0, 1]");
    }
    config_.top_p = static_cast<float>(value);
  }
  if (sampling.contains("guidance_scale")) {
    const double value = sampling.at("guidance_scale").get<double>();
    if (!(value >= 1.0 && value <= 2.0)) {
      throw std::invalid_argument("guidance_scale must be in [1, 2]");
    }
    config_.guidance_scale = static_cast<float>(value);
  }
  if (sampling.contains("seed")) {
    const long long value = sampling.at("seed").get<long long>();
    if (value < 0 || value > static_cast<long long>(INT32_MAX)) {
      throw std::invalid_argument("seed must be in [0, 2147483647]");
    }
    config_.seed = static_cast<int>(value);
  }
  if (sampling.contains("speak_first")) {
    // A UI encodes this flag as 0/1, so accept both forms.
    const nlohmann::json value = sampling.at("speak_first");
    if (value.is_boolean()) {
      config_.speak_first = value.get<bool>();
    } else if (value.is_number_integer()) {
      config_.speak_first = value.get<int>() != 0;
    } else {
      throw std::invalid_argument("speak_first must be a boolean");
    }
  }
}

bool HaruRealtimeSession::Handle(const nlohmann::json &message) {
  const std::string type = message.at("type").get<std::string>();
  const std::string client_event_id = message.value("event_id", std::string{});
  if (type == "session.update") {
    if (started_) {
      SendError("invalid_request_error", "invalid_state",
                "session.update is only allowed before the first append");
      return true;
    }
    const auto &session = message.at("session");
    const auto &audio = session.value("audio", nlohmann::json::object());
    if (audio.contains("input") && audio.at("input").contains("format")) {
      const auto &format = audio.at("input").at("format");
      const std::string format_type = format.value("type", "audio/pcm");
      const int format_rate = format.value("rate", input_sample_rate_);
      if (format_type != "audio/pcm" || format_rate != input_sample_rate_) {
        SendError("invalid_request_error", "invalid_input_audio_format",
                  "this process accepts audio/pcm at " +
                      std::to_string(input_sample_rate_) + " Hz");
        return true;
      }
    }
    if (session.contains("instructions")) {
      system_prompt_ = session.at("instructions").get<std::string>();
    }
    const auto &sglang = session.value("sglang", nlohmann::json::object());
    if (sglang.contains("reference_audio")) {
      const auto &ref = sglang.at("reference_audio");
      if (ref.value("media_type", "") == "audio/wav" && ref.contains("data")) {
        const auto bytes =
            omni_common::DecodeBase64(ref.at("data").get<std::string>());
        if (bytes) {
          const std::string wav(bytes->begin(), bytes->end());
          reference_ = PrepareReference(wav);
        }
      }
    }
    if (sglang.contains("sampling")) {
      try {
        ApplySampling(sglang.at("sampling"));
      } catch (const std::exception &error) {
        SendError("invalid_request_error", "invalid_sampling", error.what());
        return true;
      }
    }
    // Start() runs before session.updated so the first append is served
    // without waiting for the prompt prefill.
    RunOnWorker([&] {
      session_ = std::make_unique<DuplexSession>(model_, tokenizer_, config_);
      first_frame_ = session_->Start(system_prompt_, reference_);
    });
    const nlohmann::json capabilities =
        Capabilities(input_sample_rate_, model_sample_rate_);
    const nlohmann::json input_format = {{"type", "audio/pcm"},
                                         {"rate", input_sample_rate_}};
    const nlohmann::json output_format = {{"type", "audio/pcm"},
                                          {"rate", model_sample_rate_}};
    SendEvent("session.updated", {{"client_event_id", client_event_id},
                                  {"session",
                                   {{"id", "sess"},
                                    {"object", "realtime.session"},
                                    {"type", "realtime"},
                                    {"instructions", system_prompt_},
                                    {"output_modalities", {"audio"}},
                                    {"audio",
                                     {{"input", {{"format", input_format}}},
                                      {"output", {{"format", output_format}}}}},
                                    {"sglang", {{"granted", capabilities}}}}},
                                  {"capabilities", capabilities}});
    started_ = true;
    return true;
  }
  if (type == "input_audio_buffer.append") {
    if (!started_) {
      SendError("invalid_request_error", "invalid_state",
                "session.update must precede input_audio_buffer.append");
      return true;
    }
    if (input_ended_) {
      SendError("invalid_request_error", "invalid_state",
                "input_audio_buffer.append after sglang.input_audio.end");
      return true;
    }
    const std::optional<std::vector<float>> samples =
        AppendedSamples(message.at("audio"));
    if (!samples.has_value()) {
      return true;
    }
    accepted_input_samples_ += static_cast<int64_t>(samples->size());
    nlohmann::json sequence = nullptr;
    if (message.contains("sglang") && message["sglang"].contains("seq")) {
      sequence = message["sglang"]["seq"];
    }
    SendEvent("sglang.input_audio.accepted",
              {{"seq", sequence},
               {"accepted_end_ms", AcceptedEndMs()},
               {"client_event_id", client_event_id}});
    std::vector<Frame> frames;
    RunOnWorker([&] {
      if (first_frame_.has_value()) {
        frames.push_back(*first_frame_);
        first_frame_.reset();
      }
      const std::vector<float> resampled = resampler_.Process(*samples);
      pending_.insert(pending_.end(), resampled.begin(), resampled.end());
      while (pending_.size() >= static_cast<size_t>(frame_samples_)) {
        std::vector<float> frame(pending_.begin(),
                                 pending_.begin() + frame_samples_);
        pending_.erase(pending_.begin(), pending_.begin() + frame_samples_);
        StepFrame(frame, false, frames);
      }
    });
    for (const Frame &frame : frames) {
      EmitFrame(frame);
    }
    EmitPendingUnits();
    return true;
  }
  if (type == "sglang.input_audio.end") {
    if (!input_ended_) {
      input_ended_ = true;
      SendEvent("sglang.input_audio.ended",
                {{"accepted_end_ms", AcceptedEndMs()},
                 {"tail_policy", "pad"},
                 {"client_event_id", client_event_id}});
      std::vector<Frame> frames;
      RunOnWorker([&] {
        if (first_frame_.has_value()) {
          frames.push_back(*first_frame_);
          first_frame_.reset();
        }
        const std::vector<float> tail = resampler_.Flush();
        pending_.insert(pending_.end(), tail.begin(), tail.end());
        while (!pending_.empty()) {
          std::vector<float> frame(
              pending_.begin(),
              pending_.begin() +
                  std::min<int64_t>(frame_samples_, pending_.size()));
          pending_.erase(pending_.begin(),
                         pending_.begin() + static_cast<int64_t>(frame.size()));
          const bool is_padding =
              static_cast<int>(frame.size()) < frame_samples_;
          frame.resize(frame_samples_, 0.0f);
          StepFrame(frame, is_padding, frames);
        }
        while (stepped_frames_ % kNativeUnitFrames != 0) {
          StepFrame(std::vector<float>(frame_samples_, 0.0f), true, frames);
        }
        // The last unit may cover less than 400 ms of real input.
        const double partial_ms =
            AcceptedEndMs() - static_cast<double>(unit_count_) * kNativeUnitMs;
        if (partial_ms > 0.0) {
          pending_units_.push_back(partial_ms);
        }
      });
      for (const Frame &frame : frames) {
        EmitFrame(frame);
      }
      EmitPendingUnits();
      FinishResponse("completed", "stop");
      SendEvent("sglang.input_audio.drained",
                {{"accepted_end_ms", AcceptedEndMs()},
                 {"consumed_ms", AcceptedEndMs()},
                 {"discarded_ms", 0.0},
                 {"padding_ms", PaddingMs()},
                 {"client_event_id", client_event_id}});
    }
    return true;
  }
  if (type == "response.cancel") {
    if (response_open_) {
      FinishResponse("cancelled", "cancelled");
    }
    return true;
  }
  if (type == "session.close") {
    FinishResponse("cancelled", "client_closed");
    SendEvent("session.closed", {{"reason", "client_closed"},
                                 {"client_event_id", client_event_id}});
    return false;
  }
  SendError("invalid_request_error", "unsupported_event",
            "unsupported " + type);
  return true;
}

} // namespace haru
