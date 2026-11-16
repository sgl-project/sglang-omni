// SPDX-License-Identifier: Apache-2.0
#include "haru/haru_realtime.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>

#include "common/text.h"
#include "haru/haru_audio.h"

namespace haru {

namespace {

constexpr int kFrameSamples = 3200;
constexpr int kSampleRate = 24000;
constexpr int kNativeUnitMs = 400; // three 3200-sample frames
constexpr int kNativeUnitFrames = 3;
constexpr float kPcm16FullScale = 32768.0f;
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
// per side and sample, as scipy's resample_poly uses for a 3:2 ratio.
constexpr int kResampleHalfTaps = 30;
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

nlohmann::json Capabilities(int input_sample_rate) {
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
      {"output_audio_format", {{"type", "audio/pcm"}, {"rate", kSampleRate}}},
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
       {"temperature", "top_k", "top_p", "guidance_scale", "seed",
        "speak_first"}},
      {"supports_reference_audio", true},
      {"supports_instruction_updates", false},
      {"supports_sampling_updates", false},
  };
}

StreamResampler::StreamResampler(int input_rate) : input_rate_(input_rate) {}

std::vector<float> StreamResampler::Process(const std::vector<float> &input) {
  if (input_rate_ == kSampleRate) {
    return input;
  }
  buffer_.insert(buffer_.end(), input.begin(), input.end());
  input_count_ += static_cast<int64_t>(input.size());
  return Convert();
}

std::vector<float> StreamResampler::Flush() {
  if (input_rate_ == kSampleRate) {
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
    const double position = static_cast<double>(output_count_) * 2.0 / 3.0;
    const int64_t center = static_cast<int64_t>(std::floor(position));
    const int64_t last = center + kResampleHalfTaps;
    if (last >= origin_ + static_cast<int64_t>(buffer_.size())) {
      break;
    }
    const int64_t first = std::max(center - kResampleHalfTaps + 1, origin_);
    double value = 0.0;
    for (int64_t index = first; index <= last; ++index) {
      value += buffer_[static_cast<size_t>(index - origin_)] *
               ResampleKernel(position - static_cast<double>(index));
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
    HaruModel &model, VibeVoiceCodec &codec, MeanFlowHead &head,
    const omni_common::Tokenizer &tokenizer, const DuplexConfig &config,
    qwen3_asr::TranscriptionWorker &worker, std::string system_prompt,
    std::vector<float> reference, int input_sample_rate, Sender sender)
    : RealtimeConnection(std::move(sender)), model_(model), codec_(codec),
      head_(head), tokenizer_(tokenizer), config_(config), worker_(worker),
      system_prompt_(std::move(system_prompt)),
      reference_(std::move(reference)), input_sample_rate_(input_sample_rate),
      resampler_(input_sample_rate) {
  // The OpenAI Realtime protocol opens with session.created; the playground
  // waits for it before sending session.update.
  SendEvent("session.created", {{"session", {{"id", "sess"}}}});
}

void HaruRealtimeSession::RunOnWorker(std::function<void()> fn) {
  worker_.Transcribe(
      [fn = std::move(fn)](
          const std::atomic<bool> &) -> qwen3_asr::TranscriptionResult {
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

std::vector<float>
HaruRealtimeSession::AppendSamples(const nlohmann::json &message) {
  const auto bytes =
      omni_common::DecodeBase64(message.at("audio").get<std::string>());
  if (!bytes || bytes->size() % 2 != 0) {
    throw std::invalid_argument("audio must be base64 PCM16");
  }
  std::vector<float> samples(bytes->size() / 2);
  for (size_t i = 0; i < samples.size(); ++i) {
    const int16_t value =
        static_cast<int16_t>((*bytes)[2 * i] | ((*bytes)[2 * i + 1] << 8));
    samples[i] = static_cast<float>(value) / kPcm16FullScale;
  }
  return samples;
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
  std::vector<uint8_t> pcm(frame.audio.size() * 2);
  for (size_t i = 0; i < frame.audio.size(); ++i) {
    const float clamped = std::clamp(frame.audio[i], -1.0f, 1.0f);
    const int16_t value =
        static_cast<int16_t>(clamped * (kPcm16FullScale - 1.0f));
    pcm[2 * i] = static_cast<uint8_t>(value & 0xFF);
    pcm[2 * i + 1] = static_cast<uint8_t>((value >> 8) & 0xFF);
  }
  nlohmann::json audio_event = position;
  audio_event["delta"] = omni_common::EncodeBase64(pcm);
  SendEvent("response.output_audio.delta", audio_event);
  if (!frame.text.empty()) {
    transcript_ += frame.text;
    nlohmann::json text_event = position;
    text_event["delta"] = frame.text;
    SendEvent("response.output_audio_transcript.delta", text_event);
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
    config_.speak_first = sampling.at("speak_first").get<bool>();
  }
}

bool HaruRealtimeSession::Handle(const nlohmann::json &message) {
  const std::string type = message.at("type").get<std::string>();
  const std::string client_event_id = message.value("event_id", std::string{});
  if (type == "session.update") {
    const auto &session = message.at("session");
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
          reference_ = PrepareReferenceBytes(wav);
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
      session_ = std::make_unique<DuplexSession>(model_, codec_, head_,
                                                 tokenizer_, config_);
      first_frame_ = session_->Start(system_prompt_, reference_);
    });
    const nlohmann::json capabilities = Capabilities(input_sample_rate_);
    const nlohmann::json input_format = {{"type", "audio/pcm"},
                                         {"rate", input_sample_rate_}};
    const nlohmann::json output_format = {{"type", "audio/pcm"},
                                          {"rate", kSampleRate}};
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
    std::vector<float> samples;
    try {
      samples = AppendSamples(message);
    } catch (const std::invalid_argument &error) {
      SendError("invalid_request_error", "invalid_audio", error.what());
      return true;
    }
    accepted_input_samples_ += static_cast<int64_t>(samples.size());
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
      const std::vector<float> resampled = resampler_.Process(samples);
      pending_.insert(pending_.end(), resampled.begin(), resampled.end());
      while (pending_.size() >= static_cast<size_t>(kFrameSamples)) {
        std::vector<float> frame(pending_.begin(),
                                 pending_.begin() + kFrameSamples);
        pending_.erase(pending_.begin(), pending_.begin() + kFrameSamples);
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
        if (!pending_.empty()) {
          std::vector<float> frame = pending_;
          frame.resize(kFrameSamples, 0.0f);
          pending_.clear();
          StepFrame(frame, true, frames);
        }
        while (stepped_frames_ % kNativeUnitFrames != 0) {
          StepFrame(std::vector<float>(kFrameSamples, 0.0f), true, frames);
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
      pending_.clear();
    }
    return true;
  }
  if (type == "session.close") {
    FinishResponse("cancelled", "client_closed");
    SendEvent("session.closed", {{"reason", "client_closed"},
                                 {"client_event_id", client_event_id}});
    return false;
  }
  SendError("invalid_request", "unsupported_event", "unsupported " + type);
  return true;
}

} // namespace haru
