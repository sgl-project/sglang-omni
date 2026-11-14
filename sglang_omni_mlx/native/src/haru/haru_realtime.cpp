// SPDX-License-Identifier: Apache-2.0
#include "haru/haru_realtime.h"

#include <cstdint>
#include <string>

#include "common/text.h"
#include <iostream>
#include "haru/haru_audio.h"

namespace haru {

namespace {

constexpr int kFrameSamples = 3200;
constexpr int kSampleRate = 24000;
constexpr int kNativeUnitMs = 400;  // three 3200-sample frames
constexpr float kPcm16FullScale = 32768.0f;

nlohmann::json Capabilities() {
  return {
      {"interaction", "native"},
      {"native_full_duplex", true},
      {"output_modalities", {"audio"}},
      {"input_audio_format", {{"type", "audio/pcm"}, {"rate", kSampleRate}}},
      {"output_audio_format", {{"type", "audio/pcm"}, {"rate", kSampleRate}}},
      {"native_unit_ms", kNativeUnitMs},
      {"supports_reference_audio", true},
      {"sampling_parameters",
       {"temperature", "top_k", "top_p", "guidance_scale", "seed", "speak_first"}},
  };
}

} // namespace

HaruRealtimeSession::HaruRealtimeSession(
    HaruModel &model, VibeVoiceCodec &codec, MeanFlowHead &head,
    const omni_common::Tokenizer &tokenizer, const DuplexConfig &config,
    qwen3_asr::TranscriptionWorker &worker, std::string system_prompt,
    std::vector<float> reference, Sender sender)
    : RealtimeConnection(std::move(sender)),
      model_(model),
      codec_(codec),
      head_(head),
      tokenizer_(tokenizer),
      config_(config),
      worker_(worker),
      system_prompt_(std::move(system_prompt)),
      reference_(std::move(reference)) {}

void HaruRealtimeSession::RunOnWorker(std::function<void()> fn) {
  worker_.Transcribe(
      [fn = std::move(fn)](const std::atomic<bool> &)
          -> qwen3_asr::TranscriptionResult {
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

std::vector<float> HaruRealtimeSession::AppendSamples(
    const nlohmann::json &message) {
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

void HaruRealtimeSession::EmitFrame(const Frame &frame) {
  if (!response_open_) {
    response_open_ = true;
    SendEvent("response.created",
              {{"response",
                {{"id", "resp_" + std::to_string(response_index_)},
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

void HaruRealtimeSession::FinishResponse(const std::string &reason) {
  if (!response_open_) {
    return;
  }
  response_open_ = false;
  SendEvent("response.output_audio_transcript.done",
            {{"transcript", transcript_}});
  SendEvent("response.output_audio.done");
  SendEvent("response.done",
            {{"response",
              {{"id", "resp_" + std::to_string(response_index_)},
               {"status", "completed"},
               {"status_details", {{"reason", reason}}}}}});
  ++response_index_;
  transcript_.clear();
}

bool HaruRealtimeSession::Handle(const nlohmann::json &message) {
  const std::string type = message.at("type").get<std::string>();
  if (type == "session.update") {
    const auto &session = message.at("session");
    if (session.contains("instructions")) {
      system_prompt_ = session.at("instructions").get<std::string>();
    }
    const auto &sglang = session.value("sglang", nlohmann::json::object());
    if (sglang.contains("reference_audio")) {
      const auto &ref = sglang.at("reference_audio");
      if (ref.value("media_type", "") == "audio/wav" && ref.contains("data")) {
        const auto bytes = omni_common::DecodeBase64(ref.at("data").get<std::string>());
        if (bytes) {
          const std::string wav(bytes->begin(), bytes->end());
          reference_ = PrepareReferenceBytes(wav);
        }
      }
    }
    SendEvent("session.updated",
              {{"session", {{"id", "sess"}}}, {"capabilities", Capabilities()}});
    RunOnWorker([&] {
      session_ = std::make_unique<DuplexSession>(model_, codec_, head_,
                                                 tokenizer_, config_);
      first_frame_ = session_->Start(system_prompt_, reference_);
    });
    started_ = true;
    return true;
  }
  if (type == "input_audio_buffer.append") {
    SendEvent("sglang.input_audio.accepted");
    const std::vector<float> samples = AppendSamples(message);
    std::vector<Frame> frames;
    RunOnWorker([&] {
      if (first_frame_.has_value()) {
        frames.push_back(*first_frame_);
        first_frame_.reset();
      }
      pending_.insert(pending_.end(), samples.begin(), samples.end());
      while (pending_.size() >= kFrameSamples) {
        std::vector<float> frame(pending_.begin(),
                                 pending_.begin() + kFrameSamples);
        pending_.erase(pending_.begin(), pending_.begin() + kFrameSamples);
        frames.push_back(session_->Step(frame));
      }
    });
    for (const Frame &frame : frames) {
      EmitFrame(frame);
    }
    return true;
  }
  if (type == "sglang.input_audio.end") {
    if (!input_ended_) {
      input_ended_ = true;
      std::vector<Frame> frames;
      RunOnWorker([&] {
        if (!pending_.empty()) {
          std::vector<float> frame = pending_;
          frame.resize(kFrameSamples, 0.0f);
          frames.push_back(session_->Step(frame));
          pending_.clear();
        }
      });
      for (const Frame &frame : frames) {
        EmitFrame(frame);
      }
      FinishResponse("stop");
      SendEvent("sglang.input_audio.drained");
    }
    return true;
  }
  if (type == "response.cancel") {
    if (response_open_) {
      FinishResponse("cancelled");
      pending_.clear();
    }
    return true;
  }
  if (type == "session.close") {
    SendEvent("session.closed");
    return false;
  }
  SendError("invalid_request", "unsupported_event", "unsupported " + type);
  return true;
}

} // namespace haru
