// SPDX-License-Identifier: Apache-2.0
// Native MLX server for Haru-4B: full-duplex spoken dialogue on /v1/realtime.
#include <atomic>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>

#include "civetweb.h"
#include "common/asr_service.h"
#include "common/tokenizer.h"
#include "haru/haru_audio.h"
#include "haru/haru_codec.h"
#include "haru/haru_config.h"
#include "haru/haru_head.h"
#include "haru/haru_model.h"
#include "haru/haru_realtime.h"
#include "haru/haru_session.h"

namespace {

// Serves the model's capabilities as JSON for the /v1/realtime playground.
// data is the model's input sample rate.
int HandleCapabilities(mg_connection *connection, void *data) {
  const int input_sample_rate = *static_cast<const int *>(data);
  const std::string body = haru::Capabilities(input_sample_rate, 24000).dump();
  mg_printf(connection,
            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n"
            "Content-Length: %zu\r\n\r\n%s",
            body.size(), body.c_str());
  return 1;
}

class HaruServedModel : public asr_service::ServedModel {
public:
  HaruServedModel(const std::filesystem::path &model_directory,
                  int quantize_bits, std::string instructions,
                  std::vector<float> reference, int warmup_frames,
                  int input_sample_rate)
      : model_(model_directory, quantize_bits), tokenizer_(model_directory),
        instructions_(std::move(instructions)),
        reference_(std::move(reference)), warmup_frames_(warmup_frames),
        input_sample_rate_(input_sample_rate) {
    mlx::core::set_cache_limit(asr_service::kDecodeBufferCacheLimitBytes);
  }

  asr_service::Transcription
  Prepare(std::vector<float>, const asr_service::FormFields &) const override {
    throw std::logic_error("haru serves no transcriptions");
  }
  bool Transcribes() const override { return false; }
  void AddHandlers(mg_context *context,
                   omni_server::TranscriptionWorker &worker) override {
    realtime_ = [this,
                 &worker](omni_server::RealtimeConnection::Sender sender) {
      return std::make_shared<haru::HaruRealtimeSession>(
          model_, tokenizer_, session_config_, worker, instructions_,
          reference_, input_sample_rate_, std::move(sender));
    };
    asr_service::AddRealtimeHandler(context, realtime_);
    // Register the capabilities endpoint before the warm-up: the civetweb
    // listener is already accepting connections at this point, and clients
    // (playground warm-up included) probe capabilities as soon as it answers.
    mg_set_request_handler(context, "/v1/realtime/capabilities$",
                           HandleCapabilities, &input_sample_rate_);
    Warmup(worker);
  }

private:
  // First sessions pay for Metal kernel compilation inside Start(), which
  // showed up as an 8-12 s first-frame delay, so compile it at load time on the
  // worker thread that owns the MLX stream.
  void Warmup(omni_server::TranscriptionWorker &worker) {
    if (warmup_frames_ <= 0) {
      return;
    }
    {
      worker.Transcribe(
          [this](
              const std::atomic<bool> &) -> omni_common::TranscriptionResult {
            haru::DuplexSession session(model_, tokenizer_, session_config_);
            (void)session.Start(instructions_, reference_);
            // A sine tone rather than silence: the codec's encode path takes
            // different branches on degenerate all-zero frames.
            std::vector<float> frame(3200, 0.0f);
            for (size_t sample = 0; sample < frame.size(); ++sample) {
              frame[sample] =
                  0.1f * static_cast<float>(
                             std::sin(2.0 * M_PI * 220.0 *
                                      static_cast<double>(sample) / 24000.0));
            }
            for (int index = 0; index < warmup_frames_; ++index) {
              (void)session.Step(frame);
            }
            return {};
          },
          omni_server::NewCancelFlag());
    }
  }

  haru::HaruModel model_;
  omni_common::Tokenizer tokenizer_;
  haru::DuplexConfig session_config_;
  std::string instructions_;
  std::vector<float> reference_;
  int warmup_frames_;
  int input_sample_rate_;
  asr_service::RealtimeFactory realtime_;
};

} // namespace

int main(int argc, char **argv) {
  int quantize_bits = 0;
  int warmup_frames = 16;
  int input_sample_rate = 24000;
  std::filesystem::path reference_audio;
  std::filesystem::path instructions_file;

  asr_service::ServedKind haru;
  haru.model_kind = "haru";
  haru.flags = {
      {"--quantize",
       [&](const std::string &value) { quantize_bits = std::stoi(value); }},
      {"--reference-audio",
       [&](const std::string &value) { reference_audio = value; }},
      {"--instructions-file",
       [&](const std::string &value) { instructions_file = value; }},
      {"--warmup-frames",
       [&](const std::string &value) { warmup_frames = std::stoi(value); }},
      {"--input-sample-rate",
       [&](const std::string &value) { input_sample_rate = std::stoi(value); }},
  };
  haru.check_flags = [&]() {
    if (quantize_bits != 0 && quantize_bits != 8) {
      throw std::invalid_argument("--quantize must be 0 or 8");
    }
    if (warmup_frames < 0) {
      throw std::invalid_argument("--warmup-frames must be >= 0");
    }
    if (input_sample_rate != 16000 && input_sample_rate != 24000) {
      throw std::invalid_argument("--input-sample-rate must be 16000 or 24000");
    }
  };
  haru.load = [&](const std::filesystem::path &model_directory) {
    std::string instructions;
    if (!instructions_file.empty()) {
      std::ifstream file(instructions_file);
      if (!file) {
        throw std::runtime_error("cannot read --instructions-file: " +
                                 instructions_file.string());
      }
      instructions.assign((std::istreambuf_iterator<char>(file)),
                          std::istreambuf_iterator<char>());
      const auto visible = [](unsigned char c) { return std::isspace(c) == 0; };
      instructions.erase(
          instructions.begin(),
          std::find_if(instructions.begin(), instructions.end(), visible));
      instructions.erase(
          std::find_if(instructions.rbegin(), instructions.rend(), visible)
              .base(),
          instructions.end());
    }
    std::vector<float> reference;
    if (!reference_audio.empty()) {
      std::ifstream input(reference_audio, std::ios::binary);
      if (!input) {
        throw std::runtime_error("cannot read --reference-audio: " +
                                 reference_audio.string());
      }
      const std::string bytes((std::istreambuf_iterator<char>(input)),
                              std::istreambuf_iterator<char>());
      reference = haru::PrepareReference(bytes);
    }
    return std::make_unique<HaruServedModel>(
        model_directory, quantize_bits, std::move(instructions),
        std::move(reference), warmup_frames, input_sample_rate);
  };
  return asr_service::Serve(argc, argv, haru);
}
