// SPDX-License-Identifier: Apache-2.0
// Native MLX server for Haru-4B: full-duplex spoken dialogue on /v1/realtime.
#include <algorithm>
#include <cctype>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <memory>
#include <stdexcept>
#include <string>

#include "civetweb.h"
#include "common/asr_service.h"
#include "common/tokenizer.h"
#include "haru/haru_audio.h"
#include "haru/haru_config.h"
#include "haru/haru_model.h"
#include "haru/haru_realtime.h"
#include "haru/haru_session.h"

namespace {

// Serves the capabilities JSON text in data.
int HandleCapabilities(mg_connection *connection, void *data) {
  const std::string &body = *static_cast<const std::string *>(data);
  mg_printf(connection,
            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n"
            "Content-Length: %zu\r\n\r\n%s",
            body.size(), body.c_str());
  return 200;
}

class HaruServedModel : public asr_service::ServedModel {
public:
  HaruServedModel(const std::filesystem::path &model_directory,
                  std::string instructions, const std::string &reference_wav,
                  int warmup_frames, int input_sample_rate)
      : model_(model_directory), tokenizer_(model_directory),
        instructions_(std::move(instructions)),
        input_sample_rate_(input_sample_rate),
        capabilities_(
            haru::Capabilities(input_sample_rate, model_.config().sample_rate)
                .dump()) {
    mlx::core::set_cache_limit(asr_service::kDecodeBufferCacheLimitBytes);
    if (!reference_wav.empty()) {
      reference_ =
          haru::PrepareReference(reference_wav, model_.config().sample_rate);
    } else {
    }
    Warmup(warmup_frames);
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
          model_, tokenizer_, haru::DuplexConfig{}, worker, instructions_,
          reference_, input_sample_rate_, std::move(sender));
    };
    asr_service::AddRealtimeHandler(context, realtime_);
    mg_set_request_handler(context, "/v1/realtime/capabilities$",
                           HandleCapabilities, &capabilities_);
  }

private:
  // The first session otherwise waits 8 to 12 s for Metal to compile. It runs
  // in load, so a failure reaches the supervisor as a failed load.
  void Warmup(int frame_count) const {
    if (frame_count <= 0) {
      return;
    } else {
    }
    const haru::HaruConfig &config = model_.config();
    haru::DuplexSession session(model_, tokenizer_, haru::DuplexConfig{});
    (void)session.Start(instructions_, reference_);
    // A tone rather than silence: the codec's encode path takes different
    // branches on all-zero frames.
    std::vector<float> frame(static_cast<size_t>(config.frame_samples));
    for (size_t sample = 0; sample < frame.size(); ++sample) {
      frame[sample] =
          0.1f * static_cast<float>(
                     std::sin(2.0 * M_PI * 220.0 * static_cast<double>(sample) /
                              config.sample_rate));
    }
    for (int index = 0; index < frame_count; ++index) {
      (void)session.Step(frame);
    }
  }

  haru::HaruModel model_;
  omni_common::Tokenizer tokenizer_;
  std::string instructions_;
  std::vector<float> reference_;
  int input_sample_rate_;
  std::string capabilities_;
  asr_service::RealtimeFactory realtime_;
};

std::string ReadFile(const std::filesystem::path &path,
                     const std::string &flag) {
  std::ifstream file(path, std::ios::binary);
  if (!file) {
    throw std::runtime_error("cannot read " + flag + ": " + path.string());
  } else {
  }
  return {std::istreambuf_iterator<char>(file),
          std::istreambuf_iterator<char>()};
}

} // namespace

int main(int argc, char **argv) {
  int warmup_frames = 16;
  int input_sample_rate = 24000;
  std::filesystem::path reference_audio;
  std::filesystem::path instructions_file;

  asr_service::ServedKind haru;
  haru.model_kind = "haru";
  haru.flags = {
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
    if (warmup_frames < 0) {
      throw std::invalid_argument("--warmup-frames must be >= 0");
    } else {
    }
    if (input_sample_rate != 16000 && input_sample_rate != 24000) {
      throw std::invalid_argument("--input-sample-rate must be 16000 or 24000");
    } else {
    }
  };
  haru.load = [&](const std::filesystem::path &model_directory) {
    std::string instructions;
    if (!instructions_file.empty()) {
      instructions = ReadFile(instructions_file, "--instructions-file");
      const auto visible = [](unsigned char c) { return std::isspace(c) == 0; };
      instructions.erase(
          instructions.begin(),
          std::find_if(instructions.begin(), instructions.end(), visible));
      instructions.erase(
          std::find_if(instructions.rbegin(), instructions.rend(), visible)
              .base(),
          instructions.end());
    } else {
    }
    return std::make_unique<HaruServedModel>(
        model_directory, std::move(instructions),
        reference_audio.empty()
            ? std::string{}
            : ReadFile(reference_audio, "--reference-audio"),
        warmup_frames, input_sample_rate);
  };
  return asr_service::Serve(argc, argv, haru);
}
