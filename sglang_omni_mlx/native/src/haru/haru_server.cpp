// SPDX-License-Identifier: Apache-2.0
// Native MLX server for Haru-4B: full-duplex spoken dialogue on /v1/realtime.
#include <filesystem>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>

#include "common/tokenizer.h"
#include "haru/haru_audio.h"
#include "haru/haru_codec.h"
#include "haru/haru_config.h"
#include "haru/haru_head.h"
#include "haru/haru_model.h"
#include "haru/haru_realtime.h"
#include "haru/haru_session.h"
#include "qwen3_asr/asr_service.h"

namespace {

class HaruServedModel : public asr_service::ServedModel {
public:
  HaruServedModel(const std::filesystem::path &model_directory,
                  int quantize_bits, std::string instructions,
                  std::vector<float> reference, int seed)
      : model_(model_directory, quantize_bits),
        codec_(model_.checkpoint(), model_.config().codec),
        head_(model_.checkpoint(), model_.config().head),
        tokenizer_(model_directory),
        instructions_(std::move(instructions)),
        reference_(std::move(reference)) {
    session_config_.seed = seed;
    mlx::core::set_cache_limit(asr_service::kDecodeBufferCacheLimitBytes);
  }

  asr_service::Transcription
  Prepare(std::vector<float>, const asr_service::FormFields &) const override {
    throw std::logic_error("haru serves no transcriptions");
  }
  bool Transcribes() const override { return false; }
  void AddHandlers(mg_context *context,
                   qwen3_asr::TranscriptionWorker &worker) override {
    realtime_ = [this, &worker](qwen3_asr::RealtimeConnection::Sender sender) {
      return std::make_shared<haru::HaruRealtimeSession>(
          model_, codec_, head_, tokenizer_, session_config_, worker,
          instructions_, reference_, std::move(sender));
    };
    asr_service::AddRealtimeHandler(context, realtime_);
  }

private:
  haru::HaruModel model_;
  haru::VibeVoiceCodec codec_;
  haru::MeanFlowHead head_;
  omni_common::Tokenizer tokenizer_;
  haru::DuplexConfig session_config_;
  std::string instructions_;
  std::vector<float> reference_;
  asr_service::RealtimeFactory realtime_;
};

} // namespace

int main(int argc, char **argv) {
  int quantize_bits = 0;
  int seed = -1;
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
      {"--seed", [&](const std::string &value) { seed = std::stoi(value); }},
  };
  haru.check_flags = [&]() {
    if (quantize_bits != 0 && quantize_bits != 8) {
      throw std::invalid_argument("--quantize must be 0 or 8");
    }
  };
  haru.load = [&](const std::filesystem::path &model_directory) {
    std::string instructions;
    if (!instructions_file.empty()) {
      std::ifstream file(instructions_file);
      instructions.assign((std::istreambuf_iterator<char>(file)),
                          std::istreambuf_iterator<char>());
    }
    std::vector<float> reference;
    if (!reference_audio.empty()) {
      reference = haru::PrepareReference(reference_audio.string());
    }
    return std::make_unique<HaruServedModel>(model_directory, quantize_bits,
                                             std::move(instructions),
                                             std::move(reference), seed);
  };
  return asr_service::Serve(argc, argv, haru);
}
