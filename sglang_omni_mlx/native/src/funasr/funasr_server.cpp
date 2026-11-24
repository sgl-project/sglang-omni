// SPDX-License-Identifier: Apache-2.0
// Native Fun-ASR-Nano server: transcriptions over HTTP (JSON or SSE) with the
// API of qwen3_asr_server, without its realtime API, and Voxt's supervisor
// protocol.
//
//   funasr_server --model-path DIR [--model-name NAME] [--host H] [--port P]
//     [--draft-model-path DIR [--draft-bits 4|6|8]
//     [--draft-policy adaptive|fixed]]
//   funasr_server --supervised --model-kind funasr --model-directory DIR
//
// Request fields beside the audio: language (named in the prompt; Chinese and
// auto leave the model to choose), prompt (comma-separated hotwords),
// max_new_tokens (one budget for the whole request; unset or 0 is the
// default), temperature (0 only: decoding is greedy), repetition_penalty
// (decodes without the draft), stream and include_generation_metadata.
#include <memory>
#include <stdexcept>

#include "common/asr_service.h"
#include "funasr/funasr_transcriber.h"

namespace {

class FunAsrModelService : public asr_service::ServedModel {
public:
  FunAsrModelService(const std::filesystem::path &model_directory,
                     const std::filesystem::path &draft_directory,
                     int draft_quantization_bits,
                     omni_common::DraftPolicy draft_policy)
      : transcriber_(model_directory, draft_directory, draft_quantization_bits,
                     draft_policy) {
    mlx::core::set_cache_limit(asr_service::kDecodeBufferCacheLimitBytes);
  }

  asr_service::Transcription
  Prepare(std::vector<float> samples,
          const asr_service::FormFields &form) const override {
    funasr::FunAsrOptions options;
    options.language = asr_service::TextField(form, "language");
    options.prompt = asr_service::TextField(form, "prompt");
    options.max_new_tokens = asr_service::IntegerField(form, "max_new_tokens");
    options.repetition_penalty =
        asr_service::NumberField(form, "repetition_penalty")
            .value_or(options.repetition_penalty);
    if (options.max_new_tokens.value_or(0) < 0) {
      throw std::invalid_argument("max_new_tokens must not be negative");
    } else if (options.repetition_penalty <= 0) {
      throw std::invalid_argument("repetition_penalty must be positive");
    } else if (asr_service::NumberField(form, "temperature").value_or(0) != 0) {
      throw std::invalid_argument("temperature must be 0: decoding is greedy");
    } else {
    }
    return [this, samples = std::move(samples),
            options = std::move(options)](const std::atomic<bool> &cancel) {
      return transcriber_.Transcribe(samples, options, cancel);
    };
  }

private:
  funasr::FunAsrTranscriber transcriber_;
};

} // namespace

int main(int argc, char **argv) {
  std::filesystem::path draft_directory;
  int draft_quantization_bits = 0;
  omni_common::DraftPolicy draft_policy = omni_common::DraftPolicy::kFixed;
  asr_service::ServedKind kind;
  kind.model_kind = "funasr";
  kind.flags = {
      {"--draft-model-path",
       [&](const std::string &value) { draft_directory = value; }},
      {"--draft-policy",
       [&](const std::string &value) {
         draft_policy = omni_common::ParseDraftPolicy(value);
       }},
      {"--draft-bits",
       [&](const std::string &value) {
         draft_quantization_bits = std::stoi(value);
       }},
  };
  kind.check_flags = [&]() {
    if (draft_quantization_bits != 0 && draft_quantization_bits != 4 &&
        draft_quantization_bits != 6 && draft_quantization_bits != 8) {
      throw std::invalid_argument("--draft-bits must be 4, 6 or 8");
    } else if (draft_quantization_bits != 0 && draft_directory.empty()) {
      throw std::invalid_argument("--draft-bits needs --draft-model-path");
    } else {
    }
  };
  kind.load = [&](const std::filesystem::path &model_directory) {
    return std::make_unique<FunAsrModelService>(
        model_directory, draft_directory, draft_quantization_bits,
        draft_policy);
  };
  return asr_service::Serve(argc, argv, kind);
}
