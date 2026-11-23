#include <memory>
#include <stdexcept>
#include <utility>

#include "common/asr_service.h"
#include "ming3_asr/ming3_asr_transcriber.h"

namespace {

class MingModelService : public asr_service::ServedModel {
public:
  MingModelService(const std::filesystem::path &model_directory,
                   const std::filesystem::path &draft_directory,
                   int draft_quantization_bits)
      : transcriber_(model_directory, draft_directory,
                     draft_quantization_bits) {}

  asr_service::Transcription
  Prepare(std::vector<float> samples,
          const asr_service::FormFields &form) const override {
    ming3_asr::MingOptions options;
    options.language = asr_service::TextField(form, "language");
    options.prompt = asr_service::TextField(form, "prompt");
    options.hotwords = asr_service::TextField(form, "hotwords");
    options.max_new_tokens = asr_service::IntegerField(form, "max_new_tokens")
                                 .value_or(options.max_new_tokens);
    options.no_repeat_ngram_size =
        asr_service::IntegerField(form, "no_repeat_ngram_size")
            .value_or(options.no_repeat_ngram_size);
    const float temperature =
        asr_service::NumberField(form, "temperature").value_or(0.0f);
    if (temperature != 0.0f) {
      throw std::invalid_argument(
          "Ming ASR currently supports only temperature=0");
    } else {
    }
    return [this, request = transcriber_.Prepare(std::move(samples), options)](
               const std::atomic<bool> &cancel) {
      return transcriber_.Transcribe(request, cancel);
    };
  }

private:
  ming3_asr::MingTranscriber transcriber_;
};

} // namespace

int main(int argc, char **argv) {
  std::filesystem::path draft_directory;
  int draft_quantization_bits = 0;
  asr_service::ServedKind kind;
  kind.model_kind = "ming3_asr";
  kind.flags = {
      {"--draft-model-path",
       [&](const std::string &value) { draft_directory = value; }},
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
    return std::make_unique<MingModelService>(model_directory, draft_directory,
                                              draft_quantization_bits);
  };
  return asr_service::Serve(argc, argv, kind);
}
