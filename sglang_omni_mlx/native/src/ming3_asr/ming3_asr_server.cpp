#include <memory>
#include <stdexcept>
#include <utility>

#include "ming3_asr/ming3_asr_transcriber.h"
#include "qwen3_asr/asr_service.h"

namespace {

class MingModelService : public asr_service::ServedModel {
public:
  explicit MingModelService(const std::filesystem::path &model_directory)
      : transcriber_(model_directory) {}

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
  asr_service::ServedKind kind;
  kind.model_kind = "ming3_asr";
  kind.load = [](const std::filesystem::path &model_directory) {
    return std::make_unique<MingModelService>(model_directory);
  };
  return asr_service::Serve(argc, argv, kind);
}
