// SPDX-License-Identifier: Apache-2.0
#include "qwen3_tts/qwen3_tts_model.h"

#include <memory>

#include "common/http.h"
#include "common/service.h"
#include "common/tts_service.h"

namespace qwen3_tts {

class SpeechService final : public omni_server::Service {
public:
  explicit SpeechService(const std::filesystem::path &directory)
      : worker_(
            [this, directory] { model_ = std::make_unique<Model>(directory); },
            [this] { model_.reset(); }) {}

  void SetModelName(const std::string &name, bool chosen) override {
    model_name_ = chosen ? name : std::string();
  }

  void Register(mg_context *context, const std::string &) override {
    mg_set_request_handler(
        context, "/v1/audio/speech$",
        [](mg_connection *connection, void *service) {
          return static_cast<SpeechService *>(service)->Handle(connection);
        },
        this);
  }

  std::map<std::string, int> RequestStates() const override {
    return worker_.RequestStates();
  }

  void CancelAll() override { worker_.Stop(); }

private:
  int Handle(mg_connection *connection) {
    if (!omni_server::IsPost(connection)) {
      return omni_server::WriteJson(connection, 405,
                                    {{"detail", "Method Not Allowed"}});
    } else {
    }
    try {
      const auto body = omni_server::ReadBody(connection, 4 * 1024 * 1024);
      if (!body.has_value()) {
        return 413;
      } else {
      }
      const auto request =
          model_->ParseRequest(nlohmann::json::parse(*body), model_name_);
      const auto format = request.stream
                              ? tts_service::AudioFormat::StreamingPcm
                          : request.pcm ? tts_service::AudioFormat::Pcm
                                        : tts_service::AudioFormat::Wav;
      tts_service::AudioResponse response(connection, format);
      auto job = std::make_shared<tts_service::SynthesisWorker::Job<Result>>();
      job->synthesize = [this, &request, &response](
                            const std::function<void()> &check_cancelled) {
        if (request.stream) {
          return model_->Synthesize(
              request, check_cancelled, [&](const std::vector<float> &samples) {
                response.WritePcm(EncodeAudio(samples, true), kSampleRate, 1);
              });
        } else {
          return model_->Synthesize(request, check_cancelled);
        }
      };
      return response.SubmitAndRespond(
          worker_, job,
          [&](const Result &result) {
            if (!request.stream) {
              response.WritePcm(EncodeAudio(result.samples, true), kSampleRate,
                                1);
            } else {
            }
          },
          "Qwen3-TTS is busy");
    } catch (const nlohmann::json::exception &) {
      return omni_server::BadRequest(
          connection, "invalid JSON, missing field or incorrect field type");
    } catch (const std::invalid_argument &error) {
      return omni_server::BadRequest(connection, error.what());
    } catch (const std::exception &error) {
      std::cerr << "Qwen3-TTS request failed: " << error.what() << '\n';
      return omni_server::WriteJson(connection, 500,
                                    {{"detail", "Qwen3-TTS request failed"}});
    }
  }

  std::string model_name_;
  std::unique_ptr<Model> model_;
  tts_service::SynthesisWorker worker_;
};

} // namespace qwen3_tts

int main(int argc, char **argv) {
  mlx::core::set_cache_limit(1024ULL * 1024 * 1024);
  return omni_server::Serve(
      argc, argv,
      {{"qwen3_tts",
        [](const std::filesystem::path &directory) {
          return std::make_unique<qwen3_tts::SpeechService>(directory);
        },
        {},
        {}}});
}
