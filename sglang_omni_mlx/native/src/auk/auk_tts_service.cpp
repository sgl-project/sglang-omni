// SPDX-License-Identifier: Apache-2.0
#include "auk/auk_tts_service.h"

#include <iostream>
#include <typeinfo>

#include "common/audio.h"
#include "auk/auk_request.h"
#include "common/http.h"

namespace auk {

namespace {

using omni_server::Json;
constexpr size_t kMaximumBodyBytes = 16 * 1024 * 1024;

} // namespace

SpeechService::SpeechService(const std::filesystem::path &model_directory,
                             mlx::core::Dtype dtype, float max_seconds)
    : max_seconds_(max_seconds) {
  std::promise<void> loaded;
  auto ready = loaded.get_future();
  thread_ = std::thread(&SpeechService::Run, this, model_directory, dtype,
                        std::move(loaded));
  try {
    ready.get();
  } catch (...) {
    thread_.join();
    throw;
  }
}

SpeechService::~SpeechService() {
  stopping_.store(true);
  CancelAll();
  wake_.notify_all();
  thread_.join();
}

void SpeechService::Run(const std::filesystem::path &model_directory,
                        mlx::core::Dtype dtype, std::promise<void> loaded) {
  std::unique_ptr<Model> model;
  try {
    model = std::make_unique<Model>(model_directory, dtype, max_seconds_);
    loaded.set_value();
  } catch (...) {
    loaded.set_exception(std::current_exception());
    return;
  }
  while (true) {
    std::shared_ptr<Job> job;
    {
      std::unique_lock<std::mutex> lock(mutex_);
      wake_.wait(lock, [&] { return stopping_.load() || job_ != nullptr; });
      if (stopping_.load() && job_ == nullptr) {
        return;
      } else {
        job = job_;
      }
    }
    std::optional<SynthesisResult> result;
    std::exception_ptr error;
    try {
      const auto check_cancelled = [&] {
        if (stopping_.load() || job->cancelled.load()) {
          throw SynthesisCancelled();
        } else {
        }
      };
      check_cancelled();
      result = model->Synthesize(job->request, check_cancelled);
      check_cancelled();

    } catch (...) {
      error = std::current_exception();
    }
    try {
      mlx::core::synchronize();
      mlx::core::clear_cache();
    } catch (const std::exception &cleanup_error) {
      if (!error) {
        error = std::current_exception();
      } else {
        std::cerr << "synthesis cleanup failed: "
                  << typeid(cleanup_error).name() << "\n";
      }
    }
    {
      std::lock_guard<std::mutex> lock(mutex_);
      job_.reset();
    }
    if (error) {
      job->completion.set_exception(error);
    } else {
      job->completion.set_value(std::move(*result));
    }
  }
}

void SpeechService::Register(mg_context *context,
                             const std::string &model_name) {
  model_name_ = model_name;
  mg_set_request_handler(
      context, "/v1/audio/speech$",
      [](mg_connection *connection, void *service) {
        return static_cast<SpeechService *>(service)->Handle(connection, false);
      },
      this);
  mg_set_request_handler(
      context, "/generate$",
      [](mg_connection *connection, void *service) {
        return static_cast<SpeechService *>(service)->Handle(connection, true);
      },
      this);
}

int SpeechService::Handle(mg_connection *connection, bool raw_instruction) {
  if (!omni_server::IsPost(connection)) {
    return omni_server::WriteJson(connection, 405,
                                  {{"detail", "Method Not Allowed"}});
  } else {
  }
  ParsedRequest parsed;
  try {
    const std::string body =
        omni_server::ReadBody(connection, kMaximumBodyBytes);
    parsed = ParseRequest(Json::parse(body), raw_instruction, max_seconds_,
                          model_name_);
  } catch (const std::length_error &) {
    return omni_server::BadRequest(connection, "request body exceeds 16 MiB");
  } catch (const Json::exception &) {
    return omni_server::BadRequest(connection,
                                   "invalid request JSON or field type");
  } catch (const std::invalid_argument &error) {
    return omni_server::BadRequest(connection, error.what());
  }
  auto job = std::make_shared<Job>();
  job->request = std::move(parsed.synthesis);
  auto future = job->completion.get_future();
  bool accepted;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    accepted = job_ == nullptr && !stopping_.load();
    if (accepted) {
      job_ = job;
    } else {
    }
  }
  if (!accepted) {
    omni_server::WriteResponse(connection, 429, "Too Many Requests",
                               "application/json",
                               Json({{"detail", "AuK is busy"}}).dump());
    return 429;
  } else {
  }
  wake_.notify_one();
  try {
    const SynthesisResult result = future.get();
    const std::string audio =
        parsed.pcm
            ? qwen3_asr::EncodePcm16(result.samples)
            : qwen3_asr::EncodeWavPcm16(result.samples, result.sample_rate);
    if (raw_instruction) {
      std::string encoded(4 * ((audio.size() + 2) / 3) + 1, '\0');
      size_t length = encoded.size();
      mg_base64_encode(reinterpret_cast<const unsigned char *>(audio.data()),
                       audio.size(), encoded.data(), &length);
      encoded.resize(length - 1);
      return omni_server::WriteJson(
          connection, 200,
          {{"audio",
            {{"data", encoded},
             {"format", "wav"},
             {"sample_rate", result.sample_rate}}},
           {"meta_info", {{"finish_reason", {{"type", "stop"}}}}}});
    } else {
      omni_server::WriteResponse(connection, 200, "OK",
                                 parsed.pcm ? "audio/pcm" : "audio/wav", audio);
      return 200;
    }
  } catch (const SynthesisCancelled &) {
    return omni_server::WriteJson(connection, 503,
                                  {{"detail", "Synthesis cancelled"}});
  } catch (const std::exception &error) {
    std::cerr << "synthesis failed: " << typeid(error).name() << "\n";
    return omni_server::WriteJson(connection, 500,
                                  {{"detail", "Synthesis failed"}});
  }
}

std::map<std::string, int> SpeechService::RequestStates() const {
  std::lock_guard<std::mutex> lock(mutex_);
  if (job_ != nullptr) {
    return {{"running", 1}};
  } else {
    return {};
  }
}

void SpeechService::CancelAll() {
  std::lock_guard<std::mutex> lock(mutex_);
  if (job_ != nullptr) {
    job_->cancelled.store(true);
  } else {
  }
}

} // namespace auk
