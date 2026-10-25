// SPDX-License-Identifier: Apache-2.0
#include "music3/music3_service.h"

#include <iostream>

#include "common/http.h"

namespace music3 {

SpeechService::SpeechService(const std::filesystem::path &directory) {
  std::promise<void> loaded;
  auto ready = loaded.get_future();
  thread_ =
      std::thread(&SpeechService::Run, this, directory, std::move(loaded));
  try {
    ready.get();
  } catch (...) {
    thread_.join();
    throw;
  }
}

SpeechService::~SpeechService() {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    stopping_ = true;
    if (job_ != nullptr) {
      job_->cancelled.store(true);
    } else {
    }
  }
  wake_.notify_all();
  thread_.join();
}

void SpeechService::Run(const std::filesystem::path &directory,
                        std::promise<void> loaded) {
  try {
    model_ = std::make_unique<Model>(directory);
    loaded.set_value();
  } catch (...) {
    loaded.set_exception(std::current_exception());
    return;
  }
  while (true) {
    std::shared_ptr<Job> job;
    {
      std::unique_lock<std::mutex> lock(mutex_);
      wake_.wait(lock, [&] { return stopping_ || job_ != nullptr; });
      if (stopping_ && job_ == nullptr) {
        return;
      } else {
        job = job_;
      }
    }
    std::optional<Result> result;
    std::exception_ptr error;
    try {
      CheckCancelled(job->cancelled);
      job->request =
          ParseRequest(nlohmann::json::parse(job->body), model_->tokenizer(),
                       has_chosen_model_name_ ? model_name_ : std::string(),
                       model_->context_tokens());
      CheckCancelled(job->cancelled);
      result = model_->Synthesize(job->request, job->cancelled);
      CheckCancelled(job->cancelled);
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
        std::cerr << "Music3 cleanup failed: " << cleanup_error.what() << "\n";
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

void SpeechService::SetModelName(const std::string &model_name, bool chosen) {
  model_name_ = model_name;
  has_chosen_model_name_ = chosen;
}

void SpeechService::Register(mg_context *context,
                             const std::string &model_name) {
  model_name_ = model_name;
  mg_set_request_handler(
      context, "/v1/audio/speech$",
      [](mg_connection *connection, void *service) {
        return static_cast<SpeechService *>(service)->Handle(connection);
      },
      this);
}

int SpeechService::Handle(mg_connection *connection) {
  if (!omni_server::IsPost(connection)) {
    return omni_server::WriteJson(connection, 405,
                                  {{"detail", "Method Not Allowed"}});
  } else {
  }
  auto job = std::make_shared<Job>();
  std::optional<std::string> body = omni_server::ReadBody(connection, 1 << 20);
  if (!body.has_value()) {
    return 413;
  } else {
  }
  job->body = std::move(*body);
  auto future = job->completion.get_future();
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (job_ != nullptr || stopping_) {
      omni_server::WriteResponse(
          connection, 429, "Too Many Requests", "application/json",
          omni_server::Json({{"detail", "Music3 is busy"}}).dump());
      return 429;
    } else {
      job_ = job;
    }
  }
  wake_.notify_one();
  try {
    const Result result = future.get();
    omni_server::WriteResponse(connection, 200, "OK",
                               job->request.pcm ? "audio/pcm" : "audio/wav",
                               EncodeAudio(result.samples, job->request.pcm));
    return 200;
  } catch (const nlohmann::json::exception &) {
    return omni_server::BadRequest(
        connection, "invalid request JSON, missing field or field type");
  } catch (const std::invalid_argument &error) {
    return omni_server::BadRequest(connection, error.what());
  } catch (const Cancelled &) {
    return omni_server::WriteJson(connection, 503,
                                  {{"detail", "Music3 generation cancelled"}});
  } catch (const std::exception &error) {
    std::cerr << "Music3 generation failed: " << error.what() << "\n";
    return omni_server::WriteJson(connection, 500,
                                  {{"detail", "Music3 generation failed"}});
  }
}

std::map<std::string, int> SpeechService::RequestStates() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return job_ != nullptr ? std::map<std::string, int>{{"running", 1}}
                         : std::map<std::string, int>{};
}

void SpeechService::CancelAll() {
  std::lock_guard<std::mutex> lock(mutex_);
  if (job_ != nullptr) {
    job_->cancelled.store(true);
  } else {
  }
}

} // namespace music3
