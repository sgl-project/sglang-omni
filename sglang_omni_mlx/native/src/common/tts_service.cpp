// SPDX-License-Identifier: Apache-2.0
#include "common/tts_service.h"

#include <iostream>
#include <limits>
#include <typeinfo>

#include "common/audio.h"
#include "common/http.h"
#include "mlx/mlx.h"

namespace tts_service {

SynthesisWorker::SynthesisWorker(std::function<void()> load,
                                 std::function<void()> unload) {
  std::promise<void> loaded;
  auto ready = loaded.get_future();
  thread_ = std::thread(&SynthesisWorker::Run, this, std::move(load),
                        std::move(unload), std::move(loaded));
  try {
    ready.get();
  } catch (...) {
    thread_.join();
    throw;
  }
}

SynthesisWorker::~SynthesisWorker() {
  Stop();
  thread_.join();
}

void SynthesisWorker::Run(std::function<void()> load,
                          std::function<void()> unload,
                          std::promise<void> loaded) {
  try {
    load();
    loaded.set_value();
  } catch (...) {
    unload();
    loaded.set_exception(std::current_exception());
    return;
  }
  while (true) {
    std::shared_ptr<Task> job;
    {
      std::unique_lock<std::mutex> lock(mutex_);
      wake_.wait(lock, [&] { return stopping_ || job_ != nullptr; });
      if (job_ == nullptr) {
        break;
      } else {
        job = job_;
      }
    }
    std::exception_ptr error;
    try {
      const auto check_cancelled = [&] {
        if (job->cancelled.load()) {
          throw SynthesisCancelled();
        } else {
        }
      };
      check_cancelled();
      job->Synthesize(check_cancelled);
      check_cancelled();
    } catch (const SynthesisCancelled &) {
      std::cerr << "synthesis cancelled\n";
      error = std::current_exception();
    } catch (...) {
      error = std::current_exception();
    }
    try {
      mlx::core::synchronize();
      mlx::core::clear_cache();
    } catch (const std::exception &cleanup_error) {
      std::cerr << "synthesis cleanup failed: " << typeid(cleanup_error).name()
                << '\n';
      error = std::current_exception();
    }
    {
      std::lock_guard<std::mutex> lock(mutex_);
      job_.reset();
    }
    job->Complete(error);
  }
  unload();
}

SynthesisWorker::Admission
SynthesisWorker::SubmitTask(const std::shared_ptr<Task> &job) {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (stopping_) {
      return Admission::Stopping;
    } else if (job_ != nullptr) {
      return Admission::Busy;
    } else {
      job_ = job;
    }
  }
  wake_.notify_one();
  return Admission::Accepted;
}

std::map<std::string, int> SynthesisWorker::RequestStates() const {
  std::lock_guard<std::mutex> lock(mutex_);
  if (job_ != nullptr) {
    return {{"running", 1}};
  } else {
    return {};
  }
}

void SynthesisWorker::Stop() {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    stopping_ = true;
    if (job_ != nullptr) {
      job_->cancelled.store(true);
    } else {
    }
  }
  wake_.notify_all();
}

void AudioResponse::WritePcm(std::string_view pcm, int sample_rate,
                             int channels) {
  const bool streaming = format_ == AudioFormat::StreamingPcm;
  if (sample_rate <= 0 || channels <= 0 ||
      pcm.size() % (sizeof(int16_t) * static_cast<size_t>(channels)) != 0) {
    throw std::invalid_argument("invalid PCM16 audio metadata or frame size");
  } else if (headers_sent_ && (!streaming || sample_rate != sample_rate_ ||
                               channels != channels_)) {
    throw std::logic_error("audio response format changed after headers");
  } else if (streaming &&
             pcm.size() > std::numeric_limits<unsigned int>::max()) {
    throw std::length_error("PCM chunk is too large");
  } else if (streaming && pcm.empty()) {
    return;
  } else {
  }
  std::string encoded;
  std::string_view body = pcm;
  const char *content_type = "audio/pcm";
  if (format_ == AudioFormat::Wav || format_ == AudioFormat::JsonWav) {
    encoded = omni_common::EncodeWavPcm16(pcm, sample_rate, channels);
    content_type = "audio/wav";
    if (format_ == AudioFormat::JsonWav) {
      std::string base64(4 * ((encoded.size() + 2) / 3) + 1, '\0');
      size_t length = base64.size();
      mg_base64_encode(reinterpret_cast<const unsigned char *>(encoded.data()),
                       encoded.size(), base64.data(), &length);
      base64.resize(length - 1);
      encoded = omni_server::Json(
                    {{"audio",
                      {{"data", base64},
                       {"format", "wav"},
                       {"sample_rate", sample_rate}}},
                     {"meta_info", {{"finish_reason", {{"type", "stop"}}}}}})
                    .dump();
      content_type = "application/json";
    } else {
    }
    body = encoded;
  } else {
  }
  if (!headers_sent_) {
    const std::string framing =
        streaming ? "Transfer-Encoding: chunked\r\n"
                  : "Content-Length: " + std::to_string(body.size()) + "\r\n";
    sample_rate_ = sample_rate;
    channels_ = channels;
    // Note (Adrian Wei): Header writes may fail after sending a prefix.
    headers_sent_ = true;
    if (mg_printf(connection_,
                  "HTTP/1.1 200 OK\r\nContent-Type: %s\r\n%s"
                  "Connection: close\r\nX-Sample-Rate: %d\r\n"
                  "X-Channels: %d\r\nX-Bit-Depth: 16\r\n\r\n",
                  content_type, framing.c_str(), sample_rate, channels) <= 0) {
      throw std::runtime_error("failed to write speech headers");
    } else {
    }
  } else {
  }
  if (streaming) {
    if (mg_send_chunk(connection_, body.data(),
                      static_cast<unsigned int>(body.size())) < 0) {
      throw std::runtime_error("failed to write speech chunk");
    } else {
    }
  } else if (mg_write(connection_, body.data(), body.size()) !=
             static_cast<int64_t>(body.size())) {
    throw std::runtime_error("failed to write speech audio");
  } else {
  }
}

int AudioResponse::Finish() {
  if (!headers_sent_) {
    return Error(500, "Synthesis produced no audio");
  } else if (format_ == AudioFormat::StreamingPcm &&
             mg_send_chunk(connection_, "", 0) < 0) {
    throw std::runtime_error("failed to finish speech stream");
  } else {
    return 200;
  }
}

int AudioResponse::Error(int status, const std::string &detail) {
  if (headers_sent_) {
    return 200;
  } else {
    return omni_server::WriteJson(connection_, status, {{"detail", detail}});
  }
}

} // namespace tts_service
