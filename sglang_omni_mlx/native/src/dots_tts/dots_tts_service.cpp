// SPDX-License-Identifier: Apache-2.0
#include "dots_tts/dots_tts_service.h"

#include <algorithm>
#include <array>
#include <cerrno>
#include <charconv>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <iostream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <system_error>
#include <typeinfo>
#include <utility>
#include <vector>

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include "common/audio.h"
#include "common/base64.h"
#include "common/http.h"
#include "dots_tts/dots_tts_model.h"
#include "nlohmann/json.hpp"

namespace dots_tts {

namespace {

constexpr int kDefaultFlowMatchingSteps = 10;
constexpr int kDefaultMeanFlowSteps = 4;
constexpr auto kClientPollInterval = std::chrono::milliseconds(50);
constexpr int kClientLeftStatus = 499;
constexpr size_t kMaxUnreadBytesPerCheck = 64 * 1024;

int ClientSocket(const mg_connection *connection) {
  const mg_request_info *request_info = mg_get_request_info(connection);
  std::error_code listing_error;
  for (std::filesystem::directory_iterator listing("/dev/fd", listing_error);
       !listing_error && listing != std::filesystem::directory_iterator();
       listing.increment(listing_error)) {
    const std::string name = listing->path().filename().string();
    int file_descriptor = -1;
    const auto [parse_end, parse_error] = std::from_chars(
        name.data(), name.data() + name.size(), file_descriptor);
    if (parse_error != std::errc() || parse_end != name.data() + name.size()) {
      continue;
    }
    sockaddr_storage peer{};
    socklen_t peer_size = sizeof(peer);
    char address[INET6_ADDRSTRLEN] = "";
    int port = -1;
    if (getpeername(file_descriptor, reinterpret_cast<sockaddr *>(&peer),
                    &peer_size) != 0) {
      continue;
    } else if (peer.ss_family == AF_INET) {
      const auto *ipv4 = reinterpret_cast<const sockaddr_in *>(&peer);
      inet_ntop(AF_INET, &ipv4->sin_addr, address, sizeof(address));
      port = ntohs(ipv4->sin_port);
    } else if (peer.ss_family == AF_INET6) {
      const auto *ipv6 = reinterpret_cast<const sockaddr_in6 *>(&peer);
      inet_ntop(AF_INET6, &ipv6->sin6_addr, address, sizeof(address));
      port = ntohs(ipv6->sin6_port);
    }
    if (port == request_info->remote_port &&
        request_info->remote_addr == std::string(address)) {
      return file_descriptor;
    }
  }
  return -1;
}

bool ClientLeft(int client_socket) {
  if (client_socket < 0) {
    return false;
  }
  std::array<char, 4096> unread{};
  size_t dropped_bytes = 0;
  while (dropped_bytes < kMaxUnreadBytesPerCheck) {
    const ssize_t received =
        recv(client_socket, unread.data(), unread.size(), MSG_DONTWAIT);
    if (received == 0) {
      return true;
    } else if (received < 0) {
      return errno != EAGAIN && errno != EINTR;
    }
    dropped_bytes += static_cast<size_t>(received);
  }
  return false;
}

std::string SynthesizeBody(DotsTtsModel &model, const std::string &body) {
  nlohmann::json request;
  try {
    request = nlohmann::json::parse(body);
  } catch (const nlohmann::json::exception &) {
    throw std::invalid_argument("request body must be JSON");
  }
  if (request.contains("model") && !request.at("model").is_string()) {
    throw std::invalid_argument("model must be a string");
  }
  const std::string text = request.value("input", std::string());
  if (text.empty()) {
    throw std::invalid_argument("input (text to synthesize) is required");
  }

  GenerationOptions options;
  try {
    options.num_steps = request.contains("num_steps")
                            ? request.at("num_steps").get<int>()
                            : (model.is_meanflow() ? kDefaultMeanFlowSteps
                                                   : kDefaultFlowMatchingSteps);
    options.guidance_scale =
        request.value("guidance_scale", options.guidance_scale);
    options.speaker_scale =
        request.value("speaker_scale", options.speaker_scale);
    options.max_generate_length =
        request.value("max_generate_length", options.max_generate_length);
    options.eos_threshold =
        request.value("eos_threshold", options.eos_threshold);
    options.seed = request.value("seed", options.seed);
    if (request.contains("language") && request.at("language").is_string()) {
      options.language = request.at("language").get<std::string>();
    }
  } catch (const nlohmann::json::exception &error) {
    // note (guozhihao-224): a wrong-typed field surfaces as a 400, not a 500.
    throw std::invalid_argument(std::string("bad request field: ") +
                                error.what());
  }
  if (options.num_steps <= 0 || options.guidance_scale < 0 ||
      options.speaker_scale < 0 || options.max_generate_length <= 0) {
    throw std::invalid_argument(
        "num_steps and max_generate_length must be positive; "
        "guidance_scale and speaker_scale must be nonnegative");
  }

  std::optional<std::string> prompt_wav;
  std::optional<std::string> prompt_text;
  if (request.contains("references")) {
    const nlohmann::json &references = request.at("references");
    if (!references.is_array()) {
      throw std::invalid_argument("references must be an array");
    }
    if (references.size() > 1) {
      throw std::invalid_argument(
          "dots.tts supports exactly one reference (voice cloning), got " +
          std::to_string(references.size()));
    }
    if (!references.empty()) {
      const nlohmann::json &reference = references.at(0);
      if (!reference.is_object() || !reference.contains("audio_content") ||
          !reference.at("audio_content").is_string() ||
          !reference.contains("text") || !reference.at("text").is_string()) {
        throw std::invalid_argument(
            "references[0] needs audio_content (base64 WAV) and text");
      }
      prompt_text = reference.at("text").get<std::string>();
      const std::vector<uint8_t> bytes = omni_common::DecodeBase64(
          reference.at("audio_content").get<std::string>(), /*strict=*/true);
      prompt_wav = std::string(bytes.begin(), bytes.end());
    }
  }

  const SynthesisResult result =
      model.Generate(text, prompt_wav, prompt_text, options);
  return qwen3_asr::EncodeWav(result.audio, result.sample_rate);
}

int HandleSpeech(mg_connection *connection, void *data) {
  auto *service = static_cast<DotsTtsService *>(data);
  if (!omni_server::IsPost(connection)) {
    return omni_server::WriteJson(connection, 405,
                                  {{"detail", "Method Not Allowed"}});
  }
  const std::string body = omni_server::ReadBody(connection);
  service->RequestStarted();
  const auto [status, payload] = service->Synthesize(connection, body);
  service->RequestFinished();
  if (status == 200) {
    omni_server::WriteResponse(connection, 200, "OK", "audio/wav", payload);
    return 200;
  }
  if (status == kClientLeftStatus) {
    return kClientLeftStatus;
  }
  return omni_server::WriteJson(connection, status, {{"detail", payload}});
}

} // namespace

DotsTtsService::DotsTtsService(const std::filesystem::path &model_directory,
                               int max_queued_requests,
                               double request_timeout_seconds)
    : model_directory_(model_directory),
      max_queued_requests_(max_queued_requests),
      request_timeout_(
          std::chrono::duration_cast<std::chrono::steady_clock::duration>(
              std::chrono::duration<double>(request_timeout_seconds))) {
  // note (guozhihao-224): the model loads on the worker thread (MLX requires
  // load and run on one thread) before the constructor returns, so the server
  // reports ready only after the weights are actually in memory.
  worker_ = std::thread([this] { RunWorker(); });
  try {
    loaded_.get_future().get();
  } catch (...) {
    worker_.join();
    throw;
  }
}

DotsTtsService::~DotsTtsService() {
  {
    std::lock_guard<std::mutex> lock(jobs_mutex_);
    shutdown_ = true;
  }
  jobs_cv_.notify_all();
  worker_.join();
}

void DotsTtsService::Register(mg_context *context) {
  mg_set_request_handler(context, "/v1/audio/speech$", HandleSpeech, this);
}

std::pair<int, std::string>
DotsTtsService::Synthesize(mg_connection *connection, const std::string &body) {
  {
    std::lock_guard<std::mutex> lock(admission_mutex_);
    if (admitted_requests_ >= max_queued_requests_) {
      return {503, "The request queue is full."};
    }
    admitted_requests_++;
  }
  auto job = std::make_shared<Job>();
  job->cancelled = std::make_shared<std::atomic<bool>>(false);
  job->body = body;
  {
    std::lock_guard<std::mutex> lock(jobs_mutex_);
    jobs_.push_back(job);
  }
  jobs_cv_.notify_one();

  const int client_socket = ClientSocket(connection);
  const auto deadline = std::chrono::steady_clock::now() + request_timeout_;
  bool timed_out = false;
  bool client_left = false;
  {
    std::unique_lock<std::mutex> lock(jobs_mutex_);
    while (!job->finished) {
      const auto poll_at = std::min(deadline, std::chrono::steady_clock::now() +
                                                  kClientPollInterval);
      if (jobs_cv_.wait_until(lock, poll_at, [&] { return job->finished; })) {
        break;
      }
      if (std::chrono::steady_clock::now() >= deadline) {
        job->cancelled->store(true);
        timed_out = true;
        break;
      }
      client_left = ClientLeft(client_socket);
      if (client_left) {
        job->cancelled->store(true);
        break;
      }
    }
  }
  if (client_left) {
    return {kClientLeftStatus, ""};
  }
  if (timed_out) {
    return {504, "Speech generation timed out."};
  }
  if (job->error) {
    try {
      std::rethrow_exception(job->error);
    } catch (const std::invalid_argument &error) {
      return {400, error.what()};
    } catch (const std::exception &error) {
      std::cerr << "dots.tts synthesis failed: " << typeid(error).name()
                << "\n";
      return {500, "Synthesis failed."};
    }
  }
  return {200, std::move(job->wav)};
}

void DotsTtsService::RunWorker() {
  std::optional<DotsTtsModel> model;
  try {
    model.emplace(model_directory_);
    loaded_.set_value();
  } catch (const std::exception &) {
    loaded_.set_exception(std::current_exception());
    return;
  }
  for (;;) {
    std::shared_ptr<Job> job;
    {
      std::unique_lock<std::mutex> lock(jobs_mutex_);
      jobs_cv_.wait(lock, [&] { return shutdown_ || !jobs_.empty(); });
      if (shutdown_ && jobs_.empty()) {
        return;
      }
      job = jobs_.front();
      jobs_.pop_front();
    }
    if (job->cancelled->load()) {
      {
        std::lock_guard<std::mutex> lock(admission_mutex_);
        admitted_requests_--;
      }
      job->finished = true;
      jobs_cv_.notify_all();
      continue;
    }
    try {
      job->wav = SynthesizeBody(*model, job->body);
    } catch (const std::exception &) {
      job->error = std::current_exception();
    }
    {
      std::lock_guard<std::mutex> lock(admission_mutex_);
      admitted_requests_--;
    }
    {
      std::lock_guard<std::mutex> lock(jobs_mutex_);
      job->finished = true;
    }
    jobs_cv_.notify_all();
  }
}

void DotsTtsService::RequestStarted() {
  std::lock_guard<std::mutex> lock(states_mutex_);
  running_requests_++;
}

void DotsTtsService::RequestFinished() {
  std::lock_guard<std::mutex> lock(states_mutex_);
  running_requests_--;
  completed_requests_++;
}

std::map<std::string, int> DotsTtsService::RequestStates() const {
  std::lock_guard<std::mutex> lock(states_mutex_);
  std::map<std::string, int> states = {{"running", running_requests_},
                                       {"completed", completed_requests_}};
  std::lock_guard<std::mutex> admission_lock(admission_mutex_);
  if (admitted_requests_ > 1) {
    states["queued"] = admitted_requests_ - 1;
  }
  return states;
}

} // namespace dots_tts
