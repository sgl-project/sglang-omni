// SPDX-License-Identifier: Apache-2.0
#include "vad_service.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <typeinfo>

#include "audio.h"
#include "http.h"

namespace silero_vad {

namespace {

using omni_server::Json;

constexpr size_t kMaxMessageBytes = 30 * kSampleRate * sizeof(float);

struct SocketState {
  StreamState stream;
  std::vector<float> pending;
  std::string fragments;
};

int OptionalInt(const std::optional<std::string> &value, int fallback,
                const std::string &name) {
  if (!value.has_value() || value->empty()) {
    return fallback;
  } else {
  }
  size_t parsed = 0;
  const int number = std::stoi(*value, &parsed);
  if (parsed != value->size() || number < 0) {
    throw std::invalid_argument(name + " must be a non-negative integer");
  } else {
  }
  return number;
}

float OptionalProbability(const std::optional<std::string> &value,
                          float fallback, const std::string &name) {
  if (!value.has_value() || value->empty()) {
    return fallback;
  } else {
  }
  size_t parsed = 0;
  const float number = std::stof(*value, &parsed);
  if (parsed != value->size() || !(number >= 0.0f && number <= 1.0f)) {
    throw std::invalid_argument(name + " must be a number in [0, 1]");
  } else {
  }
  return number;
}

int HandleSpeechTimestamps(mg_connection *connection, void *data) {
  auto *service = static_cast<VADService *>(data);
  if (!omni_server::IsPost(connection)) {
    return omni_server::WriteJson(connection, 405,
                                  {{"detail", "Method Not Allowed"}});
  } else {
  }
  const char *content_type = mg_get_header(connection, "Content-Type");
  const auto form = qwen3_asr::ParseMultipartForm(
      content_type ? content_type : "", omni_server::ReadBody(connection));
  if (!form.has_value() || form->count("file") == 0 ||
      !form->at("file").filename.has_value()) {
    return omni_server::BadRequest(connection, "file is required");
  } else {
  }
  std::vector<float> samples;
  TimestampOptions options = service->defaults();
  try {
    samples = qwen3_asr::DecodeWav(form->at("file").value);
    if (!std::all_of(samples.begin(), samples.end(),
                     [](float sample) { return std::isfinite(sample); })) {
      throw std::invalid_argument("Audio must be finite.");
    } else {
    }
    options.threshold = OptionalProbability(
        omni_server::Field(*form, "threshold"), options.threshold, "threshold");
    options.min_speech_ms =
        OptionalInt(omni_server::Field(*form, "min_speech_duration_ms"),
                    options.min_speech_ms, "min_speech_duration_ms");
    options.min_silence_ms =
        OptionalInt(omni_server::Field(*form, "min_silence_duration_ms"),
                    options.min_silence_ms, "min_silence_duration_ms");
    options.speech_pad_ms =
        OptionalInt(omni_server::Field(*form, "speech_pad_ms"),
                    options.speech_pad_ms, "speech_pad_ms");
  } catch (const std::invalid_argument &error) {
    return omni_server::BadRequest(connection, error.what());
  } catch (const std::out_of_range &) {
    return omni_server::BadRequest(connection, "an option is out of range");
  }
  try {
    Json timestamps = Json::array();
    for (const Timestamp &timestamp :
         service->SpeechTimestamps(samples, options)) {
      timestamps.push_back(
          {{"start", timestamp.start}, {"end", timestamp.end}});
    }
    return omni_server::WriteJson(
        connection, 200,
        {{"sample_rate", kSampleRate}, {"timestamps", timestamps}});
  } catch (const std::exception &error) {
    std::cerr << "speech timestamps failed: " << typeid(error).name() << "\n";
    return omni_server::WriteJson(
        connection, 500, {{"detail", "Voice activity detection failed."}});
  }
}

void SocketReady(mg_connection *connection, void *data) {
  static_cast<VADService *>(data)->StreamOpened();
  mg_set_user_connection_data(connection, new SocketState());
}

bool SendText(mg_connection *connection, const std::string &text) {
  mg_lock_connection(connection);
  const int written = mg_websocket_write(connection, MG_WEBSOCKET_OPCODE_TEXT,
                                         text.data(), text.size());
  mg_unlock_connection(connection);
  return written > 0;
}

int SocketData(mg_connection *connection, int bits, char *data, size_t length,
               void *service_data) {
  auto *service = static_cast<VADService *>(service_data);
  auto *socket =
      static_cast<SocketState *>(mg_get_user_connection_data(connection));
  const int opcode = bits & 0x0F;
  if (socket == nullptr || opcode == MG_WEBSOCKET_OPCODE_CONNECTION_CLOSE) {
    return 0;
  } else if (opcode == MG_WEBSOCKET_OPCODE_PING) {
    mg_lock_connection(connection);
    mg_websocket_write(connection, MG_WEBSOCKET_OPCODE_PONG, data, length);
    mg_unlock_connection(connection);
    return 1;
  } else if (opcode == MG_WEBSOCKET_OPCODE_PONG) {
    return 1;
  } else {
  }
  // Note (Jiaxin Deng): checked before buffering, so a stream holds <= 30 s.
  if (opcode == MG_WEBSOCKET_OPCODE_TEXT ||
      socket->fragments.size() + length > kMaxMessageBytes) {
    SendText(connection,
             Json({{"error", "Audio must be binary messages of at most 30 s."}})
                 .dump());
    return 0;
  } else {
  }
  socket->fragments.append(data, length);
  if ((bits & 0x80) == 0) {
    return 1;
  } else {
  }
  const std::string message = std::move(socket->fragments);
  socket->fragments.clear();
  if (message.size() % sizeof(float) != 0) {
    SendText(connection,
             Json({{"error", "Audio must be float32 little-endian samples."}})
                 .dump());
    return 0;
  } else {
  }
  const size_t sample_count = message.size() / sizeof(float);
  const size_t previous = socket->pending.size();
  socket->pending.resize(previous + sample_count);
  std::memcpy(socket->pending.data() + previous, message.data(),
              message.size());
  for (size_t index = previous; index < socket->pending.size(); ++index) {
    if (!std::isfinite(socket->pending[index])) {
      SendText(connection, Json({{"error", "Audio must be finite."}}).dump());
      return 0;
    } else {
    }
  }
  try {
    const std::vector<float> probabilities =
        service->FeedSamples(socket->pending, socket->stream);
    const size_t consumed = probabilities.size() * kChunkSamples;
    socket->pending.erase(socket->pending.begin(),
                          socket->pending.begin() +
                              static_cast<std::ptrdiff_t>(consumed));
    const Json reply = {{"probability", probabilities.empty()
                                            ? Json()
                                            : Json(probabilities.back())}};
    return SendText(connection, reply.dump()) ? 1 : 0;
  } catch (const std::exception &error) {
    // Note (Jiaxin Deng): only the exception type is logged, never audio.
    std::cerr << "voice activity stream failed: " << typeid(error).name()
              << "\n";
    return 0;
  }
}

void SocketClosed(const mg_connection *connection, void *data) {
  auto *socket =
      static_cast<SocketState *>(mg_get_user_connection_data(connection));
  if (socket != nullptr) {
    delete socket;
    static_cast<VADService *>(data)->StreamClosed();
  } else {
  }
}

} // namespace

VADService::VADService(const std::filesystem::path &model_directory)
    : vad_(model_directory) {}

void VADService::Register(mg_context *context) {
  mg_set_request_handler(context, "/v1/vad/speech_timestamps$",
                         HandleSpeechTimestamps, this);
  mg_set_websocket_handler(context, "/v1/vad/stream", nullptr, SocketReady,
                           SocketData, SocketClosed, this);
}

std::map<std::string, int> VADService::RequestStates() const {
  std::lock_guard<std::mutex> lock(states_mutex_);
  return {{"running", running_requests_}, {"streams", open_streams_}};
}

std::vector<float> VADService::FeedSamples(const std::vector<float> &samples,
                                           StreamState &state) {
  std::vector<float> probabilities;
  std::lock_guard<std::mutex> lock(inference_mutex_);
  for (size_t offset = 0; offset + kChunkSamples <= samples.size();
       offset += kChunkSamples) {
    probabilities.push_back(vad_.Feed(samples.data() + offset, state));
  }
  return probabilities;
}

std::vector<Timestamp>
VADService::SpeechTimestamps(const std::vector<float> &samples,
                             const TimestampOptions &options) {
  {
    std::lock_guard<std::mutex> lock(states_mutex_);
    ++running_requests_;
  }
  std::vector<float> probabilities;
  try {
    std::lock_guard<std::mutex> lock(inference_mutex_);
    probabilities = vad_.PredictProbabilities(samples);
  } catch (...) {
    std::lock_guard<std::mutex> lock(states_mutex_);
    --running_requests_;
    throw;
  }
  {
    std::lock_guard<std::mutex> lock(states_mutex_);
    --running_requests_;
  }
  return ProbabilitiesToTimestamps(probabilities,
                                   static_cast<long>(samples.size()), options);
}

void VADService::StreamOpened() {
  std::lock_guard<std::mutex> lock(states_mutex_);
  ++open_streams_;
}

void VADService::StreamClosed() {
  std::lock_guard<std::mutex> lock(states_mutex_);
  --open_streams_;
}

} // namespace silero_vad
