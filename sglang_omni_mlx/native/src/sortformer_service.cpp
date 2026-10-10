// SPDX-License-Identifier: Apache-2.0
#include "sortformer_service.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <typeinfo>

#include "http.h"

namespace sortformer {

namespace {

using omni_server::Json;

// Note (Jiaxin Deng): Voxt's own limit on the cache and FIFO a client asks for.
constexpr int kMaxStateFrames = 4096;

struct SocketState {
  FeedOptions options;
  StreamingState stream;
  std::string fragments;
};

float ParseFloat(const std::string &name, const std::string &value,
                 float minimum, float maximum) {
  size_t parsed = 0;
  const float number = std::stof(value, &parsed);
  if (parsed != value.size() || !std::isfinite(number) || number < minimum ||
      number > maximum) {
    throw std::invalid_argument(name + " is out of range");
  } else {
  }
  return number;
}

int ParseInt(const std::string &name, const std::string &value, int minimum,
             int maximum) {
  size_t parsed = 0;
  const int number = std::stoi(value, &parsed);
  if (parsed != value.size() || number < minimum || number > maximum) {
    throw std::invalid_argument(name + " is out of range");
  } else {
  }
  return number;
}

std::optional<FeedOptions> OptionsOf(const mg_connection *connection,
                                     const SortformerService &service) {
  const char *query = mg_get_request_info(connection)->query_string;
  try {
    const FeedOptions options = ParseFeedOptions(query ? query : "");
    service.CheckStateLimits(options);
    return options;
  } catch (const std::exception &) {
    return std::nullopt;
  }
}

Json ResultJson(const FeedResult &result, const StreamingState &state) {
  const int speakers = result.speaker_count;
  Json probabilities = Json::array();
  for (int frame = 0; frame < result.frame_count; ++frame) {
    Json row = Json::array();
    for (int speaker = 0; speaker < speakers; ++speaker) {
      row.push_back(result.probabilities[static_cast<size_t>(frame) * speakers +
                                         speaker]);
    }
    probabilities.push_back(std::move(row));
  }
  Json segments = Json::array();
  for (const Segment &segment : result.segments) {
    segments.push_back({{"start", segment.start},
                        {"end", segment.end},
                        {"speaker", segment.speaker}});
  }
  return {{"frames", result.frame_count},
          {"speakers", speakers},
          {"probabilities", std::move(probabilities)},
          {"segments", std::move(segments)},
          {"state",
           {{"fifo_length", state.fifo_length()},
            {"spkcache_length", state.spkcache_length()},
            {"frames_processed", state.frames_processed}}}};
}

int SocketConnect(const mg_connection *connection, void *data) {
  // Note (Jiaxin Deng): returning 1 refuses the handshake on invalid options.
  return OptionsOf(connection, *static_cast<SortformerService *>(data))
                 .has_value()
             ? 0
             : 1;
}

void SocketReady(mg_connection *connection, void *data) {
  auto *service = static_cast<SortformerService *>(data);
  service->StreamOpened();
  mg_set_user_connection_data(
      connection,
      new SocketState{OptionsOf(connection, *service).value_or(FeedOptions{}),
                      service->NewStream(),
                      {}});
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
  auto *service = static_cast<SortformerService *>(service_data);
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
  // Note (Jiaxin Deng): checked before buffering so a client cannot grow it.
  if (opcode == MG_WEBSOCKET_OPCODE_TEXT ||
      socket->fragments.size() + length >
          service->MaxFeedSamples() * sizeof(float)) {
    SendText(connection,
             Json({{"error", "A feed is a binary message of at most " +
                                 std::to_string(service->MaxFeedSamples()) +
                                 " samples."}})
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
  const size_t sample_count = message.size() / sizeof(float);
  if (message.size() % sizeof(float) != 0 || sample_count < 2 ||
      sample_count > service->MaxFeedSamples()) {
    SendText(connection,
             Json({{"error", "A feed is 2 to " +
                                 std::to_string(service->MaxFeedSamples()) +
                                 " float32 little-endian samples."}})
                 .dump());
    return 0;
  } else {
  }
  std::vector<float> samples(sample_count);
  std::memcpy(samples.data(), message.data(), message.size());
  for (const float sample : samples) {
    if (!std::isfinite(sample)) {
      SendText(connection, Json({{"error", "Audio must be finite."}}).dump());
      return 0;
    } else {
    }
  }
  try {
    const FeedResult result =
        service->Feed(samples, socket->stream, socket->options);
    return SendText(connection, ResultJson(result, socket->stream).dump()) ? 1
                                                                           : 0;
  } catch (const std::exception &error) {
    // Note (Jiaxin Deng): returning 0 closes the socket; log the type only,
    // never audio.
    std::cerr << "diarization stream failed: " << typeid(error).name() << "\n";
    return 0;
  }
}

void SocketClosed(const mg_connection *connection, void *data) {
  auto *socket =
      static_cast<SocketState *>(mg_get_user_connection_data(connection));
  if (socket != nullptr) {
    delete socket;
    static_cast<SortformerService *>(data)->StreamClosed();
  } else {
  }
}

} // namespace

FeedOptions ParseFeedOptions(const std::string &query) {
  FeedOptions options;
  size_t start = 0;
  while (start < query.size()) {
    const size_t end = std::min(query.find('&', start), query.size());
    const std::string pair = query.substr(start, end - start);
    start = end + 1;
    if (pair.empty()) {
      continue;
    } else {
    }
    const size_t equals = pair.find('=');
    if (equals == std::string::npos) {
      throw std::invalid_argument(pair + " needs a value");
    } else {
    }
    const std::string name = pair.substr(0, equals);
    const std::string value = pair.substr(equals + 1);
    if (name == "threshold") {
      options.threshold = ParseFloat(name, value, 0.0f, 1.0f);
    } else if (name == "min_duration") {
      options.min_duration = ParseFloat(name, value, 0.0f, 3600.0f);
    } else if (name == "merge_gap") {
      options.merge_gap = ParseFloat(name, value, 0.0f, 3600.0f);
    } else if (name == "spkcache_max") {
      options.spkcache_max = ParseInt(name, value, 1, kMaxStateFrames);
    } else if (name == "fifo_max") {
      options.fifo_max = ParseInt(name, value, 0, kMaxStateFrames);
    } else {
      throw std::invalid_argument("unknown option " + name);
    }
  }
  return options;
}

SortformerService::SortformerService(
    const std::filesystem::path &model_directory)
    : model_(model_directory) {
  const ModulesConfig &modules = model_.config().modules;
  const auto samples_per_frame = static_cast<size_t>(
      std::lround(model_.frame_duration() *
                  static_cast<float>(model_.config().processor.sampling_rate)));
  // Note (Jiaxin Deng): N samples give 1 + N / hop mel frames and ceil(mel / 8)
  // after subsampling, so one sample less than update_period frames is exact.
  max_feed_samples_ = std::min(
      kMaxFeedSamples,
      static_cast<size_t>(modules.spkcache_update_period) * samples_per_frame -
          1);
}

void SortformerService::CheckStateLimits(const FeedOptions &options) const {
  const Config &config = model_.config();
  // Note (Jiaxin Deng): compression leaves spkcache_len frames whatever
  // spkcache_max is.
  const int cache_frames =
      std::max(options.spkcache_max, config.modules.spkcache_len);
  const int frames = cache_frames + options.fifo_max +
                     config.modules.chunk_left_context +
                     config.modules.spkcache_update_period;
  if (frames > config.tf_encoder.max_source_positions) {
    throw std::invalid_argument(
        "spkcache_max and fifo_max leave too few encoder positions");
  } else {
  }
}

void SortformerService::Register(mg_context *context) {
  mg_set_websocket_handler(context, "/v1/diarization/stream", SocketConnect,
                           SocketReady, SocketData, SocketClosed, this);
}

std::map<std::string, int> SortformerService::RequestStates() const {
  std::lock_guard<std::mutex> lock(states_mutex_);
  return {{"running", running_requests_}, {"streams", open_streams_}};
}

FeedResult SortformerService::Feed(const std::vector<float> &samples,
                                   StreamingState &state,
                                   const FeedOptions &options) {
  {
    std::lock_guard<std::mutex> lock(states_mutex_);
    ++running_requests_;
  }
  try {
    std::lock_guard<std::mutex> lock(inference_mutex_);
    FeedResult result = model_.Feed(samples, state, options);
    std::lock_guard<std::mutex> states_lock(states_mutex_);
    --running_requests_;
    return result;
  } catch (...) {
    std::lock_guard<std::mutex> lock(states_mutex_);
    --running_requests_;
    throw;
  }
}

void SortformerService::StreamOpened() {
  std::lock_guard<std::mutex> lock(states_mutex_);
  ++open_streams_;
}

void SortformerService::StreamClosed() {
  std::lock_guard<std::mutex> lock(states_mutex_);
  --open_streams_;
}

} // namespace sortformer
