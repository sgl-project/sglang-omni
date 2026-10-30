// SPDX-License-Identifier: Apache-2.0
#include "qwen3_asr/asr_service.h"

#include <chrono>
#include <cmath>
#include <cstring>
#include <future>
#include <iostream>
#include <stdexcept>
#include <typeinfo>

#include "civetweb.h"
#include "common/audio.h"
#include "common/http.h"
#include "common/service.h"
#include "nlohmann/json.hpp"
#include "qwen3_asr/worker.h"

namespace asr_service {

namespace {

namespace mx = mlx::core;
using omni_server::BadRequest;
using omni_server::Json;
using omni_server::ReadBody;
using omni_server::WriteJson;
using qwen3_asr::CancelFlag;
using qwen3_asr::TranscriptionResult;
using qwen3_asr::TranscriptionWorker;

constexpr auto kHeartbeatInterval = std::chrono::milliseconds(250);

struct ServerState {
  TranscriptionWorker *worker = nullptr;
  const ServedModel *model = nullptr;
  std::string model_name;
};

// Speaker segments, for models that produce them.
void AddSegments(Json &body, const TranscriptionResult &result) {
  if (!result.segments.empty()) {
    body["segments"] = qwen3_asr::SpeakerSegmentsJson(result.segments);
  } else {
  }
}

Json DoneEvent(const TranscriptionResult &result,
               bool include_generation_metadata) {
  Json event = {{"type", "transcript.text.done"}, {"text", result.text}};
  AddSegments(event, result);
  if (include_generation_metadata) {
    event["generation_metadata"] = {
        {"generated_token_count", result.generated_token_count},
        {"language",
         result.language.has_value() ? Json(*result.language) : Json()},
        {"finish_reason", qwen3_asr::FinishReasonName(result.finish_reason)}};
  } else {
  }
  return event;
}

const Json &FailureEvent() {
  static const Json event = {{"type", "error"},
                             {"error",
                              {{"type", "server_error"},
                               {"code", "transcription_failed"},
                               {"message", "Transcription failed."}}}};
  return event;
}

bool WriteSse(mg_connection *connection, const std::string &payload) {
  const std::string line = "data: " + payload + "\n\n";
  return mg_write(connection, line.data(), line.size()) > 0;
}

int HandleTranscriptions(mg_connection *connection, void *data) {
  const auto *state = static_cast<ServerState *>(data);
  if (!omni_server::IsPost(connection)) {
    return WriteJson(connection, 405, {{"detail", "Method Not Allowed"}});
  } else {
  }
  const char *content_type = mg_get_header(connection, "Content-Type");
  const auto form = omni_common::ParseMultipartForm(
      content_type ? content_type : "", ReadBody(connection));
  if (!form.has_value() || form->count("file") == 0 ||
      !form->at("file").filename.has_value()) {
    return BadRequest(connection, "file is required");
  } else {
  }
  Transcription transcription;
  try {
    transcription = state->model->Prepare(
        omni_common::DecodeWav(form->at("file").value), *form);
  } catch (const std::invalid_argument &error) {
    return BadRequest(connection, error.what());
  } catch (const std::out_of_range &) {
    return BadRequest(connection, "a numeric field is out of range");
  }
  const bool stream = omni_common::FormFlag(TextField(*form, "stream"));
  const bool include_generation_metadata =
      omni_common::FormFlag(TextField(*form, "include_generation_metadata"));
  if (include_generation_metadata && !stream) {
    return BadRequest(connection,
                      "include_generation_metadata requires stream=true");
  } else {
  }
  const CancelFlag cancel = qwen3_asr::NewCancelFlag();
  auto promise = std::make_shared<std::promise<TranscriptionResult>>();
  std::future<TranscriptionResult> future = promise->get_future();
  state->worker->Submit(std::move(transcription), cancel,
                        [promise](std::optional<TranscriptionResult> result,
                                  std::exception_ptr error) {
                          if (error) {
                            promise->set_exception(error);
                          } else {
                            promise->set_value(std::move(*result));
                          }
                        });
  if (!stream) {
    try {
      const TranscriptionResult result = future.get();
      Json body = {{"text", result.text}};
      AddSegments(body, result);
      return WriteJson(connection, 200, body);
    } catch (...) {
      return WriteJson(connection, 500, {{"detail", "Transcription failed."}});
    }
  } else {
  }
  mg_printf(connection, "HTTP/1.1 200 OK\r\nContent-Type: "
                        "text/event-stream\r\nCache-Control: no-cache\r\n"
                        "Connection: close\r\n\r\n");
  // Note (Jiaxin Deng): SSE comment heartbeats fail to write once the peer is
  // gone, which cancels the decode it was waiting for.
  while (future.wait_for(kHeartbeatInterval) != std::future_status::ready) {
    if (mg_write(connection, ":\n\n", 3) <= 0) {
      cancel->store(true);
    } else {
    }
  }
  try {
    const TranscriptionResult result = future.get();
    WriteSse(connection, DoneEvent(result, include_generation_metadata).dump());
  } catch (const qwen3_asr::TranscriptionCancelled &) {
    return 200;
  } catch (const std::exception &error) {
    std::cerr << "transcription failed: " << typeid(error).name() << "\n";
    WriteSse(connection, FailureEvent().dump());
  }
  WriteSse(connection, "[DONE]");
  return 200;
}

struct SocketState {
  std::shared_ptr<qwen3_asr::RealtimeConnection> session;
  std::string fragments;
};

void SocketReady(mg_connection *connection, void *data) {
  const auto *factory = static_cast<const RealtimeFactory *>(data);
  auto *socket = new SocketState();
  socket->session = (*factory)([connection](const std::string &text) {
    mg_lock_connection(connection);
    const int written = mg_websocket_write(connection, MG_WEBSOCKET_OPCODE_TEXT,
                                           text.data(), text.size());
    mg_unlock_connection(connection);
    return written > 0;
  });
  mg_set_user_connection_data(connection, socket);
}

int SocketData(mg_connection *connection, int bits, char *data, size_t length,
               void *) {
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
  socket->fragments.append(data, length);
  if ((bits & 0x80) == 0) {
    return 1;
  } else {
  }
  const std::string message = std::move(socket->fragments);
  socket->fragments.clear();
  nlohmann::json event;
  try {
    event = nlohmann::json::parse(message);
  } catch (const nlohmann::json::exception &) {
    socket->session->SendError("invalid_request_error", "invalid_json",
                               "Events must be JSON.");
    return 1;
  }
  if (!event.is_object()) {
    return 0;
  } else {
  }
  try {
    return socket->session->Handle(event) ? 1 : 0;
  } catch (const std::exception &error) {
    // Note (Jiaxin Deng): log the type alone, never audio or text.
    std::cerr << "realtime session failed: " << typeid(error).name() << "\n";
    return 0;
  }
}

void SocketClosed(const mg_connection *connection, void *) {
  auto *socket =
      static_cast<SocketState *>(mg_get_user_connection_data(connection));
  if (socket != nullptr) {
    socket->session->Close();
    delete socket;
  } else {
  }
}

class AsrService : public omni_server::Service {
public:
  AsrService(const std::filesystem::path &directory, const ModelLoader &load) {
    worker_ = std::make_unique<TranscriptionWorker>(
        [&]() { model_ = load(directory); });
  }

  void SetModelName(const std::string &model_name,
                    bool has_chosen_model_name) override {
    model_->set_model_name(model_name, has_chosen_model_name);
  }

  void Register(mg_context *context, const std::string &model_name) override {
    state_ = {worker_.get(), model_.get(), model_name};
    if (model_->Transcribes()) {
      mg_set_request_handler(context, "/v1/audio/transcriptions$",
                             HandleTranscriptions, &state_);
    } else {
    }
    model_->AddHandlers(context, *worker_);
  }

  std::map<std::string, int> RequestStates() const override {
    return model_->RequestStates(*worker_);
  }

  void CancelAll() override { worker_->CancelAll(); }

private:
  std::unique_ptr<ServedModel> model_;
  std::unique_ptr<TranscriptionWorker> worker_;
  ServerState state_;
};

} // namespace

std::optional<std::string> TextField(const FormFields &form,
                                     const std::string &name) {
  return omni_server::Field(form, name);
}

std::optional<int> IntegerField(const FormFields &form,
                                const std::string &name) {
  const std::optional<std::string> text = TextField(form, name);
  if (!text.has_value() || text->empty()) {
    return std::nullopt;
  } else {
  }
  size_t parsed = 0;
  int value = 0;
  try {
    value = std::stoi(*text, &parsed);
  } catch (const std::logic_error &) {
    // Note (khazic): parsed stays 0, so the check below names the field.
  }
  if (parsed != text->size()) {
    throw std::invalid_argument(name + " must be an integer");
  } else {
  }
  return value;
}

std::optional<float> NumberField(const FormFields &form,
                                 const std::string &name) {
  const std::optional<std::string> text = TextField(form, name);
  if (!text.has_value() || text->empty()) {
    return std::nullopt;
  } else {
  }
  size_t parsed = 0;
  float value = 0;
  try {
    value = std::stof(*text, &parsed);
  } catch (const std::logic_error &) {
    // Note (khazic): parsed stays 0, so the check below names the field.
  }
  if (parsed != text->size() || !std::isfinite(value)) {
    throw std::invalid_argument(name + " must be a finite number");
  } else {
  }
  return value;
}

void AddRealtimeHandler(mg_context *context, const RealtimeFactory &factory) {
  mg_set_websocket_handler(context, "/v1/realtime", nullptr, SocketReady,
                           SocketData, SocketClosed,
                           const_cast<RealtimeFactory *>(&factory));
}

int Serve(int argc, char **argv, const std::vector<ServedKind> &kinds) {
  std::vector<omni_server::ServedKind> services;
  for (const auto &kind : kinds) {
    services.push_back({
        kind.model_kind,
        [load = kind.load](const std::filesystem::path &directory) {
          return std::make_unique<AsrService>(directory, load);
        },
        kind.flags,
        kind.check_flags,
    });
  }
  return omni_server::Serve(argc, argv, services);
}

} // namespace asr_service
