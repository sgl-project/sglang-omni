// SPDX-License-Identifier: Apache-2.0
// One realtime socket's event stream, shared by the realtime sessions.
#pragma once

#include <exception>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

#include "common/worker.h"
#include "nlohmann/json.hpp"

namespace omni_server {

// One realtime socket's session: numbered events out, client events in.
// Handle runs on the socket's thread; decodes run on the worker and report
// back through the sender.
class RealtimeConnection
    : public std::enable_shared_from_this<RealtimeConnection> {
public:
  // Writes one serialized event; false once the socket is gone.
  using Sender = std::function<bool(const std::string &)>;

  explicit RealtimeConnection(Sender sender);
  virtual ~RealtimeConnection() = default;

  // Applies one client event; false once the session has completed.
  virtual bool Handle(const nlohmann::json &message) = 0;
  void SendError(const std::string &type, const std::string &code,
                 const std::string &message);
  // The client left: stop any decode in flight and send nothing more.
  void Close();

protected:
  void Send(nlohmann::ordered_json event);
  // The samples of an append event's base64 PCM16 audio; reports anything
  // else to the client and returns nothing.
  std::optional<std::vector<float>>
  AppendedSamples(const nlohmann::json &audio);
  // A session.update's session object when it asks for manual turns; reports
  // anything else to the client and returns nothing.
  std::optional<nlohmann::json>
  ManualTurnSession(const nlohmann::json &message);
  // Reports a failed decode; a cancelled one ended with the client.
  void ReportDecodeFailure(std::exception_ptr error);

  CancelFlag cancel_ = NewCancelFlag();

private:
  Sender sender_;
  std::mutex send_mutex_;
  int event_index_ = 0;
  bool closed_ = false;
};

} // namespace omni_server
