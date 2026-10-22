// SPDX-License-Identifier: Apache-2.0
// Process signals and the line-oriented supervisor protocol shared by native
// servers.
#pragma once

#include <condition_variable>
#include <mutex>
#include <string>

#include "common/http.h"

namespace omni_server {

class StopSignal {
public:
  void Set(const std::string &reason);
  std::string Wait();
  std::string Reason();

private:
  std::mutex mutex_;
  std::condition_variable stopped_;
  std::string reason_;
};

void BlockTerminationSignals();
void IgnoreBrokenPipe();
void WaitForTerminationSignal(StopSignal &stop);
void WatchShutdownCommands(StopSignal &stop);
std::string RandomHex(int character_count);
void EmitControlEvent(const Json &event);

} // namespace omni_server
