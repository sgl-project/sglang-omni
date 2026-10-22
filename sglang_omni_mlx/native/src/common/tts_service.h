// SPDX-License-Identifier: Apache-2.0
// Shared OpenAI-style TTS HTTP endpoint.
#pragma once

#include <filesystem>
#include <functional>
#include <set>
#include <string>

#include "civetweb.h"
#include "common/synthesis.h"

namespace omni_server {

class TTSService {
public:
  TTSService(std::string default_voice, std::set<std::string> voices,
             SynthesisWorker &worker);
  void Register(mg_context *context);

private:
  static int HandleSpeech(mg_connection *connection, void *data);
  int Synthesize(mg_connection *connection);

  std::string default_voice_;
  std::set<std::string> voices_;
  SynthesisWorker &worker_;
};

struct TTSModel {
  std::set<std::string> voices;
  std::string default_voice;
  SynthesisFunction synthesize;
};

struct TTSKind {
  std::string model_kind;
  std::function<TTSModel(const std::filesystem::path &)> load;
};

int ServeTTS(int argument_count, char **argument_values, const TTSKind &kind);

} // namespace omni_server
