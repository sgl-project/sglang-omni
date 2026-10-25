// SPDX-License-Identifier: Apache-2.0
// The server every native model binary runs: the transcription API (JSON or
// SSE) and Voxt's supervisor protocol. Each binary supplies how it loads and
// reads a request, its own flags, and any handlers beyond that API.
#pragma once

#include <filesystem>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "common/form.h"
#include "qwen3_asr/realtime.h"
#include "qwen3_asr/worker.h"

struct mg_context;

namespace asr_service {

using FormFields = std::map<std::string, omni_common::FormField>;
using qwen3_asr::Transcription;

// A loaded checkpoint of the served kind.
class ServedModel {
public:
  virtual ~ServedModel() = default;
  // Binds one request's form fields to its samples; throws
  // std::invalid_argument for a field it does not accept.
  virtual Transcription Prepare(std::vector<float> samples,
                                const FormFields &form) const = 0;
  // Adds handlers beyond the transcription API once the server listens.
  virtual void AddHandlers(mg_context *, qwen3_asr::TranscriptionWorker &) {}
  // False for a model that serves no transcriptions.
  virtual bool Transcribes() const { return true; }
  // Request counts for /health.
  virtual std::map<std::string, int>
  RequestStates(const qwen3_asr::TranscriptionWorker &worker) const {
    return worker.RequestStates();
  }
  // The name /v1/models lists, and whether --model-name chose it rather
  // than the server generating it; set before AddHandlers runs.
  const std::string &model_name() const { return model_name_; }
  bool has_chosen_model_name() const { return has_chosen_model_name_; }
  void set_model_name(const std::string &model_name,
                      bool has_chosen_model_name) {
    model_name_ = model_name;
    has_chosen_model_name_ = has_chosen_model_name;
  }

private:
  std::string model_name_;
  bool has_chosen_model_name_ = false;
};

using ModelLoader =
    std::function<std::unique_ptr<ServedModel>(const std::filesystem::path &)>;

struct ServedKind {
  std::string model_kind;
  ModelLoader load;
  // Flags beyond the common ones, each taking a value; a setter throws
  // std::logic_error for a value it cannot use.
  std::map<std::string, std::function<void(const std::string &)>> flags;
  // Checks the parsed flags together; throws std::invalid_argument.
  std::function<void()> check_flags;
};

// A form field as given, or as an integer or a finite number; the latter two
// throw std::invalid_argument when the field is not one, and leave out empty
// fields.
std::optional<std::string> TextField(const FormFields &form,
                                     const std::string &name);
std::optional<int> IntegerField(const FormFields &form,
                                const std::string &name);
std::optional<float> NumberField(const FormFields &form,
                                 const std::string &name);

// Builds the session of one realtime socket around the socket's sender.
using RealtimeFactory =
    std::function<std::shared_ptr<qwen3_asr::RealtimeConnection>(
        qwen3_asr::RealtimeConnection::Sender)>;

// Serves the realtime API on /v1/realtime; factory must outlive the server.
void AddRealtimeHandler(mg_context *context, const RealtimeFactory &factory);

// Runs the server for the kind --model-kind names (the first by default)
// until it is stopped; returns the exit code.
//
//   BINARY --model-path DIR [--model-name NAME] [--host H] [--port P]
//   BINARY --supervised --model-kind KIND --model-directory DIR
int Serve(int argc, char **argv, const std::vector<ServedKind> &kinds);
inline int Serve(int argc, char **argv, const ServedKind &kind) {
  return Serve(argc, argv, std::vector<ServedKind>{kind});
}

} // namespace asr_service
