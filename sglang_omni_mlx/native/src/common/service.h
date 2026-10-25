// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <filesystem>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>

struct mg_context;

namespace omni_server {

class Service {
public:
  virtual ~Service() = default;
  virtual void SetModelName(const std::string &, bool) {}
  virtual void Register(mg_context *context, const std::string &model_name) = 0;
  virtual std::map<std::string, int> RequestStates() const = 0;
  virtual void CancelAll() = 0;
};

struct ServedKind {
  std::string model_kind;
  std::function<std::unique_ptr<Service>(const std::filesystem::path &)> load;
  std::map<std::string, std::function<void(const std::string &)>> flags;
  std::function<void()> check_flags;
};

int Serve(int argc, char **argv, const std::vector<ServedKind> &kinds);

} // namespace omni_server
