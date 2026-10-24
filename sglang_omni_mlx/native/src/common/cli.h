// SPDX-License-Identifier: Apache-2.0
// Declarative command-line parsing for the native tools: one Arg per flag
// carries a name, a value hint and a typed setter, so parsing, validation and
// --help draw from a single table.
#pragma once

#include <functional>
#include <string>
#include <vector>

namespace cli {

struct Arg {
  const char *name;
  bool takes_value;
  const char *value_hint; // note (guozhihao-224): "<int>" etc. for --help.
  const char *description;
  std::function<void(const std::string &)> set;
};

bool ParseCommandLine(
    int argc, char **argv, const std::vector<Arg> &args,
    const std::function<void(const std::string &)> &positional);

} // namespace cli
