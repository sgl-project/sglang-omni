// SPDX-License-Identifier: Apache-2.0
#include "common/cli.h"

#include <iostream>
#include <stdexcept>

namespace cli {

bool ParseCommandLine(
    int argc, char **argv, const std::vector<Arg> &args,
    const std::function<void(const std::string &)> &positional) {
  for (int i = 1; i < argc; ++i) {
    const std::string argument = argv[i];
    if (argument == "--help" || argument == "-h") {
      std::cout << "usage: " << argv[0] << " [options]\n\noptions:\n";
      for (const Arg &option : args) {
        std::cout << "  " << option.name;
        if (option.takes_value) {
          std::cout << " " << option.value_hint;
        }
        std::cout << "\n      " << option.description << "\n";
      }
      return true;
    }
    const Arg *match = nullptr;
    for (const Arg &option : args) {
      if (argument == option.name) {
        match = &option;
        break;
      }
    }
    if (match == nullptr) {
      if (argument.starts_with("--")) {
        throw std::invalid_argument("unknown option " + argument);
      }
      if (positional) {
        positional(argument);
        continue;
      }
      throw std::invalid_argument("unexpected argument " + argument);
    }
    std::string value;
    if (match->takes_value) {
      if (i + 1 >= argc) {
        throw std::invalid_argument(std::string(match->name) +
                                    " needs a value");
      }
      value = argv[++i];
    }
    match->set(value);
  }
  return false;
}

} // namespace cli
