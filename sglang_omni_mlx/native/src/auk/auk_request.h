// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <string>

#include <nlohmann/json_fwd.hpp>

#include "auk/auk_model.h"

namespace auk {

struct ParsedRequest {
  SynthesisRequest synthesis;
  bool pcm;
};

ParsedRequest ParseRequest(const nlohmann::json &request, bool raw_instruction,
                           float max_seconds, const std::string &model_name);

} // namespace auk
