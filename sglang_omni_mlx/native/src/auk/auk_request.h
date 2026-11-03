// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "auk/auk_model.h"
#include "common/http.h"

namespace auk {

// The synthesis a /v1/audio/speech body, or with raw_instruction a /generate
// body, asks for; throws std::invalid_argument for a request to answer with
// 400.
SynthesisRequest ParseRequest(const omni_server::Json &request,
                              bool raw_instruction, float max_seconds);

} // namespace auk
