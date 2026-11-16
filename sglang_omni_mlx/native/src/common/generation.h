// SPDX-License-Identifier: Apache-2.0
// How an ASR or TTS generation ends: a finish reason, or cancellation.
#pragma once

#include <stdexcept>

namespace omni_common {

enum class FinishReason { kStop, kLength };

inline const char *FinishReasonName(FinishReason reason) {
  return reason == FinishReason::kStop ? "stop" : "length";
}

class GenerationCancelled : public std::runtime_error {
public:
  GenerationCancelled() : std::runtime_error("generation cancelled") {}
};

} // namespace omni_common
