// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <functional>
#include <memory>

#include "common/service.h"
#include "ming3_audio/ming3_audio_model.h"

namespace ming3_audio {

using ChatPreparation = std::function<PreparedChatRequest(ChatRequest)>;
using ChatGeneration = std::function<ChatResult(const PreparedChatRequest &,
                                                const Cancellation &)>;

std::unique_ptr<omni_server::Service>
CreateChatService(ChatPreparation prepare, ChatGeneration generate);

} // namespace ming3_audio
