// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <functional>
#include <memory>

#include "common/service.h"
#include "ming3_audio/ming3_audio_model.h"

namespace ming3_audio {

using ChatGeneration =
    std::function<ChatResult(const ChatRequest &, const Cancellation &)>;

std::unique_ptr<omni_server::Service>
CreateChatService(ChatGeneration generate);

} // namespace ming3_audio
