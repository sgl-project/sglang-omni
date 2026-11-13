// SPDX-License-Identifier: Apache-2.0
// Qwen3-ASR's audio prompt layout and its audio token counts.
#pragma once

namespace qwen3_asr {

// Qwen3-ASR encodes mel frames in chunks of 100 frames, 13 audio tokens each.
inline constexpr int kChunkFrameCount = 100;
inline constexpr int kChunkTokenCount = 13;

// How the audio part of the prompt is built. kReference follows the
// checkpoint's processor; kVoxtSwift reproduces Voxt's Swift port (one more
// mel frame, and a token count computed with float32 true division).
enum class AudioLayout { kReference, kVoxtSwift };

int ConvOutputFrames(int frame_count);
int ReferenceTokenCount(int frame_count);
int SwiftTokenCount(int frame_count);
int TokenCount(int frame_count, AudioLayout layout);

} // namespace qwen3_asr
