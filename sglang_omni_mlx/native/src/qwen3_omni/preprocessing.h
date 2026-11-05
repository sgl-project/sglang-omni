// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "common/tokenizer.h"
#include "qwen3_omni/media.h"

namespace qwen3_omni {
struct MediaEntry {
  bool user_owned;
  std::vector<int> tokens;
  std::vector<std::array<float, 3>> positions;
  std::optional<mx::array> pixels;
  std::optional<VisualGrid> grid;
  std::optional<mx::array> mel;
  int visual_token = 0;
};
struct PreparedInput {
  std::vector<int> tokens;
  mx::array positions;
  float delta;
  std::vector<MediaEntry> entries;
};
std::pair<mx::array, float>
position_ids(const std::vector<int> &tokens,
             const std::vector<MediaEntry> &entries);
PreparedInput prepare(const ChatRequest &request,
                      const omni_common::Tokenizer &tokenizer,
                      const Cancellation &cancelled);
} // namespace qwen3_omni
