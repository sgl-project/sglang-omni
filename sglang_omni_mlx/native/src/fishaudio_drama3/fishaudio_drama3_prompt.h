// SPDX-License-Identifier: Apache-2.0
// The reference-bracket-v1 prompt of Drama 3 as frame rows.
#pragma once

#include <string>
#include <vector>

#include "common/tokenizer.h"
#include "mlx/mlx.h"

namespace fishaudio_drama3 {

// A transcript and its codes [codebooks, frames], as a reference or a
// generated turn.
struct SpeechTurn {
  std::string text;
  mlx::core::array codes;
};

struct PromptFrames {
  // [rows, 1 + codebooks] int32: text rows are [token, 0, ...], audio rows
  // [first code + semantic_begin, codes of the frame...].
  mlx::core::array frames;
  // The leading rows that only depend on the references.
  int system_rows;
};

// Throws std::invalid_argument, led by location, unless codes is an integer
// [codebook_count, frames] grid with at least one frame, in [0,
// codebook_size).
void CheckCodes(const mlx::core::array &codes, int codebook_count,
                int codebook_size, const std::string &location);
// The prompt; every string is tokenized on its own, as tokenizing the joined
// prompt changes the ids at the seams.
PromptFrames BuildFrames(const omni_common::Tokenizer &tokenizer,
                         const std::vector<SpeechTurn> &references,
                         const std::vector<SpeechTurn> &history,
                         const std::string &target, int codebook_count,
                         int codebook_size, int semantic_begin);

} // namespace fishaudio_drama3
