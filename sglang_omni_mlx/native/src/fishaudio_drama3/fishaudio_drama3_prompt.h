// SPDX-License-Identifier: Apache-2.0
// The reference-bracket-v1 prompt of Drama 3 as frame rows.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "common/tokenizer.h"
#include "mlx/mlx.h"
#include "nlohmann/json.hpp"

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

// A reference's codes [codebooks][frames] from vq_codes, a grid of
// codebook_count rows of the same 1 to max_frames integers in [0,
// codebook_size); throws std::invalid_argument, led by location. Plain
// integers, so that only the worker thread makes MLX arrays.
std::vector<std::vector<int32_t>>
ReferenceCodes(const nlohmann::ordered_json &vq_codes, int codebook_count,
               int codebook_size, size_t max_frames,
               const std::string &location);
// The prompt of a validated target, references and history; every string is
// tokenized on its own, as tokenizing the joined prompt changes the ids at the
// seams.
PromptFrames BuildFrames(const omni_common::Tokenizer &tokenizer,
                         const std::vector<SpeechTurn> &references,
                         const std::vector<SpeechTurn> &history,
                         const std::string &target, int codebook_count,
                         int semantic_begin);

} // namespace fishaudio_drama3
