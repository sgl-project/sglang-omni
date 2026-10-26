// SPDX-License-Identifier: Apache-2.0
// Pieces of Voxt's Swift model ports that other native models reproduce: the
// Slaney mel filter bank at any FFT size, Swift's whitespace trimming, the
// Swift port's compiled activations, and its SentencePiece decoding.
#pragma once

#include <filesystem>
#include <set>
#include <string>
#include <vector>

#include "mlx/mlx.h"

namespace swift_port {

// Slaney-scale, Slaney-normalized filters [fft_size / 2 + 1, mel_bin_count]
// for 16 kHz audio, built in float32 in the order the Swift port builds them.
std::vector<float> SlaneyMelFilterBank(int fft_size, int mel_bin_count);

// Swift's trimmingCharacters(in: .whitespaces) on valid UTF-8, or with
// newlines, (in: .whitespacesAndNewlines).
std::string TrimWhitespace(const std::string &text, bool newlines);

// x * sigmoid(x) and max(x, 0), compiled shapeless as the Swift port compiles
// them.
mlx::core::array Silu(const mlx::core::array &x);
mlx::core::array Relu(const mlx::core::array &x);

// The pieces of a SentencePiece model file, read as the Swift port's
// SentencePieceTokenizer reads them.
class SentencePieceVocabulary {
public:
  explicit SentencePieceVocabulary(const std::filesystem::path &model_file);

  // The ids' pieces joined as SentencePieceTokenizer.decode joins them:
  // control, unused, out-of-range and dropped ids skipped, byte pieces
  // gathered into runs kept only when valid UTF-8, and the word marker read
  // as a space; not trimmed.
  std::string Decode(const std::vector<int> &ids,
                     const std::set<int> &dropped = {}) const;

private:
  std::vector<std::string> pieces_;
  // Control and unused pieces, which decoding skips.
  std::vector<bool> skipped_;
};

} // namespace swift_port
