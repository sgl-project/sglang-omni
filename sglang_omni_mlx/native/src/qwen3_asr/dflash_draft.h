// SPDX-License-Identifier: Apache-2.0
// The DFlash draft for Qwen3-ASR: a short Qwen3 decoder whose context is the
// target's hidden states, proposing a block of tokens per round.
#pragma once

#include <filesystem>
#include <vector>

#include "common/layers.h"
#include "mlx/mlx.h"
#include "qwen3_asr/qwen3_decoder.h"

namespace qwen3_asr {

class DFlashDraft {
public:
  // Loads a Speculators DFlash checkpoint trained for a target of the given
  // geometry; quantization_bits 0 keeps the weights as stored, otherwise the
  // linear layers are quantized on load.
  DFlashDraft(const std::filesystem::path &directory, int quantization_bits,
              const Qwen3DecoderConfig &target);

  // Target layers (0-indexed) whose entering hidden states form the context.
  const std::vector<int> &aux_layers() const { return aux_layers_; }
  // Tokens per block: the anchor and block_size - 1 proposals.
  int block_size() const { return block_size_; }
  int mask_token_id() const { return mask_token_id_; }
  std::vector<KVCache> NewCaches() const;
  // Appends the first token_count positions of the target hidden states
  // entering each aux layer, each [1, length, hidden], as context.
  void AppendContext(const std::vector<mlx::core::array> &aux_hidden_states,
                     int token_count, std::vector<KVCache> &caches) const;
  // Proposes the target token ids after the anchor, [block_size - 1], from
  // the block's embeddings [1, block_size, hidden]: the anchor, then mask
  // tokens, at the positions after the context.
  mlx::core::array ProposeBlock(const mlx::core::array &block_embeddings,
                                std::vector<KVCache> &caches) const;

private:
  DFlashDraft(const nlohmann::json &config, omni_common::WeightMap weights,
              int quantization_bits);

  omni_common::Checkpoint checkpoint_;
  Qwen3Decoder decoder_;
  std::vector<int> aux_layers_;
  int block_size_;
  int mask_token_id_;
};

} // namespace qwen3_asr
