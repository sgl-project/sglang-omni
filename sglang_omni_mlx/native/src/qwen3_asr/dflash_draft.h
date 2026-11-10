// SPDX-License-Identifier: Apache-2.0
// The DFlash draft for a target on the Qwen3 decoder (Qwen3-ASR, MOSS-TD): a
// short Qwen3 decoder whose context is the target's hidden states, proposing a
// block of tokens per round.
#pragma once

#include <atomic>
#include <filesystem>
#include <functional>
#include <map>
#include <vector>

#include "common/layers.h"
#include "mlx/mlx.h"
#include "qwen3_asr/qwen3_decoder.h"

namespace qwen3_asr {

struct LoadedDraft;

class DFlashDraft {
public:
  // Loads a Speculators DFlash checkpoint trained for a target of the given
  // geometry, refusing one that does not fit it; quantization_bits 0 keeps the
  // weights as stored, otherwise the linear layers are quantized on load.
  // rope_theta_scale_by_prompt_token_count maps a prompt length to the factor
  // applied to the draft's rotary base once the prompt is longer than it; the
  // largest length below the prompt wins, and none leaves the base as stored.
  DFlashDraft(const std::filesystem::path &directory, int quantization_bits,
              const Qwen3DecoderConfig &target,
              std::map<int, float> rope_theta_scale_by_prompt_token_count = {});

  // Greedy decoding with the draft: each round the target verifies the block
  // the draft proposed, cut to the tokens max_new_tokens still allows, and
  // commits the accepted tokens plus one more. Tokens go to commit until it
  // reports the output complete; returns the rounds. A positive
  // prefill_step_tokens bounds prompt chunks, leaving the last token for the
  // first logits; zero prefills the prompt in one pass.
  int Decode(const Qwen3Decoder &target, const mlx::core::array &embeddings,
             int max_new_tokens, int prefill_step_tokens,
             const std::function<bool(int)> &commit,
             const std::atomic<bool> &cancel) const;

private:
  DFlashDraft(LoadedDraft draft, int quantization_bits,
              std::map<int, float> rope_theta_scale_by_prompt_token_count);

  // Appends the first token_count positions of the target hidden states
  // entering each aux layer, each [1, length, hidden], as context.
  void AppendContext(const Qwen3Decoder &draft_decoder,
                     const std::vector<mlx::core::array> &aux_hidden_states,
                     int token_count, std::vector<KVCache> &caches) const;
  // Proposes the target token ids after the anchor, [block_size - 1], from
  // the block's embeddings [1, block_size, hidden]: the anchor, then mask
  // tokens, at the positions after the context.
  mlx::core::array ProposeBlock(const Qwen3Decoder &draft_decoder,
                                const mlx::core::array &block_embeddings,
                                std::vector<KVCache> &caches) const;

  omni_common::Checkpoint checkpoint_;
  Qwen3Decoder decoder_;
  // Target layers (0-indexed) whose entering hidden states form the context.
  std::vector<int> aux_layers_;
  // Tokens per block: the anchor and block_size - 1 proposals.
  int block_size_;
  int mask_token_id_;
  std::map<int, float> rope_theta_scale_by_prompt_token_count_;
};

} // namespace qwen3_asr
