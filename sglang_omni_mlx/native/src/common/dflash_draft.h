// SPDX-License-Identifier: Apache-2.0
// The DFlash draft for a target on the Qwen3 decoder (Qwen3-ASR, MOSS-TD): a
// short Qwen3 decoder whose context is the target's hidden states, proposing a
// block of tokens per round. A DSpark draft is the same decoder with a Markov
// head that biases each proposal by the one before it.
#pragma once

#include <atomic>
#include <filesystem>
#include <functional>
#include <string>
#include <vector>

#include "common/kv_cache.h"
#include "common/layers.h"
#include "common/qwen3_decoder.h"
#include "mlx/mlx.h"

namespace omni_common {

enum class DraftPolicy { kAdaptive, kFixed };

DraftPolicy ParseDraftPolicy(const std::string &value);

struct LoadedDraft;

// One draft decode's verification rounds, and the proposals the target
// checked and accepted; the token the target adds each round is in neither.
struct DraftStatistics {
  int verify_round_count = 0;
  int proposed_token_count = 0;
  int accepted_token_count = 0;
};

// The geometry of the target whose hidden states a draft reads and whose
// tokens it embeds and predicts.
struct DraftTarget {
  int hidden_size;
  int vocab_size;
  int layer_count;
};

class DFlashDraft {
public:
  // Loads a Speculators DFlash or DSpark checkpoint trained for a target of
  // the given geometry, refusing one that does not fit it; quantization_bits
  // 0 keeps the weights as stored, otherwise the linear layers are quantized
  // on load.
  DFlashDraft(const std::filesystem::path &directory, int quantization_bits,
              const DraftTarget &target);

  // Greedy decoding with the draft: each round the target verifies the block
  // the draft proposed, cut to the tokens max_new_tokens still allows, and
  // commits the accepted tokens plus one more. Tokens go to commit until it
  // reports the output complete. A positive prefill_step_tokens bounds prompt
  // chunks, leaving the last token for the first logits; zero prefills the
  // prompt in one pass. Adaptive backoff is opt-in here so other targets retain
  // their existing fixed-draft behavior.
  DraftStatistics Decode(const Qwen3Decoder &target,
                         const mlx::core::array &embeddings, int max_new_tokens,
                         int prefill_step_tokens,
                         const std::function<bool(int)> &commit,
                         const std::atomic<bool> &cancel,
                         DraftPolicy policy = DraftPolicy::kFixed) const;
  // The same decode over caches the caller keeps: the target's keys and
  // values and the draft's context, over the same positions, which embeddings
  // and the committed tokens but the last extend. guessed_ids, tokens likely
  // to come next, are checked in the prompt's pass and kept while the target
  // agrees. A backoff to plain decoding empties draft_caches.
  DraftStatistics Decode(const Qwen3Decoder &target,
                         const mlx::core::array &embeddings,
                         const std::vector<int> &guessed_ids,
                         int max_new_tokens, int prefill_step_tokens,
                         const std::function<bool(int)> &commit,
                         const std::atomic<bool> &cancel, DraftPolicy policy,
                         std::vector<omni_common::KVCache> &caches,
                         std::vector<omni_common::KVCache> &draft_caches) const;
  // Extends caches the caller keeps, the target's and the draft's over the
  // same positions, by embeddings [1, length, hidden] without decoding.
  void Prefill(const Qwen3Decoder &target, const mlx::core::array &embeddings,
               std::vector<omni_common::KVCache> &caches,
               std::vector<omni_common::KVCache> &draft_caches) const;
  // Empty caches for the draft's context.
  std::vector<omni_common::KVCache> NewCaches() const {
    return decoder_.NewCaches();
  }
  // Target layers (0-indexed) whose entering hidden states form the context.
  const std::vector<int> &aux_layers() const { return aux_layers_; }
  // The block the draft reads after anchor_id: the anchor, then mask tokens.
  std::vector<int> BlockIds(int anchor_id) const;
  // Appends the first token_count positions of the target hidden states
  // entering each aux layer, each [1, length, hidden], as context.
  void AppendContext(const std::vector<mlx::core::array> &aux_hidden_states,
                     int token_count,
                     std::vector<omni_common::KVCache> &caches) const;
  // Proposes the target token ids after anchor_id, int32 [proposal_count],
  // from the embeddings of BlockIds(anchor_id) [1, block_size, hidden] at the
  // positions after the context.
  mlx::core::array
  ProposeBlock(int anchor_id, const mlx::core::array &block_embeddings,
               std::vector<omni_common::KVCache> &caches) const;

private:
  DFlashDraft(LoadedDraft draft, int quantization_bits);

  omni_common::Checkpoint checkpoint_;
  Qwen3Decoder decoder_;
  std::vector<int> aux_layers_;
  int block_size_;
  int mask_token_id_;
  // note (wilsonzheng0327): DSpark's sample_from_anchor reads a proposal off
  // every block position, the anchor's included; DFlash skips the anchor.
  int proposal_count_;
  bool has_markov_head_;
};

} // namespace omni_common
