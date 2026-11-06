// SPDX-License-Identifier: Apache-2.0
#include "whisper/whisper_transcriber.h"

#include <algorithm>

namespace whisper {
namespace mx = mlx::core;

mx::array WhisperTranscriber::NextToken(mx::array logits, bool first_step,
                                        float temperature) const {
  // Note (Jiaxin Deng): the float16 decoder's logits are suppressed and
  // ranked in float32, as the reference Whisper decoder does.
  logits = mx::astype(logits, mx::float32);
  if (first_step && begin_suppression_.has_value()) {
    logits = mx::add(logits, mx::astype(*begin_suppression_, logits.dtype()));
  } else {
  }
  if (step_suppression_.has_value()) {
    logits = mx::add(logits, mx::astype(*step_suppression_, logits.dtype()));
  } else {
  }
  if (timestamp_suppression_.has_value()) {
    logits =
        mx::add(logits, mx::astype(*timestamp_suppression_, logits.dtype()));
  } else {
  }
  if (temperature <= 0.0f) {
    return mx::argmax(logits, -1);
  } else {
    return mx::reshape(
        mx::random::categorical(mx::expand_dims(
            mx::divide(logits, mx::array(temperature, logits.dtype())), 0)),
        {});
  }
}

WhisperTranscriber::WindowResult WhisperTranscriber::TranscribeWithDraft(
    const mx::array &encoder_states, const std::vector<int> &prompt_ids,
    int max_new_tokens, const std::atomic<bool> &cancel) const {
  std::vector<DecoderLayerCache> caches(model_.config().text_layer_count);
  const mx::array prompt(prompt_ids.data(),
                         {1, static_cast<int>(prompt_ids.size())}, mx::int32);
  mx::array token =
      NextToken(model_.Decode(prompt, 0, encoder_states, caches), true, 0.0f);
  mx::async_eval({token});
  WindowResult result;
  std::vector<int> generated_ids;
  std::vector<DecoderLayerCache> draft_caches(
      draft_->config().text_layer_count);
  const auto draft_prefill =
      draft_->Decode(prompt, 0, encoder_states, draft_caches);
  mx::eval({draft_prefill});
  int position = prompt_ids.size();
  bool stopped = false;
  while (static_cast<int>(generated_ids.size()) < max_new_tokens) {
    if (cancel.load()) {
      throw qwen3_asr::TranscriptionCancelled();
    } else {
    }
    const int anchor = token.item<uint32_t>();
    if (anchor == end_of_text_id_) {
      stopped = true;
      break;
    } else {
      generated_ids.push_back(anchor);
    }
    const int remaining = max_new_tokens - generated_ids.size();
    if (remaining == 0) {
      break;
    } else {
    }
    const int proposal_count = std::min(draft_steps_, remaining - 1);
    std::vector<int> block = {anchor};
    for (int i = 0; i < proposal_count; ++i) {
      if (cancel.load()) {
        throw qwen3_asr::TranscriptionCancelled();
      } else {
      }
      const mx::array input(&block.back(), {1, 1}, mx::int32);
      auto proposal = NextToken(
          draft_->Decode(input, position + i, encoder_states, draft_caches),
          false, 0.0f);
      block.push_back(proposal.item<uint32_t>());
    }
    auto predictions =
        NextToken(model_.Decode(mx::array(block.data(), {1, int(block.size())},
                                          mx::int32),
                                position, encoder_states, caches, true),
                  false, 0.0f);
    mx::eval({predictions});
    ++result.verify_round_count;
    result.proposed_draft_token_count += proposal_count;
    const auto *predicted_ids = predictions.data<uint32_t>();
    int accepted = 0;
    while (accepted < proposal_count &&
           block[accepted + 1] == int(predicted_ids[accepted])) {
      ++accepted;
      ++result.accepted_draft_token_count;
      if (block[accepted] == end_of_text_id_) {
        stopped = true;
        break;
      } else {
        generated_ids.push_back(block[accepted]);
      }
    }
    if (stopped) {
      break;
    } else {
    }
    token = mx::array(predicted_ids[accepted], mx::uint32);
    // note (ruiling): The last proposal has not yet entered the draft cache.
    if (accepted == proposal_count) {
      const mx::array input(&block.back(), {1, 1}, mx::int32);
      const auto catch_up = draft_->Decode(input, position + accepted,
                                           encoder_states, draft_caches);
      mx::eval({catch_up});
    } else {
    }
    position += accepted + 1;
    WhisperModel::TruncateCaches(caches, position);
    WhisperModel::TruncateCaches(draft_caches, position);
  }
  result.reached_token_limit = !stopped;
  result.text = DecodeText(generated_ids);
  result.generated_token_count = generated_ids.size();
  return result;
}

} // namespace whisper
