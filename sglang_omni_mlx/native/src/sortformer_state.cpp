// SPDX-License-Identifier: Apache-2.0
// Sortformer streaming state: FIFO to speaker-cache moves, AOSC compression
// and segment extraction, op for op as Swift SortformerModel.
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

#include "sortformer.h"

// Note (Jiaxin Deng): Swift never fuses a multiply and an add; segment times
// must not either.
#pragma STDC FP_CONTRACT OFF

namespace sortformer {

namespace mx = mlx::core;

namespace {

constexpr float kInfinity = std::numeric_limits<float>::infinity();

// Note (Jiaxin Deng): Swift scalar operands take the array's dtype.
mx::array Scalar(float value, const mx::array &like) {
  return mx::array(value, like.dtype());
}

void UpdateSilenceProfile(StreamingState &state, const mx::array &embeddings,
                          const mx::array &predictions,
                          float silence_threshold) {
  const mx::array is_silence =
      mx::less(mx::sum(predictions, 2), Scalar(silence_threshold, predictions));
  const mx::array silence_weights = mx::astype(is_silence, mx::float32);
  const mx::array silence_count = mx::sum(silence_weights, 1);
  const mx::array silence_sum = mx::sum(
      mx::multiply(embeddings, mx::expand_dims(silence_weights, -1)), 1);
  const mx::array updated_count =
      mx::add(state.silence_frame_count, silence_count);
  const mx::array previous_sum =
      mx::multiply(state.mean_silence_embedding,
                   mx::expand_dims(state.silence_frame_count, -1));
  const mx::array total_sum = mx::add(previous_sum, silence_sum);
  const mx::array count = mx::expand_dims(updated_count, -1);
  state.mean_silence_embedding =
      mx::divide(total_sum, mx::clip(count, Scalar(1.0f, count), std::nullopt));
  state.silence_frame_count = updated_count;
}

mx::array LogPredictionScores(const mx::array &predictions, float threshold) {
  const mx::array log_probabilities = mx::log(
      mx::clip(predictions, Scalar(threshold, predictions), std::nullopt));
  const mx::array complement =
      mx::subtract(Scalar(1.0f, predictions), predictions);
  const mx::array log_complements = mx::log(
      mx::clip(complement, Scalar(threshold, complement), std::nullopt));
  const mx::array complement_sum =
      mx::broadcast_to(mx::sum(log_complements, 2, true), predictions.shape());
  const mx::array scores =
      mx::add(mx::subtract(log_probabilities, log_complements), complement_sum);
  return mx::subtract(scores,
                      Scalar(static_cast<float>(std::log(0.5)), scores));
}

mx::array DisableLowScores(const mx::array &predictions,
                           const mx::array &scores,
                           int min_positive_per_speaker) {
  const mx::array negative_infinity(-kInfinity);
  const mx::array is_speech =
      mx::greater(predictions, Scalar(0.5f, predictions));
  mx::array result = mx::where(is_speech, scores, negative_infinity);
  const mx::array is_positive = mx::greater(result, Scalar(0.0f, result));
  const mx::array positive_count =
      mx::sum(mx::astype(is_positive, mx::float32), 1, true);
  const mx::array has_enough = mx::greater_equal(
      positive_count,
      Scalar(static_cast<float>(min_positive_per_speaker), positive_count));
  const mx::array replace = mx::logical_and(
      mx::logical_and(mx::logical_not(is_positive), is_speech), has_enough);
  return mx::where(replace, negative_infinity, result);
}

// Note (Jiaxin Deng): the boosted set comes from argpartition as in Swift, so
// ties break the way MLX's sort does.
mx::array BoostTopScores(const mx::array &scores, int boost_count,
                         float scale) {
  if (boost_count <= 0) {
    return scores;
  } else {
  }
  const int batch = scores.shape(0);
  const int frames = scores.shape(1);
  const int speakers = scores.shape(2);
  const int k = std::min(boost_count, frames);
  const float boost = -scale * static_cast<float>(std::log(0.5));
  std::vector<mx::array> boosted;
  for (int speaker = 0; speaker < speakers; ++speaker) {
    const mx::array column = mx::reshape(
        mx::slice(scores, {0, 0, speaker}, {batch, frames, speaker + 1}),
        {batch, frames});
    const mx::array top_indices = mx::slice(
        mx::argpartition(mx::negative(column), k - 1, 1), {0, 0}, {batch, k});
    const mx::array is_finite = mx::greater(column, Scalar(-kInfinity, column));
    const mx::array batch_indices = mx::broadcast_to(
        mx::expand_dims(mx::arange(batch, mx::int32), 1), top_indices.shape());
    const mx::array ones = mx::astype(mx::ones_like(top_indices), mx::float32);
    const mx::array mask =
        mx::scatter_add(mx::zeros_like(column),
                        {batch_indices, mx::astype(top_indices, mx::int32)},
                        mx::reshape(ones, {batch, k, 1, 1}), {0, 1});
    const mx::array amount =
        mx::multiply(mx::multiply(mask, Scalar(boost, mask)),
                     mx::astype(is_finite, mx::float32));
    boosted.push_back(mx::add(column, amount));
  }
  return mx::stack(boosted, -1);
}

// Note (Jiaxin Deng): disabled entries (non-finite, or a silence pad frame)
// point at frame 0.
std::pair<mx::array, mx::array> TopIndices(const mx::array &scores,
                                           int cache_length,
                                           int silence_frames_per_speaker,
                                           int max_index) {
  const int batch = scores.shape(0);
  const int frames = scores.shape(1);
  const int frames_without_silence = frames - silence_frames_per_speaker;
  const mx::array flat =
      mx::reshape(mx::transpose(scores, {0, 2, 1}), {batch, -1});
  const int k = std::min(cache_length, flat.shape(1));
  mx::array indices = mx::slice(mx::argpartition(mx::negative(flat), k - 1, 1),
                                {0, 0}, {batch, k});
  const mx::array values = mx::take_along_axis(flat, indices, 1);
  const mx::array valid = mx::greater(values, Scalar(-kInfinity, values));
  // Note (Jiaxin Deng): uint32 indices against an int32 constant promote to
  // int64, as in Swift.
  indices = mx::where(valid, indices, mx::array(max_index, mx::int32));
  mx::array sorted = mx::sort(indices, 1);
  mx::array disabled = mx::equal(sorted, mx::array(max_index, sorted.dtype()));
  sorted = mx::remainder(sorted, mx::array(frames, sorted.dtype()));
  disabled = mx::logical_or(
      disabled, mx::greater_equal(
                    sorted, mx::array(frames_without_silence, sorted.dtype())));
  sorted = mx::where(disabled, mx::array(0, mx::int32), sorted);
  return {sorted, disabled};
}

void CompressSpeakerCache(mx::array &embeddings, mx::array &predictions,
                          const mx::array &mean_silence_embedding,
                          const ModulesConfig &modules) {
  const int speakers = modules.num_speakers;
  const int cache_length = modules.spkcache_len;
  const int silence_per_speaker = modules.spkcache_sil_frames_per_spk;
  const int cache_per_speaker = cache_length / speakers - silence_per_speaker;
  const int strong_boost = static_cast<int>(std::floor(
      static_cast<float>(cache_per_speaker) * modules.strong_boost_rate));
  const int weak_boost = static_cast<int>(std::floor(
      static_cast<float>(cache_per_speaker) * modules.weak_boost_rate));
  const int min_positive = static_cast<int>(std::floor(
      static_cast<float>(cache_per_speaker) * modules.min_pos_scores_rate));

  mx::array scores =
      LogPredictionScores(predictions, modules.pred_score_threshold);
  scores = DisableLowScores(predictions, scores, min_positive);
  const int batch = scores.shape(0);
  if (modules.scores_boost_latest > 0 && scores.shape(1) > cache_length) {
    const mx::array boost_mask = mx::concatenate(
        {mx::zeros({batch, cache_length, speakers}, mx::float32),
         mx::full({batch, scores.shape(1) - cache_length, speakers},
                  modules.scores_boost_latest, mx::float32)},
        1);
    scores = mx::add(scores, boost_mask);
  } else {
  }
  scores = BoostTopScores(scores, strong_boost, 2.0f);
  scores = BoostTopScores(scores, weak_boost, 1.0f);
  if (silence_per_speaker > 0) {
    scores = mx::concatenate(
        {scores, mx::full({batch, silence_per_speaker, speakers}, kInfinity,
                          mx::float32)},
        1);
  } else {
  }
  const auto [indices, disabled] =
      TopIndices(scores, cache_length, silence_per_speaker, modules.max_index);

  const int width = embeddings.shape(2);
  const int kept = indices.shape(1);
  const mx::array embedding_indices = mx::broadcast_to(
      mx::expand_dims(indices, -1), {indices.shape(0), kept, width});
  mx::array gathered_embeddings =
      mx::take_along_axis(embeddings, embedding_indices, 1);
  const mx::array silence =
      mx::broadcast_to(mx::expand_dims(mean_silence_embedding, 1),
                       {indices.shape(0), cache_length, width});
  const mx::array disabled_mask = mx::expand_dims(disabled, -1);
  gathered_embeddings = mx::where(disabled_mask, silence, gathered_embeddings);
  const mx::array prediction_indices =
      mx::broadcast_to(mx::expand_dims(indices, -1),
                       {indices.shape(0), kept, predictions.shape(2)});
  mx::array gathered_predictions =
      mx::take_along_axis(predictions, prediction_indices, 1);
  gathered_predictions =
      mx::where(disabled_mask, mx::array(0.0f), gathered_predictions);
  mx::eval({gathered_embeddings, gathered_predictions});
  embeddings = gathered_embeddings;
  predictions = gathered_predictions;
}

} // namespace

void MaybeCompressState(StreamingState &state, int spkcache_max, int fifo_max,
                        const ModulesConfig &modules) {
  const int fifo_length = state.fifo_length();
  if (fifo_length <= fifo_max) {
    return;
  } else {
  }
  int pop_length = fifo_length - fifo_max;
  if (modules.use_aosc) {
    pop_length = std::min(pop_length, modules.spkcache_update_period);
  } else {
  }
  const int width = state.fifo.shape(2);
  const int speakers = state.fifo_preds.shape(2);
  const mx::array popped_embeddings =
      mx::slice(state.fifo, {0, 0, 0}, {1, pop_length, width});
  const mx::array popped_predictions =
      mx::slice(state.fifo_preds, {0, 0, 0}, {1, pop_length, speakers});
  if (modules.use_aosc) {
    UpdateSilenceProfile(state, popped_embeddings, popped_predictions,
                         modules.sil_threshold);
  } else {
  }
  mx::array cache = mx::concatenate({state.spkcache, popped_embeddings}, 1);
  mx::array cache_predictions =
      mx::concatenate({state.spkcache_preds, popped_predictions}, 1);
  if (cache.shape(1) > spkcache_max) {
    if (!modules.use_aosc) {
      throw std::invalid_argument(
          "Sortformer cache compression requires use_aosc");
    } else {
    }
    CompressSpeakerCache(cache, cache_predictions, state.mean_silence_embedding,
                         modules);
  } else {
  }
  state.spkcache = cache;
  state.spkcache_preds = cache_predictions;
  state.fifo =
      mx::slice(state.fifo, {0, pop_length, 0}, {1, fifo_length, width});
  state.fifo_preds = mx::slice(state.fifo_preds, {0, pop_length, 0},
                               {1, fifo_length, speakers});
  mx::eval({state.spkcache, state.spkcache_preds, state.fifo, state.fifo_preds,
            state.mean_silence_embedding, state.silence_frame_count});
}

std::vector<Segment>
ProbabilitiesToSegments(const std::vector<float> &probabilities,
                        int frame_count, int speaker_count,
                        float frame_duration, float threshold,
                        float min_duration, float merge_gap) {
  std::vector<Segment> segments;
  for (int speaker = 0; speaker < speaker_count; ++speaker) {
    std::vector<Segment> speaker_segments;
    int segment_start = -1;
    const auto close = [&](int end_frame) {
      const float start_time =
          static_cast<float>(segment_start) * frame_duration;
      const float end_time = static_cast<float>(end_frame) * frame_duration;
      if (end_time - start_time >= min_duration) {
        speaker_segments.push_back({start_time, end_time, speaker});
      } else {
      }
    };
    for (int frame = 0; frame < frame_count; ++frame) {
      const bool active =
          probabilities[static_cast<size_t>(frame) * speaker_count + speaker] >
          threshold;
      if (active) {
        if (segment_start < 0) {
          segment_start = frame;
        } else {
        }
      } else if (segment_start >= 0) {
        close(frame);
        segment_start = -1;
      } else {
      }
    }
    if (segment_start >= 0) {
      close(frame_count);
    } else {
    }
    if (merge_gap > 0 && speaker_segments.size() > 1) {
      std::vector<Segment> merged = {speaker_segments.front()};
      for (size_t i = 1; i < speaker_segments.size(); ++i) {
        const Segment &segment = speaker_segments[i];
        if (segment.start - merged.back().end <= merge_gap) {
          merged.back() = {merged.back().start, segment.end, segment.speaker};
        } else {
          merged.push_back(segment);
        }
      }
      speaker_segments = std::move(merged);
    } else {
    }
    segments.insert(segments.end(), speaker_segments.begin(),
                    speaker_segments.end());
  }
  // Note (Jiaxin Deng): Swift's sort is stable in practice; ties keep speaker
  // order.
  std::stable_sort(
      segments.begin(), segments.end(),
      [](const Segment &a, const Segment &b) { return a.start < b.start; });
  return segments;
}

} // namespace sortformer
