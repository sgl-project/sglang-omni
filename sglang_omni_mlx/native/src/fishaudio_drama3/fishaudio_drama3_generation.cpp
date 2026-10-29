// SPDX-License-Identifier: Apache-2.0
// Drama 3's generation: prefill, the per-frame step and its sampling, voices,
// and speech for text chunk by chunk.
#include <CommonCrypto/CommonDigest.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

#include "fishaudio_drama3/fishaudio_drama3_model.h"
#include "fishaudio_drama3/fishaudio_drama3_text.h"

namespace fishaudio_drama3 {

namespace mx = mlx::core;

namespace {

// Repetition-aware sampling from the training recipe: a semantic main token
// that repeats one of the last kRepetitionWindow main tokens is replaced by a
// second draw at these settings.
constexpr int kRepetitionWindow = 16;
constexpr double kRepetitionTemperature = 1.0;
constexpr int kRepetitionTopK = 30;
constexpr double kRepetitionTopP = 0.95;
constexpr double kMinTemperature = 1e-5;
constexpr int kCodebookTopK = 30;
constexpr int kPrefillChunkRows = 256;
// Frames per codec call when a chunk is decoded at once; the codec's working
// set grows with the frames of one call.
constexpr int kDecodeFrames = 64;
constexpr int kStreamIntervalFrames = 7;
constexpr size_t kVoiceCacheSize = 4;

// One id per row of logits [rows, vocabulary], drawn as the model was trained:
// temperature before top-p, the cumulative probability over the full
// vocabulary, and the token that crosses top_p dropped unless it ranks first
// or keep_crossing keeps it. Only the top_k candidates are sorted.
mx::array SampleTopKTopP(const mx::array &logits, double temperature,
                         std::optional<int> top_k, double top_p,
                         bool keep_crossing) {
  mx::array x = mx::divide(
      mx::astype(logits, mx::float32),
      omni_common::Scalar(std::max(temperature, kMinTemperature), mx::float32));
  // At low temperature the logits share a huge offset that would swamp the
  // log-sum-exp in float32.
  x = mx::subtract(x, mx::max(x, -1, true));
  const int vocabulary = x.shape(-1);
  const int kept =
      top_k.has_value() ? std::min(*top_k, vocabulary) : vocabulary;
  mx::array ids =
      SliceLast(mx::argpartition(mx::negative(x), kept - 1, -1), 0, kept);
  mx::array values = mx::take_along_axis(x, ids, -1);
  const mx::array order = mx::argsort(mx::negative(values), -1);
  values = mx::take_along_axis(values, order, -1);
  ids = mx::take_along_axis(ids, order, -1);
  const mx::array probabilities =
      mx::exp(mx::subtract(values, mx::logsumexp(x, -1, true)));
  const mx::array passed = mx::cumsum(probabilities, -1, false, !keep_crossing);
  const mx::array dropped = mx::logical_and(
      mx::greater(passed, omni_common::Scalar(top_p, mx::float32)),
      mx::greater(mx::arange(kept, mx::int32), mx::array(0, mx::int32)));
  const mx::array choice = mx::random::categorical(
      mx::where(dropped, mx::array(-std::numeric_limits<float>::infinity()),
                values),
      -1);
  return mx::squeeze(mx::take_along_axis(ids, mx::expand_dims(choice, -1), -1),
                     -1);
}

// Adds part to digest, prefixed by its length so bytes cannot move between
// parts.
void DigestPart(CC_SHA256_CTX &digest, const void *bytes, size_t byte_count) {
  const uint64_t length = byte_count;
  CC_SHA256_Update(&digest, &length, sizeof(length));
  const auto *position = static_cast<const uint8_t *>(bytes);
  while (byte_count > 0) {
    const CC_LONG piece = static_cast<CC_LONG>(
        std::min<size_t>(byte_count, std::numeric_limits<CC_LONG>::max()));
    CC_SHA256_Update(&digest, position, piece);
    position += piece;
    byte_count -= piece;
  }
}

} // namespace

mx::array Drama3Model::Prefill(const mx::array &frames, ModelCaches &caches,
                               int start, int stop,
                               const std::atomic<bool> &cancel) const {
  const mx::array batched = mx::expand_dims(frames, 0);
  const int last = frames.shape(0) - 1;
  std::optional<mx::array> hidden;
  for (int begin = start; begin < stop; begin += kPrefillChunkRows) {
    if (cancel.load()) {
      throw qwen3_asr::TranscriptionCancelled();
    } else {
    }
    const int end = std::min(begin + kPrefillChunkRows, stop);
    hidden =
        Backbone(Embed(SliceAxis(batched, 1, begin, end)), caches.backbone);
    const int labelled = std::min(end, last);
    if (labelled > begin) {
      // The tower is teacher-forced: row t is labelled by row t + 1.
      const mx::array next_frames =
          SliceAxis(batched, 1, begin + 1, labelled + 1);
      const mx::array is_semantic =
          IsSemantic(mx::squeeze(SliceLast(next_frames, 0, 1), -1));
      mx::array tower_hidden =
          Enrich(SliceAxis(*hidden, 1, 0, labelled - begin),
                 mx::squeeze(SliceLast(next_frames, 1, 2), -1), is_semantic,
                 SliceAxis(batched, 1, begin, labelled), caches.tower);
      for (int layer = shared_tower_layer_count_;
           layer < static_cast<int>(caches.tower.size()); ++layer) {
        const int codebook = layer - shared_tower_layer_count_ + 1;
        tower_hidden = TowerLayer(
            tower_hidden,
            mx::squeeze(SliceLast(next_frames, codebook, codebook + 1), -1),
            is_semantic, layer, caches.tower[layer], std::nullopt);
      }
    } else {
    }
    std::vector<mx::array> state = {*hidden};
    for (const LayerCache &cache : caches.backbone) {
      if (cache.recurrent_state.has_value()) {
        state.push_back(*cache.recurrent_state);
        state.push_back(*cache.convolution_state);
      } else {
      }
      for (const mx::array &array : cache.attention.Arrays())
        state.push_back(array);
    }
    for (const qwen3_asr::KVCache &cache : caches.tower) {
      for (const mx::array &array : cache.Arrays())
        state.push_back(array);
    }
    mx::eval(state);
    // Note (Yitong Li): each block's temporaries grow with the cache; freeing
    // them keeps a long prompt from holding one of every size at once.
    if (end < stop) {
      mx::clear_cache();
    } else {
    }
  }
  return SliceAxis(*hidden, 1, hidden->shape(1) - 1, hidden->shape(1));
}

Drama3Model::Step Drama3Model::DecodeStep(const mx::array &hidden,
                                          const mx::array &frame,
                                          const mx::array &history,
                                          ModelCaches &caches,
                                          const Sampling &sampling) const {
  const mx::array logits = mx::astype(
      checkpoint_.Linear(mx::squeeze(hidden, 1), "semantic_head"), mx::float32);
  const mx::array first = mx::take(
      candidate_ids_, SampleTopKTopP(logits, sampling.temperature,
                                     sampling.top_k, sampling.top_p, true));
  const mx::array retry = mx::take(
      candidate_ids_, SampleTopKTopP(logits, kRepetitionTemperature,
                                     kRepetitionTopK, kRepetitionTopP, true));
  const mx::array repeated = mx::any(
      mx::equal(mx::expand_dims(history, 0), mx::expand_dims(first, 1)), -1);
  const mx::array main_token =
      mx::where(mx::logical_and(repeated, IsSemantic(first)), retry, first);
  const mx::array semantic_codes = mx::clip(
      mx::subtract(main_token, mx::array(semantic_begin_, main_token.dtype())),
      mx::array(0, main_token.dtype()),
      mx::array(codebook_size_ - 1, main_token.dtype()));
  const mx::array is_semantic = mx::expand_dims(IsSemantic(main_token), 1);
  const std::optional<int> head_top_k =
      sampling.top_k.has_value() ? std::min(*sampling.top_k, kCodebookTopK)
                                 : kCodebookTopK;

  std::vector<mx::array> codes = {mx::expand_dims(semantic_codes, 1)};
  mx::array tower_hidden =
      Enrich(hidden, codes.back(), is_semantic, frame, caches.tower);
  for (int layer = shared_tower_layer_count_;
       layer < static_cast<int>(caches.tower.size()); ++layer) {
    tower_hidden = TowerLayer(tower_hidden, codes.back(), is_semantic, layer,
                              caches.tower[layer], std::nullopt);
    codes.push_back(mx::expand_dims(
        SampleTopKTopP(
            mx::squeeze(
                CodebookHead(tower_hidden, layer - shared_tower_layer_count_),
                1),
            sampling.temperature, head_top_k, sampling.top_p, false),
        1));
  }
  std::vector<mx::array> columns = {mx::expand_dims(main_token, 1)};
  for (const mx::array &codebook_codes : codes) {
    columns.push_back(mx::astype(codebook_codes, mx::int32));
  }
  const mx::array new_frame =
      mx::astype(mx::expand_dims(mx::concatenate(columns, -1), 1), mx::int32);
  const mx::array next_hidden = Backbone(Embed(new_frame), caches.backbone);
  const mx::array next_history =
      mx::concatenate({SliceAxis(history, 0, 1, history.shape(0)), main_token});
  return {new_frame, next_hidden, next_history, main_token};
}

bool Drama3Model::GenerateChunk(
    const PromptFrames &prompt, Voice &voice, const Sampling &sampling,
    int max_frames, const std::atomic<bool> &cancel,
    const std::function<void(const mx::array &)> &on_frame) const {
  if (!voice.prefix.has_value()) {
    ModelCaches fresh;
    fresh.backbone.resize(layer_count_);
    fresh.tower.resize(codebook_count_ - 1 + shared_tower_layer_count_);
    Prefill(prompt.frames, fresh, 0, prompt.system_rows, cancel);
    voice.prefix = fresh;
  } else {
  }
  ModelCaches caches = *voice.prefix;
  const int rows = prompt.frames.shape(0);
  const mx::array hidden =
      Prefill(prompt.frames, caches, prompt.system_rows, rows, cancel);
  const mx::array last_row =
      mx::expand_dims(SliceAxis(prompt.frames, 0, rows - 1, rows), 0);
  const mx::array history = mx::squeeze(
      SliceLast(SliceAxis(prompt.frames, 0,
                          std::max(0, rows - kRepetitionWindow), rows),
                0, 1),
      -1);
  // Note (Yitong Li): step n + 1 is queued before step n's main token is read,
  // so the GPU never waits for the host; the one wasted step after the end
  // only touches this chunk's caches, and draws its random keys as the
  // reference does.
  Step pending = DecodeStep(hidden, last_row, history, caches, sampling);
  mx::async_eval(
      {pending.frame, pending.hidden, pending.history, pending.main_token});
  // Note (Yitong Li): a turn often ends with <|semantic:0|> right before the
  // end; that frame decodes to a click, so it is held back one step and
  // dropped at the end.
  std::optional<mx::array> held;
  for (int frame_index = 0; frame_index < max_frames; ++frame_index) {
    if (cancel.load()) {
      throw qwen3_asr::TranscriptionCancelled();
    } else {
    }
    const Step current = pending;
    if (frame_index + 1 < max_frames) {
      pending = DecodeStep(current.hidden, current.frame, current.history,
                           caches, sampling);
      mx::async_eval(
          {pending.frame, pending.hidden, pending.history, pending.main_token});
    } else {
    }
    const int token = current.main_token.item<int>();
    if (token == end_of_speech_id_) {
      return true;
    } else {
    }
    if (held.has_value()) {
      on_frame(*held);
    } else {
    }
    held = mx::reshape(SliceLast(current.frame, 1, 1 + codebook_count_),
                       {codebook_count_});
    mx::eval(*held);
    if (token != semantic_begin_) {
      on_frame(*held);
      held = std::nullopt;
    } else {
    }
  }
  if (held.has_value()) {
    on_frame(*held);
  } else {
  }
  return false;
}

std::shared_ptr<Voice>
Drama3Model::GetVoice(const std::vector<ReferenceInput> &references) {
  CC_SHA256_CTX digest;
  CC_SHA256_Init(&digest);
  for (const ReferenceInput &reference : references) {
    if (reference.codes.has_value()) {
      DigestPart(digest, "codes", 5);
      for (const std::vector<int32_t> &codebook : *reference.codes) {
        DigestPart(digest, codebook.data(), codebook.size() * sizeof(int32_t));
      }
    } else {
      DigestPart(digest, reference.samples.data(),
                 reference.samples.size() * sizeof(float));
    }
    DigestPart(digest, reference.text.data(), reference.text.size());
  }
  std::string reference_key(CC_SHA256_DIGEST_LENGTH, '\0');
  CC_SHA256_Final(reinterpret_cast<unsigned char *>(reference_key.data()),
                  &digest);
  for (auto cached = voices_.begin(); cached != voices_.end(); ++cached) {
    if (cached->first == reference_key) {
      voices_.splice(voices_.begin(), voices_, cached);
      return voices_.front().second;
    } else {
    }
  }
  auto voice = std::make_shared<Voice>();
  for (const ReferenceInput &reference : references) {
    if (reference.codes.has_value()) {
      const std::vector<std::vector<int32_t>> &grid = *reference.codes;
      const size_t frame_count = grid.empty() ? 0 : grid.front().size();
      std::vector<int32_t> codes;
      for (const std::vector<int32_t> &codebook : grid) {
        if (codebook.size() != frame_count) {
          throw std::invalid_argument("reference codes must be a grid");
        } else {
        }
        codes.insert(codes.end(), codebook.begin(), codebook.end());
      }
      voice->references.push_back(
          {reference.text, mx::array(codes.data(),
                                     {static_cast<int>(grid.size()),
                                      static_cast<int>(frame_count)},
                                     mx::int32)});
    } else {
      voice->references.push_back(
          {reference.text, codec_.Encode(reference.samples)});
      // Temporaries sized by this reference rarely fit a later allocation.
      mx::clear_cache();
    }
  }
  // The prompt checks the codes before any is cached.
  for (size_t index = 0; index < voice->references.size(); ++index) {
    CheckCodes(voice->references[index].codes, codebook_count_, codebook_size_,
               "reference " + std::to_string(index + 1));
  }
  voices_.emplace_front(reference_key, voice);
  while (voices_.size() > kVoiceCacheSize) {
    voices_.pop_back();
  }
  return voice;
}

SynthesisResult Drama3Model::Synthesize(
    const std::string &text, Voice &voice, const Sampling &sampling,
    int max_frames, bool stream, const std::atomic<bool> &cancel,
    const std::function<void(const mx::array &)> &on_frame,
    const std::function<void(std::vector<float>)> &on_audio) {
  ValidateText(text, "text");
  const std::vector<std::string> chunks = SplitText(text);
  CodecStream codec_stream(codec_);
  const auto decode = [&](const std::vector<mx::array> &codes, size_t begin,
                          size_t end) {
    std::vector<float> audio;
    for (size_t piece = begin; piece < end; piece += kDecodeFrames) {
      const size_t piece_end = std::min(end, piece + kDecodeFrames);
      const std::vector<mx::array> frames(codes.begin() + piece,
                                          codes.begin() + piece_end);
      const mx::array samples =
          mx::astype(codec_stream.Push(mx::stack(frames, 1)), mx::float32);
      mx::eval(samples);
      audio.insert(audio.end(), samples.data<float>(),
                   samples.data<float>() + samples.size());
    }
    return audio;
  };
  std::optional<SpeechTurn> pinned;
  std::optional<SpeechTurn> latest;
  SynthesisResult result;
  try {
    for (size_t chunk_index = 0; chunk_index < chunks.size(); ++chunk_index) {
      const bool is_last = chunk_index + 1 == chunks.size();
      std::vector<SpeechTurn> history;
      for (const std::optional<SpeechTurn> &turn : {pinned, latest}) {
        if (turn.has_value()) {
          history.push_back(*turn);
        } else {
        }
      }
      const PromptFrames prompt = BuildFrames(
          tokenizer_, voice.references, history, chunks[chunk_index],
          codebook_count_, codebook_size_, semantic_begin_);
      result.prompt_row_count += prompt.frames.shape(0);
      std::vector<mx::array> codes;
      size_t emitted = 0;
      const bool ended = GenerateChunk(
          prompt, voice, sampling, max_frames, cancel,
          [&](const mx::array &frame) {
            on_frame(frame);
            codes.push_back(frame);
            if (stream && codes.size() - emitted >=
                              static_cast<size_t>(kStreamIntervalFrames)) {
              on_audio(decode(codes, emitted, codes.size()));
              emitted = codes.size();
            } else {
            }
          });
      result.frame_count += static_cast<int>(codes.size());
      if (!ended) {
        result.finish_reason = qwen3_asr::FinishReason::kLength;
      } else {
      }
      if (!codes.empty()) {
        const SpeechTurn turn = {chunks[chunk_index], mx::stack(codes, 1)};
        if (!pinned.has_value()) {
          pinned = turn;
        } else {
          latest = turn;
        }
      } else {
      }
      if (codes.size() > emitted || is_last) {
        std::vector<float> audio = decode(codes, emitted, codes.size());
        // Buffers sized by this prompt and chunk rarely fit a later one.
        mx::clear_cache();
        on_audio(std::move(audio));
      } else {
        mx::clear_cache();
      }
    }
  } catch (...) {
    mx::clear_cache();
    throw;
  }
  return result;
}

} // namespace fishaudio_drama3
