// SPDX-License-Identifier: Apache-2.0
#include "qwen3_tts/qwen3_tts_codec.h"

#include <algorithm>
#include <mutex>

#include <mlx/memory.h>

namespace qwen3_tts {

namespace mx = mlx::core;

namespace {

// Note (Jiaxin Deng): the CUDA vocoder decodes a whole request in 300-frame
// chunks, each with up to 25 frames of left context.
constexpr int kChunkFrames = 300;
constexpr int kLeftContextFrames = 25;

// Note (Codex): on M5 Pro, this limit reduced long-chunk footprint without
// slowing generation. Note (Codex): MLX cache limits are process-wide; the
// native server owns one model worker.
constexpr size_t kCodecCacheLimitBytes = 512ULL * 1024 * 1024;
std::mutex codec_cache_limit_mutex;

class ScopedCodecCacheLimit {
public:
  explicit ScopedCodecCacheLimit(bool enabled)
      : lock_(codec_cache_limit_mutex), enabled_(enabled) {
    if (enabled_) {
      previous_cache_limit_bytes_ = mx::set_cache_limit(0);
      mx::set_cache_limit(
          std::min(previous_cache_limit_bytes_, kCodecCacheLimitBytes));
    } else {
    }
  }
  ~ScopedCodecCacheLimit() {
    if (enabled_) {
      mx::set_cache_limit(previous_cache_limit_bytes_);
    } else {
    }
  }
  ScopedCodecCacheLimit(const ScopedCodecCacheLimit &) = delete;
  ScopedCodecCacheLimit &operator=(const ScopedCodecCacheLimit &) = delete;

private:
  std::unique_lock<std::mutex> lock_;
  bool enabled_;
  size_t previous_cache_limit_bytes_ = 0;
};

} // namespace

SpeechStream::SpeechStream(const SpeechDecoder &decoder, bool streamed,
                           int first_chunk_frames)
    : decoder_(decoder), streamed_(streamed),
      first_chunk_frames_(
          first_chunk_frames > 0
              ? std::min(first_chunk_frames, kStreamedChunkFrames)
              : kStreamedChunkFrames),
      feature_state_(streamed ? decoder.NewStreamState() : std::nullopt),
      wave_state_(feature_state_
                      ? std::optional<SpeechDecoderState>(SpeechDecoderState{})
                      : std::nullopt) {}

int SpeechStream::ChunkFrames() const {
  if (!streamed_) {
    return kChunkFrames;
  } else if (emitted_frames_ == 0) {
    return first_chunk_frames_;
  } else {
    return kStreamedChunkFrames;
  }
}

std::vector<float> SpeechStream::Push(const mx::array &frame_codes) {
  frames_.push_back(frame_codes);
  if (frame_codes.data<int32_t>()[0] > 0) {
    ++voiced_frames_;
  } else {
  }
  const int chunk_frames = ChunkFrames();
  if (static_cast<int>(frames_.size()) - emitted_frames_ < chunk_frames) {
    return {};
  } else if (streamed_) {
    return DecodeChunk(emitted_frames_ + chunk_frames);
  } else {
    const std::vector<float> chunk =
        DecodeChunk(emitted_frames_ + chunk_frames);
    held_samples_.insert(held_samples_.end(), chunk.begin(), chunk.end());
    return {};
  }
}

std::vector<float> SpeechStream::Finish() {
  std::vector<float> samples = std::move(held_samples_);
  while (emitted_frames_ < static_cast<int>(frames_.size())) {
    const std::vector<float> chunk = DecodeChunk(std::min(
        static_cast<int>(frames_.size()), emitted_frames_ + ChunkFrames()));
    samples.insert(samples.end(), chunk.begin(), chunk.end());
  }
  // Note (Jiaxin Deng): as the CUDA vocoder's whole-request decode, the audio
  // keeps one frame of samples per frame whose first code is not 0.
  const size_t kept_samples =
      static_cast<size_t>(voiced_frames_) * decoder_.samples_per_frame();
  if (!streamed_ && kept_samples < samples.size()) {
    samples.resize(kept_samples);
  } else {
  }
  return samples;
}

std::vector<float> SpeechStream::DecodeChunk(int end_frame) {
  const ScopedCodecCacheLimit cache_limit(!streamed_);
  if (feature_state_) {
    return DecodeIncremental(end_frame);
  } else {
  }
  const int context_frames = std::min(kLeftContextFrames, emitted_frames_);
  const int start_frame = emitted_frames_ - context_frames;
  const std::vector<mx::array> window(frames_.begin() + start_frame,
                                      frames_.begin() + end_frame);
  const int samples_per_frame = decoder_.samples_per_frame();
  const mx::array audio =
      mx::slice(decoder_.Decode(mx::stack(window, 0)),
                {context_frames * samples_per_frame},
                {(end_frame - start_frame) * samples_per_frame});
  mx::eval(audio);
  emitted_frames_ = end_frame;
  std::vector<float> samples(audio.data<float>(),
                             audio.data<float>() + audio.size());
  // Note (Jiaxin Deng): a chunk's upsampling buffers run to gigabytes and
  // are not needed again until the next chunk, so MLX's cache returns them.
  mx::clear_cache();
  return samples;
}

std::vector<float> SpeechStream::DecodeIncremental(int end_frame) {
  std::vector<float> samples;
  const int samples_per_frame = decoder_.samples_per_frame();
  int feature_block_start = -1;
  std::optional<mx::array> features;
  while (emitted_frames_ < end_frame) {
    const int start = wave_state_->decoded_frames;
    const int block_start =
        (start / kStreamedChunkFrames) * kStreamedChunkFrames;
    if (feature_block_start != block_start) {
      const int block_end =
          std::min(end_frame, block_start + kStreamedChunkFrames);
      const std::vector<mx::array> window(frames_.begin() + block_start,
                                          frames_.begin() + block_end);
      mx::array codes = mx::stack(window, 0);
      std::optional<SpeechDecoderState> preview;
      if (block_end - block_start < kStreamedChunkFrames) {
        // A causal preview preserves the complete block's matrix shapes.
        // Commit front-end state only when every code in the block is real.
        preview = *feature_state_;
        codes = mx::concatenate(
            {codes,
             mx::broadcast_to(window.back(),
                              {kStreamedChunkFrames - (block_end - block_start),
                               codes.shape(1)})},
            0);
      } else {
      }
      features = decoder_.DecodeFeatures(codes, preview ? &*preview
                                                        : &*feature_state_);
      feature_block_start = block_start;
    } else {
    }
    const int offset = start - block_start;
    const int step_frames =
        offset == 0 ? kDefaultFirstChunkFrames : kStreamedChunkFrames - offset;
    const int stop = std::min(end_frame, start + step_frames);
    std::optional<SpeechDecoderState> preview;
    if (stop - start < step_frames) {
      // The waveform's substep shapes also remain fixed at a partial packet.
      preview = *wave_state_;
    } else {
    }
    SpeechDecoderState &state = preview ? *preview : *wave_state_;
    const int stride = features->shape(1) / kStreamedChunkFrames;
    const mx::array input =
        mx::slice(*features, {0, offset * stride, 0},
                  {1, (offset + step_frames) * stride, features->shape(2)});
    const mx::array audio =
        mx::slice(decoder_.DecodeSamples(input, step_frames, &state),
                  {(emitted_frames_ - start) * samples_per_frame},
                  {(stop - start) * samples_per_frame});
    // Retain evaluated features for the next substep of this same block.
    std::vector<mx::array> outputs = {audio, *features};
    for (const auto &[name, value] : feature_state_->convolution_history) {
      outputs.push_back(value);
    }
    for (const auto &history : feature_state_->attention) {
      if (history) {
        outputs.push_back(history->first);
        outputs.push_back(history->second);
      } else {
      }
    }
    if (!preview) {
      for (const auto &[name, value] : state.convolution_history) {
        outputs.push_back(value);
      }
    } else {
    }
    mx::eval(outputs);
    samples.insert(samples.end(), audio.data<float>(),
                   audio.data<float>() + audio.size());
    emitted_frames_ = stop;
  }
  mx::clear_cache();
  return samples;
}

} // namespace qwen3_tts
