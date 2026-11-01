// SPDX-License-Identifier: Apache-2.0
// Higgs Audio v3's audio codec decoder (raw codec frames to 24 kHz samples),
// the delay pattern the language model speaks in, and the streaming schedule
// that decodes rows as generation produces them.
#pragma once

#include <cstdint>
#include <vector>

#include "common/layers.h"
#include "mlx/mlx.h"

namespace higgs_tts {

constexpr int kCodebookCount = 8;
// Each codebook's 1024 codec entries, then its begin- and end-of-audio codes.
constexpr int kCodecEntryCount = 1024;
constexpr int kBeginOfAudioCode = 1024;
constexpr int kEndOfAudioCode = 1025;
constexpr int kAudioCodeCount = 1026;

class AudioCodec {
public:
  // weights are the codec's tensors with the checkpoint's codec prefix
  // removed, as stored (PyTorch layouts).
  explicit AudioCodec(omni_common::WeightMap weights);

  // Raw codec frames [frames, codebooks] int32, each code below
  // kCodecEntryCount, to float32 samples [frames * samples_per_frame].
  mlx::core::array Decode(const mlx::core::array &frames) const;
  int samples_per_frame() const;
  int sample_rate() const;

private:
  static omni_common::WeightMap PrepareWeights(omni_common::WeightMap weights);
  mlx::core::array ResidualUnit(const mlx::core::array &x,
                                const std::string &prefix, int dilation) const;

  omni_common::Checkpoint checkpoint_;
};

// Delayed rows [rows, codebooks] to raw frames [rows - codebooks + 1,
// codebooks]: codebook k of frame f is row f + k's. Begin- and end-of-audio
// codes left in a frame become entry 0.
mlx::core::array RawFrames(const std::vector<int32_t> &delayed_codes,
                           int first_row, int row_end);

// Note (Jiaxin Deng): the CUDA vocoder's streaming schedule
// (HiggsStreamingVocoderScheduler): the first 20 frames at row 27, the next
// window at row 150, then one every 75 rows, each leaving the last 4 frames
// for the next and decoded with 8 frames of left context.
class CodecStream {
public:
  explicit CodecStream(const AudioCodec &codec) : codec_(codec) {}

  // Adds one delayed row's codes [codebooks]; returns the samples a
  // completed window adds, empty until one does.
  std::vector<float> Push(const mlx::core::array &row_codes);
  // Decodes every frame not yet returned.
  std::vector<float> Finish();

private:
  std::vector<float> DecodeWindow(int frame_end, bool is_final);
  int row_count() const;

  const AudioCodec &codec_;
  std::vector<int32_t> delayed_codes_;
  int emitted_frames_ = 0;
  int next_decode_rows_ = 0;
};

} // namespace higgs_tts
