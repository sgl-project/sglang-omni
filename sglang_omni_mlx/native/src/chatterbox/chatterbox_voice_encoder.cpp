// SPDX-License-Identifier: Apache-2.0
// Chatterbox-Turbo's voice encoder.
#include "chatterbox/chatterbox_voice_encoder.h"

#include <algorithm>
#include <cmath>
#include <optional>
#include <string>

#include "chatterbox/chatterbox_speech_tokenizer.h"

namespace chatterbox {

namespace mx = mlx::core;

namespace {

constexpr int kMelCount = 40;
constexpr int kLstmCount = 3;
constexpr int kWindowFrames = 160;
// Windows start every round(16000 / 1.3 / 160) = 77 frames.
constexpr int kWindowStepFrames = 77;
// A trailing partial window is kept when it is at least this full.
constexpr double kMinimumWindowCoverage = 0.8;
constexpr int kWindowBatchSize = 32;
constexpr float kNormalizationEps = 1e-8f;
// Leading and trailing audio more than 20 dB below the loudest RMS frame is
// trimmed first.
constexpr int kTrimFrameSamples = 2048;
constexpr int kTrimHopSamples = 512;
constexpr float kTrimTopDecibels = 20.0f;
constexpr float kTrimRmsFloor = 1e-10f;
constexpr int kPairwiseBlockSize = 128;

// Note (Jiaxin Deng): float32 pairwise summation in blocks of 8 and 128, so the
// trim's RMS gate decides each frame the same way.
float PairwiseSum(const float *values, int count) {
  if (count < 8) {
    float sum = 0.0f;
    for (int i = 0; i < count; ++i) {
      sum += values[i];
    }
    return sum;
  } else if (count <= kPairwiseBlockSize) {
    float partial_sums[8];
    std::copy(values, values + 8, partial_sums);
    int i = 8;
    for (; i < count - count % 8; i += 8) {
      for (int j = 0; j < 8; ++j) {
        partial_sums[j] += values[i + j];
      }
    }
    float sum = ((partial_sums[0] + partial_sums[1]) +
                 (partial_sums[2] + partial_sums[3])) +
                ((partial_sums[4] + partial_sums[5]) +
                 (partial_sums[6] + partial_sums[7]));
    for (; i < count; ++i) {
      sum += values[i];
    }
    return sum;
  } else {
    int half = count / 2;
    half -= half % 8;
    return PairwiseSum(values, half) + PairwiseSum(values + half, count - half);
  }
}

std::vector<float> TrimSilence(const std::vector<float> &samples) {
  const int sample_count = static_cast<int>(samples.size());
  if (sample_count < kTrimFrameSamples) {
    return samples;
  } else {
  }
  const int frame_count =
      1 + (sample_count - kTrimFrameSamples) / kTrimHopSamples;
  std::vector<float> rms_decibels(frame_count);
  std::vector<float> squares(kTrimFrameSamples);
  for (int frame = 0; frame < frame_count; ++frame) {
    const float *start = samples.data() + frame * kTrimHopSamples;
    for (int i = 0; i < kTrimFrameSamples; ++i) {
      squares[i] = start[i] * start[i];
    }
    const float rms = std::sqrt(PairwiseSum(squares.data(), kTrimFrameSamples) /
                                static_cast<float>(kTrimFrameSamples));
    rms_decibels[frame] = 20.0f * std::log10(std::max(rms, kTrimRmsFloor));
  }
  const float threshold =
      *std::max_element(rms_decibels.begin(), rms_decibels.end()) -
      kTrimTopDecibels;
  int first_frame = 0;
  while (rms_decibels[first_frame] < threshold) {
    ++first_frame;
  }
  int last_frame = frame_count - 1;
  while (rms_decibels[last_frame] < threshold) {
    --last_frame;
  }
  const int start_sample = first_frame * kTrimHopSamples;
  const int end_sample = std::min(
      (last_frame + 1) * kTrimHopSamples + kTrimFrameSamples, sample_count);
  return std::vector<float>(samples.begin() + start_sample,
                            samples.begin() + end_sample);
}

} // namespace

VoiceEncoder::VoiceEncoder(const omni_common::Checkpoint &checkpoint)
    : checkpoint_(checkpoint), mel_filters_(SlaneyMelFilters16k(kMelCount)) {
  mx::eval(mel_filters_);
}

mx::array VoiceEncoder::EmbedWindows(const mx::array &mel_windows) const {
  mx::array x = mel_windows;
  for (int layer = 1; layer <= kLstmCount; ++layer) {
    const std::string prefix = "ve.lstm" + std::to_string(layer);
    const mx::array input_gates =
        mx::addmm(checkpoint_.Weight(prefix + ".bias"), x,
                  mx::transpose(checkpoint_.Weight(prefix + ".Wx")));
    std::optional<omni_common::LstmCell> state;
    std::vector<mx::array> hidden_states;
    hidden_states.reserve(kWindowFrames);
    for (int frame = 0; frame < x.shape(1); ++frame) {
      const mx::array gates = mx::squeeze(
          mx::slice(input_gates, {0, frame, 0},
                    {input_gates.shape(0), frame + 1, input_gates.shape(2)}),
          1);
      state = omni_common::LstmStep(gates, state,
                                    checkpoint_.Weight(prefix + ".Wh"));
      hidden_states.push_back(state->hidden);
    }
    x = mx::stack(hidden_states, -2);
  }
  const mx::array last_hidden =
      mx::squeeze(mx::slice(x, {0, x.shape(1) - 1, 0},
                            {x.shape(0), x.shape(1), x.shape(2)}),
                  1);
  const mx::array projected =
      mx::maximum(checkpoint_.Linear(last_hidden, "ve.proj"), mx::array(0.0f));
  return mx::divide(
      projected, mx::add(mx::linalg::norm(projected, std::vector<int>{1}, true),
                         mx::array(kNormalizationEps)));
}

mx::array VoiceEncoder::Embed(const std::vector<float> &samples_16k) const {
  const std::vector<float> trimmed = TrimSilence(samples_16k);
  mx::array spectrum = Spectrum16k(trimmed);
  mx::eval(spectrum);
  const int frame_count = spectrum.shape(0);
  const int bin_count = spectrum.shape(1);
  // Note (Jiaxin Deng): the magnitude is computed on the CPU as
  // larger * sqrt(fma(ratio, ratio, 1)), then squared and multiplied with the
  // filters by the CPU's BLAS, so it rounds as the float32 host math does.
  const mx::complex64_t *bins = spectrum.data<mx::complex64_t>();
  std::vector<float> power(static_cast<size_t>(frame_count) * bin_count);
  for (size_t i = 0; i < power.size(); ++i) {
    const float real = std::fabs(bins[i].real());
    const float imaginary = std::fabs(bins[i].imag());
    const float larger = std::max(real, imaginary);
    const float smaller = std::min(real, imaginary);
    float magnitude = 0.0f;
    if (larger > 0.0f) {
      const float ratio = smaller / larger;
      magnitude = larger * std::sqrt(std::fma(ratio, ratio, 1.0f));
    } else {
    }
    power[i] = magnitude * magnitude;
  }
  const mx::array power_array(power.begin(), {frame_count, bin_count});
  // [mel bands, frames]
  mx::array mel = mx::matmul(mel_filters_, mx::transpose(power_array),
                             mx::Device(mx::Device::cpu));
  mel = mx::copy(mx::transpose(mel));

  const int coverable_frames =
      std::max(frame_count - kWindowFrames + kWindowStepFrames, 0);
  int window_count = coverable_frames / kWindowStepFrames;
  const int remainder = coverable_frames % kWindowStepFrames;
  if (window_count == 0 ||
      static_cast<double>(remainder + kWindowFrames - kWindowStepFrames) /
              kWindowFrames >=
          kMinimumWindowCoverage) {
    ++window_count;
  } else {
  }
  const int target_frames =
      kWindowFrames + kWindowStepFrames * (window_count - 1);
  if (target_frames > frame_count) {
    mel = mx::pad(mel, {{0, target_frames - frame_count}, {0, 0}});
  } else {
  }
  std::vector<mx::array> windows;
  windows.reserve(window_count);
  for (int window = 0; window < window_count; ++window) {
    const int start = window * kWindowStepFrames;
    windows.push_back(
        mx::slice(mel, {start, 0}, {start + kWindowFrames, kMelCount}));
  }
  const mx::array mel_windows = mx::stack(windows);

  std::vector<mx::array> window_embeddings;
  for (int start = 0; start < window_count; start += kWindowBatchSize) {
    const int stop = std::min(start + kWindowBatchSize, window_count);
    window_embeddings.push_back(EmbedWindows(mx::slice(
        mel_windows, {start, 0, 0}, {stop, kWindowFrames, kMelCount})));
  }
  const mx::array mean_embedding =
      mx::stack({mx::mean(mx::concatenate(window_embeddings, 0), 0, false)});
  return mx::divide(
      mean_embedding,
      mx::add(mx::linalg::norm(mean_embedding, std::vector<int>{1}, true),
              mx::array(kNormalizationEps)));
}

} // namespace chatterbox
