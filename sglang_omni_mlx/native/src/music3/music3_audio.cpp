// SPDX-License-Identifier: Apache-2.0
#include "music3/music3_model.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <utility>

namespace music3 {

uint64_t AcousticSeed(uint64_t seed, int chunk_index) {
  // Note (Akazaakane): the reference uses BLAKE2b-64 with this personalization
  // for each window.
  constexpr std::array<uint64_t, 8> initialization = {
      0x6a09e667f3bcc908ULL, 0xbb67ae8584caa73bULL, 0x3c6ef372fe94f82bULL,
      0xa54ff53a5f1d36f1ULL, 0x510e527fade682d1ULL, 0x9b05688c2b3e6c1fULL,
      0x1f83d9abfb41bd6bULL, 0x5be0cd19137e2179ULL};
  constexpr uint8_t permutations[12][16] = {
      {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
      {14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3},
      {11, 8, 12, 0, 5, 2, 15, 13, 10, 14, 3, 6, 7, 1, 9, 4},
      {7, 9, 3, 1, 13, 12, 11, 14, 2, 6, 5, 10, 4, 0, 15, 8},
      {9, 0, 5, 7, 2, 4, 10, 15, 14, 1, 11, 12, 6, 8, 3, 13},
      {2, 12, 6, 10, 0, 11, 8, 3, 4, 13, 7, 5, 15, 14, 1, 9},
      {12, 5, 1, 15, 14, 13, 4, 10, 0, 7, 6, 3, 9, 2, 8, 11},
      {13, 11, 7, 14, 12, 1, 3, 9, 5, 0, 15, 4, 8, 6, 2, 10},
      {6, 15, 14, 9, 11, 3, 0, 8, 12, 2, 13, 7, 1, 4, 10, 5},
      {10, 2, 8, 4, 7, 6, 1, 5, 15, 11, 9, 14, 3, 12, 13, 0},
      {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
      {14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3}};
  std::array<uint8_t, 128> bytes{};
  size_t length = 0;
  const auto append = [&](uint64_t value, int count) {
    for (int i = 0; i < count; ++i)
      bytes[length++] = static_cast<uint8_t>(value >> (8 * i));
  };
  append(seed, 8);
  append(3, 4);
  for (char character : std::string("dit"))
    bytes[length++] = character;
  const std::string index = std::to_string(chunk_index);
  append(index.size(), 4);
  for (char character : index)
    bytes[length++] = character;
  std::array<uint64_t, 16> message{};
  for (size_t i = 0; i < bytes.size(); ++i)
    message[i / 8] |= static_cast<uint64_t>(bytes[i]) << (8 * (i % 8));
  auto hash = initialization;
  hash[0] ^= 0x01010008;
  const std::string personalization = "minimax-ttm";
  for (size_t i = 0; i < personalization.size(); ++i)
    hash[6 + i / 8] ^=
        static_cast<uint64_t>(static_cast<uint8_t>(personalization[i]))
        << (8 * (i % 8));
  std::array<uint64_t, 16> state{};
  std::copy(hash.begin(), hash.end(), state.begin());
  std::copy(initialization.begin(), initialization.end(), state.begin() + 8);
  state[12] ^= length;
  state[14] = ~state[14];
  const auto mix = [&](int a, int b, int c, int d, uint64_t x, uint64_t y) {
    state[a] = state[a] + state[b] + x;
    state[d] = std::rotr(state[d] ^ state[a], 32);
    state[c] += state[d];
    state[b] = std::rotr(state[b] ^ state[c], 24);
    state[a] = state[a] + state[b] + y;
    state[d] = std::rotr(state[d] ^ state[a], 16);
    state[c] += state[d];
    state[b] = std::rotr(state[b] ^ state[c], 63);
  };
  for (const auto &order : permutations) {
    mix(0, 4, 8, 12, message[order[0]], message[order[1]]);
    mix(1, 5, 9, 13, message[order[2]], message[order[3]]);
    mix(2, 6, 10, 14, message[order[4]], message[order[5]]);
    mix(3, 7, 11, 15, message[order[6]], message[order[7]]);
    mix(0, 5, 10, 15, message[order[8]], message[order[9]]);
    mix(1, 6, 11, 12, message[order[10]], message[order[11]]);
    mix(2, 7, 8, 13, message[order[12]], message[order[13]]);
    mix(3, 4, 9, 14, message[order[14]], message[order[15]]);
  }
  return (hash[0] ^ state[0] ^ state[8]) & 0x7fffffffffffffffULL;
}

std::vector<float> ResampleStereo(const mlx::core::array &waveform) {
  // Note (Akazaakane): match the reference's six-lobe Hann sinc, 0.99 rolloff
  // and zero padding.
  constexpr int input_period = 441;
  constexpr int output_period = 320;
  constexpr int lobes = 6;
  constexpr float base_frequency = output_period * 0.99f;
  constexpr int width = 9;
  const int input_samples = waveform.shape(1);
  mlx::core::eval(waveform);
  const float *source = waveform.data<float>();
  const int output_samples = static_cast<int>(
      (static_cast<int64_t>(input_samples) * output_period + input_period - 1) /
      input_period);
  std::array<std::vector<std::pair<int, float>>, output_period> filters;
  for (int phase = 0; phase < output_period; ++phase) {
    for (int index = -width; index < width + input_period; ++index) {
      const float position = (-static_cast<float>(phase) / output_period +
                              static_cast<float>(index) / input_period) *
                             base_frequency;
      if (std::abs(position) >= lobes) {
        continue;
      } else {
      }
      const float angle = position * static_cast<float>(M_PI);
      const float window = std::cos(angle / lobes / 2);
      const float sinc = angle == 0 ? 1.0f : std::sin(angle) / angle;
      filters[phase].emplace_back(
          index, sinc * (window * window * (base_frequency / input_period)));
    }
  }
  std::vector<float> samples(static_cast<size_t>(output_samples) * 2);
  for (int index = 0; index < output_samples; ++index) {
    const int source_start = index / output_period * input_period;
    for (int channel = 0; channel < 2; ++channel) {
      float value = 0;
      for (const auto &[offset, weight] : filters[index % output_period]) {
        const int source_index = source_start + offset;
        if (source_index >= 0 && source_index < input_samples) {
          value += source[channel * input_samples + source_index] * weight;
        } else {
        }
      }
      if (!std::isfinite(value)) {
        throw std::runtime_error("Music3 generated nonfinite audio");
      } else {
        samples[2 * index + channel] = value;
      }
    }
  }
  return samples;
}

std::string EncodeAudio(const std::vector<float> &samples, bool pcm) {
  std::string bytes;
  bytes.reserve(samples.size() * 2 + (pcm ? 0 : 44));
  const auto append = [&](uint32_t value, int count) {
    for (int i = 0; i < count; ++i)
      bytes.push_back(static_cast<char>(value >> (8 * i)));
  };
  if (!pcm) {
    bytes += "RIFF";
    append(36 + samples.size() * 2, 4);
    bytes += "WAVEfmt ";
    append(16, 4);
    append(1, 2);
    append(2, 2);
    append(kSampleRate, 4);
    append(kSampleRate * 4, 4);
    append(4, 2);
    append(16, 2);
    bytes += "data";
    append(samples.size() * 2, 4);
  } else {
  }
  for (float sample : samples) {
    if (!std::isfinite(sample)) {
      throw std::runtime_error("Music3 generated nonfinite audio");
    } else {
    }
    const int16_t value = static_cast<int16_t>(
        std::clamp(std::nearbyint(sample * 32768.0f), -32768.0f, 32767.0f));
    append(static_cast<uint16_t>(value), 2);
  }
  return bytes;
}

} // namespace music3
