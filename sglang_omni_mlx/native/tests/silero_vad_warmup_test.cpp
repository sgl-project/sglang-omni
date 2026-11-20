// SPDX-License-Identifier: Apache-2.0
// Guards the startup warmup in SileroVAD: after the model primed its
// kernels with a throwaway stream, a fresh stream must produce exactly
// the probabilities the reference table below records (captured from
// the same checkpoint before the warmup existed). Any state the warmup
// leaks into later streams, and any accidental math change, shifts
// these numbers.
//
//   silero_vad_warmup_test <silero model directory> [--print]
//
// --print prints the probabilities instead of checking them (how the
// reference table was captured).
#include <cmath>
#include <cstdio>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "silero_vad/silero_vad.h"

namespace {

constexpr int kChunks = 40;

// Captured from silero-vad-v6 (2ebf4a5e) without the warmup; the Signal
// below is deterministic, so these are the model's exact outputs.
constexpr float kReference[kChunks] = {
    0.00358793838f, 0.0018748492f,  0.00276426692f, 0.00560356071f,
    0.00550419232f, 0.00566963386f, 0.00398608297f, 0.00415684748f,
    0.00430993782f, 0.00403698254f, 0.00278176623f, 0.00402937829f,
    0.00332010747f, 0.00475009345f, 0.00245313998f, 0.00697505893f,
    0.00497487839f, 0.00399414822f, 0.0031836608f,  0.0162935946f,
    0.00364598888f, 0.00361632765f, 0.00537434965f, 0.00322319195f,
    0.00458811736f, 0.00383747276f, 0.00253334665f, 0.00507128378f,
    0.00313595217f, 0.00225963979f, 0.00325980037f, 0.00247484585f,
    0.00332444627f, 0.00458183f,    0.00418136455f, 0.0084137097f,
    0.00324639678f, 0.00525660673f, 0.00759702409f, 0.00646079564f,
};

void Require(bool condition, std::string_view message) {
  if (!condition) {
    throw std::runtime_error(std::string(message));
  }
}

// A deterministic non-zero signal: all zeros could hide a state leak.
std::vector<float> Signal() {
  std::vector<float> samples(kChunks * silero_vad::kChunkSamples);
  unsigned state = 12345;
  for (float &sample : samples) {
    state = state * 1664525u + 1013904223u;
    sample = static_cast<float>((state >> 8) % 2000) / 1000.0f - 1.0f;
  }
  return samples;
}

} // namespace

int main(int argc, char **argv) {
  if (argc < 2) {
    std::cerr << "usage: " << argv[0]
              << " <silero model directory> [--print]\n";
    return 2;
  }
  const bool print = argc > 2 && std::string(argv[2]) == "--print";
  const std::vector<float> samples = Signal();

  silero_vad::SileroVAD model(argv[1]);
  Require(model.WarmedUp(),
          "the constructor must prime the kernels with its throwaway "
          "warmup stream (reverting the warmup fails this check)");
  silero_vad::StreamState state;
  const std::vector<float> probabilities =
      model.Feed(samples.data(), kChunks, state);
  Require(probabilities.size() == kChunks, "one probability per chunk");
  for (int chunk = 0; chunk < kChunks; ++chunk) {
    if (print) {
      std::printf("    %.9gf,\n", static_cast<double>(probabilities[chunk]));
      continue;
    }
    Require(probabilities[chunk] == kReference[chunk],
            "a fresh stream after the warmup must match the reference "
            "bit for bit (chunk " +
                std::to_string(chunk) + ")");
  }
  if (!print) {
    std::cout << "ok: " << kChunks
              << " probabilities identical to the reference\n";
  }
  return 0;
}
