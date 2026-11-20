// SPDX-License-Identifier: Apache-2.0
// The file path (PredictProbabilities) must return exactly the per-chunk
// path's probabilities while syncing far less often.
//   silero_vad_batched_test --model-path DIR
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include "silero_vad/silero_vad.h"

int main(int argc, char **argv) {
  std::string model_path;
  for (int i = 1; i < argc; ++i) {
    if (std::string(argv[i]) == "--model-path") {
      model_path = argv[++i];
    }
  }
  if (model_path.empty()) {
    std::fprintf(stderr, "usage: silero_vad_batched_test --model-path DIR\n");
    return 1;
  }
  const silero_vad::SileroVAD vad(model_path);

  // 400 chunks plus a tail, so padding and the final partial block run too.
  std::mt19937 rng(7);
  std::normal_distribution<float> noise(0.0f, 0.1f);
  std::vector<float> samples(400 * silero_vad::kChunkSamples + 123);
  for (float &sample : samples) {
    sample = noise(rng);
  }

  const std::vector<float> batched = vad.PredictProbabilities(samples);

  std::vector<float> padded(samples);
  padded.resize((samples.size() + silero_vad::kChunkSamples - 1) /
                    silero_vad::kChunkSamples * silero_vad::kChunkSamples,
                0.0f);
  silero_vad::StreamState state;
  std::vector<float> per_chunk;
  for (size_t offset = 0; offset < padded.size();
       offset += silero_vad::kChunkSamples) {
    per_chunk.push_back(vad.Feed(padded.data() + offset, 1, state)[0]);
  }

  if (batched.size() != per_chunk.size()) {
    std::fprintf(stderr, "size: %zu vs %zu\n", batched.size(),
                 per_chunk.size());
    return 1;
  }
  for (size_t i = 0; i < batched.size(); ++i) {
    if (batched[i] != per_chunk[i]) {
      std::fprintf(stderr, "chunk %zu: %.9g vs %.9g\n", i, batched[i],
                   per_chunk[i]);
      return 1;
    }
  }
  // 401 chunks must cost two GPU round trips, not one per chunk; without the
  // batched path this accessor does not exist and the count would not hold.
  const size_t want_syncs = (per_chunk.size() + 249) / 250;
  if (vad.FilePathSyncs() != want_syncs) {
    std::fprintf(stderr, "syncs: %zu vs %zu\n", vad.FilePathSyncs(),
                 want_syncs);
    return 1;
  }
  std::printf("ok: %zu probabilities identical, %zu syncs\n", batched.size(),
              vad.FilePathSyncs());
  return 0;
}
