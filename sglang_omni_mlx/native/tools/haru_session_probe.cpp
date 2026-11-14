// SPDX-License-Identifier: Apache-2.0
// DuplexSession end-to-end test: a character card + a synthetic voice reference,
// speak-first, then a few silent user frames. Prints tokens, text and audio sums.
#include <cmath>
#include <filesystem>
#include <iostream>
#include <vector>

#include "common/audio.h"
#include "common/layers.h"
#include "common/tokenizer.h"
#include "haru/haru_audio.h"
#include "haru/haru_codec.h"
#include "haru/haru_config.h"
#include "haru/haru_head.h"
#include "haru/haru_model.h"
#include "haru/haru_session.h"

namespace mx = mlx::core;

int main(int argc, char **argv) {
  if (argc < 2 || argc > 3) {
    std::cerr << "usage: haru_session_probe <model-dir> [seed]\n";
    return 2;
  }
  try {
    const std::filesystem::path directory = argv[1];
    const int seed = argc >= 3 ? std::stoi(argv[2]) : 42;
    const haru::HaruConfig config =
        haru::HaruConfig::FromFile(directory / "config.json");
    omni_common::Checkpoint checkpoint(
        omni_common::LoadSafetensors(directory),
        omni_common::ReadJson(directory / "config.json"));
    haru::HaruModel model(directory);
    haru::VibeVoiceCodec codec(checkpoint, config.codec);
    haru::MeanFlowHead head(checkpoint, config.head);
    omni_common::Tokenizer tokenizer(directory);

    // A 3 s reference voice at 16 kHz, decoded at 24 kHz through
    // PrepareReferenceBytes (WAV decode + resample + silence trim).
    const int ref_samples = 3 * 16000;
    std::vector<float> ref16k(ref_samples);
    for (int i = 0; i < ref_samples; ++i) {
      ref16k[i] =
          0.05f * std::sin(2.0f * 3.14159265f * 220.0f * i / 16000.0f);
    }
    const std::string wav =
        omni_common::WavBytes(omni_common::Pcm16Bytes(ref16k), 16000);
    std::vector<float> reference = haru::PrepareReferenceBytes(wav);

    haru::DuplexConfig session_config;
    session_config.speak_first = true;
    session_config.seed = seed;
    session_config.temperature = 0.0f;  // greedy, deterministic
    haru::DuplexSession session(model, codec, head, tokenizer, session_config);

    haru::Frame frame =
        session.Start("You are a friendly voice assistant named Haru.",
                      reference);
    std::cout << "frame " << frame.index << " token " << frame.token
              << " text \"" << frame.text << "\" audio_sum "
              << [&] {
                   double s = 0;
                   for (float v : frame.audio) s += v;
                   return s;
                 }()
              << "\n";

    std::vector<float> silence(config.frame_samples, 0.0f);
    for (int i = 0; i < 3; ++i) {
      frame = session.Step(silence);
      std::cout << "frame " << frame.index << " token " << frame.token
                << " text \"" << frame.text << "\" done " << frame.done << "\n";
    }
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "haru_session_probe: " << error.what() << "\n";
    return 1;
  }
}
