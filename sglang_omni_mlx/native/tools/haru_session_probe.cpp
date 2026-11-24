// SPDX-License-Identifier: Apache-2.0
// DuplexSession end-to-end test: a character card + a synthetic voice
// reference, speak-first, then a few silent user frames. Prints tokens, text
// and audio sums.
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <iostream>
#include <vector>

#include <CommonCrypto/CommonDigest.h>

#include "common/audio.h"
#include "common/layers.h"
#include "common/tokenizer.h"
#include "haru/haru_audio.h"
#include "haru/haru_config.h"
#include "haru/haru_model.h"
#include "haru/haru_session.h"

namespace mx = mlx::core;

int main(int argc, char **argv) {
  if (argc < 2 || argc > 4) {
    std::cerr
        << "usage: haru_session_probe <model-dir> [seed] [quantize-bits]\n";
    return 2;
  }
  try {
    const std::filesystem::path directory = argv[1];
    const int seed = argc >= 3 ? std::stoi(argv[2]) : 42;
    const int quantize_bits = argc >= 4 ? std::stoi(argv[3]) : 0;
    const auto load_start = std::chrono::steady_clock::now();
    const haru::HaruConfig config =
        haru::HaruConfig::FromFile(directory / "config.json");
    omni_common::Checkpoint checkpoint(
        omni_common::LoadSafetensors(directory),
        omni_common::ReadJson(directory / "config.json"));
    haru::HaruModel model(directory, quantize_bits);
    omni_common::Tokenizer tokenizer(directory);
    const auto load_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                             std::chrono::steady_clock::now() - load_start)
                             .count();

    const auto infer_start = std::chrono::steady_clock::now();

    // A 3 s reference voice at 16 kHz, decoded at 24 kHz through
    // PrepareReference (WAV decode + resample + silence trim).
    const int ref_samples = 3 * 16000;
    std::vector<float> ref16k(ref_samples);
    for (int i = 0; i < ref_samples; ++i) {
      ref16k[i] = 0.05f * std::sin(2.0f * 3.14159265f * 220.0f * i / 16000.0f);
    }
    const std::string wav =
        omni_common::WavBytes(omni_common::Pcm16Bytes(ref16k), 16000);
    std::vector<float> reference = haru::PrepareReference(wav);

    haru::DuplexConfig session_config;
    session_config.speak_first = true;
    session_config.seed = seed;
    session_config.temperature = 0.0f; // greedy, deterministic
    haru::DuplexSession session(model, tokenizer, session_config);

    haru::Frame frame = session.Start(
        "You are a friendly voice assistant named Haru.", reference);
    std::vector<float> all_audio = frame.audio;
    std::cout << "frame " << frame.index << " token " << frame.token
              << " text \"" << frame.text << "\" audio_sum " <<
        [&] {
          double s = 0;
          for (float v : frame.audio)
            s += v;
          return s;
        }() << "\n";

    std::vector<float> silence(config.frame_samples, 0.0f);
    for (int i = 0; i < 3; ++i) {
      frame = session.Step(silence);
      all_audio.insert(all_audio.end(), frame.audio.begin(), frame.audio.end());
      std::cout << "frame " << frame.index << " token " << frame.token
                << " text \"" << frame.text << "\" done " << frame.done << "\n";
    }

    // A second turn on the same session: a short voiced input followed by
    // silence, so the golden covers more than the opening frames.
    std::vector<float> speech(config.frame_samples);
    for (int i = 0; i < config.frame_samples; ++i) {
      speech[i] = 0.05f * std::sin(2.0f * 3.14159265f * 180.0f * i /
                                   static_cast<float>(config.sample_rate));
    }
    std::vector<float> turn2_audio;
    std::string turn2_tokens;
    for (int i = 0; i < 2; ++i) {
      frame = session.Step(speech);
      turn2_audio.insert(turn2_audio.end(), frame.audio.begin(),
                         frame.audio.end());
      turn2_tokens += std::to_string(frame.token) + " ";
    }
    for (int i = 0; i < 2; ++i) {
      frame = session.Step(silence);
      turn2_audio.insert(turn2_audio.end(), frame.audio.begin(),
                         frame.audio.end());
      turn2_tokens += std::to_string(frame.token) + " ";
    }
    std::cout << "turn2_tokens " << turn2_tokens << "\n";
    unsigned char turn2_digest[CC_SHA256_DIGEST_LENGTH];
    CC_SHA256(turn2_audio.data(), turn2_audio.size() * sizeof(float),
              turn2_digest);
    char turn2_hex[CC_SHA256_DIGEST_LENGTH * 2 + 1];
    for (int i = 0; i < CC_SHA256_DIGEST_LENGTH; ++i) {
      std::snprintf(turn2_hex + 2 * i, 3, "%02x", turn2_digest[i]);
    }
    turn2_hex[CC_SHA256_DIGEST_LENGTH * 2] = '\0';
    std::cout << "turn2_audio_sha256 " << turn2_hex << "\n";

    // A deterministic fingerprint of the decoded audio, for the audio golden.
    unsigned char digest[CC_SHA256_DIGEST_LENGTH];
    CC_SHA256(all_audio.data(), all_audio.size() * sizeof(float), digest);
    char hex[CC_SHA256_DIGEST_LENGTH * 2 + 1];
    for (int i = 0; i < CC_SHA256_DIGEST_LENGTH; ++i) {
      std::snprintf(hex + 2 * i, 3, "%02x", digest[i]);
    }
    hex[CC_SHA256_DIGEST_LENGTH * 2] = '\0';
    const auto infer_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                              std::chrono::steady_clock::now() - infer_start)
                              .count();
    std::cout << "audio_sha256 " << hex << "\n";
    std::cout << "load_ms " << load_ms << "\n";
    std::cout << "infer_ms " << infer_ms << "\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "haru_session_probe: " << error.what() << "\n";
    return 1;
  }
}
