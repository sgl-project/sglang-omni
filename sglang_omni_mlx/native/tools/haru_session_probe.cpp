// SPDX-License-Identifier: Apache-2.0
// One seeded greedy DuplexSession: a character card, a synthetic voice
// reference, speak-first, then silent and voiced user frames. Prints each
// frame's token and text, and the SHA-256 of the model's audio.
#include <cmath>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include <CommonCrypto/CommonDigest.h>

#include "common/audio.h"
#include "common/layers.h"
#include "common/tokenizer.h"
#include "haru/haru_audio.h"
#include "haru/haru_model.h"
#include "haru/haru_session.h"

namespace {

constexpr int kReferenceSampleRate = 16000;
constexpr int kReferenceSeconds = 3;
constexpr float kToneAmplitude = 0.05f;

std::vector<float> Tone(int sample_count, int sample_rate, float frequency) {
  std::vector<float> samples(static_cast<size_t>(sample_count));
  for (int sample = 0; sample < sample_count; ++sample) {
    samples[static_cast<size_t>(sample)] =
        kToneAmplitude *
        std::sin(2.0f * static_cast<float>(M_PI) * frequency *
                 static_cast<float>(sample) / static_cast<float>(sample_rate));
  }
  return samples;
}

} // namespace

int main(int argc, char **argv) {
  if (argc < 2 || argc > 4 ||
      (argc == 4 && std::string(argv[3]) != "--full-precision")) {
    std::cerr << "usage: haru_session_probe <model-dir> [seed] "
                 "[--full-precision]\n";
    return 2;
  } else {
  }
  try {
    if (argc == 4) {
      omni_common::SetLoadTimeQuantization(false);
    } else {
    }
    const haru::HaruModel model(argv[1]);
    const omni_common::Tokenizer tokenizer(argv[1]);
    const haru::HaruConfig &config = model.config();

    haru::DuplexConfig session_config;
    session_config.speak_first = true;
    session_config.seed = argc >= 3 ? std::stoi(argv[2]) : 42;
    session_config.temperature = 0.0f;
    haru::DuplexSession session(model, tokenizer, session_config);

    const std::vector<float> silence(static_cast<size_t>(config.frame_samples));
    const std::vector<float> voice =
        Tone(config.frame_samples, config.sample_rate, 180.0f);
    const std::string reference_wav = omni_common::WavBytes(
        omni_common::Pcm16Bytes(Tone(kReferenceSeconds * kReferenceSampleRate,
                                     kReferenceSampleRate, 220.0f)),
        kReferenceSampleRate);
    std::vector<haru::Frame> frames = {session.Start(
        "You are a friendly voice assistant named Haru.",
        haru::PrepareReference(reference_wav, config.sample_rate))};
    for (const std::vector<float> *user_frame :
         {&silence, &silence, &silence, &voice, &voice, &silence, &silence}) {
      frames.push_back(session.Step(*user_frame));
    }

    std::vector<float> audio;
    for (const haru::Frame &frame : frames) {
      std::cout << "frame " << frame.index << " token " << frame.token
                << " text \"" << frame.text << "\"\n";
      audio.insert(audio.end(), frame.audio.begin(), frame.audio.end());
    }
    unsigned char digest[CC_SHA256_DIGEST_LENGTH];
    CC_SHA256(audio.data(), static_cast<CC_LONG>(audio.size() * sizeof(float)),
              digest);
    std::cout << "audio_sha256 ";
    for (const unsigned char byte : digest) {
      std::cout << std::hex << std::setw(2) << std::setfill('0')
                << static_cast<int>(byte);
    }
    std::cout << "\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "haru_session_probe: " << error.what() << "\n";
    return 1;
  }
}
