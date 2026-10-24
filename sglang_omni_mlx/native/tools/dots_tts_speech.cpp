// SPDX-License-Identifier: Apache-2.0
// Synthesizes speech with the native dots.tts runtime and writes one WAV file
// for parity checks. Run with --help for the full option table.
//
//   dots_tts_speech --model-path DIR --text TEXT [--ref-audio a.wav]
//     [--ref-text S] [--num-steps N] [--guidance-scale F] [--speaker-scale F]
//     [--seed N] [--max-generate-length N] [--eos-threshold F]
//     [--language L] [--no-trim-onset] [--out out.wav]
#include <chrono>
#include <fstream>
#include <iostream>
#include <optional>
#include <sstream>
#include <string>

#include "common/audio.h"
#include "common/cli.h"
#include "dots_tts/dots_tts_model.h"

namespace {

// note (guozhihao-224): parses a CLI value naming its flag, so a bad number
// fails with a readable message instead of an uncaught stoi/stof throw.
int ParseInt(const std::string &flag, const std::string &raw) {
  try {
    return std::stoi(raw);
  } catch (const std::exception &) {
    throw std::invalid_argument(flag + " wants an integer, got " + raw);
  }
}

float ParseFloat(const std::string &flag, const std::string &raw) {
  try {
    return std::stof(raw);
  } catch (const std::exception &) {
    throw std::invalid_argument(flag + " wants a number, got " + raw);
  }
}

} // namespace

int main(int argc, char **argv) {
  std::string model_path;
  std::string out_path = "out.wav";
  std::string ref_audio;
  std::string ref_text;
  dots_tts::GenerationOptions options;
  std::string text;
  try {
    const std::vector<cli::Arg> kOptions = {
        {"--model-path", true, "<path>",
         "converted dots.tts checkpoint directory",
         [&](const std::string &v) { model_path = v; }},
        {"--out", true, "<wav>", "output WAV path",
         [&](const std::string &v) { out_path = v; }},
        {"--text", true, "<text>", "text to synthesize",
         [&](const std::string &v) { text = v; }},
        {"--ref-audio", true, "<wav>", "reference audio for voice cloning",
         [&](const std::string &v) { ref_audio = v; }},
        {"--ref-text", true, "<text>", "reference transcript",
         [&](const std::string &v) { ref_text = v; }},
        {"--num-steps", true, "<int>", "flow-matching steps",
         [&](const std::string &v) {
           options.num_steps = ParseInt("--num-steps", v);
         }},
        {"--guidance-scale", true, "<float>", "classifier-free guidance scale",
         [&](const std::string &v) {
           options.guidance_scale = ParseFloat("--guidance-scale", v);
         }},
        {"--speaker-scale", true, "<float>", "speaker embedding scale",
         [&](const std::string &v) {
           options.speaker_scale = ParseFloat("--speaker-scale", v);
         }},
        {"--seed", true, "<int>", "random seed",
         [&](const std::string &v) { options.seed = ParseInt("--seed", v); }},
        {"--max-generate-length", true, "<int>",
         "max latent patches to generate",
         [&](const std::string &v) {
           options.max_generate_length = ParseInt("--max-generate-length", v);
         }},
        {"--eos-threshold", true, "<float>",
         "EOS probability that stops generation",
         [&](const std::string &v) {
           options.eos_threshold = ParseFloat("--eos-threshold", v);
         }},
        {"--language", true, "<code>", "speech language tag (zh, en, yue, ...)",
         [&](const std::string &v) { options.language = v; }},
        {"--no-trim-onset", false, nullptr, "keep the vocoder onset transient",
         [&](const std::string &) { options.trim_onset = false; }},
    };
    if (cli::ParseCommandLine(
            argc, argv, kOptions,
            [&](const std::string &token) { text = token; })) {
      return 0;
    }
    if (model_path.empty()) {
      throw std::invalid_argument("--model-path is required");
    }
    if (text.empty()) {
      throw std::invalid_argument("--text is required");
    }

    const auto load_started = std::chrono::steady_clock::now();
    const dots_tts::DotsTtsModel model(model_path);
    std::cerr << "loaded in "
              << std::chrono::duration<double>(
                     std::chrono::steady_clock::now() - load_started)
                     .count()
              << " s\n";

    std::optional<std::string> prompt_wav;
    if (!ref_audio.empty()) {
      std::ifstream stream(ref_audio, std::ios::binary);
      std::ostringstream bytes;
      bytes << stream.rdbuf();
      prompt_wav = bytes.str();
    }

    const auto started = std::chrono::steady_clock::now();
    const dots_tts::SynthesisResult result = model.Generate(
        text, prompt_wav,
        ref_text.empty() ? std::optional<std::string>(std::nullopt)
                         : std::optional<std::string>(ref_text),
        options);
    const std::string wav_bytes =
        qwen3_asr::EncodeWav(result.audio, result.sample_rate);
    std::ofstream out(out_path, std::ios::binary);
    out.write(wav_bytes.data(), static_cast<std::streamsize>(wav_bytes.size()));
    const double seconds = std::chrono::duration<double>(
                               std::chrono::steady_clock::now() - started)
                               .count();
    std::cout << "wrote " << out_path << " ("
              << result.audio.size() / result.sample_rate << " s, "
              << result.num_patches << " patches, " << seconds << " s)\n";
  } catch (const std::exception &error) {
    std::cerr << argv[0] << ": " << error.what() << "\n";
    return 2;
  }
  return 0;
}