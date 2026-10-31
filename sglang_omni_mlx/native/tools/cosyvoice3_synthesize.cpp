// SPDX-License-Identifier: Apache-2.0
// Synthesizes speech with the native Fun-CosyVoice3 runtime and prints one
// JSON line of timings; with --tokens-output, also the text ids, the sampled
// speech tokens and the first step's logits, for parity checks against the
// reference. --tokens-input vocodes given speech tokens instead of sampling.
// --reference-audio clones the voice of a WAV file, with its transcript from
// --reference-text; the tokens output then also holds what it encodes to.
// --reference-input takes that encoding from a JSON file of the same fields
// instead, such as one from the reference front end.
//
//   cosyvoice3_synthesize --model-path DIR --text TEXT [--instructions I]
//       [--reference-audio ref.wav [--reference-text T]]
//       [--seed N] [--temperature T] [--top-p P] [--top-k K]
//       [--repetition-penalty R] [--max-new-tokens N] [--output out.wav]
//       [--tokens-output tokens.json] [--tokens-input tokens.json]
//       [--mel-output mel.npy] [--mel-input mel.npy]
//
// --mel-output saves the flow's mel; --mel-input vocodes a given mel.
#include <atomic>
#include <chrono>
#include <fstream>
#include <iostream>
#include <iterator>
#include <optional>
#include <string>
#include <vector>

#include "cosyvoice3/cosyvoice3_model.h"
#include "nlohmann/json.hpp"
#include "voxtral_tts/voxtral_tts_codec.h"

namespace {

double SecondsSince(std::chrono::steady_clock::time_point start) {
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
      .count();
}

} // namespace

int main(int argc, char **argv) {
  namespace mx = mlx::core;
  std::string model_path;
  cosyvoice3::SpeechText text;
  cosyvoice3::SamplingOptions options;
  std::string output_path;
  std::string tokens_output_path;
  std::string tokens_input_path;
  std::string mel_output_path;
  std::string mel_input_path;
  std::string reference_audio_path;
  std::string reference_input_path;
  for (int i = 1; i < argc; ++i) {
    const std::string argument = argv[i];
    if (i + 1 >= argc) {
      std::cerr << argv[0] << ": missing value for " << argument << "\n";
      return 2;
    } else {
    }
    const std::string value = argv[++i];
    if (argument == "--model-path") {
      model_path = value;
    } else if (argument == "--text") {
      text.text = value;
    } else if (argument == "--instructions") {
      text.instructions = omni_common::StripUnicodeWhitespace(value);
    } else if (argument == "--reference-audio") {
      reference_audio_path = value;
    } else if (argument == "--reference-input") {
      reference_input_path = value;
    } else if (argument == "--reference-text") {
      text.reference_text = omni_common::StripUnicodeWhitespace(value);
    } else if (argument == "--seed") {
      options.seed = std::stoull(value);
    } else if (argument == "--temperature") {
      options.temperature = std::stof(value);
    } else if (argument == "--top-p") {
      options.top_p = std::stof(value);
    } else if (argument == "--top-k") {
      options.top_k = std::stoi(value);
    } else if (argument == "--repetition-penalty") {
      options.repetition_penalty = std::stof(value);
    } else if (argument == "--max-new-tokens") {
      options.max_new_tokens = std::stoi(value);
    } else if (argument == "--output") {
      output_path = value;
    } else if (argument == "--tokens-output") {
      tokens_output_path = value;
    } else if (argument == "--tokens-input") {
      tokens_input_path = value;
    } else if (argument == "--mel-output") {
      mel_output_path = value;
    } else if (argument == "--mel-input") {
      mel_input_path = value;
    } else {
      std::cerr << argv[0] << ": unknown option " << argument << "\n";
      return 2;
    }
  }
  // Note (Jiaxin Deng): as the server, blank instructions or transcripts are
  // none.
  if (text.instructions.has_value() && text.instructions->empty()) {
    text.instructions.reset();
  } else {
  }
  if (text.reference_text.has_value() && text.reference_text->empty()) {
    text.reference_text.reset();
  } else {
  }
  const auto load_started = std::chrono::steady_clock::now();
  const cosyvoice3::CosyVoice3Model model(model_path);
  const double load_seconds = SecondsSince(load_started);

  std::optional<cosyvoice3::Reference> reference;
  nlohmann::json reference_line;
  double reference_seconds = 0.0;
  if (!reference_audio_path.empty()) {
    const auto reference_started = std::chrono::steady_clock::now();
    std::ifstream file(reference_audio_path, std::ios::binary);
    const std::string bytes((std::istreambuf_iterator<char>(file)),
                            std::istreambuf_iterator<char>());
    reference = model.EncodeReference(cosyvoice3::ReadReferenceAudio(bytes));
    reference_seconds = SecondsSince(reference_started);
    const mx::array &mel = reference->prompt_mel;
    const mx::array &embedding = reference->speaker_embedding;
    reference_line = {
        {"speech_tokens", reference->speech_tokens},
        {"flow_tokens", reference->flow_tokens},
        {"speaker_embedding",
         std::vector<float>(embedding.data<float>(),
                            embedding.data<float>() + embedding.size())},
        {"prompt_mel_frames", mel.shape(1)},
        {"prompt_mel", std::vector<float>(mel.data<float>(),
                                          mel.data<float>() + mel.size())}};
  } else if (!reference_input_path.empty()) {
    const nlohmann::json given =
        nlohmann::json::parse(std::ifstream(reference_input_path));
    const std::vector<float> mel =
        given["prompt_mel"].get<std::vector<float>>();
    const std::vector<float> embedding =
        given["speaker_embedding"].get<std::vector<float>>();
    reference = cosyvoice3::Reference{
        given["speech_tokens"].get<std::vector<int>>(),
        given["flow_tokens"].get<std::vector<int>>(),
        mx::array(mel.data(),
                  {1, static_cast<int>(mel.size()) / cosyvoice3::kMelBinCount,
                   cosyvoice3::kMelBinCount},
                  mx::float32),
        mx::array(embedding.data(), {1, cosyvoice3::kSpeakerEmbeddingSize},
                  mx::float32)};
  } else {
  }
  // Note (Jiaxin Deng): as on CUDA, the LLM's speech prompt is the reference's
  // tokens only when its transcript is given.
  const std::vector<int> prompt_speech_tokens =
      reference.has_value() && text.reference_text.has_value()
          ? reference->speech_tokens
          : std::vector<int>();
  const auto [prompt_ids, target_ids] = model.EncodeText(text);
  std::vector<int> prompt_text_ids = prompt_ids;
  prompt_text_ids.insert(prompt_text_ids.end(), target_ids.begin(),
                         target_ids.end());
  std::vector<float> first_logits;
  if (!tokens_output_path.empty()) {
    std::vector<qwen3_asr::KVCache> caches = model.NewCaches();
    const mx::array logits = mx::astype(
        model.ForwardEmbeddings(
            model.BuildPromptEmbeddings(prompt_text_ids, prompt_speech_tokens),
            caches),
        mx::float32);
    mx::eval(logits);
    first_logits.assign(logits.data<float>(),
                        logits.data<float>() + logits.size());
  } else {
  }

  const auto started = std::chrono::steady_clock::now();
  cosyvoice3::GeneratedSpeech generated;
  if (!tokens_input_path.empty()) {
    generated.speech_tokens =
        nlohmann::json::parse(std::ifstream(tokens_input_path))["speech_tokens"]
            .get<std::vector<int>>();
  } else {
    const std::atomic<bool> cancel(false);
    generated = model.Generate(prompt_ids, target_ids, prompt_speech_tokens,
                               options, cancel, [](int) {});
  }
  const double tokens_seconds = SecondsSince(started);
  const std::vector<int> kept =
      cosyvoice3::TrimSilentRuns(generated.speech_tokens);
  const mx::array mel = mel_input_path.empty()
                            ? model.Mel(kept, reference, [] {})
                            : mx::load(mel_input_path);
  if (!mel_output_path.empty()) {
    mx::save(mel_output_path, mel);
  } else {
  }
  const mx::array samples = model.Samples(mel);
  const double seconds = SecondsSince(started);
  const std::vector<float> audio(samples.data<float>(),
                                 samples.data<float>() + samples.size());
  const double audio_seconds =
      static_cast<double>(audio.size()) / cosyvoice3::kSampleRate;

  if (!output_path.empty()) {
    std::ofstream(output_path, std::ios::binary) << voxtral_tts::WavBytes(
        voxtral_tts::Pcm16Bytes(audio), cosyvoice3::kSampleRate);
  } else {
  }
  if (!tokens_output_path.empty()) {
    std::ofstream(tokens_output_path) << nlohmann::json{
        {"text_ids", prompt_text_ids},
        {"speech_tokens", generated.speech_tokens},
        {"vocoded_tokens", kept},
        {"first_logits", first_logits},
        {"reference", reference_line}}.dump();
  } else {
  }
  const nlohmann::json line = {
      {"text_token_count", prompt_text_ids.size()},
      {"speech_token_count", generated.speech_tokens.size()},
      {"finish_reason", qwen3_asr::FinishReasonName(generated.finish_reason)},
      {"load_seconds", load_seconds},
      {"reference_seconds", reference_seconds},
      {"token_seconds", tokens_seconds},
      {"seconds", seconds},
      {"audio_seconds", audio_seconds},
      {"real_time_factor", audio_seconds > 0 ? seconds / audio_seconds : 0.0},
      {"peak_memory_bytes", mx::get_peak_memory()},
  };
  std::cout << line.dump() << std::endl;
  return 0;
}
