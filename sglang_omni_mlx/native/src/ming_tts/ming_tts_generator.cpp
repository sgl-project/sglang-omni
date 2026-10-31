// SPDX-License-Identifier: Apache-2.0
#include "ming_tts/ming_tts_generator.h"

#include <CoreFoundation/CoreFoundation.h>

#include <algorithm>
#include <array>
#include <cassert>
#include <stdexcept>

namespace ming_tts {
namespace mx = mlx::core;

namespace {

constexpr float kStopProbabilityThreshold = 0.5f;
constexpr int kMinimumGeneratedPatches = 5;

bool HasText(const std::string &text) {
  CFStringRef unicode = CFStringCreateWithBytes(
      kCFAllocatorDefault, reinterpret_cast<const UInt8 *>(text.data()),
      static_cast<CFIndex>(text.size()), kCFStringEncodingUTF8, false);
  if (unicode == nullptr) {
    throw std::invalid_argument("text must be valid UTF-8");
  } else {
  }
  const auto whitespace =
      CFCharacterSetGetPredefined(kCFCharacterSetWhitespaceAndNewline);
  bool found = false;
  for (CFIndex i = 0; i < CFStringGetLength(unicode); ++i) {
    const UniChar character = CFStringGetCharacterAtIndex(unicode, i);
    // Note (altale): Python also strips the U+001C through U+001F separators.
    if (!CFCharacterSetIsCharacterMember(whitespace, character) &&
        !(character >= 0x1c && character <= 0x1f)) {
      found = true;
      break;
    } else {
    }
  }
  CFRelease(unicode);
  return found;
}

} // namespace

MingFrontend::MingFrontend(const std::filesystem::path &directory)
    : tokenizer_(omni_common::Tokenizer::FromJson(
          omni_common::ReadJson(directory / "tokenizer.json"))) {
  for (const auto *token : {"<role>", "</role>", "<audio>", "<|endoftext|>",
                            "<spk>", "</spk>", "<audioPatch>"}) {
    const int id = tokenizer_.AddedTokenId(token);
    if (id < 0) {
      throw std::invalid_argument(std::string("missing Ming special token: ") +
                                  token);
    } else {
    }
  }
}

Prompt MingFrontend::BuildPrompt(const std::string &text,
                                 const std::string &description,
                                 const std::string &instructions,
                                 const std::string &reference_text,
                                 int reference_tokens) const {
  if (!HasText(text)) {
    throw std::invalid_argument("input must be nonempty");
  } else {
  }
  if (reference_tokens > 0 && !HasText(reference_text)) {
    throw std::invalid_argument("ref_text must be nonempty");
  } else {
  }
  Prompt prompt;
  auto &input_ids = prompt.input_ids;
  const auto append = [&](const std::string &part) {
    const auto tokens = tokenizer_.Encode(part);
    input_ids.insert(input_ids.end(), tokens.begin(), tokens.end());
  };
  append("<role>HUMAN</role>");
  append(description.empty()
             ? "Please generate speech based on the following description.\n"
             : description);
  if (reference_tokens > 0) {
    append("  speaker_1:<spk>");
    prompt.speaker_position = static_cast<int>(input_ids.size());
    input_ids.push_back(tokenizer_.AddedTokenId("<audioPatch>"));
    append("</spk>\n");
  } else {
  }
  constexpr std::array markers{
      "Genre: ", "Mood: ", "Instrument: ", "Theme: ", "Duration: "};
  if (!std::all_of(markers.begin(), markers.end(), [&](const auto *marker) {
        return text.find(marker) != std::string::npos;
      })) {
    append(" Text input:\n");
  } else {
  }
  append(reference_text);
  append(text);
  append("<role>ASSISTANT</role>");
  if (!instructions.empty()) {
    append(instructions);
    input_ids.push_back(tokenizer_.AddedTokenId("<|endoftext|>"));
  } else {
  }
  input_ids.push_back(tokenizer_.AddedTokenId("<audio>"));
  if (reference_tokens > 0) {
    prompt.reference_start = static_cast<int>(input_ids.size());
    input_ids.insert(input_ids.end(), reference_tokens,
                     tokenizer_.AddedTokenId("<audioPatch>"));
  } else {
  }
  return prompt;
}

MingGenerator::MingGenerator(const std::filesystem::path &directory)
    : backbone_(directory), acoustic_(directory) {}

GenerationState MingGenerator::Start(
    const Prompt &prompt, int max_steps,
    const std::optional<ReferenceConditioning> &reference) const {
  const auto &input_ids = prompt.input_ids;
  assert(!input_ids.empty() && max_steps > 0);
  const auto &config = backbone_.config();
  if (input_ids.size() + max_steps - 1 >
          static_cast<size_t>(config.max_position_embeddings) ||
      std::any_of(input_ids.begin(), input_ids.end(), [&](int token) {
        return token < 0 || token >= config.vocab_size;
      })) {
    throw std::invalid_argument("invalid token ids or generation length");
  } else {
  }
  const int token_count = static_cast<int>(input_ids.size());
  mx::array embeddings =
      backbone_.EmbedTokens(mx::array(input_ids.data(), {1, token_count}));
  mx::array history = mx::zeros(
      {1, acoustic_.history_frames(), acoustic_.latent_dim()}, mx::float32);
  if (reference) {
    const auto &latents = reference->latents;
    const int frames = latents.shape(1);
    assert(
        latents.shape(0) == 1 && latents.shape(2) == acoustic_.latent_dim() &&
        frames % acoustic_.patch_size() == 0 && prompt.speaker_position >= 0);
    const int patches = frames / acoustic_.patch_size();
    assert(prompt.reference_start + patches == token_count);
    embeddings = mx::slice_update(
        embeddings,
        mx::astype(
            mx::expand_dims(acoustic_.ProjectSpeaker(reference->speaker), 1),
            embeddings.dtype()),
        {0, prompt.speaker_position, 0},
        {1, prompt.speaker_position + 1, config.hidden_size});
    const auto projected = acoustic_.Aggregate(mx::reshape(
        latents, {patches, acoustic_.patch_size(), acoustic_.latent_dim()}));
    embeddings = mx::slice_update(
        embeddings,
        mx::astype(mx::reshape(projected, {1, patches, config.hidden_size}),
                   embeddings.dtype()),
        {0, prompt.reference_start, 0}, {1, token_count, config.hidden_size});
    const int kept = std::min(frames, acoustic_.history_frames());
    history = mx::slice_update(
        history, mx::slice(latents, {0, frames - kept, 0}, latents.shape()),
        {0, acoustic_.history_frames() - kept, 0}, history.shape());
  } else {
  }
  auto caches = backbone_.NewCaches();
  const mx::array positions =
      mx::broadcast_to(mx::arange(token_count), {3, 1, token_count});
  const mx::array prefill = backbone_.Forward(embeddings, positions, caches);
  const mx::array hidden =
      mx::slice(prefill, {0, prefill.shape(1) - 1, 0}, prefill.shape());
  mx::eval(hidden, history);
  return {std::move(caches), hidden, history, std::nullopt, max_steps};
}

mx::array MingGenerator::Step(GenerationState &state, const mx::array &noise,
                              const mx::array &timesteps,
                              const mx::array &sde_random, float cfg,
                              float sigma, float temperature) const {
  assert(!state.finished);
  if (state.feedback) {
    const mx::array positions =
        mx::full({3, 1, 1}, mx::array(state.caches.front().offset()));
    state.hidden = backbone_.Forward(mx::expand_dims(*state.feedback, 1),
                                     positions, state.caches);
  } else {
  }
  const auto tail =
      acoustic_.Step(state.hidden, state.history, noise, timesteps, sde_random,
                     cfg, sigma, temperature);
  mx::eval(tail.latents, tail.feedback, tail.stop_probability);
  const float probability =
      mx::astype(tail.stop_probability, mx::float32).item<float>();
  const bool stop = probability > kStopProbabilityThreshold &&
                    state.steps + 1 >= kMinimumGeneratedPatches;
  const bool length = state.steps + 1 >= state.max_steps;
  ++state.steps;
  state.finished = stop || length;
  if (state.finished) {
    state.caches.clear();
    state.feedback.reset();
  } else {
    const mx::array appended =
        mx::concatenate({state.history, tail.latents}, 1);
    state.history = mx::slice(
        appended, {0, appended.shape(1) - acoustic_.history_frames(), 0},
        appended.shape());
    state.feedback = mx::astype(tail.feedback, state.hidden.dtype());
    mx::eval(state.history, *state.feedback);
  }
  return mx::squeeze(tail.latents, 0);
}
} // namespace ming_tts
