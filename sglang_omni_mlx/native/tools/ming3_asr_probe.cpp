#include <algorithm>
#include <atomic>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "common/audio.h"
#include "common/tokenizer.h"
#include "ming3_asr/ming3_asr_model.h"

namespace mx = mlx::core;

int main(int argc, char **argv) {
  try {
    if (argc < 4) {
      throw std::invalid_argument("usage: ming3_asr_probe OPERATION INPUT "
                                  "OUTPUT [MODEL_DIRECTORY] [CHUNK_SIZE]");
    } else {
    }
    const std::string operation = argv[1];
    const std::string input = argv[2];
    const std::string output = argv[3];
    if (operation == "delta") {
      const auto tensors = mx::load_safetensors(input).first;
      const auto result = ming3_asr::GatedDelta(
          tensors.at("queries"), tensors.at("keys"), tensors.at("values"),
          tensors.at("decay"), tensors.at("beta"), tensors.at("state"));
      mx::save_safetensors(
          output, {{"output", result.output}, {"state", result.next_state}});
    } else if (argc < 5) {
      throw std::invalid_argument("model directory is required");
    } else if (operation == "tokenizer") {
      const omni_common::Tokenizer tokenizer(argv[4]);
      const auto texts =
          omni_common::ReadJson(input).get<std::vector<std::string>>();
      auto results = nlohmann::json::array();
      for (const auto &text : texts) {
        const auto ids = tokenizer.Encode(text);
        results.push_back(
            {{"ids", ids}, {"decoded", tokenizer.Decode(ids, false)}});
      }
      std::ofstream destination(output);
      destination.exceptions(std::ios::badbit | std::ios::failbit);
      destination << results.dump();
    } else if (operation == "encode") {
      const ming3_asr::MingModel model(argv[4]);
      std::ifstream stream(input, std::ios::binary);
      if (!stream) {
        throw std::runtime_error("cannot read " + input);
      } else {
      }
      std::ostringstream bytes;
      bytes << stream.rdbuf();
      const auto samples = omni_common::DecodeWav(bytes.str());
      const std::atomic<bool> cancel(false);
      mx::save_safetensors(
          output, {{"features", ming3_asr::ComputeAudioFeatures(samples)},
                   {"audio", model.EncodeAudio(samples, cancel)}});
    } else if (operation == "forward") {
      const ming3_asr::MingModel model(argv[4]);
      const auto embeddings =
          mx::load_safetensors(input).first.at("embeddings");
      const int chunk_size =
          argc > 5 ? std::stoi(argv[5]) : embeddings.shape(1);
      if (chunk_size < 1) {
        throw std::invalid_argument("chunk size must be positive");
      } else {
      }
      const std::atomic<bool> cancel(false);
      auto cache = model.NewCache();
      std::vector<mx::array> hidden;
      for (int start = 0; start < embeddings.shape(1); start += chunk_size) {
        hidden.push_back(model.Forward(
            mx::slice(embeddings, {0, start, 0},
                      {1, std::min(start + chunk_size, embeddings.shape(1)),
                       embeddings.shape(2)}),
            cache, cancel));
      }
      std::unordered_map<std::string, mx::array> tensors{
          {"hidden", mx::concatenate(hidden, 1)}};
      tensors.emplace("logits", model.LastTokenLogits(hidden.back()));
      for (size_t layer = 0; layer < cache.layers().size(); ++layer) {
        const auto &layer_cache = cache.layers()[layer];
        const std::string prefix = std::to_string(layer) + ".";
        if (const auto *delta =
                std::get_if<ming3_asr::DeltaCache>(&layer_cache)) {
          const auto &state = delta->state.value();
          tensors.emplace(prefix + "recurrent", state.recurrent);
          for (size_t index = 0; index < state.convolution.size(); ++index) {
            tensors.emplace(prefix + "conv." + std::to_string(index),
                            state.convolution[index]);
          }
        } else {
          const auto &state =
              std::get<ming3_asr::LatentAttentionCache>(layer_cache)
                  .state.value();
          tensors.emplace(prefix + "keys", state.keys);
          tensors.emplace(prefix + "values", state.values);
        }
      }
      mx::save_safetensors(output, tensors);
    } else {
      throw std::invalid_argument("unknown probe operation " + operation);
    }
    return 0;
  } catch (const std::exception &error) {
    std::cerr << argv[0] << ": " << error.what() << '\n';
    return 1;
  }
}
