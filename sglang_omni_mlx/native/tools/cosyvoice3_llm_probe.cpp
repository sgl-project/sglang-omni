// SPDX-License-Identifier: Apache-2.0
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "fun_cosyvoice3/cosyvoice3_model.h"
#include "nlohmann/json.hpp"

namespace mx = mlx::core;

namespace {

std::vector<int> ParseTokenIds(const std::string &value) {
  const auto tokens = nlohmann::json::parse(value);
  if (!tokens.is_array()) {
    throw std::invalid_argument("token ids must be a JSON array of integers");
  } else {
  }
  std::vector<int> token_ids;
  for (const auto &token : tokens) {
    if (!token.is_number_integer() || token < 0 || token > 151935) {
      throw std::invalid_argument("token ids must be integers in [0, 151935]");
    } else {
      token_ids.push_back(token.get<int>());
    }
  }
  return token_ids;
}

} // namespace

int main(int argc, char **argv) {
  try {
    std::string model_path;
    std::string logits_output;
    std::string embeddings_output;
    std::vector<int> text_token_ids;
    std::vector<int> prompt_speech_token_ids;
    std::vector<int> decode_token_ids;
    for (int i = 1; i < argc; ++i) {
      const std::string argument = argv[i];
      if (argument == "--help") {
        std::cout << "Usage: cosyvoice3_llm_probe --model-path DIR "
                     "--logits-output FILE.npy\n"
                  << "  [--text-ids '[1,2]'] [--prompt-speech-ids '[3,4]']\n"
                  << "  [--decode-ids '[5,6]'] [--embeddings-output FILE.npy]\n"
                  << "Loads converted CosyVoice3 safetensors, not raw llm.pt.\n"
                  << "Prompt: [SOS, text, TASK, prompt speech]. Text ids must "
                     "include\n"
                  << "any reference transcript and tokenizer control tokens.\n"
                  << "Logits are float32 [1 + decode token count, 6761]: "
                     "prefill, then\n"
                  << "one row after each teacher-forced decode id using the KV "
                     "cache.\n"
                  << "No tokenizer, sampling, vocoder, ONNX, or serving is "
                     "involved.\n";
        return 0;
      } else if (i + 1 >= argc) {
        throw std::invalid_argument("missing value for " + argument);
      } else {
      }
      const std::string value = argv[++i];
      if (argument == "--model-path") {
        model_path = value;
      } else if (argument == "--logits-output") {
        logits_output = value;
      } else if (argument == "--embeddings-output") {
        embeddings_output = value;
      } else if (argument == "--text-ids") {
        text_token_ids = ParseTokenIds(value);
      } else if (argument == "--prompt-speech-ids") {
        prompt_speech_token_ids = ParseTokenIds(value);
      } else if (argument == "--decode-ids") {
        decode_token_ids = ParseTokenIds(value);
      } else {
        throw std::invalid_argument("unknown option " + argument);
      }
    }
    if (model_path.empty() || logits_output.empty()) {
      throw std::invalid_argument(
          "--model-path and --logits-output are required; see --help");
    } else {
    }
    for (const int token_id : prompt_speech_token_ids) {
      if (token_id >= fun_cosyvoice3::kSpeechTokenSize) {
        throw std::invalid_argument("prompt speech ids must be in [0, 6560]");
      } else {
      }
    }
    for (const int token_id : decode_token_ids) {
      if (token_id >= fun_cosyvoice3::kTotalSpeechVocabularySize) {
        throw std::invalid_argument("decode ids must be in [0, 6760]");
      } else {
      }
    }

    const fun_cosyvoice3::CosyVoice3Model model(model_path);
    const auto embeddings =
        model.BuildPromptEmbeddings(text_token_ids, prompt_speech_token_ids);
    auto caches = model.NewCaches();
    std::vector<mx::array> logits;
    logits.push_back(model.ForwardEmbeddings(embeddings, caches));
    mx::eval(logits.back());
    for (const int token_id : decode_token_ids) {
      logits.push_back(
          model.ForwardEmbeddings(model.SpeechEmbeddings({token_id}), caches));
      mx::eval(logits.back());
    }
    const auto output =
        mx::astype(mx::reshape(mx::concatenate(logits, 1),
                               {static_cast<int>(logits.size()),
                                fun_cosyvoice3::kTotalSpeechVocabularySize}),
                   mx::float32);
    mx::save(logits_output, output);
    if (!embeddings_output.empty()) {
      mx::save(embeddings_output, mx::astype(embeddings, mx::float32));
    } else {
    }
    std::cout << nlohmann::json{{"prompt_token_count", embeddings.shape(1)},
                                {"decode_token_count", decode_token_ids.size()},
                                {"cache_token_count", caches.front().offset()},
                                {"logits_shape", output.shape()},
                                {"logits_output", logits_output}}
                     .dump()
              << '\n';
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "cosyvoice3_llm_probe: " << error.what() << '\n';
    return 1;
  }
}
