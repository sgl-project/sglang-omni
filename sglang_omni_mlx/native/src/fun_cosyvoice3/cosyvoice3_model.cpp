// SPDX-License-Identifier: Apache-2.0
#include "fun_cosyvoice3/cosyvoice3_model.h"

#include <stdexcept>
#include <string>
#include <utility>

namespace fun_cosyvoice3 {

namespace mx = mlx::core;

namespace {

constexpr int kHiddenSize = 896;
constexpr int kIntermediateSize = 4864;
constexpr int kLayerCount = 24;
constexpr int kAttentionHeadCount = 14;
constexpr int kKeyValueHeadCount = 2;
constexpr int kHeadDimension = 64;
constexpr int kTextVocabularySize = 151936;
constexpr int kMaxPositionEmbeddings = 32768;
constexpr float kRmsNormEpsilon = 1e-6f;
constexpr float kRopeTheta = 1000000.0f;

qwen3_asr::Checkpoint
LoadCheckpoint(const std::filesystem::path &model_directory) {
  auto config = qwen3_asr::ReadJson(model_directory / "config.json");
  const auto architecture = config.value("llm", nlohmann::json::object());
  const nlohmann::json expected = {
      {"hidden_size", kHiddenSize},
      {"intermediate_size", kIntermediateSize},
      {"num_hidden_layers", kLayerCount},
      {"num_attention_heads", kAttentionHeadCount},
      {"num_key_value_heads", kKeyValueHeadCount},
      {"head_dim", kHeadDimension},
      {"vocab_size", kTextVocabularySize},
      {"max_position_embeddings", kMaxPositionEmbeddings},
      {"rms_norm_eps", kRmsNormEpsilon},
      {"rope_theta", kRopeTheta}};
  if (!architecture.is_object()) {
    throw std::invalid_argument("CosyVoice3 config.llm must be an object");
  } else {
  }
  for (const auto &[name, value] : expected.items()) {
    if (architecture.contains(name) &&
        (!architecture.at(name).is_number() ||
         architecture.at(name).get<float>() != value.get<float>())) {
      throw std::invalid_argument("unsupported CosyVoice3 llm." + name);
    } else {
    }
  }
  if ((architecture.contains("model_type") &&
       architecture.at("model_type") != "qwen2") ||
      (architecture.contains("rope_scaling") &&
       !architecture.at("rope_scaling").is_null()) ||
      architecture.value("use_sliding_window", false)) {
    throw std::invalid_argument("unsupported CosyVoice3 Qwen2 architecture");
  } else {
  }

  int bits = 0;
  int group_size = 0;
  if (config.contains("quantization") && !config.at("quantization").is_null()) {
    auto &quantization = config.at("quantization");
    bits = quantization.value("bits", 4);
    group_size = quantization.value("group_size", 64);
    if ((bits != 2 && bits != 3 && bits != 4 && bits != 6 && bits != 8) ||
        (group_size != 32 && group_size != 64 && group_size != 128) ||
        quantization.value("mode", std::string("affine")) != "affine") {
      throw std::invalid_argument("unsupported CosyVoice3 affine quantization");
    } else {
    }
    quantization["bits"] = bits;
    quantization["group_size"] = group_size;
  } else {
    config.erase("quantization");
  }

  qwen3_asr::WeightMap weights;
  for (auto &[name, tensor] : qwen3_asr::LoadSafetensors(model_directory)) {
    std::string mapped_name;
    if (name.starts_with("qwen2.") && !name.ends_with("lm_head.weight")) {
      auto suffix = name.substr(6);
      if (suffix.starts_with("model.")) {
        suffix.erase(0, 6);
      } else {
      }
      mapped_name = "model." + suffix;
    } else if (name == "llm.speech_embedding.weight" ||
               name == "speech_embedding.weight") {
      mapped_name = "speech_embedding.weight";
    } else if (name == "llm.llm_decoder.weight" ||
               name == "llm_decoder.weight") {
      mapped_name = "llm_decoder.weight";
    } else {
      continue;
    }
    if (!weights.emplace(mapped_name, tensor).second) {
      throw std::invalid_argument("ambiguous CosyVoice3 tensor: " +
                                  mapped_name);
    } else {
    }
  }

  const auto require_shape = [&](const std::string &name,
                                 const mx::Shape &shape, bool packed = false) {
    const auto found = weights.find(name);
    if (found == weights.end()) {
      throw std::invalid_argument("CosyVoice3 checkpoint is missing " + name);
    } else if (found->second.shape() != shape) {
      throw std::invalid_argument("unexpected CosyVoice3 tensor shape: " +
                                  name);
    } else {
    }
    const auto dtype = found->second.dtype();
    if (packed ? dtype != mx::uint32
               : (dtype != mx::float16 && dtype != mx::bfloat16 &&
                  dtype != mx::float32)) {
      throw std::invalid_argument("unexpected CosyVoice3 tensor dtype: " +
                                  name);
    } else {
    }
  };
  require_shape("model.embed_tokens.weight",
                {kTextVocabularySize, kHiddenSize});
  require_shape("model.norm.weight", {kHiddenSize});
  require_shape("speech_embedding.weight",
                {kTotalSpeechVocabularySize, kHiddenSize});
  require_shape("llm_decoder.weight",
                {kTotalSpeechVocabularySize, kHiddenSize});
  for (int i = 0; i < kLayerCount; ++i) {
    const auto prefix = "model.layers." + std::to_string(i);
    require_shape(prefix + ".input_layernorm.weight", {kHiddenSize});
    require_shape(prefix + ".post_attention_layernorm.weight", {kHiddenSize});
    const std::vector<std::pair<std::string, mx::Shape>> projections = {
        {"self_attn.q_proj", {kHiddenSize, kHiddenSize}},
        {"self_attn.k_proj",
         {kKeyValueHeadCount * kHeadDimension, kHiddenSize}},
        {"self_attn.v_proj",
         {kKeyValueHeadCount * kHeadDimension, kHiddenSize}},
        {"self_attn.o_proj", {kHiddenSize, kHiddenSize}},
        {"mlp.gate_proj", {kIntermediateSize, kHiddenSize}},
        {"mlp.up_proj", {kIntermediateSize, kHiddenSize}},
        {"mlp.down_proj", {kHiddenSize, kIntermediateSize}}};
    for (const auto &[suffix, shape] : projections) {
      const auto projection = prefix + "." + suffix;
      if (weights.contains(projection + ".scales")) {
        if (bits == 0) {
          throw std::invalid_argument(
              "quantized CosyVoice3 weights require quantization config");
        } else {
        }
        require_shape(projection + ".weight", {shape[0], shape[1] * bits / 32},
                      true);
        require_shape(projection + ".scales",
                      {shape[0], shape[1] / group_size});
        require_shape(projection + ".biases",
                      {shape[0], shape[1] / group_size});
      } else {
        require_shape(projection + ".weight", shape);
      }
      if (suffix == "self_attn.q_proj" || suffix == "self_attn.k_proj" ||
          suffix == "self_attn.v_proj") {
        require_shape(projection + ".bias", {shape[0]});
      } else if (weights.contains(projection + ".bias")) {
        throw std::invalid_argument("unexpected CosyVoice3 projection bias: " +
                                    projection);
      } else {
      }
    }
  }
  // Note (lgyStoic): Match the legacy loader's FP16 speech tables without
  // changing the backbone precision.
  for (const std::string name :
       {"speech_embedding.weight", "llm_decoder.weight"}) {
    weights.insert_or_assign(name, mx::astype(weights.at(name), mx::float16));
  }
  return qwen3_asr::Checkpoint(std::move(weights), config);
}

mx::array TokenIds(const std::vector<int> &token_ids, int vocabulary_size) {
  for (const int token_id : token_ids) {
    if (token_id < 0 || token_id >= vocabulary_size) {
      throw std::invalid_argument("CosyVoice3 token id out of range: " +
                                  std::to_string(token_id));
    } else {
    }
  }
  return mx::array(token_ids.data(), {1, static_cast<int>(token_ids.size())},
                   mx::int32);
}

} // namespace

CosyVoice3Model::CosyVoice3Model(const std::filesystem::path &model_directory)
    : checkpoint_(LoadCheckpoint(model_directory)),
      decoder_(checkpoint_, "model",
               {kLayerCount, kAttentionHeadCount, kKeyValueHeadCount,
                kHeadDimension, kRmsNormEpsilon, kRopeTheta, false, false}) {}

mx::array CosyVoice3Model::BuildPromptEmbeddings(
    const std::vector<int> &text_token_ids,
    const std::vector<int> &prompt_speech_token_ids) const {
  std::vector<mx::array> pieces = {SpeechEmbeddings({kSosTokenId})};
  if (!text_token_ids.empty()) {
    pieces.push_back(
        decoder_.EmbedTokens(TokenIds(text_token_ids, kTextVocabularySize)));
  } else {
  }
  pieces.push_back(SpeechEmbeddings({kTaskTokenId}));
  if (!prompt_speech_token_ids.empty()) {
    pieces.push_back(
        checkpoint_.Embed(TokenIds(prompt_speech_token_ids, kSpeechTokenSize),
                          "speech_embedding"));
  } else {
  }
  return mx::concatenate(pieces, 1);
}

mx::array CosyVoice3Model::SpeechEmbeddings(
    const std::vector<int> &speech_token_ids) const {
  if (speech_token_ids.empty()) {
    throw std::invalid_argument(
        "CosyVoice3 speech token input must not be empty");
  } else {
    return checkpoint_.Embed(
        TokenIds(speech_token_ids, kTotalSpeechVocabularySize),
        "speech_embedding");
  }
}

mx::array CosyVoice3Model::ForwardEmbeddings(
    const mx::array &embeddings,
    std::vector<qwen3_asr::KVCache> &caches) const {
  if (embeddings.ndim() != 3 || embeddings.shape(0) != 1 ||
      embeddings.shape(1) == 0 || embeddings.shape(2) != kHiddenSize ||
      caches.size() != kLayerCount) {
    throw std::invalid_argument(
        "CosyVoice3 requires [1, tokens, 896] embeddings and 24 caches");
  } else {
  }
  const int offset = caches.front().offset();
  for (const auto &cache : caches) {
    if (cache.offset() != offset) {
      throw std::invalid_argument("CosyVoice3 cache offsets must match");
    } else {
    }
  }
  if (embeddings.shape(1) > kMaxPositionEmbeddings - offset) {
    throw std::invalid_argument(
        "CosyVoice3 input exceeds the 32768-token context");
  } else {
  }
  const auto hidden_states = decoder_.Forward(embeddings, caches);
  const int length = hidden_states.shape(1);
  const auto last = checkpoint_.RmsNorm(
      mx::slice(hidden_states, {0, length - 1, 0}, {1, length, kHiddenSize}),
      "model.norm", kRmsNormEpsilon);
  return checkpoint_.Linear(last, "llm_decoder");
}

std::vector<qwen3_asr::KVCache> CosyVoice3Model::NewCaches() const {
  return decoder_.NewCaches();
}

} // namespace fun_cosyvoice3
