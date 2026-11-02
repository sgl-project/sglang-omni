// SPDX-License-Identifier: Apache-2.0
#include "cosyvoice3/cosyvoice3_model.h"

#include <algorithm>
#include <array>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>

namespace cosyvoice3 {

namespace mx = mlx::core;

namespace {

constexpr int kHiddenSize = 896;
constexpr int kIntermediateSize = 4864;
constexpr int kLayerCount = 24;
constexpr int kAttentionHeadCount = 14;
constexpr int kKeyValueHeadCount = 2;
constexpr int kHeadDimension = 64;
constexpr int kTextVocabularySize = 151936;
constexpr int kMaxPositionEmbeddings = kMaxContextTokens;
constexpr float kRmsNormEpsilon = 1e-6f;
constexpr float kRopeTheta = 1000000.0f;
// The recent speech tokens a sampled token must not repeat, else it is drawn
// again from the whole distribution without it.
constexpr size_t kRepetitionWindow = 10;
constexpr int kMaxSilentRun = 5;
constexpr float kGreedyTemperature = 1e-6f;
// Speech tokens of silence and breath.
const std::unordered_set<int> kSilentTokens = {1,   2,    28,   29,   55,  248,
                                               494, 2241, 2242, 2322, 2323};
const std::string kPromptPrefix = "You are a helpful assistant.<|endofprompt|>";
const std::string kInstructPrefix = "You are a helpful assistant. ";
const std::string kEndOfPrompt = "<|endofprompt|>";
constexpr int kFirstSpecialTokenId = 151646;
// The special tokens the CUDA server adds to the tokenizer, in id order from
// kFirstSpecialTokenId.
constexpr const char *kSpecialTokens[] = {
    "<|endofprompt|>",
    "[breath]",
    "<strong>",
    "</strong>",
    "[noise]",
    "[laughter]",
    "[cough]",
    "[clucking]",
    "[accent]",
    "[quick_breath]",
    "<laughter>",
    "</laughter>",
    "[hissing]",
    "[sigh]",
    "[vocalized-noise]",
    "[lipsmack]",
    "[mn]",
    "<|endofsystem|>",
    "[AA]",
    "[AA0]",
    "[AA1]",
    "[AA2]",
    "[AE]",
    "[AE0]",
    "[AE1]",
    "[AE2]",
    "[AH]",
    "[AH0]",
    "[AH1]",
    "[AH2]",
    "[AO]",
    "[AO0]",
    "[AO1]",
    "[AO2]",
    "[AW]",
    "[AW0]",
    "[AW1]",
    "[AW2]",
    "[AY]",
    "[AY0]",
    "[AY1]",
    "[AY2]",
    "[B]",
    "[CH]",
    "[D]",
    "[DH]",
    "[EH]",
    "[EH0]",
    "[EH1]",
    "[EH2]",
    "[ER]",
    "[ER0]",
    "[ER1]",
    "[ER2]",
    "[EY]",
    "[EY0]",
    "[EY1]",
    "[EY2]",
    "[F]",
    "[G]",
    "[HH]",
    "[IH]",
    "[IH0]",
    "[IH1]",
    "[IH2]",
    "[IY]",
    "[IY0]",
    "[IY1]",
    "[IY2]",
    "[JH]",
    "[K]",
    "[L]",
    "[M]",
    "[N]",
    "[NG]",
    "[OW]",
    "[OW0]",
    "[OW1]",
    "[OW2]",
    "[OY]",
    "[OY0]",
    "[OY1]",
    "[OY2]",
    "[P]",
    "[R]",
    "[S]",
    "[SH]",
    "[T]",
    "[TH]",
    "[UH]",
    "[UH0]",
    "[UH1]",
    "[UH2]",
    "[UW]",
    "[UW0]",
    "[UW1]",
    "[UW2]",
    "[V]",
    "[W]",
    "[Y]",
    "[Z]",
    "[ZH]",
    "[a]",
    "[ai]",
    "[an]",
    "[ang]",
    "[ao]",
    "[b]",
    "[c]",
    "[ch]",
    "[d]",
    "[e]",
    "[ei]",
    "[en]",
    "[eng]",
    "[f]",
    "[g]",
    "[h]",
    "[i]",
    "[ian]",
    "[in]",
    "[ing]",
    "[iu]",
    "[ià]",
    "[iàn]",
    "[iàng]",
    "[iào]",
    "[iá]",
    "[ián]",
    "[iáng]",
    "[iáo]",
    "[iè]",
    "[ié]",
    "[iòng]",
    "[ióng]",
    "[iù]",
    "[iú]",
    "[iā]",
    "[iān]",
    "[iāng]",
    "[iāo]",
    "[iē]",
    "[iě]",
    "[iōng]",
    "[iū]",
    "[iǎ]",
    "[iǎn]",
    "[iǎng]",
    "[iǎo]",
    "[iǒng]",
    "[iǔ]",
    "[j]",
    "[k]",
    "[l]",
    "[m]",
    "[n]",
    "[o]",
    "[ong]",
    "[ou]",
    "[p]",
    "[q]",
    "[r]",
    "[s]",
    "[sh]",
    "[t]",
    "[u]",
    "[uang]",
    "[ue]",
    "[un]",
    "[uo]",
    "[uà]",
    "[uài]",
    "[uàn]",
    "[uàng]",
    "[uá]",
    "[uái]",
    "[uán]",
    "[uáng]",
    "[uè]",
    "[ué]",
    "[uì]",
    "[uí]",
    "[uò]",
    "[uó]",
    "[uā]",
    "[uāi]",
    "[uān]",
    "[uāng]",
    "[uē]",
    "[uě]",
    "[uī]",
    "[uō]",
    "[uǎ]",
    "[uǎi]",
    "[uǎn]",
    "[uǎng]",
    "[uǐ]",
    "[uǒ]",
    "[vè]",
    "[w]",
    "[x]",
    "[y]",
    "[z]",
    "[zh]",
    "[à]",
    "[ài]",
    "[àn]",
    "[àng]",
    "[ào]",
    "[á]",
    "[ái]",
    "[án]",
    "[áng]",
    "[áo]",
    "[è]",
    "[èi]",
    "[èn]",
    "[èng]",
    "[èr]",
    "[é]",
    "[éi]",
    "[én]",
    "[éng]",
    "[ér]",
    "[ì]",
    "[ìn]",
    "[ìng]",
    "[í]",
    "[ín]",
    "[íng]",
    "[ò]",
    "[òng]",
    "[òu]",
    "[ó]",
    "[óng]",
    "[óu]",
    "[ù]",
    "[ùn]",
    "[ú]",
    "[ún]",
    "[ā]",
    "[āi]",
    "[ān]",
    "[āng]",
    "[āo]",
    "[ē]",
    "[ēi]",
    "[ēn]",
    "[ēng]",
    "[ě]",
    "[ěi]",
    "[ěn]",
    "[ěng]",
    "[ěr]",
    "[ī]",
    "[īn]",
    "[īng]",
    "[ō]",
    "[ōng]",
    "[ōu]",
    "[ū]",
    "[ūn]",
    "[ǎ]",
    "[ǎi]",
    "[ǎn]",
    "[ǎng]",
    "[ǎo]",
    "[ǐ]",
    "[ǐn]",
    "[ǐng]",
    "[ǒ]",
    "[ǒng]",
    "[ǒu]",
    "[ǔ]",
    "[ǔn]",
    "[ǘ]",
    "[ǚ]",
    "[ǜ]",
};

// Note (Jiaxin Deng): lgyStoic's Qwen2 checkpoint mapping and checks; the
// flow, HiFT and speaker encoder weights keep their stored names.
omni_common::Checkpoint
LoadCheckpoint(const std::filesystem::path &model_directory) {
  auto config = omni_common::ReadJson(model_directory / "config.json");
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
  // Note (Jiaxin Deng): the MLX checkpoint's config.llm names only these
  // sizes, so they are required; the decoder's other settings are the 0.5B
  // model's constants, checked when given, and its tensor shapes below.
  const nlohmann::json required = {{"llm_input_size", kHiddenSize},
                                   {"llm_output_size", kHiddenSize},
                                   {"speech_token_size", kSpeechTokenSize},
                                   {"extended_vocab_size", 200}};
  for (const auto &[name, value] : required.items()) {
    if (!architecture.contains(name) || architecture.at(name) != value) {
      throw std::invalid_argument("CosyVoice3 config.llm." + name +
                                  " must be " + value.dump());
    } else {
    }
  }
  for (const auto &[name, value] : expected.items()) {
    if (architecture.contains(name) &&
        (!architecture.at(name).is_number() ||
         architecture.at(name).get<float>() != value.get<float>())) {
      throw std::invalid_argument("unsupported CosyVoice3 llm." + name);
    } else {
    }
  }
  if (config.contains("quantization") && !config.at("quantization").is_null()) {
    throw std::invalid_argument(
        "CosyVoice3 serves the float16 checkpoint, not a quantized one");
  } else {
    config.erase("quantization");
  }

  omni_common::WeightMap weights;
  for (auto &[name, tensor] : omni_common::LoadSafetensors(model_directory)) {
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
    } else if ((name.starts_with("flow.") &&
                !name.ends_with("rotary_embed.inv_freq")) ||
               name.starts_with("hifigan.") || name.starts_with("campplus.")) {
      mapped_name = name;
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
                                 const mx::Shape &shape) {
    const auto found = weights.find(name);
    if (found == weights.end()) {
      throw std::invalid_argument("CosyVoice3 checkpoint is missing " + name);
    } else if (found->second.shape() != shape) {
      throw std::invalid_argument("unexpected CosyVoice3 tensor shape: " +
                                  name);
    } else if (found->second.dtype() != mx::float16) {
      throw std::invalid_argument("CosyVoice3 tensor is not float16: " + name);
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
      require_shape(projection + ".weight", shape);
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
  require_shape("flow.input_embedding.weight",
                {kSpeechTokenSize, kMelBinCount});
  require_shape("flow.spk_embed_affine_layer.weight",
                {kMelBinCount, kSpeakerEmbeddingSize});
  require_shape("hifigan.conv_pre.conv.weight", {512, 5, kMelBinCount});
  return omni_common::Checkpoint(std::move(weights), config);
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

// The S3 speech tokenizer v3 checkpoint the model directory holds in
// speech_tokenizer/.
omni_common::Checkpoint
LoadSpeechTokenizer(const std::filesystem::path &directory) {
  const nlohmann::json config =
      omni_common::ReadJson(directory / "config.json");
  if (config.value("model_type", std::string()) != "s3_tokenizer_v3" ||
      config.value("n_audio_layer", 0) != 12 ||
      config.value("n_audio_state", 0) != 1280 ||
      config.value("n_mels", 0) != 128 ||
      config.value("n_codebook_size", 0) != kSpeechTokenSize) {
    throw std::invalid_argument(
        "speech_tokenizer/ must hold the S3 speech tokenizer v3");
  } else {
  }
  return omni_common::Checkpoint(omni_common::LoadSafetensors(directory),
                                 nlohmann::json::object());
}

} // namespace

CosyVoice3Model::CosyVoice3Model(const std::filesystem::path &model_directory)
    : checkpoint_(LoadCheckpoint(model_directory)),
      decoder_(checkpoint_, "model",
               {kLayerCount, kAttentionHeadCount, kKeyValueHeadCount,
                kHeadDimension, kRmsNormEpsilon, kRopeTheta, false, false}),
      tokenizer_(std::make_unique<omni_common::Tokenizer>(model_directory)),
      flow_(checkpoint_), hift_(checkpoint_), speaker_encoder_(checkpoint_),
      speech_tokenizer_checkpoint_(
          LoadSpeechTokenizer(model_directory / "speech_tokenizer")),
      speech_tokenizer_(speech_tokenizer_checkpoint_) {}

std::vector<int> CosyVoice3Model::Encode(const std::string &text) const {
  // Note (Jiaxin Deng): as Hugging Face splits added tokens out first,
  // leftmost and then longest, and encodes the text between them.
  std::vector<int> ids;
  size_t ordinary_start = 0;
  size_t position = 0;
  while (position < text.size()) {
    int match = -1;
    size_t match_size = 0;
    for (size_t i = 0; i < std::size(kSpecialTokens); ++i) {
      const std::string_view token = kSpecialTokens[i];
      if (token.size() > match_size &&
          text.compare(position, token.size(), token) == 0) {
        match = static_cast<int>(i);
        match_size = token.size();
      } else {
      }
    }
    if (match >= 0) {
      const std::vector<int> ordinary = tokenizer_->Encode(
          text.substr(ordinary_start, position - ordinary_start));
      ids.insert(ids.end(), ordinary.begin(), ordinary.end());
      ids.push_back(kFirstSpecialTokenId + match);
      position += match_size;
      ordinary_start = position;
    } else {
      ++position;
    }
  }
  const std::vector<int> ordinary =
      tokenizer_->Encode(text.substr(ordinary_start));
  ids.insert(ids.end(), ordinary.begin(), ordinary.end());
  return ids;
}

std::pair<std::vector<int>, std::vector<int>>
CosyVoice3Model::EncodeText(const SpeechText &text) const {
  // Note (Jiaxin Deng): the CUDA server's prompt text: instructions, else
  // the reference transcript, each kept as given when it already ends its
  // prompt.
  std::string prompt = kPromptPrefix;
  if (text.instructions.has_value()) {
    prompt = text.instructions->find(kEndOfPrompt) != std::string::npos
                 ? *text.instructions
                 : kInstructPrefix + *text.instructions + kEndOfPrompt;
  } else if (text.reference_text.has_value()) {
    prompt = text.reference_text->find(kEndOfPrompt) != std::string::npos
                 ? *text.reference_text
                 : kPromptPrefix + *text.reference_text;
  } else {
  }
  return {Encode(prompt), Encode(text.text)};
}

Reference CosyVoice3Model::EncodeReference(const ReferenceAudio &audio) const {
  const mx::array tokens = speech_tokenizer_.Encode(audio.samples_16k);
  mx::eval(tokens);
  const int32_t *token_values = tokens.data<int32_t>();
  std::vector<int> speech_tokens(token_values, token_values + tokens.size());
  const mx::array mel = PromptMel(audio.samples_24k);
  // Note (Jiaxin Deng): the flow takes two mel frames per prompt token, so
  // the longer of the two is trimmed, as on CUDA.
  const int flow_token_count = std::min(static_cast<int>(speech_tokens.size()),
                                        mel.shape(1) / kMelFramesPerToken);
  if (flow_token_count == 0) {
    throw std::invalid_argument("reference audio gives no speech tokens");
  } else {
  }
  Reference reference{
      speech_tokens,
      std::vector<int>(speech_tokens.begin(),
                       speech_tokens.begin() + flow_token_count),
      mx::slice(mel, {0, 0, 0},
                {1, kMelFramesPerToken * flow_token_count, kMelBinCount}),
      speaker_encoder_.Embed(audio.samples_16k)};
  mx::eval({reference.prompt_mel, reference.speaker_embedding});
  return reference;
}

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

GeneratedSpeech
CosyVoice3Model::Generate(const std::vector<int> &prompt_text_ids,
                          const std::vector<int> &target_text_ids,
                          const std::vector<int> &prompt_speech_tokens,
                          const SamplingOptions &options,
                          const std::atomic<bool> &cancel,
                          const std::function<void(int)> &on_token) const {
  // Note (Jiaxin Deng): the CUDA server's length contract, counted on the
  // target text alone: no stop before 2 tokens per text token, and at most
  // 20 per text token.
  const int target_token_count =
      std::max(static_cast<int>(target_text_ids.size()), 1);
  const int max_tokens = options.max_new_tokens.value_or(
      std::min(kMaxSpeechTokens, 20 * target_token_count));
  const int min_tokens = std::min(2 * target_token_count, max_tokens);
  // Note (Jiaxin Deng): as SGLang, a temperature this low samples greedily
  // rather than overflow the scaled logits.
  const bool greedy =
      options.temperature < kGreedyTemperature || options.top_k == 1;

  std::vector<int> text_ids = prompt_text_ids;
  text_ids.insert(text_ids.end(), target_text_ids.begin(),
                  target_text_ids.end());
  std::vector<qwen3_asr::KVCache> caches = NewCaches();
  mx::array logits = ForwardEmbeddings(
      BuildPromptEmbeddings(text_ids, prompt_speech_tokens), caches);
  const mx::array vocabulary =
      mx::arange(kTotalSpeechVocabularySize, mx::int32);
  const mx::array is_control =
      mx::greater_equal(vocabulary, mx::array(kSpeechTokenSize));
  const mx::array negative_infinity =
      mx::array(-std::numeric_limits<float>::infinity());
  mx::array seen = mx::zeros({kTotalSpeechVocabularySize}, mx::bool_);
  mx::array random_key = mx::random::key(options.seed);
  const auto next_key = [&]() {
    const auto [carried, drawn] = mx::random::split(random_key);
    random_key = carried;
    return drawn;
  };

  GeneratedSpeech generated;
  generated.finish_reason = qwen3_asr::FinishReason::kLength;
  while (static_cast<int>(generated.speech_tokens.size()) < max_tokens) {
    if (cancel.load()) {
      throw qwen3_asr::TranscriptionCancelled();
    } else {
    }
    mx::array row = mx::astype(mx::reshape(logits, {-1}), mx::float32);
    if (static_cast<int>(generated.speech_tokens.size()) < min_tokens) {
      row = mx::where(is_control, negative_infinity, row);
    } else {
    }
    if (options.repetition_penalty != 1.0f) {
      const mx::array penalty = mx::array(options.repetition_penalty);
      row = mx::where(seen,
                      mx::where(mx::greater(row, mx::array(0.0f)),
                                mx::divide(row, penalty),
                                mx::multiply(row, penalty)),
                      row);
    } else {
    }
    int token = 0;
    if (greedy) {
      token = mx::argmax(row).item<int>();
    } else {
      const mx::array scaled = mx::divide(row, mx::array(options.temperature));
      // Note (Jiaxin Deng): top-k and top-p both cut the sorted
      // probabilities of the whole row, as the CUDA sampler does.
      const mx::array order = mx::argsort(mx::negative(scaled));
      const mx::array sorted = mx::take(scaled, order);
      const mx::array probabilities = mx::softmax(sorted, -1, true);
      mx::array keep =
          mx::less_equal(mx::subtract(mx::cumsum(probabilities), probabilities),
                         mx::array(options.top_p));
      if (options.top_k > 0) {
        keep = mx::logical_and(keep,
                               mx::less(mx::arange(kTotalSpeechVocabularySize),
                                        mx::array(options.top_k)));
      } else {
      }
      const mx::array choice = mx::random::categorical(
          mx::where(keep, sorted, negative_infinity), -1, next_key());
      token = mx::take(order, choice).item<int>();
      const auto &recent = generated.speech_tokens;
      const size_t window_start = recent.size() > kRepetitionWindow
                                      ? recent.size() - kRepetitionWindow
                                      : 0;
      // Note (yexiaodong): a speech token among the last ten is drawn again
      // from the whole distribution without it, as the reference sampler
      // does to break loops.
      if (token < kSpeechTokenSize &&
          std::find(recent.begin() + window_start, recent.end(), token) !=
              recent.end()) {
        token = mx::random::categorical(
                    mx::where(mx::equal(vocabulary, mx::array(token)),
                              negative_infinity, scaled),
                    -1, next_key())
                    .item<int>();
      } else {
      }
    }
    if (token >= kSpeechTokenSize) {
      generated.finish_reason = qwen3_asr::FinishReason::kStop;
      break;
    } else {
    }
    generated.speech_tokens.push_back(token);
    seen = mx::logical_or(seen, mx::equal(vocabulary, mx::array(token)));
    on_token(token);
    logits = ForwardEmbeddings(SpeechEmbeddings({token}), caches);
  }
  return generated;
}

mx::array CosyVoice3Model::Mel(const std::vector<int> &speech_tokens,
                               const std::optional<Reference> &reference,
                               const std::function<void()> &on_step) const {
  if (reference.has_value()) {
    return flow_.Decode(speech_tokens, reference->flow_tokens,
                        reference->prompt_mel, reference->speaker_embedding,
                        on_step);
  } else {
    return flow_.Decode(
        speech_tokens, {}, mx::zeros({1, 0, kMelBinCount}, mx::float16),
        mx::zeros({1, kSpeakerEmbeddingSize}, mx::float32), on_step);
  }
}

mx::array CosyVoice3Model::Samples(const mx::array &mel) const {
  mx::array samples = hift_.Decode(mel);
  mx::eval(samples);
  return samples;
}

mx::array CosyVoice3Model::Vocode(const std::vector<int> &speech_tokens,
                                  const std::optional<Reference> &reference,
                                  const std::function<void()> &on_step) const {
  const mx::array mel = Mel(speech_tokens, reference, on_step);
  mx::eval(mel);
  on_step();
  return Samples(mel);
}

ReferenceAudio ReadReferenceAudio(std::string_view wav_bytes) {
  const DecodedAudio decoded = DecodeWav(wav_bytes);
  // Note (Jiaxin Deng): each encoder's rate is resampled from the file's, as
  // the CUDA server loads the reference once per rate.
  return {Resample(decoded.samples, decoded.sample_rate, 16000),
          Resample(decoded.samples, decoded.sample_rate, kSampleRate)};
}

std::vector<int> TrimSilentRuns(const std::vector<int> &speech_tokens) {
  std::vector<int> kept;
  int silent_run = 0;
  for (const int token : speech_tokens) {
    silent_run = kSilentTokens.count(token) > 0 ? silent_run + 1 : 0;
    if (silent_run <= kMaxSilentRun) {
      kept.push_back(token);
    } else {
    }
  }
  return kept;
}

} // namespace cosyvoice3
