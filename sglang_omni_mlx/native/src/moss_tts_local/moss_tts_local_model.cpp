// SPDX-License-Identifier: Apache-2.0
#include "moss_tts_local/moss_tts_local_model.h"

#include <cmath>
#include <limits>
#include <stdexcept>

#include "common/text.h"

namespace moss_tts_local {

namespace mx = mlx::core;

namespace {

constexpr const char *kBackbonePrefix = "transformer";
constexpr const char *kLocalPrefix = "local_transformer";
constexpr int kQuantizationGroupSize = 64;
// The user turn's template around its fields, as the checkpoint's processor
// renders it.
constexpr const char *kUserRolePrefix = "user\n";
constexpr const char *kReferencePrefix = "<user_inst>\n- Reference(s):\n";
constexpr const char *kTextFieldPrefix = "\n- Text:\n";
constexpr const char *kUserSuffix = "\n</user_inst>";
constexpr const char *kAssistantTurnPrefix = "\n";
constexpr const char *kAssistantRolePrefix = "assistant\n";

// Whether a checkpoint weight is quantized at load: the backbone's and the
// local transformer's matrices. The heads and the audio embeddings that
// decide every code stay in bfloat16.
bool IsQuantized(const std::string &name, const mx::array &weight) {
  const auto ends_with = [&](const std::string &suffix) {
    return name.size() >= suffix.size() &&
           name.compare(name.size() - suffix.size(), suffix.size(), suffix) ==
               0;
  };
  return weight.ndim() == 2 && ends_with(".weight") &&
         weight.shape(-1) % kQuantizationGroupSize == 0 &&
         (name.rfind("transformer.layers.", 0) == 0 ||
          name == "transformer.embed_tokens.weight" ||
          name.rfind("local_transformer.h.", 0) == 0);
}

std::vector<mx::array> LogSoftmaxGraph(const std::vector<mx::array> &inputs) {
  // Note (Jiaxin Deng): mlx.nn.log_softmax's op order, kept for bit identity.
  const mx::array shifted =
      mx::subtract(inputs[0], mx::stop_gradient(mx::max(inputs[0], -1, true)));
  return {mx::subtract(shifted, mx::logsumexp(shifted, -1, true))};
}

// Compiled shapeless as mlx.nn does.
mx::array LogSoftmax(const mx::array &logits) {
  static const auto compiled = mx::compile(LogSoftmaxGraph, true);
  return compiled({logits})[0];
}

mx::array NegativeInfinity(mx::Dtype dtype) {
  return mx::array(-std::numeric_limits<float>::infinity(), dtype);
}

// Logits [1, vocabulary] with every candidate outside the top_k most likely
// at -inf.
mx::array KeepTopK(const mx::array &logits, int top_k) {
  const int vocabulary = logits.shape(-1);
  if (top_k <= 0 || top_k >= vocabulary) {
    return logits;
  } else {
  }
  const mx::array log_probabilities = LogSoftmax(logits);
  const mx::array dropped = mx::slice(
      mx::argpartition(mx::negative(log_probabilities), top_k - 1, -1),
      {0, top_k}, {1, vocabulary});
  const mx::array masked =
      mx::put_along_axis(log_probabilities, dropped,
                         NegativeInfinity(log_probabilities.dtype()), -1);
  return mx::where(mx::equal(masked, NegativeInfinity(masked.dtype())),
                   NegativeInfinity(logits.dtype()), logits);
}

// Logits [1, vocabulary] keeping the smallest set of the most likely
// candidates whose probability reaches top_p.
mx::array KeepTopP(const mx::array &logits, double top_p) {
  if (top_p <= 0.0 || top_p >= 1.0) {
    return logits;
  } else {
  }
  const int vocabulary = logits.shape(-1);
  const mx::array log_probabilities = LogSoftmax(logits);
  const mx::array probabilities = mx::exp(log_probabilities);
  const mx::array ascending = mx::argsort(log_probabilities, -1);
  const mx::array cumulative =
      mx::cumsum(mx::take_along_axis(probabilities, ascending, -1), -1);
  const mx::array inverse = mx::put_along_axis(
      mx::zeros_like(ascending), ascending,
      mx::reshape(mx::arange(vocabulary, ascending.dtype()), {1, vocabulary}),
      -1);
  // Note (Jiaxin Deng): the most likely candidate always stays, as CUDA's
  // filter keeps it; a top_p below bfloat16's resolution would otherwise
  // leave none. For any larger top_p it passes anyway.
  const mx::array most_likely =
      mx::equal(mx::reshape(mx::arange(vocabulary), {1, vocabulary}),
                mx::argmax(log_probabilities, -1, true));
  const mx::array kept = mx::where(
      mx::logical_or(mx::greater(mx::take_along_axis(cumulative, inverse, -1),
                                 mx::array(static_cast<float>(1.0 - top_p),
                                           cumulative.dtype())),
                     most_likely),
      log_probabilities, NegativeInfinity(log_probabilities.dtype()));
  return mx::where(mx::equal(kept, NegativeInfinity(kept.dtype())),
                   NegativeInfinity(logits.dtype()), logits);
}

// Note (Jiaxin Deng): the repetition penalty of the mlx-audio port: the logit
// of each code seen earlier, seen_codes [vocabulary] true, is divided by the
// penalty when positive and multiplied by it when not.
mx::array PenalizeRepeats(const mx::array &logits,
                          const std::optional<mx::array> &seen_codes,
                          double penalty) {
  if (!seen_codes.has_value() || penalty == 1.0) {
    return logits;
  } else {
  }
  const mx::Dtype dtype = logits.dtype();
  const mx::array penalized = mx::where(
      mx::greater(logits, mx::array(0.0f, dtype)),
      mx::divide(logits, mx::array(static_cast<float>(penalty), dtype)),
      mx::multiply(logits, mx::array(static_cast<float>(penalty), dtype)));
  return mx::where(mx::expand_dims(*seen_codes, 0), penalized, logits);
}

// One sampled index [1] from logits [1, vocabulary], drawing on MLX's global
// random state as the reference sampler does.
mx::array SampleIndex(const mx::array &logits,
                      const SamplingParameters &parameters, int top_k,
                      const std::optional<mx::array> &seen_codes) {
  // A temperature too small for a float divides by zero; it samples greedily.
  const bool sampled = static_cast<float>(parameters.temperature) >=
                       std::numeric_limits<float>::min();
  mx::array scores =
      sampled ? mx::divide(logits,
                           mx::array(static_cast<float>(parameters.temperature),
                                     logits.dtype()))
              : logits;
  scores = PenalizeRepeats(scores, seen_codes, parameters.repetition_penalty);
  if (!sampled) {
    return mx::astype(mx::argmax(scores, -1), mx::int32);
  } else {
    scores = KeepTopP(KeepTopK(scores, top_k), parameters.top_p);
    return mx::astype(mx::random::categorical(scores, -1), mx::int32);
  }
}

} // namespace

MossTtsLocalModel::LoadedWeights
MossTtsLocalModel::LoadWeights(const std::filesystem::path &directory,
                               int quantization_bits) {
  if (quantization_bits != 4 && quantization_bits != 8 &&
      quantization_bits != 16) {
    throw std::invalid_argument("quantization bits must be 4, 8 or 16");
  } else {
  }
  LoadedWeights loaded;
  omni_common::WeightMap checkpoint = omni_common::LoadSafetensors(directory);
  while (!checkpoint.empty()) {
    // Note (Jiaxin Deng): each entry leaves the map as it is converted, so
    // the bfloat16 checkpoint is never resident at once.
    auto entry = checkpoint.extract(checkpoint.begin());
    const std::string &name = entry.key();
    mx::array &array = entry.mapped();
    // The full-vocabulary text head is unused: the binary head decides.
    if (name == "text_lm_head.weight") {
      continue;
    } else if (quantization_bits != 16 && IsQuantized(name, array)) {
      std::vector<mx::array> parts = mx::quantize(array, kQuantizationGroupSize,
                                                  quantization_bits, "affine");
      mx::eval(parts);
      const std::string prefix = name.substr(0, name.size() - 7);
      loaded.weights.insert_or_assign(name, parts[0]);
      loaded.weights.insert_or_assign(prefix + ".scales", parts[1]);
      loaded.weights.insert_or_assign(prefix + ".biases", parts[2]);
    } else {
      mx::eval(array);
      loaded.weights.insert_or_assign(name, array);
    }
  }
  mx::clear_cache();
  loaded.quantization = nlohmann::json::object();
  if (quantization_bits != 16) {
    loaded.quantization["quantization"] = {
        {"group_size", kQuantizationGroupSize}, {"bits", quantization_bits}};
  } else {
  }
  return loaded;
}

MossTtsLocalModel::MossTtsLocalModel(
    const std::filesystem::path &model_directory,
    const std::filesystem::path &codec_directory, int quantization_bits)
    : MossTtsLocalModel(model_directory, codec_directory,
                        omni_common::ReadJson(model_directory / "config.json"),
                        LoadWeights(model_directory, quantization_bits)) {}

MossTtsLocalModel::MossTtsLocalModel(
    const std::filesystem::path &model_directory,
    const std::filesystem::path &codec_directory, const nlohmann::json &config,
    LoadedWeights weights)
    : checkpoint_(std::move(weights.weights), weights.quantization),
      codec_(codec_directory, mx::bfloat16), prompt_encoder_(model_directory),
      backbone_(checkpoint_, kBackbonePrefix,
                qwen3_asr::Qwen3DecoderConfig::FromJson(
                    config.at("language_config"))) {
  if (config.at("model_type") != "moss_tts_local" ||
      config.value("local_text_head_mode", std::string()) != "binary") {
    throw std::runtime_error(
        "expected a MOSS-TTS-Local v1.5 checkpoint with a binary text head");
  } else {
  }
  const nlohmann::json &local = config.at("gpt2_config");
  if (local.at("position_embedding_type") != "rope" ||
      local.at("activation_function") != "silu") {
    throw std::runtime_error("unsupported local transformer");
  } else {
  }
  backbone_norm_eps_ =
      config.at("language_config").at("rms_norm_eps").get<float>();
  local_layer_count_ = config.value("local_transformer_layers", 1);
  local_head_count_ = local.at("n_head").get<int>();
  local_norm_eps_ = local.at("layer_norm_epsilon").get<float>();
  local_rope_base_ = local.at("rope_base").get<float>();
  codebook_count_ = config.at("n_vq").get<int>();
  audio_pad_code_ = config.at("audio_pad_code").get<int>();
  audio_end_id_ = config.at("audio_end_token_id").get<int>();
  audio_assistant_slot_id_ =
      config.at("audio_assistant_slot_token_id").get<int>();
  if (codec_.sample_rate() != config.at("sampling_rate").get<int>() ||
      codebook_count_ > codec_.codebook_count()) {
    throw std::runtime_error("the codec does not match the model");
  } else {
  }
}

PromptEncoder::PromptEncoder(const std::filesystem::path &model_directory)
    : tokenizer_(model_directory) {
  const nlohmann::json config =
      omni_common::ReadJson(model_directory / "config.json");
  codebook_count_ = config.at("n_vq").get<int>();
  audio_pad_code_ = config.at("audio_pad_code").get<int>();
  im_start_id_ = config.at("im_start_token_id").get<int>();
  im_end_id_ = config.at("im_end_token_id").get<int>();
  audio_start_id_ = config.at("audio_start_token_id").get<int>();
}

std::vector<std::vector<int>>
PromptEncoder::Rows(const SpeechPrompt &prompt) const {
  // Note (Jiaxin Deng): the checkpoint's processor strips each field and
  // writes None for a missing or blank one.
  const auto field = [](const std::optional<std::string> &value) {
    const std::string stripped =
        value.has_value() ? omni_common::StripUnicodeWhitespace(*value) : "";
    return stripped.empty() ? std::string("None") : stripped;
  };
  const std::optional<std::string> token_count =
      prompt.token_count.has_value()
          ? std::optional<std::string>(std::to_string(*prompt.token_count))
          : std::nullopt;
  std::vector<int> ids = {im_start_id_};
  const auto append = [&](const std::string &text) {
    const std::vector<int> encoded = tokenizer_.Encode(text);
    ids.insert(ids.end(), encoded.begin(), encoded.end());
  };
  append(kUserRolePrefix);
  append(kReferencePrefix);
  append("None");
  append("\n- Instruction:\n" + field(prompt.instruction) + "\n- Tokens:\n" +
         field(token_count) + "\n- Quality:\nNone\n- Sound Event:\nNone\n" +
         "- Ambient Sound:\nNone\n- Language:\n" + field(prompt.language) +
         kTextFieldPrefix);
  append(prompt.text);
  append(kUserSuffix);
  ids.push_back(im_end_id_);
  append(kAssistantTurnPrefix);
  ids.push_back(im_start_id_);
  append(kAssistantRolePrefix);
  ids.push_back(audio_start_id_);
  std::vector<std::vector<int>> rows;
  rows.reserve(ids.size());
  for (const int id : ids) {
    std::vector<int> row(codebook_count_ + 1, audio_pad_code_);
    row[0] = id;
    rows.push_back(std::move(row));
  }
  return rows;
}

mx::array MossTtsLocalModel::EmbedRows(const mx::array &rows) const {
  const int length = rows.shape(1);
  const auto column = [&](int index) {
    return mx::reshape(mx::slice(rows, {0, 0, index}, {1, length, index + 1}),
                       {1, length});
  };
  mx::array embeddings = backbone_.EmbedTokens(column(0));
  for (int codebook = 0; codebook < codebook_count_; ++codebook) {
    const mx::array codes = column(codebook + 1);
    const mx::array valid = mx::not_equal(codes, mx::array(audio_pad_code_));
    const mx::array safe_codes =
        mx::astype(mx::where(valid, codes, mx::array(0)), mx::int32);
    embeddings =
        mx::add(embeddings,
                mx::multiply(
                    checkpoint_.Embed(safe_codes, "audio_embeddings." +
                                                      std::to_string(codebook)),
                    mx::expand_dims(valid, -1)));
  }
  return embeddings;
}

mx::array MossTtsLocalModel::LocalAttention(const mx::array &x,
                                            const std::string &prefix) const {
  const int length = x.shape(1);
  const int hidden_size = x.shape(2);
  const int head_dim = hidden_size / local_head_count_;
  const std::vector<mx::array> projections =
      mx::split(checkpoint_.Linear(x, prefix + ".c_attn"), 3, -1);
  // Rotary embedding of adjacent pairs, with cosines computed in float32.
  const mx::array positions = mx::arange(0.0, length, mx::float32);
  const mx::array inverse_frequencies = mx::divide(
      mx::array(1.0f),
      mx::power(mx::array(local_rope_base_),
                mx::divide(mx::arange(0.0, head_dim, 2.0, mx::float32),
                           mx::array(static_cast<float>(head_dim)))));
  const mx::array frequencies =
      mx::multiply(mx::reshape(positions, {length, 1}),
                   mx::reshape(inverse_frequencies, {1, head_dim / 2}));
  const mx::array cosine =
      mx::astype(mx::reshape(mx::repeat(mx::cos(frequencies), 2, -1),
                             {1, length, 1, head_dim}),
                 x.dtype());
  const mx::array sine =
      mx::astype(mx::reshape(mx::repeat(mx::sin(frequencies), 2, -1),
                             {1, length, 1, head_dim}),
                 x.dtype());
  const auto heads = [&](const mx::array &projection, bool rotate) {
    const mx::array split =
        mx::reshape(projection, {1, length, local_head_count_, head_dim});
    if (!rotate) {
      return mx::transpose(split, {0, 2, 1, 3});
    } else {
    }
    const mx::Shape stop = {1, length, local_head_count_, head_dim};
    const mx::array even = mx::slice(split, {0, 0, 0, 0}, stop, {1, 1, 1, 2});
    const mx::array odd = mx::slice(split, {0, 0, 0, 1}, stop, {1, 1, 1, 2});
    const mx::array rotated_half =
        mx::reshape(mx::stack({mx::negative(odd), even}, -1), split.shape());
    return mx::transpose(
        mx::add(mx::multiply(split, cosine), mx::multiply(rotated_half, sine)),
        {0, 2, 1, 3});
  };
  const mx::array visible =
      mx::less_equal(mx::reshape(mx::arange(length), {1, length}),
                     mx::reshape(mx::arange(length), {length, 1}));
  const mx::array mask = mx::reshape(
      mx::astype(
          mx::where(visible, mx::array(0.0f),
                    mx::array(static_cast<float>(mx::finfo(x.dtype()).min))),
          x.dtype()),
      {1, 1, length, length});
  const mx::array attended = mx::fast::scaled_dot_product_attention(
      heads(projections[0], true), heads(projections[1], true),
      heads(projections[2], false),
      static_cast<float>(std::pow(static_cast<double>(head_dim), -0.5)), "",
      mask);
  return checkpoint_.Linear(mx::reshape(mx::transpose(attended, {0, 2, 1, 3}),
                                        {1, length, hidden_size}),
                            prefix + ".c_proj");
}

mx::array MossTtsLocalModel::LocalLast(const mx::array &inputs) const {
  const std::string local = kLocalPrefix;
  mx::array hidden = inputs;
  for (int layer = 0; layer < local_layer_count_; ++layer) {
    const std::string prefix = local + ".h." + std::to_string(layer);
    hidden = mx::add(
        hidden, LocalAttention(checkpoint_.LayerNorm(hidden, prefix + ".ln_1",
                                                     local_norm_eps_),
                               prefix + ".attn"));
    hidden = mx::add(
        hidden,
        checkpoint_.Linear(omni_common::Silu(checkpoint_.Linear(
                               checkpoint_.LayerNorm(hidden, prefix + ".ln_2",
                                                     local_norm_eps_),
                               prefix + ".mlp.fc_in")),
                           prefix + ".mlp.fc_out"));
  }
  const mx::array normed =
      checkpoint_.LayerNorm(hidden, local + ".ln_f", local_norm_eps_);
  const int length = normed.shape(1);
  const int hidden_size = normed.shape(2);
  return mx::reshape(
      mx::slice(normed, {0, length - 1, 0}, {1, length, hidden_size}),
      {1, hidden_size});
}

qwen3_asr::FinishReason MossTtsLocalModel::Generate(
    const std::vector<std::vector<int>> &prompt_rows,
    const GenerationParameters &parameters, const std::atomic<bool> &cancel,
    const std::function<void(const mx::array &)> &on_frame) const {
  const int channels = codebook_count_ + 1;
  std::vector<int> flat;
  flat.reserve(prompt_rows.size() * channels);
  for (const std::vector<int> &row : prompt_rows) {
    if (static_cast<int>(row.size()) != channels) {
      throw std::invalid_argument("a prompt row has the wrong width");
    } else {
    }
    flat.insert(flat.end(), row.begin(), row.end());
  }
  mx::random::seed(parameters.seed);
  const mx::array candidates =
      mx::array({audio_assistant_slot_id_, audio_end_id_});
  // Which codes each codebook has produced so far, for the penalty.
  std::vector<std::optional<mx::array>> seen_codes(codebook_count_);
  std::vector<qwen3_asr::KVCache> caches = backbone_.NewCaches();
  mx::array rows =
      mx::array(flat.begin(),
                {1, static_cast<int>(prompt_rows.size()), channels}, mx::int32);
  for (int frame = 0; frame < parameters.max_frames; ++frame) {
    if (cancel.load()) {
      throw qwen3_asr::TranscriptionCancelled();
    } else {
    }
    const mx::array hidden = backbone_.Forward(EmbedRows(rows), caches);
    const int length = hidden.shape(1);
    const int hidden_size = hidden.shape(2);
    mx::array local_inputs = mx::reshape(
        checkpoint_.RmsNorm(
            mx::slice(hidden, {0, length - 1, 0}, {1, length, hidden_size}),
            std::string(kBackbonePrefix) + ".norm", backbone_norm_eps_),
        {1, 1, hidden_size});
    mx::array local_hidden = LocalLast(local_inputs);
    // Note (Jiaxin Deng): the binary head's two candidates are the audio
    // slot (continue) and the end of audio.
    const mx::array decision = mx::take(
        candidates,
        SampleIndex(checkpoint_.Linear(local_hidden, "local_text_lm_head"),
                    parameters.text, std::min(parameters.text.top_k, 2),
                    std::nullopt),
        0);
    mx::eval(decision);
    if (decision.item<int32_t>() != audio_assistant_slot_id_) {
      return qwen3_asr::FinishReason::kStop;
    } else {
    }
    std::vector<mx::array> codes;
    for (int codebook = 0; codebook < codebook_count_; ++codebook) {
      const mx::array code = SampleIndex(
          checkpoint_.Linear(local_hidden,
                             "audio_lm_heads." + std::to_string(codebook)),
          parameters.audio, parameters.audio.top_k, seen_codes[codebook]);
      codes.push_back(code);
      if (codebook + 1 < codebook_count_) {
        local_inputs = mx::concatenate(
            {local_inputs,
             mx::expand_dims(
                 checkpoint_.Embed(code, "audio_embeddings." +
                                             std::to_string(codebook)),
                 1)},
            1);
        local_hidden = LocalLast(local_inputs);
      } else {
      }
    }
    const mx::array frame_codes = mx::concatenate(codes, 0);
    std::vector<mx::array> evaluated = {frame_codes};
    if (parameters.audio.repetition_penalty != 1.0) {
      for (int codebook = 0; codebook < codebook_count_; ++codebook) {
        const mx::array produced = mx::equal(
            mx::arange(checkpoint_
                           .Weight("audio_lm_heads." +
                                   std::to_string(codebook) + ".weight")
                           .shape(0)),
            codes[codebook]);
        seen_codes[codebook] =
            seen_codes[codebook].has_value()
                ? mx::logical_or(*seen_codes[codebook], produced)
                : produced;
        evaluated.push_back(*seen_codes[codebook]);
      }
    } else {
    }
    mx::eval(evaluated);
    rows = mx::reshape(
        mx::concatenate({mx::array({audio_assistant_slot_id_}), frame_codes},
                        0),
        {1, 1, channels});
    on_frame(frame_codes);
  }
  return qwen3_asr::FinishReason::kLength;
}

} // namespace moss_tts_local
