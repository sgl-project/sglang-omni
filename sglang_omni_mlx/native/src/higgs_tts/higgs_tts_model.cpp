// SPDX-License-Identifier: Apache-2.0
#include "higgs_tts/higgs_tts_model.h"

#include <limits>
#include <optional>
#include <stdexcept>

namespace higgs_tts {

namespace mx = mlx::core;

namespace {

constexpr const char *kDecoderPrefix = "model";
constexpr const char *kAudioEmbedding = "audio_embedding";
constexpr const char *kCheckpointLayerPrefix = "body.layers.";
constexpr const char *kCheckpointNorm = "body.norm.weight";
constexpr const char *kCheckpointTextEmbedding =
    "tied.embedding.text_embedding.weight";
constexpr const char *kCheckpointAudioEmbedding =
    "tied.embedding.modality_embeddings.0.embedding.weight";
constexpr const char *kCheckpointCodecPrefix =
    "tied.embedding.modality_embeddings.0.model.";
constexpr int kQuantizationGroupSize = 64;
// Note (Jiaxin Deng): the reference samplers' greedy threshold.
constexpr double kGreedyTemperature = 1e-5;

// A scalar of x's dtype, as Python scalars combine with arrays.
mx::array Scalar(double value, const mx::array &x) {
  return mx::array(static_cast<float>(value), x.dtype());
}

std::vector<mx::array> LogSoftmaxGraph(const std::vector<mx::array> &inputs) {
  const mx::array shifted =
      mx::subtract(inputs[0], mx::stop_gradient(mx::max(inputs[0], -1, true)));
  return {mx::subtract(shifted, mx::logsumexp(shifted, -1, true))};
}

// Note (Jiaxin Deng): mlx.nn.log_softmax's graph, compiled shapeless as it
// is, for bit identity.
mx::array LogSoftmax(const mx::array &logits) {
  static const auto compiled = mx::compile(LogSoftmaxGraph, true);
  return compiled({logits})[0];
}

// logits with -inf wherever filtered log-probabilities are -inf.
mx::array KeepUnfiltered(const mx::array &logits,
                         const mx::array &filtered_logprobs) {
  const mx::array negative_infinity =
      Scalar(-std::numeric_limits<double>::infinity(), logits);
  return mx::where(mx::equal(filtered_logprobs, negative_infinity),
                   negative_infinity, logits);
}

bool IsQuantizedWeight(const std::string &name) {
  return name == std::string(kDecoderPrefix) + ".embed_tokens.weight" ||
         (name.rfind(std::string(kDecoderPrefix) + ".layers.", 0) == 0 &&
          name.find("_proj.weight") != std::string::npos);
}

} // namespace

nlohmann::json HiggsTtsModel::CheckpointConfig(int quantize_bits) {
  if (quantize_bits == 0) {
    return nlohmann::json::object();
  } else {
    return {
        {"quantization",
         {{"group_size", kQuantizationGroupSize}, {"bits", quantize_bits}}}};
  }
}

HiggsTtsModel::LoadedWeights
HiggsTtsModel::LoadWeights(const std::filesystem::path &directory,
                           const nlohmann::json &config, int quantize_bits) {
  if (quantize_bits != 0 && quantize_bits != 8) {
    throw std::invalid_argument("quantize_bits must be 0 or 8");
  } else {
  }
  const std::string codec_prefix = kCheckpointCodecPrefix;
  const std::string layer_prefix = kCheckpointLayerPrefix;
  const std::string decoder_prefix = kDecoderPrefix;
  LoadedWeights weights;
  omni_common::WeightMap loaded = omni_common::LoadSafetensors(directory);
  // Note (Jiaxin Deng): each entry leaves the loaded map as it is read, so a
  // bfloat16 tensor quantized below is freed instead of staying resident.
  for (auto entry = loaded.begin(); entry != loaded.end();
       entry = loaded.erase(entry)) {
    const std::string &name = entry->first;
    const mx::array &array = entry->second;
    std::string renamed;
    if (name.rfind(codec_prefix, 0) == 0) {
      weights.codec.insert_or_assign(name.substr(codec_prefix.size()), array);
      continue;
    } else if (name.rfind(layer_prefix, 0) == 0) {
      renamed = decoder_prefix + ".layers." + name.substr(layer_prefix.size());
    } else if (name == kCheckpointNorm) {
      renamed = decoder_prefix + ".norm.weight";
    } else if (name == kCheckpointTextEmbedding) {
      renamed = decoder_prefix + ".embed_tokens.weight";
    } else if (name == kCheckpointAudioEmbedding) {
      renamed = std::string(kAudioEmbedding) + ".weight";
    } else {
      continue;
    }
    if (quantize_bits != 0 && IsQuantizedWeight(renamed)) {
      // Note (Jiaxin Deng): each tensor is quantized and evaluated alone, so
      // the bfloat16 checkpoint is never resident at once.
      const std::vector<mx::array> quantized =
          mx::quantize(array, kQuantizationGroupSize, quantize_bits);
      mx::eval(quantized);
      const std::string stem = renamed.substr(0, renamed.size() - 7);
      weights.model.insert_or_assign(renamed, quantized[0]);
      weights.model.insert_or_assign(stem + ".scales", quantized[1]);
      weights.model.insert_or_assign(stem + ".biases", quantized[2]);
    } else {
      weights.model.insert_or_assign(renamed, array);
    }
  }
  if (config.at("audio_encoder_config").at("num_codebooks").get<int>() !=
          kCodebookCount ||
      config.at("audio_encoder_config").at("vocab_size").get<int>() !=
          kAudioCodeCount) {
    throw std::runtime_error("the checkpoint's audio codebooks are not the "
                             "8 x 1026 Higgs Audio v3 speaks");
  } else {
  }
  return weights;
}

HiggsTtsModel::HiggsTtsModel(const std::filesystem::path &model_directory,
                             int quantize_bits)
    : HiggsTtsModel(
          model_directory,
          omni_common::ReadJson(model_directory / "config.json"), quantize_bits,
          LoadWeights(model_directory,
                      omni_common::ReadJson(model_directory / "config.json"),
                      quantize_bits)) {}

HiggsTtsModel::HiggsTtsModel(const std::filesystem::path &model_directory,
                             const nlohmann::json &config, int quantize_bits,
                             LoadedWeights weights)
    : checkpoint_(std::move(weights.model), CheckpointConfig(quantize_bits)),
      codec_(std::move(weights.codec)), tokenizer_(model_directory),
      decoder_(checkpoint_, kDecoderPrefix,
               {config.at("text_config").at("num_hidden_layers").get<int>(),
                config.at("text_config").at("num_attention_heads").get<int>(),
                config.at("text_config").at("num_key_value_heads").get<int>(),
                config.at("text_config").at("head_dim").get<int>(),
                config.at("text_config").at("rms_norm_eps").get<float>(),
                config.at("text_config")
                    .at("rope_parameters")
                    .at("rope_theta")
                    .get<float>()}),
      norm_eps_(config.at("text_config").at("rms_norm_eps").get<float>()),
      tts_token_id_(tokenizer_.AddedTokenId("<|tts|>")),
      text_token_id_(tokenizer_.AddedTokenId("<|text|>")),
      audio_token_id_(tokenizer_.AddedTokenId("<|audio|>")),
      codebook_offsets_(mx::multiply(mx::arange(kCodebookCount, mx::int32),
                                     mx::array(kAudioCodeCount))) {
  if (tts_token_id_ < 0 || text_token_id_ < 0 || audio_token_id_ < 0) {
    throw std::runtime_error("the tokenizer lacks Higgs Audio v3's "
                             "<|tts|>, <|text|> or <|audio|> token");
  } else {
  }
  mx::eval(codebook_offsets_);
}

std::vector<int> HiggsTtsModel::EncodePrompt(const std::string &text) const {
  std::vector<int> ids = {tts_token_id_, text_token_id_};
  const std::vector<int> text_ids = tokenizer_.Encode(text);
  ids.insert(ids.end(), text_ids.begin(), text_ids.end());
  ids.push_back(audio_token_id_);
  return ids;
}

qwen3_asr::FinishReason HiggsTtsModel::Generate(
    const std::vector<int> &prompt_ids, const SamplingOptions &sampling,
    int max_rows, const std::atomic<bool> &cancel,
    const std::function<void(const mx::array &)> &on_row) const {
  const mx::array ids = mx::array(
      prompt_ids.begin(), {1, static_cast<int>(prompt_ids.size())}, mx::int32);
  std::vector<qwen3_asr::KVCache> caches = decoder_.NewCaches();
  mx::array hidden = LastHidden(decoder_.EmbedTokens(ids), caches);
  // Note (Jiaxin Deng): the CUDA sampler's delay pattern. Row r opens
  // codebooks up to r, the rest held at begin of audio; once the first
  // codebook ends, the other codebooks get kCodebookCount - 2 more rows.
  int delayed_rows = 0;
  std::optional<int> rows_until_end;
  for (int row = 0; row < max_rows; ++row) {
    if (cancel.load()) {
      throw qwen3_asr::TranscriptionCancelled();
    } else {
    }
    mx::array sampled = SampleRow(hidden, sampling);
    mx::eval(sampled);
    std::vector<int32_t> codes(sampled.data<int32_t>(),
                               sampled.data<int32_t>() + kCodebookCount);
    bool done = false;
    if (delayed_rows < kCodebookCount) {
      for (int codebook = delayed_rows + 1; codebook < kCodebookCount;
           ++codebook) {
        codes[codebook] = kBeginOfAudioCode;
      }
      ++delayed_rows;
    } else if (rows_until_end.has_value()) {
      --*rows_until_end;
      done = *rows_until_end <= 0;
    } else if (codes[0] == kEndOfAudioCode) {
      rows_until_end = kCodebookCount - 2;
    } else {
    }
    const mx::array row_codes =
        mx::array(codes.begin(), {kCodebookCount}, mx::int32);
    if (!done && row + 1 < max_rows) {
      hidden = LastHidden(EmbedRow(row_codes), caches);
      // Note (Minyi Tang): the decoder steps on the GPU while on_row
      // handles this row.
      mx::async_eval(hidden);
    } else {
    }
    on_row(row_codes);
    if (done) {
      return qwen3_asr::FinishReason::kStop;
    } else {
    }
  }
  return qwen3_asr::FinishReason::kLength;
}

mx::array
HiggsTtsModel::LastHidden(const mx::array &embeddings,
                          std::vector<qwen3_asr::KVCache> &caches) const {
  const mx::array hidden = decoder_.Forward(embeddings, caches);
  const int length = hidden.shape(1);
  const int dimension = hidden.shape(2);
  return checkpoint_.RmsNorm(
      mx::reshape(mx::slice(hidden, {0, length - 1, 0}, {1, length, dimension}),
                  {1, dimension}),
      std::string(kDecoderPrefix) + ".norm", norm_eps_);
}

mx::array HiggsTtsModel::SampleRow(const mx::array &hidden,
                                   const SamplingOptions &sampling) const {
  mx::array logits =
      mx::reshape(checkpoint_.TiedProjection(hidden, kAudioEmbedding),
                  {kCodebookCount, kAudioCodeCount});
  if (sampling.temperature <= kGreedyTemperature || sampling.top_k == 1) {
    return mx::astype(mx::argmax(logits, -1), mx::int32);
  } else {
  }
  // Note (Jiaxin Deng): the MLX reference sampler's ops in its order and
  // dtypes, so a seed draws the same codes: temperature, then top-k and
  // top-p as masks of log-probabilities, then a categorical draw.
  logits = mx::divide(logits, Scalar(sampling.temperature, logits));
  if (sampling.top_k > 0 && sampling.top_k < kAudioCodeCount) {
    const mx::array logprobs = LogSoftmax(logits);
    const mx::array dropped = mx::slice(
        mx::argpartition(mx::negative(logprobs), sampling.top_k - 1, -1),
        {0, sampling.top_k}, {kCodebookCount, kAudioCodeCount});
    logits = KeepUnfiltered(
        logits,
        mx::put_along_axis(
            logprobs, dropped,
            Scalar(-std::numeric_limits<double>::infinity(), logprobs), -1));
  } else {
  }
  if (sampling.top_p > 0.0 && sampling.top_p < 1.0) {
    const mx::array logprobs = LogSoftmax(logits);
    const mx::array sorted_indices = mx::argsort(logprobs, -1);
    const mx::array cumulative_sorted = mx::cumsum(
        mx::take_along_axis(mx::exp(logprobs), sorted_indices, -1), -1);
    const mx::array inverse_indices = mx::put_along_axis(
        mx::zeros_like(sorted_indices), sorted_indices,
        mx::arange(kAudioCodeCount, sorted_indices.dtype()), -1);
    const mx::array cumulative =
        mx::take_along_axis(cumulative_sorted, inverse_indices, -1);
    logits = KeepUnfiltered(
        logits,
        mx::where(
            mx::greater(cumulative, Scalar(1.0 - sampling.top_p, cumulative)),
            logprobs,
            Scalar(-std::numeric_limits<double>::infinity(), logprobs)));
  } else {
  }
  return mx::astype(mx::random::categorical(logits, -1), mx::int32);
}

mx::array HiggsTtsModel::EmbedRow(const mx::array &codes) const {
  return mx::expand_dims(
      mx::sum(checkpoint_.Embed(mx::add(mx::reshape(codes, {1, kCodebookCount}),
                                        codebook_offsets_),
                                kAudioEmbedding),
              -2),
      0);
}

} // namespace higgs_tts
