// SPDX-License-Identifier: Apache-2.0
#include "whisper/whisper_model.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <stdexcept>
#include <string_view>
#include <utility>

#include "common/audio.h"
#include "common/generation.h"
#include "common/npz.h"
#include "nlohmann/json.hpp"

namespace whisper {

namespace mx = mlx::core;
using omni_common::Gelu;

namespace {

constexpr int kMlpQuantizationBits = 8;
constexpr int kQuantizationGroupSize = 64;

constexpr float kLayerNormEpsilon = 1e-5f;

// Note (khazic): fixed sinusoids of the encoder positions, computed in double
// precision.
mx::array Sinusoids(int length, int channels) {
  const int half = channels / 2;
  const double log_timescale_increment =
      std::log(10000.0) / static_cast<double>(std::max(half - 1, 1));
  std::vector<double> inverse_timescales(half);
  for (int i = 0; i < half; ++i)
    inverse_timescales[i] =
        std::exp(-log_timescale_increment * static_cast<double>(i));
  std::vector<float> values(static_cast<size_t>(length) * channels);
  for (int position = 0; position < length; ++position) {
    for (int i = 0; i < half; ++i) {
      const double scaled_time =
          static_cast<double>(position) * inverse_timescales[i];
      values[static_cast<size_t>(position) * channels + i] =
          static_cast<float>(std::sin(scaled_time));
      values[static_cast<size_t>(position) * channels + half + i] =
          static_cast<float>(std::cos(scaled_time));
    }
  }
  return mx::array(values.data(), {length, channels}, mx::float32);
}

WhisperConfig ReadConfig(const std::filesystem::path &model_directory,
                         bool draft = false) {
  const nlohmann::json config =
      omni_common::ReadJson(model_directory / "config.json");
  if (config.contains("quantization") ||
      config.contains("quantization_config")) {
    throw std::runtime_error("quantized Whisper checkpoints are not served");
  } else if (draft && !config.contains("n_mels") &&
             config.value("model_type", "") == "whisper") {
    if (!config.value("tie_word_embeddings", true) ||
        config.value("scale_embedding", false)) {
      throw std::runtime_error(
          "the Whisper draft needs tied, unscaled token embeddings");
    } else {
    }
    return {config.at("num_mel_bins").get<int>(),
            config.at("max_source_positions").get<int>(),
            config.at("d_model").get<int>(),
            config.at("encoder_attention_heads").get<int>(),
            config.at("encoder_layers").get<int>(),
            config.at("vocab_size").get<int>(),
            config.at("max_target_positions").get<int>(),
            config.at("decoder_attention_heads").get<int>(),
            config.at("decoder_layers").get<int>()};
  } else {
  }
  return {config.at("n_mels").get<int>(),
          config.at("n_audio_ctx").get<int>(),
          config.at("n_audio_state").get<int>(),
          config.at("n_audio_head").get<int>(),
          config.at("n_audio_layer").get<int>(),
          config.at("n_vocab").get<int>(),
          config.at("n_text_ctx").get<int>(),
          config.at("n_text_head").get<int>(),
          config.at("n_text_layer").get<int>()};
}

std::string NativeDraftWeightName(std::string name) {
  name.erase(0, std::string_view("model.").size());
  constexpr std::pair<std::string_view, std::string_view> replacements[] = {
      {"decoder.layers.", "decoder.blocks."},
      {".embed_tokens.weight", ".token_embedding.weight"},
      {".embed_positions.weight", ".positional_embedding"},
      {".self_attn_layer_norm.", ".attn_ln."},
      {".encoder_attn_layer_norm.", ".cross_attn_ln."},
      {".final_layer_norm.", ".mlp_ln."},
      {".self_attn.", ".attn."},
      {".encoder_attn.", ".cross_attn."},
      {".q_proj.", ".query."},
      {".k_proj.", ".key."},
      {".v_proj.", ".value."},
      {".out_proj.", ".out."},
      {".fc1.", ".mlp1."},
      {".fc2.", ".mlp2."},
      {"decoder.layer_norm.", "decoder.ln."},
  };
  for (const auto &[source, destination] : replacements) {
    if (const auto offset = name.find(source); offset != std::string::npos) {
      name.replace(offset, source.size(), destination);
    } else {
    }
  }
  return name;
}

// The tensors of a checkpoint's *.safetensors, else of its weights.npz; the
// safetensors ones stay unread until evaluated.
std::unordered_map<std::string, mx::array>
LoadWeights(const std::filesystem::path &model_directory,
            bool decoder_only = false) {
  std::vector<std::filesystem::path> weight_files;
  for (const auto &entry :
       std::filesystem::directory_iterator(model_directory)) {
    if (entry.path().extension() == ".safetensors") {
      weight_files.push_back(entry.path());
    } else {
    }
  }
  std::sort(weight_files.begin(), weight_files.end());
  std::unordered_map<std::string, mx::array> weights;
  // Note (khazic): mlx-community converted the older checkpoints to weights.npz
  // only.
  if (!weight_files.empty()) {
    for (const auto &path : weight_files) {
      auto [loaded, metadata] = mx::load_safetensors(path.string());
      for (auto &[name, array] : loaded) {
        if (!decoder_only || name.starts_with("decoder.")) {
          weights.insert_or_assign(name, array);
        } else if (name.starts_with("model.decoder.")) {
          weights.insert_or_assign(NativeDraftWeightName(name), array);
        } else {
        }
      }
    }
  } else if (std::filesystem::exists(model_directory / "weights.npz")) {
    weights = npz::Load(model_directory / "weights.npz");
    if (decoder_only) {
      std::erase_if(weights, [](const auto &entry) {
        return !entry.first.starts_with("decoder.");
      });
    } else {
    }
  } else {
    throw std::runtime_error("no Whisper weights in " +
                             model_directory.string());
  }
  return weights;
}

// [batch, length, width] to [batch, heads, length, width / heads].
mx::array SplitHeads(const mx::array &x, int head_count) {
  return mx::transpose(mx::reshape(x, {x.shape(0), x.shape(1), head_count,
                                       x.shape(2) / head_count}),
                       {0, 2, 1, 3});
}

// The target's weights, with the fixed encoder positions the checkpoint
// leaves out.
omni_common::WeightMap
TargetWeights(const std::filesystem::path &model_directory,
              const WhisperConfig &config) {
  omni_common::WeightMap weights = LoadWeights(model_directory);
  weights.erase("alignment_heads");
  // Note (khazic): the checkpoint leaves out the fixed encoder positions.
  if (weights.count("encoder.positional_embedding") == 0) {
    weights.insert_or_assign(
        "encoder.positional_embedding",
        mx::astype(Sinusoids(config.audio_context_length, config.audio_width),
                   weights.at("encoder.conv1.weight").dtype()));
  } else {
  }
  return weights;
}

// A draft's decoder weights in the target's dtype; refuses a draft whose
// decoder cannot read the target's encoder states.
omni_common::WeightMap
DraftWeights(const std::filesystem::path &draft_directory,
             const WhisperConfig &config, const WhisperConfig &target_config,
             mx::Dtype dtype) {
  const std::string draft = "the Whisper draft at " + draft_directory.string();
  if (config.audio_width != target_config.audio_width) {
    throw std::runtime_error(draft + " reads encoder states of width " +
                             std::to_string(config.audio_width) +
                             ", the target's are " +
                             std::to_string(target_config.audio_width));
  } else if (config.vocabulary_size != target_config.vocabulary_size) {
    throw std::runtime_error(draft + " has a vocabulary of " +
                             std::to_string(config.vocabulary_size) +
                             ", the target's is " +
                             std::to_string(target_config.vocabulary_size));
  } else if (config.text_context_length < target_config.text_context_length) {
    throw std::runtime_error(draft + " places " +
                             std::to_string(config.text_context_length) +
                             " text positions, fewer than the target's " +
                             std::to_string(target_config.text_context_length));
  } else {
  }
  // Note (XXXXRT666): the draft decodes the target's encoder states, so only
  // its decoder is loaded; an encoder unlike the target's only lowers how many
  // proposals the target accepts.
  omni_common::WeightMap weights =
      LoadWeights(draft_directory, /*decoder_only=*/true);
  for (auto &[name, weight] : weights) {
    weight = mx::astype(weight, dtype);
  }
  return weights;
}

} // namespace

WhisperModel::WhisperModel(const std::filesystem::path &model_directory)
    : config_(ReadConfig(model_directory)),
      // Note (XXXXRT666): the decoder's MLPs run in 8 bits, quantized at load;
      // its attention stays in the checkpoint dtype, since quantizing it
      // changed large-v3's mixed-language transcripts.
      checkpoint_(omni_common::QuantizedCheckpoint(
          TargetWeights(model_directory, config_), nlohmann::json::object(),
          kQuantizationGroupSize, [](const std::string &name) {
            return name.starts_with("decoder.blocks.") &&
                           name.find(".mlp") != std::string::npos
                       ? kMlpQuantizationBits
                       : 0;
          })) {}

WhisperModel::WhisperModel(const std::filesystem::path &draft_directory,
                           const WhisperModel &target)
    : config_(ReadConfig(draft_directory, /*draft=*/true)),
      checkpoint_(DraftWeights(draft_directory, config_, target.config_,
                               target.checkpoint_
                                   .Weight("decoder.token_embedding.weight")
                                   .dtype()),
                  nlohmann::json::object()) {}

mx::array WhisperModel::Attention(const mx::array &queries_input,
                                  const mx::array &keys,
                                  const mx::array &values,
                                  const std::string &prefix, int head_count,
                                  bool is_causal) const {
  const int batch = queries_input.shape(0);
  const int length = queries_input.shape(1);
  const int width = queries_input.shape(2);
  const mx::array queries = SplitHeads(
      checkpoint_.Linear(queries_input, prefix + ".query"), head_count);
  const float scale = std::pow(static_cast<float>(width / head_count), -0.5f);
  const mx::array attended = mx::fast::scaled_dot_product_attention(
      queries, keys, values, scale, /*window_size=*/-1,
      is_causal ? "causal" : "");
  return checkpoint_.Linear(mx::reshape(mx::transpose(attended, {0, 2, 1, 3}),
                                        {batch, length, width}),
                            prefix + ".out");
}

mx::array WhisperModel::EncoderLayer(const mx::array &x, int layer) const {
  const std::string prefix = "encoder.blocks." + std::to_string(layer);
  const int head_count = config_.audio_head_count;
  const mx::array normed =
      checkpoint_.LayerNorm(x, prefix + ".attn_ln", kLayerNormEpsilon);
  const mx::array keys =
      SplitHeads(checkpoint_.Linear(normed, prefix + ".attn.key"), head_count);
  const mx::array values = SplitHeads(
      checkpoint_.Linear(normed, prefix + ".attn.value"), head_count);
  const mx::array hidden = mx::add(
      x, Attention(normed, keys, values, prefix + ".attn", head_count, false));
  return mx::add(
      hidden,
      checkpoint_.Linear(Gelu(checkpoint_.Linear(
                             checkpoint_.LayerNorm(hidden, prefix + ".mlp_ln",
                                                   kLayerNormEpsilon),
                             prefix + ".mlp1")),
                         prefix + ".mlp2"));
}

mx::array WhisperModel::Encode(std::span<const float> padded_samples,
                               const std::atomic<bool> &cancel) const {
  std::vector<mx::array> mel_windows;
  for (size_t offset = 0; offset < padded_samples.size();
       offset += kWindowSampleCount) {
    mel_windows.push_back(omni_common::WhisperWindowFeatures(
        padded_samples.data() + offset, kWindowSampleCount,
        config_.mel_bin_count));
  }
  const mx::array mel_batch = mel_windows.size() == 1
                                  ? mx::expand_dims(mel_windows.front(), 0)
                                  : mx::stack(mel_windows);
  // Note (Jiaxin Deng): the float32 log-mel is cast to the weights' dtype, as
  // the reference Whisper does; fed as float32 it promotes the whole encoder,
  // and through the cross-attention the decoder, to float32.
  mx::array x =
      mx::astype(mel_batch, checkpoint_.Weight("encoder.conv1.weight").dtype());
  x = Gelu(checkpoint_.Conv1d(x, "encoder.conv1", 1, 1));
  x = Gelu(checkpoint_.Conv1d(x, "encoder.conv2", 2, 1));
  x = mx::add(x, mx::slice(checkpoint_.Weight("encoder.positional_embedding"),
                           {0, 0}, {x.shape(1), x.shape(2)}));
  for (int layer = 0; layer < config_.audio_layer_count; ++layer) {
    if (cancel.load()) {
      throw omni_common::GenerationCancelled();
    } else {
    }
    x = EncoderLayer(x, layer);
    // Note (Jiaxin Deng): evaluated layer by layer; as one graph the whole
    // encoder keeps each layer's temporaries until it finishes, about 0.6 GB
    // more at its peak for large-v3-turbo, at no measurable speed gain.
    mx::eval(x);
  }
  return checkpoint_.LayerNorm(x, "encoder.ln_post", kLayerNormEpsilon);
}

mx::array WhisperModel::DecoderLayer(const mx::array &x, int layer,
                                     const mx::array &encoder_states,
                                     bool is_causal,
                                     DecoderLayerCache &cache) const {
  const std::string prefix = "decoder.blocks." + std::to_string(layer);
  const int head_count = config_.text_head_count;
  const mx::array normed =
      checkpoint_.LayerNorm(x, prefix + ".attn_ln", kLayerNormEpsilon);
  const auto [keys, values] = cache.self_attention.UpdateAndFetch(
      SplitHeads(checkpoint_.Linear(normed, prefix + ".attn.key"), head_count),
      SplitHeads(checkpoint_.Linear(normed, prefix + ".attn.value"),
                 head_count));
  mx::array hidden =
      mx::add(x, Attention(normed, keys, values, prefix + ".attn", head_count,
                           is_causal));
  if (!cache.cross_keys.has_value()) {
    cache.cross_keys = SplitHeads(
        checkpoint_.Linear(encoder_states, prefix + ".cross_attn.key"),
        head_count);
    cache.cross_values = SplitHeads(
        checkpoint_.Linear(encoder_states, prefix + ".cross_attn.value"),
        head_count);
  } else {
  }
  hidden = mx::add(
      hidden, Attention(checkpoint_.LayerNorm(hidden, prefix + ".cross_attn_ln",
                                              kLayerNormEpsilon),
                        *cache.cross_keys, *cache.cross_values,
                        prefix + ".cross_attn", head_count, false));
  return mx::add(
      hidden,
      checkpoint_.Linear(Gelu(checkpoint_.Linear(
                             checkpoint_.LayerNorm(hidden, prefix + ".mlp_ln",
                                                   kLayerNormEpsilon),
                             prefix + ".mlp1")),
                         prefix + ".mlp2"));
}

mx::array WhisperModel::Decode(const mx::array &token_ids, int start_position,
                               const mx::array &encoder_states,
                               std::vector<DecoderLayerCache> &caches,
                               int logit_count) const {
  const int length = token_ids.shape(1);
  std::vector<int32_t> position_values(length);
  for (int i = 0; i < length; ++i)
    position_values[i] = start_position + i;
  const mx::array position_ids(position_values.data(), {length}, mx::int32);
  mx::array hidden =
      mx::add(mx::take(checkpoint_.Weight("decoder.token_embedding.weight"),
                       token_ids, 0),
              mx::expand_dims(
                  mx::take(checkpoint_.Weight("decoder.positional_embedding"),
                           position_ids, 0),
                  0));
  for (int layer = 0; layer < config_.text_layer_count; ++layer) {
    hidden =
        DecoderLayer(hidden, layer, encoder_states, length > 1, caches[layer]);
  }
  hidden = checkpoint_.LayerNorm(hidden, "decoder.ln", kLayerNormEpsilon);
  const int batch_size = hidden.shape(0);
  const int width = hidden.shape(2);
  const mx::array rows =
      mx::reshape(mx::slice(hidden, {0, length - logit_count, 0},
                            {batch_size, length, width}),
                  {batch_size * logit_count, width});
  // Note (khazic): tied output projection: the token embedding used as a linear
  // layer.
  return mx::matmul(rows, mx::transpose(checkpoint_.Weight(
                              "decoder.token_embedding.weight")));
}

void KeepCacheRows(std::vector<DecoderLayerCache> &caches,
                   const mx::array &rows) {
  for (auto &cache : caches) {
    cache.self_attention.KeepRows(rows);
    if (cache.cross_keys.has_value()) {
      cache.cross_keys = mx::take(*cache.cross_keys, rows, 0);
      cache.cross_values = mx::take(*cache.cross_values, rows, 0);
    } else {
    }
  }
}

void TruncateCaches(std::vector<DecoderLayerCache> &caches, int token_count) {
  for (auto &cache : caches) {
    cache.self_attention.Trim(
        std::max(0, cache.self_attention.offset() - token_count));
  }
}

} // namespace whisper
