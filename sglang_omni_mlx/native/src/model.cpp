// SPDX-License-Identifier: Apache-2.0
#include "model.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>

#include "nlohmann/json.hpp"

namespace qwen3_asr {

namespace mx = mlx::core;

namespace {

constexpr int kKvCacheStepTokens = 256;

std::vector<mx::array> GeluGraph(const std::vector<mx::array> &inputs) {
  // Note (Jiaxin Deng): mlx.nn.gelu's op order, kept for bit identity.
  const mx::array &x = inputs[0];
  return {mx::divide(
      mx::multiply(x, mx::add(mx::array(1.0f),
                              mx::erf(mx::divide(
                                  x, mx::array(static_cast<float>(M_SQRT2)))))),
      mx::array(2.0f))};
}

std::vector<mx::array> SiluGraph(const std::vector<mx::array> &inputs) {
  return {mx::multiply(inputs[0], mx::sigmoid(inputs[0]))};
}

// Note (Jiaxin Deng): compiled shapeless as mlx.nn does, so each activation
// is one fused kernel instead of one per elementwise op.
mx::array Gelu(const mx::array &x) {
  static const auto compiled = mx::compile(GeluGraph, true);
  return compiled({x})[0];
}

mx::array Silu(const mx::array &x) {
  static const auto compiled = mx::compile(SiluGraph, true);
  return compiled({x})[0];
}

mx::array SinusoidalPositions(int length, int channels) {
  const double timescale_step = std::log(10000.0) / (channels / 2 - 1);
  const mx::array inverse_timescales =
      mx::exp(mx::multiply(mx::array(static_cast<float>(-timescale_step)),
                           mx::arange(channels / 2, mx::float32)));
  const mx::array scaled_time =
      mx::multiply(mx::expand_dims(mx::arange(length, mx::float32), 1),
                   mx::expand_dims(inverse_timescales, 0));
  return mx::concatenate({mx::sin(scaled_time), mx::cos(scaled_time)}, 1);
}

} // namespace

std::pair<mx::array, mx::array>
KVCache::UpdateAndFetch(const mx::array &keys, const mx::array &values) {
  const int new_token_count = keys.shape(2);
  if (!keys_.has_value() || offset_ + new_token_count > keys_->shape(2)) {
    const int step_count =
        (new_token_count + kKvCacheStepTokens - 1) / kKvCacheStepTokens;
    const mx::Shape grown = {keys.shape(0), keys.shape(1),
                             step_count * kKvCacheStepTokens, keys.shape(3)};
    const mx::array extra_keys = mx::zeros(grown, keys.dtype());
    const mx::array extra_values = mx::zeros(grown, values.dtype());
    if (!keys_.has_value()) {
      keys_ = extra_keys;
      values_ = extra_values;
    } else {
      const mx::Shape kept = {keys_->shape(0), keys_->shape(1), offset_,
                              keys_->shape(3)};
      keys_ = mx::concatenate(
          {mx::slice(*keys_, {0, 0, 0, 0}, kept), extra_keys}, 2);
      values_ = mx::concatenate(
          {mx::slice(*values_, {0, 0, 0, 0}, kept), extra_values}, 2);
    }
  } else {
  }
  const mx::Shape start = {0, 0, offset_, 0};
  const mx::Shape stop = {keys.shape(0), keys.shape(1),
                          offset_ + new_token_count, keys.shape(3)};
  keys_ = mx::slice_update(*keys_, keys, start, stop);
  values_ = mx::slice_update(*values_, values, start, stop);
  offset_ += new_token_count;
  const mx::Shape fetched = {keys_->shape(0), keys_->shape(1), offset_,
                             keys_->shape(3)};
  return {mx::slice(*keys_, {0, 0, 0, 0}, fetched),
          mx::slice(*values_, {0, 0, 0, 0}, fetched)};
}

Qwen3ASR::Qwen3ASR(const std::filesystem::path &model_directory) {
  std::ifstream config_stream(model_directory / "config.json");
  if (!config_stream) {
    throw std::runtime_error("cannot read config.json in " +
                             model_directory.string());
  } else {
  }
  const nlohmann::json config = nlohmann::json::parse(config_stream);
  const nlohmann::json &audio = config.at("thinker_config").at("audio_config");
  const nlohmann::json &text = config.at("thinker_config").at("text_config");
  audio_ = {audio.at("num_mel_bins").get<int>(),
            audio.at("encoder_layers").get<int>(),
            audio.at("encoder_attention_heads").get<int>(),
            audio.at("d_model").get<int>(),
            audio.at("n_window").get<int>(),
            audio.at("n_window_infer").get<int>()};
  text_ = {text.at("num_hidden_layers").get<int>(),
           text.at("num_attention_heads").get<int>(),
           text.at("num_key_value_heads").get<int>(),
           text.at("head_dim").get<int>(),
           text.at("rms_norm_eps").get<float>(),
           text.at("rope_theta").get<float>()};
  if (config.contains("quantization")) {
    const nlohmann::json &quantization = config.at("quantization");
    quantization_ = {quantization.at("group_size").get<int>(),
                     quantization.at("bits").get<int>(),
                     quantization.value("mode", std::string("affine"))};
  } else {
  }
  std::vector<std::filesystem::path> weight_files;
  for (const auto &entry :
       std::filesystem::directory_iterator(model_directory)) {
    if (entry.path().extension() == ".safetensors") {
      weight_files.push_back(entry.path());
    } else {
    }
  }
  std::sort(weight_files.begin(), weight_files.end());
  for (const auto &path : weight_files) {
    auto [loaded, metadata] = mx::load_safetensors(path.string());
    for (auto &[name, array] : loaded) {
      weights_.insert_or_assign(name, array);
    }
  }
  std::vector<mx::array> parameters;
  parameters.reserve(weights_.size());
  for (const auto &[name, array] : weights_)
    parameters.push_back(array);
  mx::eval(parameters);
}

const mx::array &Qwen3ASR::Weight(const std::string &name) const {
  const auto found = weights_.find(name);
  if (found == weights_.end()) {
    throw std::runtime_error("checkpoint is missing " + name);
  } else {
  }
  return found->second;
}

bool Qwen3ASR::Has(const std::string &name) const {
  return weights_.count(name) > 0;
}

mx::array Qwen3ASR::Linear(const mx::array &x,
                           const std::string &prefix) const {
  if (Has(prefix + ".scales")) {
    const std::optional<mx::array> biases =
        Has(prefix + ".biases")
            ? std::optional<mx::array>(Weight(prefix + ".biases"))
            : std::nullopt;
    mx::array y = mx::quantized_matmul(
        x, Weight(prefix + ".weight"), Weight(prefix + ".scales"), biases, true,
        quantization_.group_size, quantization_.bits, quantization_.mode);
    if (Has(prefix + ".bias")) {
      return mx::add(y, Weight(prefix + ".bias"));
    } else {
      return y;
    }
  } else if (Has(prefix + ".bias")) {
    return mx::addmm(Weight(prefix + ".bias"), x,
                     mx::transpose(Weight(prefix + ".weight")));
  } else {
    return mx::matmul(x, mx::transpose(Weight(prefix + ".weight")));
  }
}

mx::array Qwen3ASR::LayerNorm(const mx::array &x,
                              const std::string &prefix) const {
  return mx::fast::layer_norm(x, Weight(prefix + ".weight"),
                              Weight(prefix + ".bias"), 1e-5f);
}

mx::array Qwen3ASR::RmsNorm(const mx::array &x,
                            const std::string &prefix) const {
  return mx::fast::rms_norm(x, Weight(prefix + ".weight"), text_.rms_norm_eps);
}

mx::array Qwen3ASR::Conv2d(const mx::array &x,
                           const std::string &prefix) const {
  return mx::add(mx::conv2d(x, Weight(prefix + ".weight"), {2, 2}, {1, 1}),
                 Weight(prefix + ".bias"));
}

mx::array Qwen3ASR::AudioEncoderLayer(const mx::array &x, int layer) const {
  const std::string prefix = "audio_tower.layers." + std::to_string(layer);
  const int batch = x.shape(0);
  const int length = x.shape(1);
  const int width = x.shape(2);
  const int head_count = audio_.encoder_attention_heads;
  const int head_dim = audio_.d_model / head_count;
  const mx::array normed = LayerNorm(x, prefix + ".self_attn_layer_norm");
  const auto project = [&](const std::string &name) {
    return mx::transpose(
        mx::reshape(Linear(normed, prefix + ".self_attn." + name),
                    {batch, length, head_count, head_dim}),
        {0, 2, 1, 3});
  };
  const mx::array attended = mx::fast::scaled_dot_product_attention(
      project("q_proj"), project("k_proj"), project("v_proj"),
      static_cast<float>(std::pow(static_cast<double>(head_dim), -0.5)));
  mx::array hidden =
      mx::add(x, Linear(mx::reshape(mx::transpose(attended, {0, 2, 1, 3}),
                                    {batch, length, width}),
                        prefix + ".self_attn.out_proj"));
  return mx::add(
      hidden,
      Linear(Gelu(Linear(LayerNorm(hidden, prefix + ".final_layer_norm"),
                         prefix + ".fc1")),
             prefix + ".fc2"));
}

mx::array Qwen3ASR::EncodeAudio(const mx::array &mel,
                                AudioLayout layout) const {
  const int chunk_frame_count = audio_.n_window * 2;
  const int frame_count = mel.shape(-1);
  const int mel_bins = mel.shape(0);
  std::vector<int> chunk_starts;
  std::vector<int> chunk_lengths;
  for (int start = 0; start < frame_count; start += chunk_frame_count) {
    chunk_starts.push_back(start);
    chunk_lengths.push_back(std::min(chunk_frame_count, frame_count - start));
  }
  const int longest_chunk =
      *std::max_element(chunk_lengths.begin(), chunk_lengths.end());
  std::vector<mx::array> padded_chunks;
  for (size_t i = 0; i < chunk_starts.size(); ++i) {
    const mx::array chunk =
        mx::slice(mel, {0, chunk_starts[i]},
                  {mel_bins, chunk_starts[i] + chunk_lengths[i]});
    padded_chunks.push_back(
        mx::pad(chunk, {{0, 0}, {0, longest_chunk - chunk_lengths[i]}}));
  }
  mx::array x = mx::expand_dims(mx::stack(padded_chunks), -1);
  x = Gelu(Conv2d(x, "audio_tower.conv2d1"));
  x = Gelu(Conv2d(x, "audio_tower.conv2d2"));
  x = Gelu(Conv2d(x, "audio_tower.conv2d3"));
  const int chunk_count = x.shape(0);
  const int frequency_bins = x.shape(1);
  const int conv_frames = x.shape(2);
  const int channels = x.shape(3);
  x = Linear(mx::reshape(mx::transpose(x, {0, 2, 3, 1}),
                         {chunk_count, conv_frames, channels * frequency_bins}),
             "audio_tower.conv_out");
  x = mx::add(
      x, mx::expand_dims(SinusoidalPositions(conv_frames, audio_.d_model), 0));

  std::vector<int> credited_lengths;
  for (const int length : chunk_lengths) {
    credited_lengths.push_back(layout == AudioLayout::kReference
                                   ? ConvOutputFrames(length)
                                   : SwiftTokenCount(length));
  }
  // Note (Jiaxin Deng): the Swift port credits each chunk by its own length
  // formula and keeps that many rows of the padded conv output.
  std::vector<mx::array> kept_rows;
  for (int i = 0; i < chunk_count; ++i) {
    const int kept = std::min(credited_lengths[i], conv_frames);
    kept_rows.push_back(
        mx::reshape(mx::slice(x, {i, 0, 0}, {i + 1, kept, x.shape(2)}),
                    {kept, x.shape(2)}));
  }
  mx::array hidden_states = mx::concatenate(kept_rows, 0);

  // Note (Jiaxin Deng): attention stays within windows of chunks, equal-length
  // windows batched together, as in the Swift encoder.
  const int chunks_per_window =
      std::max(1, audio_.n_window_infer / chunk_frame_count);
  std::vector<int> window_lengths;
  for (int start = 0; start < chunk_count; start += chunks_per_window) {
    int total = 0;
    for (int i = start; i < std::min(start + chunks_per_window, chunk_count);
         ++i) {
      total += credited_lengths[i];
    }
    window_lengths.push_back(total);
  }
  const int token_count = hidden_states.shape(0);
  std::vector<std::pair<int, int>> window_bounds;
  int window_start = 0;
  for (const int window_length : window_lengths) {
    const int window_end = std::min(window_start + window_length, token_count);
    if (window_end > window_start) {
      window_bounds.emplace_back(window_start, window_end);
    } else {
    }
    window_start = window_end;
  }
  if (window_start < token_count) {
    window_bounds.emplace_back(window_start, token_count);
  } else {
  }
  const int width = hidden_states.shape(1);
  std::set<int> lengths;
  for (const auto &[start, end] : window_bounds)
    lengths.insert(end - start);
  std::map<size_t, mx::array> encoded_windows;
  for (const int length : lengths) {
    std::vector<size_t> same_length;
    std::vector<mx::array> rows;
    for (size_t index = 0; index < window_bounds.size(); ++index) {
      const auto &[start, end] = window_bounds[index];
      if (end - start == length) {
        same_length.push_back(index);
        rows.push_back(mx::slice(hidden_states, {start, 0}, {end, width}));
      } else {
      }
    }
    mx::array batch = mx::stack(rows);
    for (int layer = 0; layer < audio_.encoder_layers; ++layer) {
      batch = AudioEncoderLayer(batch, layer);
    }
    for (size_t row = 0; row < same_length.size(); ++row) {
      encoded_windows.insert_or_assign(
          same_length[row],
          mx::reshape(mx::slice(batch, {static_cast<int>(row), 0, 0},
                                {static_cast<int>(row) + 1, length, width}),
                      {length, width}));
    }
  }
  std::vector<mx::array> ordered;
  for (size_t index = 0; index < window_bounds.size(); ++index) {
    ordered.push_back(encoded_windows.at(index));
  }
  hidden_states = LayerNorm(mx::concatenate(ordered, 0), "audio_tower.ln_post");
  return Linear(Gelu(Linear(hidden_states, "audio_tower.proj1")),
                "audio_tower.proj2");
}

mx::array Qwen3ASR::EmbedTokens(const mx::array &ids) const {
  const std::string prefix = "model.embed_tokens";
  if (Has(prefix + ".scales")) {
    const std::optional<mx::array> biases =
        Has(prefix + ".biases") ? std::optional<mx::array>(mx::take(
                                      Weight(prefix + ".biases"), ids, 0))
                                : std::nullopt;
    return mx::dequantize(mx::take(Weight(prefix + ".weight"), ids, 0),
                          mx::take(Weight(prefix + ".scales"), ids, 0), biases,
                          quantization_.group_size, quantization_.bits,
                          quantization_.mode);
  } else {
    return mx::take(Weight(prefix + ".weight"), ids, 0);
  }
}

mx::array Qwen3ASR::TextDecoderLayer(const mx::array &x, int layer,
                                     KVCache &cache) const {
  const std::string prefix = "model.layers." + std::to_string(layer);
  const int batch = x.shape(0);
  const int length = x.shape(1);
  const int head_count = text_.num_attention_heads;
  const int kv_head_count = text_.num_key_value_heads;
  const int head_dim = text_.head_dim;
  const mx::array normed = RmsNorm(x, prefix + ".input_layernorm");
  mx::array queries =
      RmsNorm(mx::reshape(Linear(normed, prefix + ".self_attn.q_proj"),
                          {batch, length, head_count, head_dim}),
              prefix + ".self_attn.q_norm");
  mx::array keys =
      RmsNorm(mx::reshape(Linear(normed, prefix + ".self_attn.k_proj"),
                          {batch, length, kv_head_count, head_dim}),
              prefix + ".self_attn.k_norm");
  const mx::array values =
      mx::reshape(Linear(normed, prefix + ".self_attn.v_proj"),
                  {batch, length, kv_head_count, head_dim});
  queries = mx::fast::rope(mx::transpose(queries, {0, 2, 1, 3}), head_dim,
                           false, text_.rope_theta, 1.0f, cache.offset());
  keys = mx::fast::rope(mx::transpose(keys, {0, 2, 1, 3}), head_dim, false,
                        text_.rope_theta, 1.0f, cache.offset());
  auto [cached_keys, cached_values] =
      cache.UpdateAndFetch(keys, mx::transpose(values, {0, 2, 1, 3}));
  const mx::array attended = mx::fast::scaled_dot_product_attention(
      queries, cached_keys, cached_values,
      static_cast<float>(std::pow(static_cast<double>(head_dim), -0.5)),
      length > 1 ? "causal" : "");
  const mx::array hidden =
      mx::add(x, Linear(mx::reshape(mx::transpose(attended, {0, 2, 1, 3}),
                                    {batch, length, -1}),
                        prefix + ".self_attn.o_proj"));
  const mx::array mlp_input =
      RmsNorm(hidden, prefix + ".post_attention_layernorm");
  return mx::add(
      hidden,
      Linear(mx::multiply(Silu(Linear(mlp_input, prefix + ".mlp.gate_proj")),
                          Linear(mlp_input, prefix + ".mlp.up_proj")),
             prefix + ".mlp.down_proj"));
}

mx::array Qwen3ASR::Decode(const mx::array &embeddings,
                           std::vector<KVCache> &caches) const {
  mx::array hidden = embeddings;
  for (int layer = 0; layer < text_.num_hidden_layers; ++layer) {
    hidden = TextDecoderLayer(hidden, layer, caches[layer]);
  }
  const int length = hidden.shape(1);
  const mx::array last =
      RmsNorm(mx::slice(hidden, {0, length - 1, 0},
                        {hidden.shape(0), length, hidden.shape(2)}),
              "model.norm");
  // Note (Jiaxin Deng): tied output projection; the embedding table is the
  // linear layer.
  const std::string prefix = "model.embed_tokens";
  mx::array logits = [&]() {
    if (Has(prefix + ".scales")) {
      const std::optional<mx::array> biases =
          Has(prefix + ".biases")
              ? std::optional<mx::array>(Weight(prefix + ".biases"))
              : std::nullopt;
      return mx::quantized_matmul(last, Weight(prefix + ".weight"),
                                  Weight(prefix + ".scales"), biases, true,
                                  quantization_.group_size, quantization_.bits,
                                  quantization_.mode);
    } else {
      return mx::matmul(last, mx::transpose(Weight(prefix + ".weight")));
    }
  }();
  return mx::reshape(logits, {logits.shape(-1)});
}

std::vector<KVCache> Qwen3ASR::NewCaches() const {
  return std::vector<KVCache>(text_.num_hidden_layers);
}

} // namespace qwen3_asr
