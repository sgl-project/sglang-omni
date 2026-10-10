// SPDX-License-Identifier: Apache-2.0
#include "model.h"

#include <algorithm>
#include <cmath>
#include <map>
#include <set>
#include <stdexcept>
#include <vector>

namespace qwen3_asr {

namespace mx = mlx::core;

namespace {

constexpr float kLayerNormEps = 1e-5f;

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

Qwen3ASR::Qwen3ASR(const std::filesystem::path &model_directory)
    : Qwen3ASR(ReadJson(model_directory / "config.json"), model_directory) {}

Qwen3ASR::Qwen3ASR(const nlohmann::json &config,
                   const std::filesystem::path &model_directory)
    : checkpoint_(LoadSafetensors(model_directory), config),
      decoder_(checkpoint_, "model",
               Qwen3DecoderConfig::FromJson(
                   config.at("thinker_config").at("text_config"))) {
  const nlohmann::json &audio = config.at("thinker_config").at("audio_config");
  audio_ = {audio.at("num_mel_bins").get<int>(),
            audio.at("encoder_layers").get<int>(),
            audio.at("encoder_attention_heads").get<int>(),
            audio.at("d_model").get<int>(),
            audio.at("n_window").get<int>(),
            audio.at("n_window_infer").get<int>()};
}

mx::array Qwen3ASR::Conv2d(const mx::array &x,
                           const std::string &prefix) const {
  return mx::add(
      mx::conv2d(x, checkpoint_.Weight(prefix + ".weight"), {2, 2}, {1, 1}),
      checkpoint_.Weight(prefix + ".bias"));
}

mx::array Qwen3ASR::AudioEncoderLayer(const mx::array &x, int layer) const {
  const std::string prefix = "audio_tower.layers." + std::to_string(layer);
  const int batch = x.shape(0);
  const int length = x.shape(1);
  const int width = x.shape(2);
  const int head_count = audio_.encoder_attention_heads;
  const int head_dim = audio_.d_model / head_count;
  const mx::array normed =
      checkpoint_.LayerNorm(x, prefix + ".self_attn_layer_norm", kLayerNormEps);
  const auto project = [&](const std::string &name) {
    return mx::transpose(
        mx::reshape(checkpoint_.Linear(normed, prefix + ".self_attn." + name),
                    {batch, length, head_count, head_dim}),
        {0, 2, 1, 3});
  };
  const mx::array attended = mx::fast::scaled_dot_product_attention(
      project("q_proj"), project("k_proj"), project("v_proj"),
      static_cast<float>(std::pow(static_cast<double>(head_dim), -0.5)));
  mx::array hidden = mx::add(
      x, checkpoint_.Linear(mx::reshape(mx::transpose(attended, {0, 2, 1, 3}),
                                        {batch, length, width}),
                            prefix + ".self_attn.out_proj"));
  return mx::add(
      hidden, checkpoint_.Linear(
                  Gelu(checkpoint_.Linear(
                      checkpoint_.LayerNorm(
                          hidden, prefix + ".final_layer_norm", kLayerNormEps),
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
  x = checkpoint_.Linear(
      mx::reshape(mx::transpose(x, {0, 2, 3, 1}),
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
  hidden_states = checkpoint_.LayerNorm(mx::concatenate(ordered, 0),
                                        "audio_tower.ln_post", kLayerNormEps);
  return checkpoint_.Linear(
      Gelu(checkpoint_.Linear(hidden_states, "audio_tower.proj1")),
      "audio_tower.proj2");
}

mx::array Qwen3ASR::EmbedTokens(const mx::array &ids) const {
  return decoder_.EmbedTokens(ids);
}

mx::array Qwen3ASR::Decode(const mx::array &embeddings,
                           std::vector<KVCache> &caches) const {
  return decoder_.LastLogits(decoder_.Forward(embeddings, caches));
}

std::vector<KVCache> Qwen3ASR::NewCaches() const {
  return decoder_.NewCaches();
}

} // namespace qwen3_asr
