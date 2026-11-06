// SPDX-License-Identifier: Apache-2.0
#include "qwen3_asr/qwen3_asr_model.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace qwen3_asr {

namespace mx = mlx::core;

namespace {

constexpr float kLayerNormEps = 1e-5f;
// Chunks, a second of audio each, that the convolutions take at a time.
constexpr int kConvolutionBlockChunks = 64;
// Note (Yucheng Hu): clips longer than one block take smaller blocks, since
// two blocks' activations are alive at once.
constexpr int kLongAudioConvolutionBlockChunks = 8;
// Note (Yucheng Hu): these minimums keep conv_out and the attention fc2 on
// the ungrouped encoder's matmul kernels, since split-K changes output bits.
constexpr int kMinimumConvOutChunksPerGroup = 197;
constexpr int kMinimumWindowsPerGroup = 24;
// Names are relative to the layer.
constexpr std::array<const char *, 6> kEncoderLinears = {"self_attn.q_proj",
                                                         "self_attn.k_proj",
                                                         "self_attn.v_proj",
                                                         "self_attn.out_proj",
                                                         "fc1",
                                                         "fc2"};

struct WindowBatch {
  std::vector<size_t> window_indices;
  mx::array batch;
};

std::vector<std::pair<int, int>> GroupBounds(int member_count,
                                             int minimum_group_size) {
  const int group_count = std::max(1, member_count / minimum_group_size);
  std::vector<std::pair<int, int>> bounds;
  for (int group = 0; group < group_count; ++group) {
    bounds.emplace_back(group * member_count / group_count,
                        (group + 1) * member_count / group_count);
  }
  return bounds;
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

Qwen3ASR::Qwen3ASR(const std::filesystem::path &model_directory)
    : Qwen3ASR(omni_common::ReadJson(model_directory / "config.json"),
               model_directory) {}

Qwen3ASR::Qwen3ASR(const nlohmann::json &config,
                   const std::filesystem::path &model_directory)
    : checkpoint_(omni_common::LoadSafetensors(model_directory), config),
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

mx::array Qwen3ASR::AudioEncoderLayer(const mx::array &x, int layer,
                                      const CastWeights &cast_weights) const {
  const std::string prefix = "audio_tower.layers." + std::to_string(layer);
  const int batch = x.shape(0);
  const int length = x.shape(1);
  const int width = x.shape(2);
  const int head_count = audio_.encoder_attention_heads;
  const int head_dim = audio_.d_model / head_count;
  const auto linear = [&](const mx::array &input, const std::string &name) {
    const auto cast_weight = cast_weights.find(name);
    if (cast_weight != cast_weights.end()) {
      return checkpoint_.Linear(input, prefix + "." + name,
                                cast_weight->second);
    } else {
      return checkpoint_.Linear(input, prefix + "." + name);
    }
  };
  const mx::array normed =
      checkpoint_.LayerNorm(x, prefix + ".self_attn_layer_norm", kLayerNormEps);
  const auto project = [&](const std::string &name) {
    return mx::transpose(mx::reshape(linear(normed, "self_attn." + name),
                                     {batch, length, head_count, head_dim}),
                         {0, 2, 1, 3});
  };
  const mx::array attended = mx::fast::scaled_dot_product_attention(
      project("q_proj"), project("k_proj"), project("v_proj"),
      static_cast<float>(std::pow(static_cast<double>(head_dim), -0.5)));
  mx::array hidden =
      mx::add(x, linear(mx::reshape(mx::transpose(attended, {0, 2, 1, 3}),
                                    {batch, length, width}),
                        "self_attn.out_proj"));
  return mx::add(
      hidden,
      linear(omni_common::Gelu(linear(
                 checkpoint_.LayerNorm(hidden, prefix + ".final_layer_norm",
                                       kLayerNormEps),
                 "fc1")),
             "fc2"));
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
  const int chunk_count = static_cast<int>(chunk_starts.size());
  const bool is_long_audio = chunk_count > kConvolutionBlockChunks;
  const int block_chunks = is_long_audio ? kLongAudioConvolutionBlockChunks
                                         : kConvolutionBlockChunks;
  std::vector<int> credited_lengths;
  for (const int length : chunk_lengths) {
    credited_lengths.push_back(layout == AudioLayout::kReference
                                   ? ConvOutputFrames(length)
                                   : SwiftTokenCount(length));
  }
  if (is_long_audio) {
    // Note (Yucheng Hu): the STFT temporaries are freed before the first
    // convolution block runs, instead of being held alongside it.
    mx::eval(mel);
  } else {
  }
  // Note (Yucheng Hu): MLX casts the convolution weights and biases to the
  // activation dtype in every block, so they are cast once per clip.
  std::vector<std::pair<mx::array, mx::array>> convolutions;
  mx::Dtype input_dtype = mel.dtype();
  for (const std::string prefix :
       {"audio_tower.conv2d1", "audio_tower.conv2d2", "audio_tower.conv2d3"}) {
    const mx::array &weight = checkpoint_.Weight(prefix + ".weight");
    const mx::array &bias = checkpoint_.Weight(prefix + ".bias");
    const mx::Dtype weight_dtype =
        mx::promote_types(input_dtype, weight.dtype());
    const mx::Dtype output_dtype =
        mx::promote_types(weight_dtype, bias.dtype());
    convolutions.emplace_back(mx::astype(weight, weight_dtype),
                              mx::astype(bias, output_dtype));
    input_dtype = output_dtype;
  }
  // The convolutions take one block of chunks at a time, so a long clip holds
  // two blocks' activations at most: each block but the last starts on the GPU
  // once built, after which the block before it is waited for. conv_out takes
  // groups of at least kMinimumConvOutChunksPerGroup chunks, each started once
  // built, so the blocks of a finished group are freed.
  const std::vector<std::pair<int, int>> conv_out_groups =
      is_long_audio ? GroupBounds(chunk_count, kMinimumConvOutChunksPerGroup)
                    : std::vector<std::pair<int, int>>{{0, chunk_count}};
  std::vector<mx::array> kept_row_groups;
  for (const auto &[group_start, group_end] : conv_out_groups) {
    std::vector<mx::array> conv_blocks;
    for (int block_start = group_start; block_start < group_end;
         block_start += block_chunks) {
      const int block_end = std::min(block_start + block_chunks, group_end);
      std::vector<mx::array> padded_chunks;
      for (int i = block_start; i < block_end; ++i) {
        const mx::array chunk =
            mx::slice(mel, {0, chunk_starts[i]},
                      {mel_bins, chunk_starts[i] + chunk_lengths[i]});
        padded_chunks.push_back(
            mx::pad(chunk, {{0, 0}, {0, longest_chunk - chunk_lengths[i]}}));
      }
      mx::array x = mx::expand_dims(mx::stack(padded_chunks), -1);
      for (const auto &[weight, bias] : convolutions) {
        x = omni_common::Gelu(
            mx::add(mx::conv2d(x, weight, {2, 2}, {1, 1}), bias));
      }
      const int block_chunk_count = x.shape(0);
      const int frequency_bins = x.shape(1);
      const int conv_frames = x.shape(2);
      const int channels = x.shape(3);
      conv_blocks.push_back(mx::reshape(
          mx::transpose(x, {0, 2, 3, 1}),
          {block_chunk_count, conv_frames, channels * frequency_bins}));
      if (block_end < chunk_count) {
        mx::async_eval(conv_blocks.back());
        if (conv_blocks.size() > 1) {
          mx::eval(conv_blocks[conv_blocks.size() - 2]);
        } else {
        }
      } else {
      }
    }
    mx::array x = checkpoint_.Linear(conv_blocks.size() == 1
                                         ? conv_blocks[0]
                                         : mx::concatenate(conv_blocks, 0),
                                     "audio_tower.conv_out");
    const int conv_frames = x.shape(1);
    x = mx::add(x, mx::expand_dims(
                       SinusoidalPositions(conv_frames, audio_.d_model), 0));

    // Note (Jiaxin Deng): the Swift port credits each chunk by its own length
    // formula and keeps that many rows of the padded conv output.
    std::vector<mx::array> kept_rows;
    for (int i = group_start; i < group_end; ++i) {
      const int kept = std::min(credited_lengths[i], conv_frames);
      kept_rows.push_back(
          mx::reshape(mx::slice(x, {i - group_start, 0, 0},
                                {i - group_start + 1, kept, x.shape(2)}),
                      {kept, x.shape(2)}));
    }
    kept_row_groups.push_back(mx::concatenate(kept_rows, 0));
    if (group_end < chunk_count) {
      mx::async_eval(kept_row_groups.back());
    } else {
    }
  }
  // Note (Yucheng Hu): the casts are freed with the convolution tail instead
  // of being held through the attention.
  convolutions.clear();
  mx::array hidden_states = kept_row_groups.size() == 1
                                ? kept_row_groups[0]
                                : mx::concatenate(kept_row_groups, 0);
  if (is_long_audio) {
    // Note (Yucheng Hu): the convolution temporaries are freed before the
    // first attention layer runs, instead of being held alongside it.
    mx::eval(hidden_states);
  } else {
  }

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
  // Note (Yucheng Hu): step i holds the i-th window group of every length, so
  // the short last window runs beside the first full group instead of alone.
  std::vector<std::vector<WindowBatch>> steps;
  for (const int length : lengths) {
    std::vector<size_t> same_length;
    for (size_t index = 0; index < window_bounds.size(); ++index) {
      const auto &[start, end] = window_bounds[index];
      if (end - start == length) {
        same_length.push_back(index);
      } else {
      }
    }
    const int same_length_count = static_cast<int>(same_length.size());
    const std::vector<std::pair<int, int>> window_groups =
        is_long_audio
            ? GroupBounds(same_length_count, kMinimumWindowsPerGroup)
            : std::vector<std::pair<int, int>>{{0, same_length_count}};
    steps.resize(std::max(steps.size(), window_groups.size()));
    for (size_t group = 0; group < window_groups.size(); ++group) {
      const auto &[group_start, group_end] = window_groups[group];
      std::vector<mx::array> rows;
      for (int i = group_start; i < group_end; ++i) {
        const auto &[start, end] = window_bounds[same_length[i]];
        rows.push_back(mx::slice(hidden_states, {start, 0}, {end, width}));
      }
      steps[group].push_back(
          {std::vector<size_t>(same_length.begin() + group_start,
                               same_length.begin() + group_end),
           mx::stack(rows)});
    }
  }
  for (std::vector<WindowBatch> &step : steps) {
    for (int layer = 0; layer < audio_.encoder_layers; ++layer) {
      // Note (Yucheng Hu): the step's batches share each weight's cast to the
      // residual stream's dtype, which is the cast their matmuls would insert.
      CastWeights cast_weights;
      for (const char *name : kEncoderLinears) {
        const std::string linear_prefix =
            "audio_tower.layers." + std::to_string(layer) + "." + name;
        if (checkpoint_.Has(linear_prefix + ".scales")) {
          // A quantized linear has no weight to cast.
        } else {
          const mx::array transposed_weight =
              mx::transpose(checkpoint_.Weight(linear_prefix + ".weight"));
          cast_weights.emplace(
              name, mx::astype(transposed_weight,
                               mx::promote_types(hidden_states.dtype(),
                                                 transposed_weight.dtype())));
        }
      }
      std::vector<mx::array> layer_inputs;
      std::vector<mx::array> layer_outputs;
      for (WindowBatch &window_batch : step) {
        layer_inputs.push_back(window_batch.batch);
        window_batch.batch =
            AudioEncoderLayer(window_batch.batch, layer, cast_weights);
        layer_outputs.push_back(window_batch.batch);
      }
      if (is_long_audio) {
        // Note (Yucheng Hu): MLX frees temporaries per buffer, so a long clip
        // evaluates one layer of a step at a time.
        mx::async_eval(layer_outputs);
        mx::eval(layer_inputs);
      } else {
      }
    }
    for (const WindowBatch &window_batch : step) {
      const int length = window_batch.batch.shape(1);
      for (int row = 0;
           row < static_cast<int>(window_batch.window_indices.size()); ++row) {
        encoded_windows.insert_or_assign(
            window_batch.window_indices[row],
            mx::reshape(mx::slice(window_batch.batch, {row, 0, 0},
                                  {row + 1, length, width}),
                        {length, width}));
      }
    }
  }
  std::vector<mx::array> ordered;
  for (size_t index = 0; index < window_bounds.size(); ++index) {
    ordered.push_back(encoded_windows.at(index));
  }
  hidden_states = checkpoint_.LayerNorm(mx::concatenate(ordered, 0),
                                        "audio_tower.ln_post", kLayerNormEps);
  return checkpoint_.Linear(
      omni_common::Gelu(checkpoint_.Linear(hidden_states, "audio_tower.proj1")),
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
