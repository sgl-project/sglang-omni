// SPDX-License-Identifier: Apache-2.0
#include "qwen3_asr/qwen3_asr_model.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace qwen3_asr {

namespace mx = mlx::core;

namespace {

constexpr float kLayerNormEps = 1e-5f;
// The audio encoder's weights and activations.
const mx::Dtype kEncoderDtype = mx::float16;

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

// Each encoder layer's q, k and v projections stacked as one qkv_proj, whose
// rows are q's, then k's, then v's.
void StackAttentionProjections(omni_common::WeightMap &weights,
                               int encoder_layers) {
  for (int layer = 0; layer < encoder_layers; ++layer) {
    const std::string prefix =
        "audio_tower.layers." + std::to_string(layer) + ".self_attn.";
    for (const std::string part : {"weight", "scales", "biases", "bias"}) {
      if (weights.contains(prefix + "q_proj." + part)) {
        std::vector<mx::array> projections;
        for (const std::string projection : {"q_proj.", "k_proj.", "v_proj."}) {
          const auto found = weights.find(prefix + projection + part);
          projections.push_back(found->second);
          weights.erase(found);
        }
        weights.insert_or_assign(prefix + "qkv_proj." + part,
                                 mx::concatenate(projections, 0));
      } else {
      }
    }
  }
}

} // namespace

Qwen3ASR::Qwen3ASR(const std::filesystem::path &model_directory)
    : Qwen3ASR(omni_common::ReadJson(model_directory / "config.json"),
               model_directory) {}

Qwen3ASR::Qwen3ASR(const nlohmann::json &config,
                   const std::filesystem::path &model_directory)
    : checkpoint_(
          [&config, &model_directory] {
            omni_common::WeightMap weights =
                omni_common::LoadSafetensors(model_directory);
            for (auto &[name, weight] : weights) {
              if (name.starts_with("audio_tower.") &&
                  mx::issubdtype(weight.dtype(), mx::floating)) {
                weight = mx::astype(weight, kEncoderDtype);
              } else {
              }
            }
            StackAttentionProjections(weights, config.at("thinker_config")
                                                   .at("audio_config")
                                                   .at("encoder_layers")
                                                   .get<int>());
            return weights;
          }(),
          config),
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

mx::array
Qwen3ASR::AudioEncoderLayer(const mx::array &x, int layer,
                            const std::vector<WindowRun> &runs) const {
  const std::string prefix = "audio_tower.layers." + std::to_string(layer);
  const int width = x.shape(1);
  const int head_count = audio_.encoder_attention_heads;
  const int head_dim = audio_.d_model / head_count;
  const mx::array qkv = checkpoint_.Linear(
      checkpoint_.LayerNorm(x, prefix + ".self_attn_layer_norm", kLayerNormEps),
      prefix + ".self_attn.qkv_proj");
  std::vector<mx::array> attended;
  for (const WindowRun &run : runs) {
    const int end = run.start + run.windows * run.length;
    const auto heads = [&](int part) {
      return mx::transpose(
          mx::reshape(mx::slice(qkv, {run.start, part * width},
                                {end, (part + 1) * width}),
                      {run.windows, run.length, head_count, head_dim}),
          {0, 2, 1, 3});
    };
    const mx::array windows = mx::fast::scaled_dot_product_attention(
        heads(0), heads(1), heads(2),
        static_cast<float>(std::pow(static_cast<double>(head_dim), -0.5)));
    attended.push_back(mx::reshape(mx::transpose(windows, {0, 2, 1, 3}),
                                   {run.windows * run.length, width}));
  }
  mx::array hidden = mx::add(
      x, checkpoint_.Linear(attended.size() == 1 ? attended[0]
                                                 : mx::concatenate(attended, 0),
                            prefix + ".self_attn.out_proj"));
  return mx::add(
      hidden, checkpoint_.Linear(
                  omni_common::Gelu(checkpoint_.Linear(
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
  mx::array x =
      mx::astype(mx::expand_dims(mx::stack(padded_chunks), -1), kEncoderDtype);
  for (const std::string convolution : {"conv2d1", "conv2d2", "conv2d3"}) {
    const std::string prefix = "audio_tower." + convolution;
    x = ConvolutionGelu(x, checkpoint_.Weight(prefix + ".weight"),
                        checkpoint_.Weight(prefix + ".bias"));
  }
  const int chunk_count = x.shape(0);
  const int frequency_bins = x.shape(1);
  const int conv_frames = x.shape(2);
  const int channels = x.shape(3);
  x = checkpoint_.Linear(
      mx::reshape(mx::transpose(x, {0, 2, 3, 1}),
                  {chunk_count, conv_frames, channels * frequency_bins}),
      "audio_tower.conv_out");
  x = mx::add(x, mx::expand_dims(mx::astype(SinusoidalPositions(conv_frames,
                                                                audio_.d_model),
                                            kEncoderDtype),
                                 0));

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
  // Every layer but attention takes all tokens at once; attention takes each
  // run of consecutive equal-length windows as one batch.
  std::vector<WindowRun> runs;
  int window_start = 0;
  const auto add_window = [&runs](int start, int length) {
    if (!runs.empty() && runs.back().length == length) {
      ++runs.back().windows;
    } else {
      runs.push_back({start, 1, length});
    }
  };
  for (const int window_length : window_lengths) {
    const int window_end = std::min(window_start + window_length, token_count);
    if (window_end > window_start) {
      add_window(window_start, window_end - window_start);
    } else {
    }
    window_start = window_end;
  }
  if (window_start < token_count) {
    add_window(window_start, token_count - window_start);
  } else {
  }
  for (int layer = 0; layer < audio_.encoder_layers; ++layer) {
    hidden_states = AudioEncoderLayer(hidden_states, layer, runs);
  }
  hidden_states = checkpoint_.LayerNorm(hidden_states, "audio_tower.ln_post",
                                        kLayerNormEps);
  return checkpoint_.Linear(
      omni_common::Gelu(checkpoint_.Linear(hidden_states, "audio_tower.proj1")),
      "audio_tower.proj2");
}

mx::array Qwen3ASR::EmbedTokens(const mx::array &ids) const {
  return decoder_.EmbedTokens(ids);
}

mx::array Qwen3ASR::Decode(const mx::array &embeddings,
                           std::vector<Qwen3ASRKVCache> &caches,
                           int rows) const {
  return decoder_.LastLogits(decoder_.Forward(embeddings, caches), rows);
}

std::vector<Qwen3ASRKVCache> Qwen3ASR::NewCaches() const {
  return decoder_.NewCaches();
}

} // namespace qwen3_asr
