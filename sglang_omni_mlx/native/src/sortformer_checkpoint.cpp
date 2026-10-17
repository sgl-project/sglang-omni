// SPDX-License-Identifier: Apache-2.0
#include "sortformer_checkpoint.h"

#include <algorithm>
#include <fstream>
#include <set>
#include <stdexcept>

#include "nlohmann/json.hpp"

namespace sortformer {

namespace mx = mlx::core;

namespace {

nlohmann::json Section(const nlohmann::json &config, const char *key) {
  // Note (Jiaxin Deng): Swift decodes a missing section from the top level.
  return config.contains(key) ? config.at(key) : config;
}

Config ParseConfig(const nlohmann::json &json) {
  Config config;
  config.num_speakers = json.value("num_speakers", 4);
  const nlohmann::json fc = Section(json, "fc_encoder_config");
  FastConformerConfig &f = config.fc_encoder;
  f.hidden_size = fc.value("hidden_size", 512);
  f.num_hidden_layers = fc.value("num_hidden_layers", 18);
  f.num_attention_heads = fc.value("num_attention_heads", 8);
  f.intermediate_size = fc.value("intermediate_size", 2048);
  f.num_mel_bins = fc.value("num_mel_bins", 80);
  f.conv_kernel_size = fc.value("conv_kernel_size", 9);
  f.subsampling_factor = fc.value("subsampling_factor", 8);
  f.subsampling_conv_channels = fc.value("subsampling_conv_channels", 256);
  f.subsampling_conv_kernel_size = fc.value("subsampling_conv_kernel_size", 3);
  f.subsampling_conv_stride = fc.value("subsampling_conv_stride", 2);
  f.attention_bias = fc.value("attention_bias", true);
  f.scale_input = fc.value("scale_input", true);
  const nlohmann::json tf = Section(json, "tf_encoder_config");
  TransformerConfig &t = config.tf_encoder;
  t.d_model = tf.value("d_model", 192);
  t.encoder_layers = tf.value("encoder_layers", 18);
  t.encoder_attention_heads = tf.value("encoder_attention_heads", 8);
  t.encoder_ffn_dim = tf.value("encoder_ffn_dim", 768);
  t.layer_norm_eps = tf.value("layer_norm_eps", 1e-5f);
  t.max_source_positions = tf.value("max_source_positions", 1500);
  t.k_proj_bias = tf.value("k_proj_bias", false);
  const nlohmann::json modules = Section(json, "modules_config");
  ModulesConfig &m = config.modules;
  m.num_speakers = modules.value("num_speakers", 4);
  m.fc_d_model = modules.value("fc_d_model", 512);
  m.tf_d_model = modules.value("tf_d_model", 192);
  m.subsampling_factor = modules.value("subsampling_factor", 8);
  m.chunk_len = modules.value("chunk_len", 188);
  m.fifo_len = modules.value("fifo_len", 0);
  m.spkcache_len = modules.value("spkcache_len", 188);
  m.spkcache_update_period = modules.value("spkcache_update_period", 188);
  m.chunk_left_context = modules.value("chunk_left_context", 1);
  m.chunk_right_context = modules.value("chunk_right_context", 1);
  m.spkcache_sil_frames_per_spk =
      modules.value("spkcache_sil_frames_per_spk", 5);
  m.pred_score_threshold = modules.value("pred_score_threshold", 1e-6f);
  m.max_index = modules.value("max_index", 10000);
  m.scores_boost_latest = modules.value("scores_boost_latest", 0.5f);
  m.sil_threshold = modules.value("sil_threshold", 0.1f);
  m.strong_boost_rate = modules.value("strong_boost_rate", 0.3f);
  m.weak_boost_rate = modules.value("weak_boost_rate", 0.7f);
  m.min_pos_scores_rate = modules.value("min_pos_scores_rate", 0.5f);
  m.use_aosc = modules.value("use_aosc", false);
  const nlohmann::json processor = Section(json, "processor_config");
  ProcessorConfig &p = config.processor;
  p.feature_size = processor.value("feature_size", 80);
  p.sampling_rate = processor.value("sampling_rate", 16000);
  p.hop_length = processor.value("hop_length", 160);
  p.fft_size = processor.value("n_fft", 512);
  p.window_length = processor.value("win_length", 400);
  p.preemphasis = processor.value("preemphasis", 0.97f);
  return config;
}

} // namespace

Config LoadConfig(const std::filesystem::path &model_directory) {
  std::ifstream config_stream(model_directory / "config.json");
  if (!config_stream) {
    throw std::runtime_error("cannot read config.json in " +
                             model_directory.string());
  } else {
  }
  const Config config = ParseConfig(nlohmann::json::parse(config_stream));
  // Note (Jiaxin Deng): only the v2.1 (AOSC) path is ported; older checkpoints
  // peak-normalize and standardize their features.
  if (!config.modules.use_aosc) {
    throw std::runtime_error(
        "only Sortformer checkpoints with use_aosc are supported");
  } else {
  }
  return config;
}

namespace {

std::set<std::string> ExpectedKeys(const Config &config) {
  std::set<std::string> keys;
  const auto add_linear = [&](const std::string &prefix, bool bias) {
    keys.insert(prefix + ".weight");
    if (bias) {
      keys.insert(prefix + ".bias");
    } else {
    }
  };
  for (const char *layer :
       {"layers_0", "layers_2", "layers_3", "layers_5", "layers_6", "linear"}) {
    add_linear(std::string("fc_encoder.subsampling.") + layer, true);
  }
  for (int layer = 0; layer < config.fc_encoder.num_hidden_layers; ++layer) {
    const std::string prefix = "fc_encoder.layers." + std::to_string(layer);
    for (const char *norm : {"norm_feed_forward1", "norm_self_att", "norm_conv",
                             "norm_feed_forward2", "norm_out"}) {
      add_linear(prefix + "." + norm, true);
    }
    for (const char *feed_forward : {".feed_forward1", ".feed_forward2"}) {
      add_linear(prefix + feed_forward + ".linear1", true);
      add_linear(prefix + feed_forward + ".linear2", true);
    }
    for (const char *projection : {"q_proj", "k_proj", "v_proj", "o_proj"}) {
      add_linear(prefix + ".self_attn." + projection,
                 config.fc_encoder.attention_bias);
    }
    add_linear(prefix + ".self_attn.relative_k_proj", false);
    keys.insert(prefix + ".self_attn.bias_u");
    keys.insert(prefix + ".self_attn.bias_v");
    add_linear(prefix + ".conv.pointwise_conv1", true);
    add_linear(prefix + ".conv.depthwise_conv", true);
    add_linear(prefix + ".conv.pointwise_conv2", true);
    add_linear(prefix + ".conv.norm", true);
    keys.insert(prefix + ".conv.norm.running_mean");
    keys.insert(prefix + ".conv.norm.running_var");
  }
  keys.insert("tf_encoder.embed_positions.weight");
  for (int layer = 0; layer < config.tf_encoder.encoder_layers; ++layer) {
    const std::string prefix = "tf_encoder.layers." + std::to_string(layer);
    add_linear(prefix + ".self_attn.q_proj", true);
    add_linear(prefix + ".self_attn.k_proj", config.tf_encoder.k_proj_bias);
    add_linear(prefix + ".self_attn.v_proj", true);
    add_linear(prefix + ".self_attn.out_proj", true);
    add_linear(prefix + ".self_attn_layer_norm", true);
    add_linear(prefix + ".fc1", true);
    add_linear(prefix + ".fc2", true);
    add_linear(prefix + ".final_layer_norm", true);
  }
  for (const char *head : {"encoder_proj", "first_hidden_to_hidden",
                           "single_hidden_to_spks", "hidden_to_spks"}) {
    add_linear(std::string("sortformer_modules.") + head, true);
  }
  return keys;
}

// Note (Jiaxin Deng): mirrors Swift SortformerModel.sanitize; checkpoints
// already in MLX layout pass through unchanged.
std::unordered_map<std::string, mx::array>
Sanitize(std::unordered_map<std::string, mx::array> weights) {
  bool already_converted = false;
  for (const auto &[name, array] : weights) {
    if (name.find("subsampling.layers_") != std::string::npos) {
      already_converted = true;
    } else {
    }
  }
  std::unordered_map<std::string, mx::array> sanitized;
  for (auto &[name, array] : weights) {
    if (name.find("num_batches_tracked") != std::string::npos) {
      continue;
    } else {
    }
    std::string key = name;
    mx::array value = array;
    if (!already_converted) {
      const std::string from = "subsampling.layers.";
      const size_t found = key.find("fc_encoder." + from);
      if (found != std::string::npos) {
        key.replace(key.find(from), from.size(), "subsampling.layers_");
      } else {
      }
      const bool is_weight = key.find("weight") != std::string::npos;
      if (key.find("subsampling") != std::string::npos && is_weight &&
          key.find("linear") == std::string::npos && value.ndim() == 4) {
        value = mx::transpose(value, {0, 2, 3, 1});
      } else if ((key.find("pointwise_conv1") != std::string::npos ||
                  key.find("pointwise_conv2") != std::string::npos ||
                  key.find("depthwise_conv") != std::string::npos) &&
                 is_weight && value.ndim() == 3) {
        value = mx::transpose(value, {0, 2, 1});
      } else {
      }
    } else {
    }
    sanitized.insert_or_assign(key, value);
  }
  return sanitized;
}

} // namespace

std::unordered_map<std::string, mx::array>
LoadCheckpoint(const std::filesystem::path &model_directory,
               const Config &config) {
  std::vector<std::filesystem::path> weight_files;
  for (const auto &entry :
       std::filesystem::directory_iterator(model_directory)) {
    if (entry.path().extension() == ".safetensors") {
      weight_files.push_back(entry.path());
    } else {
    }
  }
  std::sort(weight_files.begin(), weight_files.end());
  std::unordered_map<std::string, mx::array> loaded_weights;
  for (const auto &path : weight_files) {
    auto [loaded, metadata] = mx::load_safetensors(path.string());
    for (auto &[name, array] : loaded) {
      loaded_weights.insert_or_assign(name, array);
    }
  }
  std::unordered_map<std::string, mx::array> weights =
      Sanitize(std::move(loaded_weights));
  const std::set<std::string> expected = ExpectedKeys(config);
  for (const auto &[name, array] : weights) {
    if (expected.count(name) == 0) {
      throw std::runtime_error("unexpected Sortformer checkpoint key " + name);
    } else {
    }
  }
  for (const std::string &name : expected) {
    if (weights.count(name) == 0) {
      throw std::runtime_error("Sortformer checkpoint is missing " + name);
    } else {
    }
  }
  std::vector<mx::array> parameters;
  parameters.reserve(weights.size());
  for (const auto &[name, array] : weights)
    parameters.push_back(array);
  mx::eval(parameters);
  return weights;
}

} // namespace sortformer
