// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Prince Canuma and contributors.
#include "qwen3_tts/qwen3_tts_model.h"

#include <set>

namespace qwen3_tts {
namespace mx = mlx::core;
using qwen3_asr::WeightMap;

namespace {

void ConvolutionShapes(WeightShapes &shapes, const std::string &prefix,
                       int input, int output, int kernel, bool bias = true,
                       int groups = 1) {
  shapes[prefix + ".weight"] = {output, kernel, input / groups};
  if (bias) {
    shapes[prefix + ".bias"] = {output};
  } else {
  }
}

void CodecTransformerShapes(WeightShapes &shapes, const std::string &prefix,
                            const nlohmann::json &config, bool encoder) {
  const int hidden = config.at("hidden_size");
  const int head_dim = config.at("head_dim");
  const int heads = config.at("num_attention_heads");
  const int kv_heads = config.at("num_key_value_heads");
  const int intermediate = config.at("intermediate_size");
  if (hidden <= 0 || head_dim <= 0 || head_dim % 2 || heads <= 0 ||
      kv_heads <= 0 || heads % kv_heads ||
      config.value("attention_bias", false)) {
    throw std::invalid_argument("unsupported Qwen3-TTS codec transformer");
  } else {
  }
  for (int layer = 0; layer < config.at("num_hidden_layers").get<int>();
       ++layer) {
    const auto stem = prefix + ".layers." + std::to_string(layer);
    for (const auto &name : {"input_layernorm", "post_attention_layernorm"}) {
      shapes[stem + "." + name + ".weight"] = {hidden};
      if (encoder) {
        shapes[stem + "." + name + ".bias"] = {hidden};
      } else {
      }
    }
    shapes[stem + ".self_attn_layer_scale.scale"] = {hidden};
    shapes[stem + ".mlp_layer_scale.scale"] = {hidden};
    LinearShapes(shapes, stem + ".self_attn.q_proj", hidden, heads * head_dim);
    LinearShapes(shapes, stem + ".self_attn.k_proj", hidden,
                 kv_heads * head_dim);
    LinearShapes(shapes, stem + ".self_attn.v_proj", hidden,
                 kv_heads * head_dim);
    LinearShapes(shapes, stem + ".self_attn.o_proj", heads * head_dim, hidden);
    if (encoder) {
      LinearShapes(shapes, stem + ".mlp.fc1", hidden, intermediate);
      LinearShapes(shapes, stem + ".mlp.fc2", intermediate, hidden);
    } else {
      LinearShapes(shapes, stem + ".mlp.gate_proj", hidden, intermediate);
      LinearShapes(shapes, stem + ".mlp.up_proj", hidden, intermediate);
      LinearShapes(shapes, stem + ".mlp.down_proj", intermediate, hidden);
    }
  }
  if (!encoder) {
    shapes[prefix + ".norm.weight"] = {hidden};
    LinearShapes(shapes, prefix + ".input_proj", config.at("latent_dim"),
                 hidden, true);
    LinearShapes(shapes, prefix + ".output_proj", hidden,
                 config.at("latent_dim"), true);
  } else {
  }
}

} // namespace

WeightMap AudioCodec::LoadWeights(const std::filesystem::path &directory,
                                  const nlohmann::json &config, bool encoder) {
  const auto &decoder = config.at("decoder_config");
  if (config.at("output_sample_rate") != kSampleRate ||
      config.at("decode_upsample_rate") != kSamplesPerFrame ||
      decoder.at("num_quantizers") != kCodeGroups ||
      decoder.at("codebook_size") != kCodebookSize ||
      decoder.at("num_semantic_quantizers") != 1 ||
      decoder.value("sliding_window", 72) != 72) {
    throw std::invalid_argument(
        "expected Qwen3-TTS 24 kHz 16-group speech tokenizer");
  } else {
  }
  WeightShapes shapes;
  std::set<std::string> transposed;
  const int codebook_dim = decoder.at("codebook_dim");
  for (const auto &part : {std::string("rvq_first"), std::string("rvq_rest")}) {
    const auto stem = "decoder.quantizer." + part;
    ConvolutionShapes(shapes, stem + ".input_proj", codebook_dim,
                      codebook_dim / 2, 1, false);
    ConvolutionShapes(shapes, stem + ".output_proj", codebook_dim / 2,
                      codebook_dim, 1, false);
    const int count = part == "rvq_first" ? 1 : kCodeGroups - 1;
    for (int group = 0; group < count; ++group) {
      shapes[stem + ".vq.layers." + std::to_string(group) +
             ".codebook.embed.weight"] = {kCodebookSize, codebook_dim / 2};
    }
  }
  const int latent = decoder.at("latent_dim");
  ConvolutionShapes(shapes, "decoder.pre_conv.conv", codebook_dim, latent, 3);
  CodecTransformerShapes(shapes, "decoder.pre_transformer", decoder, false);
  int upsampling = 1;
  const auto ratios = decoder.at("upsampling_ratios").get<std::vector<int>>();
  for (size_t index = 0; index < ratios.size(); ++index) {
    if (ratios[index] <= 0) {
      throw std::invalid_argument("invalid codec upsampling ratio");
    } else {
    }
    upsampling *= ratios[index];
    const auto stem = "decoder.upsample." + std::to_string(index);
    ConvolutionShapes(shapes, stem + ".0.conv", latent, latent, ratios[index]);
    transposed.insert(stem + ".0.conv.weight");
    ConvolutionShapes(shapes, stem + ".1.dwconv.conv", latent, latent, 7, true,
                      latent);
    shapes[stem + ".1.norm.weight"] = {latent};
    shapes[stem + ".1.norm.bias"] = {latent};
    shapes[stem + ".1.gamma"] = {latent};
    LinearShapes(shapes, stem + ".1.pwconv1", latent, latent * 4, true);
    LinearShapes(shapes, stem + ".1.pwconv2", latent * 4, latent, true);
  }
  int channels = decoder.at("decoder_dim");
  ConvolutionShapes(shapes, "decoder.decoder.0.conv", latent, channels, 7);
  const auto rates = decoder.at("upsample_rates").get<std::vector<int>>();
  for (size_t index = 0; index < rates.size(); ++index) {
    if (rates[index] <= 0) {
      throw std::invalid_argument("invalid waveform upsampling rate");
    } else {
    }
    upsampling *= rates[index];
    const auto stem = "decoder.decoder." + std::to_string(index + 1) + ".block";
    shapes[stem + ".0.alpha"] = {channels};
    shapes[stem + ".0.beta"] = {channels};
    ConvolutionShapes(shapes, stem + ".1.conv", channels, channels / 2,
                      rates[index] * 2);
    transposed.insert(stem + ".1.conv.weight");
    channels /= 2;
    for (int residual = 2; residual < 5; ++residual) {
      const auto unit = stem + "." + std::to_string(residual);
      for (const auto &activation : {"act1", "act2"}) {
        shapes[unit + "." + activation + ".alpha"] = {channels};
        shapes[unit + "." + activation + ".beta"] = {channels};
      }
      ConvolutionShapes(shapes, unit + ".conv1.conv", channels, channels, 7);
      ConvolutionShapes(shapes, unit + ".conv2.conv", channels, channels, 1);
    }
  }
  if (upsampling != kSamplesPerFrame) {
    throw std::invalid_argument(
        "codec upsampling does not produce 1920 samples per frame");
  } else {
  }
  const auto final = "decoder.decoder." + std::to_string(rates.size() + 1);
  shapes[final + ".alpha"] = {channels};
  shapes[final + ".beta"] = {channels};
  ConvolutionShapes(
      shapes, "decoder.decoder." + std::to_string(rates.size() + 2) + ".conv",
      channels, 1, 7);
  if (encoder) {
    const auto &configuration = config.at("encoder_config");
    if (configuration.at("pad_mode") != "constant" ||
        configuration.at("audio_channels") != 1 ||
        configuration.at("num_residual_layers") != 1 ||
        configuration.at("use_conv_shortcut") != false ||
        configuration.at("hidden_act") != "gelu" ||
        configuration.at("sampling_rate") != kSampleRate ||
        configuration.at("upsampling_ratios") != nlohmann::json({8, 6, 5, 4}) ||
        configuration.at("num_semantic_quantizers") != 1 ||
        configuration.at("codebook_size") != kCodebookSize ||
        configuration.value("sliding_window", 250) != 250 ||
        configuration.value("_frame_rate", 12.5f) != 12.5f ||
        config.at("encoder_valid_num_quantizers") != kCodeGroups) {
      throw std::invalid_argument(
          "unsupported reference codec encoder configuration");
    } else {
    }
    int width = configuration.at("num_filters");
    ConvolutionShapes(shapes, "encoder.encoder.layers.0.conv", 1, width,
                      configuration.at("kernel_size"));
    const auto downsampling =
        configuration.at("upsampling_ratios").get<std::vector<int>>();
    int layer = 1;
    for (auto ratio = downsampling.rbegin(); ratio != downsampling.rend();
         ++ratio) {
      const auto stem = "encoder.encoder.layers." + std::to_string(layer);
      const int compressed = width / configuration.at("compress").get<int>();
      ConvolutionShapes(shapes, stem + ".block.1.conv", width, compressed,
                        configuration.at("residual_kernel_size"));
      ConvolutionShapes(shapes, stem + ".block.3.conv", compressed, width, 1);
      ConvolutionShapes(shapes,
                        "encoder.encoder.layers." + std::to_string(layer + 2) +
                            ".conv",
                        width, width * 2, 2 * *ratio);
      width *= 2;
      layer += 3;
    }
    const int hidden = configuration.at("hidden_size");
    ConvolutionShapes(
        shapes, "encoder.encoder.layers." + std::to_string(layer + 1) + ".conv",
        width, hidden, configuration.at("last_kernel_size"));
    CodecTransformerShapes(shapes, "encoder.encoder_transformer", configuration,
                           true);
    ConvolutionShapes(shapes, "encoder.downsample.conv", hidden, hidden, 4,
                      false);
    const int dimension =
        configuration.at("vector_quantization_hidden_dimension");
    for (const auto &part :
         {std::string("semantic_residual_vector_quantizer"),
          std::string("acoustic_residual_vector_quantizer")}) {
      const auto stem = "encoder.quantizer." + part;
      ConvolutionShapes(shapes, stem + ".input_proj", hidden, dimension, 1,
                        false);
      ConvolutionShapes(shapes, stem + ".output_proj", dimension, hidden, 1,
                        false);
      const int count = part.starts_with("semantic")
                            ? 1
                            : configuration.at("num_quantizers").get<int>() - 1;
      for (int group = 0; group < count; ++group) {
        shapes[stem + ".layers." + std::to_string(group) +
               ".codebook.embed.weight"] = {kCodebookSize, dimension};
      }
    }
  } else {
  }
  auto raw = qwen3_asr::LoadSafetensors(directory);
  WeightMap weights;
  for (const auto &[name, weight] : raw) {
    if (!encoder && name.starts_with("encoder.")) {
      continue;
    } else if (name.find("._codebook.") != std::string::npos ||
               name.find(".codebook.") != std::string::npos) {
      const auto marker = name.find("._codebook.") != std::string::npos
                              ? "._codebook."
                              : ".codebook.";
      const auto position = name.find(marker);
      const auto statistic = name.substr(position + std::string(marker).size());
      if (statistic == "embedding_sum" || statistic == "embed_sum") {
        const auto usage =
            raw.at(name.substr(0, position) + marker + "cluster_usage");
        weights.emplace(name.substr(0, position) + ".codebook.embed.weight",
                        weight / mx::maximum(mx::expand_dims(usage, 1),
                                             mx::array(1e-5f, usage.dtype())));
      } else if (statistic == "cluster_usage" || statistic == "initialized") {
      } else {
        throw std::runtime_error("unexpected codec statistic: " + name);
      }
    } else if (weight.ndim() == 3 && name.ends_with(".weight")) {
      weights.emplace(name,
                      mx::transpose(weight, transposed.contains(name)
                                                ? std::vector<int>{1, 2, 0}
                                                : std::vector<int>{0, 2, 1}));
    } else {
      weights.emplace(name, weight);
    }
  }
  ValidateWeights(weights, shapes);
  return weights;
}

} // namespace qwen3_tts
