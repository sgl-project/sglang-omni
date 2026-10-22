// SPDX-License-Identifier: Apache-2.0
#include "voxtral_tts/config.h"

#include <cctype>
#include <sstream>
#include <stdexcept>

#include "common/layers.h"

namespace voxtral_tts {

namespace {

constexpr float kDefaultAcousticNormEpsilon = 1e-5f;
constexpr const char *kDefaultEncoderConvolutionStrides = "2, 2, 2, 1";

std::vector<int> ParseIntegerList(const std::string &values,
                                  const std::string &field_name) {
  std::vector<int> parsed_values;
  std::stringstream value_stream(values);
  std::string value_text;
  while (std::getline(value_stream, value_text, ',')) {
    size_t parsed_characters = 0;
    const int value = std::stoi(value_text, &parsed_characters);
    while (parsed_characters < value_text.size() &&
           std::isspace(
               static_cast<unsigned char>(value_text[parsed_characters]))) {
      ++parsed_characters;
    }
    if (parsed_characters != value_text.size() || value < 1) {
      throw std::invalid_argument(field_name +
                                  " must contain positive integers");
    } else {
    }
    parsed_values.push_back(value);
  }
  if (parsed_values.empty()) {
    throw std::invalid_argument(field_name + " must not be empty");
  } else {
  }
  return parsed_values;
}

} // namespace

VoxtralConfig
VoxtralConfig::Load(const std::filesystem::path &checkpoint_directory) {
  const nlohmann::json parameters =
      qwen3_asr::ReadJson(checkpoint_directory / "params.json");
  const nlohmann::json &multimodal = parameters.at("multimodal");
  const nlohmann::json &audio_values = multimodal.at("audio_model_args");
  const nlohmann::json &tokenizer_values =
      multimodal.at("audio_tokenizer_args");
  const nlohmann::json &acoustic_values =
      audio_values.at("acoustic_transformer_args");
  const nlohmann::json &voice_values = tokenizer_values.at("voice");

  VoxtralConfig config{
      .text = {parameters.at("dim").get<int>(),
               parameters.at("n_layers").get<int>(),
               parameters.at("head_dim").get<int>(),
               parameters.at("hidden_dim").get<int>(),
               parameters.at("n_heads").get<int>(),
               parameters.at("n_kv_heads").get<int>(),
               parameters.at("vocab_size").get<int>(),
               parameters.at("rope_theta").get<float>(),
               parameters.at("norm_eps").get<float>()},
      .audio = {audio_values.at("semantic_codebook_size").get<int>(),
                audio_values.at("acoustic_codebook_size").get<int>(),
                audio_values.at("n_acoustic_codebook").get<int>(),
                {acoustic_values.at("input_dim").get<int>(),
                 acoustic_values.at("dim").get<int>(),
                 acoustic_values.at("n_layers").get<int>(),
                 acoustic_values.at("head_dim").get<int>(),
                 acoustic_values.at("hidden_dim").get<int>(),
                 acoustic_values.at("n_heads").get<int>(),
                 acoustic_values.at("n_kv_heads").get<int>(),
                 acoustic_values.at("use_biases").get<bool>(),
                 acoustic_values.value("norm_eps",
                                       kDefaultAcousticNormEpsilon)},
                audio_values.at("audio_token_id").get<int>()},
      .audio_tokenizer =
          {tokenizer_values.at("sampling_rate").get<int>(),
           tokenizer_values.at("pretransform_patch_size").get<int>(),
           tokenizer_values.at("patch_proj_kernel_size").get<int>(),
           tokenizer_values.at("semantic_codebook_size").get<int>(),
           tokenizer_values.at("semantic_dim").get<int>(),
           tokenizer_values.at("acoustic_codebook_size").get<int>(),
           tokenizer_values.at("acoustic_dim").get<int>(),
           tokenizer_values.at("causal").get<bool>(),
           tokenizer_values.at("attn_sliding_window_size").get<int>(),
           tokenizer_values.at("half_attn_window_upon_downsampling")
               .get<bool>(),
           ParseIntegerList(tokenizer_values.value(
                                "encoder_convs_strides_str",
                                std::string(kDefaultEncoderConvolutionStrides)),
                            "encoder_convs_strides_str"),
           tokenizer_values.at("dim").get<int>(),
           tokenizer_values.at("hidden_dim").get<int>(),
           tokenizer_values.at("head_dim").get<int>(),
           tokenizer_values.at("n_heads").get<int>(),
           tokenizer_values.at("n_kv_heads").get<int>(),
           tokenizer_values.at("qk_norm_eps").get<float>(),
           tokenizer_values.at("qk_norm").get<bool>(),
           tokenizer_values.at("use_biases").get<bool>(),
           tokenizer_values.at("norm_eps").get<float>(),
           tokenizer_values.at("layer_scale").get<bool>(),
           tokenizer_values.at("layer_scale_init").is_null()
               ? std::nullopt
               : std::optional<float>(
                     tokenizer_values.at("layer_scale_init").get<float>()),
           ParseIntegerList(
               tokenizer_values.at("decoder_transformer_lengths_str")
                   .get<std::string>(),
               "decoder_transformer_lengths_str"),
           ParseIntegerList(tokenizer_values.at("decoder_convs_kernels_str")
                                .get<std::string>(),
                            "decoder_convs_kernels_str"),
           ParseIntegerList(tokenizer_values.at("decoder_convs_strides_str")
                                .get<std::string>(),
                            "decoder_convs_strides_str")},
      .voices = {},
      .checkpoint_directory = checkpoint_directory};

  for (const auto &[voice_name, token_count] : voice_values.items()) {
    config.voices.emplace(voice_name, token_count.get<int>());
  }
  const AudioTokenizerConfig &audio_tokenizer = config.audio_tokenizer;
  if (audio_tokenizer.transformer_lengths.size() !=
          audio_tokenizer.convolution_kernels.size() ||
      audio_tokenizer.transformer_lengths.size() !=
          audio_tokenizer.convolution_strides.size() ||
      audio_tokenizer.transformer_lengths.size() !=
          audio_tokenizer.encoder_convolution_strides.size()) {
    throw std::invalid_argument(
        "Voxtral audio tokenizer configuration lengths differ");
  } else {
  }
  return config;
}

} // namespace voxtral_tts
