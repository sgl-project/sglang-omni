// SPDX-License-Identifier: Apache-2.0
#include "funasr/funasr_model.h"

#include <cmath>
#include <limits>
#include <map>
#include <stdexcept>

#include "common/kaldi_fbank.h"

namespace funasr {

namespace mx = mlx::core;
using omni_common::Qwen3DecoderConfig;
using omni_common::ReadJson;
using omni_common::WeightMap;

namespace {

constexpr const char *kEncoderPrefix = "model.audio_tower.layers.";
constexpr const char *kAdaptorPrefix = "model.multi_modal_projector";
constexpr const char *kDecoderPrefix = "model.language_model";
// The reference's fbank: 16 kHz, 25 ms windows, 80 mel bins.
constexpr int kSampleRate = 16000;
constexpr int kWindowMilliseconds = 25;
constexpr int kMelBinCount = 80;
constexpr int kLowFrameRateHalvings = 3;

int TimestampBlockCount(const nlohmann::json &audio) {
  return audio.at("num_timestamp_prediction_layers").get<int>();
}

int BlockCount(const nlohmann::json &audio) {
  return audio.at("num_hidden_layers").get<int>() - TimestampBlockCount(audio);
}

// Note (wilsonzheng0327): the audio tower is SenseVoice's encoder as one
// layer list, so its weights take SenseVoice's names: layer 0, the main
// stack, then the timestamp stack, whose last layers' final norms are
// after_norm and tp_norm, with the query, key and value projections fused
// and the FSMN kernels in MLX's layout.
WeightMap EncoderLayout(const WeightMap &weights, const nlohmann::json &audio) {
  const int block_count = BlockCount(audio);
  const int layer_count = audio.at("num_hidden_layers").get<int>();
  static const std::map<std::string, std::string> renamed = {
      {"input_layernorm.weight", "norm1.weight"},
      {"input_layernorm.bias", "norm1.bias"},
      {"post_attention_layernorm.weight", "norm2.weight"},
      {"post_attention_layernorm.bias", "norm2.bias"},
      {"mlp.fc1.weight", "feed_forward.w_1.weight"},
      {"mlp.fc1.bias", "feed_forward.w_1.bias"},
      {"mlp.fc2.weight", "feed_forward.w_2.weight"},
      {"mlp.fc2.bias", "feed_forward.w_2.bias"},
      {"self_attn.o_proj.weight", "self_attn.linear_out.weight"},
      {"self_attn.o_proj.bias", "self_attn.linear_out.bias"},
  };
  const std::string tower = kEncoderPrefix;
  WeightMap loaded;
  for (const auto &[name, weight] : weights) {
    if (!name.starts_with(tower)) {
      loaded.insert_or_assign(name, weight);
      continue;
    } else {
    }
    const size_t dot = name.find('.', tower.size());
    const int layer = std::stoi(name.substr(tower.size(), dot - tower.size()));
    const std::string parameter = name.substr(dot + 1);
    const std::string prefix =
        layer == 0 ? std::string("encoder.encoders0.0")
        : layer < block_count
            ? "encoder.encoders." + std::to_string(layer - 1)
            : "encoder.tp_encoders." + std::to_string(layer - block_count);
    const std::string attention = tower + std::to_string(layer) + ".self_attn.";
    if (parameter.starts_with("self_attn.q_proj.")) {
      const std::string kind = parameter.substr(parameter.rfind('.') + 1);
      loaded.insert_or_assign(
          prefix + ".self_attn.linear_q_k_v." + kind,
          mx::concatenate({weight, weights.at(attention + "k_proj." + kind),
                           weights.at(attention + "v_proj." + kind)},
                          0));
    } else if (parameter.starts_with("self_attn.k_proj.") ||
               parameter.starts_with("self_attn.v_proj.")) {
      // Fused into the query's entry above.
    } else if (parameter == "self_attn.fsmn.conv.weight") {
      loaded.insert_or_assign(prefix + ".self_attn.fsmn_block.weight",
                              mx::contiguous(mx::transpose(weight, {0, 2, 1})));
    } else if (parameter.starts_with("final_layernorm.") &&
               (layer == block_count - 1 || layer == layer_count - 1)) {
      loaded.insert_or_assign((layer == block_count - 1 ? "encoder.after_norm."
                                                        : "encoder.tp_norm.") +
                                  parameter.substr(parameter.rfind('.') + 1),
                              weight);
    } else if (renamed.contains(parameter)) {
      loaded.insert_or_assign(prefix + "." + renamed.at(parameter), weight);
    } else {
      throw std::runtime_error("unexpected Fun-ASR encoder weight " + name);
    }
  }
  return loaded;
}

// Audio tokens the prompt holds for a clip of frame_count stacked frames:
// three halvings, each rounding up.
int AudioTokenCount(int frame_count) {
  for (int halving = 0; halving < kLowFrameRateHalvings; ++halving) {
    frame_count = (frame_count + 1) / 2;
  }
  return frame_count;
}

} // namespace

FunAsrNano::FunAsrNano(const std::filesystem::path &model_directory)
    : FunAsrNano(ReadJson(model_directory / "config.json"),
                 ReadJson(model_directory / "processor_config.json")
                     .at("feature_extractor"),
                 model_directory) {}

FunAsrNano::FunAsrNano(const nlohmann::json &config,
                       const nlohmann::json &feature_extractor,
                       const std::filesystem::path &model_directory)
    : mel_filters_(mx::array(0.0f)),
      checkpoint_(EncoderLayout(omni_common::LoadSafetensors(model_directory),
                                config.at("audio_config")),
                  config),
      decoder_(checkpoint_, kDecoderPrefix,
               Qwen3DecoderConfig::FromJson(config.at("text_config"))) {
  const nlohmann::json &audio = config.at("audio_config");
  const nlohmann::json &adaptor = config.at("adaptor_config");
  encoder_.input_size = audio.at("num_mel_bins").get<int>() *
                        audio.at("num_stacked_frames").get<int>();
  encoder_.model_width = audio.at("hidden_size").get<int>();
  encoder_.head_count = audio.at("num_attention_heads").get<int>();
  encoder_.block_count = BlockCount(audio);
  encoder_.tp_block_count = TimestampBlockCount(audio);
  encoder_.kernel_size = audio.at("fsmn_kernel_size").get<int>();
  lfr_m_ = feature_extractor.at("num_frames_lfr").get<int>();
  lfr_n_ = feature_extractor.at("stride_lfr").get<int>();
  hop_length_ =
      kSampleRate * feature_extractor.at("frame_shift").get<int>() / 1000;
  adaptor_layer_count_ = adaptor.at("num_hidden_layers").get<int>();
  adaptor_head_count_ = adaptor.at("num_attention_heads").get<int>();
  adaptor_norm_eps_ = adaptor.at("layer_norm_eps").get<float>();
  // Note (wilsonzheng0327): the graph below is the published checkpoint's:
  // an 80-bin 16 kHz fbank of 25 ms windows, frames stacked as wide as the
  // first layer, ReLU feed-forwards, a 1e-5 encoder norm, heads splitting the
  // widths evenly, and the tied output embedding.
  if (config.value("model_type", std::string()) != "fun_asr_nano" ||
      feature_extractor.at("sampling_rate").get<int>() != kSampleRate ||
      feature_extractor.at("feature_size").get<int>() != kMelBinCount ||
      audio.at("num_mel_bins").get<int>() != kMelBinCount ||
      feature_extractor.at("frame_length").get<int>() != kWindowMilliseconds ||
      audio.at("num_stacked_frames").get<int>() != lfr_m_ || lfr_n_ < 1 ||
      hop_length_ < 1 || audio.at("hidden_act").get<std::string>() != "relu" ||
      adaptor.at("hidden_act").get<std::string>() != "relu" ||
      adaptor.at("projector_hidden_act").get<std::string>() != "relu" ||
      audio.at("layer_norm_eps").get<float>() != 1e-5f ||
      encoder_.head_count < 1 ||
      encoder_.model_width % encoder_.head_count != 0 ||
      encoder_.block_count < 1 || encoder_.tp_block_count < 1 ||
      encoder_.kernel_size < 1 || adaptor_head_count_ < 1 ||
      config.at("text_config").at("hidden_size").get<int>() %
              adaptor_head_count_ !=
          0 ||
      !config.value("tie_word_embeddings", true)) {
    throw std::runtime_error(
        "only the published Fun-ASR-Nano checkpoint layout is supported");
  } else {
  }
  window_ = omni_common::SymmetricHammingWindow(kSampleRate *
                                                kWindowMilliseconds / 1000);
  const int fft_size =
      omni_common::PaddedFftSize(static_cast<int>(window_.size()));
  const int frequency_count = fft_size / 2 + 1;
  const std::vector<double> banks =
      omni_common::KaldiMelBanks(fft_size, kMelBinCount);
  std::vector<float> filters(banks.size());
  for (int bin = 0; bin < frequency_count; ++bin) {
    for (int mel = 0; mel < kMelBinCount; ++mel) {
      filters[bin * kMelBinCount + mel] =
          static_cast<float>(banks[mel * frequency_count + bin]);
    }
  }
  mel_filters_ =
      mx::array(filters.data(), {frequency_count, kMelBinCount}, mx::float32);
  mx::async_eval(mel_filters_);
}

mx::array FunAsrNano::Features(const std::vector<float> &samples) const {
  // Note (wilsonzheng0327): the reference's fbank floors at float epsilon and
  // keeps no CMVN statistics.
  return omni_common::LowFrameRateFrames(
      omni_common::KaldiFbank(samples, window_, hop_length_, mel_filters_,
                              std::numeric_limits<float>::epsilon()),
      lfr_m_, lfr_n_);
}

mx::array FunAsrNano::AdaptorLayer(const mx::array &x,
                                   const std::string &prefix) const {
  const int length = x.shape(1);
  const int width = x.shape(2);
  const int head_width = width / adaptor_head_count_;
  const mx::array normed =
      checkpoint_.LayerNorm(x, prefix + ".input_layernorm", adaptor_norm_eps_);
  const auto project = [&](const std::string &name) {
    return mx::transpose(
        mx::reshape(checkpoint_.Linear(normed, prefix + ".self_attn." + name),
                    {1, length, adaptor_head_count_, head_width}),
        {0, 2, 1, 3});
  };
  const mx::array attended = mx::fast::scaled_dot_product_attention(
      project("q_proj"), project("k_proj"), project("v_proj"),
      1.0f / std::sqrt(static_cast<float>(head_width)));
  const mx::array hidden = mx::add(
      x, checkpoint_.Linear(mx::reshape(mx::transpose(attended, {0, 2, 1, 3}),
                                        {1, length, width}),
                            prefix + ".self_attn.o_proj"));
  const mx::array expanded = checkpoint_.Linear(
      checkpoint_.LayerNorm(hidden, prefix + ".post_attention_layernorm",
                            adaptor_norm_eps_),
      prefix + ".mlp.fc1");
  return mx::add(hidden,
                 checkpoint_.Linear(
                     mx::maximum(expanded, mx::array(0.0f, expanded.dtype())),
                     prefix + ".mlp.fc2"));
}

mx::array FunAsrNano::EncodeAudio(const mx::array &features) const {
  const std::string adaptor = kAdaptorPrefix;
  const mx::array &embedding_weight =
      checkpoint_.Weight(std::string(kDecoderPrefix) + ".embed_tokens.weight");
  mx::array x = omni_common::SanmEncode(
      checkpoint_,
      mx::expand_dims(mx::astype(features, embedding_weight.dtype()), 0),
      encoder_);
  const mx::array projected = checkpoint_.Linear(x, adaptor + ".linear_1");
  x = checkpoint_.Linear(
      mx::maximum(projected, mx::array(0.0f, projected.dtype())),
      adaptor + ".linear_2");
  for (int layer = 0; layer < adaptor_layer_count_; ++layer) {
    x = AdaptorLayer(x, adaptor + ".layers." + std::to_string(layer));
  }
  // Note (wilsonzheng0327): the adaptor keeps every frame, and the prompt
  // takes its first AudioTokenCount rows, as the reference does.
  const int token_count = AudioTokenCount(features.shape(0));
  return mx::reshape(mx::slice(x, {0, 0, 0}, {1, token_count, x.shape(2)}),
                     {token_count, x.shape(2)});
}

} // namespace funasr
