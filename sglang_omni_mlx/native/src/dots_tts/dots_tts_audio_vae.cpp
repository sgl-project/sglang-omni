// SPDX-License-Identifier: Apache-2.0
// dots.tts audio VAE encoder and BigVGAN decoder.
#include "dots_tts/dots_tts_acoustics.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>

#include "common/layers.h"

namespace dots_tts {
namespace mx = mlx::core;
namespace {

mx::array LeakyRelu(const mx::array &x, float slope) {
  return mx::maximum(x, mx::multiply(x, mx::array(slope, x.dtype())));
}

} // namespace

mx::array AudioEncoder::operator()(const mx::array &x) const {
  mx::array out = pre_conv(x);
  out = LeakyRelu(out, act_slope);
  for (size_t i = 0; i < down_convs.size(); ++i) {
    out = down_convs[i](out);
    out = res_stacks[i](out);
    out = LeakyRelu(out, act_slope);
  }
  return post_conv(out);
}

mx::array BigVganDecoder::operator()(const mx::array &x) const {
  mx::array out = conv_pre(x);
  const int stages = static_cast<int>(ups.size());
  for (int i = 0; i < stages; ++i) {
    out = ups[i](out);
    mx::array summed(0.0f);
    bool first = true;
    for (int j = 0; j < num_kernels; ++j) {
      const AMPBlock1 &rb = resblocks[i * num_kernels + j];
      const mx::array block = rb(out);
      summed = first ? block : mx::add(summed, block);
      first = false;
    }
    out = mx::divide(summed, mx::array(static_cast<float>(num_kernels)));
  }
  out = act_post(out);
  return mx::clip(conv_post(out), mx::array(-1.0f), mx::array(1.0f));
}

mx::array AudioVAE::Encode(const mx::array &wav) const {
  // note (guozhihao-224): the model always loads the full encode path, so a
  // missing piece is a build bug, not a runtime condition.
  assert(encoder.has_value() && enc_lin0.has_value() && enc_slstm.has_value() &&
         enc_lin2.has_value() && pre_proj.has_value());

  mx::array x = mx::astype(wav, mx::float32);
  x = mx::transpose(x, {0, 2, 1});
  x = (*encoder)(x);
  x = (*enc_lin0)(x);
  x = (*enc_slstm)(x);
  x = (*enc_lin2)(x);
  x = (*pre_proj)(x);
  return mx::transpose(x, {0, 2, 1});
}

mx::array AudioVAE::Decode(const mx::array &latent) const {
  // note (guozhihao-224): [B, latent_dim, T] -> [B, 1, T*1920].
  mx::array x = mx::astype(latent, mx::float32);
  x = mx::transpose(x, {0, 2, 1});
  x = post_proj(x);
  x = mi_lin0(x);
  x = slstm(x);
  x = mi_lin2(x);
  const mx::array out = decoder(x);
  return mx::transpose(out, {0, 2, 1});
}

AudioVAE LoadAudioVae(const DotsCheckpoint &checkpoint) {
  const nlohmann::json &vocoder_config = checkpoint.config().at("vocoder");
  const int mutual_information_layer_count =
      vocoder_config.value("mi_num_layers", 4);

  const auto conv1d = [&](const std::string &name, bool causal, int stride,
                          int dilation) {
    return nn::Conv1d(
        Conv1dWeight(checkpoint.Weight(name + ".weight"), mx::float32),
        checkpoint.Weight(name + ".bias"), causal, stride, dilation);
  };
  const auto convt1d = [&](const std::string &name, int stride, bool causal) {
    return nn::ConvTranspose1d(
        ConvTranspose1dWeight(checkpoint.Weight(name + ".weight"), mx::float32),
        checkpoint.Weight(name + ".bias"), stride, causal);
  };
  const auto filter_weight = [&](const std::string &name, int channels) {
    return FilterWeight(checkpoint.Weight(name), channels);
  };
  const auto resampler = [&](const std::string &prefix, int channels) {
    return AliasFreeResampler{
        filter_weight(prefix + ".upsample.filter", channels),
        filter_weight(prefix + ".downsample.lowpass.filter", channels),
        mx::astype(checkpoint.Weight(prefix + ".act.alpha"), mx::float32),
        mx::astype(checkpoint.Weight(prefix + ".act.beta"), mx::float32),
        /*ratio=*/2};
  };
  const bool causal = vocoder_config.value("causal", true);
  const bool use_bias_at_final =
      vocoder_config.value("use_bias_at_final", true);
  const int num_kernels =
      static_cast<int>(vocoder_config.at("resblock_kernel_sizes").size());
  const int initial = vocoder_config.at("upsample_initial_channel").get<int>();
  const std::vector<int> upsample_rates =
      vocoder_config.at("upsample_rates").get<std::vector<int>>();
  const std::vector<std::vector<int>> dils =
      vocoder_config.at("resblock_dilation_sizes")
          .get<std::vector<std::vector<int>>>();

  nn::Conv1d post_proj(
      Conv1dWeight(checkpoint.Weight("post_proj.weight"), mx::float32),
      checkpoint.Weight("post_proj.bias"), false);

  std::vector<mx::array> slstm_ih, slstm_hh, slstm_bi, slstm_bh;
  for (int layer_index = 0; layer_index < mutual_information_layer_count;
       ++layer_index) {
    slstm_ih.push_back(checkpoint.Weight("dec_mi_layer.1.lstm.weight_ih_l" +
                                         std::to_string(layer_index)));
    slstm_hh.push_back(checkpoint.Weight("dec_mi_layer.1.lstm.weight_hh_l" +
                                         std::to_string(layer_index)));
    slstm_bi.push_back(checkpoint.Weight("dec_mi_layer.1.lstm.bias_ih_l" +
                                         std::to_string(layer_index)));
    slstm_bh.push_back(checkpoint.Weight("dec_mi_layer.1.lstm.bias_hh_l" +
                                         std::to_string(layer_index)));
  }
  SLSTM slstm{std::move(slstm_ih), std::move(slstm_hh), std::move(slstm_bi),
              std::move(slstm_bh), mutual_information_layer_count};

  // note (guozhihao-224): conv_pre stays non-causal even in a causal decoder.
  nn::Conv1d conv_pre(
      Conv1dWeight(checkpoint.Weight("decoder.conv_pre.weight"), mx::float32),
      checkpoint.Weight("decoder.conv_pre.bias"), false);
  std::vector<nn::ConvTranspose1d> ups;
  std::vector<AMPBlock1> resblocks;
  for (size_t i = 0; i < upsample_rates.size(); ++i) {
    const std::string prefix = "decoder.ups." + std::to_string(i) + ".0";
    ups.push_back(convt1d(prefix, upsample_rates[i], causal));
    const int ch = initial / (1 << (i + 1));
    for (size_t j = 0; j < dils.size(); ++j) {
      const std::string rp =
          "decoder.resblocks." + std::to_string(i * dils.size() + j);
      std::vector<nn::Conv1d> convs1, convs2;
      std::vector<AliasFreeResampler> acts;
      for (size_t d = 0; d < dils.size(); ++d) {
        convs1.push_back(
            conv1d(rp + ".convs1." + std::to_string(d), causal, 1, dils[j][d]));
        convs2.push_back(
            conv1d(rp + ".convs2." + std::to_string(d), causal, 1, 1));
      }
      for (int a = 0; a < 6; ++a)
        acts.push_back(resampler(rp + ".activations." + std::to_string(a), ch));
      resblocks.push_back(
          AMPBlock1{std::move(convs1), std::move(convs2), std::move(acts)});
    }
  }
  const int final_ch = initial / (1 << upsample_rates.size());
  AliasFreeResampler act_post = resampler("decoder.activation_post", final_ch);
  nn::Conv1d conv_post(
      Conv1dWeight(checkpoint.Weight("decoder.conv_post.weight"), mx::float32),
      use_bias_at_final && checkpoint.Has("decoder.conv_post.bias")
          ? std::optional<mx::array>(
                checkpoint.Weight("decoder.conv_post.bias"))
          : std::nullopt,
      causal);
  BigVganDecoder decoder{std::move(conv_pre),  std::move(ups),
                         std::move(resblocks), std::move(act_post),
                         std::move(conv_post), num_kernels};

  const bool causal_encoder = vocoder_config.value("causal_encoder", true);
  const std::vector<int> down_rates =
      vocoder_config.at("downsample_rates").get<std::vector<int>>();
  constexpr int kEncoderBlocks = 6;
  constexpr int kDilationBase = 2;
  const std::string audio_encoder_generator_prefix = "audio_encoder.generator";
  nn::Conv1d pre_conv(
      Conv1dWeight(
          checkpoint.Weight(audio_encoder_generator_prefix + ".0.layer.weight"),
          mx::float32),
      checkpoint.Weight(audio_encoder_generator_prefix + ".0.layer.bias"),
      causal_encoder);
  std::vector<nn::Conv1d> down_convs;
  std::vector<ResStack> res_stacks;
  for (size_t i = 0; i < down_rates.size(); ++i) {
    const int base_idx = static_cast<int>(2 + 3 * i);
    down_convs.push_back(nn::Conv1d(
        Conv1dWeight(checkpoint.Weight(audio_encoder_generator_prefix + "." +
                                       std::to_string(base_idx) +
                                       ".layer.weight"),
                     mx::float32),
        checkpoint.Weight(audio_encoder_generator_prefix + "." +
                          std::to_string(base_idx) + ".layer.bias"),
        causal_encoder, down_rates[i]));
    const int rs_idx = base_idx + 1;
    std::vector<nn::Conv1d> convs1, convs2;
    for (int b = 0; b < kEncoderBlocks; ++b) {
      const int dil = static_cast<int>(std::pow(kDilationBase, b));
      convs1.push_back(nn::Conv1d(
          Conv1dWeight(checkpoint.Weight(audio_encoder_generator_prefix + "." +
                                         std::to_string(rs_idx) + ".layers." +
                                         std::to_string(b) + ".2.weight"),
                       mx::float32),
          checkpoint.Weight(audio_encoder_generator_prefix + "." +
                            std::to_string(rs_idx) + ".layers." +
                            std::to_string(b) + ".2.bias"),
          causal_encoder, 1, dil));
      convs2.push_back(nn::Conv1d(
          Conv1dWeight(checkpoint.Weight(audio_encoder_generator_prefix + "." +
                                         std::to_string(rs_idx) + ".layers." +
                                         std::to_string(b) + ".5.weight"),
                       mx::float32),
          checkpoint.Weight(audio_encoder_generator_prefix + "." +
                            std::to_string(rs_idx) + ".layers." +
                            std::to_string(b) + ".5.bias"),
          causal_encoder, 1));
    }
    res_stacks.push_back(ResStack{std::move(convs1), std::move(convs2)});
  }
  const int post_idx = static_cast<int>(2 + 3 * down_rates.size());
  nn::Conv1d post_conv(
      Conv1dWeight(checkpoint.Weight(audio_encoder_generator_prefix + "." +
                                     std::to_string(post_idx) +
                                     ".layer.weight"),
                   mx::float32),
      checkpoint.Weight(audio_encoder_generator_prefix + "." +
                        std::to_string(post_idx) + ".layer.bias"),
      /*causal=*/false);
  AudioEncoder encoder{std::move(pre_conv), std::move(down_convs),
                       std::move(res_stacks), std::move(post_conv), 0.2f};

  std::vector<mx::array> enc_ih, enc_hh, enc_bi, enc_bh;
  for (int layer_index = 0; layer_index < mutual_information_layer_count;
       ++layer_index) {
    enc_ih.push_back(checkpoint.Weight("enc_mi_layer.1.lstm.weight_ih_l" +
                                       std::to_string(layer_index)));
    enc_hh.push_back(checkpoint.Weight("enc_mi_layer.1.lstm.weight_hh_l" +
                                       std::to_string(layer_index)));
    enc_bi.push_back(checkpoint.Weight("enc_mi_layer.1.lstm.bias_ih_l" +
                                       std::to_string(layer_index)));
    enc_bh.push_back(checkpoint.Weight("enc_mi_layer.1.lstm.bias_hh_l" +
                                       std::to_string(layer_index)));
  }
  SLSTM enc_slstm{std::move(enc_ih), std::move(enc_hh), std::move(enc_bi),
                  std::move(enc_bh), mutual_information_layer_count};
  nn::Conv1d pre_proj(
      Conv1dWeight(checkpoint.Weight("pre_proj.weight"), mx::float32),
      checkpoint.Weight("pre_proj.bias"), false);

  AudioVAE vae;
  vae.post_proj = std::move(post_proj);
  vae.mi_lin0 = nn::Linear(checkpoint.Weight("dec_mi_layer.0.weight"),
                           checkpoint.Weight("dec_mi_layer.0.bias"));
  vae.slstm = std::move(slstm);
  vae.mi_lin2 = nn::Linear(checkpoint.Weight("dec_mi_layer.2.weight"),
                           checkpoint.Weight("dec_mi_layer.2.bias"));
  vae.decoder = std::move(decoder);
  vae.encoder = std::move(encoder);
  vae.enc_lin0 = nn::Linear(checkpoint.Weight("enc_mi_layer.0.weight"),
                            checkpoint.Weight("enc_mi_layer.0.bias"));
  vae.enc_slstm = std::move(enc_slstm);
  vae.enc_lin2 = nn::Linear(checkpoint.Weight("enc_mi_layer.2.weight"),
                            checkpoint.Weight("enc_mi_layer.2.bias"));
  vae.pre_proj = std::move(pre_proj);
  return vae;
}

} // namespace dots_tts
