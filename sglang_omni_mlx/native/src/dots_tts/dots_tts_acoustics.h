// SPDX-License-Identifier: Apache-2.0
// The dots.tts acoustic networks and their building blocks: the flow-matching
// DiT and solver, the AudioVAE (BigVGAN vocoder + SLSTM) and the VAE semantic
// encoder, plus the linear/attention/convolution primitives they share.
#pragma once

#include <optional>
#include <vector>

#include "dots_tts/dots_tts_checkpoint.h"
#include "mlx/mlx.h"

namespace dots_tts::nn {

namespace mx = mlx::core;

struct Linear {
  mx::array weight;
  std::optional<mx::array> bias;

  Linear() : weight(0.0f) {}
  Linear(mx::array weight, std::optional<mx::array> bias = std::nullopt)
      : weight(std::move(weight)), bias(std::move(bias)) {}
  mx::array operator()(const mx::array &x) const {
    // note (guozhihao-224): compute in the input's dtype so a bf16 flow path
    // stays on Metal's bf16 matmul; fp32 inputs keep the reference rounding.
    const mx::Dtype in_dtype = x.dtype();
    mx::array y = mx::matmul(x, mx::transpose(mx::astype(weight, in_dtype)));
    if (bias.has_value()) {
      y = mx::add(y, mx::astype(*bias, in_dtype));
    }
    return y;
  }
};

// note (guozhihao-224): RMS norm with a learned per-feature weight; eps is
// torch's finfo(fp32).eps as the reference resolves.
struct RMSNorm {
  mx::array weight;
  float eps = 1.1920928955078125e-07f;
  RMSNorm(mx::array weight, float eps = 1.1920928955078125e-07f)
      : weight(std::move(weight)), eps(eps) {}
  mx::array operator()(const mx::array &x) const;
};

struct LayerNorm {
  int dim = 0;
  float eps = 1e-5f;
  explicit LayerNorm(int dim, float eps = 1e-5f) : dim(dim), eps(eps) {}
  mx::array operator()(const mx::array &x) const;
};

mx::array ReplicatePad(const mx::array &x, int left, int right);

struct Conv1d {
  mx::array weight;
  std::optional<mx::array> bias;
  bool causal = false;
  int stride = 1;
  int dilation = 1;
  int groups = 1;
  int left_padding = 0;
  int padding = 0;

  Conv1d() : weight(0.0f) {}
  Conv1d(mx::array weight, std::optional<mx::array> bias, bool causal_,
         int stride_ = 1, int dilation_ = 1, int groups_ = 1)
      : weight(std::move(weight)), bias(std::move(bias)), causal(causal_),
        stride(stride_), dilation(dilation_), groups(groups_) {
    const int kernel = this->weight.shape(1);
    if (causal) {
      left_padding = dilation * (kernel - 1);
      padding = 0;
    } else {
      left_padding = 0;
      padding = (kernel * dilation - dilation) / 2;
    }
  }

  mx::array operator()(const mx::array &x) const {
    mx::array input = x;
    if (causal && left_padding > 0) {
      input = mx::pad(input, {{0, 0}, {left_padding, 0}, {0, 0}});
    }
    mx::array y = mx::conv1d(input, weight, stride, padding, dilation, groups);
    if (bias.has_value()) {
      y = mx::add(y, *bias);
    }
    return y;
  }
};

struct ConvTranspose1d {
  mx::array weight;
  std::optional<mx::array> bias;
  int stride = 1;
  bool causal = false;

  ConvTranspose1d() : weight(0.0f) {}
  ConvTranspose1d(mx::array weight, std::optional<mx::array> bias, int stride_,
                  bool causal_)
      : weight(std::move(weight)), bias(std::move(bias)), stride(stride_),
        causal(causal_) {}

  mx::array operator()(const mx::array &x) const {
    mx::array y = mx::conv_transpose1d(x, weight, stride, /*padding=*/0,
                                       /*dilation=*/1, /*output_padding=*/0,
                                       /*groups=*/1);
    if (bias.has_value()) {
      y = mx::add(y, *bias);
    }
    if (causal) {
      y = mx::slice(y, {0, 0, 0},
                    {y.shape(0), y.shape(1) - stride, y.shape(2)});
    }
    return y;
  }
};

enum class Activation { GeluTanh, Silu };

struct Mlp {
  Linear fc1;
  Linear fc2;
  Activation activation = Activation::GeluTanh;

  Mlp(Linear fc1_, Linear fc2_, Activation activation_)
      : fc1(std::move(fc1_)), fc2(std::move(fc2_)), activation(activation_) {}

  mx::array operator()(const mx::array &x) const;
};

mx::array GeluTanh(const mx::array &x);

enum class NormLayer { LayerNorm, RMSNorm };

struct RotaryEmbedding {
  int dim = 0;
  float theta = 10000.0f;

  RotaryEmbedding(int dim, float theta_ = 10000.0f) : dim(dim), theta(theta_) {}
  mx::array Apply(const mx::array &t) const;
};

struct MultiHeadAttention {
  int hidden_size = 0;
  int num_heads = 0;
  int head_dim = 0;
  float scale = 1.0f;
  bool qkv_bias = false;
  bool qk_norm = false;
  bool rotary_bias = false;
  float rotary_theta = 10000.0f;
  NormLayer norm_layer = NormLayer::LayerNorm;

  Linear q_proj;
  Linear k_proj;
  Linear v_proj;
  Linear o_proj;
  std::optional<mx::array> q_norm_weight;
  std::optional<mx::array> k_norm_weight;
  std::optional<RotaryEmbedding> rotary;

  MultiHeadAttention(int hidden_size_, int num_heads_, bool qkv_bias_,
                     bool qk_norm_, NormLayer norm_layer_, bool rotary_bias_,
                     float rotary_theta_);
  mx::array operator()(const mx::array &q, const std::optional<mx::array> &k,
                       const std::optional<mx::array> &v,
                       const std::optional<mx::array> &mask) const;
};

} // namespace dots_tts::nn

namespace dots_tts {

namespace mx = mlx::core;

mx::array SnakeBeta(const mx::array &x, const mx::array &alpha,
                    const mx::array &beta);

struct AliasFreeResampler {
  mx::array up_filter{0.0f};
  mx::array down_filter{0.0f};
  mx::array alpha{0.0f};
  mx::array beta{0.0f};
  int ratio = 2;

  mx::array operator()(const mx::array &x) const;
};

struct AMPBlock1 {
  std::vector<nn::Conv1d> convs1;
  std::vector<nn::Conv1d> convs2;
  std::vector<AliasFreeResampler> acts;

  mx::array operator()(const mx::array &x) const;
};

struct SLSTM {
  std::vector<mx::array> weight_ih;
  std::vector<mx::array> weight_hh;
  std::vector<mx::array> bias_ih;
  std::vector<mx::array> bias_hh;
  int num_layers = 1;

  mx::array operator()(const mx::array &x) const;
};

struct ResStack {
  std::vector<nn::Conv1d> convs1;
  std::vector<nn::Conv1d> convs2;

  mx::array operator()(const mx::array &x) const;
};

struct EncoderLayer {
  nn::RMSNorm attn_norm;
  nn::MultiHeadAttention attn;
  nn::RMSNorm ffn_norm;
  nn::Mlp ffn;

  EncoderLayer(nn::RMSNorm attn_norm_, nn::MultiHeadAttention attn_,
               nn::RMSNorm ffn_norm_, nn::Mlp ffn_)
      : attn_norm(std::move(attn_norm_)), attn(std::move(attn_)),
        ffn_norm(std::move(ffn_norm_)), ffn(std::move(ffn_)) {}

  mx::array operator()(const mx::array &x, const mx::array &mask) const;
};

struct SemanticEncoderCache {
  std::vector<mx::array> layer_normed;
};

struct VAESemanticEncoder {
  nn::Conv1d ds_proj;
  nn::Linear in_proj;
  std::vector<EncoderLayer> layers;
  nn::Linear out_proj;
  int out_ds_rate = 2;

  VAESemanticEncoder(nn::Conv1d ds_proj_, nn::Linear in_proj_,
                     std::vector<EncoderLayer> layers_, nn::Linear out_proj_,
                     int out_ds_rate_)
      : ds_proj(std::move(ds_proj_)), in_proj(std::move(in_proj_)),
        layers(std::move(layers_)), out_proj(std::move(out_proj_)),
        out_ds_rate(out_ds_rate_) {}

  mx::array operator()(const mx::array &x) const;
  void Seed(SemanticEncoderCache &cache, const mx::array &x) const;
  mx::array Append(SemanticEncoderCache &cache, const mx::array &x) const;
};

struct TimestepEmbedder {
  nn::Linear mlp_fc0;
  nn::Linear mlp_fc2;
  int frequency_embedding_size = 256;
  float max_period = 10000.0f;

  TimestepEmbedder(nn::Linear fc0, nn::Linear fc2, int freq = 256,
                   float max_period_ = 10000.0f)
      : mlp_fc0(std::move(fc0)), mlp_fc2(std::move(fc2)),
        frequency_embedding_size(freq), max_period(max_period_) {}

  mx::array operator()(const mx::array &t) const;
};

struct FinalLayer {
  nn::Linear adaLN_linear;
  nn::LayerNorm norm;
  nn::Linear linear;

  FinalLayer(nn::Linear adaLN, nn::LayerNorm norm_, nn::Linear linear_)
      : adaLN_linear(std::move(adaLN)), norm(std::move(norm_)),
        linear(std::move(linear_)) {}

  mx::array operator()(const mx::array &x, const mx::array &conditioning) const;
};

struct DiTBlock {
  nn::MultiHeadAttention attn;
  nn::Mlp ffn;
  nn::Linear adaLN_linear;
  nn::LayerNorm norm1;
  nn::LayerNorm norm2;

  DiTBlock(nn::MultiHeadAttention attn_, nn::Mlp ffn_, nn::Linear adaLN,
           int hidden_size)
      : attn(std::move(attn_)), ffn(std::move(ffn_)),
        adaLN_linear(std::move(adaLN)), norm1(hidden_size), norm2(hidden_size) {
  }

  mx::array operator()(const mx::array &x, const mx::array &conditioning,
                       const std::optional<mx::array> &mask) const;
};

struct DiT {
  nn::Linear input_layer;
  TimestepEmbedder time_embedder;
  // note (guozhihao-224): present on MeanFlow checkpoints only
  // (config.meanflow.use_duration_embedding); its embedding is added to the
  // conditioning alongside the timestep embedding.
  std::optional<TimestepEmbedder> duration_embedder;
  std::vector<DiTBlock> blocks;
  FinalLayer output_layer;

  DiT(nn::Linear input, TimestepEmbedder time, std::vector<DiTBlock> blocks_,
      FinalLayer output,
      std::optional<TimestepEmbedder> duration_embedder_ = std::nullopt)
      : input_layer(std::move(input)), time_embedder(std::move(time)),
        duration_embedder(std::move(duration_embedder_)),
        blocks(std::move(blocks_)), output_layer(std::move(output)) {}

  mx::array
  operator()(const mx::array &x, const mx::array &timesteps,
             const std::optional<mx::array> &attn_mask,
             const std::optional<mx::array> &speaker_conditioning,
             const std::optional<mx::array> &duration = std::nullopt) const;
};

mx::array FmSolverStep(const DiT &dit, const nn::Linear &coordinate_proj,
                       const mx::array &t, const mx::array &z,
                       const mx::array &input_sequence,
                       const mx::array &unconditional_sequence,
                       const std::optional<mx::array> &attn_mask,
                       const mx::array &speaker_conditioning,
                       float guidance_scale);

struct FlowSolver {
  const DiT *dit;
  nn::Linear coordinate_proj;
  int latent_dim = 128;

  FlowSolver(const DiT &dit_, nn::Linear coordinate_proj_, int latent_dim_)
      : dit(&dit_), coordinate_proj(std::move(coordinate_proj_)),
        latent_dim(latent_dim_) {}

  mx::array Denoise(const mx::array &input_sequence,
                    const mx::array &unconditional_sequence,
                    const std::optional<mx::array> &attn_mask,
                    const mx::array &speaker_conditioning, float guidance_scale,
                    int num_steps, const mx::array &noise) const;

  mx::array DenoiseMeanFlow(const mx::array &input_sequence,
                            const std::optional<mx::array> &attn_mask,
                            const mx::array &speaker_conditioning,
                            int num_steps, const mx::array &noise) const;
};

struct FlowHead {
  nn::Linear hidden_proj;
  nn::Linear latent_proj;
  nn::Linear xvec_linear;
  mx::array xvec_norm_weight{0.0f};
  mx::array xvec_norm_bias{0.0f};
  DiT dit;
  nn::Linear coordinate_proj;
  VAESemanticEncoder patch_encoder;
};

FlowHead LoadFlowHead(const DotsCheckpoint &checkpoint);

struct AudioEncoder {
  nn::Conv1d pre_conv;
  std::vector<nn::Conv1d> down_convs;
  std::vector<ResStack> res_stacks;
  nn::Conv1d post_conv;
  float act_slope = 0.2f;

  mx::array operator()(const mx::array &x) const;
};

struct BigVganDecoder {
  nn::Conv1d conv_pre;
  std::vector<nn::ConvTranspose1d> ups;
  std::vector<AMPBlock1> resblocks;
  AliasFreeResampler act_post;
  nn::Conv1d conv_post;
  int num_kernels = 3;

  mx::array operator()(const mx::array &x) const;
};

struct AudioVAE {
  nn::Conv1d post_proj;
  nn::Linear mi_lin0;
  SLSTM slstm;
  nn::Linear mi_lin2;
  BigVganDecoder decoder;
  std::optional<AudioEncoder> encoder;
  std::optional<nn::Linear> enc_lin0;
  std::optional<SLSTM> enc_slstm;
  std::optional<nn::Linear> enc_lin2;
  std::optional<nn::Conv1d> pre_proj;

  mx::array Encode(const mx::array &wav) const;
  mx::array Decode(const mx::array &latent) const;
};

AudioVAE LoadAudioVae(const DotsCheckpoint &checkpoint);

} // namespace dots_tts
