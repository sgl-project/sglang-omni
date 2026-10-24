// SPDX-License-Identifier: Apache-2.0
// dots.tts patch encoder and flow-matching DiT.
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

mx::array Modulate(const mx::array &x, const mx::array &shift,
                   const mx::array &scale) {
  return mx::add(mx::multiply(x, mx::add(mx::array(1.0f, x.dtype()),
                                         mx::expand_dims(scale, 1))),
                 mx::expand_dims(shift, 1));
}

mx::array TimestepEmbedding(const mx::array &t, int dim, float max_period) {
  const int half = dim / 2;
  const mx::array freqs =
      mx::exp(mx::multiply(mx::array(static_cast<float>(-std::log(max_period))),
                           mx::divide(mx::arange(half, mx::float32),
                                      mx::array(static_cast<float>(half)))));
  const mx::array args =
      mx::multiply(mx::expand_dims(mx::astype(t, mx::float32), -1),
                   mx::expand_dims(freqs, 0));
  std::vector<mx::array> pieces = {mx::cos(args), mx::sin(args)};
  if (dim % 2) {
    pieces.push_back(
        mx::zeros_like(mx::slice(mx::cos(args), {0, 0}, {args.shape(0), 1})));
  }
  return mx::concatenate(pieces, -1);
}

} // namespace

mx::array TimestepEmbedder::operator()(const mx::array &t) const {
  const mx::array emb =
      TimestepEmbedding(t, frequency_embedding_size, max_period);
  // note (guozhihao-224): keep the conditioning path on the flow dtype so the
  // adaLN output and everything after it does not promote a bf16 flow back to
  // fp32 (the cos/sin embedding math above stays fp32 for precision).
  return mx::astype(mlp_fc2(qwen3_asr::Silu(mlp_fc0(emb))), t.dtype());
}

mx::array FinalLayer::operator()(const mx::array &x,
                                 const mx::array &conditioning) const {
  const mx::array mod = adaLN_linear(qwen3_asr::Silu(conditioning));
  const std::vector<mx::array> pieces = mx::split(mod, 2, -1);
  const mx::array normalized = norm(x);
  return linear(Modulate(normalized, pieces[0], pieces[1]));
}

mx::array DiTBlock::operator()(const mx::array &x,
                               const mx::array &conditioning,
                               const std::optional<mx::array> &mask) const {
  const mx::array mod = adaLN_linear(qwen3_asr::Silu(conditioning));
  const std::vector<mx::array> chunks = mx::split(mod, 6, -1);
  const mx::array &shift_attn = chunks[0];
  const mx::array &scale_attn = chunks[1];
  const mx::array &gate_attn = chunks[2];
  const mx::array &shift_ffn = chunks[3];
  const mx::array &scale_ffn = chunks[4];
  const mx::array &gate_ffn = chunks[5];
  mx::array out =
      mx::add(x, mx::multiply(mx::expand_dims(gate_attn, 1),
                              attn(Modulate(norm1(x), shift_attn, scale_attn),
                                   std::nullopt, std::nullopt, mask)));
  return mx::add(out,
                 mx::multiply(mx::expand_dims(gate_ffn, 1),
                              ffn(Modulate(norm2(out), shift_ffn, scale_ffn))));
}

mx::array DiT::operator()(const mx::array &x, const mx::array &timesteps,
                          const std::optional<mx::array> &attn_mask,
                          const std::optional<mx::array> &speaker_conditioning,
                          const std::optional<mx::array> &duration) const {
  mx::array conditioning = time_embedder(timesteps);

  if (duration.has_value() && duration_embedder.has_value()) {
    conditioning = mx::add(conditioning, (*duration_embedder)(*duration));
  }
  if (speaker_conditioning.has_value()) {
    conditioning = mx::add(conditioning, *speaker_conditioning);
  }
  mx::array out = input_layer(x);
  for (const DiTBlock &block : blocks) {
    out = block(out, conditioning, attn_mask);
  }
  return output_layer(out, conditioning);
}

mx::array FmSolverStep(const DiT &dit, const nn::Linear &coordinate_proj,
                       const mx::array &t, const mx::array &z,
                       const mx::array &input_sequence,
                       const mx::array &unconditional_history,
                       const std::optional<mx::array> &attn_mask,
                       const mx::array &speaker_conditioning,
                       float guidance_scale) {
  const int batch_size = input_sequence.shape(0);
  const int patch_size = z.shape(1);
  const int latent_start = input_sequence.shape(1) - patch_size;

  const mx::array z_proj = coordinate_proj(z);
  const mx::array conditional_sequence =
      mx::concatenate({mx::slice(input_sequence, {0, 0, 0},
                                 {input_sequence.shape(0), latent_start,
                                  input_sequence.shape(2)}),
                       z_proj},
                      1);
  const mx::array unconditional_sequence =
      mx::concatenate({mx::slice(unconditional_history, {0, 0, 0},
                                 {unconditional_history.shape(0), latent_start,
                                  unconditional_history.shape(2)}),
                       z_proj},
                      1);
  const mx::array concatenated_sequence =
      mx::concatenate({conditional_sequence, unconditional_sequence}, 0);
  const mx::array shared_timestep = mx::broadcast_to(mx::reshape(t, {1}), {2});
  const mx::array batched_speaker_conditioning = mx::concatenate(
      {speaker_conditioning, mx::zeros_like(speaker_conditioning)}, 0);

  mx::array velocity = dit(concatenated_sequence, shared_timestep, attn_mask,
                           batched_speaker_conditioning);
  velocity =
      mx::slice(velocity, {0, latent_start, 0},
                {velocity.shape(0), velocity.shape(1), velocity.shape(2)});
  const mx::array conditional_velocity = mx::slice(
      velocity, {0, 0, 0}, {batch_size, velocity.shape(1), velocity.shape(2)});
  const mx::array unconditional_velocity =
      mx::slice(velocity, {batch_size, 0, 0},
                {velocity.shape(0), velocity.shape(1), velocity.shape(2)});
  return mx::add(
      conditional_velocity,
      mx::multiply(mx::array(guidance_scale, velocity.dtype()),
                   mx::subtract(conditional_velocity, unconditional_velocity)));
}

mx::array FlowSolver::Denoise(const mx::array &input_sequence,
                              const mx::array &unconditional_sequence,
                              const std::optional<mx::array> &attn_mask,
                              const mx::array &speaker_conditioning,
                              float guidance_scale, int num_steps,
                              const mx::array &noise) const {
  // note (guozhihao-224): a non-positive step count would make dt infinite and
  // turn the solver into a no-op that pipes raw noise through the vocoder.
  if (num_steps < 1) {
    throw std::invalid_argument("dots.tts num_steps must be >= 1");
  }
  // note (guozhihao-224): run the flow solver in bf16 for Metal's bf16 fast
  // paths; the Euler integration amplifies fp32 rounding into audible noise,
  // but a few ulps of bf16 metal rounding stay below that bar (parity is
  // relaxed here anyway). The result is cast back to fp32 so the AR history
  // stays fp32.
  mx::array z = mx::astype(noise, mx::bfloat16);
  const mx::array input_bf16 = mx::astype(input_sequence, mx::bfloat16);
  const mx::array unconditional_bf16 =
      mx::astype(unconditional_sequence, mx::bfloat16);
  const mx::array speaker_conditioning_bf16 =
      mx::astype(speaker_conditioning, mx::bfloat16);
  const float dt = 1.0f / num_steps;
  for (int k = 0; k < num_steps; ++k) {
    const mx::array t = mx::array(static_cast<float>(k * dt), mx::bfloat16);
    const mx::array v = FmSolverStep(*dit, coordinate_proj, t, z, input_bf16,
                                     unconditional_bf16, attn_mask,
                                     speaker_conditioning_bf16, guidance_scale);
    z = mx::add(z, mx::multiply(mx::array(dt, mx::bfloat16), v));
  }
  return mx::astype(z, mx::float32);
}

mx::array FlowSolver::DenoiseMeanFlow(const mx::array &input_sequence,
                                      const std::optional<mx::array> &attn_mask,
                                      const mx::array &speaker_conditioning,
                                      int num_steps,
                                      const mx::array &noise) const {
  // note (guozhihao-224): same bf16 treatment as Denoise, but a single branch
  // with no classifier-free guidance, and each step also feeds the segment
  // duration to the DiT's duration embedder.
  if (num_steps < 1) {
    throw std::invalid_argument("dots.tts num_steps must be >= 1");
  }
  mx::array z = mx::astype(noise, mx::bfloat16);
  const mx::array input_bf16 = mx::astype(input_sequence, mx::bfloat16);
  const mx::array speaker_conditioning_bf16 =
      mx::astype(speaker_conditioning, mx::bfloat16);
  const int patch_size = noise.shape(1);
  const int latent_start = input_sequence.shape(1) - patch_size;
  const float dt = 1.0f / num_steps;
  for (int k = 0; k < num_steps; ++k) {
    const mx::array t = mx::array(static_cast<float>(k * dt), mx::bfloat16);
    const mx::array step_dt = mx::array(dt, mx::bfloat16);
    const mx::array z_proj = coordinate_proj(z);
    const mx::array sequence = mx::concatenate(
        {mx::slice(input_bf16, {0, 0, 0},
                   {input_bf16.shape(0), latent_start, input_bf16.shape(2)}),
         z_proj},
        1);
    mx::array velocity =
        (*dit)(sequence, t, attn_mask, speaker_conditioning_bf16, step_dt);
    velocity =
        mx::slice(velocity, {0, latent_start, 0},
                  {velocity.shape(0), velocity.shape(1), velocity.shape(2)});
    z = mx::add(z, mx::multiply(step_dt, velocity));
  }
  return mx::astype(z, mx::float32);
}

FlowHead LoadFlowHead(const DotsCheckpoint &checkpoint) {
  const nlohmann::json &cfg = checkpoint.config();
  const bool meanflow =
      cfg.contains("meanflow") && cfg.at("meanflow").value("enabled", false);
  const int flow_hidden_size = cfg.at("DiT").at("hidden_size").get<int>();
  const int patch_layer_count =
      cfg.at("PatchEncoder").at("num_layers").get<int>();
  const int patch_hidden = cfg.at("PatchEncoder").at("hidden_size").get<int>();
  const int patch_heads = cfg.at("PatchEncoder").at("num_heads").get<int>();
  const int dit_layers = cfg.at("DiT").at("num_layers").get<int>();
  const int dit_heads = cfg.at("DiT").at("num_heads").get<int>();

  const auto linear = [&](const std::string &name) {
    return nn::Linear(
        checkpoint.Weight(name + ".weight"),
        checkpoint.Has(name + ".bias")
            ? std::optional<mx::array>(checkpoint.Weight(name + ".bias"))
            : std::nullopt);
  };

  nn::Linear hidden_proj = linear("hidden_proj");
  nn::Linear latent_proj = linear("latent_proj");
  nn::Linear xvec_linear = linear("xvec_proj.0");
  mx::array xvec_norm_weight = checkpoint.Weight("xvec_proj.1.weight");
  mx::array xvec_norm_bias = checkpoint.Weight("xvec_proj.1.bias");

  nn::Linear input_layer = linear("velocity_field_predictor.input_layer");
  TimestepEmbedder time_embedder(
      linear("velocity_field_predictor.time_embedder.mlp.0"),
      linear("velocity_field_predictor.time_embedder.mlp.2"));
  std::optional<TimestepEmbedder> duration_embedder;
  if (meanflow &&
      checkpoint.Has("velocity_field_predictor.duration_embedder.mlp.0"
                     ".weight")) {
    duration_embedder.emplace(
        linear("velocity_field_predictor.duration_embedder.mlp.0"),
        linear("velocity_field_predictor.duration_embedder.mlp.2"));
  }
  std::vector<DiTBlock> blocks;
  for (int i = 0; i < dit_layers; ++i) {
    const std::string bp =
        "velocity_field_predictor.blocks." + std::to_string(i);
    nn::MultiHeadAttention attn(flow_hidden_size, dit_heads, /*qkv_bias=*/false,
                                /*qk_norm=*/true, nn::NormLayer::RMSNorm,
                                /*rotary=*/true,
                                cfg.at("DiT").value("rotary_theta", 10000.0f));
    attn.q_proj = linear(bp + ".attn.q_proj");
    attn.k_proj = linear(bp + ".attn.k_proj");
    attn.v_proj = linear(bp + ".attn.v_proj");
    attn.o_proj = linear(bp + ".attn.o_proj");
    attn.q_norm_weight = checkpoint.Weight(bp + ".attn.q_norm.weight");
    attn.k_norm_weight = checkpoint.Weight(bp + ".attn.k_norm.weight");
    nn::Mlp ffn(linear(bp + ".ffn.fc1"), linear(bp + ".ffn.fc2"),
                /*activation=*/nn::Activation::GeluTanh);
    nn::Linear adaLN = linear(bp + ".adaLN_modulation.1");
    blocks.emplace_back(std::move(attn), std::move(ffn), std::move(adaLN),
                        flow_hidden_size);
  }
  FinalLayer output_layer(
      linear("velocity_field_predictor.output_layer.adaLN_modulation.1"),
      nn::LayerNorm(flow_hidden_size),
      linear("velocity_field_predictor.output_layer.linear"));
  DiT dit{std::move(input_layer), std::move(time_embedder), std::move(blocks),
          std::move(output_layer), std::move(duration_embedder)};
  nn::Linear coordinate_proj = linear("coordinate_proj");

  nn::Conv1d ds_proj(
      Conv1dWeight(checkpoint.Weight("patch_encoder.ds_proj.weight"),
                   mx::float32),
      checkpoint.Weight("patch_encoder.ds_proj.bias"),
      /*causal=*/true,
      /*stride=*/2);
  nn::Linear in_proj = linear("patch_encoder.in_proj");
  std::vector<EncoderLayer> encoder_layers;
  for (int i = 0; i < patch_layer_count; ++i) {
    const std::string lp = "patch_encoder.encoder.layers." + std::to_string(i);
    nn::RMSNorm attn_norm(checkpoint.Weight(lp + ".attn_norm.weight"));
    nn::RMSNorm ffn_norm(checkpoint.Weight(lp + ".ffn_norm.weight"));
    nn::MultiHeadAttention attn(patch_hidden, patch_heads, /*qkv_bias=*/false,
                                /*qk_norm=*/false, nn::NormLayer::RMSNorm,
                                /*rotary=*/false, 10000.0f);
    attn.q_proj = linear(lp + ".attn.q_proj");
    attn.k_proj = linear(lp + ".attn.k_proj");
    attn.v_proj = linear(lp + ".attn.v_proj");
    attn.o_proj = linear(lp + ".attn.o_proj");
    nn::Mlp ffn(linear(lp + ".ffn.fc1"), linear(lp + ".ffn.fc2"),
                /*activation=*/nn::Activation::Silu);
    encoder_layers.emplace_back(std::move(attn_norm), std::move(attn),
                                std::move(ffn_norm), std::move(ffn));
  }
  nn::Linear out_proj = linear("patch_encoder.out_proj");
  VAESemanticEncoder patch_encoder{std::move(ds_proj), std::move(in_proj),
                                   std::move(encoder_layers),
                                   std::move(out_proj), /*out_ds_rate=*/2};

  return FlowHead{std::move(hidden_proj),     std::move(latent_proj),
                  std::move(xvec_linear),     std::move(xvec_norm_weight),
                  std::move(xvec_norm_bias),  std::move(dit),
                  std::move(coordinate_proj), std::move(patch_encoder)};
}
namespace {

mx::array CausalAttentionMask(int query_len, int key_len, int offset) {
  std::vector<char> keep(static_cast<size_t>(query_len) * key_len, 0);
  for (int q = 0; q < query_len; ++q) {
    for (int k = 0; k < key_len && k <= offset + q; ++k) {
      keep[static_cast<size_t>(q) * key_len + k] = 1;
    }
  }
  return mx::array(keep.data(), {query_len, key_len}, mx::bool_);
}

} // namespace

mx::array SnakeBeta(const mx::array &x, const mx::array &alpha,
                    const mx::array &beta) {
  const mx::array x32 = mx::astype(x, mx::float32);
  const mx::array a = mx::exp(mx::astype(alpha, mx::float32));
  const mx::array b = mx::exp(mx::astype(beta, mx::float32));
  const mx::array s = mx::sin(mx::multiply(x32, a));
  return mx::add(
      x32, mx::multiply(mx::divide(mx::ones_like(b),
                                   mx::add(b, mx::array(1e-9f, mx::float32))),
                        mx::multiply(s, s)));
}

mx::array AliasFreeResampler::operator()(const mx::array &x) const {
  const int c = x.shape(2);
  const int k_up = up_filter.shape(1);
  const int k_down = down_filter.shape(1);
  const mx::array x32 = mx::astype(x, mx::float32);
  mx::array up =
      mx::multiply(mx::array(static_cast<float>(ratio)),
                   mx::conv_transpose1d(x32, up_filter, ratio, /*padding=*/0,
                                        /*dilation=*/1, /*output_padding=*/0,
                                        /*groups=*/c));
  up = mx::slice(up, {0, 0, 0},
                 {up.shape(0), up.shape(1) - (k_up - ratio), up.shape(2)});
  const mx::array act = SnakeBeta(up, alpha, beta);
  const mx::array padded = nn::ReplicatePad(act, k_down - 1, 0);
  return mx::conv1d(padded, down_filter, ratio, /*padding=*/0, /*dilation=*/1,
                    /*groups=*/c);
}

mx::array AMPBlock1::operator()(const mx::array &x) const {
  mx::array out = x;
  const int count = static_cast<int>(convs1.size());
  for (int j = 0; j < count; ++j) {
    mx::array xt = acts[2 * j](out);
    xt = convs1[j](xt);
    xt = acts[2 * j + 1](xt);
    xt = convs2[j](xt);
    out = mx::add(xt, out);
  }
  return out;
}

mx::array SLSTM::operator()(const mx::array &x) const {
  const mx::array residual = x;
  const int batch = x.shape(0);
  const int time = x.shape(1);
  const int hidden = x.shape(2);
  mx::array current = x;

  for (int layer = 0; layer < num_layers; ++layer) {
    const mx::array bias = mx::add(bias_ih[layer], bias_hh[layer]);
    const mx::array gates_in =
        mx::add(mx::matmul(current, mx::transpose(weight_ih[layer])),
                mx::broadcast_to(bias, {batch, time, 4 * hidden}));
    std::optional<qwen3_asr::LstmCell> state;
    std::vector<mx::array> outputs;
    outputs.reserve(time);

    for (int step = 0; step < time; ++step) {
      const mx::array gates = mx::squeeze(
          mx::slice(gates_in, {0, step, 0}, {batch, step + 1, 4 * hidden}), 1);
      state = qwen3_asr::LstmStep(gates, state, weight_hh[layer]);
      outputs.push_back(state->hidden);
    }
    current = mx::stack(outputs, 1);
  }
  return mx::add(current, residual);
}

mx::array ResStack::operator()(const mx::array &x) const {
  mx::array out = x;
  const int count = static_cast<int>(convs1.size());
  for (int j = 0; j < count; ++j) {
    mx::array h = mx::maximum(out, mx::multiply(out, mx::array(0.01f)));
    h = convs1[j](h);
    h = mx::maximum(h, mx::multiply(h, mx::array(0.01f)));
    h = convs2[j](h);
    out = mx::add(out, h);
  }
  return out;
}

mx::array EncoderLayer::operator()(const mx::array &x,
                                   const mx::array &mask) const {
  mx::array h = attn_norm(x);
  h = attn(h, std::nullopt, std::nullopt, mask);
  h = mx::add(x, h);
  mx::array h2 = ffn_norm(h);
  h2 = ffn(h2);
  return mx::add(h, h2);
}

mx::array VAESemanticEncoder::operator()(const mx::array &x) const {
  const int time = x.shape(1);
  assert(time % (2 * out_ds_rate) == 0);

  const int reduced = time / 2;
  const mx::array ones = mx::ones({reduced, reduced}, mx::bool_);
  const mx::array mask = mx::tril(ones);
  mx::array out = ds_proj(x);
  out = in_proj(out);
  for (const EncoderLayer &layer : layers) {
    out = layer(out, mask);
  }

  const int b = out.shape(0);
  const int sd = out.shape(1);
  const int h = out.shape(2);
  const int s = sd / out_ds_rate;
  out = mx::reshape(out, {b, s, out_ds_rate, h});
  out = mx::reshape(out, {b, s, out_ds_rate * h});
  return out_proj(out);
}

void VAESemanticEncoder::Seed(SemanticEncoderCache &cache,
                              const mx::array &x) const {
  mx::array rows = in_proj(ds_proj(x));
  cache.layer_normed.clear();
  cache.layer_normed.reserve(layers.size());
  for (size_t i = 0; i < layers.size(); ++i) {
    const EncoderLayer &layer = layers[i];
    cache.layer_normed.push_back(layer.attn_norm(rows));
    const int len = cache.layer_normed[i].shape(1);
    const mx::array h = mx::add(
        rows, layer.attn(cache.layer_normed[i], std::nullopt, std::nullopt,
                         CausalAttentionMask(len, len, /*offset=*/0)));
    rows = mx::add(h, layer.ffn(layer.ffn_norm(h)));
  }
  mx::eval(cache.layer_normed);
}

mx::array VAESemanticEncoder::Append(SemanticEncoderCache &cache,
                                     const mx::array &x) const {
  const mx::array rows = in_proj(ds_proj(x));
  const int r = rows.shape(1);
  const int prev =
      cache.layer_normed.empty() ? 0 : cache.layer_normed[0].shape(1);
  mx::array new_rows = mx::slice(rows, {0, prev, 0}, {1, r, rows.shape(2)});
  const bool first_call = cache.layer_normed.empty();
  if (first_call) {
    cache.layer_normed.reserve(layers.size());
  }
  for (size_t i = 0; i < layers.size(); ++i) {
    const EncoderLayer &layer = layers[i];
    const mx::array norm_new = layer.attn_norm(new_rows);
    if (first_call) {
      cache.layer_normed.push_back(norm_new);
    } else {
      cache.layer_normed[i] =
          mx::concatenate({cache.layer_normed[i], norm_new}, 1);
    }
    const int n = norm_new.shape(1);
    const mx::array h = mx::add(
        new_rows,
        layer.attn(
            norm_new, cache.layer_normed[i], cache.layer_normed[i],
            CausalAttentionMask(n, cache.layer_normed[i].shape(1), prev)));
    new_rows = mx::add(h, layer.ffn(layer.ffn_norm(h)));
  }

  mx::eval(new_rows);
  mx::eval(cache.layer_normed);
  const int h = new_rows.shape(2);
  const int take = std::min(out_ds_rate, new_rows.shape(1));
  mx::array grouped = mx::slice(new_rows, {0, new_rows.shape(1) - take, 0},
                                {1, new_rows.shape(1), h});
  grouped = mx::reshape(grouped, {new_rows.shape(0), 1, take, h});
  grouped = mx::reshape(grouped, {new_rows.shape(0), 1, take * h});
  return out_proj(grouped);
}

} // namespace dots_tts
