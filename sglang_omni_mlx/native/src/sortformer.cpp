// SPDX-License-Identifier: Apache-2.0
#include "sortformer.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "sortformer_checkpoint.h"

namespace sortformer {

namespace mx = mlx::core;

namespace {

// Note (Jiaxin Deng): mlx-swift casts a scalar operand to the array's dtype
// (float16 stays float16); a C++ float32 scalar would promote it.
mx::array Scalar(float value, const mx::array &like) {
  return mx::array(value, like.dtype());
}

std::vector<mx::array> ReluGraph(const std::vector<mx::array> &inputs) {
  return {mx::maximum(inputs[0], Scalar(0.0f, inputs[0]))};
}

std::vector<mx::array> SiluGraph(const std::vector<mx::array> &inputs) {
  return {mx::multiply(inputs[0], mx::sigmoid(inputs[0]))};
}

// Note (Jiaxin Deng): compiled shapeless to match MLXNN relu and silu.
mx::array Relu(const mx::array &x) {
  static const auto compiled = mx::compile(ReluGraph, true);
  return compiled({x})[0];
}

mx::array Silu(const mx::array &x) {
  static const auto compiled = mx::compile(SiluGraph, true);
  return compiled({x})[0];
}

int SubsampledLength(int frame_count, int stage_count) {
  int length = frame_count;
  for (int stage = 0; stage < stage_count; ++stage) {
    length = static_cast<int>(
                 std::floor((static_cast<float>(length) - 1.0f) / 2.0f)) +
             1;
  }
  return length;
}

// Note (Jiaxin Deng): computed in float32 and cast to the activations' dtype,
// as Swift does.
mx::array RelativePositionalEncoding(int length, int width, mx::Dtype dtype) {
  std::vector<float> positions;
  for (int position = length - 1; position >= -(length - 1); --position) {
    positions.push_back(static_cast<float>(position));
  }
  std::vector<float> dimensions;
  for (int dimension = 0; dimension < width; dimension += 2) {
    dimensions.push_back(static_cast<float>(dimension));
  }
  const int position_count = static_cast<int>(positions.size());
  const mx::array position_array(positions.data(), {position_count},
                                 mx::float32);
  const mx::array dimension_array(
      dimensions.data(), {static_cast<int>(dimensions.size())}, mx::float32);
  const mx::array divisor = mx::exp(mx::multiply(
      dimension_array, mx::array(static_cast<float>(
                           -std::log(10000.0) / static_cast<double>(width)))));
  const mx::array angles = mx::multiply(mx::expand_dims(position_array, 1),
                                        mx::expand_dims(divisor, 0));
  const mx::array encoding =
      mx::reshape(mx::stack({mx::sin(angles), mx::cos(angles)}, -1),
                  {position_count, width});
  return mx::astype(mx::expand_dims(encoding, 0), dtype);
}

mx::array RelativeShift(const mx::array &x) {
  const int batch = x.shape(0);
  const int heads = x.shape(1);
  const int query_length = x.shape(2);
  const int position_length = x.shape(3);
  mx::array padded = mx::pad(x, {{0, 0}, {0, 0}, {0, 0}, {1, 0}});
  padded =
      mx::reshape(padded, {batch, heads, position_length + 1, query_length});
  padded = mx::slice(padded, {0, 0, 1, 0},
                     {batch, heads, position_length + 1, query_length});
  return mx::reshape(padded, {batch, heads, query_length, position_length});
}

} // namespace

SortformerModel::SortformerModel(const std::filesystem::path &model_directory)
    : config_(LoadConfig(model_directory)), features_(config_.processor),
      weights_(LoadCheckpoint(model_directory, config_)) {}

const mx::array &SortformerModel::Weight(const std::string &name) const {
  const auto found = weights_.find(name);
  if (found == weights_.end()) {
    throw std::runtime_error("Sortformer checkpoint is missing " + name);
  } else {
  }
  return found->second;
}

float SortformerModel::frame_duration() const {
  return static_cast<float>(config_.processor.hop_length *
                            config_.fc_encoder.subsampling_factor) /
         static_cast<float>(config_.processor.sampling_rate);
}

mx::array SortformerModel::Linear(const mx::array &x,
                                  const std::string &prefix) const {
  // Note (Jiaxin Deng): addmm with bias, matmul without, as MLXNN Linear.
  const auto bias = weights_.find(prefix + ".bias");
  if (bias != weights_.end()) {
    return mx::addmm(bias->second, x,
                     mx::transpose(Weight(prefix + ".weight")));
  } else {
    return mx::matmul(x, mx::transpose(Weight(prefix + ".weight")));
  }
}

mx::array SortformerModel::LayerNorm(const mx::array &x,
                                     const std::string &prefix,
                                     float eps) const {
  return mx::fast::layer_norm(x, Weight(prefix + ".weight"),
                              Weight(prefix + ".bias"), eps);
}

mx::array SortformerModel::PreEncode(const mx::array &features) const {
  const FastConformerConfig &fc = config_.fc_encoder;
  const int stride = fc.subsampling_conv_stride;
  const int padding = (fc.subsampling_conv_kernel_size - 1) / 2;
  const auto conv = [&](const mx::array &x, const std::string &name,
                        int conv_stride, int conv_padding, int groups) {
    const std::string prefix = "fc_encoder.subsampling." + name;
    return mx::add(mx::conv2d(x, Weight(prefix + ".weight"),
                              {conv_stride, conv_stride},
                              {conv_padding, conv_padding}, {1, 1}, groups),
                   Weight(prefix + ".bias"));
  };
  const int channels = fc.subsampling_conv_channels;
  mx::array h = mx::expand_dims(mx::transpose(features, {0, 2, 1}), -1);
  h = Relu(conv(h, "layers_0", stride, padding, 1));
  h = Relu(conv(conv(h, "layers_2", stride, padding, channels), "layers_3", 1,
                0, 1));
  h = Relu(conv(conv(h, "layers_5", stride, padding, channels), "layers_6", 1,
                0, 1));
  const int batch = h.shape(0);
  const int time = h.shape(1);
  const int frequency = h.shape(2);
  const int channel_count = h.shape(3);
  h = mx::reshape(mx::transpose(h, {0, 1, 3, 2}),
                  {batch, time, channel_count * frequency});
  return Linear(h, "fc_encoder.subsampling.linear");
}

mx::array
SortformerModel::RelativePositionAttention(const mx::array &x,
                                           const mx::array &position_embedding,
                                           const std::string &prefix) const {
  const int heads = config_.fc_encoder.num_attention_heads;
  const int head_dim = config_.fc_encoder.hidden_size / heads;
  const int batch = x.shape(0);
  const auto project = [&](const mx::array &input, const std::string &name,
                           int input_batch) {
    return mx::transpose(mx::reshape(Linear(input, prefix + "." + name),
                                     {input_batch, -1, heads, head_dim}),
                         {0, 2, 1, 3});
  };
  const mx::array q = project(x, "q_proj", batch);
  const mx::array k = project(x, "k_proj", batch);
  const mx::array v = project(x, "v_proj", batch);
  const mx::array q_by_time = mx::transpose(q, {0, 2, 1, 3});
  const mx::array p = project(position_embedding, "relative_k_proj", 1);
  const mx::array q_with_bias_u = mx::transpose(
      mx::add(q_by_time, Weight(prefix + ".bias_u")), {0, 2, 1, 3});
  const mx::array q_with_bias_v = mx::transpose(
      mx::add(q_by_time, Weight(prefix + ".bias_v")), {0, 2, 1, 3});
  const mx::array matrix_ac =
      mx::matmul(q_with_bias_u, mx::transpose(k, {0, 1, 3, 2}));
  mx::array matrix_bd =
      mx::matmul(q_with_bias_v, mx::transpose(p, {0, 1, 3, 2}));
  matrix_bd = RelativeShift(matrix_bd);
  matrix_bd = mx::slice(matrix_bd, {0, 0, 0, 0},
                        {matrix_bd.shape(0), matrix_bd.shape(1),
                         matrix_bd.shape(2), matrix_ac.shape(3)});
  mx::array scores = mx::add(matrix_ac, matrix_bd);
  scores = mx::divide(scores,
                      Scalar(std::sqrt(static_cast<float>(head_dim)), scores));
  const mx::array attention = mx::softmax(scores, -1);
  const mx::array out =
      mx::reshape(mx::transpose(mx::matmul(attention, v), {0, 2, 1, 3}),
                  {batch, -1, heads * head_dim});
  return Linear(out, prefix + ".o_proj");
}

mx::array
SortformerModel::ConformerConvolution(const mx::array &x,
                                      const std::string &prefix) const {
  const auto conv = [&](const mx::array &input, const std::string &name,
                        int padding, int groups) {
    return mx::add(mx::conv1d(input, Weight(prefix + "." + name + ".weight"), 1,
                              padding, 1, groups),
                   Weight(prefix + "." + name + ".bias"));
  };
  mx::array h = conv(x, "pointwise_conv1", 0, 1);
  const std::vector<mx::array> halves = mx::split(h, 2, -1);
  h = mx::multiply(halves[0], mx::sigmoid(halves[1]));
  h = conv(h, "depthwise_conv", (config_.fc_encoder.conv_kernel_size - 1) / 2,
           config_.fc_encoder.hidden_size);
  const mx::array &running_var = Weight(prefix + ".norm.running_var");
  h = mx::add(
      mx::multiply(
          mx::divide(
              mx::subtract(h, Weight(prefix + ".norm.running_mean")),
              mx::sqrt(mx::add(running_var, Scalar(1e-5f, running_var)))),
          Weight(prefix + ".norm.weight")),
      Weight(prefix + ".norm.bias"));
  h = Silu(h);
  return conv(h, "pointwise_conv2", 0, 1);
}

mx::array SortformerModel::ConformerLayer(const mx::array &x,
                                          const mx::array &position_embedding,
                                          int layer) const {
  constexpr float kFeedForwardFactor = 0.5f;
  constexpr float kEps = 1e-5f;
  const std::string prefix = "fc_encoder.layers." + std::to_string(layer);
  const auto feed_forward = [&](const mx::array &input,
                                const std::string &name) {
    return Linear(Silu(Linear(input, prefix + "." + name + ".linear1")),
                  prefix + "." + name + ".linear2");
  };
  mx::array residual = x;
  mx::array h = feed_forward(LayerNorm(x, prefix + ".norm_feed_forward1", kEps),
                             "feed_forward1");
  residual = mx::add(residual, mx::multiply(h, Scalar(kFeedForwardFactor, h)));
  h = RelativePositionAttention(
      LayerNorm(residual, prefix + ".norm_self_att", kEps), position_embedding,
      prefix + ".self_attn");
  residual = mx::add(residual, h);
  h = ConformerConvolution(LayerNorm(residual, prefix + ".norm_conv", kEps),
                           prefix + ".conv");
  residual = mx::add(residual, h);
  h = feed_forward(LayerNorm(residual, prefix + ".norm_feed_forward2", kEps),
                   "feed_forward2");
  residual = mx::add(residual, mx::multiply(h, Scalar(kFeedForwardFactor, h)));
  return LayerNorm(residual, prefix + ".norm_out", kEps);
}

mx::array SortformerModel::FastConformer(const mx::array &embeddings) const {
  mx::array x = embeddings;
  if (config_.fc_encoder.scale_input) {
    const float scale = static_cast<float>(
        std::sqrt(static_cast<double>(config_.fc_encoder.hidden_size)));
    x = mx::multiply(x, Scalar(scale, x));
  } else {
  }
  const mx::array position_embedding = RelativePositionalEncoding(
      x.shape(1), config_.fc_encoder.hidden_size, x.dtype());
  for (int layer = 0; layer < config_.fc_encoder.num_hidden_layers; ++layer) {
    x = ConformerLayer(x, position_embedding, layer);
  }
  return x;
}

mx::array SortformerModel::TransformerLayer(const mx::array &x,
                                            const mx::array &mask,
                                            int layer) const {
  const TransformerConfig &tf = config_.tf_encoder;
  const std::string prefix = "tf_encoder.layers." + std::to_string(layer);
  const int batch = x.shape(0);
  const int length = x.shape(1);
  const int heads = tf.encoder_attention_heads;
  const int head_dim = tf.d_model / heads;
  const float scale = std::pow(static_cast<float>(head_dim), -0.5f);
  const auto project = [&](const std::string &name) {
    return mx::transpose(mx::reshape(Linear(x, prefix + ".self_attn." + name),
                                     {batch, -1, heads, head_dim}),
                         {0, 2, 1, 3});
  };
  const mx::array q = project("q_proj");
  const mx::array k = project("k_proj");
  const mx::array v = project("v_proj");
  mx::array scores = mx::matmul(mx::multiply(q, Scalar(scale, q)),
                                mx::transpose(k, {0, 1, 3, 2}));
  scores = mx::add(scores, mask);
  const mx::array attention = mx::softmax(scores, -1);
  const mx::array attended =
      mx::reshape(mx::transpose(mx::matmul(attention, v), {0, 2, 1, 3}),
                  {batch, length, tf.d_model});
  mx::array h = mx::add(x, Linear(attended, prefix + ".self_attn.out_proj"));
  h = LayerNorm(h, prefix + ".self_attn_layer_norm", tf.layer_norm_eps);
  const mx::array residual = h;
  h = Linear(Relu(Linear(h, prefix + ".fc1")), prefix + ".fc2");
  h = mx::add(residual, h);
  return LayerNorm(h, prefix + ".final_layer_norm", tf.layer_norm_eps);
}

mx::array
SortformerModel::SpeakerProbabilities(const mx::array &embeddings) const {
  const int length = embeddings.shape(1);
  // Note (Jiaxin Deng): Swift still adds a float32 mask, promoting float16
  // attention scores to float32 from the first layer on.
  const mx::array valid =
      mx::less(mx::expand_dims(mx::arange(length, mx::int32), 0),
               mx::reshape(mx::array(length, mx::int32), {1, 1}));
  const mx::array additive_mask =
      mx::expand_dims(mx::multiply(mx::subtract(mx::array(1.0f),
                                                mx::astype(valid, mx::float32)),
                                   mx::array(-1e4f)),
                      {1, 2});
  const mx::array positions = mx::arange(length, mx::int32);
  mx::array x =
      mx::add(embeddings, mx::take(Weight("tf_encoder.embed_positions.weight"),
                                   positions, 0));
  for (int layer = 0; layer < config_.tf_encoder.encoder_layers; ++layer) {
    x = TransformerLayer(x, additive_mask, layer);
  }
  mx::array h = Relu(x);
  h = Relu(Linear(h, "sortformer_modules.first_hidden_to_hidden"));
  h = mx::sigmoid(Linear(h, "sortformer_modules.single_hidden_to_spks"));
  return mx::multiply(h, mx::expand_dims(valid, 2));
}

StreamingState SortformerModel::InitStreamingState() const {
  const int width = config_.fc_encoder.hidden_size;
  const int speakers = config_.modules.num_speakers;
  return StreamingState{mx::zeros({1, 0, width}, mx::float32),
                        mx::zeros({1, 0, speakers}, mx::float32),
                        mx::zeros({1, 0, width}, mx::float32),
                        mx::zeros({1, 0, speakers}, mx::float32),
                        0,
                        mx::zeros({1, width}, mx::float32),
                        mx::zeros({1}, mx::float32)};
}

FeedResult SortformerModel::Feed(const std::vector<float> &samples,
                                 StreamingState &state,
                                 const FeedOptions &options) const {
  const ModulesConfig &modules = config_.modules;
  const float frame_duration_seconds = frame_duration();
  const float chunk_time_offset =
      static_cast<float>(state.frames_processed) * frame_duration_seconds;
  const mx::array features = features_(samples);

  // Note (Jiaxin Deng): pre-encoded in the checkpoint dtype, as in Swift.
  const mx::Dtype model_dtype =
      Weight("sortformer_modules.encoder_proj.weight").dtype();
  mx::array chunk_embeddings = PreEncode(mx::astype(features, model_dtype));
  const int chunk_length = SubsampledLength(features.shape(2), 3);
  const int width = chunk_embeddings.shape(2);
  chunk_embeddings =
      mx::slice(chunk_embeddings, {0, 0, 0}, {1, chunk_length, width});

  const int cache_length = state.spkcache_length();
  const int fifo_length = state.fifo_length();
  const int left_context = modules.use_aosc ? modules.chunk_left_context : 0;
  std::vector<mx::array> parts;
  if (cache_length > 0) {
    parts.push_back(state.spkcache);
  } else {
  }
  if (fifo_length > 0) {
    parts.push_back(state.fifo);
  } else {
  }
  int left_length = 0;
  if (left_context > 0 && fifo_length > 0) {
    left_length = std::min(left_context, fifo_length);
    parts.push_back(mx::slice(state.fifo, {0, fifo_length - left_length, 0},
                              {1, fifo_length, width}));
  } else {
  }
  parts.push_back(chunk_embeddings);
  // Note (Jiaxin Deng): float32 once the state holds frames, as in Swift.
  const mx::array all_embeddings = mx::concatenate(parts, 1);

  const mx::array encoded =
      Linear(FastConformer(all_embeddings), "sortformer_modules.encoder_proj");
  const mx::array all_predictions = SpeakerProbabilities(encoded);
  const int speakers = all_predictions.shape(2);
  const int chunk_start = cache_length + fifo_length + left_length;
  const mx::array chunk_predictions =
      mx::slice(all_predictions, {0, chunk_start, 0},
                {1, chunk_start + chunk_length, speakers});
  const mx::array cache_predictions =
      mx::slice(all_predictions, {0, 0, 0}, {1, cache_length, speakers});
  const mx::array fifo_predictions =
      mx::slice(all_predictions, {0, cache_length, 0},
                {1, cache_length + fifo_length, speakers});
  mx::eval({chunk_predictions, chunk_embeddings, cache_predictions,
            fifo_predictions});

  if (cache_length > 0) {
    state.spkcache_preds = cache_predictions;
  } else {
  }
  const mx::array previous_fifo_predictions =
      fifo_length > 0 ? fifo_predictions : state.fifo_preds;
  state.fifo = mx::concatenate({state.fifo, chunk_embeddings}, 1);
  state.fifo_preds =
      mx::concatenate({previous_fifo_predictions, chunk_predictions}, 1);
  mx::eval({state.fifo, state.fifo_preds});
  state.frames_processed += chunk_length;

  FeedResult result;
  const mx::array host_predictions =
      mx::reshape(mx::astype(chunk_predictions, mx::float32), {-1});
  mx::eval(host_predictions);
  const float *data = host_predictions.data<float>();
  result.probabilities.assign(data, data + host_predictions.size());
  result.frame_count = chunk_length;
  result.speaker_count = speakers;
  result.segments = ProbabilitiesToSegments(
      result.probabilities, chunk_length, speakers, frame_duration_seconds,
      options.threshold, options.min_duration, options.merge_gap);
  for (Segment &segment : result.segments) {
    segment.start = segment.start + chunk_time_offset;
    segment.end = segment.end + chunk_time_offset;
  }
  MaybeCompressState(state, options.spkcache_max, options.fifo_max, modules);
  return result;
}

} // namespace sortformer
