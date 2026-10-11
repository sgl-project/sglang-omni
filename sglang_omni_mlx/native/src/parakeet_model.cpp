// SPDX-License-Identifier: Apache-2.0
#include "parakeet_model.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <regex>
#include <sstream>
#include <stdexcept>

#include "swift_port.h"

namespace parakeet {

namespace mx = mlx::core;

namespace {

constexpr float kLayerNormEps = 1e-5f;
constexpr float kBatchNormEps = 1e-5f;
constexpr float kFeatureStdGuard = 1e-5f;
constexpr int kSubsamplingStride = 2;
constexpr int kSubsamplingPadding = 1;
constexpr float kPositionTimescale = 10000.0f;

// Note (Dayuxiaoshui): NeMo exports Infinity and NaN, which JSON lacks; the
// Swift port reads them as null.
nlohmann::json ReadConfig(const std::filesystem::path &path) {
  std::ifstream stream(path, std::ios::binary);
  if (!stream) {
    throw std::runtime_error("cannot read " + path.string());
  } else {
  }
  std::ostringstream contents;
  contents << stream.rdbuf();
  return nlohmann::json::parse(
      std::regex_replace(contents.str(), std::regex("-?Infinity|NaN"), "null"));
}

qwen3_asr::WeightMap
BFloat16Weights(const std::filesystem::path &model_directory) {
  qwen3_asr::WeightMap weights = qwen3_asr::LoadSafetensors(model_directory);
  for (auto &[name, weight] : weights) {
    if (mx::issubdtype(weight.dtype(), mx::floating)) {
      weight = mx::astype(weight, mx::bfloat16);
    } else {
    }
  }
  return weights;
}

} // namespace

ParakeetModel::ParakeetModel(const std::filesystem::path &model_directory)
    : ParakeetModel(ReadConfig(model_directory / "config.json"),
                    model_directory) {}

ParakeetModel::ParakeetModel(const nlohmann::json &config,
                             const std::filesystem::path &model_directory)
    : features_(front_end_),
      checkpoint_(BFloat16Weights(model_directory), config) {
  const nlohmann::json &preprocessor = config.at("preprocessor");
  const nlohmann::json &encoder = config.at("encoder");
  const nlohmann::json &decoding = config.at("decoding");
  config_.feature_count = preprocessor.at("features").get<int>();
  config_.model_width = encoder.at("d_model").get<int>();
  config_.head_count = encoder.at("n_heads").get<int>();
  config_.layer_count = encoder.at("n_layers").get<int>();
  config_.subsampling_factor = encoder.at("subsampling_factor").get<int>();
  config_.prediction_layer_count =
      config.at("decoder").at("prednet").at("pred_rnn_layers").get<int>();
  config_.vocabulary =
      config.at("joint").at("vocabulary").get<std::vector<std::string>>();
  config_.durations = decoding.at("durations").get<std::vector<int>>();
  const nlohmann::json max_symbols =
      decoding.value("greedy", nlohmann::json::object())
          .value("max_symbols", nlohmann::json());
  config_.max_symbols_per_frame =
      max_symbols.is_number_integer() ? max_symbols.get<int>() : 0;
  // Note (Dayuxiaoshui): the Swift port takes Int(seconds * rate) in float.
  const int sample_rate = preprocessor.at("sample_rate").get<int>();
  front_end_.feature_size = config_.feature_count;
  front_end_.sampling_rate = sample_rate;
  front_end_.fft_size = preprocessor.at("n_fft").get<int>();
  front_end_.window_length =
      static_cast<int>(preprocessor.at("window_size").get<float>() *
                       static_cast<float>(sample_rate));
  front_end_.hop_length =
      static_cast<int>(preprocessor.at("window_stride").get<float>() *
                       static_cast<float>(sample_rate));
  front_end_.preemphasis = preprocessor.value("preemph", 0.97f);
  features_ = sortformer::FeatureExtractor(front_end_);
  prediction_width_ =
      checkpoint_.Weight("decoder.prediction.dec_rnn.lstm.0.Wh").shape(1);
  compiled_decode_ = mx::compile([this](const std::vector<mx::array> &inputs) {
    return DecodeGraph(inputs);
  });
  // Note (Dayuxiaoshui): the graph below is the published TDT checkpoint's:
  // per-feature normalized Hann features, 8x dw-striding subsampling,
  // unscaled relative-position attention, batch-normed convolutions and a
  // ReLU joint.
  if (decoding.at("model_type").get<std::string>() != "tdt" ||
      preprocessor.at("normalize").get<std::string>() != "per_feature" ||
      preprocessor.at("window").get<std::string>() != "hann" ||
      preprocessor.value("pad_to", 0) != 0 ||
      encoder.at("subsampling").get<std::string>() != "dw_striding" ||
      config_.subsampling_factor != 8 ||
      encoder.at("causal_downsampling").get<bool>() ||
      encoder.at("self_attention_model").get<std::string>() != "rel_pos" ||
      encoder.at("xscaling").get<bool>() ||
      encoder.at("conv_norm_type").get<std::string>() != "batch_norm" ||
      config.at("joint").at("jointnet").at("activation").get<std::string>() !=
          "relu" ||
      config_.head_count < 1 || config_.model_width % 2 != 0 ||
      config_.model_width % config_.head_count != 0 ||
      config_.durations.empty()) {
    throw std::runtime_error(
        "only the per-feature, dw-striding, rel_pos Parakeet TDT checkpoint "
        "is supported");
  } else {
  }
  // Note (Jiaxin Deng): a blank with duration 0 leaves the decoder state
  // unchanged, so greedy TDT decoding ends only with a symbol limit.
  if (config_.max_symbols_per_frame < 1) {
    throw std::runtime_error(
        "decoding.greedy.max_symbols must be a positive integer");
  } else if (std::any_of(config_.durations.begin(), config_.durations.end(),
                         [](int duration) { return duration < 0; })) {
    throw std::runtime_error("decoding.durations must be nonnegative");
  } else if (checkpoint_.Weight("joint.joint_net.2.bias").shape(0) !=
             blank_id() + 1 + static_cast<int>(config_.durations.size())) {
    throw std::runtime_error("the joint's outputs must be the vocabulary, "
                             "the blank and one per decoding.durations entry");
  } else if (checkpoint_.Weight("decoder.prediction.embed.weight").shape(0) <=
             blank_id()) {
    throw std::runtime_error(
        "the prediction embedding must have a row for the blank");
  } else {
  }
}

double ParakeetModel::FrameSeconds(int frames) const {
  return static_cast<double>(frames * config_.subsampling_factor *
                             front_end_.hop_length) /
         front_end_.sampling_rate;
}

mx::array ParakeetModel::Features(const std::vector<float> &samples) const {
  // Note (Dayuxiaoshui): a contiguous [frames, features] copy, so the
  // reductions over frames run the kernels, and the order, they run in Swift.
  const mx::array mel =
      mx::contiguous(mx::transpose(mx::squeeze(features_(samples), 0)));
  const int frame_count = mel.shape(0);
  const mx::array mean = mx::mean(mel, 0, true);
  const mx::array centered = mx::subtract(mel, mean);
  const mx::array variance =
      mx::divide(mx::sum(mx::square(centered), 0, true),
                 mx::array(static_cast<float>(std::max(frame_count - 1, 1))));
  const mx::array normalized = mx::divide(
      centered, mx::add(mx::sqrt(variance), mx::array(kFeatureStdGuard)));
  return mx::expand_dims(normalized, 0);
}

mx::array ParakeetModel::Conv2d(const mx::array &x, const std::string &prefix,
                                int groups) const {
  const int kernel = checkpoint_.Weight(prefix + ".weight").shape(1);
  const int stride = kernel == 1 ? 1 : kSubsamplingStride;
  const int padding = kernel == 1 ? 0 : kSubsamplingPadding;
  return mx::add(mx::conv2d(x, checkpoint_.Weight(prefix + ".weight"),
                            {stride, stride}, {padding, padding}, {1, 1},
                            groups),
                 checkpoint_.Weight(prefix + ".bias"));
}

mx::array ParakeetModel::FeedForward(const mx::array &x,
                                     const std::string &prefix) const {
  return checkpoint_.Linear(
      swift_port::Silu(checkpoint_.Linear(x, prefix + ".linear1")),
      prefix + ".linear2");
}

mx::array ParakeetModel::Attention(const mx::array &x,
                                   const mx::array &positions,
                                   const std::string &prefix) const {
  const int time = x.shape(1);
  const int position_count = positions.shape(1);
  const int head_count = config_.head_count;
  const int head_width = config_.model_width / head_count;
  const float scale = std::pow(static_cast<float>(head_width), -0.5f);
  const auto heads = [&](const mx::array &projected, int length) {
    return mx::transpose(
        mx::reshape(projected, {1, length, head_count, head_width}),
        {0, 2, 1, 3});
  };
  const mx::array query_heads =
      mx::reshape(checkpoint_.Linear(x, prefix + ".linear_q"),
                  {1, time, head_count, head_width});
  const mx::Dtype dtype = query_heads.dtype();
  const mx::array content_queries = mx::transpose(
      mx::add(query_heads,
              mx::astype(checkpoint_.Weight(prefix + ".pos_bias_u"), dtype)),
      {0, 2, 1, 3});
  const mx::array position_queries = mx::transpose(
      mx::add(query_heads,
              mx::astype(checkpoint_.Weight(prefix + ".pos_bias_v"), dtype)),
      {0, 2, 1, 3});
  const mx::array keys =
      heads(checkpoint_.Linear(x, prefix + ".linear_k"), time);
  const mx::array values =
      heads(checkpoint_.Linear(x, prefix + ".linear_v"), time);
  const mx::array position_keys = heads(
      checkpoint_.Linear(positions, prefix + ".linear_pos"), position_count);
  mx::array position_scores =
      mx::matmul(position_queries, mx::swapaxes(position_keys, -2, -1));
  // Note (Dayuxiaoshui): the relative shift: a left zero column, then rows
  // reread one step along, aligns score (i, j) with position i - j.
  position_scores = mx::pad(position_scores, {{0, 0}, {0, 0}, {0, 0}, {1, 0}});
  position_scores = mx::reshape(
      mx::slice(mx::reshape(position_scores,
                            {1, head_count, position_count + 1, time}),
                {0, 0, 1, 0}, {1, head_count, position_count + 1, time}),
      {1, head_count, time, position_count});
  position_scores = mx::multiply(
      mx::slice(position_scores, {0, 0, 0, 0}, {1, head_count, time, time}),
      mx::array(scale, position_scores.dtype()));
  const mx::array attended = mx::fast::scaled_dot_product_attention(
      content_queries, keys, values, scale, "array", position_scores);
  return checkpoint_.Linear(mx::reshape(mx::transpose(attended, {0, 2, 1, 3}),
                                        {1, time, config_.model_width}),
                            prefix + ".linear_out");
}

mx::array ParakeetModel::Convolution(const mx::array &x,
                                     const std::string &prefix) const {
  const std::vector<mx::array> halves = mx::split(
      mx::conv1d(x, checkpoint_.Weight(prefix + ".pointwise_conv1.weight")), 2,
      2);
  const mx::array gated = mx::multiply(halves[0], mx::sigmoid(halves[1]));
  const mx::array &depthwise =
      checkpoint_.Weight(prefix + ".depthwise_conv.weight");
  mx::array convolved =
      mx::conv1d(gated, depthwise, 1, (depthwise.shape(1) - 1) / 2, 1,
                 config_.model_width);
  // Note (Dayuxiaoshui): batch norm with its running statistics, in MLXNN's
  // order.
  const std::string norm = prefix + ".batch_norm";
  convolved = mx::multiply(
      mx::subtract(convolved, checkpoint_.Weight(norm + ".running_mean")),
      mx::rsqrt(mx::add(checkpoint_.Weight(norm + ".running_var"),
                        mx::array(kBatchNormEps, convolved.dtype()))));
  convolved =
      mx::add(mx::multiply(checkpoint_.Weight(norm + ".weight"), convolved),
              checkpoint_.Weight(norm + ".bias"));
  return mx::conv1d(swift_port::Silu(convolved),
                    checkpoint_.Weight(prefix + ".pointwise_conv2.weight"));
}

mx::array ParakeetModel::Encode(const mx::array &features) const {
  const std::string pre_encode = "encoder.pre_encode";
  mx::array x = mx::expand_dims(mx::astype(features, mx::bfloat16), 3);
  x = swift_port::Relu(Conv2d(x, pre_encode + ".conv.0", 1));
  const int channels = x.shape(3);
  for (const auto &[depthwise, pointwise] :
       {std::pair{".conv.2", ".conv.3"}, std::pair{".conv.5", ".conv.6"}}) {
    x = swift_port::Relu(Conv2d(Conv2d(x, pre_encode + depthwise, channels),
                                pre_encode + pointwise, 1));
  }
  const int time = x.shape(1);
  const int frequency_bins = x.shape(2);
  x = checkpoint_.Linear(mx::reshape(mx::transpose(x, {0, 1, 3, 2}),
                                     {1, time, channels * frequency_bins}),
                         pre_encode + ".out");

  // Note (Dayuxiaoshui): each position row depends only on its position, so
  // building the rows this length needs equals slicing the Swift port's
  // 5000-position table; float math in its order keeps the values.
  const int width = config_.model_width;
  const int row_count = 2 * time - 1;
  const float log_timescale =
      static_cast<float>(std::log(static_cast<double>(kPositionTimescale))) /
      static_cast<float>(width);
  std::vector<float> table(static_cast<size_t>(row_count) * width);
  for (int row = 0; row < row_count; ++row) {
    const float position = static_cast<float>(time - 1 - row);
    for (int column = 0; column < width; column += 2) {
      const float angle =
          position * std::exp(-static_cast<float>(column) * log_timescale);
      table[static_cast<size_t>(row) * width + column] = std::sin(angle);
      table[static_cast<size_t>(row) * width + column + 1] = std::cos(angle);
    }
  }
  const mx::array positions = mx::astype(
      mx::array(table.data(), {1, row_count, width}, mx::float32), x.dtype());

  const mx::array half(0.5f, x.dtype());
  for (int layer = 0; layer < config_.layer_count; ++layer) {
    const std::string prefix = "encoder.layers." + std::to_string(layer);
    const auto norm = [&](const mx::array &input, const std::string &name) {
      return checkpoint_.LayerNorm(input, prefix + "." + name, kLayerNormEps);
    };
    x = mx::add(x, mx::multiply(half, FeedForward(norm(x, "norm_feed_forward1"),
                                                  prefix + ".feed_forward1")));
    x = mx::add(x, Attention(norm(x, "norm_self_att"), positions,
                             prefix + ".self_attn"));
    x = mx::add(x, Convolution(norm(x, "norm_conv"), prefix + ".conv"));
    x = mx::add(x, mx::multiply(half, FeedForward(norm(x, "norm_feed_forward2"),
                                                  prefix + ".feed_forward2")));
    x = norm(x, "norm_out");
  }
  return x;
}

std::pair<mx::array, mx::array> ParakeetModel::InitialState() const {
  const mx::array zeros = mx::zeros(
      {config_.prediction_layer_count, 1, prediction_width_}, mx::bfloat16);
  return {zeros, zeros};
}

std::vector<mx::array>
ParakeetModel::DecodeGraph(const std::vector<mx::array> &inputs) const {
  const mx::array &frame = inputs[0];
  const mx::array &token = inputs[1];
  const mx::array &hidden = inputs[2];
  const mx::array &cell = inputs[3];
  const std::string prediction = "decoder.prediction";
  const mx::array embedded = checkpoint_.Embed(token, prediction + ".embed");
  mx::array output = mx::where(
      mx::expand_dims(mx::equal(token, mx::array(blank_id(), mx::int32)), 2),
      mx::zeros_like(embedded), embedded);
  std::vector<mx::array> hidden_states;
  std::vector<mx::array> cell_states;
  for (int layer = 0; layer < config_.prediction_layer_count; ++layer) {
    const std::string lstm =
        prediction + ".dec_rnn.lstm." + std::to_string(layer);
    const auto layer_state = [&](const mx::array &state) {
      return mx::squeeze(
          mx::slice(state, {layer, 0, 0}, {layer + 1, 1, prediction_width_}),
          0);
    };
    // Note (Dayuxiaoshui): MLXNN's LSTM over one step, from the given state.
    mx::array gates =
        mx::addmm(checkpoint_.Weight(lstm + ".bias"), output,
                  mx::transpose(checkpoint_.Weight(lstm + ".Wx")));
    gates = mx::addmm(
        mx::squeeze(mx::slice(gates, {0, 0, 0}, {1, 1, gates.shape(2)}), 1),
        layer_state(hidden), mx::transpose(checkpoint_.Weight(lstm + ".Wh")));
    const std::vector<mx::array> pieces = mx::split(gates, 4, -1);
    const mx::array next_cell =
        mx::add(mx::multiply(mx::sigmoid(pieces[1]), layer_state(cell)),
                mx::multiply(mx::sigmoid(pieces[0]), mx::tanh(pieces[2])));
    const mx::array next_hidden =
        mx::multiply(mx::sigmoid(pieces[3]), mx::tanh(next_cell));
    hidden_states.push_back(next_hidden);
    cell_states.push_back(next_cell);
    output = mx::expand_dims(next_hidden, 1);
  }
  const mx::array combined =
      mx::add(mx::expand_dims(checkpoint_.Linear(frame, "joint.enc"), 2),
              mx::expand_dims(checkpoint_.Linear(output, "joint.pred"), 1));
  const mx::array logits = mx::reshape(
      checkpoint_.Linear(swift_port::Relu(combined), "joint.joint_net.2"),
      {-1});
  const int class_count = blank_id() + 1;
  return {
      mx::stack({mx::astype(mx::argmax(mx::slice(logits, {0}, {class_count})),
                            mx::int32),
                 mx::astype(mx::argmax(mx::slice(logits, {class_count},
                                                 {logits.shape(0)})),
                            mx::int32)}),
      mx::stack(hidden_states, 0), mx::stack(cell_states, 0)};
}

DecoderStep ParakeetModel::Decode(const mx::array &frame, int token,
                                  const mx::array &hidden,
                                  const mx::array &cell) const {
  const std::vector<mx::array> outputs = compiled_decode_(
      {frame, mx::array({token}, {1, 1}, mx::int32), hidden, cell});
  mx::eval(outputs);
  const int32_t *decisions = outputs[0].data<int32_t>();
  return {decisions[0], decisions[1], outputs[1], outputs[2]};
}

} // namespace parakeet
