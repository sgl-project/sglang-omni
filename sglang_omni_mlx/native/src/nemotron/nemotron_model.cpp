// SPDX-License-Identifier: Apache-2.0
#include "nemotron/nemotron_model.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "common/fastconformer.h"
#include "common/swift_port.h"

namespace nemotron {

namespace mx = mlx::core;

namespace {

constexpr float kLayerNormEps = 1e-5f;
// Note (Dayuxiaoshui): causal dw-striding pads two frames before and one after
// on both the time and frequency axes, so each output sees only its past.
constexpr int kCausalPadBefore = 2;
constexpr int kCausalPadAfter = 1;
constexpr int kSubsamplingStride = 2;

mx::array Scalar(float value, mx::Dtype dtype) {
  return mx::array(value, dtype);
}

} // namespace

NemotronModel::NemotronModel(const std::filesystem::path &model_directory)
    : NemotronModel(omni_common::ReadJson(model_directory / "config.json"),
                    model_directory) {}

NemotronModel::NemotronModel(const nlohmann::json &config,
                             const std::filesystem::path &model_directory)
    : checkpoint_(omni_common::LoadSafetensors(model_directory), config) {
  const nlohmann::json &encoder = config.at("encoder");
  const nlohmann::json &context = config.at("default_att_context_size");
  config_.feature_count = config.at("preprocessor").at("features").get<int>();
  config_.model_width = encoder.at("d_model").get<int>();
  config_.head_count = encoder.at("n_heads").get<int>();
  config_.layer_count = encoder.at("n_layers").get<int>();
  config_.subsampling_factor = encoder.at("subsampling_factor").get<int>();
  config_.conv_kernel_size = encoder.at("conv_kernel_size").get<int>();
  config_.left_context_frames = context.at(0).get<int>();
  config_.right_context_frames = context.at(1).get<int>();
  config_.prompt_count = config.at("prompt").at("num_prompts").get<int>();
  config_.prompt_indices = config.at("prompt")
                               .at("prompt_dictionary")
                               .get<std::map<std::string, int>>();
  config_.default_language = config.at("default_language").get<std::string>();
  config_.prediction_layer_count =
      config.at("decoder").at("pred_rnn_layers").get<int>();
  config_.vocabulary = config.at("vocabulary").get<std::vector<std::string>>();
  if (config.contains("max_symbols") && !config.at("max_symbols").is_null()) {
    config_.max_symbols_per_frame = config.at("max_symbols").get<int>();
  } else {
  }
  const bool prompts_in_range =
      !config_.prompt_indices.empty() &&
      std::all_of(config_.prompt_indices.begin(), config_.prompt_indices.end(),
                  [&](const auto &entry) {
                    return entry.second >= 0 &&
                           entry.second < config_.prompt_count;
                  });
  // Note (Dayuxiaoshui): positions fill sine and cosine columns in pairs and
  // heads split the width evenly, and the joint's last class is the blank
  // after the vocabulary.
  const int vocabulary_size = static_cast<int>(config_.vocabulary.size());
  if (config_.model_width % 2 != 0 || config_.head_count < 1 ||
      config_.model_width % config_.head_count != 0 ||
      config.at("decoder").at("vocab_size").get<int>() != vocabulary_size ||
      config.at("joint").at("num_classes").get<int>() != vocabulary_size) {
    throw std::runtime_error(
        "Nemotron config: d_model must be even and split across n_heads, and "
        "vocab_size and num_classes must equal the vocabulary's size");
  } else {
  }
  // Note (Dayuxiaoshui): the graph below is the published checkpoint's: 8x
  // causal subsampling, unscaled inputs, bias-free layer-normed convolutions,
  // a ReLU joint and unnormalized features.
  if (config_.subsampling_factor != 8 ||
      encoder.at("conv_context_size").get<std::string>() != "causal" ||
      encoder.at("xscaling").get<bool>() ||
      encoder.at("use_bias").get<bool>() ||
      encoder.at("conv_norm_type").get<std::string>() != "layer_norm" ||
      config.at("joint").at("activation").get<std::string>() != "relu" ||
      config.at("preprocessor").at("normalize").get<std::string>() != "NA" ||
      !prompts_in_range) {
    throw std::runtime_error(
        "only the causal, unnormalized 8x Nemotron streaming checkpoint is "
        "supported");
  } else {
  }
}

mx::array NemotronModel::Conv2d(const mx::array &x, const std::string &prefix,
                                int stride, int groups) const {
  return mx::add(mx::conv2d(x, checkpoint_.Weight(prefix + ".weight"),
                            {stride, stride}, {0, 0}, {1, 1}, groups),
                 checkpoint_.Weight(prefix + ".bias"));
}

mx::array NemotronModel::Subsample(const mx::array &mel) const {
  const auto causal_pad = [](const mx::array &input) {
    return mx::pad(input, {{0, 0},
                           {kCausalPadBefore, kCausalPadAfter},
                           {kCausalPadBefore, kCausalPadAfter},
                           {0, 0}});
  };
  const std::string prefix = "encoder.pre_encode";
  mx::array x = mx::expand_dims(mel, 3);
  x = swift_port::Relu(
      Conv2d(causal_pad(x), prefix + ".conv.0", kSubsamplingStride, 1));
  const int channels = x.shape(3);
  // Note (Dayuxiaoshui): conv.1 and conv.4 are the ReLUs between stages.
  for (const auto &[depthwise, pointwise] :
       {std::pair{".conv.2", ".conv.3"}, std::pair{".conv.5", ".conv.6"}}) {
    x = Conv2d(causal_pad(x), prefix + depthwise, kSubsamplingStride, channels);
    x = swift_port::Relu(Conv2d(x, prefix + pointwise, 1, 1));
  }
  const int frame_count = x.shape(1);
  const int frequency_bins = x.shape(2);
  x = mx::reshape(mx::transpose(x, {0, 1, 3, 2}),
                  {1, frame_count, channels * frequency_bins});
  return checkpoint_.Linear(x, prefix + ".out");
}

mx::array NemotronModel::Encode(const mx::array &x, int chunk_frames,
                                std::vector<LayerCache> &caches) const {
  const int cached = caches.front().attention_inputs.has_value()
                         ? caches.front().attention_inputs->shape(1)
                         : 0;
  // Note (XXXXRT666): one chunk sees the frames cached so far; several share
  // one window length, so a stream's first windows are padded to the full
  // context and the padding is masked.
  const int padding = x.shape(1) > chunk_frames
                          ? std::max(0, config_.left_context_frames - cached)
                          : 0;
  const mx::array positions =
      mx::astype(omni_common::RelativePositionTable(
                     cached + padding + chunk_frames, config_.model_width),
                 x.dtype());
  mx::array hidden = x;
  for (int layer = 0; layer < config_.layer_count; ++layer) {
    hidden =
        Layer(hidden, layer, chunk_frames, padding, positions, caches[layer]);
  }
  return hidden;
}

mx::array NemotronModel::Layer(const mx::array &x, int layer, int chunk_frames,
                               int padding_frames,
                               const mx::array &relative_positions,
                               LayerCache &cache) const {
  const std::string prefix = "encoder.layers." + std::to_string(layer);
  const mx::Dtype dtype = x.dtype();
  const mx::array half = Scalar(0.5f, dtype);
  mx::array hidden = mx::add(
      x, mx::multiply(
             half, omni_common::ConformerFeedForward(
                       checkpoint_,
                       checkpoint_.LayerNorm(x, prefix + ".norm_feed_forward1",
                                             kLayerNormEps),
                       prefix + ".feed_forward1")));

  const mx::array attention_input =
      checkpoint_.LayerNorm(hidden, prefix + ".norm_self_att", kLayerNormEps);
  mx::array keys_values =
      cache.attention_inputs.has_value()
          ? mx::concatenate({*cache.attention_inputs, attention_input}, 1)
          : attention_input;
  if (padding_frames > 0) {
    keys_values = mx::pad(keys_values, {{0, 0}, {padding_frames, 0}, {0, 0}});
  } else {
  }
  hidden = mx::add(hidden, omni_common::ChunkedRelativePositionAttention(
                               checkpoint_, attention_input, keys_values,
                               chunk_frames, padding_frames, relative_positions,
                               prefix + ".self_attn", config_.head_count));
  const int key_count = keys_values.shape(1);
  cache.attention_inputs = mx::slice(
      keys_values,
      {0, std::max(padding_frames, key_count - config_.left_context_frames), 0},
      {1, key_count, config_.model_width});

  const std::string conv_prefix = prefix + ".conv";
  const std::vector<mx::array> halves = mx::split(
      mx::conv1d(
          checkpoint_.LayerNorm(hidden, prefix + ".norm_conv", kLayerNormEps),
          checkpoint_.Weight(conv_prefix + ".pointwise_conv1.weight")),
      2, 2);
  const mx::array gated = mx::multiply(halves[0], mx::sigmoid(halves[1]));
  const int history_frames = config_.conv_kernel_size - 1;
  const mx::array convolution_input = mx::concatenate(
      {cache.convolution_inputs.value_or(
           mx::zeros({1, history_frames, config_.model_width}, gated.dtype())),
       gated},
      1);
  const int input_frames = convolution_input.shape(1);
  cache.convolution_inputs = mx::slice(
      convolution_input, {0, std::max(0, input_frames - history_frames), 0},
      {1, input_frames, config_.model_width});
  mx::array convolved =
      mx::conv1d(convolution_input,
                 checkpoint_.Weight(conv_prefix + ".depthwise_conv.weight"), 1,
                 0, 1, config_.model_width);
  // Note (Dayuxiaoshui): conv_norm_type layer_norm keeps the batch_norm name.
  convolved = swift_port::Silu(checkpoint_.LayerNorm(
      convolved, conv_prefix + ".batch_norm", kLayerNormEps));
  hidden = mx::add(
      hidden,
      mx::conv1d(convolved,
                 checkpoint_.Weight(conv_prefix + ".pointwise_conv2.weight")));

  hidden = mx::add(
      hidden, mx::multiply(half, omni_common::ConformerFeedForward(
                                     checkpoint_,
                                     checkpoint_.LayerNorm(
                                         hidden, prefix + ".norm_feed_forward2",
                                         kLayerNormEps),
                                     prefix + ".feed_forward2")));
  return checkpoint_.LayerNorm(hidden, prefix + ".norm_out", kLayerNormEps);
}

int NemotronModel::PromptIndex(
    const std::optional<std::string> &language) const {
  const auto found =
      config_.prompt_indices.find(language.value_or(config_.default_language));
  if (found != config_.prompt_indices.end()) {
    return found->second;
  } else {
  }
  const auto fallback = config_.prompt_indices.find(config_.default_language);
  return fallback != config_.prompt_indices.end() ? fallback->second : 0;
}

mx::array NemotronModel::ApplyPrompt(const mx::array &encoded,
                                     int prompt_index) const {
  const int frame_count = encoded.shape(1);
  std::vector<float> one_hot(
      static_cast<size_t>(frame_count) * config_.prompt_count, 0.0f);
  for (int frame = 0; frame < frame_count; ++frame) {
    one_hot[static_cast<size_t>(frame) * config_.prompt_count + prompt_index] =
        1.0f;
  }
  const mx::array conditioned = mx::concatenate(
      {encoded, mx::astype(mx::array(one_hot.data(),
                                     {1, frame_count, config_.prompt_count},
                                     mx::float32),
                           encoded.dtype())},
      2);
  return checkpoint_.Linear(
      swift_port::Relu(checkpoint_.Linear(conditioned, "prompt_kernel.0")),
      "prompt_kernel.2");
}

std::pair<mx::array, LstmState>
NemotronModel::Predict(std::optional<int> token,
                       const std::optional<LstmState> &state) const {
  const std::string prefix = "decoder.prediction";
  const int hidden_width =
      checkpoint_.Weight(prefix + ".dec_rnn.lstm.0.Wh").shape(1);
  // Note (Dayuxiaoshui): with no token yet the input is float32 zeros, which
  // promotes that first step's LSTM to float32, as in Swift.
  mx::array output =
      token.has_value()
          ? checkpoint_.Embed(mx::array({*token}, {1, 1}, mx::int32),
                              prefix + ".embed")
          : mx::zeros({1, 1, hidden_width}, mx::float32);
  LstmState next;
  for (int layer = 0; layer < config_.prediction_layer_count; ++layer) {
    const std::string lstm_prefix =
        prefix + ".dec_rnn.lstm." + std::to_string(layer);
    mx::array gates =
        mx::addmm(checkpoint_.Weight(lstm_prefix + ".bias"), output,
                  mx::transpose(checkpoint_.Weight(lstm_prefix + ".Wx")));
    gates = mx::reshape(gates, {1, gates.shape(-1)});
    const omni_common::LstmCell step = omni_common::LstmStep(
        gates,
        state.has_value() ? std::optional<omni_common::LstmCell>(
                                {state->hidden[layer], state->cell[layer]})
                          : std::nullopt,
        checkpoint_.Weight(lstm_prefix + ".Wh"));
    next.hidden.push_back(step.hidden);
    next.cell.push_back(step.cell);
    output = mx::expand_dims(step.hidden, 1);
  }
  return {output, next};
}

mx::array NemotronModel::ProjectFrames(const mx::array &frames) const {
  return checkpoint_.Linear(frames, "joint.enc");
}

mx::array NemotronModel::JointTokens(const mx::array &projected_frames,
                                     const mx::array &prediction) const {
  const mx::array logits = checkpoint_.Linear(
      swift_port::Relu(mx::add(projected_frames,
                               checkpoint_.Linear(prediction, "joint.pred"))),
      "joint.joint_net.2");
  return mx::astype(mx::reshape(mx::argmax(logits, -1), {-1}), mx::int32);
}

} // namespace nemotron
