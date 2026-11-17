// SPDX-License-Identifier: Apache-2.0
#include "common/layers.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "yaml-cpp/yaml.h"

namespace omni_common {

namespace mx = mlx::core;

namespace {

nlohmann::json YamlToJson(const YAML::Node &node) {
  if (node.IsMap()) {
    nlohmann::json object = nlohmann::json::object();
    for (const auto &entry : node) {
      object[entry.first.as<std::string>()] = YamlToJson(entry.second);
    }
    return object;
  } else if (node.IsSequence()) {
    nlohmann::json array = nlohmann::json::array();
    for (const auto &element : node) {
      array.push_back(YamlToJson(element));
    }
    return array;
  } else if (!node.IsScalar()) {
    return nullptr;
  } else if (node.Tag() == "!") {
    // A quoted scalar is a string whatever it spells.
    return node.Scalar();
  } else {
  }
  // Note (XXXXRT666): PyYAML's safe loader, which OmegaConf reads config.yaml
  // with, takes these as booleans; yaml-cpp's own decoding adds y, n and any
  // casing.
  static const std::set<std::string> true_words{
      "true", "True", "TRUE", "yes", "Yes", "YES", "on", "On", "ON"};
  static const std::set<std::string> false_words{
      "false", "False", "FALSE", "no", "No", "NO", "off", "Off", "OFF"};
  long long integer = 0;
  double number = 0.0;
  if (true_words.contains(node.Scalar())) {
    return true;
  } else if (false_words.contains(node.Scalar())) {
    return false;
  } else if (YAML::convert<long long>::decode(node, integer)) {
    return integer;
  } else if (YAML::convert<double>::decode(node, number)) {
    return number;
  } else {
    return node.Scalar();
  }
}

std::vector<mx::array> GeluGraph(const std::vector<mx::array> &inputs) {
  // Note (Jiaxin Deng): mlx.nn.gelu's op order, kept for bit identity.
  // Note (Dayuxiaoshui): constants take x's dtype as Python scalars do, so a
  // half-precision encoder is not promoted to float32.
  const mx::array &x = inputs[0];
  const mx::Dtype dtype = x.dtype();
  return {mx::divide(
      mx::multiply(
          x, mx::add(mx::array(1.0f, dtype),
                     mx::erf(mx::divide(
                         x, mx::array(static_cast<float>(M_SQRT2), dtype))))),
      mx::array(2.0f, dtype))};
}

std::vector<mx::array> TanhGeluGraph(const std::vector<mx::array> &inputs) {
  const mx::array &x = inputs[0];
  const mx::Dtype dtype = x.dtype();
  const mx::array inner = mx::multiply(
      mx::array(static_cast<float>(std::sqrt(2.0 / M_PI)), dtype),
      mx::add(x, mx::multiply(mx::array(0.044715f, dtype),
                              mx::power(x, mx::array(3.0f, dtype)))));
  return {mx::multiply(mx::multiply(mx::array(0.5f, dtype), x),
                       mx::add(mx::array(1.0f, dtype), mx::tanh(inner)))};
}

std::vector<mx::array> SiluGraph(const std::vector<mx::array> &inputs) {
  return {mx::multiply(inputs[0], mx::sigmoid(inputs[0]))};
}

std::vector<mx::array> SwigluGraph(const std::vector<mx::array> &inputs) {
  return {mx::multiply(SiluGraph({inputs[0]})[0], inputs[1])};
}

std::vector<mx::array> SnakeGraph(const std::vector<mx::array> &inputs) {
  const mx::array &x = inputs[0];
  const mx::array &alpha = inputs[1];
  const mx::array &beta = inputs[2];
  return {mx::add(x, mx::divide(mx::square(mx::sin(mx::multiply(alpha, x))),
                                mx::add(beta, Scalar(1e-9f, beta))))};
}

std::vector<mx::array>
LogScaleSnakeGraph(const std::vector<mx::array> &inputs) {
  return SnakeGraph({inputs[0], mx::exp(inputs[1]), mx::exp(inputs[2])});
}

} // namespace

mx::array Scalar(float value, mx::Dtype dtype) {
  return mx::array(value, dtype);
}

mx::array Scalar(float value, const mx::array &like) {
  return Scalar(value, like.dtype());
}

// Note (Jiaxin Deng): compiled shapeless as mlx.nn does, so each activation
// is one fused kernel instead of one per elementwise op.
mx::array Gelu(const mx::array &x) {
  static const auto compiled = mx::compile(GeluGraph, true);
  return compiled({x})[0];
}

mx::array TanhGelu(const mx::array &x) {
  static const auto compiled = mx::compile(TanhGeluGraph, true);
  return compiled({x})[0];
}

mx::array Silu(const mx::array &x) {
  static const auto compiled = mx::compile(SiluGraph, true);
  return compiled({x})[0];
}

// Note (XXXXRT666): one kernel instead of Silu's and the multiply's, which
// takes a dependent dispatch per layer off a decode step.
mx::array Swiglu(const mx::array &gate, const mx::array &up) {
  static const auto compiled = mx::compile(SwigluGraph, true);
  return compiled({gate, up})[0];
}

mx::array Snake(const mx::array &x, const mx::array &alpha,
                const mx::array &beta, bool log_scale) {
  static const auto compiled = mx::compile(SnakeGraph, true);
  static const auto compiled_log_scale = mx::compile(LogScaleSnakeGraph, true);
  return (log_scale ? compiled_log_scale : compiled)({x, alpha, beta})[0];
}

mx::array RouteToExperts(const mx::array &hidden, const mx::array &indices,
                         const ExpertRunner &run_experts) {
  const int tokens = hidden.shape(0), width = hidden.shape(1);
  const int experts_per_token = indices.shape(1);
  // Note (XXXXRT666): A decode token's experts are all distinct, so sorting
  // leaves gather_mm nothing to group.
  if (tokens > 1) {
    const mx::array flat_indices = mx::reshape(indices, {-1});
    const mx::array order = mx::argsort(flat_indices);
    const mx::array token_indices =
        mx::floor_divide(order, mx::array(experts_per_token, mx::uint32));
    const mx::array routed =
        run_experts(mx::expand_dims(mx::take(hidden, token_indices, 0), 1),
                    mx::take(flat_indices, order, 0), true);
    const mx::array inverse_order = mx::put_along_axis(
        mx::zeros(order.shape(), mx::uint32), order,
        mx::arange(tokens * experts_per_token, mx::uint32), 0);
    return mx::reshape(mx::take(routed, inverse_order, 0),
                       {tokens, experts_per_token, routed.shape(-1)});
  } else {
    return mx::squeeze(
        run_experts(mx::reshape(hidden, {tokens, 1, 1, width}), indices, false),
        -2);
  }
}

std::string ReadFile(const std::filesystem::path &path) {
  std::ifstream stream(path, std::ios::binary);
  if (!stream) {
    throw std::runtime_error("cannot read " + path.string());
  } else {
  }
  std::ostringstream contents;
  contents << stream.rdbuf();
  return contents.str();
}

nlohmann::json ReadJson(const std::filesystem::path &path) {
  std::ifstream stream(path);
  if (!stream) {
    throw std::runtime_error("cannot read " + path.string());
  } else {
  }
  return nlohmann::json::parse(stream);
}

nlohmann::json ReadYaml(const std::filesystem::path &path) {
  std::ifstream stream(path);
  if (!stream) {
    throw std::runtime_error("cannot read " + path.string());
  } else {
  }
  return YamlToJson(YAML::Load(stream));
}

WeightMap LoadSafetensors(const std::filesystem::path &model_directory) {
  std::vector<std::filesystem::path> weight_files;
  for (const auto &entry :
       std::filesystem::directory_iterator(model_directory)) {
    if (entry.path().extension() == ".safetensors") {
      weight_files.push_back(entry.path());
    } else {
    }
  }
  if (weight_files.empty()) {
    throw std::runtime_error("no .safetensors files in " +
                             model_directory.string());
  } else {
  }
  std::sort(weight_files.begin(), weight_files.end());
  WeightMap weights;
  for (const auto &path : weight_files) {
    auto [loaded, metadata] = mx::load_safetensors(path.string());
    for (auto &[name, array] : loaded) {
      weights.insert_or_assign(name, array);
    }
  }
  return weights;
}

Checkpoint::Checkpoint(WeightMap weights, const nlohmann::json &config)
    : weights_(std::move(weights)) {
  if (config.contains("quantization")) {
    const nlohmann::json &quantization = config.at("quantization");
    quantization_ = {quantization.at("group_size").get<int>(),
                     quantization.at("bits").get<int>(),
                     quantization.value("mode", std::string("affine"))};
    for (const auto &[prefix, layer_quantization] : quantization.items()) {
      if (layer_quantization.is_object()) {
        quantization_by_prefix_.insert_or_assign(
            prefix, QuantizationConfig{
                        layer_quantization.at("group_size").get<int>(),
                        layer_quantization.at("bits").get<int>(),
                        layer_quantization.value("mode", quantization_.mode)});
      } else {
      }
    }
  } else {
  }
  std::vector<mx::array> parameters;
  parameters.reserve(weights_.size());
  for (const auto &[name, array] : weights_)
    parameters.push_back(array);
  mx::eval(parameters);
}

const mx::array &Checkpoint::Weight(const std::string &name) const {
  const auto found = weights_.find(name);
  if (found == weights_.end()) {
    throw std::runtime_error("checkpoint is missing " + name);
  } else {
  }
  return found->second;
}

bool Checkpoint::Has(const std::string &name) const {
  return weights_.count(name) > 0;
}

void Checkpoint::RequireShape(const std::string &name,
                              const mx::Shape &shape) const {
  if (Weight(name).shape() != shape) {
    throw std::invalid_argument("checkpoint tensor has an unexpected shape: " +
                                name);
  } else {
  }
}

const QuantizationConfig &
Checkpoint::Quantization(const std::string &prefix) const {
  const auto found = quantization_by_prefix_.find(prefix);
  if (found != quantization_by_prefix_.end()) {
    return found->second;
  } else {
    return quantization_;
  }
}

mx::array Checkpoint::Linear(const mx::array &x,
                             const std::string &prefix) const {
  if (Has(prefix + ".scales")) {
    const std::optional<mx::array> biases =
        Has(prefix + ".biases")
            ? std::optional<mx::array>(Weight(prefix + ".biases"))
            : std::nullopt;
    const QuantizationConfig &quantization = Quantization(prefix);
    mx::array y = mx::quantized_matmul(
        x, Weight(prefix + ".weight"), Weight(prefix + ".scales"), biases, true,
        quantization.group_size, quantization.bits, quantization.mode);
    if (Has(prefix + ".bias")) {
      return mx::add(y, Weight(prefix + ".bias"));
    } else {
      return y;
    }
  } else if (Has(prefix + ".bias")) {
    return mx::addmm(Weight(prefix + ".bias"), x,
                     mx::transpose(Weight(prefix + ".weight")));
  } else {
    return mx::matmul(x, mx::transpose(Weight(prefix + ".weight")));
  }
}

mx::array Checkpoint::Conv1d(const mx::array &x, const std::string &prefix,
                             int stride, int padding, int dilation,
                             int groups) const {
  const mx::array y = mx::conv1d(x, Weight(prefix + ".weight"), stride, padding,
                                 dilation, groups);
  if (Has(prefix + ".bias")) {
    return mx::add(y, Weight(prefix + ".bias"));
  } else {
    return y;
  }
}

mx::array Checkpoint::Conv2d(const mx::array &x, const std::string &prefix,
                             int stride, int padding, int groups) const {
  const mx::array y =
      mx::conv2d(x, Weight(prefix + ".weight"), {stride, stride},
                 {padding, padding}, {1, 1}, groups);
  if (Has(prefix + ".bias")) {
    return mx::add(y, Weight(prefix + ".bias"));
  } else {
    return y;
  }
}

mx::array Checkpoint::LayerNorm(const mx::array &x, const std::string &prefix,
                                float eps) const {
  return mx::fast::layer_norm(x, Weight(prefix + ".weight"),
                              Weight(prefix + ".bias"), eps);
}

mx::array Checkpoint::RmsNorm(const mx::array &x, const std::string &prefix,
                              float eps) const {
  return mx::fast::rms_norm(x, Weight(prefix + ".weight"), eps);
}

mx::array Checkpoint::Embed(const mx::array &ids,
                            const std::string &prefix) const {
  if (Has(prefix + ".scales")) {
    const std::optional<mx::array> biases =
        Has(prefix + ".biases") ? std::optional<mx::array>(mx::take(
                                      Weight(prefix + ".biases"), ids, 0))
                                : std::nullopt;
    const QuantizationConfig &quantization = Quantization(prefix);
    return mx::dequantize(mx::take(Weight(prefix + ".weight"), ids, 0),
                          mx::take(Weight(prefix + ".scales"), ids, 0), biases,
                          quantization.group_size, quantization.bits,
                          quantization.mode);
  } else {
    return mx::take(Weight(prefix + ".weight"), ids, 0);
  }
}

mx::array Checkpoint::TiedProjection(const mx::array &x,
                                     const std::string &prefix) const {
  if (Has(prefix + ".scales")) {
    const std::optional<mx::array> biases =
        Has(prefix + ".biases")
            ? std::optional<mx::array>(Weight(prefix + ".biases"))
            : std::nullopt;
    const QuantizationConfig &quantization = Quantization(prefix);
    return mx::quantized_matmul(
        x, Weight(prefix + ".weight"), Weight(prefix + ".scales"), biases, true,
        quantization.group_size, quantization.bits, quantization.mode);
  } else {
    return mx::matmul(x, mx::transpose(Weight(prefix + ".weight")));
  }
}

mx::array RelativeShift(const mx::array &scores, int key_count) {
  const int batch = scores.shape(0);
  const int heads = scores.shape(1);
  const int query_length = scores.shape(2);
  const int position_length = scores.shape(3);
  if (position_length < query_length + key_count - 1) {
    throw std::invalid_argument(
        "relative shift needs a position for every query-key distance");
  } else {
  }
  // Note (XXXXRT666): the padded reshape as a view: row i starts query_length
  // - 1 - i columns in, so no copy of the scores is made.
  return mx::as_strided(
      scores, {batch, heads, query_length, key_count},
      {static_cast<int64_t>(heads) * query_length * position_length,
       static_cast<int64_t>(query_length) * position_length,
       position_length - 1, 1},
      query_length - 1);
}

void FoldBatchNorms(WeightMap &weights, const std::string &norm_name,
                    float eps) {
  const std::string suffix = "." + norm_name + ".running_mean";
  std::vector<std::string> modules;
  for (const auto &[name, weight] : weights) {
    if (name.ends_with(suffix)) {
      modules.push_back(name.substr(0, name.size() - suffix.size()));
    } else {
    }
  }
  const auto float_weight = [&](const std::string &name) {
    return mx::astype(weights.at(name), mx::float32);
  };
  for (const std::string &module : modules) {
    const std::string norm = module + "." + norm_name;
    const std::string convolution = module + ".depthwise_conv";
    const mx::array scale =
        mx::multiply(float_weight(norm + ".weight"),
                     mx::rsqrt(mx::add(float_weight(norm + ".running_var"),
                                       mx::array(eps))));
    const mx::array convolution_bias = weights.contains(convolution + ".bias")
                                           ? float_weight(convolution + ".bias")
                                           : mx::zeros_like(scale);
    const mx::array &kernel = weights.at(convolution + ".weight");
    mx::Shape scale_shape(kernel.ndim(), 1);
    scale_shape[0] = kernel.shape(0);
    const mx::Dtype dtype = kernel.dtype();
    weights.insert_or_assign(
        convolution + ".weight",
        mx::astype(mx::multiply(mx::astype(kernel, mx::float32),
                                mx::reshape(scale, scale_shape)),
                   dtype));
    weights.insert_or_assign(
        convolution + ".bias",
        mx::astype(
            mx::add(
                mx::multiply(mx::subtract(convolution_bias,
                                          float_weight(norm + ".running_mean")),
                             scale),
                float_weight(norm + ".bias")),
            dtype));
    for (const char *part : {".weight", ".bias", ".running_mean",
                             ".running_var", ".num_batches_tracked"}) {
      weights.erase(norm + part);
    }
  }
}

LstmCell LstmStep(const mx::array &gates, const std::optional<LstmCell> &state,
                  const mx::array &recurrent_weight) {
  const mx::array all_gates =
      state.has_value()
          ? mx::addmm(gates, state->hidden, mx::transpose(recurrent_weight))
          : gates;
  const std::vector<mx::array> pieces = mx::split(all_gates, 4, -1);
  const mx::array input_gate = mx::sigmoid(pieces[0]);
  const mx::array forget_gate = mx::sigmoid(pieces[1]);
  const mx::array candidate = mx::tanh(pieces[2]);
  const mx::array output_gate = mx::sigmoid(pieces[3]);
  const mx::array cell = state.has_value()
                             ? mx::add(mx::multiply(forget_gate, state->cell),
                                       mx::multiply(input_gate, candidate))
                             : mx::multiply(input_gate, candidate);
  return {mx::multiply(output_gate, mx::tanh(cell)), cell};
}

} // namespace omni_common
