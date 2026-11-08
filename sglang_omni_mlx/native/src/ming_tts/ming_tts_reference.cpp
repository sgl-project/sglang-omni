// SPDX-License-Identifier: Apache-2.0
#include "ming_tts/ming_tts_reference.h"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <iterator>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace ming_tts {
namespace mx = mlx::core;

namespace {

constexpr const char *kCamPlusSafetensors = "campplus.safetensors";
constexpr const char *kCamPlusOnnx = "campplus.onnx";
// ONNX's ModelProto.graph, GraphProto.initializer and the TensorProto fields
// read here, with its FLOAT data type and EXTERNAL data location.
constexpr int kModelGraphField = 7;
constexpr int kGraphInitializerField = 5;
constexpr int kTensorDimsField = 1;
constexpr int kTensorDataTypeField = 2;
constexpr int kTensorFloatDataField = 4;
constexpr int kTensorNameField = 8;
constexpr int kTensorRawDataField = 9;
constexpr int kTensorDataLocationField = 14;
constexpr uint64_t kOnnxFloat = 1;
constexpr uint64_t kOnnxExternalData = 1;
// Protobuf wire types.
constexpr int kVarint = 0;
constexpr int kFixed64 = 1;
constexpr int kLengthDelimited = 2;
constexpr int kFixed32 = 5;

// The fields of one protobuf message, read in wire order.
class ProtobufFields {
public:
  explicit ProtobufFields(std::string_view bytes) : bytes_(bytes) {}

  bool Done() const { return position_ == bytes_.size(); }

  // The next field's number and wire type.
  std::pair<int, int> Tag() {
    const uint64_t tag = Varint();
    return {static_cast<int>(tag >> 3), static_cast<int>(tag & 7)};
  }

  uint64_t Varint() {
    uint64_t value = 0;
    for (int shift = 0; shift < 64; shift += 7) {
      const uint8_t byte = static_cast<uint8_t>(Take(1)[0]);
      value |= static_cast<uint64_t>(byte & 0x7f) << shift;
      if ((byte & 0x80) == 0) {
        return value;
      } else {
      }
    }
    throw std::runtime_error("malformed ONNX file: varint too long");
  }

  std::string_view Bytes() { return Take(Varint()); }

  std::string_view Take(uint64_t count) {
    if (count > bytes_.size() - position_) {
      throw std::runtime_error("malformed ONNX file: field past its message");
    } else {
    }
    const std::string_view taken = bytes_.substr(position_, count);
    position_ += count;
    return taken;
  }

  void Skip(int wire_type) {
    if (wire_type == kVarint) {
      Varint();
    } else if (wire_type == kFixed64) {
      Take(8);
    } else if (wire_type == kLengthDelimited) {
      Bytes();
    } else if (wire_type == kFixed32) {
      Take(4);
    } else {
      throw std::runtime_error("malformed ONNX file: wire type " +
                               std::to_string(wire_type));
    }
  }

private:
  std::string_view bytes_;
  size_t position_ = 0;
};

// A TensorProto as an array, or nothing for a tensor that is not float32.
std::optional<std::pair<std::string, mx::array>>
OnnxTensor(std::string_view message) {
  ProtobufFields fields(message);
  std::string name;
  mx::Shape shape;
  uint64_t data_type = 0;
  std::string_view raw_data;
  std::vector<float> float_data;
  bool external = false;
  while (!fields.Done()) {
    const auto [field, wire_type] = fields.Tag();
    if (field == kTensorDimsField && wire_type == kVarint) {
      shape.push_back(static_cast<int>(fields.Varint()));
    } else if (field == kTensorDimsField && wire_type == kLengthDelimited) {
      ProtobufFields dims(fields.Bytes());
      while (!dims.Done()) {
        shape.push_back(static_cast<int>(dims.Varint()));
      }
    } else if (field == kTensorDataTypeField && wire_type == kVarint) {
      data_type = fields.Varint();
    } else if (field == kTensorFloatDataField &&
               wire_type == kLengthDelimited) {
      const std::string_view packed = fields.Bytes();
      const size_t start = float_data.size();
      float_data.resize(start + packed.size() / sizeof(float));
      std::memcpy(float_data.data() + start, packed.data(),
                  packed.size() / sizeof(float) * sizeof(float));
    } else if (field == kTensorFloatDataField && wire_type == kFixed32) {
      float value;
      std::memcpy(&value, fields.Take(sizeof(float)).data(), sizeof(float));
      float_data.push_back(value);
    } else if (field == kTensorNameField && wire_type == kLengthDelimited) {
      name = std::string(fields.Bytes());
    } else if (field == kTensorRawDataField && wire_type == kLengthDelimited) {
      raw_data = fields.Bytes();
    } else if (field == kTensorDataLocationField && wire_type == kVarint) {
      external = fields.Varint() == kOnnxExternalData;
    } else {
      fields.Skip(wire_type);
    }
  }
  size_t element_count = 1;
  for (const int dimension : shape) {
    element_count *= dimension;
  }
  if (data_type != kOnnxFloat) {
    return std::nullopt;
  } else if (external) {
    throw std::runtime_error("ONNX tensor " + name +
                             " keeps its data in another file");
  } else if (!raw_data.empty()) {
    if (raw_data.size() != element_count * sizeof(float)) {
      throw std::runtime_error("malformed ONNX file: tensor " + name +
                               " has the wrong byte count");
    } else {
    }
    float_data.resize(element_count);
    std::memcpy(float_data.data(), raw_data.data(), raw_data.size());
  } else if (float_data.size() != element_count) {
    throw std::runtime_error("malformed ONNX file: tensor " + name +
                             " has the wrong element count");
  } else {
  }
  return std::pair{name, mx::array(float_data.begin(), shape, mx::float32)};
}

// The float32 initializers of an ONNX model, by name.
omni_common::WeightMap OnnxInitializers(const std::filesystem::path &path) {
  std::ifstream stream(path, std::ios::binary);
  if (!stream) {
    throw std::runtime_error("cannot read " + path.string());
  } else {
  }
  const std::string bytes((std::istreambuf_iterator<char>(stream)),
                          std::istreambuf_iterator<char>());
  omni_common::WeightMap initializers;
  ProtobufFields model(bytes);
  while (!model.Done()) {
    const auto [field, wire_type] = model.Tag();
    if (field == kModelGraphField && wire_type == kLengthDelimited) {
      ProtobufFields graph(model.Bytes());
      while (!graph.Done()) {
        const auto [graph_field, graph_wire_type] = graph.Tag();
        if (graph_field == kGraphInitializerField &&
            graph_wire_type == kLengthDelimited) {
          if (auto tensor = OnnxTensor(graph.Bytes())) {
            initializers.insert_or_assign(tensor->first, tensor->second);
          } else {
          }
        } else {
          graph.Skip(graph_wire_type);
        }
      }
    } else {
      model.Skip(wire_type);
    }
  }
  return initializers;
}

// CampPlus in a model directory, an ONNX export in PyTorch layout, as
// CamPlusEncoder names its weights: campplus.safetensors, which the
// mlx-community checkpoints carry, else the official campplus.onnx, whose
// initializers it holds.
omni_common::WeightMap CamPlusWeights(const std::filesystem::path &directory) {
  const std::filesystem::path safetensors = directory / kCamPlusSafetensors;
  const bool has_safetensors = std::filesystem::is_regular_file(safetensors);
  const std::filesystem::path source =
      has_safetensors ? safetensors : directory / kCamPlusOnnx;
  const omni_common::WeightMap stored =
      has_safetensors ? mx::load_safetensors(source.string()).first
                      : OnnxInitializers(source);
  omni_common::WeightMap weights;
  const auto take = [&](const std::string &name) {
    const auto found = stored.find(name);
    if (found == stored.end()) {
      throw std::runtime_error(source.string() + " has no " + name);
    } else {
    }
    return mx::astype(found->second, mx::float32);
  };
  const auto conv1d = [](const mx::array &weight) {
    return mx::transpose(weight, {0, 2, 1});
  };
  // Note (altale): The exported checkpoint retains numeric names for folded
  // convolutions.
  // Note (XXXXRT666): the export folds the batch norm after each convolution
  // into it; they are numbered three apart in network order.
  int folded = 4423;
  const auto fold = [&](const std::string &convolution, bool two_d) {
    const mx::array weight = take("onnx::Conv_" + std::to_string(folded));
    weights.insert_or_assign(convolution + ".weight",
                             two_d ? mx::transpose(weight, {0, 2, 3, 1})
                                   : conv1d(weight));
    weights.insert_or_assign(convolution + ".bias",
                             take("onnx::Conv_" + std::to_string(folded + 1)));
    folded += 3;
  };
  const auto batch_norm = [&](const std::string &prefix,
                              const std::string &stored_prefix) {
    for (const char *field : {".running_mean", ".running_var"}) {
      weights.insert_or_assign(prefix + field, take(stored_prefix + field));
    }
    if (stored.count(stored_prefix + ".weight") > 0) {
      for (const char *field : {".weight", ".bias"}) {
        weights.insert_or_assign(prefix + field, take(stored_prefix + field));
      }
    } else {
    }
  };
  fold("head.conv1", true);
  for (const std::string layer : {"head.layer1", "head.layer2"}) {
    fold(layer + ".0.conv1", true);
    fold(layer + ".0.conv2", true);
    fold(layer + ".0.shortcut.0", true);
    fold(layer + ".1.conv1", true);
    fold(layer + ".1.conv2", true);
  }
  fold("head.conv2", true);
  fold("tdnn.linear", false);
  for (int block = 0; block < 3; ++block) {
    for (int layer = 0; layer < omni_common::kCamPlusBlockLayerCounts[block];
         ++layer) {
      const std::string prefix = omni_common::CamPlusLayerPrefix(block, layer);
      const std::string stored_prefix = "xvector.block" +
                                        std::to_string(block + 1) + ".tdnnd" +
                                        std::to_string(layer + 1);
      batch_norm(prefix + ".nonlinear1.0",
                 stored_prefix + ".nonlinear1.batchnorm");
      fold(prefix + ".linear1", false);
      for (const char *name : {".linear_local", ".linear1", ".linear2"}) {
        const std::string attention = ".cam_layer" + std::string(name);
        weights.insert_or_assign(
            prefix + attention + ".weight",
            conv1d(take(stored_prefix + attention + ".weight")));
        if (std::string(name) != ".linear_local") {
          weights.insert_or_assign(prefix + attention + ".bias",
                                   take(stored_prefix + attention + ".bias"));
        } else {
        }
      }
    }
    const std::string transit = "transits." + std::to_string(block);
    const std::string stored_transit =
        "xvector.transit" + std::to_string(block + 1);
    batch_norm(transit + ".nonlinear.0",
               stored_transit + ".nonlinear.batchnorm");
    // The last transit carries the output batch norm folded into it.
    if (block == 2) {
      fold(transit + ".linear", false);
    } else {
      weights.insert_or_assign(transit + ".linear.weight",
                               conv1d(take(stored_transit + ".linear.weight")));
    }
  }
  weights.insert_or_assign("dense.linear.weight",
                           conv1d(take("xvector.dense.linear.weight")));
  batch_norm("dense.nonlinear.0", "xvector.dense.nonlinear.batchnorm");
  return weights;
}

} // namespace

bool HasCamPlusWeights(const std::filesystem::path &directory) {
  return std::filesystem::is_regular_file(directory / kCamPlusSafetensors) ||
         std::filesystem::is_regular_file(directory / kCamPlusOnnx);
}

MingReferenceEncoder::MingReferenceEncoder(
    const std::filesystem::path &directory)
    : speaker_(CamPlusWeights(directory)), audio_(directory) {
  const auto config = omni_common::ReadJson(directory / "config.json");
  const auto &audio = config.at("audio_tokenizer_config");
  // Note (altale): Pad to complete acoustic patches after both encoder
  // reductions.
  alignment_samples_ = config.at("ditar_config").at("patch_size").get<int>() *
                       audio.at("enc_kwargs").value("hop_size", 320) *
                       std::max(1, audio.at("patch_size").get<int>());
}

ReferenceConditioning
MingReferenceEncoder::Encode(const ReferenceAudio &audio) const {
  std::vector<float> padded = audio.samples_44k;
  padded.resize(static_cast<size_t>(PatchCount(padded.size())) *
                    alignment_samples_,
                0.0f);
  const auto latents =
      mx::astype(audio_.Encode(mx::array(padded.data(),
                                         {1, static_cast<int>(padded.size())})),
                 mx::float32);
  const auto speaker = speaker_.Embed(audio.samples_16k);
  mx::eval(latents, speaker);
  return {latents, speaker};
}

int MingReferenceEncoder::PatchCount(size_t sample_count) const {
  return static_cast<int>((sample_count + alignment_samples_ - 1) /
                          alignment_samples_);
}

} // namespace ming_tts
