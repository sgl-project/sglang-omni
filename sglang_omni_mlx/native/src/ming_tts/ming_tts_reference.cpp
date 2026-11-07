// SPDX-License-Identifier: Apache-2.0
#include "ming_tts/ming_tts_reference.h"

#include <algorithm>
#include <stdexcept>
#include <string>

namespace ming_tts {
namespace mx = mlx::core;

namespace {

// campplus.safetensors, an ONNX export of CampPlus in PyTorch layout, as
// CamPlusEncoder names its weights.
omni_common::WeightMap CamPlusWeights(const std::filesystem::path &path) {
  const auto stored = mx::load_safetensors(path.string()).first;
  omni_common::WeightMap weights;
  const auto take = [&](const std::string &name) {
    const auto found = stored.find(name);
    if (found == stored.end()) {
      throw std::runtime_error(path.string() + " has no " + name);
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

MingReferenceEncoder::MingReferenceEncoder(
    const std::filesystem::path &directory)
    : speaker_(CamPlusWeights(directory / "campplus.safetensors")),
      audio_(directory) {
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
