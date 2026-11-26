// SPDX-License-Identifier: Apache-2.0
#include "sensevoice/sensevoice_model.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <stdexcept>

#include "common/kaldi_fbank.h"
#include "common/layers.h"

namespace sensevoice {

namespace mx = mlx::core;

namespace {

// The Swift port's fbank: the lowest mel at 20 Hz and a log floor.
constexpr float kLowestMelHz = 20.0f;
constexpr float kLogFloor = 1e-10f;
// Embedding rows of the query frames: the event and emotion queries, and
// the text normalization ones with and without inverse normalization.
constexpr int kEventQueryId = 1;
constexpr int kEmotionQueryId = 2;
constexpr int kWithItnId = 14;
constexpr int kWithoutItnId = 15;
// The query rows the Swift port's embedding table holds.
constexpr int kEmbeddingRows = 16;

// Note (Dayuxiaoshui): the bracketed list after <LearnRateCoef> N in the
// component a tag opens, as the Swift port's regex reads am.mvn; words that
// are not numbers are skipped.
std::vector<float> BracketedValues(const std::string &text,
                                   const std::string &tag) {
  const size_t component = text.find(tag);
  const size_t coefficient = component == std::string::npos
                                 ? std::string::npos
                                 : text.find("<LearnRateCoef>", component);
  const size_t open = coefficient == std::string::npos
                          ? std::string::npos
                          : text.find('[', coefficient);
  const size_t close =
      open == std::string::npos ? std::string::npos : text.find(']', open);
  if (close == std::string::npos) {
    throw std::runtime_error("am.mvn has no " + tag + " statistics");
  } else {
  }
  std::istringstream words(text.substr(open + 1, close - open - 1));
  std::vector<float> values;
  for (std::string word; words >> word;) {
    char *end = nullptr;
    const float value = std::strtof(word.c_str(), &end);
    if (end == word.c_str() + word.size()) {
      values.push_back(value);
    } else {
    }
  }
  return values;
}

// The Swift port's mel filters with the HTK scale and no normalization:
// [fft_size / 2 + 1, mel_count], built in float in its order.
std::vector<float> HtkMelFilters(int sample_rate, int fft_size, int mel_count,
                                 float lowest_hz) {
  const int frequency_count = fft_size / 2 + 1;
  const auto hertz_to_mel = [](float hertz) {
    return 2595.0f * std::log10(1.0f + hertz / 700.0f);
  };
  const auto mel_to_hertz = [](float mel) {
    return 700.0f * (std::pow(10.0f, mel / 2595.0f) - 1.0f);
  };
  const float lowest_mel = hertz_to_mel(lowest_hz);
  const float highest_mel =
      hertz_to_mel(static_cast<float>(sample_rate) / 2.0f);
  std::vector<float> edges(mel_count + 2);
  for (int i = 0; i < mel_count + 2; ++i) {
    edges[i] = mel_to_hertz(lowest_mel + static_cast<float>(i) *
                                             (highest_mel - lowest_mel) /
                                             static_cast<float>(mel_count + 1));
  }
  std::vector<float> filters(static_cast<size_t>(frequency_count) * mel_count,
                             0.0f);
  for (int i = 0; i < frequency_count; ++i) {
    const float frequency = static_cast<float>(i) *
                            static_cast<float>(sample_rate) /
                            static_cast<float>(fft_size);
    for (int j = 0; j < mel_count; ++j) {
      const float low = edges[j];
      const float center = edges[j + 1];
      const float high = edges[j + 2];
      float &weight = filters[static_cast<size_t>(i) * mel_count + j];
      if (frequency >= low && frequency < center) {
        weight = (frequency - low) / (center - low);
      } else if (frequency >= center && frequency <= high) {
        weight = (high - frequency) / (high - center);
      } else {
      }
    }
  }
  return filters;
}

// Note (XXXXRT666): the checkpoint is float32; the encoder runs in bfloat16,
// its FSMN kernels already in MLX's layout.
omni_common::WeightMap HalfPrecisionWeights(omni_common::WeightMap weights) {
  for (auto &[name, weight] : weights) {
    if (name.ends_with(".fsmn_block.weight")) {
      weight = mx::contiguous(mx::transpose(weight, {0, 2, 1}));
    } else {
    }
    if (mx::issubdtype(weight.dtype(), mx::floating)) {
      weight = mx::astype(weight, mx::bfloat16);
    } else {
    }
  }
  return weights;
}

} // namespace

SenseVoiceModel::SenseVoiceModel(const std::filesystem::path &model_directory)
    : SenseVoiceModel(omni_common::ReadJson(model_directory / "config.json"),
                      model_directory) {}

SenseVoiceModel::SenseVoiceModel(const nlohmann::json &config,
                                 const std::filesystem::path &model_directory)
    : checkpoint_(
          HalfPrecisionWeights(omni_common::LoadSafetensors(model_directory)),
          config),
      mel_filters_(mx::array(0.0f)) {
  const nlohmann::json &encoder = config.at("encoder_conf");
  const nlohmann::json &front_end = config.at("frontend_conf");
  config_.vocabulary_size = config.at("vocab_size").get<int>();
  config_.encoder.input_size = config.at("input_size").get<int>();
  config_.encoder.model_width = encoder.at("output_size").get<int>();
  config_.encoder.head_count = encoder.at("attention_heads").get<int>();
  config_.encoder.block_count = encoder.at("num_blocks").get<int>();
  config_.encoder.tp_block_count = encoder.at("tp_blocks").get<int>();
  config_.encoder.kernel_size = encoder.at("kernel_size").get<int>();
  config_.encoder.sanm_shift =
      encoder.value("sanm_shift", encoder.value("sanm_shfit", 0));
  config_.sample_rate = front_end.at("fs").get<int>();
  config_.mel_count = front_end.at("n_mels").get<int>();
  config_.frame_length_ms = front_end.at("frame_length").get<int>();
  config_.frame_shift_ms = front_end.at("frame_shift").get<int>();
  config_.lfr_m = front_end.at("lfr_m").get<int>();
  config_.lfr_n = front_end.at("lfr_n").get<int>();
  // Note (Dayuxiaoshui): the graph below is the published checkpoint's:
  // 16 kHz Hamming fbank, pre-norm layers, stacked frames as wide as the
  // first layer, and heads splitting the width evenly.
  if (config.value("model_type", "sensevoice") != "sensevoice" ||
      config_.sample_rate != 16000 ||
      front_end.at("window").get<std::string>() != "hamming" ||
      !encoder.value("normalize_before", true) ||
      config_.encoder.input_size != config_.lfr_m * config_.mel_count ||
      config_.encoder.head_count < 1 ||
      config_.encoder.model_width % config_.encoder.head_count != 0 ||
      config_.encoder.block_count < 1 || config_.lfr_m < 1 ||
      config_.lfr_n < 1) {
    throw std::runtime_error(
        "only the pre-norm 16 kHz Hamming SenseVoice checkpoint is supported");
  } else {
  }
  const int window_length =
      config_.sample_rate * config_.frame_length_ms / 1000;
  // Note (Jiaxin Deng): shapes the graph needs, checked at load rather than
  // on the first request (a zero hop would divide by zero in Fbank).
  const int fsmn_left_padding =
      (config_.encoder.kernel_size - 1) / 2 + config_.encoder.sanm_shift;
  if (window_length < 2 ||
      config_.sample_rate * config_.frame_shift_ms / 1000 < 1 ||
      config_.mel_count < 1 || config_.encoder.input_size % 2 != 0 ||
      config_.encoder.kernel_size < 1 || config_.encoder.sanm_shift < 0 ||
      fsmn_left_padding > config_.encoder.kernel_size - 1 ||
      config_.encoder.tp_block_count < 0 || config_.vocabulary_size < 1) {
    throw std::runtime_error("SenseVoice config has unsupported shapes");
  } else {
  }
  const mx::array &embedding_weight = checkpoint_.Weight("embed.weight");
  const mx::array &ctc_weight = checkpoint_.Weight("ctc.ctc_lo.weight");
  if (embedding_weight.ndim() != 2 ||
      embedding_weight.shape(0) < kEmbeddingRows || ctc_weight.ndim() != 2 ||
      ctc_weight.shape(0) != config_.vocabulary_size) {
    throw std::runtime_error(
        "SenseVoice checkpoint's embedding or CTC head does not match its "
        "config");
  } else {
  }
  const int fft_size = omni_common::PaddedFftSize(window_length);
  window_ = omni_common::SymmetricHammingWindow(window_length);
  std::vector<float> filters = HtkMelFilters(config_.sample_rate, fft_size,
                                             config_.mel_count, kLowestMelHz);
  mel_filters_ = mx::array(filters.data(),
                           {fft_size / 2 + 1, config_.mel_count}, mx::float32);
  std::vector<float> means;
  std::vector<float> inverse_stddevs;
  if (std::filesystem::exists(model_directory / "am.mvn")) {
    const std::string statistics =
        omni_common::ReadFile(model_directory / "am.mvn");
    means = BracketedValues(statistics, "<AddShift>");
    inverse_stddevs = BracketedValues(statistics, "<Rescale>");
  } else if (config.contains("cmvn_means") && config.contains("cmvn_istd")) {
    means = config.at("cmvn_means").get<std::vector<float>>();
    inverse_stddevs = config.at("cmvn_istd").get<std::vector<float>>();
  } else {
  }
  if (!means.empty()) {
    if (static_cast<int>(means.size()) != config_.encoder.input_size ||
        inverse_stddevs.size() != means.size()) {
      throw std::runtime_error("CMVN statistics must match input_size");
    } else {
    }
    cmvn_means_ =
        mx::array(means.data(), {config_.encoder.input_size}, mx::float32);
    cmvn_inverse_stddevs_ = mx::array(
        inverse_stddevs.data(), {config_.encoder.input_size}, mx::float32);
  } else {
  }
  mx::async_eval(mel_filters_);
}

mx::array SenseVoiceModel::Features(const std::vector<float> &samples) const {
  const int hop_length = config_.sample_rate * config_.frame_shift_ms / 1000;
  mx::array stacked = omni_common::LowFrameRateFrames(
      omni_common::KaldiFbank(samples, window_, hop_length, mel_filters_,
                              kLogFloor),
      config_.lfr_m, config_.lfr_n);
  if (cmvn_means_.has_value()) {
    stacked =
        mx::multiply(mx::add(stacked, *cmvn_means_), *cmvn_inverse_stddevs_);
  } else {
  }
  return stacked;
}

mx::array SenseVoiceModel::Logits(const mx::array &features, int language_id,
                                  bool use_itn) const {
  const auto query = [&](std::vector<int32_t> ids) {
    return checkpoint_.Embed(
        mx::array(ids.data(), {1, static_cast<int>(ids.size())}, mx::int32),
        "embed");
  };
  const mx::array x = mx::concatenate(
      {query({language_id}), query({kEventQueryId, kEmotionQueryId}),
       query({use_itn ? kWithItnId : kWithoutItnId}),
       mx::expand_dims(mx::astype(features, mx::bfloat16), 0)},
      1);
  const mx::array logits = checkpoint_.Linear(
      omni_common::SanmEncode(checkpoint_, x, config_.encoder), "ctc.ctc_lo");
  return mx::squeeze(logits, 0);
}

} // namespace sensevoice
