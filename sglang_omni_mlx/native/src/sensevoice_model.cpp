// SPDX-License-Identifier: Apache-2.0
#include "sensevoice_model.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <stdexcept>

#include "swift_port.h"

namespace sensevoice {

namespace mx = mlx::core;

namespace {

constexpr float kLayerNormEps = 1e-5f;
// Kaldi fbank: int16-scaled samples, per-frame DC removal, preemphasis, the
// lowest mel at 20 Hz and a log floor.
constexpr float kInt16Scale = 32768.0f;
constexpr float kPreemphasis = 0.97f;
constexpr float kLowestMelHz = 20.0f;
constexpr float kLogFloor = 1e-10f;
constexpr double kPositionTimescale = 10000.0;
// Embedding rows of the query frames: the event and emotion queries, and
// the text normalization ones with and without inverse normalization.
constexpr int kEventQueryId = 1;
constexpr int kEmotionQueryId = 2;
constexpr int kWithItnId = 14;
constexpr int kWithoutItnId = 15;
// The query rows the Swift port's embedding table holds.
constexpr int kEmbeddingRows = 16;

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

// The Swift port's symmetric Hamming window, in float.
std::vector<float> HammingWindow(int size) {
  std::vector<float> window(size);
  const float denominator = static_cast<float>(size - 1);
  for (int n = 0; n < size; ++n) {
    const float phase =
        2.0f * static_cast<float>(M_PI) * static_cast<float>(n) / denominator;
    window[n] = 0.54f - 0.46f * std::cos(phase);
  }
  return window;
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

int NextPowerOfTwo(int value) {
  int power = 1;
  while (power < value) {
    power <<= 1;
  }
  return power;
}

} // namespace

SenseVoiceModel::SenseVoiceModel(const std::filesystem::path &model_directory)
    : SenseVoiceModel(qwen3_asr::ReadJson(model_directory / "config.json"),
                      model_directory) {}

SenseVoiceModel::SenseVoiceModel(const nlohmann::json &config,
                                 const std::filesystem::path &model_directory)
    : checkpoint_(qwen3_asr::LoadSafetensors(model_directory), config),
      mel_filters_(mx::array(0.0f)) {
  const nlohmann::json &encoder = config.at("encoder_conf");
  const nlohmann::json &front_end = config.at("frontend_conf");
  config_.vocabulary_size = config.at("vocab_size").get<int>();
  config_.input_size = config.at("input_size").get<int>();
  config_.model_width = encoder.at("output_size").get<int>();
  config_.head_count = encoder.at("attention_heads").get<int>();
  config_.feed_forward_width = encoder.at("linear_units").get<int>();
  config_.block_count = encoder.at("num_blocks").get<int>();
  config_.tp_block_count = encoder.at("tp_blocks").get<int>();
  config_.kernel_size = encoder.at("kernel_size").get<int>();
  config_.sanm_shift =
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
      config_.input_size != config_.lfr_m * config_.mel_count ||
      config_.head_count < 1 || config_.model_width % config_.head_count != 0 ||
      config_.block_count < 1 || config_.lfr_m < 1 || config_.lfr_n < 1) {
    throw std::runtime_error(
        "only the pre-norm 16 kHz Hamming SenseVoice checkpoint is supported");
  } else {
  }
  const int window_length =
      config_.sample_rate * config_.frame_length_ms / 1000;
  // Note (Jiaxin Deng): shapes the graph needs, checked at load rather than
  // on the first request (a zero hop would divide by zero in Fbank).
  const int fsmn_left_padding =
      (config_.kernel_size - 1) / 2 + config_.sanm_shift;
  if (window_length < 2 ||
      config_.sample_rate * config_.frame_shift_ms / 1000 < 1 ||
      config_.mel_count < 1 || config_.input_size % 2 != 0 ||
      config_.kernel_size < 1 || config_.sanm_shift < 0 ||
      fsmn_left_padding > config_.kernel_size - 1 ||
      config_.tp_block_count < 0 || config_.vocabulary_size < 1) {
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
  const int fft_size = NextPowerOfTwo(window_length);
  window_ = HammingWindow(window_length);
  std::vector<float> filters = HtkMelFilters(config_.sample_rate, fft_size,
                                             config_.mel_count, kLowestMelHz);
  mel_filters_ = mx::array(filters.data(),
                           {fft_size / 2 + 1, config_.mel_count}, mx::float32);
  std::vector<float> means;
  std::vector<float> inverse_stddevs;
  if (std::filesystem::exists(model_directory / "am.mvn")) {
    const std::string statistics = ReadFile(model_directory / "am.mvn");
    means = BracketedValues(statistics, "<AddShift>");
    inverse_stddevs = BracketedValues(statistics, "<Rescale>");
  } else if (config.contains("cmvn_means") && config.contains("cmvn_istd")) {
    means = config.at("cmvn_means").get<std::vector<float>>();
    inverse_stddevs = config.at("cmvn_istd").get<std::vector<float>>();
  } else {
  }
  if (!means.empty()) {
    if (static_cast<int>(means.size()) != config_.input_size ||
        inverse_stddevs.size() != means.size()) {
      throw std::runtime_error("CMVN statistics must match input_size");
    } else {
    }
    cmvn_means_ = mx::array(means.data(), {config_.input_size}, mx::float32);
    cmvn_inverse_stddevs_ =
        mx::array(inverse_stddevs.data(), {config_.input_size}, mx::float32);
  } else {
  }
  mx::eval(mel_filters_);
}

mx::array SenseVoiceModel::Fbank(const std::vector<float> &samples) const {
  const int window_length =
      config_.sample_rate * config_.frame_length_ms / 1000;
  const int hop_length = config_.sample_rate * config_.frame_shift_ms / 1000;
  const int sample_count = static_cast<int>(samples.size());
  if (sample_count < window_length) {
    return mx::zeros({0, config_.mel_count}, mx::float32);
  } else {
  }
  const mx::array audio =
      mx::multiply(mx::array(samples.data(), {sample_count}, mx::float32),
                   mx::array(kInt16Scale));
  const int frame_count = 1 + (sample_count - window_length) / hop_length;
  mx::array frames =
      mx::as_strided(audio, {frame_count, window_length}, {hop_length, 1}, 0);
  frames = mx::subtract(frames, mx::mean(frames, 1, true));
  // Note (Dayuxiaoshui): preemphasis within each frame; the first sample
  // takes itself as its predecessor.
  const mx::array preemphasis(kPreemphasis);
  const mx::array head = mx::slice(frames, {0, 0}, {frame_count, 1});
  const mx::array first = mx::subtract(head, mx::multiply(preemphasis, head));
  const mx::array rest = mx::subtract(
      mx::slice(frames, {0, 1}, {frame_count, window_length}),
      mx::multiply(preemphasis, mx::slice(frames, {0, 0},
                                          {frame_count, window_length - 1})));
  frames = mx::concatenate({first, rest}, 1);
  frames = mx::multiply(
      frames, mx::array(window_.data(), {window_length}, mx::float32));
  const int fft_size = NextPowerOfTwo(window_length);
  if (fft_size > window_length) {
    frames = mx::concatenate(
        {frames,
         mx::zeros({frame_count, fft_size - window_length}, mx::float32)},
        1);
  } else {
  }
  const mx::array power = mx::square(mx::abs(mx::fft::rfft(frames, 1)));
  return mx::log(
      mx::maximum(mx::matmul(power, mel_filters_), mx::array(kLogFloor)));
}

mx::array SenseVoiceModel::Features(const std::vector<float> &samples) const {
  const mx::array fbank = Fbank(samples);
  const int time = fbank.shape(0);
  const int lfr_frames = (time + config_.lfr_n - 1) / config_.lfr_n;
  if (lfr_frames == 0) {
    return mx::zeros({0, config_.input_size}, mx::float32);
  } else {
  }
  // Note (Dayuxiaoshui): frame i stacks frames i * lfr_n - (lfr_m - 1) / 2
  // onward, the first and last frames repeated past either end, as the
  // Swift port's padding does; a gather moves the same values.
  const int left_pad = (config_.lfr_m - 1) / 2;
  std::vector<int32_t> indices;
  indices.reserve(static_cast<size_t>(lfr_frames) * config_.lfr_m);
  for (int frame = 0; frame < lfr_frames; ++frame) {
    for (int offset = 0; offset < config_.lfr_m; ++offset) {
      indices.push_back(
          std::clamp(frame * config_.lfr_n + offset - left_pad, 0, time - 1));
    }
  }
  mx::array stacked = mx::reshape(
      mx::take(fbank,
               mx::array(indices.data(), {static_cast<int>(indices.size())},
                         mx::int32),
               0),
      {lfr_frames, config_.input_size});
  if (cmvn_means_.has_value()) {
    stacked =
        mx::multiply(mx::add(stacked, *cmvn_means_), *cmvn_inverse_stddevs_);
  } else {
  }
  return stacked;
}

mx::array SenseVoiceModel::EncoderLayer(const mx::array &x,
                                        const std::string &prefix,
                                        bool residual) const {
  const std::string attention = prefix + ".self_attn";
  const int width = config_.model_width;
  const int head_count = config_.head_count;
  const int head_width = width / head_count;
  const int time = x.shape(1);
  const mx::array normed =
      checkpoint_.LayerNorm(x, prefix + ".norm1", kLayerNormEps);
  const std::vector<mx::array> qkv =
      mx::split(checkpoint_.Linear(normed, attention + ".linear_q_k_v"), 3, -1);
  // Note (Dayuxiaoshui): the FSMN memory, a depthwise convolution over the
  // values padded to keep their length, added back onto them.
  const int left_padding = (config_.kernel_size - 1) / 2 + config_.sanm_shift;
  const int right_padding = config_.kernel_size - 1 - left_padding;
  const mx::array memory = mx::add(
      mx::conv1d(
          mx::pad(qkv[2], {{0, 0}, {left_padding, right_padding}, {0, 0}}),
          mx::transpose(checkpoint_.Weight(attention + ".fsmn_block.weight"),
                        {0, 2, 1}),
          1, 0, 1, width),
      qkv[2]);
  const auto heads = [&](const mx::array &projected) {
    return mx::transpose(
        mx::reshape(projected, {1, time, head_count, head_width}),
        {0, 2, 1, 3});
  };
  const float scale = 1.0f / std::sqrt(static_cast<float>(head_width));
  const mx::array attended = mx::fast::scaled_dot_product_attention(
      heads(qkv[0]), heads(qkv[1]), heads(qkv[2]), scale);
  const mx::array attention_output = mx::add(
      checkpoint_.Linear(
          mx::reshape(mx::transpose(attended, {0, 2, 1, 3}), {1, time, width}),
          attention + ".linear_out"),
      memory);
  // Note (Dayuxiaoshui): the first layer widens its input, so it has no
  // residual around attention; as in the Swift port, only a width change
  // drops it.
  const mx::array hidden =
      residual ? mx::add(x, attention_output) : attention_output;
  const mx::array feed_forward = checkpoint_.Linear(
      swift_port::Relu(checkpoint_.Linear(
          checkpoint_.LayerNorm(hidden, prefix + ".norm2", kLayerNormEps),
          prefix + ".feed_forward.w_1")),
      prefix + ".feed_forward.w_2");
  return mx::add(hidden, feed_forward);
}

mx::array SenseVoiceModel::LogProbabilities(const mx::array &features,
                                            int language_id,
                                            bool use_itn) const {
  const auto query = [&](std::vector<int32_t> ids) {
    return checkpoint_.Embed(
        mx::array(ids.data(), {1, static_cast<int>(ids.size())}, mx::int32),
        "embed");
  };
  mx::array x = mx::concatenate({query({language_id}),
                                 query({kEventQueryId, kEmotionQueryId}),
                                 query({use_itn ? kWithItnId : kWithoutItnId}),
                                 mx::expand_dims(features, 0)},
                                1);
  x = mx::multiply(x, mx::array(static_cast<float>(std::sqrt(
                          static_cast<double>(config_.model_width)))));
  // Note (Dayuxiaoshui): sinusoids at positions 1 onward, sines then
  // cosines, in MLX ops as the Swift port builds them.
  const int time = x.shape(1);
  const int half_width = std::max(config_.input_size / 2, 1);
  std::vector<float> positions(time);
  for (int index = 0; index < time; ++index) {
    positions[index] = static_cast<float>(index + 1);
  }
  const float log_increment =
      static_cast<float>(std::log(kPositionTimescale) /
                         static_cast<double>(std::max(half_width - 1, 1)));
  const mx::array inverse_timescales = mx::exp(mx::multiply(
      mx::arange(half_width, mx::float32), mx::array(-log_increment)));
  const mx::array scaled_time = mx::multiply(
      mx::expand_dims(mx::array(positions.data(), {time}, mx::float32), 1),
      mx::expand_dims(inverse_timescales, 0));
  const mx::array encoding =
      mx::concatenate({mx::sin(scaled_time), mx::cos(scaled_time)}, 1);
  x = mx::add(x, mx::astype(mx::expand_dims(encoding, 0), x.dtype()));

  x = EncoderLayer(x, "encoder.encoders0.0",
                   config_.input_size == config_.model_width);
  for (int layer = 0; layer < config_.block_count - 1; ++layer) {
    x = EncoderLayer(x, "encoder.encoders." + std::to_string(layer), true);
  }
  x = checkpoint_.LayerNorm(x, "encoder.after_norm", kLayerNormEps);
  for (int layer = 0; layer < config_.tp_block_count; ++layer) {
    x = EncoderLayer(x, "encoder.tp_encoders." + std::to_string(layer), true);
  }
  x = checkpoint_.LayerNorm(x, "encoder.tp_norm", kLayerNormEps);
  const mx::array logits = checkpoint_.Linear(x, "ctc.ctc_lo");
  return mx::squeeze(mx::subtract(logits, mx::logsumexp(logits, -1, true)), 0);
}

} // namespace sensevoice
