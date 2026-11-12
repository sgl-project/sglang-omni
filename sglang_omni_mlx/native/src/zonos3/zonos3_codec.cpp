// SPDX-License-Identifier: Apache-2.0
// A port of mlx-audio's zonos3/codec.py (CodecDecoder and CodecStream), op for
// op, so whole-utterance decodes match it sample for sample.
#include "zonos3/zonos3_codec.h"

#include <cmath>
#include <complex>
#include <stdexcept>
#include <string>

namespace zonos3 {

namespace mx = mlx::core;

namespace {

constexpr const char *kArchitecture = "mossdual_istft";
// Query rows per windowed-attention block of a whole-utterance decode.
constexpr int kAttentionBlockRows = 512;
// Frames per streaming trunk pass.
constexpr int kMaxStreamChunkFrames = 32;
constexpr float kRopeBase = 10000.0f;

mx::array Scalar(float value) { return mx::array(value); }

// 0.5 * x * (1 + erf(x / sqrt(2))), in the reference's order.
mx::array Gelu(const mx::array &x) {
  return mx::multiply(
      mx::multiply(Scalar(0.5f), x),
      mx::add(Scalar(1.0f),
              mx::erf(mx::multiply(
                  x, Scalar(static_cast<float>(1.0 / std::sqrt(2.0)))))));
}

mx::array Rows(const mx::array &x, int begin, int end) {
  return mx::slice(x, {begin, 0}, {end, x.shape(1)});
}

mx::array Columns(const mx::array &x, int begin, int end) {
  return mx::slice(x, {0, begin}, {x.shape(0), end});
}

// h[t] = sum_k x[t + k - (K - 1)] * w[:, k] for x [T, C], w [C, K]; the K - 1
// rows before x are history, zeros when unset.
mx::array CausalDepthwise(const mx::array &x, const mx::array &weight,
                          const std::optional<mx::array> &history) {
  const int rows = x.shape(0);
  const int taps = weight.shape(1);
  const mx::array previous = history.has_value()
                                 ? mx::astype(*history, x.dtype())
                                 : mx::zeros({taps - 1, x.shape(1)}, x.dtype());
  const mx::array padded = mx::concatenate({previous, x}, 0);
  mx::array h = mx::multiply(Rows(padded, 0, rows),
                             mx::squeeze(Columns(weight, 0, 1), 1));
  for (int k = 1; k < taps; ++k) {
    h = mx::add(h, mx::multiply(Rows(padded, k, k + rows),
                                mx::squeeze(Columns(weight, k, k + 1), 1)));
  }
  return h;
}

// out[t * patch + j, c] = in[t, c * patch + j].
mx::array PatchUp(const mx::array &x, int patch) {
  const int rows = x.shape(0);
  const int channels = x.shape(1);
  return mx::reshape(
      mx::transpose(mx::reshape(x, {rows, channels / patch, patch}), {0, 2, 1}),
      {rows * patch, channels / patch});
}

std::vector<int> IntList(const nlohmann::json &values) {
  std::vector<int> out;
  for (const auto &value : values) {
    out.push_back(value.get<int>());
  }
  return out;
}

std::vector<double> HostVector(const mx::array &array) {
  const mx::array as_float = mx::astype(array, mx::float32);
  mx::eval(as_float);
  const float *data = as_float.data<float>();
  return std::vector<double>(data, data + as_float.size());
}

std::vector<std::vector<double>> HostMatrix(const mx::array &array) {
  const std::vector<double> flat = HostVector(array);
  const int rows = array.shape(0);
  const int columns = array.shape(1);
  std::vector<std::vector<double>> matrix(rows);
  for (int row = 0; row < rows; ++row) {
    matrix[row].assign(flat.begin() + row * columns,
                       flat.begin() + (row + 1) * columns);
  }
  return matrix;
}

// Each row of x [rows, columns] on the host.
std::vector<std::vector<float>> HostRows(const mx::array &x) {
  const mx::array contiguous = mx::contiguous(mx::astype(x, mx::float32));
  mx::eval(contiguous);
  const float *data = contiguous.data<float>();
  const int rows = x.shape(0);
  const int columns = x.shape(1);
  std::vector<std::vector<float>> out(rows);
  for (int row = 0; row < rows; ++row) {
    out[row].assign(data + row * columns, data + (row + 1) * columns);
  }
  return out;
}

// torchaudio-style polyphase resampling of x [samples]:
// out[n * to + i] = sum_k kernel[i][k] x[n * from + k - width].
mx::array SincResample(const mx::array &x, const mx::array &kernel, int from,
                       int to, int width) {
  const long long length = x.shape(0);
  const long long total = (to * length + from - 1) / from;
  const long long blocks = (total + to - 1) / to;
  const int taps = 2 * width + from;
  const long long pad_right = blocks * from + from - (length + width);
  const mx::array padded = mx::concatenate(
      {mx::zeros({width}, x.dtype()), x,
       mx::zeros({static_cast<int>(std::max<long long>(pad_right, 0))},
                 x.dtype())});
  const int block_count = static_cast<int>(blocks);
  const mx::array first = mx::reshape(
      mx::slice(padded, {0}, {block_count * from}), {block_count, from});
  const mx::array second = Columns(
      mx::reshape(mx::slice(padded, {from}, {from + block_count * from}),
                  {block_count, from}),
      0, taps - from);
  const mx::array windows = mx::concatenate({first, second}, 1);
  return mx::slice(
      mx::reshape(mx::matmul(windows, mx::transpose(kernel)), {-1}), {0},
      {static_cast<int>(total)});
}

} // namespace

CodecDecoder::CodecDecoder(const std::filesystem::path &codec_directory)
    : weights_(codec_directory, ""), mixer_(mx::array(0.0f)),
      resample_kernel_(mx::array(0.0f)) {
  if (weights_.architecture() != kArchitecture) {
    throw std::runtime_error(codec_directory.string() +
                             " is not a ZONOS3 iSTFT decoder checkpoint");
  } else {
  }
  sample_rate_ = weights_.Meta("sample_rate").get<int>();
  hop_ = weights_.Meta("hop").get<int>();
  const int internal_hop = weights_.Meta("internal_hop").get<int>();
  codebook_count_ = weights_.Meta("n_codebooks").get<int>();
  codebook_size_ = weights_.Meta("codebook_size").get<int>();
  semantic_depth_ = weights_.Meta("sem_head.depth").get<int>();
  head_depth_ = weights_.Meta("head.depth").get<int>();
  head_hop_ = weights_.Meta("head.hop").get<int>();
  fft_sizes_ = IntList(weights_.Meta("head.n_fft"));
  max_log_magnitude_ = weights_.Meta("head.max_log_magnitude").get<float>();
  headroom_db_ = weights_.Meta("headroom_db").get<double>();
  layer_norm_epsilon_ = weights_.Meta("ln_eps").get<float>();
  convnext_epsilon_ = weights_.Meta("convnext_eps").get<float>();
  resample_from_ = weights_.Meta("resample.orig").get<int>();
  resample_to_ = weights_.Meta("resample.new").get<int>();
  resample_width_ = weights_.Meta("resample.width").get<int>();
  if (sample_rate_ != 44100 || hop_ != 2940 || internal_hop != 2944 ||
      codebook_count_ != 10 || fft_sizes_.size() != 2) {
    throw std::runtime_error(codec_directory.string() +
                             ": inconsistent decoder dimensions");
  } else {
  }
  const std::vector<int> output_sizes = IntList(weights_.Meta("stage.out_dim"));
  const std::vector<int> model_sizes = IntList(weights_.Meta("stage.d_model"));
  const std::vector<int> head_counts = IntList(weights_.Meta("stage.n_head"));
  const std::vector<int> layer_counts = IntList(weights_.Meta("stage.n_layer"));
  const std::vector<int> contexts = IntList(weights_.Meta("stage.context"));
  const std::vector<int> patches = IntList(weights_.Meta("stage.patch"));
  for (size_t stage = 0; stage < model_sizes.size(); ++stage) {
    stages_.push_back({output_sizes.at(stage), model_sizes.at(stage),
                       head_counts.at(stage), layer_counts.at(stage),
                       contexts.at(stage), patches.at(stage)});
  }
  for (int branch = 0; branch < 2; ++branch) {
    windows_.push_back(weights_.Table("head.window." + std::to_string(branch)));
  }
  mixer_ = weights_.Table("head.mixer.weight");
  resample_kernel_ = weights_.Table("resample.kernel");
  head_config_.fft_sizes = fft_sizes_;
  head_config_.hop = head_hop_;
  head_config_.windows = {HostVector(windows_[0]), HostVector(windows_[1])};
  head_config_.mixer = HostMatrix(mixer_);
  head_config_.max_log_magnitude = max_log_magnitude_;
}

float CodecDecoder::OutputGain(std::optional<double> headroom_db) const {
  const double decibels = headroom_db.has_value() ? *headroom_db : headroom_db_;
  return static_cast<float>(std::pow(10.0, -decibels / 20.0));
}

mx::array
CodecDecoder::ConvNextBlock(const mx::array &x, const std::string &prefix,
                            const std::optional<mx::array> &history) const {
  mx::array h =
      CausalDepthwise(x, weights_.Table(prefix + ".dw.weight"), history);
  h = mx::add(h, weights_.Table(prefix + ".dw.bias"));
  h = mx::fast::layer_norm(h, weights_.Table(prefix + ".norm.weight"),
                           weights_.Table(prefix + ".norm.bias"),
                           convnext_epsilon_);
  h = Gelu(weights_.Linear(h, prefix + ".pw1.weight", prefix + ".pw1.bias"));
  h = weights_.Linear(h, prefix + ".pw2.weight", prefix + ".pw2.bias");
  return mx::add(x, h);
}

mx::array
CodecDecoder::Latent(const mx::array &codes,
                     std::vector<std::optional<mx::array>> *histories) const {
  const int frames = codes.shape(0);
  mx::array x = mx::add(
      mx::take(weights_.Table("sem.table"),
               mx::reshape(mx::slice(codes, {0, 0}, {frames, 1}), {frames}), 0),
      weights_.Table("sem.bias"));
  for (int block = 0; block < semantic_depth_; ++block) {
    const std::string prefix = "sem_head." + std::to_string(block);
    if (histories == nullptr) {
      x = ConvNextBlock(x, prefix, std::nullopt);
    } else {
      std::optional<mx::array> &history = (*histories)[block];
      const mx::array output = ConvNextBlock(x, prefix, history);
      const int taps = weights_.Table(prefix + ".dw.weight").shape(1);
      const mx::array previous =
          history.has_value() ? *history
                              : mx::zeros({taps - 1, x.shape(1)}, x.dtype());
      const mx::array joined = mx::concatenate({previous, x}, 0);
      history = Rows(joined, joined.shape(0) - (taps - 1), joined.shape(0));
      x = output;
    }
  }
  x = weights_.Linear(x, "sem_head.out.weight", "sem_head.out.bias");
  for (int codebook = 1; codebook < codebook_count_; ++codebook) {
    x = mx::add(
        x,
        mx::take(
            weights_.Table("res." + std::to_string(codebook - 1) + ".table"),
            mx::reshape(mx::slice(codes, {0, codebook}, {frames, codebook + 1}),
                        {frames}),
            0));
  }
  return mx::add(x, weights_.Table("res.bias"));
}

std::pair<mx::array, mx::array>
CodecDecoder::Head(mx::array x,
                   std::vector<std::optional<mx::array>> *histories) const {
  for (int block = 0; block < head_depth_; ++block) {
    const std::string prefix = "head." + std::to_string(block);
    if (histories == nullptr) {
      x = ConvNextBlock(x, prefix, std::nullopt);
    } else {
      std::optional<mx::array> &history = (*histories)[block];
      const mx::array output = ConvNextBlock(x, prefix, history);
      const int taps = weights_.Table(prefix + ".dw.weight").shape(1);
      const mx::array previous =
          history.has_value() ? *history
                              : mx::zeros({taps - 1, x.shape(1)}, x.dtype());
      const mx::array joined = mx::concatenate({previous, x}, 0);
      history = Rows(joined, joined.shape(0) - (taps - 1), joined.shape(0));
      x = output;
    }
    mx::eval(x);
  }
  x = mx::fast::layer_norm(x, weights_.Table("head.norm.weight"),
                           weights_.Table("head.norm.bias"), convnext_epsilon_);
  return {weights_.Linear(x, "head.out.0.weight", "head.out.0.bias"),
          weights_.Linear(x, "head.out.1.weight", "head.out.1.bias")};
}

mx::array CodecDecoder::TransformerLayer(const mx::array &x,
                                         const std::string &prefix,
                                         const CodecStage &stage) const {
  const int rows = x.shape(0);
  const int width = x.shape(1);
  const int head_size = width / stage.head_count;
  const mx::array normed = mx::fast::layer_norm(
      x, weights_.Table(prefix + ".norm1.weight"),
      weights_.Table(prefix + ".norm1.bias"), layer_norm_epsilon_);
  const mx::array query_key_value =
      mx::transpose(mx::reshape(weights_.Linear(normed, prefix + ".qkv.weight"),
                                {rows, 3, stage.head_count, head_size}),
                    {1, 2, 0, 3});
  const auto part = [&](int index) {
    return mx::squeeze(
        mx::slice(query_key_value, {index, 0, 0, 0},
                  {index + 1, stage.head_count, rows, head_size}),
        0);
  };
  const mx::array queries =
      mx::fast::rope(part(0), head_size, true, kRopeBase, 1.0f, 0);
  const mx::array keys =
      mx::fast::rope(part(1), head_size, true, kRopeBase, 1.0f, 0);
  const mx::array values = part(2);
  const float scale =
      static_cast<float>(std::pow(static_cast<double>(head_size), -0.5));
  std::vector<mx::array> blocks;
  for (int begin = 0; begin < rows; begin += kAttentionBlockRows) {
    const int end = std::min(begin + kAttentionBlockRows, rows);
    const int left = std::max(0, begin - stage.context + 1);
    const mx::array difference =
        mx::subtract(mx::expand_dims(mx::arange(begin, end), 1),
                     mx::expand_dims(mx::arange(left, end), 0));
    const mx::array mask =
        mx::logical_and(mx::greater_equal(difference, mx::array(0)),
                        mx::less(difference, mx::array(stage.context)));
    const auto window = [&](const mx::array &tensor, int from, int to) {
      return mx::expand_dims(
          mx::slice(tensor, {0, from, 0}, {stage.head_count, to, head_size}),
          0);
    };
    blocks.push_back(
        mx::squeeze(mx::fast::scaled_dot_product_attention(
                        window(queries, begin, end), window(keys, left, end),
                        window(values, left, end), scale, "array", mask),
                    0));
  }
  const mx::array attended =
      blocks.size() == 1 ? blocks[0] : mx::concatenate(blocks, 1);
  const mx::array output =
      mx::reshape(mx::transpose(attended, {1, 0, 2}), {rows, width});
  mx::array y =
      mx::add(x, mx::multiply(weights_.Linear(output, prefix + ".wo.weight"),
                              weights_.Table(prefix + ".attn_scale")));
  const mx::array normed_again = mx::fast::layer_norm(
      y, weights_.Table(prefix + ".norm2.weight"),
      weights_.Table(prefix + ".norm2.bias"), layer_norm_epsilon_);
  const mx::array feedforward = weights_.Linear(
      Gelu(weights_.Linear(normed_again, prefix + ".ffn1.weight")),
      prefix + ".ffn2.weight");
  return mx::add(
      y, mx::multiply(feedforward, weights_.Table(prefix + ".ffn_scale")));
}

mx::array CodecDecoder::Istft(const mx::array &spectrum, int branch) const {
  const int fft_size = fft_sizes_[branch];
  const int bins = fft_size / 2 + 1;
  const int columns = spectrum.shape(0);
  const mx::array magnitude = mx::exp(
      mx::minimum(Columns(spectrum, 0, bins), Scalar(max_log_magnitude_)));
  const mx::array phase = Columns(spectrum, bins, 2 * bins);
  const mx::array real = mx::multiply(magnitude, mx::cos(phase));
  const mx::array imaginary = mx::multiply(magnitude, mx::sin(phase));
  const mx::array bin_index = mx::arange(bins);
  const mx::array edge =
      mx::astype(mx::logical_and(mx::greater(bin_index, mx::array(0)),
                                 mx::less(bin_index, mx::array(bins - 1))),
                 mx::float32);
  const mx::array complex_spectrum =
      mx::add(real, mx::multiply(mx::array(std::complex<float>(0.0f, 1.0f)),
                                 mx::multiply(imaginary, edge)));
  const mx::array frames = mx::multiply(
      mx::astype(mx::fft::irfft(complex_spectrum, fft_size, -1), mx::float32),
      windows_[branch]);
  const int overlap = fft_size / head_hop_;
  const mx::array chunks = mx::reshape(frames, {columns, overlap, head_hop_});
  mx::array overlap_add =
      mx::zeros({columns + overlap - 1, head_hop_}, mx::float32);
  for (int slot = 0; slot < overlap; ++slot) {
    overlap_add =
        mx::add(overlap_add,
                mx::pad(mx::squeeze(mx::slice(chunks, {0, slot, 0},
                                              {columns, slot + 1, head_hop_}),
                                    1),
                        {{slot, overlap - 1 - slot}, {0, 0}}));
  }
  const mx::array squared_window =
      mx::reshape(mx::square(windows_[branch]), {overlap, head_hop_});
  mx::array envelope =
      mx::zeros({columns + overlap - 1, head_hop_}, mx::float32);
  for (int slot = 0; slot < overlap; ++slot) {
    const mx::array rows = Rows(envelope, slot, slot + columns);
    envelope = mx::slice_update(
        envelope, mx::add(rows, Rows(squared_window, slot, slot + 1)),
        {slot, 0}, {slot + columns, head_hop_});
  }
  const int pad = fft_size / 2;
  const int length = columns * head_hop_;
  const mx::array normalized =
      mx::divide(mx::reshape(overlap_add, {-1}),
                 mx::maximum(mx::reshape(envelope, {-1}), Scalar(1e-12f)));
  return mx::slice(normalized, {pad}, {pad + length});
}

mx::array CodecDecoder::Decode(const mx::array &codes) const {
  if (codes.ndim() != 2 || codes.shape(1) != codebook_count_) {
    throw std::invalid_argument("codes must be [frames, " +
                                std::to_string(codebook_count_) + "]");
  } else {
  }
  const int frames = codes.shape(0);
  if (frames == 0) {
    return mx::zeros({0}, mx::float32);
  } else {
  }
  const mx::array ids = mx::astype(codes, mx::int32);
  if (mx::any(mx::logical_or(mx::less(ids, mx::array(0)),
                             mx::greater_equal(ids, mx::array(codebook_size_))))
          .item<bool>()) {
    throw std::invalid_argument("codes contain non-codec tokens");
  } else {
  }
  mx::array x = Latent(ids, nullptr);
  mx::eval(x);
  for (size_t stage_index = 0; stage_index < stages_.size(); ++stage_index) {
    const CodecStage &stage = stages_[stage_index];
    const std::string prefix = "stage." + std::to_string(stage_index);
    x = weights_.Linear(x, prefix + ".in.weight");
    for (int layer = 0; layer < stage.layer_count; ++layer) {
      x = TransformerLayer(x, prefix + ".l" + std::to_string(layer), stage);
      mx::eval(x);
    }
    if (stage.output_size > 0) {
      x = weights_.Linear(x, prefix + ".out.weight");
    } else {
    }
    if (stage.patch > 1) {
      x = PatchUp(x, stage.patch);
    } else {
    }
  }
  const auto [first, second] = Head(x, nullptr);
  const mx::array branches =
      mx::expand_dims(mx::stack({Istft(first, 0), Istft(second, 1)}, -1), 0);
  const int taps = mixer_.shape(1);
  const mx::array mixed = mx::reshape(
      mx::slice(mx::conv1d(branches, mx::expand_dims(mx::transpose(mixer_), 0),
                           1, (taps - 1) / 2),
                {0, 0, 0}, {1, branches.shape(1), 1}),
      {-1});
  mx::array audio = SincResample(mixed, resample_kernel_, resample_from_,
                                 resample_to_, resample_width_);
  mx::eval(audio);
  // Every length is a new set of buffer sizes; without this the cache grows
  // over a long stream of requests.
  mx::clear_cache();
  const int want = frames * hop_;
  if (audio.shape(0) >= want) {
    return mx::slice(audio, {0}, {want});
  } else {
    return mx::pad(audio, {{0, want - audio.shape(0)}});
  }
}

CodecStream::CodecStream(const CodecDecoder &decoder)
    : decoder_(decoder), semantic_histories_(decoder.semantic_depth_),
      head_histories_(decoder.head_depth_), head_(decoder.head_config_),
      resampler_(HostMatrix(decoder.resample_kernel_), decoder.resample_from_,
                 decoder.resample_to_, decoder.resample_width_) {
  for (const CodecStage &stage : decoder.stages_) {
    caches_.emplace_back(stage.layer_count);
  }
}

std::vector<float> CodecStream::Decode(const mx::array &codes) {
  if (codes.ndim() != 2 || codes.shape(1) != decoder_.codebook_count_) {
    throw std::invalid_argument("codes must be [frames, codebooks]");
  } else {
  }
  std::vector<float> samples;
  for (int begin = 0; begin < codes.shape(0); begin += kMaxStreamChunkFrames) {
    const int end = std::min(begin + kMaxStreamChunkFrames, codes.shape(0));
    const std::vector<float> chunk = DecodeChunk(Rows(codes, begin, end));
    samples.insert(samples.end(), chunk.begin(), chunk.end());
  }
  return samples;
}

std::vector<float> CodecStream::DecodeChunk(const mx::array &codes) {
  const Weights &weights = decoder_.weights_;
  mx::array x =
      decoder_.Latent(mx::maximum(mx::astype(codes, mx::int32), mx::array(0)),
                      &semantic_histories_);
  mx::eval(x);
  int positions_per_frame = 1;
  for (size_t stage_index = 0; stage_index < decoder_.stages_.size();
       ++stage_index) {
    const CodecStage &stage = decoder_.stages_[stage_index];
    const std::string stage_prefix = "stage." + std::to_string(stage_index);
    x = weights.Linear(x, stage_prefix + ".in.weight");
    for (int layer = 0; layer < stage.layer_count; ++layer) {
      const std::string prefix = stage_prefix + ".l" + std::to_string(layer);
      const int rows = x.shape(0);
      const int width = x.shape(1);
      const int head_size = width / stage.head_count;
      const int first_position = frames_ * positions_per_frame;
      const mx::array normed = mx::fast::layer_norm(
          x, weights.Table(prefix + ".norm1.weight"),
          weights.Table(prefix + ".norm1.bias"), decoder_.layer_norm_epsilon_);
      const mx::array query_key_value = mx::transpose(
          mx::reshape(weights.Linear(normed, prefix + ".qkv.weight"),
                      {rows, 3, stage.head_count, head_size}),
          {1, 2, 0, 3});
      const auto part = [&](int index) {
        return mx::squeeze(
            mx::slice(query_key_value, {index, 0, 0, 0},
                      {index + 1, stage.head_count, rows, head_size}),
            0);
      };
      const mx::array queries = mx::fast::rope(part(0), head_size, true,
                                               kRopeBase, 1.0f, first_position);
      mx::array keys = mx::fast::rope(part(1), head_size, true, kRopeBase, 1.0f,
                                      first_position);
      mx::array values = part(2);
      auto &cache = caches_[stage_index][layer];
      if (cache.has_value()) {
        keys = mx::concatenate({cache->first, keys}, 1);
        values = mx::concatenate({cache->second, values}, 1);
      } else {
      }
      const int cached = keys.shape(1) - rows;
      const mx::array difference = mx::subtract(
          mx::expand_dims(mx::arange(first_position, first_position + rows), 1),
          mx::expand_dims(
              mx::arange(first_position - cached, first_position + rows), 0));
      const mx::array mask =
          mx::logical_and(mx::greater_equal(difference, mx::array(0)),
                          mx::less(difference, mx::array(stage.context)));
      const float scale =
          static_cast<float>(std::pow(static_cast<double>(head_size), -0.5));
      const mx::array attended =
          mx::squeeze(mx::fast::scaled_dot_product_attention(
                          mx::expand_dims(queries, 0), mx::expand_dims(keys, 0),
                          mx::expand_dims(values, 0), scale, "array", mask),
                      0);
      x = mx::add(
          x, mx::multiply(
                 weights.Linear(mx::reshape(mx::transpose(attended, {1, 0, 2}),
                                            {rows, width}),
                                prefix + ".wo.weight"),
                 weights.Table(prefix + ".attn_scale")));
      const mx::array normed_again = mx::fast::layer_norm(
          x, weights.Table(prefix + ".norm2.weight"),
          weights.Table(prefix + ".norm2.bias"), decoder_.layer_norm_epsilon_);
      const mx::array feedforward = weights.Linear(
          Gelu(weights.Linear(normed_again, prefix + ".ffn1.weight")),
          prefix + ".ffn2.weight");
      x = mx::add(
          x, mx::multiply(feedforward, weights.Table(prefix + ".ffn_scale")));
      const int keep = stage.context - 1;
      if (keep > 0) {
        const int total = keys.shape(1);
        const int from = std::max(0, total - keep);
        cache = std::make_pair(
            mx::slice(keys, {0, from, 0}, {stage.head_count, total, head_size}),
            mx::slice(values, {0, from, 0},
                      {stage.head_count, total, head_size}));
        mx::eval(x, cache->first, cache->second);
      } else {
        cache.reset();
        mx::eval(x);
      }
    }
    if (stage.output_size > 0) {
      x = weights.Linear(x, stage_prefix + ".out.weight");
    } else {
    }
    if (stage.patch > 1) {
      x = PatchUp(x, stage.patch);
      positions_per_frame *= stage.patch;
    } else {
    }
  }
  const auto [first, second] = decoder_.Head(x, &head_histories_);
  mx::eval(first, second);
  for (const auto &history : semantic_histories_) {
    if (history.has_value()) {
      mx::eval(*history);
    } else {
    }
  }
  for (const auto &history : head_histories_) {
    if (history.has_value()) {
      mx::eval(*history);
    } else {
    }
  }
  const std::vector<float> internal =
      head_.Push({HostRows(first), HostRows(second)});
  frames_ += codes.shape(0);
  return resampler_.Push(internal);
}

std::vector<float> CodecStream::Flush() {
  std::vector<float> samples = resampler_.Push(head_.Flush());
  const std::vector<float> rest = resampler_.Flush();
  samples.insert(samples.end(), rest.begin(), rest.end());
  mx::clear_cache();
  return samples;
}

} // namespace zonos3
