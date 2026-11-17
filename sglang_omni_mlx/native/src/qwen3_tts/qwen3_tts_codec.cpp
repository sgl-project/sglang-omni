// SPDX-License-Identifier: Apache-2.0
#include "qwen3_tts/qwen3_tts_codec.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <mutex>
#include <optional>
#include <stdexcept>

#include <mlx/memory.h>

namespace qwen3_tts {

namespace mx = mlx::core;

namespace {

constexpr const char *kPrefix = "decoder";
constexpr float kCodebookUsageEps = 1e-5f;
constexpr float kSnakeEps = 1e-9f;
constexpr float kConvNextNormEps = 1e-6f;
constexpr std::array<int, 3> kResidualDilations = {1, 3, 9};
// Note (Jiaxin Deng): the CUDA vocoder decodes a whole request in 300-frame
// chunks, each with up to 25 frames of left context.
constexpr int kChunkFrames = 300;
constexpr int kLeftContextFrames = 25;
constexpr int kStreamedChunkFrames = 25;

// Note (Codex): on M5 Pro, this limit reduced long-chunk footprint without
// slowing generation. Note (Codex): MLX cache limits are process-wide; the
// native server owns one model worker.
constexpr size_t kCodecCacheLimitBytes = 512ULL * 1024 * 1024;
std::mutex codec_cache_limit_mutex;

class ScopedCodecCacheLimit {
public:
  explicit ScopedCodecCacheLimit(bool enabled)
      : lock_(codec_cache_limit_mutex), enabled_(enabled) {
    if (enabled_) {
      previous_cache_limit_bytes_ = mx::set_cache_limit(0);
      mx::set_cache_limit(
          std::min(previous_cache_limit_bytes_, kCodecCacheLimitBytes));
    } else {
    }
  }
  ~ScopedCodecCacheLimit() {
    if (enabled_) {
      mx::set_cache_limit(previous_cache_limit_bytes_);
    } else {
    }
  }
  ScopedCodecCacheLimit(const ScopedCodecCacheLimit &) = delete;
  ScopedCodecCacheLimit &operator=(const ScopedCodecCacheLimit &) = delete;

private:
  std::unique_lock<std::mutex> lock_;
  bool enabled_;
  size_t previous_cache_limit_bytes_ = 0;
};

bool EndsWith(const std::string &text, const std::string &suffix) {
  return text.size() >= suffix.size() &&
         text.compare(text.size() - suffix.size(), suffix.size(), suffix) == 0;
}

} // namespace

omni_common::WeightMap
SpeechDecoder::PrepareWeights(omni_common::WeightMap weights) {
  omni_common::WeightMap prepared;
  const std::string usage_suffix = "._codebook.cluster_usage";
  for (auto &[name, array] : weights) {
    if (name.rfind(std::string(kPrefix) + ".", 0) != 0) {
      continue;
    } else if (EndsWith(name, usage_suffix)) {
      const std::string base =
          name.substr(0, name.size() - usage_suffix.size());
      const mx::array &sums = weights.at(base + "._codebook.embedding_sum");
      prepared.insert_or_assign(
          base + ".codebook.embed.weight",
          mx::divide(sums,
                     mx::expand_dims(
                         mx::maximum(array, mx::array(kCodebookUsageEps)), 1)));
      continue;
    } else if (EndsWith(name, "._codebook.embedding_sum")) {
      continue;
    } else {
    }
    // Checkpoint kernels are [out, in, kernel], and a transposed
    // convolution's [in, out, kernel]; MLX wants [out, kernel, in].
    const bool transposed =
        array.ndim() == 3 && ((name.find(".upsample.") != std::string::npos &&
                               EndsWith(name, ".0.conv.weight")) ||
                              EndsWith(name, ".block.1.conv.weight"));
    if (transposed) {
      prepared.insert_or_assign(name, mx::transpose(array, {1, 2, 0}));
    } else if (array.ndim() == 3) {
      prepared.insert_or_assign(name, mx::transpose(array, {0, 2, 1}));
    } else {
      prepared.insert_or_assign(name, array);
    }
  }
  return prepared;
}

SpeechDecoder::SpeechDecoder(
    const std::filesystem::path &speech_tokenizer_directory)
    : SpeechDecoder(
          omni_common::ReadJson(speech_tokenizer_directory / "config.json"),
          omni_common::LoadSafetensors(speech_tokenizer_directory)) {}

SpeechDecoder::SpeechDecoder(const nlohmann::json &config,
                             omni_common::WeightMap weights)
    : checkpoint_(PrepareWeights(std::move(weights)),
                  nlohmann::json::object()) {
  const nlohmann::json &decoder = config.at("decoder_config");
  codebook_count_ = decoder.at("num_quantizers").get<int>();
  semantic_codebook_count_ = decoder.at("num_semantic_quantizers").get<int>();
  head_count_ = decoder.at("num_attention_heads").get<int>();
  head_dim_ = decoder.at("head_dim").get<int>();
  transformer_layer_count_ = decoder.at("num_hidden_layers").get<int>();
  rms_norm_eps_ = decoder.at("rms_norm_eps").get<float>();
  sliding_window_frames_ = decoder.at("sliding_window").get<int>();
  upsampling_ratios_ = decoder.at("upsampling_ratios").get<std::vector<int>>();
  upsample_rates_ = decoder.at("upsample_rates").get<std::vector<int>>();
  samples_per_frame_ = config.at("decode_upsample_rate").get<int>();
  sample_rate_ = config.at("output_sample_rate").get<int>();
  if (decoder.at("num_key_value_heads").get<int>() != head_count_) {
    throw std::runtime_error("the speech decoder's attention must have as "
                             "many key/value heads as query heads");
  } else {
  }
  rope_theta_ = decoder.at("rope_theta").get<float>();
}

mx::array SpeechDecoder::Snake(const mx::array &x,
                               const std::string &prefix) const {
  const mx::array alpha = mx::exp(checkpoint_.Weight(prefix + ".alpha"));
  const mx::array beta = mx::exp(checkpoint_.Weight(prefix + ".beta"));
  return mx::add(
      x, mx::multiply(
             mx::divide(mx::array(1.0f), mx::add(beta, mx::array(kSnakeEps))),
             mx::power(mx::sin(mx::multiply(x, alpha)), mx::array(2.0f))));
}

mx::array SpeechDecoder::CausalConvolution(const mx::array &x,
                                           const std::string &prefix,
                                           int dilation, int groups) const {
  const mx::array &weight = checkpoint_.Weight(prefix + ".weight");
  const int padding = (weight.shape(1) - 1) * dilation;
  const mx::array padded =
      padding > 0 ? mx::pad(x, {{0, 0}, {padding, 0}, {0, 0}}) : x;
  const mx::array y = mx::conv1d(padded, weight, 1, 0, dilation, groups);
  if (checkpoint_.Has(prefix + ".bias")) {
    return mx::add(y, checkpoint_.Weight(prefix + ".bias"));
  } else {
    return y;
  }
}

mx::array SpeechDecoder::TransposedConvolution(const mx::array &x,
                                               const std::string &prefix,
                                               int stride) const {
  const mx::array &weight = checkpoint_.Weight(prefix + ".weight");
  const mx::array y = mx::add(mx::conv_transpose1d(x, weight, stride),
                              checkpoint_.Weight(prefix + ".bias"));
  const int trim = weight.shape(1) - stride;
  if (trim > 0) {
    return mx::slice(y, {0, 0, 0}, {y.shape(0), y.shape(1) - trim, y.shape(2)});
  } else {
    return y;
  }
}

mx::array SpeechDecoder::TransformerLayer(const mx::array &x, int layer) const {
  const std::string prefix =
      std::string(kPrefix) + ".pre_transformer.layers." + std::to_string(layer);
  const int batch = x.shape(0);
  const int length = x.shape(1);
  const mx::array normed =
      checkpoint_.RmsNorm(x, prefix + ".input_layernorm", rms_norm_eps_);
  const auto heads = [&](const std::string &projection) {
    return mx::transpose(
        mx::reshape(
            checkpoint_.Linear(normed, prefix + ".self_attn." + projection),
            {batch, length, head_count_, head_dim_}),
        {0, 2, 1, 3});
  };
  const auto rotate = [&](const mx::array &projected) {
    return mx::fast::rope(projected, head_dim_, false, rope_theta_, 1.0f, 0);
  };
  // Note (Jiaxin Deng): the decoder's attention is causal within its
  // sliding window, as the CUDA vocoder's; a full causal mask changes the
  // samples of chunks longer than the window.
  const mx::array attended = mx::fast::scaled_dot_product_attention(
      rotate(heads("q_proj")), rotate(heads("k_proj")), heads("v_proj"),
      static_cast<float>(std::pow(static_cast<double>(head_dim_), -0.5)),
      sliding_window_frames_, "causal");
  const mx::array attention_output = checkpoint_.Linear(
      mx::reshape(mx::transpose(attended, {0, 2, 1, 3}), {batch, length, -1}),
      prefix + ".self_attn.o_proj");
  const mx::array hidden = mx::add(
      x,
      mx::multiply(checkpoint_.Weight(prefix + ".self_attn_layer_scale.scale"),
                   attention_output));
  const mx::array mlp_input = checkpoint_.RmsNorm(
      hidden, prefix + ".post_attention_layernorm", rms_norm_eps_);
  const mx::array mlp_output = checkpoint_.Linear(
      omni_common::Swiglu(
          checkpoint_.Linear(mlp_input, prefix + ".mlp.gate_proj"),
          checkpoint_.Linear(mlp_input, prefix + ".mlp.up_proj")),
      prefix + ".mlp.down_proj");
  return mx::add(hidden, mx::multiply(checkpoint_.Weight(
                                          prefix + ".mlp_layer_scale.scale"),
                                      mlp_output));
}

mx::array SpeechDecoder::Decode(const mx::array &codes) const {
  const std::string prefix = kPrefix;
  const int frame_count = codes.shape(0);
  // [1, frames] per codebook; each quantizer sums its codebooks' rows and
  // projects them back to the codec's dimension.
  const auto quantized = [&](const std::string &quantizer, int first_codebook,
                             int codebook_count) {
    const std::string quantizer_prefix = prefix + ".quantizer." + quantizer;
    std::optional<mx::array> sum;
    for (int codebook = 0; codebook < codebook_count; ++codebook) {
      const mx::array ids =
          mx::reshape(mx::slice(codes, {0, first_codebook + codebook},
                                {frame_count, first_codebook + codebook + 1}),
                      {1, frame_count});
      const mx::array rows = checkpoint_.Embed(
          ids, quantizer_prefix + ".vq.layers." + std::to_string(codebook) +
                   ".codebook.embed");
      sum = sum.has_value()
                ? mx::add(*sum, rows)
                : mx::add(mx::zeros(rows.shape(), mx::float32), rows);
    }
    return mx::conv1d(
        *sum, checkpoint_.Weight(quantizer_prefix + ".output_proj.weight"));
  };
  mx::array hidden = quantized("rvq_first", 0, semantic_codebook_count_);
  if (codebook_count_ > semantic_codebook_count_) {
    hidden =
        mx::add(hidden, quantized("rvq_rest", semantic_codebook_count_,
                                  codebook_count_ - semantic_codebook_count_));
  } else {
  }
  hidden = CausalConvolution(hidden, prefix + ".pre_conv.conv", 1, 1);

  const std::string transformer = prefix + ".pre_transformer";
  hidden = checkpoint_.Linear(hidden, transformer + ".input_proj");
  for (int layer = 0; layer < transformer_layer_count_; ++layer) {
    hidden = TransformerLayer(hidden, layer);
  }
  hidden = checkpoint_.Linear(
      checkpoint_.RmsNorm(hidden, transformer + ".norm", rms_norm_eps_),
      transformer + ".output_proj");

  for (size_t stage = 0; stage < upsampling_ratios_.size(); ++stage) {
    const std::string stage_prefix =
        prefix + ".upsample." + std::to_string(stage);
    hidden = TransposedConvolution(hidden, stage_prefix + ".0.conv",
                                   upsampling_ratios_[stage]);
    const std::string block = stage_prefix + ".1";
    const mx::array residual = hidden;
    mx::array x =
        CausalConvolution(hidden, block + ".dwconv.conv", 1, hidden.shape(2));
    x = checkpoint_.LayerNorm(x, block + ".norm", kConvNextNormEps);
    x = omni_common::Gelu(checkpoint_.Linear(x, block + ".pwconv1"));
    x = checkpoint_.Linear(x, block + ".pwconv2");
    hidden = mx::add(residual,
                     mx::multiply(checkpoint_.Weight(block + ".gamma"), x));
  }

  const std::string stack = prefix + ".decoder";
  mx::array wave = CausalConvolution(hidden, stack + ".0.conv", 1, 1);
  for (size_t stage = 0; stage < upsample_rates_.size(); ++stage) {
    const std::string block =
        stack + "." + std::to_string(stage + 1) + ".block";
    wave = Snake(wave, block + ".0");
    wave =
        TransposedConvolution(wave, block + ".1.conv", upsample_rates_[stage]);
    for (size_t unit = 0; unit < kResidualDilations.size(); ++unit) {
      const std::string unit_prefix = block + "." + std::to_string(unit + 2);
      mx::array x = Snake(wave, unit_prefix + ".act1");
      x = CausalConvolution(x, unit_prefix + ".conv1.conv",
                            kResidualDilations[unit], 1);
      x = Snake(x, unit_prefix + ".act2");
      x = CausalConvolution(x, unit_prefix + ".conv2.conv", 1, 1);
      wave = mx::add(x, wave);
    }
  }
  const int output_stage = static_cast<int>(upsample_rates_.size()) + 1;
  wave = Snake(wave, stack + "." + std::to_string(output_stage));
  wave = CausalConvolution(
      wave, stack + "." + std::to_string(output_stage + 1) + ".conv", 1, 1);
  return mx::reshape(mx::clip(wave, mx::array(-1.0f), mx::array(1.0f)), {-1});
}

SpeechStream::SpeechStream(const SpeechDecoder &decoder, bool streamed,
                           int first_chunk_frames)
    : decoder_(decoder), streamed_(streamed),
      first_chunk_frames_(
          first_chunk_frames > 0
              ? std::min(first_chunk_frames, kStreamedChunkFrames)
              : kStreamedChunkFrames) {}

int SpeechStream::ChunkFrames() const {
  if (!streamed_) {
    return kChunkFrames;
  } else if (emitted_frames_ == 0) {
    return first_chunk_frames_;
  } else {
    return kStreamedChunkFrames;
  }
}

std::vector<float> SpeechStream::Push(const mx::array &frame_codes) {
  frames_.push_back(frame_codes);
  if (frame_codes.data<int32_t>()[0] > 0) {
    ++voiced_frames_;
  } else {
  }
  const int chunk_frames = ChunkFrames();
  if (static_cast<int>(frames_.size()) - emitted_frames_ < chunk_frames) {
    return {};
  } else if (streamed_) {
    return DecodeChunk(emitted_frames_ + chunk_frames);
  } else {
    const std::vector<float> chunk =
        DecodeChunk(emitted_frames_ + chunk_frames);
    held_samples_.insert(held_samples_.end(), chunk.begin(), chunk.end());
    return {};
  }
}

std::vector<float> SpeechStream::Finish() {
  std::vector<float> samples = std::move(held_samples_);
  while (emitted_frames_ < static_cast<int>(frames_.size())) {
    const std::vector<float> chunk = DecodeChunk(std::min(
        static_cast<int>(frames_.size()), emitted_frames_ + ChunkFrames()));
    samples.insert(samples.end(), chunk.begin(), chunk.end());
  }
  // Note (Jiaxin Deng): as the CUDA vocoder's whole-request decode, the audio
  // keeps one frame of samples per frame whose first code is not 0.
  const size_t kept_samples =
      static_cast<size_t>(voiced_frames_) * decoder_.samples_per_frame();
  if (!streamed_ && kept_samples < samples.size()) {
    samples.resize(kept_samples);
  } else {
  }
  return samples;
}

std::vector<float> SpeechStream::DecodeChunk(int end_frame) {
  const ScopedCodecCacheLimit cache_limit(!streamed_);
  const int context_frames = std::min(kLeftContextFrames, emitted_frames_);
  const int start_frame = emitted_frames_ - context_frames;
  const std::vector<mx::array> window(frames_.begin() + start_frame,
                                      frames_.begin() + end_frame);
  const int samples_per_frame = decoder_.samples_per_frame();
  const mx::array audio =
      mx::slice(decoder_.Decode(mx::stack(window, 0)),
                {context_frames * samples_per_frame},
                {(end_frame - start_frame) * samples_per_frame});
  mx::eval(audio);
  emitted_frames_ = end_frame;
  std::vector<float> samples(audio.data<float>(),
                             audio.data<float>() + audio.size());
  // Note (Jiaxin Deng): a chunk's upsampling buffers run to gigabytes and
  // are not needed again until the next chunk, so MLX's cache returns them.
  mx::clear_cache();
  return samples;
}

} // namespace qwen3_tts
