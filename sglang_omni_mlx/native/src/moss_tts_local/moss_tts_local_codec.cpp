// SPDX-License-Identifier: Apache-2.0
#include "moss_tts_local/moss_tts_local_codec.h"

#include <algorithm>
#include <cmath>
#include <set>
#include <stdexcept>

namespace moss_tts_local {

namespace mx = mlx::core;

namespace {

constexpr float kLayerNormEps = 1e-5f;
constexpr int kChunkFrames = 25;
constexpr float kPcm16FullScale = 32767.0f;

// A weight-normed 1x1 convolution's kernel g * v / |v|, in MLX's [out, 1, in]
// layout.
mx::array WeightNormKernel(const mx::array &magnitude,
                           const mx::array &direction) {
  const mx::array norm =
      mx::sqrt(mx::sum(mx::square(direction), std::vector<int>{1, 2}, true));
  return mx::transpose(mx::divide(mx::multiply(magnitude, direction), norm),
                       {0, 2, 1});
}

// A 1x1 convolution over x [batch, channels, frames], as the weight-normed
// Conv1d it was trained as.
mx::array PointwiseConvolution(const omni_common::Checkpoint &checkpoint,
                               const mx::array &x, const std::string &prefix) {
  const mx::array y = mx::conv1d(mx::transpose(x, {0, 2, 1}),
                                 checkpoint.Weight(prefix + ".weight"));
  return mx::transpose(mx::add(y, checkpoint.Weight(prefix + ".bias")),
                       {0, 2, 1});
}

// x times value as a scalar of x's dtype, as Python scalars combine.
mx::array Scale(const mx::array &x, double value) {
  return mx::multiply(mx::array(static_cast<float>(value), x.dtype()), x);
}

mx::array ExactGelu(const mx::array &x) {
  const mx::Dtype dtype = x.dtype();
  return mx::multiply(
      Scale(x, 0.5),
      mx::add(mx::array(1.0f, dtype),
              mx::erf(mx::divide(
                  x, mx::array(static_cast<float>(std::sqrt(2.0)), dtype)))));
}

// Rotary embedding of adjacent pairs at positions offset onward, in float32.
mx::array Rotate(const mx::array &x, int offset, float max_period) {
  const int head_dim = x.shape(3);
  const int frames = x.shape(2);
  const mx::array frequencies = mx::exp(mx::multiply(
      mx::arange(static_cast<double>(head_dim / 2), mx::float32),
      mx::array(static_cast<float>(-std::log(static_cast<double>(max_period)) *
                                   2.0 / head_dim))));
  const mx::array positions =
      mx::arange(static_cast<double>(offset),
                 static_cast<double>(offset + frames), mx::float32);
  const mx::array phase =
      mx::multiply(mx::reshape(positions, {1, 1, frames, 1}),
                   mx::reshape(frequencies, {1, 1, 1, head_dim / 2}));
  const mx::array cosine = mx::cos(phase);
  const mx::array sine = mx::sin(phase);
  mx::Shape paired = x.shape();
  paired.back() = head_dim / 2;
  paired.push_back(2);
  const mx::array pairs = mx::reshape(mx::astype(x, mx::float32), paired);
  mx::Shape half = paired;
  half.back() = 1;
  mx::Shape start(paired.size(), 0);
  const mx::array real = mx::squeeze(mx::slice(pairs, start, half), -1);
  start.back() = 1;
  mx::Shape stop = paired;
  const mx::array imaginary = mx::squeeze(mx::slice(pairs, start, stop), -1);
  const mx::array rotated = mx::stack(
      {mx::subtract(mx::multiply(real, cosine), mx::multiply(imaginary, sine)),
       mx::add(mx::multiply(real, sine), mx::multiply(imaginary, cosine))},
      -1);
  return mx::astype(mx::reshape(rotated, x.shape()), x.dtype());
}

omni_common::WeightMap
LoadDecoderWeights(const std::filesystem::path &directory,
                   mx::Dtype transformer_dtype) {
  omni_common::WeightMap raw;
  omni_common::WeightMap checkpoint = omni_common::LoadSafetensors(directory);
  while (!checkpoint.empty()) {
    // Note (Jiaxin Deng): each tensor leaves the map as it is cast, so the
    // float32 checkpoint is never resident at once.
    auto entry = checkpoint.extract(checkpoint.begin());
    const std::string &name = entry.key();
    const mx::array &array = entry.mapped();
    if (name.rfind("decoder.", 0) == 0) {
      mx::array cast = mx::astype(array, transformer_dtype);
      mx::eval(cast);
      raw.insert_or_assign(name, cast);
    } else if (name.rfind("quantizer.", 0) == 0) {
      raw.insert_or_assign(name, array);
    } else {
    }
  }
  omni_common::WeightMap weights;
  for (auto &[name, array] : raw) {
    const std::string suffix = ".parametrizations.weight.original1";
    if (name.size() > suffix.size() &&
        name.compare(name.size() - suffix.size(), suffix.size(), suffix) == 0) {
      const std::string prefix = name.substr(0, name.size() - suffix.size());
      weights.insert_or_assign(
          prefix + ".weight",
          WeightNormKernel(
              raw.at(prefix + ".parametrizations.weight.original0"), array));
    } else if (name.find(".parametrizations.") == std::string::npos) {
      weights.insert_or_assign(name, array);
    } else {
    }
  }
  return weights;
}

} // namespace

AudioCodec::AudioCodec(const std::filesystem::path &codec_directory,
                       mx::Dtype transformer_dtype)
    : checkpoint_(LoadDecoderWeights(codec_directory, transformer_dtype),
                  nlohmann::json::object()),
      transformer_dtype_(transformer_dtype) {
  const nlohmann::json config =
      omni_common::ReadJson(codec_directory / "config.json");
  sample_rate_ = config.at("sampling_rate").get<int>();
  channel_count_ = config.at("number_channels").get<int>();
  if (!config.value("enable_channel_interleave", true) || channel_count_ < 1) {
    throw std::runtime_error("the codec must interleave its channels");
  } else {
  }
  codebook_count_ =
      config.at("quantizer_kwargs").at("num_quantizers").get<int>();
  const int downsample_rate = config.at("downsample_rate").get<int>();
  // Note (Jiaxin Deng): as the codec's builder, each transformer attends over
  // its context duration at the frame rate it runs at.
  double frame_rate = static_cast<double>(sample_rate_ * channel_count_) /
                      (static_cast<double>(downsample_rate) * channel_count_);
  int interleaved_samples_per_frame = 1;
  for (const nlohmann::json &module : config.at("decoder_kwargs")) {
    const std::string module_type = module.at("module_type").get<std::string>();
    Stage stage;
    if (module_type == "PatchedPretransform") {
      stage.kind = StageKind::kPatch;
      stage.patch_size = module.at("patch_size").get<int>();
      frame_rate *= stage.patch_size;
      interleaved_samples_per_frame *= stage.patch_size;
    } else if (module_type == "Transformer") {
      if (!module.at("causal").get<bool>() ||
          module.at("positional_embedding") != "rope" ||
          module.value("gating", std::string("none")) != "none" ||
          module.value("norm", std::string("layer_norm")) != "layer_norm") {
        throw std::runtime_error("unsupported codec transformer");
      } else {
      }
      stage.kind = StageKind::kTransformer;
      stage.layer_count = module.at("num_layers").get<int>();
      stage.head_count = module.at("num_heads").get<int>();
      stage.model_dimension = module.at("d_model").get<int>();
      stage.context_frames = static_cast<int>(std::lround(
          frame_rate *
          module.value("context_duration", config
                                               .at("causal_transformer_context_"
                                                   "duration")
                                               .get<double>())));
      if (module.value("max_period", 10000.0) != 10000.0) {
        throw std::runtime_error("unsupported codec rotary period");
      } else {
      }
    } else {
      throw std::runtime_error("unsupported codec module " + module_type);
    }
    stages_.push_back(stage);
  }
  samples_per_frame_ = interleaved_samples_per_frame / channel_count_;
}

AudioCodec::StageCaches AudioCodec::NewCaches() const {
  StageCaches caches;
  for (const Stage &stage : stages_) {
    caches.emplace_back(stage.layer_count);
  }
  return caches;
}

mx::array AudioCodec::DequantizeCodes(const mx::array &codes) const {
  const int codebooks = codes.shape(0);
  const int frames = codes.shape(1);
  if (codebooks > codebook_count_) {
    throw std::invalid_argument("more codebooks than the codec has");
  } else {
  }
  const mx::array output_projection =
      checkpoint_.Weight("quantizer.output_proj.weight");
  const int rvq_dimension = output_projection.shape(2);
  mx::array embedding = mx::zeros({1, rvq_dimension, frames}, mx::float32);
  for (int codebook = 0; codebook < codebooks; ++codebook) {
    const std::string prefix =
        "quantizer.quantizers." + std::to_string(codebook);
    const mx::array ids =
        mx::slice(codes, {codebook, 0}, {codebook + 1, frames});
    const mx::array entries =
        mx::transpose(checkpoint_.Embed(ids, prefix + ".codebook"), {0, 2, 1});
    embedding = mx::add(
        embedding, mx::astype(PointwiseConvolution(
                                  checkpoint_, mx::astype(entries, mx::float32),
                                  prefix + ".out_proj"),
                              mx::float32));
  }
  return mx::astype(
      PointwiseConvolution(checkpoint_, embedding, "quantizer.output_proj"),
      mx::float32);
}

mx::array AudioCodec::AttentionStep(const mx::array &x,
                                    const std::string &prefix,
                                    const Stage &stage,
                                    AttentionCache &cache) const {
  const int frames = x.shape(1);
  const int head_dim = stage.model_dimension / stage.head_count;
  const mx::array qkv = mx::reshape(checkpoint_.Linear(x, prefix + ".in_proj"),
                                    {1, frames, 3, stage.head_count, head_dim});
  const auto part = [&](int index) {
    return mx::transpose(mx::squeeze(mx::slice(qkv, {0, 0, index, 0, 0},
                                               {1, frames, index + 1,
                                                stage.head_count, head_dim}),
                                     2),
                         {0, 2, 1, 3});
  };
  const mx::array queries = Rotate(part(0), cache.offset, 10000.0f);
  const mx::array new_keys = Rotate(part(1), cache.offset, 10000.0f);
  const mx::array new_values = part(2);
  const mx::array keys = cache.keys.has_value()
                             ? mx::concatenate({*cache.keys, new_keys}, 2)
                             : new_keys;
  const mx::array values = cache.values.has_value()
                               ? mx::concatenate({*cache.values, new_values}, 2)
                               : new_values;
  const int key_start = cache.keys.has_value()
                            ? cache.offset - cache.keys->shape(2)
                            : cache.offset;
  const int key_count = keys.shape(2);
  const mx::array distance = mx::subtract(
      mx::reshape(mx::arange(cache.offset, cache.offset + frames, mx::int32),
                  {1, 1, frames, 1}),
      mx::reshape(mx::arange(key_start, key_start + key_count, mx::int32),
                  {1, 1, 1, key_count}));
  const mx::array allowed =
      mx::logical_and(mx::greater_equal(distance, mx::array(0)),
                      mx::less(distance, mx::array(stage.context_frames)));
  const mx::array mask = mx::astype(
      mx::where(allowed, mx::array(0.0f),
                mx::array(static_cast<float>(mx::finfo(x.dtype()).min))),
      x.dtype());
  const mx::array attended = mx::fast::scaled_dot_product_attention(
      queries, keys, values,
      static_cast<float>(std::pow(static_cast<double>(head_dim), -0.5)), "",
      mask);
  cache.offset += frames;
  const int kept = std::min(key_count, stage.context_frames - 1);
  if (kept <= 0) {
    cache.keys.reset();
    cache.values.reset();
  } else {
    const mx::Shape stop = {1, stage.head_count, key_count, head_dim};
    cache.keys = mx::slice(keys, {0, 0, key_count - kept, 0}, stop);
    cache.values = mx::slice(values, {0, 0, key_count - kept, 0}, stop);
  }
  return checkpoint_.Linear(mx::reshape(mx::transpose(attended, {0, 2, 1, 3}),
                                        {1, frames, stage.model_dimension}),
                            prefix + ".out_proj");
}

mx::array
AudioCodec::TransformerStep(const mx::array &x, int stage_index,
                            std::vector<AttentionCache> &caches) const {
  const Stage &stage = stages_[stage_index];
  const std::string prefix = "decoder." + std::to_string(stage_index);
  mx::array hidden = checkpoint_.Linear(
      mx::astype(mx::transpose(x, {0, 2, 1}), transformer_dtype_),
      prefix + ".input_proj");
  for (int layer = 0; layer < stage.layer_count; ++layer) {
    const std::string layer_prefix =
        prefix + ".transformer.layers." + std::to_string(layer);
    const mx::array attended = AttentionStep(
        checkpoint_.LayerNorm(hidden, layer_prefix + ".norm1", kLayerNormEps),
        layer_prefix + ".self_attn", stage, caches[layer]);
    hidden = mx::add(
        hidden,
        mx::multiply(checkpoint_.Weight(layer_prefix + ".layer_scale_1.scale"),
                     attended));
    const mx::array feed_forward = checkpoint_.Linear(
        ExactGelu(checkpoint_.Linear(
            checkpoint_.LayerNorm(hidden, layer_prefix + ".norm2",
                                  kLayerNormEps),
            layer_prefix + ".ffn.0")),
        layer_prefix + ".ffn.2");
    hidden = mx::add(
        hidden,
        mx::multiply(checkpoint_.Weight(layer_prefix + ".layer_scale_2.scale"),
                     feed_forward));
  }
  return mx::transpose(checkpoint_.Linear(hidden, prefix + ".output_proj"),
                       {0, 2, 1});
}

mx::array AudioCodec::DecodeStep(const mx::array &codes,
                                 StageCaches &caches) const {
  mx::array audio = DequantizeCodes(codes);
  for (size_t index = 0; index < stages_.size(); ++index) {
    const Stage &stage = stages_[index];
    if (stage.kind == StageKind::kTransformer) {
      audio = TransformerStep(audio, static_cast<int>(index), caches[index]);
    } else {
      const int channels = audio.shape(1) / stage.patch_size;
      const int frames = audio.shape(2);
      audio = mx::reshape(
          mx::transpose(
              mx::reshape(audio, {1, channels, stage.patch_size, frames}),
              {0, 1, 3, 2}),
          {1, channels, frames * stage.patch_size});
    }
  }
  return mx::reshape(mx::astype(audio, mx::float32), {-1});
}

CodecStream::CodecStream(const AudioCodec &codec, int first_chunk_frames)
    : codec_(codec), caches_(codec.NewCaches()),
      first_chunk_frames_(first_chunk_frames) {}

std::vector<float> CodecStream::Push(const mx::array &frame_codes) {
  pending_frames_.push_back(frame_codes);
  const int chunk_frames =
      emitted_first_chunk_ ? kChunkFrames : first_chunk_frames_;
  if (static_cast<int>(pending_frames_.size()) < chunk_frames) {
    return {};
  } else {
    emitted_first_chunk_ = true;
    return DecodePending();
  }
}

std::vector<float> CodecStream::Finish() {
  if (pending_frames_.empty()) {
    return {};
  } else {
    return DecodePending();
  }
}

std::vector<float> CodecStream::DecodePending() {
  const mx::array codes = mx::transpose(mx::stack(pending_frames_, 0), {1, 0});
  pending_frames_.clear();
  const mx::array samples = codec_.DecodeStep(codes, caches_);
  mx::eval(samples);
  const float *data = samples.data<float>();
  return std::vector<float>(data, data + samples.size());
}

std::string Pcm16Bytes(const std::vector<float> &samples) {
  std::string bytes;
  bytes.reserve(samples.size() * 2);
  for (const float sample : samples) {
    // Note (Minyi Tang): NaN becomes silence, since std::clamp passes it
    // through and casting NaN to an integer is undefined.
    const float bounded =
        std::isnan(sample) ? 0.0f : std::clamp(sample, -1.0f, 1.0f);
    const auto value = static_cast<int16_t>(bounded * kPcm16FullScale);
    bytes.push_back(static_cast<char>(value & 0xff));
    bytes.push_back(static_cast<char>((value >> 8) & 0xff));
  }
  return bytes;
}

std::string WavBytes(const std::string &pcm, int sample_rate,
                     int channel_count) {
  std::string bytes;
  const auto append = [&](uint32_t value, int size) {
    for (int index = 0; index < size; ++index) {
      bytes.push_back(static_cast<char>((value >> (8 * index)) & 0xff));
    }
  };
  const auto data_size = static_cast<uint32_t>(pcm.size());
  bytes += "RIFF";
  append(36 + data_size, 4);
  bytes += "WAVEfmt ";
  append(16, 4); // fmt chunk size
  append(1, 2);  // PCM
  append(channel_count, 2);
  append(sample_rate, 4);
  append(sample_rate * channel_count * 2, 4); // bytes per second
  append(channel_count * 2, 2);               // bytes per sample frame
  append(16, 2);                              // bits per sample
  bytes += "data";
  append(data_size, 4);
  return bytes + pcm;
}

} // namespace moss_tts_local
