// SPDX-License-Identifier: Apache-2.0
#include "ming_tts/ming_tts_audio.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <complex>
#include <stdexcept>

#include "ming_tts/ming_tts_model.h"
#include "ming_tts/ming_tts_request.h"

namespace ming_tts {
namespace mx = mlx::core;

AudioTransformer::AudioTransformer(const omni_common::Checkpoint &checkpoint,
                                   std::string prefix,
                                   const nlohmann::json &config)
    : checkpoint_(checkpoint), prefix_(std::move(prefix)),
      hidden_size_(config.at("hidden_size")),
      layers_(config.at("num_hidden_layers")),
      heads_(config.at("num_attention_heads")),
      kv_heads_(config.value("num_key_value_heads", nlohmann::json()).is_null()
                    ? heads_
                    : config.at("num_key_value_heads").get<int>()),
      epsilon_(config.value("rms_norm_eps", 1e-6f)),
      rope_theta_(config.value("rope_theta", 1000000.0f)) {
  if (config.value("hidden_act", std::string("silu")) != "silu") {
    throw std::invalid_argument("AudioVAE native supports SiLU only");
  } else {
  }
  const auto rope = config.value(
      "rope_parameters", config.value("rope_scaling", nlohmann::json()));
  if (!rope.is_null()) {
    if (rope.value("rope_type", rope.value("type", std::string("default"))) !=
        "default") {
      throw std::invalid_argument("AudioVAE native supports default RoPE only");
    } else {
      rope_theta_ = rope.value("rope_theta", rope_theta_);
    }
  } else {
  }
  const int window =
      config.value("use_sliding_window", false) &&
              !config.value("sliding_window", nlohmann::json(4096)).is_null()
          ? config.value("sliding_window", 4096)
          : 0;
  const auto layer_types = config.value("layer_types", nlohmann::json());
  for (int layer = 0; layer < layers_; ++layer) {
    const std::string kind =
        layer_types.is_null()
            ? (window > 0 && layer >= config.value("max_window_layers", 28)
                   ? "sliding_attention"
                   : "full_attention")
            : layer_types.at(layer).get<std::string>();
    if (kind == "full_attention") {
      windows_.push_back(0);
    } else if (kind == "sliding_attention" && window > 0) {
      windows_.push_back(window);
    } else {
      throw std::invalid_argument("unsupported AudioVAE attention window");
    }
  }
}

mx::array AudioTransformer::Forward(mx::array hidden,
                                    std::vector<AudioCache> &caches) const {
  assert(caches.size() == static_cast<size_t>(layers_));
  const int batch = hidden.shape(0), length = hidden.shape(1),
            head_dim = hidden_size_ / heads_;
  for (int layer = 0; layer < layers_; ++layer) {
    auto &cache = caches[layer];
    const int past = cache.keys ? cache.keys->shape(2) : 0;
    const mx::array query =
        mx::reshape(mx::arange(past, past + length), {-1, 1});
    const mx::array key = mx::reshape(mx::arange(past + length), {1, -1});
    mx::array mask = mx::greater_equal(query, key);
    if (windows_[layer] > 0) {
      mask = mx::logical_and(
          mask, mx::less(mx::subtract(query, key), mx::array(windows_[layer])));
    } else {
    }
    const std::string prefix = prefix_ + ".layers." + std::to_string(layer);
    const mx::array normed =
        checkpoint_.RmsNorm(hidden, prefix + ".input_layernorm", epsilon_);
    const auto project = [&](const std::string &name, int heads) {
      return mx::transpose(
          mx::reshape(checkpoint_.Linear(normed, prefix + ".self_attn." + name),
                      {batch, length, heads, head_dim}),
          {0, 2, 1, 3});
    };
    const mx::array queries =
        mx::fast::rope(project("q_proj", heads_), head_dim, false, rope_theta_,
                       1.0f, cache.offset);
    mx::array keys = mx::fast::rope(project("k_proj", kv_heads_), head_dim,
                                    false, rope_theta_, 1.0f, cache.offset);
    mx::array values = project("v_proj", kv_heads_);
    if (cache.keys) {
      keys = mx::concatenate({*cache.keys, keys}, 2);
      values = mx::concatenate({*cache.values, values}, 2);
    } else {
    }
    const mx::array attended = mx::fast::scaled_dot_product_attention(
        queries, keys, values,
        static_cast<float>(std::pow(static_cast<double>(head_dim), -0.5)), "",
        mask);
    cache.offset += length;
    const int trim =
        windows_[layer] > 0 ? std::max(0, keys.shape(2) - windows_[layer]) : 0;
    cache.keys = mx::slice(keys, {0, 0, trim, 0}, keys.shape());
    cache.values = mx::slice(values, {0, 0, trim, 0}, values.shape());
    hidden = mx::add(
        hidden,
        checkpoint_.Linear(mx::reshape(mx::transpose(attended, {0, 2, 1, 3}),
                                       {batch, length, hidden_size_}),
                           prefix + ".self_attn.o_proj"));
    const mx::array mlp_input = checkpoint_.RmsNorm(
        hidden, prefix + ".post_attention_layernorm", epsilon_);
    hidden = mx::add(
        hidden, checkpoint_.Linear(
                    mx::multiply(
                        omni_common::Silu(checkpoint_.Linear(
                            mlp_input, prefix + ".mlp.gate_proj")),
                        checkpoint_.Linear(mlp_input, prefix + ".mlp.up_proj")),
                    prefix + ".mlp.down_proj"));
  }
  return checkpoint_.RmsNorm(hidden, prefix_ + ".norm", epsilon_);
}

std::vector<AudioCache> AudioTransformer::NewCaches() const {
  return std::vector<AudioCache>(layers_);
}

MingAudioDecoder::MingAudioDecoder(const std::filesystem::path &directory)
    : MingAudioDecoder(omni_common::ReadJson(directory / "config.json"),
                       directory) {}

MingAudioDecoder::MingAudioDecoder(const nlohmann::json &config,
                                   const std::filesystem::path &directory)
    : patch_size_(config.at("audio_tokenizer_config").at("patch_size")),
      hop_length_(config.at("audio_tokenizer_config")
                      .at("dec_kwargs")
                      .at("output_dim")),
      latent_dim_(config.at("audio_tokenizer_config")
                      .at("dec_kwargs")
                      .at("latent_dim")),
      sample_rate_(config.at("audio_tokenizer_config").at("sample_rate")),
      checkpoint_(
          [&] {
            if (sample_rate_ != kAudioSampleRate ||
                !config.at("audio_tokenizer_config")
                     .value("semantic_module_kwargs", nlohmann::json())
                     .is_null()) {
              throw std::invalid_argument(
                  "unsupported Ming AudioVAE decoder configuration");
            } else {
              return LoadComponentWeights(directory, {"audio.decoder."});
            }
          }(),
          config),
      transformer_(
          checkpoint_, "audio.decoder.decoder",
          config.at("audio_tokenizer_config").at("dec_kwargs").at("backbone")) {
}

mx::array MingAudioDecoder::Interpolate(const mx::array &hidden) const {
  const mx::array positions = mx::clip(
      mx::subtract(mx::divide(mx::add(mx::arange(hidden.shape(1) * patch_size_),
                                      mx::array(0.5f)),
                              mx::array(patch_size_)),
                   mx::array(0.5f)),
      mx::array(0), mx::array(hidden.shape(1) - 1));
  const mx::array left = mx::astype(mx::floor(positions), mx::int32);
  const mx::array right =
      mx::minimum(mx::add(left, mx::array(1)), mx::array(hidden.shape(1) - 1));
  const mx::array weight = mx::reshape(
      mx::astype(mx::subtract(positions, left), hidden.dtype()), {1, -1, 1});
  return mx::add(
      mx::multiply(mx::take(hidden, left, 1),
                   mx::subtract(mx::array(1, hidden.dtype()), weight)),
      mx::multiply(mx::take(hidden, right, 1), weight));
}

mx::array MingAudioDecoder::Synthesize(const mx::array &hidden,
                                       AudioDecoderState &state, bool streaming,
                                       bool last_chunk) const {
  const auto head =
      mx::split(mx::astype(checkpoint_.Linear(hidden, "audio.decoder.head.out"),
                           mx::float32),
                2, -1);
  const mx::array magnitude = mx::minimum(mx::exp(head[0]), mx::array(100.0f));
  const mx::array spectrum = mx::multiply(
      magnitude, mx::add(mx::cos(head[1]),
                         mx::multiply(mx::array(std::complex<float>(0, 1)),
                                      mx::sin(head[1]))));
  const int batch = hidden.shape(0), frames = hidden.shape(1),
            fft_size = 4 * hop_length_;
  const int size = (frames - 1) * hop_length_ + fft_size;
  const int buffer_length = fft_size - hop_length_, pad = buffer_length / 2;
  const mx::array window = mx::astype(
      checkpoint_.Weight("audio.decoder.head.istft.window"), mx::float32);
  const mx::array inverse =
      mx::multiply(mx::fft::irfft(spectrum, fft_size, -1), window);
  const mx::array indices =
      mx::reshape(mx::add(mx::multiply(mx::reshape(mx::arange(frames), {-1, 1}),
                                       mx::array(hop_length_)),
                          mx::arange(fft_size)),
                  {-1});
  const mx::array batch_indices =
      mx::reshape(mx::add(mx::reshape(indices, {1, -1}),
                          mx::multiply(mx::reshape(mx::arange(batch), {-1, 1}),
                                       mx::array(size))),
                  {-1});
  mx::array audio = mx::reshape(
      mx::scatter_add(mx::zeros({batch * size}, mx::float32), batch_indices,
                      mx::reshape(inverse, {-1, 1}), 0),
      {batch, size});
  mx::array envelope = mx::reshape(
      mx::scatter_add(mx::zeros({size}, mx::float32), indices,
                      mx::reshape(mx::broadcast_to(mx::multiply(window, window),
                                                   {frames, fft_size}),
                                  {-1, 1}),
                      0),
      {1, size});
  if (!streaming) {
    return mx::divide(mx::slice(audio, {0, pad}, {batch, size - pad}),
                      mx::slice(envelope, {0, pad}, {1, size - pad}));
  } else if (!state.audio_overlap) {
    audio = mx::slice(audio, {0, pad}, audio.shape());
    envelope = mx::slice(envelope, {0, pad}, envelope.shape());
  } else {
    audio = mx::slice_update(
        audio,
        mx::add(mx::slice(audio, {0, 0}, {batch, buffer_length}),
                *state.audio_overlap),
        {0, 0}, {batch, buffer_length});
    envelope = mx::slice_update(
        envelope,
        mx::add(mx::slice(envelope, {0, 0}, {1, buffer_length}),
                *state.envelope_overlap),
        {0, 0}, {1, buffer_length});
  }
  state.audio_overlap =
      mx::slice(audio, {0, audio.shape(1) - buffer_length}, audio.shape());
  state.envelope_overlap = mx::slice(
      envelope, {0, envelope.shape(1) - buffer_length}, envelope.shape());
  const int end = audio.shape(1) - (last_chunk ? pad : buffer_length);
  return mx::divide(mx::slice(audio, {0, 0}, {batch, end}),
                    mx::slice(envelope, {0, 0}, {1, end}));
}

mx::array MingAudioDecoder::DecodeFull(const mx::array &latents) const {
  assert(latents.ndim() == 3 && latents.shape(2) == latent_dim_ &&
         latents.size() > 0);
  mx::array hidden = checkpoint_.Linear(
      mx::astype(latents, WeightDtype(checkpoint_, "audio.decoder.fc1")),
      "audio.decoder.fc1");
  if (patch_size_ != -1) {
    hidden = Interpolate(hidden);
  } else {
  }
  AudioDecoderState state;
  state.caches = transformer_.NewCaches();
  return Synthesize(transformer_.Forward(hidden, state.caches), state, false,
                    true);
}

mx::array MingAudioDecoder::DecodeChunk(const mx::array &latents,
                                        AudioDecoderState &state,
                                        bool last_chunk) const {
  assert(latents.ndim() == 3 && latents.shape(2) == latent_dim_ &&
         latents.size() > 0);
  if (state.caches.empty()) {
    state.caches = transformer_.NewCaches();
  } else {
  }
  mx::array hidden = checkpoint_.Linear(
      mx::astype(latents, WeightDtype(checkpoint_, "audio.decoder.fc1")),
      "audio.decoder.fc1");
  if (patch_size_ != -1) {
    // Note (XXXXRT666): upsampling a frame reads only its neighbours, so a
    // chunk holds back just its last frame until the next one comes.
    const mx::array frames =
        state.pending ? mx::concatenate({*state.pending, hidden}, 1) : hidden;
    const int ready = frames.shape(1) - (last_chunk ? 0 : 1);
    if (ready == 0) {
      state.pending = frames;
      mx::eval(*state.pending);
      return mx::zeros({latents.shape(0), 0}, mx::float32);
    } else {
    }
    const int start = state.left ? patch_size_ : 0;
    const mx::array upsampled = Interpolate(
        state.left ? mx::concatenate({*state.left, frames}, 1) : frames);
    hidden = mx::slice(
        upsampled, {0, start, 0},
        {upsampled.shape(0), start + ready * patch_size_, upsampled.shape(2)});
    if (last_chunk) {
      state.pending.reset();
      state.left.reset();
    } else {
      state.left = mx::slice(frames, {0, ready - 1, 0},
                             {frames.shape(0), ready, frames.shape(2)});
      state.pending = mx::slice(frames, {0, ready, 0}, frames.shape());
    }
  } else {
  }
  const mx::array waveform = Synthesize(
      transformer_.Forward(hidden, state.caches), state, true, last_chunk);
  std::vector<mx::array> arrays{waveform};
  for (const auto &cache : state.caches) {
    arrays.push_back(*cache.keys);
    arrays.push_back(*cache.values);
  }
  for (const auto *array : {&state.pending, &state.left, &state.audio_overlap,
                            &state.envelope_overlap}) {
    if (*array) {
      arrays.push_back(**array);
    } else {
    }
  }
  mx::eval(arrays);
  if (last_chunk) {
    state = AudioDecoderState{};
  } else {
  }
  return waveform;
}

MingAudioEncoder::MingAudioEncoder(const std::filesystem::path &directory)
    : MingAudioEncoder(omni_common::ReadJson(directory / "config.json"),
                       directory) {}

MingAudioEncoder::MingAudioEncoder(const nlohmann::json &config,
                                   const std::filesystem::path &directory)
    : patch_size_(config.at("audio_tokenizer_config").at("patch_size")),
      hop_length_(config.at("audio_tokenizer_config")
                      .at("enc_kwargs")
                      .value("hop_size", 320)),
      input_dim_(
          config.at("audio_tokenizer_config").at("enc_kwargs").at("input_dim")),
      latent_dim_(config.at("audio_tokenizer_config")
                      .at("enc_kwargs")
                      .at("latent_dim")),
      checkpoint_(
          [&] {
            const auto &audio = config.at("audio_tokenizer_config");
            if (audio.at("sample_rate") != kAudioSampleRate ||
                !audio.value("semantic_module_kwargs", nlohmann::json())
                     .is_null()) {
              throw std::invalid_argument(
                  "unsupported Ming AudioVAE encoder configuration");
            } else {
              return LoadComponentWeights(directory, {"audio.encoder."});
            }
          }(),
          config),
      transformer_(
          checkpoint_, "audio.encoder.encoder",
          config.at("audio_tokenizer_config").at("enc_kwargs").at("backbone")) {
  if (patch_size_ != -1) {
    auto backbone =
        config.at("audio_tokenizer_config").at("enc_kwargs").at("backbone");
    backbone["num_hidden_layers"] = 4;
    if (!backbone.value("layer_types", nlohmann::json()).is_null()) {
      auto kinds = backbone.at("layer_types").get<std::vector<std::string>>();
      kinds.resize(std::min<size_t>(4, kinds.size()));
      backbone["layer_types"] = kinds;
    } else {
    }
    aggregator_.emplace(checkpoint_, "audio.encoder.aggregator", backbone);
  } else {
  }
}

mx::array MingAudioEncoder::Encode(const mx::array &waveform) const {
  assert(waveform.ndim() == 2 && waveform.size() > 0);
  const int batch = waveform.shape(0), length = waveform.shape(1);
  const int count = (length + hop_length_ - 1) / hop_length_;
  const int needed = (count - 1) * hop_length_ + input_dim_;
  const mx::array padded =
      mx::pad(waveform, {{0, 0}, {0, std::max(0, needed - length)}});
  const mx::array indices =
      mx::add(mx::multiply(mx::reshape(mx::arange(count), {-1, 1}),
                           mx::array(hop_length_)),
              mx::arange(input_dim_));
  const mx::array frames = mx::take(padded, indices, 1);
  auto caches = transformer_.NewCaches();
  mx::array hidden = transformer_.Forward(
      checkpoint_.Linear(
          checkpoint_.Linear(
              mx::astype(frames, WeightDtype(checkpoint_, "audio.encoder.fc1")),
              "audio.encoder.fc1"),
          "audio.encoder.fc2"),
      caches);
  if (aggregator_) {
    const int padding =
        (patch_size_ - hidden.shape(1) % patch_size_) % patch_size_;
    hidden = mx::reshape(mx::pad(hidden, {{0, 0}, {0, padding}, {0, 0}}),
                         {-1, patch_size_, transformer_.hidden_size()});
    const mx::array cls = mx::astype(
        mx::broadcast_to(checkpoint_.Weight("audio.encoder.cls_embed"),
                         {hidden.shape(0), 1, transformer_.hidden_size()}),
        hidden.dtype());
    hidden = mx::reshape(mx::concatenate({hidden, cls}, 1),
                         {batch, -1, transformer_.hidden_size()});
    auto aggregate_caches = aggregator_->NewCaches();
    hidden =
        mx::reshape(aggregator_->Forward(hidden, aggregate_caches),
                    {batch, -1, patch_size_ + 1, transformer_.hidden_size()});
    hidden = mx::squeeze(
        mx::slice(hidden, {0, 0, patch_size_, 0}, hidden.shape()), 2);
  } else {
  }
  const auto posterior =
      mx::split(checkpoint_.Linear(hidden, "audio.encoder.fc3"), 2, -1);
  const mx::array random =
      mx::random::normal(posterior[0].shape(), posterior[0].dtype());
  const mx::array deviation =
      mx::add(mx::logaddexp(posterior[1], mx::array(0, posterior[1].dtype())),
              mx::array(1e-4f, posterior[1].dtype()));
  return mx::add(posterior[0], mx::multiply(deviation, random));
}

} // namespace ming_tts
