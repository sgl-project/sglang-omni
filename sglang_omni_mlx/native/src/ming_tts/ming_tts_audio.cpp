// SPDX-License-Identifier: Apache-2.0
#include "ming_tts/ming_tts_audio.h"

#include <algorithm>
#include <cassert>
#include <complex>
#include <stdexcept>

#include "ming_tts/ming_tts_model.h"

namespace ming_tts {
namespace mx = mlx::core;

MingAudioDecoder::MingAudioDecoder(const std::filesystem::path &directory)
    : MingAudioDecoder(qwen3_asr::ReadJson(directory / "config.json"),
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
            if (sample_rate_ != 44100 ||
                std::min(hop_length_, latent_dim_) <= 0 ||
                config.at("audio_tokenizer_config")
                        .at("enc_kwargs")
                        .at("latent_dim")
                        .get<int>() != latent_dim_ ||
                (patch_size_ != -1 && patch_size_ < 1) ||
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
  RequireShape(checkpoint_, "audio.decoder.fc1.weight",
               {transformer_.hidden_size(), latent_dim_});
  RequireShape(checkpoint_, "audio.decoder.fc1.bias",
               {transformer_.hidden_size()});
  RequireShape(checkpoint_, "audio.decoder.head.out.weight",
               {hop_length_ * 4 + 2, transformer_.hidden_size()});
  RequireShape(checkpoint_, "audio.decoder.head.out.bias",
               {hop_length_ * 4 + 2});
  RequireShape(checkpoint_, "audio.decoder.head.istft.window",
               {hop_length_ * 4});
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
  if (patch_size_ == -1) {
  } else if (!state.pending) {
    if (!last_chunk) {
      state.pending = hidden;
      mx::eval(*state.pending);
      return mx::zeros({latents.shape(0), 0}, mx::float32);
    } else {
      hidden = Interpolate(hidden);
    }
  } else {
    std::vector<mx::array> parts{
        *state.pending,
        mx::slice(hidden, {0, 0, 0}, {hidden.shape(0), 1, hidden.shape(2)})};
    int start = 0;
    if (state.left) {
      parts.insert(parts.begin(), *state.left);
      start = patch_size_;
    } else {
    }
    mx::array previous = Interpolate(mx::concatenate(parts, 1));
    previous = mx::slice(previous, {0, start, 0},
                         {hidden.shape(0),
                          start + state.pending->shape(1) * patch_size_,
                          hidden.shape(2)});
    const mx::array left =
        mx::slice(*state.pending, {0, state.pending->shape(1) - 1, 0},
                  state.pending->shape());
    if (last_chunk) {
      const mx::array tail = Interpolate(mx::concatenate({left, hidden}, 1));
      hidden = mx::concatenate(
          {previous, mx::slice(tail, {0, patch_size_, 0}, tail.shape())}, 1);
      state.pending.reset();
      state.left.reset();
    } else {
      state.pending = hidden;
      state.left = left;
      hidden = previous;
    }
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
    : MingAudioEncoder(qwen3_asr::ReadJson(directory / "config.json"),
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
            if (audio.at("sample_rate") != 44100 ||
                std::min({hop_length_, input_dim_, latent_dim_}) <= 0 ||
                (patch_size_ != -1 && patch_size_ < 1) ||
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
  const int hidden = transformer_.hidden_size();
  RequireShape(checkpoint_, "audio.encoder.fc1.weight", {hidden, input_dim_});
  RequireShape(checkpoint_, "audio.encoder.fc2.weight", {hidden, hidden});
  RequireShape(checkpoint_, "audio.encoder.fc2.bias", {hidden});
  RequireShape(checkpoint_, "audio.encoder.fc3.weight",
               {2 * latent_dim_, hidden});
  RequireShape(checkpoint_, "audio.encoder.fc3.bias", {2 * latent_dim_});
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
    RequireShape(checkpoint_, "audio.encoder.cls_embed", {1, 1, hidden});
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
