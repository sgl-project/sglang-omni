// SPDX-License-Identifier: Apache-2.0
#include "cosyvoice3/cosyvoice3_vocoder.h"

#include <cmath>
#include <stdexcept>

namespace cosyvoice3 {

namespace mx = mlx::core;

namespace {

constexpr int kDitBlockCount = 22;
constexpr int kDitHeadCount = 16;
constexpr int kDitHeadSize = 64;
constexpr int kTimeFrequencyCount = 128;
constexpr int kConvPositionKernel = 31;
constexpr int kConvPositionGroups = 16;
constexpr int kPreLookaheadFrames = 3;
constexpr int kFlowStepCount = 10;
constexpr float kGuidanceRate = 0.7f;
constexpr float kNormEps = 1e-6f;
constexpr int kHarmonicCount = 9;
// Samples per mel frame: the upsampling rates 8, 5, 3 times the iSTFT hop.
constexpr int kSamplesPerFrame = 480;
constexpr int kFftSize = 16;
constexpr int kFftHop = 4;
constexpr int kFftBins = kFftSize / 2 + 1;
constexpr int kUpsampleRates[] = {8, 5, 3};
constexpr int kUpsampleKernels[] = {16, 11, 7};
constexpr int kResBlockKernels[] = {3, 7, 11};
constexpr int kSourceResBlockKernels[] = {7, 7, 11};
// Each source branch's downsampling stride, to the rate of its stage.
constexpr int kSourceStrides[] = {15, 3, 1};
constexpr int kResBlockDilations[] = {1, 3, 5};
constexpr float kSineAmplitude = 0.1f;
constexpr float kNoiseStd = 0.003f;
constexpr float kVoicedThreshold = 10.0f;
constexpr float kAudioLimit = 0.99f;
constexpr float kMaxMagnitude = 100.0f;
const std::string kEstimator = "flow.decoder.estimator.";

// A Python float operand of an MLX op: it takes the other operand's dtype.
mx::array Scalar(double value, mx::Dtype dtype) {
  return mx::array(static_cast<float>(value), dtype);
}

mx::array PlainLayerNorm(const mx::array &x) {
  return mx::fast::layer_norm(x, std::nullopt, std::nullopt, kNormEps);
}

mx::array LeakyRelu(const mx::array &x, double slope) {
  return mx::maximum(mx::multiply(Scalar(slope, x.dtype()), x), x);
}

mx::array Mish(const mx::array &x) {
  return mx::multiply(x,
                      mx::tanh(mx::logaddexp(x, mx::array(0.0f, x.dtype()))));
}

mx::array TanhGelu(const mx::array &x) {
  const mx::Dtype dtype = x.dtype();
  const mx::array cubic = mx::power(x, mx::array(3.0f, dtype));
  const mx::array inner =
      mx::multiply(Scalar(std::sqrt(2.0 / M_PI), dtype),
                   mx::add(x, mx::multiply(Scalar(0.044715, dtype), cubic)));
  return mx::multiply(mx::multiply(Scalar(0.5, dtype), x),
                      mx::add(Scalar(1.0, dtype), mx::tanh(inner)));
}

// Interleaved rotation of the first 64 channels, as the DiT applies rotary
// embeddings before it splits heads.
mx::array Rotate(const mx::array &x, const mx::array &cos,
                 const mx::array &sin) {
  const int batch = x.shape(0);
  const int length = x.shape(1);
  const int width = x.shape(2);
  const mx::array rotated = mx::slice(x, {0, 0, 0}, {batch, length, 64});
  const mx::array passed = mx::slice(x, {0, 0, 64}, {batch, length, width});
  const mx::array pairs = mx::reshape(rotated, {batch, length, 32, 2});
  const mx::array even =
      mx::squeeze(mx::slice(pairs, {0, 0, 0, 0}, {batch, length, 32, 1}), 3);
  const mx::array odd =
      mx::squeeze(mx::slice(pairs, {0, 0, 0, 1}, {batch, length, 32, 2}), 3);
  const mx::array half_turn = mx::reshape(
      mx::stack({mx::negative(odd), even}, -1), {batch, length, 64});
  const mx::array result = mx::astype(
      mx::add(mx::multiply(rotated, cos), mx::multiply(half_turn, sin)),
      x.dtype());
  return mx::concatenate({result, passed}, -1);
}

// The periodic Hann window torch.stft uses.
mx::array HannWindow() {
  std::vector<float> window(kFftSize);
  for (int i = 0; i < kFftSize; ++i) {
    window[i] =
        static_cast<float>(0.5 * (1 - std::cos(2 * M_PI * i / kFftSize)));
  }
  return mx::array(window.data(), {kFftSize}, mx::float32);
}

// Overlap-adds frames [count, 16] at hop 4 into [(count + 3) * 4] samples.
mx::array OverlapAdd(const mx::array &frames) {
  const int count = frames.shape(0);
  std::optional<mx::array> total;
  for (int block = 0; block < kFftSize / kFftHop; ++block) {
    const mx::array shifted = mx::pad(
        mx::slice(frames, {0, block * kFftHop}, {count, (block + 1) * kFftHop}),
        {{block, kFftSize / kFftHop - 1 - block}, {0, 0}});
    total = total.has_value() ? mx::add(*total, shifted) : shifted;
  }
  return mx::reshape(*total, {-1});
}

} // namespace

Flow::Flow(const omni_common::Checkpoint &checkpoint)
    : checkpoint_(checkpoint),
      dtype_probe_(checkpoint.Weight("flow.input_embedding.weight")),
      // Note (yexiaodong): one fixed noise draw, so a request's mel depends
      // only on its tokens.
      noise_(mx::random::normal({1, kMelBinCount, kMaxMelFrames}, mx::float32,
                                0.0f, 1.0f, mx::random::key(0))),
      rope_cos_(mx::array(0.0f)), rope_sin_(mx::array(0.0f)) {
  const mx::array inverse_frequencies = mx::divide(
      mx::array(1.0f),
      mx::power(
          mx::array(10000.0f),
          mx::divide(mx::astype(mx::arange(0, kDitHeadSize, 2), mx::float32),
                     mx::array(static_cast<float>(kDitHeadSize)))));
  const mx::array angles =
      mx::repeat(mx::outer(mx::astype(mx::arange(kMaxMelFrames), mx::float32),
                           inverse_frequencies),
                 2, -1);
  rope_cos_ = mx::cos(angles);
  rope_sin_ = mx::sin(angles);
  mx::eval({noise_, rope_cos_, rope_sin_});
}

mx::array Flow::Decode(const std::vector<int> &speech_tokens,
                       const std::vector<int> &prompt_tokens,
                       const mx::array &prompt_mel,
                       const mx::array &speaker_embedding,
                       const std::function<void()> &on_step) const {
  const mx::Dtype dtype = dtype_probe_.dtype();
  const int token_count =
      static_cast<int>(prompt_tokens.size() + speech_tokens.size());
  const int prompt_frames = prompt_mel.shape(1);
  const int frame_count = kMelFramesPerToken * token_count;
  if (speech_tokens.empty() ||
      prompt_frames !=
          kMelFramesPerToken * static_cast<int>(prompt_tokens.size()) ||
      frame_count > kMaxMelFrames) {
    throw std::invalid_argument(
        "CosyVoice3 flow needs speech tokens, two prompt mel frames per "
        "prompt token, and at most 15000 mel frames");
  } else {
  }
  // Note (Jiaxin Deng): F.normalize's clamp in float32; an all-zero
  // embedding (no speaker) stays zero rather than dividing by zero.
  const mx::array embedding = mx::astype(speaker_embedding, mx::float32);
  const mx::array speaker_features = checkpoint_.Linear(
      mx::astype(
          mx::divide(embedding,
                     mx::maximum(
                         mx::linalg::norm(embedding, std::vector<int>{1}, true),
                         mx::array(1e-12f))),
          dtype),
      "flow.spk_embed_affine_layer");

  std::vector<int> token_ids = prompt_tokens;
  token_ids.insert(token_ids.end(), speech_tokens.begin(), speech_tokens.end());
  const mx::array ids =
      mx::clip(mx::array(token_ids.data(), {1, token_count}, mx::int32),
               mx::array(0), mx::array(dtype_probe_.shape(0) - 1));
  const mx::array embedded = checkpoint_.Embed(ids, "flow.input_embedding");
  mx::array lookahead = mx::add(
      mx::conv1d(mx::pad(embedded, {{0, 0}, {0, kPreLookaheadFrames}, {0, 0}}),
                 checkpoint_.Weight("flow.pre_lookahead_layer.conv1.weight")),
      checkpoint_.Weight("flow.pre_lookahead_layer.conv1.bias"));
  lookahead = mx::add(
      mx::conv1d(mx::pad(LeakyRelu(lookahead, 0.01), {{0, 0}, {2, 0}, {0, 0}}),
                 checkpoint_.Weight("flow.pre_lookahead_layer.conv2.weight")),
      checkpoint_.Weight("flow.pre_lookahead_layer.conv2.bias"));
  const mx::array mu =
      mx::repeat(mx::add(lookahead, embedded), kMelFramesPerToken, 1);
  const mx::array mel_conditioning = mx::concatenate(
      {mx::astype(prompt_mel, dtype),
       mx::zeros({1, frame_count - prompt_frames, kMelBinCount}, dtype)},
      1);

  // Note (Jiaxin Deng): classifier-free guidance runs the conditional and
  // unconditional halves as one batch of two.
  const mx::array mu_pair = mx::concatenate({mu, mx::zeros_like(mu)}, 0);
  const mx::array speaker_pair =
      mx::concatenate({speaker_features, mx::zeros_like(speaker_features)}, 0);
  const mx::array conditioning_pair =
      mx::concatenate({mel_conditioning, mx::zeros_like(mel_conditioning)}, 0);
  mx::array t_span = mx::linspace(0, 1, kFlowStepCount + 1, dtype);
  t_span = mx::subtract(
      Scalar(1.0, dtype),
      mx::cos(mx::multiply(mx::multiply(t_span, Scalar(0.5, dtype)),
                           Scalar(M_PI, dtype))));
  const auto span = [&](int step) {
    return mx::slice(t_span, {step}, {step + 1});
  };
  mx::array x = mx::transpose(
      mx::astype(mx::slice(noise_, {0, 0, 0}, {1, kMelBinCount, frame_count}),
                 dtype),
      {0, 2, 1});
  mx::array t = span(0);
  mx::array dt = mx::squeeze(mx::subtract(span(1), span(0)));
  for (int step = 1; step <= kFlowStepCount; ++step) {
    on_step();
    const mx::array velocity =
        Velocity(mx::concatenate({x, x}, 0), mu_pair,
                 mx::concatenate({t, t}, 0), speaker_pair, conditioning_pair);
    const std::vector<mx::array> halves = mx::split(velocity, 2, 0);
    const mx::array guided = mx::subtract(
        mx::multiply(Scalar(1.0 + kGuidanceRate, dtype), halves[0]),
        mx::multiply(Scalar(kGuidanceRate, dtype), halves[1]));
    x = mx::add(x, mx::multiply(dt, guided));
    // Note (yexiaodong): one evaluation per step bounds the graph and the
    // memory it holds.
    mx::eval(x);
    t = mx::add(t, dt);
    if (step < kFlowStepCount) {
      dt = mx::squeeze(mx::subtract(span(step + 1), t));
    } else {
    }
  }
  return mx::transpose(
      mx::slice(x, {0, prompt_frames, 0}, {1, frame_count, kMelBinCount}),
      {0, 2, 1});
}

mx::array Flow::Velocity(const mx::array &x, const mx::array &mu,
                         const mx::array &timesteps,
                         const mx::array &speaker_features,
                         const mx::array &mel_conditioning) const {
  const mx::Dtype dtype = x.dtype();
  const int batch = x.shape(0);
  const int length = x.shape(1);
  const mx::array frequencies = mx::exp(
      mx::multiply(mx::astype(mx::arange(kTimeFrequencyCount), mx::float32),
                   mx::array(static_cast<float>(-std::log(10000.0) /
                                                (kTimeFrequencyCount - 1)))));
  const mx::array time_angles = mx::multiply(
      mx::expand_dims(mx::multiply(Scalar(1000.0, dtype), timesteps), 1),
      mx::expand_dims(frequencies, 0));
  // Note (yuhui): the float32 sinusoid is cast before the time MLP so
  // it does not promote the half-precision DiT.
  mx::array timestep_embedding = mx::astype(
      mx::concatenate({mx::sin(time_angles), mx::cos(time_angles)}, -1), dtype);
  timestep_embedding = checkpoint_.Linear(
      omni_common::Silu(checkpoint_.Linear(
          timestep_embedding, kEstimator + "time_embed.time_mlp_0")),
      kEstimator + "time_embed.time_mlp_2");

  const mx::array speakers =
      mx::broadcast_to(mx::expand_dims(speaker_features, 1),
                       {batch, length, speaker_features.shape(-1)});
  mx::array hidden = checkpoint_.Linear(
      mx::concatenate({x, mel_conditioning, mu, speakers}, -1),
      kEstimator + "input_embed.proj");
  mx::array position = hidden;
  for (const char *conv : {"conv1", "conv2"}) {
    const std::string prefix =
        kEstimator + "input_embed.conv_pos_embed." + conv;
    position = Mish(mx::add(
        mx::conv1d(
            mx::pad(position, {{0, 0}, {kConvPositionKernel - 1, 0}, {0, 0}}),
            checkpoint_.Weight(prefix + ".weight"), 1, 0, 1,
            kConvPositionGroups),
        checkpoint_.Weight(prefix + ".bias")));
  }
  hidden = mx::add(position, hidden);

  for (int block = 0; block < kDitBlockCount; ++block) {
    hidden = Block(hidden, timestep_embedding, block);
  }
  const std::vector<mx::array> modulation =
      mx::split(checkpoint_.Linear(omni_common::Silu(timestep_embedding),
                                   kEstimator + "norm_out.linear"),
                2, -1);
  hidden =
      mx::add(mx::multiply(PlainLayerNorm(hidden),
                           mx::expand_dims(
                               mx::add(Scalar(1.0, dtype), modulation[0]), 1)),
              mx::expand_dims(modulation[1], 1));
  return checkpoint_.Linear(hidden, kEstimator + "proj_out");
}

mx::array Flow::Block(const mx::array &x, const mx::array &timestep_embedding,
                      int block) const {
  const std::string prefix =
      kEstimator + "transformer_blocks." + std::to_string(block) + ".";
  const mx::Dtype dtype = x.dtype();
  const int batch = x.shape(0);
  const int length = x.shape(1);
  const int width = x.shape(2);
  // Shift, scale and gate for attention, then for the feed-forward.
  const std::vector<mx::array> modulation =
      mx::split(checkpoint_.Linear(omni_common::Silu(timestep_embedding),
                                   prefix + "attn_norm.linear"),
                6, -1);
  const auto modulate = [&](const mx::array &input, int shift, int scale) {
    return mx::add(
        mx::multiply(
            PlainLayerNorm(input),
            mx::expand_dims(mx::add(Scalar(1.0, dtype), modulation[scale]), 1)),
        mx::expand_dims(modulation[shift], 1));
  };
  const mx::array normed = modulate(x, 0, 1);
  const mx::array cos =
      mx::expand_dims(mx::slice(rope_cos_, {0, 0}, {length, 64}), 0);
  const mx::array sin =
      mx::expand_dims(mx::slice(rope_sin_, {0, 0}, {length, 64}), 0);
  const auto heads = [&](const mx::array &projection) {
    return mx::transpose(
        mx::reshape(projection, {batch, length, kDitHeadCount, kDitHeadSize}),
        {0, 2, 1, 3});
  };
  const mx::array q =
      heads(Rotate(checkpoint_.Linear(normed, prefix + "attn.to_q"), cos, sin));
  const mx::array k =
      heads(Rotate(checkpoint_.Linear(normed, prefix + "attn.to_k"), cos, sin));
  const mx::array v = heads(checkpoint_.Linear(normed, prefix + "attn.to_v"));
  const mx::array attended = mx::reshape(
      mx::transpose(mx::fast::scaled_dot_product_attention(
                        q, k, v, 1.0f / std::sqrt(float(kDitHeadSize))),
                    {0, 2, 1, 3}),
      {batch, length, width});
  mx::array hidden = mx::add(
      x, mx::multiply(mx::expand_dims(modulation[2], 1),
                      checkpoint_.Linear(attended, prefix + "attn.to_out_0")));
  const mx::array feed_forward =
      checkpoint_.Linear(TanhGelu(checkpoint_.Linear(modulate(hidden, 3, 4),
                                                     prefix + "ff.ff_0_0")),
                         prefix + "ff.ff_2");
  return mx::add(hidden,
                 mx::multiply(mx::expand_dims(modulation[5], 1), feed_forward));
}

Hift::Hift(const omni_common::Checkpoint &checkpoint)
    : checkpoint_(checkpoint), window_(HannWindow()),
      initial_phases_(mx::array(0.0f)) {
  // Note (yexiaodong): the first harmonic starts at phase zero and the
  // others at fixed random phases, so the source is the same every request.
  const mx::array phases =
      mx::random::uniform({1, kHarmonicCount}, mx::float32, mx::random::key(1));
  initial_phases_ = mx::concatenate(
      {mx::zeros({1, 1}), mx::slice(phases, {0, 1}, {1, kHarmonicCount})}, 1);
  mx::eval({window_, initial_phases_});
}

mx::array Hift::Conv(const mx::array &x, const std::string &prefix, int left,
                     int right, int stride, int dilation) const {
  return mx::add(mx::conv1d(mx::pad(x, {{0, 0}, {left, right}, {0, 0}}),
                            checkpoint_.Weight(prefix + ".conv.weight"), stride,
                            0, dilation),
                 checkpoint_.Weight(prefix + ".conv.bias"));
}

mx::array Hift::ResBlock(const mx::array &x, const std::string &prefix,
                         int kernel_size) const {
  const auto snake = [&](const mx::array &input, const std::string &name) {
    const mx::array &alpha = checkpoint_.Weight(prefix + "." + name + ".alpha");
    const mx::Dtype dtype = alpha.dtype();
    const mx::array magnitude = mx::abs(alpha);
    const mx::array divisor =
        mx::where(mx::less(magnitude, Scalar(1e-9, dtype)), Scalar(1e-4, dtype),
                  mx::multiply(mx::sign(alpha),
                               mx::maximum(magnitude, Scalar(1e-4, dtype))));
    return mx::add(input,
                   mx::divide(mx::power(mx::sin(mx::multiply(input, alpha)),
                                        mx::array(2.0f, input.dtype())),
                              divisor));
  };
  mx::array hidden = x;
  for (int i = 0; i < 3; ++i) {
    const std::string index = std::to_string(i);
    const int dilation = kResBlockDilations[i];
    mx::array branch = Conv(snake(hidden, "activations1." + index),
                            prefix + ".convs1." + index,
                            dilation * (kernel_size - 1), 0, 1, dilation);
    branch = Conv(snake(branch, "activations2." + index),
                  prefix + ".convs2." + index, kernel_size - 1, 0);
    hidden = mx::add(branch, hidden);
  }
  return hidden;
}

mx::array Hift::HarmonicSource(const mx::array &f0) const {
  const int sample_count = f0.shape(1);
  const int frame_count = sample_count / kSamplesPerFrame;
  const mx::array harmonics = mx::reshape(
      mx::arange(1, kHarmonicCount + 1, mx::int32), {1, 1, kHarmonicCount});
  mx::array radians =
      mx::remainder(mx::divide(mx::multiply(f0, harmonics),
                               mx::array(static_cast<float>(kSampleRate))),
                    mx::array(1.0f));
  radians = mx::concatenate(
      {mx::add(mx::slice(radians, {0, 0, 0}, {1, 1, kHarmonicCount}),
               mx::expand_dims(initial_phases_, 1)),
       mx::slice(radians, {0, 1, 0}, {1, sample_count, kHarmonicCount})},
      1);
  // Linear interpolation without corner alignment down to one value a frame.
  const mx::array source = mx::clip(
      mx::subtract(
          mx::multiply(
              mx::add(mx::astype(mx::arange(frame_count), mx::float32),
                      mx::array(0.5f)),
              mx::array(static_cast<float>(sample_count) / frame_count)),
          mx::array(0.5f)),
      mx::array(0.0f), mx::array(static_cast<float>(sample_count - 1)));
  const mx::array low = mx::astype(mx::floor(source), mx::int32);
  const mx::array high =
      mx::minimum(mx::add(low, mx::array(1)), mx::array(sample_count - 1));
  const mx::array weight =
      mx::expand_dims(mx::subtract(source, mx::astype(low, mx::float32)), 1);
  const mx::array low_values = mx::take(radians, low, 1);
  const mx::array frame_radians = mx::add(
      low_values, mx::multiply(weight, mx::subtract(mx::take(radians, high, 1),
                                                    low_values)));
  const mx::array phase = mx::repeat(
      mx::multiply(mx::multiply(mx::multiply(mx::cumsum(frame_radians, 1),
                                             mx::array(2.0f)),
                                mx::array(static_cast<float>(M_PI))),
                   mx::array(static_cast<float>(kSamplesPerFrame))),
      kSamplesPerFrame, 1);
  const mx::array sines =
      mx::multiply(mx::sin(phase), mx::array(kSineAmplitude));
  const mx::array voiced =
      mx::astype(mx::greater(f0, mx::array(kVoicedThreshold)), mx::float32);
  const mx::array noise_amplitude =
      mx::add(mx::multiply(voiced, mx::array(kNoiseStd)),
              mx::divide(mx::multiply(mx::subtract(mx::array(1.0f), voiced),
                                      mx::array(kSineAmplitude)),
                         mx::array(3.0f)));
  const mx::array noise = mx::multiply(
      noise_amplitude,
      mx::random::uniform(sines.shape(), mx::float32, mx::random::key(0)));
  return mx::tanh(
      checkpoint_.Linear(mx::add(mx::multiply(sines, voiced), noise),
                         "hifigan.m_source.l_linear"));
}

mx::array Hift::Decode(const mx::array &mel) const {
  const mx::Dtype dtype =
      checkpoint_.Weight("hifigan.conv_pre.conv.weight").dtype();
  const int frame_count = mel.shape(2);
  const mx::array frames = mx::transpose(mel, {0, 2, 1});

  // Note (yuhui): pitch runs in float32, since its error accumulates in
  // the harmonic source's phase.
  mx::array pitch = mx::astype(frames, mx::float32);
  for (int i = 0; i < 5; ++i) {
    pitch = Conv(pitch, "hifigan.f0_predictor.condnet_" + std::to_string(2 * i),
                 i == 0 ? 0 : 2, i == 0 ? 3 : 0);
    pitch = mx::where(mx::greater(pitch, mx::array(0.0f)), pitch,
                      mx::subtract(mx::exp(pitch), mx::array(1.0f)));
  }
  const mx::array f0 =
      mx::abs(checkpoint_.Linear(pitch, "hifigan.f0_predictor.classifier"));
  const mx::array source =
      mx::reshape(HarmonicSource(mx::repeat(f0, kSamplesPerFrame, 1)), {-1});

  const int sample_count = source.shape(0);
  const int half = kFftSize / 2;
  const mx::array padded = mx::concatenate(
      {mx::slice(source, {half}, {0}, {-1}), source,
       mx::slice(source, {sample_count - 2}, {sample_count - half - 2}, {-1})});
  const int fft_frames = sample_count / kFftHop + 1;
  const mx::array starts =
      mx::multiply(mx::arange(fft_frames, mx::int32), mx::array(kFftHop));
  const mx::array spectrum = mx::fft::fft(
      mx::multiply(mx::take(padded, mx::add(mx::expand_dims(starts, 1),
                                            mx::arange(kFftSize, mx::int32))),
                   window_),
      -1);
  const mx::array bins = mx::slice(spectrum, {0, 0}, {fft_frames, kFftBins});
  // Note (yuhui): the float32 source spectrum is cast so it does not
  // promote the half-precision convolutions.
  const mx::array source_spectrum = mx::astype(
      mx::expand_dims(mx::concatenate({mx::real(bins), mx::imag(bins)}, -1), 0),
      dtype);

  mx::array hidden = Conv(mx::astype(frames, dtype), "hifigan.conv_pre", 0, 4);
  int resblock = 0;
  for (int stage = 0; stage < 3; ++stage) {
    const std::string index = std::to_string(stage);
    hidden = mx::repeat(LeakyRelu(hidden, 0.1), kUpsampleRates[stage], 1);
    hidden =
        Conv(hidden, "hifigan.ups." + index, kUpsampleKernels[stage] - 1, 0);
    if (stage == 2) {
      hidden = mx::concatenate(
          {mx::slice(hidden, {0, 1, 0}, {1, 2, hidden.shape(2)}), hidden}, 1);
    } else {
    }
    const int stride = kSourceStrides[stage];
    const mx::array downsampled =
        stride == 1
            ? Conv(source_spectrum, "hifigan.source_downs." + index, 0, 0)
            : Conv(source_spectrum, "hifigan.source_downs." + index, stride - 1,
                   0, stride);
    hidden = mx::add(hidden,
                     ResBlock(downsampled, "hifigan.source_resblocks." + index,
                              kSourceResBlockKernels[stage]));
    std::vector<mx::array> branches;
    for (int kernel : kResBlockKernels) {
      branches.push_back(ResBlock(
          hidden, "hifigan.resblocks." + std::to_string(resblock++), kernel));
    }
    hidden = mx::mean(mx::stack(branches, 0), 0);
  }
  hidden = mx::astype(Conv(LeakyRelu(hidden, 0.01), "hifigan.conv_post", 6, 0),
                      mx::float32);
  hidden = mx::squeeze(hidden, 0);
  const int count = hidden.shape(0);
  const mx::array magnitude =
      mx::minimum(mx::exp(mx::slice(hidden, {0, 0}, {count, kFftBins})),
                  mx::array(kMaxMagnitude));
  const mx::array phase =
      mx::sin(mx::slice(hidden, {0, kFftBins}, {count, 2 * kFftBins}));
  const mx::array real = mx::multiply(magnitude, mx::cos(phase));
  const mx::array imaginary = mx::multiply(magnitude, mx::sin(phase));
  const auto mirrored = [&](const mx::array &values) {
    return mx::slice(values, {0, kFftBins - 2}, {count, 0}, {1, -1});
  };
  const mx::array full_spectrum = mx::add(
      mx::astype(mx::concatenate({real, mirrored(real)}, -1), mx::complex64),
      mx::multiply(
          mx::array(mx::complex64_t{0.0f, 1.0f}),
          mx::astype(mx::concatenate(
                         {imaginary, mx::negative(mirrored(imaginary))}, -1),
                     mx::complex64)));
  const mx::array windowed =
      mx::multiply(mx::real(mx::fft::ifft(full_spectrum, -1)), window_);
  const mx::array window_sum = mx::maximum(
      OverlapAdd(mx::broadcast_to(mx::square(window_), {count, kFftSize})),
      mx::array(1e-8f));
  const mx::array samples = mx::divide(OverlapAdd(windowed), window_sum);
  const int sample_total = samples.shape(0);
  return mx::clip(mx::slice(samples, {half}, {sample_total - half}),
                  mx::array(-kAudioLimit), mx::array(kAudioLimit));
}

} // namespace cosyvoice3
