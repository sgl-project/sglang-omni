// SPDX-License-Identifier: Apache-2.0
#include "voxtral_tts/voxtral_tts_model.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace voxtral_tts {

namespace mx = mlx::core;

namespace {

constexpr const char *kDecoderPrefix = "language_model.model.model";
constexpr const char *kAcousticPrefix = "acoustic_transformer";
constexpr const char *kCodebookEmbeddings =
    "audio_codebook_embeddings.embeddings";
constexpr const char *kCodecWeightPrefix = "audio_tokenizer.";
// Note (Minyi Tang): the reference flow (#2788 and the CUDA model): Euler
// steps between 8 evenly spaced timesteps, guided 1.2 toward the
// conditioned velocity, with sinusoid time embeddings of period up to 10^4.
constexpr int kFlowTimesteps = 8;
constexpr double kGuidanceScale = 1.2;
constexpr float kTimeEmbeddingMaxPeriod = 10000.0f;
constexpr float kDefaultAcousticNormEps = 1e-5f;

// x times value as a scalar of x's dtype, as Python scalars combine.
mx::array Scale(const mx::array &x, double value) {
  return mx::multiply(x, mx::array(static_cast<float>(value), x.dtype()));
}

} // namespace

VoxtralTtsModel::LoadedWeights
VoxtralTtsModel::LoadWeights(const std::filesystem::path &directory) {
  LoadedWeights weights;
  for (const auto &[name, array] : qwen3_asr::LoadSafetensors(directory)) {
    if (name.rfind(kCodecWeightPrefix, 0) == 0) {
      weights.codec.insert_or_assign(name, array);
    } else {
      weights.model.insert_or_assign(name, array);
    }
  }
  return weights;
}

VoxtralTtsModel::VoxtralTtsModel(const std::filesystem::path &model_directory)
    : VoxtralTtsModel(model_directory,
                      qwen3_asr::ReadJson(model_directory / "config.json"),
                      LoadWeights(model_directory)) {}

VoxtralTtsModel::VoxtralTtsModel(const std::filesystem::path &model_directory,
                                 const nlohmann::json &config,
                                 LoadedWeights weights)
    : checkpoint_(std::move(weights.model), config),
      codec_(std::move(weights.codec),
             config.at("multimodal").at("audio_tokenizer_args")),
      tokenizer_(model_directory),
      decoder_(
          checkpoint_, kDecoderPrefix,
          {config.at("n_layers").get<int>(), config.at("n_heads").get<int>(),
           config.at("n_kv_heads").get<int>(), config.at("head_dim").get<int>(),
           config.at("norm_eps").get<float>(),
           config.at("rope_theta").get<float>(),
           /*has_query_key_norm=*/false,
           /*traditional_rope=*/true}),
      norm_eps_(config.at("norm_eps").get<float>()), codebook_offsets_(0),
      time_frequencies_(0.0f) {
  const nlohmann::json &audio = config.at("multimodal").at("audio_model_args");
  const nlohmann::json &acoustic = audio.at("acoustic_transformer_args");
  semantic_code_count_ = audio.at("semantic_codebook_size").get<int>();
  acoustic_codebook_count_ = audio.at("n_acoustic_codebook").get<int>();
  acoustic_levels_ = audio.at("acoustic_codebook_size").get<int>();
  acoustic_layer_count_ = acoustic.at("n_layers").get<int>();
  acoustic_head_count_ = acoustic.at("n_heads").get<int>();
  acoustic_key_value_head_count_ = acoustic.at("n_kv_heads").get<int>();
  acoustic_head_dim_ = acoustic.at("head_dim").get<int>();
  acoustic_norm_eps_ = acoustic.value("norm_eps", kDefaultAcousticNormEps);

  // Every codebook keeps rows for the special codes before its own.
  std::vector<int> offsets = {0};
  int next_offset = semantic_code_count_ + kSpecialAudioCodeCount;
  for (int codebook = 0; codebook < acoustic_codebook_count_; ++codebook) {
    offsets.push_back(next_offset);
    next_offset += acoustic_levels_ + kSpecialAudioCodeCount;
  }
  codebook_offsets_ =
      mx::array(offsets.begin(), {static_cast<int>(offsets.size())}, mx::int32);
  const int frequency_count = acoustic.at("dim").get<int>() / 2;
  time_frequencies_ = mx::exp(mx::divide(
      mx::multiply(
          mx::negative(mx::log(mx::array(kTimeEmbeddingMaxPeriod))),
          mx::arange(static_cast<double>(frequency_count), mx::float32)),
      mx::array(static_cast<float>(frequency_count))));

  std::vector<mx::array> loaded = {codebook_offsets_, time_frequencies_};
  for (const auto &[voice, frame_count] : tokenizer_.voice_frame_counts()) {
    const auto [arrays, metadata] = mx::load_safetensors(
        (model_directory / "voice_embedding" / (voice + ".safetensors"))
            .string());
    const mx::array embedding = arrays.at("embedding");
    if (embedding.ndim() != 2 || embedding.shape(0) != frame_count) {
      throw std::runtime_error("voice " + voice +
                               " does not have the frames tekken.json lists");
    } else {
    }
    voices_.emplace(voice, embedding);
    loaded.push_back(embedding);
  }
  mx::eval(loaded);
}

qwen3_asr::FinishReason VoxtralTtsModel::Generate(
    const std::vector<int> &prompt_ids, const std::string &voice,
    int max_frames, const std::atomic<bool> &cancel,
    const std::function<void(const mx::array &)> &on_frame) const {
  const auto found = voices_.find(voice);
  if (found == voices_.end()) {
    throw std::invalid_argument("unknown voice " + voice);
  } else {
  }
  const mx::array &voice_rows = found->second;
  if (std::count(prompt_ids.begin(), prompt_ids.end(),
                 tokenizer_.audio_token_id()) != voice_rows.shape(0)) {
    throw std::invalid_argument("the prompt's audio placeholders do not "
                                "match voice " +
                                voice);
  } else {
  }
  const mx::array ids = mx::array(
      prompt_ids.begin(), {1, static_cast<int>(prompt_ids.size())}, mx::int32);
  const mx::array embeddings = decoder_.EmbedTokens(ids);
  // The voice's rows replace the audio placeholders in order.
  const mx::array placeholders =
      mx::equal(mx::reshape(ids, {-1}), mx::array(tokenizer_.audio_token_id()));
  const mx::array row_indices =
      mx::clip(mx::subtract(mx::cumsum(mx::astype(placeholders, mx::int32), 0),
                            mx::array(1)),
               mx::array(0), mx::array(voice_rows.shape(0) - 1));
  const mx::array prompt_embeddings =
      mx::where(mx::reshape(placeholders, {1, -1, 1}),
                mx::expand_dims(mx::astype(mx::take(voice_rows, row_indices, 0),
                                           embeddings.dtype()),
                                0),
                embeddings);

  std::vector<qwen3_asr::KVCache> caches = decoder_.NewCaches();
  mx::array hidden = LastHidden(prompt_embeddings, caches);
  for (int frame = 0; frame < max_frames; ++frame) {
    if (cancel.load()) {
      throw qwen3_asr::TranscriptionCancelled();
    } else {
    }
    const mx::array codes = PredictFrame(hidden);
    mx::eval(codes);
    if (codes.data<int32_t>()[0] == kEndOfAudioCode) {
      return qwen3_asr::FinishReason::kStop;
    } else {
    }
    if (frame + 1 < max_frames) {
      hidden = LastHidden(EmbedFrame(codes), caches);
      // Note (Minyi Tang): the decoder steps on the GPU while on_frame
      // builds the codec's window.
      mx::async_eval(hidden);
    } else {
    }
    on_frame(codes);
  }
  return qwen3_asr::FinishReason::kLength;
}

mx::array
VoxtralTtsModel::LastHidden(const mx::array &embeddings,
                            std::vector<qwen3_asr::KVCache> &caches) const {
  const mx::array hidden = decoder_.Forward(embeddings, caches);
  const int length = hidden.shape(1);
  const int dimension = hidden.shape(2);
  return checkpoint_.RmsNorm(
      mx::reshape(mx::slice(hidden, {0, length - 1, 0}, {1, length, dimension}),
                  {1, dimension}),
      std::string(kDecoderPrefix) + ".norm", norm_eps_);
}

mx::array VoxtralTtsModel::PredictFrame(const mx::array &hidden) const {
  const std::string head = kAcousticPrefix;
  const mx::Dtype dtype = hidden.dtype();
  const mx::array logits =
      mx::astype(checkpoint_.Linear(hidden, head + ".semantic_codebook_output"),
                 mx::float32);
  // Neither the empty code nor the output's alignment padding is a choice.
  const mx::array indices = mx::arange(logits.shape(1));
  const mx::array semantic = mx::astype(
      mx::argmax(
          mx::where(mx::logical_and(
                        mx::greater(indices, mx::array(0)),
                        mx::less(indices, mx::array(semantic_code_count_ +
                                                    kSpecialAudioCodeCount))),
                    logits, mx::array(-std::numeric_limits<float>::infinity())),
          -1, true),
      mx::int32);

  mx::array latents = mx::random::normal({1, acoustic_codebook_count_}, dtype);
  const mx::array timesteps = mx::linspace(0.0, 1.0, kFlowTimesteps, dtype);
  // The unconditioned half sees no language hidden state.
  const mx::array language_features =
      checkpoint_.Linear(mx::concatenate({hidden, mx::zeros_like(hidden)}, 0),
                         head + ".llm_projection");
  for (int step = 0; step + 1 < kFlowTimesteps; ++step) {
    const mx::array timestep = mx::slice(timesteps, {step}, {step + 1});
    const mx::array step_size =
        mx::subtract(mx::slice(timesteps, {step + 1}, {step + 2}), timestep);
    const mx::array angles = mx::multiply(
        mx::reshape(timestep, {1, 1}), mx::expand_dims(time_frequencies_, 0));
    const mx::array time_embedding = mx::astype(
        mx::concatenate({mx::cos(angles), mx::sin(angles)}, -1), dtype);
    const mx::array velocity = Velocity(
        mx::concatenate({latents, latents}, 0),
        checkpoint_.Linear(mx::concatenate({time_embedding, time_embedding}, 0),
                           head + ".time_projection"),
        language_features);
    const mx::array guided = mx::add(
        Scale(mx::slice(velocity, {0, 0}, {1, acoustic_codebook_count_}),
              kGuidanceScale),
        Scale(mx::slice(velocity, {1, 0}, {2, acoustic_codebook_count_}),
              1.0 - kGuidanceScale));
    latents = mx::add(latents, mx::multiply(guided, step_size));
  }
  // Latents in [-1, 1] round to the nearest of the evenly spaced levels.
  const mx::array levels =
      mx::round(Scale(Scale(mx::add(mx::clip(latents, mx::array(-1.0f, dtype),
                                             mx::array(1.0f, dtype)),
                                    mx::array(1.0f, dtype)),
                            0.5),
                      acoustic_levels_ - 1));
  const mx::array acoustic =
      mx::add(mx::astype(levels, mx::int32), mx::array(kSpecialAudioCodeCount));
  return mx::reshape(mx::concatenate({semantic, acoustic}, 1), {-1});
}

mx::array VoxtralTtsModel::Velocity(const mx::array &latents,
                                    const mx::array &time_features,
                                    const mx::array &language_features) const {
  const std::string head = kAcousticPrefix;
  mx::array sequence =
      mx::stack({checkpoint_.Linear(latents, head + ".input_projection"),
                 time_features, language_features},
                1);
  for (int layer = 0; layer < acoustic_layer_count_; ++layer) {
    sequence = AcousticLayer(sequence, layer);
  }
  const int batch = sequence.shape(0);
  const int dimension = sequence.shape(2);
  // The velocity is read at the latents' position.
  return checkpoint_.Linear(
      mx::reshape(mx::slice(checkpoint_.RmsNorm(sequence, head + ".norm",
                                                acoustic_norm_eps_),
                            {0, 0, 0}, {batch, 1, dimension}),
                  {batch, dimension}),
      head + ".acoustic_codebook_output");
}

mx::array VoxtralTtsModel::AcousticLayer(const mx::array &x, int layer) const {
  const std::string prefix =
      std::string(kAcousticPrefix) + ".layers." + std::to_string(layer);
  const int batch = x.shape(0);
  const int length = x.shape(1);
  const mx::array normed =
      checkpoint_.RmsNorm(x, prefix + ".attention_norm", acoustic_norm_eps_);
  // Bidirectional attention over the three positions, keys and values
  // repeated for each query head that shares them.
  const auto heads = [&](const std::string &projection, int head_count,
                         int repeats) {
    return mx::transpose(
        mx::repeat(mx::reshape(checkpoint_.Linear(
                                   normed, prefix + ".attention." + projection),
                               {batch, length, head_count, acoustic_head_dim_}),
                   repeats, 2),
        {0, 2, 1, 3});
  };
  const int repeats = acoustic_head_count_ / acoustic_key_value_head_count_;
  const mx::array queries =
      Scale(heads("wq", acoustic_head_count_, 1),
            std::pow(static_cast<double>(acoustic_head_dim_), -0.5));
  const mx::array keys = heads("wk", acoustic_key_value_head_count_, repeats);
  const mx::array values = heads("wv", acoustic_key_value_head_count_, repeats);
  const mx::array weights = mx::astype(
      mx::softmax(
          mx::astype(mx::matmul(queries, mx::transpose(keys, {0, 1, 3, 2})),
                     mx::float32),
          -1),
      values.dtype());
  const mx::array hidden = mx::add(
      x,
      checkpoint_.Linear(
          mx::reshape(mx::transpose(mx::matmul(weights, values), {0, 2, 1, 3}),
                      {batch, length, -1}),
          prefix + ".attention.wo"));
  const mx::array feed_forward_input =
      checkpoint_.RmsNorm(hidden, prefix + ".ffn_norm", acoustic_norm_eps_);
  return mx::add(
      hidden,
      checkpoint_.Linear(
          mx::multiply(qwen3_asr::Silu(checkpoint_.Linear(
                           feed_forward_input, prefix + ".feed_forward.w1")),
                       checkpoint_.Linear(feed_forward_input,
                                          prefix + ".feed_forward.w3")),
          prefix + ".feed_forward.w2"));
}

mx::array VoxtralTtsModel::EmbedFrame(const mx::array &codes) const {
  return mx::sum(
      checkpoint_.Embed(mx::expand_dims(mx::add(codes, codebook_offsets_), 0),
                        kCodebookEmbeddings),
      1, true);
}

} // namespace voxtral_tts
