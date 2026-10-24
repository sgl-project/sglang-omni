// SPDX-License-Identifier: Apache-2.0
#include "dots_tts/dots_tts_model.h"

#include <algorithm>
#include <cassert>
#include <numeric>
#include <stdexcept>
#include <utility>

#include "common/audio.h"
#include "common/layers.h"
#include "dots_tts/dots_tts_audio.h"
#include "dots_tts/dots_tts_checkpoint.h"

namespace dots_tts {

namespace mx = mlx::core;

struct DotsTtsModel::FlowState {
  std::vector<mx::array> flow_history;
  std::vector<mx::array> unconditional_flow_history;
  int flow_sequence_length = 0;
  std::vector<qwen3_asr::KVCache> llm_caches;
  std::vector<mx::array> denorm_patch_history;
  mx::array null_speaker_conditioning{0.0f};
  bool drop_regenerated = false;
  std::optional<mx::array> last_prefill_hidden;
  mx::array rng_key{0.0f};
  SemanticEncoderCache patch_cache;
};

namespace {

qwen3_asr::Qwen3DecoderConfig
BackboneDecoderConfig(const nlohmann::json &config) {
  qwen3_asr::Qwen3DecoderConfig decoder;
  const int hidden_size = config.at("hidden_size").get<int>();
  decoder.num_hidden_layers = config.at("num_hidden_layers").get<int>();
  decoder.num_attention_heads = config.at("num_attention_heads").get<int>();
  decoder.num_key_value_heads = config.at("num_key_value_heads").get<int>();
  decoder.head_dim =
      config.value("head_dim", hidden_size / decoder.num_attention_heads);
  decoder.rope_theta = config.value("rope_theta", 1000000.0f);
  decoder.rms_norm_eps = config.value("rms_norm_eps", 1e-6f);
  decoder.qk_norm = false;
  decoder.apply_final_norm = true;
  decoder.compute_float32 = true;
  return decoder;
}

} // namespace

DotsLLM::DotsLLM(const DotsCheckpoint &checkpoint)
    : checkpoint_(checkpoint),
      decoder_(checkpoint.BackboneCheckpoint(), "llm.model",
               BackboneDecoderConfig(checkpoint.config())) {}

mx::array DotsLLM::EmbedTokens(const mx::array &ids) const {
  return mx::astype(decoder_.EmbedTokens(ids), mx::float32);
}

mx::array DotsLLM::Forward(const mx::array &embeddings,
                           std::vector<qwen3_asr::KVCache> &caches) const {
  return decoder_.Forward(embeddings, caches);
}

std::vector<qwen3_asr::KVCache> DotsLLM::NewCaches() const {
  return decoder_.NewCaches();
}

mx::array DotsLLM::EosLogits(const mx::array &hidden) const {
  const mx::array activated =
      qwen3_asr::Silu(checkpoint_.BackboneLinear(hidden, "eos_proj.0"));
  return checkpoint_.BackboneLinear(activated, "eos_proj.2");
}

DotsTtsModel::DotsTtsModel(const std::filesystem::path &model_directory) {
  checkpoint_ = std::make_unique<DotsCheckpoint>(model_directory);
  tokenizer_ = std::make_unique<DotsTokenizer>(model_directory);
  llm_ = std::make_unique<DotsLLM>(*checkpoint_);

  const nlohmann::json &cfg = checkpoint_->config();
  patch_size_ = cfg.at("patch_size").get<int>();
  latent_dim_ = cfg.at("latent_dim").get<int>();
  flow_hidden_size_ = cfg.at("DiT").at("hidden_size").get<int>();
  sample_rate_ = cfg.at("vocoder").value("sample_rate", 48000);
  max_sequence_length_ =
      cfg.value("max_position_embeddings", max_sequence_length_);
  meanflow_ =
      cfg.contains("meanflow") && cfg.at("meanflow").value("enabled", false);
  const std::vector<int> rates =
      cfg.at("vocoder").at("upsample_rates").get<std::vector<int>>();
  hop_size_ =
      std::accumulate(rates.begin(), rates.end(), 1, std::multiplies<int>());

  FlowHead head = LoadFlowHead(*checkpoint_);
  hidden_proj_ = std::make_unique<nn::Linear>(std::move(head.hidden_proj));
  latent_proj_ = std::make_unique<nn::Linear>(std::move(head.latent_proj));
  xvec_linear_ = std::make_unique<nn::Linear>(std::move(head.xvec_linear));
  xvec_norm_weight_ = std::move(head.xvec_norm_weight);
  xvec_norm_bias_ = std::move(head.xvec_norm_bias);
  dit_ = std::make_unique<DiT>(std::move(head.dit));
  coordinate_proj_ =
      std::make_unique<nn::Linear>(std::move(head.coordinate_proj));
  patch_encoder_ =
      std::make_unique<VAESemanticEncoder>(std::move(head.patch_encoder));

  vae_ = std::make_unique<AudioVAE>(LoadAudioVae(*checkpoint_));
  speaker_ = std::make_unique<CAMPPlus>(LoadSpeaker(*checkpoint_));
  fbank_ = std::make_unique<KaldiFbank>();
  flow_solver_ =
      std::make_unique<FlowSolver>(*dit_, *coordinate_proj_, latent_dim_);
}

mx::array DotsTtsModel::XvectorFromAudio48k(
    const std::vector<float> &audio_samples_48khz) const {
  const std::vector<float> audio_samples_16khz =
      Resample(audio_samples_48khz, sample_rate_, kSpeakerFbankSampleRate);
  const int maximum_sample_count = kSpeakerFbankSampleRate * 10;
  std::vector<float> cropped_audio_samples = audio_samples_16khz;
  if (static_cast<int>(cropped_audio_samples.size()) > maximum_sample_count) {
    cropped_audio_samples.resize(static_cast<size_t>(maximum_sample_count));
  }
  const std::vector<float> fbank_features = (*fbank_)(cropped_audio_samples);
  const int frame_count =
      static_cast<int>(fbank_features.size()) / kSpeakerMelBins;
  const mx::array fbank(fbank_features.data(), {frame_count, kSpeakerMelBins},
                        mx::float32);
  const mx::array speaker_embedding = (*speaker_)(mx::expand_dims(fbank, 0));
  return speaker_embedding;
}

DotsTtsModel::PromptConditioningResult DotsTtsModel::PreparePromptConditioning(
    const std::vector<float> &prompt_audio48k, float speaker_scale,
    bool use_prompt_prefill, mx::array rng_key) const {
  if (prompt_audio48k.empty()) {
    return {PromptConditioning{}, rng_key};
  }

  const int chunk = patch_size_ * hop_size_;
  std::vector<float> padded = prompt_audio48k;
  const int sample_count = static_cast<int>(padded.size());
  const int target = ((sample_count + chunk - 1) / chunk) * chunk;
  if (target > sample_count) {
    padded.resize(static_cast<size_t>(target), 0.0f);
  }
  const mx::array speaker_embedding =
      mx::multiply(XvectorFromAudio48k(padded), mx::array(speaker_scale));
  mx::array speaker_conditioning = mx::add(
      mx::matmul(speaker_embedding, mx::transpose(xvec_linear_->weight)),
      *xvec_linear_->bias);
  speaker_conditioning = mx::fast::layer_norm(
      speaker_conditioning, xvec_norm_weight_, xvec_norm_bias_, /*eps=*/1e-5f);
  // note (guozhihao-224): the prompt contributes its sampled latents only when
  // prefill is enabled (audio and transcript both present), matching the
  // reference codec, which keeps prompt_latents null otherwise; a reference
  // without a transcript must not consume the target audio's spans.
  if (!use_prompt_prefill) {
    return {
        PromptConditioning{speaker_conditioning, std::nullopt, std::nullopt},
        rng_key};
  }

  const mx::array wav(padded.data(), {1, 1, static_cast<int>(padded.size())},
                      mx::float32);
  const mx::array latent = vae_->Encode(wav);
  const std::vector<mx::array> halves = mx::split(latent, 2, 1);
  const mx::array &mean = halves[0];
  const mx::array &log_std = halves[1];
  const std::pair<mx::array, mx::array> prompt_keys =
      mx::random::split(rng_key);
  rng_key = prompt_keys.second;
  const mx::array noise =
      mx::random::normal(mean.shape(), mx::float32, prompt_keys.first);
  mx::array sampled = mx::add(mean, mx::multiply(noise, mx::exp(log_std)));
  sampled = mx::transpose(sampled, {0, 2, 1});
  sampled = mx::slice(sampled, {0, 0, 0},
                      {1, sampled.shape(1) - patch_size_, latent_dim_});
  mx::array normalized = mx::divide(
      mx::subtract(sampled, mx::expand_dims(checkpoint_->latent_mean(), 0)),
      mx::expand_dims(mx::sqrt(checkpoint_->latent_var()), 0));
  const int t = normalized.shape(1);
  const int s = t / patch_size_;
  const int keep = s * patch_size_;
  const mx::array patch_slice =
      mx::slice(normalized, {0, 0, 0}, {1, keep, normalized.shape(2)});
  const mx::array prompt_patches =
      mx::reshape(patch_slice, {1, s, patch_size_, latent_dim_});
  const mx::array prompt_denorm =
      mx::slice(sampled, {0, 0, 0}, {1, keep, sampled.shape(2)});
  return {
      PromptConditioning{speaker_conditioning, prompt_patches, prompt_denorm},
      rng_key};
}

std::vector<mx::array>
DotsTtsModel::BuildFmAttnMask(int flow_sequence_length) const {
  const int p = patch_size_;
  const int total = flow_sequence_length + p;
  std::vector<char> mask(static_cast<size_t>(total) * total, 0);
  const int block_start = flow_sequence_length - 1;

  if (block_start > 0) {
    for (int row = 0; row < block_start; ++row) {
      for (int col = 0; col <= row; ++col)
        mask[static_cast<size_t>(row) * total + col] = 1;
    }
  }
  if (flow_sequence_length > 0) {
    for (int col = 0; col < total; ++col)
      mask[static_cast<size_t>(block_start) * total + col] = 1;
  }
  for (int row = flow_sequence_length; row < total; ++row) {
    for (int col = 0; col < total; ++col)
      mask[static_cast<size_t>(row) * total + col] = 1;
  }

  const mx::array mask_flat(mask.data(), {total, total}, mx::bool_);
  return {mx::expand_dims(mask_flat, 0)};
}

void DotsTtsModel::AppendHiddenChunk(FlowState *state,
                                     const mx::array &hidden) const {
  mx::array last = mx::slice(hidden, {0, hidden.shape(1) - 1, 0},
                             {1, hidden.shape(1), hidden.shape(2)});
  mx::array projected =
      mx::add(mx::matmul(last, mx::transpose(hidden_proj_->weight)),
              *hidden_proj_->bias);
  state->flow_history.push_back(projected);
  if (!meanflow_) {
    const mx::array zeros = mx::zeros_like(last);
    mx::array null_projected =
        mx::add(mx::matmul(zeros, mx::transpose(hidden_proj_->weight)),
                *hidden_proj_->bias);
    state->unconditional_flow_history.push_back(null_projected);
  }
  state->flow_sequence_length += projected.shape(1);
}

void DotsTtsModel::AppendHistoryChunk(FlowState *state,
                                      const mx::array &latent) const {
  mx::array history_latent =
      mx::add(mx::matmul(latent, mx::transpose(latent_proj_->weight)),
              *latent_proj_->bias);
  state->flow_history.push_back(history_latent);
  state->unconditional_flow_history.push_back(history_latent);
  state->flow_sequence_length += history_latent.shape(1);
}

int DotsTtsModel::Prefill(
    const std::vector<int> &schedule_ids, const mx::array &schedule,
    FlowState *state, const std::vector<int> &span_positions,
    const std::optional<mx::array> &prompt_patches,
    const std::optional<mx::array> &prompt_latents) const {
  const int prompt_patch_count =
      prompt_patches.has_value() ? static_cast<int>(prompt_patches->shape(1))
                                 : 0;
  const int prefill_end = span_positions[prompt_patch_count];

  if (prefill_end == 0) {
    state->last_prefill_hidden = std::nullopt;
    return 0;
  }
  const mx::array head_ids = mx::slice(schedule, {0, 0}, {1, prefill_end});
  mx::array inputs_embeds = llm_->EmbedTokens(head_ids);
  if (prompt_patch_count > 0) {
    assert(prompt_latents.has_value());
    mx::array patch_emb = (*patch_encoder_)(*prompt_latents);
    for (int i = 0; i < prompt_patch_count; ++i) {
      const int span = span_positions[i];
      const mx::array row =
          mx::slice(patch_emb, {0, i, 0}, {1, i + 1, patch_emb.shape(2)});
      const mx::Shape start = {0, span, 0};
      const mx::Shape stop = {1, span + 1, inputs_embeds.shape(2)};
      inputs_embeds = mx::slice_update(inputs_embeds, row, start, stop);
    }
  }

  std::vector<qwen3_asr::KVCache> &caches = state->llm_caches;
  const mx::array llm_hiddens = llm_->Forward(inputs_embeds, caches);
  state->last_prefill_hidden =
      mx::slice(llm_hiddens, {0, prefill_end - 1, 0},
                {1, prefill_end, llm_hiddens.shape(2)});

  int cursor = 0;
  for (int prompt_index = 0; prompt_index < prompt_patch_count;
       ++prompt_index) {
    const int span_position = span_positions[prompt_index];
    if (span_position > cursor) {
      AppendHiddenChunk(state,
                        mx::slice(llm_hiddens, {0, span_position - 1, 0},
                                  {1, span_position, llm_hiddens.shape(2)}));
    }
    const mx::array prompt_patch =
        mx::slice(*prompt_patches, {0, prompt_index, 0, 0},
                  {1, prompt_index + 1, patch_size_, latent_dim_});
    const mx::array patch_flat =
        mx::reshape(prompt_patch, {1, patch_size_, latent_dim_});
    AppendHistoryChunk(state, patch_flat);
    const bool next_is_span =
        span_position + 1 < static_cast<int>(schedule_ids.size()) &&
        (schedule_ids[span_position + 1] == tokenizer_->audio_gen_span_id() ||
         schedule_ids[span_position + 1] == tokenizer_->audio_comp_span_id());
    if (next_is_span) {
      AppendHiddenChunk(
          state, mx::slice(llm_hiddens, {0, span_position, 0},
                           {1, span_position + 1, llm_hiddens.shape(2)}));
    }
    cursor = span_position + 1;
  }
  if (prefill_end > cursor) {
    AppendHiddenChunk(state, *state->last_prefill_hidden);
  }
  return prefill_end;
}

mx::array DotsTtsModel::ReencodeAndStep(FlowState *state,
                                        const mx::array &denorm_patch) const {
  state->denorm_patch_history.push_back(denorm_patch);
  const mx::array history = mx::concatenate(state->denorm_patch_history, 1);
  const mx::array emb = (*patch_encoder_).Append(state->patch_cache, history);
  return llm_->Forward(emb, state->llm_caches);
}

bool DotsTtsModel::ShouldStopAfterCurrentAudio(const mx::array &llm_hiddens,
                                               float eos_threshold) const {
  const mx::array logits = llm_->EosLogits(llm_hiddens);
  const mx::array probs = mx::softmax(mx::astype(logits, mx::float32), -1);
  const mx::array eos_p = mx::reshape(
      mx::slice(probs, {0, probs.shape(1) - 1, 1}, {1, probs.shape(1), 2}), {});
  mx::eval(eos_p);
  return eos_p.item<float>() > eos_threshold;
}

SynthesisResult
DotsTtsModel::Generate(const std::string &text,
                       const std::optional<std::string> &prompt_wav,
                       const std::optional<std::string> &prompt_text,
                       const GenerationOptions &options) const {
  mx::array rng_key = mx::random::key(static_cast<uint64_t>(options.seed));
  if (options.max_generate_length > kMaxGenerateLength) {
    throw std::invalid_argument("dots.tts max_generate_length must be in [1, " +
                                std::to_string(kMaxGenerateLength) + "]");
  }

  std::vector<float> prompt_audio48k;
  if (prompt_wav.has_value()) {
    const qwen3_asr::WavAudio wav = qwen3_asr::DecodeWavAudio(*prompt_wav);
    std::vector<float> trimmed = TrimSilence(wav.samples, /*top_db=*/30.0f);
    prompt_audio48k = Resample(trimmed, wav.sample_rate, sample_rate_);
    // note (guozhihao-224): the speaker fbank needs at least one 25 ms window
    // at 16 kHz; shorter references crash the x-vector trunk.
    if (static_cast<int>(prompt_audio48k.size()) <
        kSpeakerFbankSampleRate * 25 / 1000) {
      throw std::invalid_argument(
          "reference audio is too short for speaker embedding");
    }
  }
  const bool use_prompt_prefill = !prompt_audio48k.empty() &&
                                  prompt_text.has_value() &&
                                  !prompt_text->empty();

  const std::vector<int> schedule_ids = tokenizer_->BuildGenerationSchedule(
      text, prompt_text, options.language, options.max_generate_length);
  if (static_cast<int>(schedule_ids.size()) > max_sequence_length_) {
    throw std::invalid_argument("dots.tts schedule exceeds context length " +
                                std::to_string(max_sequence_length_));
  }

  const PromptConditioningResult prompt_conditioning =
      PreparePromptConditioning(prompt_audio48k, options.speaker_scale,
                                use_prompt_prefill, rng_key);
  const PromptConditioning &conditioning = prompt_conditioning.conditioning;
  rng_key = prompt_conditioning.next_rng_key;
  const int prompt_patch_count =
      conditioning.prompt_patches.has_value()
          ? static_cast<int>(conditioning.prompt_patches->shape(1))
          : 0;
  const mx::array schedule(schedule_ids.data(),
                           {1, static_cast<int>(schedule_ids.size())},
                           mx::int32);
  const int audio_gen_span = tokenizer_->audio_gen_span_id();
  const int audio_comp_span = tokenizer_->audio_comp_span_id();
  std::vector<int> span_positions;
  for (size_t i = 0; i < schedule_ids.size(); ++i) {
    if (schedule_ids[i] == audio_gen_span ||
        schedule_ids[i] == audio_comp_span) {
      span_positions.push_back(static_cast<int>(i));
    }
  }
  if (static_cast<int>(span_positions.size()) < prompt_patch_count + 1) {
    throw std::invalid_argument(
        "dots.tts schedule has too few audio spans for the prompt");
  }

  FlowState state;
  state.llm_caches = llm_->NewCaches();
  state.null_speaker_conditioning =
      mx::zeros({1, flow_hidden_size_}, mx::float32);
  state.drop_regenerated = use_prompt_prefill;
  state.rng_key = std::move(rng_key);
  state.denorm_patch_history.clear();
  if (conditioning.prompt_denorm_latents.has_value()) {
    for (int i = 0; i < prompt_patch_count; ++i) {
      state.denorm_patch_history.push_back(mx::slice(
          *conditioning.prompt_denorm_latents, {0, i * patch_size_, 0},
          {1, (i + 1) * patch_size_,
           conditioning.prompt_denorm_latents->shape(2)}));
    }
  }
  if (!state.denorm_patch_history.empty()) {
    (*patch_encoder_)
        .Seed(state.patch_cache,
              mx::concatenate(state.denorm_patch_history, 1));
  }

  const int position =
      Prefill(schedule_ids, schedule, &state, span_positions,
              conditioning.prompt_patches, conditioning.prompt_denorm_latents);
  mx::array llm_hiddens = state.last_prefill_hidden.value_or(mx::array(0.0f));
  const int schedule_length = static_cast<int>(schedule_ids.size());
  int span_cursor = static_cast<int>(
      std::lower_bound(span_positions.begin(), span_positions.end(), position) -
      span_positions.begin());

  std::vector<mx::array> emitted;
  bool stopped = false;
  int current_position = position;
  while (current_position < schedule_length && !stopped) {
    const int token_id = schedule_ids[current_position];
    if (token_id == audio_gen_span || token_id == audio_comp_span) {
      const bool stop_after =
          state.drop_regenerated
              ? false
              : ShouldStopAfterCurrentAudio(llm_hiddens, options.eos_threshold);
      const mx::array flow_sequence = mx::concatenate(state.flow_history, 1);
      const std::vector<mx::array> attn_mask =
          BuildFmAttnMask(state.flow_sequence_length);
      const mx::array pad =
          mx::zeros({1, patch_size_, flow_hidden_size_}, flow_sequence.dtype());
      const mx::array input_sequence = mx::concatenate({flow_sequence, pad}, 1);
      const std::pair<mx::array, mx::array> flow_keys =
          mx::random::split(state.rng_key);
      state.rng_key = flow_keys.second;
      const mx::array noise =
          mx::astype(mx::random::normal({1, patch_size_, latent_dim_},
                                        mx::float32, flow_keys.first),
                     flow_sequence.dtype());
      const mx::array speaker_conditioning =
          conditioning.speaker_conditioning.has_value()
              ? *conditioning.speaker_conditioning
              : state.null_speaker_conditioning;
      // note (guozhihao-224): MeanFlow checkpoints run one branch with a
      // duration-conditioned DiT and no classifier-free guidance.
      const mx::array patch = [&]() -> mx::array {
        if (meanflow_) {
          return flow_solver_->DenoiseMeanFlow(input_sequence, attn_mask[0],
                                               speaker_conditioning,
                                               options.num_steps, noise);
        }
        const mx::array unconditional_sequence = mx::concatenate(
            {mx::concatenate(state.unconditional_flow_history, 1), pad}, 1);
        return flow_solver_->Denoise(input_sequence, unconditional_sequence,
                                     attn_mask[0], speaker_conditioning,
                                     options.guidance_scale, options.num_steps,
                                     noise);
      }();
      mx::eval(patch);

      AppendHistoryChunk(&state, patch);
      const mx::array denorm = mx::add(
          mx::multiply(patch,
                       mx::expand_dims(mx::sqrt(checkpoint_->latent_var()), 0)),
          mx::expand_dims(checkpoint_->latent_mean(), 0));
      llm_hiddens = ReencodeAndStep(&state, denorm);
      if (current_position + 1 < schedule_length &&
          (schedule_ids[current_position + 1] == audio_gen_span ||
           schedule_ids[current_position + 1] == audio_comp_span)) {
        AppendHiddenChunk(&state, llm_hiddens);
      }
      ++current_position;
      ++span_cursor;
      if (state.drop_regenerated) {
        state.drop_regenerated = false;
      } else {
        emitted.push_back(denorm);
      }
      if (stop_after) {
        stopped = true;
      }
      continue;
    } else {
      const int next_audio_position =
          span_cursor < static_cast<int>(span_positions.size())
              ? span_positions[span_cursor]
              : schedule_length;
      const mx::array text_chunk =
          mx::slice(schedule, {0, current_position}, {1, next_audio_position});
      const mx::array embeddings = llm_->EmbedTokens(text_chunk);
      llm_hiddens = llm_->Forward(embeddings, state.llm_caches);
      AppendHiddenChunk(&state, llm_hiddens);
      current_position = next_audio_position;
    }
  }

  if (emitted.empty()) {
    throw std::runtime_error("dots.tts produced no payload latents");
  }

  const mx::array latents = mx::concatenate(emitted, 1);
  const mx::array latents_ct = mx::transpose(latents, {0, 2, 1});
  mx::array wav = (*vae_).Decode(latents_ct);
  const int wave_len = wav.shape(2);
  wav = mx::reshape(wav, {wave_len});
  mx::eval(wav);
  const float *data = wav.data<float>();
  std::vector<float> audio(data, data + wav.size());
  if (options.trim_onset) {
    audio = CleanOnset(audio, sample_rate_);
  }
  return SynthesisResult{std::move(audio), sample_rate_,
                         static_cast<int>(emitted.size())};
}

} // namespace dots_tts
