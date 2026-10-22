// SPDX-License-Identifier: Apache-2.0
#include "voxtral_tts/model.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>
#include <utility>

#include "mlx/fast.h"
#include "mlx/ops.h"
#include "mlx/random.h"
#include "voxtral_tts/speech_tokenizer.h"

namespace voxtral_tts {

namespace mx = mlx::core;

namespace {

constexpr int kSpecialTokenCount = 2;
constexpr int kAcousticDecodeIterations = 8;
constexpr float kClassifierFreeGuidanceAlpha = 1.2f;
constexpr float kMinimumSemanticClusterUsage = 1e-5f;
constexpr char kAudioCodeEmbeddingKey[] =
    "mm_audio_embeddings.audio_codebook_embeddings.embeddings.weight";

mx::array InterleaveQueryKeyWeight(const mx::array &weight, int head_count,
                                   int head_dimension) {
  const int half_head_dimension = head_dimension / 2;
  const int input_dimension = weight.shape(1);
  return mx::reshape(
      mx::transpose(mx::reshape(weight, {head_count, half_head_dimension, 2,
                                         input_dimension}),
                    {0, 2, 1, 3}),
      {head_count * head_dimension, input_dimension});
}

qwen3_asr::WeightMap LoadTextWeights(const qwen3_asr::WeightMap &source_weights,
                                     const TextConfig &config) {
  static const std::map<std::string, std::string> global_keys = {
      {"norm.weight", "norm.weight"},
      {"mm_audio_embeddings.tok_embeddings.weight", "embed_tokens.weight"}};
  static const std::map<std::string, std::string> layer_keys = {
      {"attention.wq.weight", "self_attn.q_proj.weight"},
      {"attention.wk.weight", "self_attn.k_proj.weight"},
      {"attention.wv.weight", "self_attn.v_proj.weight"},
      {"attention.wo.weight", "self_attn.o_proj.weight"},
      {"attention_norm.weight", "input_layernorm.weight"},
      {"feed_forward.w1.weight", "mlp.gate_proj.weight"},
      {"feed_forward.w2.weight", "mlp.down_proj.weight"},
      {"feed_forward.w3.weight", "mlp.up_proj.weight"},
      {"ffn_norm.weight", "post_attention_layernorm.weight"}};

  qwen3_asr::WeightMap text_weights;
  for (const auto &[checkpoint_name, weight] : source_weights) {
    const auto global_key = global_keys.find(checkpoint_name);
    if (global_key != global_keys.end()) {
      text_weights.emplace(global_key->second, weight);
      continue;
    } else {
    }
    if (checkpoint_name.rfind("layers.", 0) != 0) {
      continue;
    } else {
    }
    const size_t layer_number_start = std::string("layers.").size();
    const size_t layer_number_end =
        checkpoint_name.find('.', layer_number_start);
    if (layer_number_end == std::string::npos) {
      throw std::invalid_argument("invalid Voxtral layer weight name: " +
                                  checkpoint_name);
    } else {
    }
    const std::string layer_number = checkpoint_name.substr(
        layer_number_start, layer_number_end - layer_number_start);
    const std::string layer_weight_name =
        checkpoint_name.substr(layer_number_end + 1);
    const auto layer_key = layer_keys.find(layer_weight_name);
    if (layer_key == layer_keys.end()) {
      continue;
    } else {
    }
    mx::array remapped_weight = weight;
    if (layer_weight_name == "attention.wq.weight") {
      remapped_weight = InterleaveQueryKeyWeight(weight, config.head_count,
                                                 config.head_dimension);
    } else if (layer_weight_name == "attention.wk.weight") {
      remapped_weight = InterleaveQueryKeyWeight(weight, config.kv_head_count,
                                                 config.head_dimension);
    } else {
    }
    text_weights.emplace("layers." + layer_number + "." + layer_key->second,
                         remapped_weight);
  }
  return text_weights;
}

qwen3_asr::WeightMap
LoadAcousticWeights(const qwen3_asr::WeightMap &source_weights) {
  qwen3_asr::WeightMap acoustic_weights;
  for (const auto &[checkpoint_name, weight] : source_weights) {
    if (checkpoint_name.rfind("acoustic_transformer.", 0) != 0) {
      continue;
    } else {
    }
    const std::string layer_name =
        checkpoint_name.substr(std::string("acoustic_transformer.").size());
    if (layer_name == "time_embedding.inv_freq") {
      continue;
    } else {
    }
    acoustic_weights.emplace(layer_name, weight);
  }
  return acoustic_weights;
}

std::set<int>
TransposedConvolutionBlockIndices(const AudioTokenizerConfig &config) {
  std::set<int> transposed_block_indices;
  int decoder_block_index = 1;
  for (size_t stack_index = 0;
       stack_index + 1 < config.transformer_lengths.size(); ++stack_index) {
    ++decoder_block_index;
    if (config.convolution_kernels[stack_index + 1] != 1 ||
        config.convolution_strides[stack_index + 1] != 1) {
      transposed_block_indices.insert(decoder_block_index);
      ++decoder_block_index;
    } else {
    }
  }
  return transposed_block_indices;
}

bool IsTransposedConvolution(const std::string &module_name,
                             const std::set<int> &transposed_block_indices) {
  const std::string prefix = "audio_tokenizer.decoder_blocks.";
  if (module_name.rfind(prefix, 0) != 0) {
    return false;
  } else {
  }
  const size_t block_number_start = prefix.size();
  const size_t block_number_end = module_name.find('.', block_number_start);
  if (block_number_end == std::string::npos) {
    return false;
  } else {
  }
  const int block_index = std::stoi(module_name.substr(
      block_number_start, block_number_end - block_number_start));
  return transposed_block_indices.contains(block_index);
}

qwen3_asr::WeightMap
LoadAudioTokenizerWeights(const qwen3_asr::WeightMap &source_weights,
                          const AudioTokenizerConfig &config) {
  const std::set<int> transposed_block_indices =
      TransposedConvolutionBlockIndices(config);
  qwen3_asr::WeightMap audio_weights;
  const std::string semantic_embedding_key =
      "audio_tokenizer.quantizer.semantic_codebook.embedding_sum";
  const std::string semantic_usage_key =
      "audio_tokenizer.quantizer.semantic_codebook.cluster_usage";
  const auto embedding_sum = source_weights.find(semantic_embedding_key);
  const auto cluster_usage = source_weights.find(semantic_usage_key);
  if (embedding_sum == source_weights.end() ||
      cluster_usage == source_weights.end()) {
    throw std::runtime_error(
        "Voxtral checkpoint is missing semantic codebook weights");
  } else {
  }
  const mx::array usage = mx::maximum(
      mx::expand_dims(mx::astype(cluster_usage->second, mx::float32), 1),
      mx::array(kMinimumSemanticClusterUsage, mx::float32));
  audio_weights.emplace(
      "semantic_embedding",
      mx::astype(mx::astype(embedding_sum->second, mx::float32) / usage,
                 embedding_sum->second.dtype()));

  for (const auto &[checkpoint_name, checkpoint_weight] : source_weights) {
    if (checkpoint_name.rfind("audio_tokenizer.", 0) != 0) {
      continue;
    } else {
    }
    const std::string decoder_name =
        checkpoint_name.substr(std::string("audio_tokenizer.").size());
    if (decoder_name.rfind("quantizer.", 0) == 0 ||
        decoder_name.ends_with(".parametrizations.weight.original0") ||
        decoder_name.ends_with(".parametrizations.weight.original1")) {
      continue;
    } else {
    }
    if (decoder_name.ends_with(".conv.weight")) {
      const bool is_transposed_convolution = IsTransposedConvolution(
          "audio_tokenizer." + decoder_name, transposed_block_indices);
      const mx::array reordered_weight =
          is_transposed_convolution
              ? mx::transpose(checkpoint_weight, {1, 2, 0})
              : mx::transpose(checkpoint_weight, {0, 2, 1});
      audio_weights.emplace(decoder_name, reordered_weight);
    } else {
      audio_weights.emplace(decoder_name, checkpoint_weight);
    }
  }

  for (const auto &[checkpoint_name, checkpoint_weight] : source_weights) {
    if (!checkpoint_name.ends_with(".parametrizations.weight.original1")) {
      continue;
    } else {
    }
    const std::string module_name = checkpoint_name.substr(
        0, checkpoint_name.size() -
               std::string(".parametrizations.weight.original1").size());
    const std::string scale_name =
        module_name + ".parametrizations.weight.original0";
    const auto scale_weight = source_weights.find(scale_name);
    if (scale_weight == source_weights.end()) {
      throw std::runtime_error(
          "Voxtral checkpoint is missing weight norm scale " + scale_name);
    } else {
    }
    const mx::array weight_norm = mx::astype(checkpoint_weight, mx::float32);
    const mx::array scale = mx::astype(scale_weight->second, mx::float32);
    const mx::array norm =
        mx::sqrt(mx::sum(mx::multiply(weight_norm, weight_norm), {1, 2}, true));
    const mx::array effective_weight = mx::astype(
        weight_norm * scale / mx::maximum(norm, mx::array(1e-12f, mx::float32)),
        checkpoint_weight.dtype());
    const std::string decoder_name =
        module_name.substr(std::string("audio_tokenizer.").size());
    const bool is_transposed_convolution =
        IsTransposedConvolution(module_name, transposed_block_indices);
    const mx::array reordered_weight =
        is_transposed_convolution ? mx::transpose(effective_weight, {1, 2, 0})
                                  : mx::transpose(effective_weight, {0, 2, 1});
    const std::string weight_name = decoder_name + ".weight";
    audio_weights.insert_or_assign(weight_name, reordered_weight);
  }
  return audio_weights;
}

} // namespace

struct VoxtralModel::TextDecoder {
  const qwen3_asr::Checkpoint &checkpoint;
  TextConfig config;

  std::vector<qwen3_asr::KVCache> NewCaches() const {
    return std::vector<qwen3_asr::KVCache>(config.layer_count);
  }

  mx::array Embed(const mx::array &token_ids) const {
    return checkpoint.Embed(token_ids, "embed_tokens");
  }

  mx::array Forward(const mx::array &embeddings,
                    std::vector<qwen3_asr::KVCache> &caches) const {
    const int batch_size = embeddings.shape(0);
    const int token_count = embeddings.shape(1);
    mx::array hidden_states = embeddings;
    for (int layer_index = 0; layer_index < config.layer_count; ++layer_index) {
      const std::string layer_prefix = "layers." + std::to_string(layer_index);
      const mx::array normalized =
          checkpoint.RmsNorm(hidden_states, layer_prefix + ".input_layernorm",
                             config.norm_epsilon);
      mx::array queries = mx::reshape(
          checkpoint.Linear(normalized, layer_prefix + ".self_attn.q_proj"),
          {batch_size, token_count, config.head_count, config.head_dimension});
      mx::array keys = mx::reshape(
          checkpoint.Linear(normalized, layer_prefix + ".self_attn.k_proj"),
          {batch_size, token_count, config.kv_head_count,
           config.head_dimension});
      const mx::array values = mx::reshape(
          checkpoint.Linear(normalized, layer_prefix + ".self_attn.v_proj"),
          {batch_size, token_count, config.kv_head_count,
           config.head_dimension});
      queries = mx::fast::rope(mx::transpose(queries, {0, 2, 1, 3}),
                               config.head_dimension, false, config.rope_theta,
                               1.0f, caches[layer_index].offset());
      keys = mx::fast::rope(mx::transpose(keys, {0, 2, 1, 3}),
                            config.head_dimension, false, config.rope_theta,
                            1.0f, caches[layer_index].offset());
      auto [cached_keys, cached_values] = caches[layer_index].UpdateAndFetch(
          keys, mx::transpose(values, {0, 2, 1, 3}));
      const mx::array attended = mx::fast::scaled_dot_product_attention(
          queries, cached_keys, cached_values,
          std::pow(static_cast<double>(config.head_dimension), -0.5),
          token_count > 1 ? "causal" : "");
      const mx::array attention_output =
          checkpoint.Linear(mx::reshape(mx::transpose(attended, {0, 2, 1, 3}),
                                        {batch_size, token_count, -1}),
                            layer_prefix + ".self_attn.o_proj");
      hidden_states = hidden_states + attention_output;
      const mx::array mlp_input = checkpoint.RmsNorm(
          hidden_states, layer_prefix + ".post_attention_layernorm",
          config.norm_epsilon);
      const mx::array gated =
          qwen3_asr::Silu(
              checkpoint.Linear(mlp_input, layer_prefix + ".mlp.gate_proj")) *
          checkpoint.Linear(mlp_input, layer_prefix + ".mlp.up_proj");
      hidden_states = hidden_states +
                      checkpoint.Linear(gated, layer_prefix + ".mlp.down_proj");
    }
    const int last_token_index = token_count - 1;
    const mx::array last_hidden_state =
        mx::slice(hidden_states, {0, last_token_index, 0},
                  {batch_size, token_count, hidden_states.shape(2)});
    return mx::squeeze(
        checkpoint.RmsNorm(last_hidden_state, "norm", config.norm_epsilon), 1);
  }
};

struct VoxtralModel::AcousticHead {
  const qwen3_asr::Checkpoint &checkpoint;
  AudioModelConfig config;

  mx::array Linear(const mx::array &input, const std::string &name) const {
    return checkpoint.Linear(input, name);
  }

  mx::array PredictVelocity(const mx::array &acoustic_latents,
                            const mx::array &language_hidden_states,
                            const mx::array &timestep_embedding) const {
    const mx::array acoustic_features =
        Linear(mx::astype(acoustic_latents, language_hidden_states.dtype()),
               "input_projection");
    const mx::array time_features =
        Linear(mx::astype(timestep_embedding, language_hidden_states.dtype()),
               "time_projection");
    const mx::array language_features =
        Linear(language_hidden_states, "llm_projection");
    mx::array hidden_states =
        mx::stack({acoustic_features, time_features, language_features}, 1);
    const AcousticConfig &transformer = config.transformer;
    const int batch_size = hidden_states.shape(0);
    const int token_count = hidden_states.shape(1);
    const int kv_repeats = transformer.head_count / transformer.kv_head_count;
    for (int layer_index = 0; layer_index < transformer.layer_count;
         ++layer_index) {
      const std::string layer_prefix = "layers." + std::to_string(layer_index);
      const mx::array attention_input =
          checkpoint.RmsNorm(hidden_states, layer_prefix + ".attention_norm",
                             transformer.norm_epsilon);
      mx::array queries =
          mx::reshape(Linear(attention_input, layer_prefix + ".attention.wq"),
                      {batch_size, token_count, transformer.head_count,
                       transformer.head_dimension});
      mx::array keys =
          mx::reshape(Linear(attention_input, layer_prefix + ".attention.wk"),
                      {batch_size, token_count, transformer.kv_head_count,
                       transformer.head_dimension});
      mx::array values =
          mx::reshape(Linear(attention_input, layer_prefix + ".attention.wv"),
                      {batch_size, token_count, transformer.kv_head_count,
                       transformer.head_dimension});
      if (kv_repeats > 1) {
        keys = mx::repeat(keys, kv_repeats, 2);
        values = mx::repeat(values, kv_repeats, 2);
      } else {
      }
      queries = mx::transpose(queries, {0, 2, 1, 3}) *
                mx::array(1.0f / std::sqrt(transformer.head_dimension),
                          queries.dtype());
      keys = mx::transpose(keys, {0, 2, 1, 3});
      values = mx::transpose(values, {0, 2, 1, 3});
      const mx::array attention_scores = mx::astype(
          mx::matmul(queries, mx::transpose(keys, {0, 1, 3, 2})), mx::float32);
      const mx::array attention_weights =
          mx::astype(softmax(attention_scores, -1), values.dtype());
      const mx::array attended = mx::matmul(attention_weights, values);
      const mx::array attention_output = Linear(
          mx::reshape(mx::transpose(attended, {0, 2, 1, 3}),
                      {batch_size, token_count,
                       transformer.head_count * transformer.head_dimension}),
          layer_prefix + ".attention.wo");
      hidden_states = hidden_states + attention_output;
      const mx::array feed_forward_input = checkpoint.RmsNorm(
          hidden_states, layer_prefix + ".ffn_norm", transformer.norm_epsilon);
      const mx::array feed_forward_gated =
          qwen3_asr::Silu(
              Linear(feed_forward_input, layer_prefix + ".feed_forward.w1")) *
          Linear(feed_forward_input, layer_prefix + ".feed_forward.w3");
      hidden_states = hidden_states + Linear(feed_forward_gated,
                                             layer_prefix + ".feed_forward.w2");
    }
    const mx::array normalized =
        checkpoint.RmsNorm(hidden_states, "norm", transformer.norm_epsilon);
    const mx::array first_position =
        mx::slice(normalized, {0, 0, 0}, {batch_size, 1, normalized.shape(2)});
    return Linear(mx::squeeze(first_position, 1), "acoustic_codebook_output");
  }

  mx::array DecodeOneFrame(const mx::array &semantic_codes,
                           const mx::array &language_hidden_states) const {
    const int batch_size = semantic_codes.shape(0);
    const mx::array should_decode =
        mx::not_equal(semantic_codes, mx::array(1, mx::int32));
    mx::array sampled =
        mx::random::normal({batch_size, config.acoustic_codebook_count},
                           language_hidden_states.dtype());
    const mx::array unconditional_hidden_states =
        mx::zeros_like(language_hidden_states);
    const mx::array timesteps = mx::linspace(
        0.0, 1.0, kAcousticDecodeIterations, language_hidden_states.dtype());
    for (int iteration_index = 0;
         iteration_index + 1 < kAcousticDecodeIterations; ++iteration_index) {
      const mx::array timestep =
          mx::slice(timesteps, {iteration_index}, {iteration_index + 1});
      const mx::array step_size =
          mx::slice(timesteps, {iteration_index + 1}, {iteration_index + 2}) -
          timestep;
      const int acoustic_dimension = config.transformer.dimension;
      const int half_dimension = acoustic_dimension / 2;
      const mx::array frequencies =
          mx::exp(-std::log(10000.0f) *
                  mx::astype(mx::arange(half_dimension), mx::float32) /
                  static_cast<float>(half_dimension));
      const mx::array batch_timesteps =
          mx::broadcast_to(mx::reshape(timestep, {1, 1}), {batch_size, 1});
      const mx::array angles =
          batch_timesteps * mx::expand_dims(frequencies, 0);
      const mx::array timestep_embedding =
          mx::astype(mx::concatenate({mx::cos(angles), mx::sin(angles)}, -1),
                     language_hidden_states.dtype());
      const mx::array duplicated_latents =
          mx::concatenate({sampled, sampled}, 0);
      const mx::array duplicated_language = mx::concatenate(
          {language_hidden_states, unconditional_hidden_states}, 0);
      const mx::array duplicated_timestep =
          mx::concatenate({timestep_embedding, timestep_embedding}, 0);
      const mx::array velocity = PredictVelocity(
          duplicated_latents, duplicated_language, duplicated_timestep);
      const mx::array conditional_velocity =
          mx::slice(velocity, {0, 0}, {batch_size, velocity.shape(1)});
      const mx::array unconditional_velocity = mx::slice(
          velocity, {batch_size, 0}, {2 * batch_size, velocity.shape(1)});
      const mx::Dtype velocity_dtype = conditional_velocity.dtype();
      const mx::array guided_velocity =
          mx::array(kClassifierFreeGuidanceAlpha, velocity_dtype) *
              conditional_velocity +
          mx::array(1.0f - kClassifierFreeGuidanceAlpha, velocity_dtype) *
              unconditional_velocity;
      sampled = sampled + guided_velocity * step_size;
    }
    const mx::Dtype sampled_dtype = sampled.dtype();
    sampled = mx::clip(sampled, mx::array(-1.0f, sampled_dtype),
                       mx::array(1.0f, sampled_dtype));
    const mx::array quantized = mx::round(
        (sampled + mx::array(1.0f, sampled_dtype)) *
        mx::array(0.5f, sampled_dtype) *
        mx::array(static_cast<float>(config.acoustic_codebook_size - 1),
                  sampled_dtype));
    const mx::array acoustic_codes =
        mx::astype(quantized, mx::int32) + kSpecialTokenCount;
    return mx::where(mx::expand_dims(should_decode, 1), acoustic_codes,
                     mx::array(kSpecialTokenCount, mx::int32));
  }

  mx::array operator()(const mx::array &language_hidden_states) const {
    mx::array semantic_logits =
        mx::astype(Linear(language_hidden_states, "semantic_codebook_output"),
                   mx::float32);
    const mx::array semantic_indices = mx::arange(semantic_logits.shape(1));
    const mx::array allowed_codes =
        mx::less(semantic_indices,
                 mx::array(config.semantic_codebook_size + kSpecialTokenCount,
                           mx::int32));
    semantic_logits =
        mx::where(mx::expand_dims(allowed_codes, 0), semantic_logits,
                  mx::array(-std::numeric_limits<float>::infinity()));
    const mx::array not_empty_token =
        mx::not_equal(semantic_indices, mx::array(0, mx::int32));
    semantic_logits =
        mx::where(mx::expand_dims(not_empty_token, 0), semantic_logits,
                  mx::array(-std::numeric_limits<float>::infinity()));
    const mx::array semantic_codes =
        mx::astype(mx::argmax(semantic_logits, -1), mx::int32);
    const mx::array acoustic_codes =
        DecodeOneFrame(semantic_codes, language_hidden_states);
    return mx::concatenate({mx::expand_dims(semantic_codes, 1), acoustic_codes},
                           1);
  }
};

VoxtralModel::~VoxtralModel() = default;

VoxtralModel::VoxtralModel(const VoxtralConfig &config)
    : config_(config), audio_code_embeddings_(mx::array(0.0f)) {
  const nlohmann::json parameters =
      qwen3_asr::ReadJson(config.checkpoint_directory / "params.json");
  qwen3_asr::WeightMap source_weights =
      qwen3_asr::LoadSafetensors(config.checkpoint_directory);
  const auto audio_embedding = source_weights.find(kAudioCodeEmbeddingKey);
  if (audio_embedding == source_weights.end()) {
    throw std::runtime_error(std::string("Voxtral checkpoint is missing ") +
                             kAudioCodeEmbeddingKey);
  } else {
  }
  audio_code_embeddings_ = audio_embedding->second;
  text_checkpoint_ = std::make_unique<qwen3_asr::Checkpoint>(
      LoadTextWeights(source_weights, config.text), parameters);
  acoustic_checkpoint_ = std::make_unique<qwen3_asr::Checkpoint>(
      LoadAcousticWeights(source_weights), parameters);
  audio_tokenizer_ = std::make_unique<AudioTokenizerDecoder>(
      config.audio_tokenizer,
      std::make_unique<qwen3_asr::Checkpoint>(
          LoadAudioTokenizerWeights(source_weights, config.audio_tokenizer),
          parameters));
  text_decoder_ = std::make_unique<TextDecoder>(
      TextDecoder{*text_checkpoint_, config.text});
  acoustic_head_ = std::make_unique<AcousticHead>(
      AcousticHead{*acoustic_checkpoint_, config.audio});
  mx::eval(audio_code_embeddings_);
}

const VoxtralConfig &VoxtralModel::config() const { return config_; }

mx::array VoxtralModel::LoadVoiceEmbedding(const std::string &voice) const {
  const auto voice_entry = config_.voices.find(voice);
  if (voice_entry == config_.voices.end()) {
    throw std::invalid_argument("unknown Voxtral voice: " + voice);
  } else {
  }
  const std::filesystem::path voice_path = config_.checkpoint_directory /
                                           "voice_embedding" /
                                           (voice + ".safetensors");
  if (!std::filesystem::is_regular_file(voice_path)) {
    throw std::runtime_error(
        "native Voxtral voice embedding is missing; run "
        "voxtral_tts_convert_voices on the checkpoint first, expected " +
        voice_path.string());
  } else {
  }
  auto [weights, metadata] = mx::load_safetensors(voice_path.string());
  if (weights.size() != 1 || !weights.contains("voice_embedding")) {
    throw std::runtime_error(
        "Voxtral voice safetensors must contain voice_embedding");
  } else {
  }
  mx::array voice_embedding = weights.at("voice_embedding");
  if (voice_embedding.ndim() == 3 && voice_embedding.shape(0) == 1) {
    voice_embedding = mx::squeeze(voice_embedding, 0);
  } else {
  }
  if (voice_embedding.ndim() != 2 ||
      voice_embedding.shape(1) != config_.text.dimension) {
    throw std::runtime_error(
        "Voxtral voice embedding has an unexpected shape: " +
        voice_path.string());
  } else {
  }
  return mx::astype(voice_embedding, audio_code_embeddings_.dtype());
}

GeneratedCodes
VoxtralModel::Generate(const std::vector<int> &prompt_token_ids,
                       const mx::array &voice_embedding, int max_new_tokens,
                       uint64_t seed,
                       const std::atomic<bool> &cancelled) const {
  if (prompt_token_ids.empty() || max_new_tokens < 1) {
    throw std::invalid_argument(
        "Voxtral prompt and max_new_tokens must be positive");
  } else {
  }
  mx::random::seed(seed);
  const mx::array input_ids(prompt_token_ids.data(),
                            {1, static_cast<int>(prompt_token_ids.size())},
                            mx::int32);
  mx::array prompt_embeddings = text_decoder_->Embed(input_ids);
  int voice_frame_index = 0;
  for (size_t prompt_index = 0; prompt_index < prompt_token_ids.size();
       ++prompt_index) {
    if (prompt_token_ids[prompt_index] != config_.audio.audio_token_id) {
      continue;
    } else {
    }
    if (voice_frame_index >= voice_embedding.shape(0)) {
      throw std::invalid_argument(
          "Voxtral voice embedding has fewer frames than prompt placeholders");
    } else {
    }
    const mx::array voice_frame =
        mx::reshape(mx::take(voice_embedding, voice_frame_index, 0),
                    {1, 1, voice_embedding.shape(1)});
    prompt_embeddings = mx::slice_update(
        prompt_embeddings, voice_frame, {0, static_cast<int>(prompt_index), 0},
        {1, static_cast<int>(prompt_index) + 1, prompt_embeddings.shape(2)});
    ++voice_frame_index;
  }
  if (voice_frame_index != voice_embedding.shape(0)) {
    throw std::invalid_argument("Voxtral voice embedding frame count does not "
                                "match prompt placeholders");
  } else {
  }
  std::vector<qwen3_asr::KVCache> caches = text_decoder_->NewCaches();
  mx::array language_hidden_states =
      text_decoder_->Forward(prompt_embeddings, caches);
  std::vector<std::vector<int>> generated_frames;
  generated_frames.reserve(max_new_tokens);
  std::string finish_reason = "length";
  for (int frame_index = 0; frame_index < max_new_tokens; ++frame_index) {
    if (cancelled.load()) {
      throw std::runtime_error("Voxtral speech generation cancelled");
    } else {
    }
    const mx::array audio_codes = (*acoustic_head_)(language_hidden_states);
    mx::eval(audio_codes);
    const int semantic_code = static_cast<int>(
        mx::slice(audio_codes, {0, 0}, {1, 1}).item<uint32_t>());
    if (semantic_code == 1) {
      finish_reason = "stop";
      break;
    } else {
    }
    std::vector<int> frame_codes;
    frame_codes.reserve(config_.audio.acoustic_codebook_count + 1);
    for (int codebook_index = 0;
         codebook_index <= config_.audio.acoustic_codebook_count;
         ++codebook_index) {
      frame_codes.push_back(static_cast<int>(
          mx::slice(audio_codes, {0, codebook_index}, {1, codebook_index + 1})
              .item<uint32_t>()));
    }
    if (frame_index + 1 < max_new_tokens) {
      std::vector<int> codebook_offsets = {0};
      const int semantic_codebook_size =
          config_.audio.semantic_codebook_size + kSpecialTokenCount;
      int running_offset = semantic_codebook_size;
      for (int codebook_index = 0;
           codebook_index < config_.audio.acoustic_codebook_count;
           ++codebook_index) {
        codebook_offsets.push_back(running_offset);
        running_offset +=
            config_.audio.acoustic_codebook_size + kSpecialTokenCount;
      }
      std::vector<int> shifted_codes = frame_codes;
      for (size_t codebook_index = 0; codebook_index < shifted_codes.size();
           ++codebook_index) {
        shifted_codes[codebook_index] += codebook_offsets[codebook_index];
      }
      const mx::array code_indices(shifted_codes.data(),
                                   {1, static_cast<int>(shifted_codes.size())},
                                   mx::int32);
      const mx::array feedback =
          mx::sum(mx::take(audio_code_embeddings_, code_indices, 0), 1);
      language_hidden_states =
          text_decoder_->Forward(mx::expand_dims(feedback, 1), caches);
      mx::eval(language_hidden_states);
    } else {
    }
    generated_frames.push_back(std::move(frame_codes));
  }
  if (generated_frames.empty()) {
    throw std::runtime_error("Voxtral TTS generated no audio frames");
  } else {
  }
  return {std::move(generated_frames), finish_reason};
}

std::vector<float>
VoxtralModel::DecodeAudio(const std::vector<std::vector<int>> &frames,
                          const std::atomic<bool> &cancelled) const {
  if (cancelled.load()) {
    throw std::runtime_error("Voxtral speech generation cancelled");
  } else {
  }
  if (frames.empty()) {
    throw std::invalid_argument(
        "Voxtral audio decoder requires at least one frame");
  } else {
  }
  const int codebook_count = config_.audio.acoustic_codebook_count + 1;
  for (const std::vector<int> &frame : frames) {
    if (frame.size() != static_cast<size_t>(codebook_count)) {
      throw std::invalid_argument(
          "Voxtral audio frame has an unexpected codebook count");
    } else {
    }
  }
  const int decoded_frame_count = static_cast<int>(frames.size()) + 2;
  std::vector<int> decoder_codes;
  decoder_codes.reserve(static_cast<size_t>(codebook_count) *
                        static_cast<size_t>(decoded_frame_count));
  const int first_frame = 0;
  for (int codebook_index = 0; codebook_index < codebook_count;
       ++codebook_index) {
    decoder_codes.push_back(frames[first_frame][codebook_index] -
                            kSpecialTokenCount);
    decoder_codes.push_back(frames[first_frame][codebook_index] -
                            kSpecialTokenCount);
    for (const std::vector<int> &frame : frames) {
      decoder_codes.push_back(frame[codebook_index] - kSpecialTokenCount);
    }
  }
  const mx::array audio_code_tensor(decoder_codes.data(),
                                    {1, codebook_count, decoded_frame_count},
                                    mx::int32);
  mx::array waveform =
      mx::reshape(audio_tokenizer_->Decode(audio_code_tensor), {-1});
  if (cancelled.load()) {
    throw std::runtime_error("Voxtral speech generation cancelled");
  } else {
  }
  const int trim_sample_count = 2 * audio_tokenizer_->downsample_factor();
  if (waveform.shape(0) <= trim_sample_count) {
    throw std::runtime_error("Voxtral audio decoder returned too few samples");
  } else {
  }
  waveform = mx::slice(waveform, {trim_sample_count}, {waveform.shape(0)});
  const int fade_sample_count = audio_tokenizer_->sampling_rate_hz() / 100;
  const int fade_length = std::min(fade_sample_count, waveform.shape(0));
  const mx::array fade = mx::linspace(0.0, 1.0, fade_sample_count, mx::float32);
  const mx::array first_samples = mx::slice(waveform, {0}, {fade_length});
  waveform = mx::slice_update(
      waveform, first_samples * mx::slice(fade, {0}, {fade_length}), {0},
      {fade_length});
  waveform = mx::astype(waveform, mx::float32);
  mx::eval(waveform);
  if (cancelled.load()) {
    throw std::runtime_error("Voxtral speech generation cancelled");
  } else {
  }
  return std::vector<float>(waveform.data<float>(),
                            waveform.data<float>() + waveform.size());
}

} // namespace voxtral_tts
