// SPDX-License-Identifier: Apache-2.0
// A port of mlx-audio's zonos3/backbone.py, generation.py and model.py, op for
// op, so a seeded request samples the same codes. The gated delta recurrence
// of multi-row calls is the scalar-gate Metal kernel of mlx-lm
// (mlx_lm/models/gated_delta.py, MIT, Copyright 2025 Apple Inc.) that the
// reference vendors; one-row calls take its compiled ops step.
#include "zonos3/zonos3_model.h"

#include <CommonCrypto/CommonDigest.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

#include "common/layers.h"
#include "common/npz.h"

namespace zonos3 {

namespace mx = mlx::core;

namespace {

// Global attention keys and values grow in blocks of this many rows.
constexpr int kKeyValueBlockRows = 256;
constexpr float kGatedDeltaNormEpsilon = 1e-5f;
constexpr float kQueryKeyL2Epsilon = 1e-6f;
// Reference audio beyond this many seconds is not read.
constexpr int kMaxReferenceSeconds = 60;
constexpr size_t kReferenceCacheEntries = 4;
constexpr double kFramesPerSecond = 15.0;
// The speaking rate a speed of 1 asks for, in UTF-8 bytes per second.
constexpr double kUnitSpeakingRate = 15.193;

mx::array Scalar(float value) { return mx::array(value); }

mx::array Silu(const mx::array &x) { return mx::multiply(x, mx::sigmoid(x)); }

mx::array Columns(const mx::array &x, int begin, int end) {
  return mx::slice(x, {0, begin}, {x.shape(0), end});
}

mx::array Rows(const mx::array &x, int begin, int end) {
  mx::Shape start(x.ndim(), 0);
  mx::Shape stop = x.shape();
  start[0] = begin;
  stop[0] = end;
  return mx::slice(x, start, stop);
}

const char *kGatedDeltaSource = R"(
        auto n = thread_position_in_grid.z;
        auto b_idx = n / Hv;
        auto hv_idx = n % Hv;
        auto hk_idx = hv_idx / (Hv / Hk);
        constexpr int n_per_t = Dk / 32;

        auto q_ = q + b_idx * T * Hk * Dk + hk_idx * Dk;
        auto k_ = k + b_idx * T * Hk * Dk + hk_idx * Dk;

        auto v_ = v + b_idx * T * Hv * Dv + hv_idx * Dv;
        y += b_idx * T * Hv * Dv + hv_idx * Dv;

        auto dk_idx = thread_position_in_threadgroup.x;
        auto dv_idx = thread_position_in_grid.y;

        auto i_state = state_in + (n * Dv + dv_idx) * Dk;
        auto o_state = state_out + (n * Dv + dv_idx) * Dk;

        float state[n_per_t];
        for (int i = 0; i < n_per_t; ++i) {
          auto s_idx = n_per_t * dk_idx + i;
          state[i] = static_cast<float>(i_state[s_idx]);
        }

        auto g_ = g + b_idx * T * Hv;
        auto beta_ = beta + b_idx * T * Hv;

        for (int t = 0; t < T; ++t) {
          float kv_mem = 0.0f;
          for (int i = 0; i < n_per_t; ++i) {
            auto s_idx = n_per_t * dk_idx + i;
            state[i] = state[i] * g_[hv_idx];
            kv_mem += state[i] * k_[s_idx];
          }
          kv_mem = simd_sum(kv_mem);

          auto delta = (v_[dv_idx] - kv_mem) * beta_[hv_idx];

          float out = 0.0f;
          for (int i = 0; i < n_per_t; ++i) {
            auto s_idx = n_per_t * dk_idx + i;
            state[i] = state[i] + k_[s_idx] * delta;
            out += state[i] * q_[s_idx];
          }
          out = simd_sum(out);
          if (thread_index_in_simdgroup == 0) {
            y[dv_idx] = static_cast<InT>(out);
          }
          q_ += Hk * Dk;
          k_ += Hk * Dk;
          v_ += Hv * Dv;
          y += Hv * Dv;
          g_ += Hv;
          beta_ += Hv;
        }
        for (int i = 0; i < n_per_t; ++i) {
          auto s_idx = n_per_t * dk_idx + i;
          o_state[s_idx] = static_cast<StT>(state[i]);
        }
)";

std::vector<mx::array> GatedDeltaStepGraph(const std::vector<mx::array> &in) {
  const mx::array &queries = in[0];
  const mx::array &keys = in[1];
  const mx::array &values = in[2];
  const mx::array &gates = in[3];
  const mx::array &beta = in[4];
  mx::array state = mx::multiply(in[5], mx::expand_dims(gates, {-1, -2}));
  const mx::array memory =
      mx::sum(mx::multiply(state, mx::expand_dims(keys, -2)), -1);
  const mx::array delta = mx::multiply(mx::subtract(values, memory),
                                       mx::expand_dims(beta, -1));
  state = mx::add(state, mx::multiply(mx::expand_dims(keys, -2),
                                      mx::expand_dims(delta, -1)));
  const mx::array output =
      mx::sum(mx::multiply(state, mx::expand_dims(queries, -2)), -1);
  return {output, state};
}

// y [B, T, Hv, Dv] and the final state of the gated delta rule over q, k
// [B, T, Hk, Dk], v [B, T, Hv, Dv], gates and beta [B, T, Hv] from state
// [B, Hv, Dv, Dk], all float32.
std::pair<mx::array, mx::array>
GatedDelta(const mx::array &queries, const mx::array &keys,
           const mx::array &values, const mx::array &gates,
           const mx::array &beta, const mx::array &state) {
  const int batch = keys.shape(0);
  const int rows = keys.shape(1);
  const int key_heads = keys.shape(2);
  const int key_size = keys.shape(3);
  const int value_heads = values.shape(2);
  const int value_size = values.shape(3);
  if (rows == 1 || key_size % 32 != 0) {
    static const auto step = mx::compile(GatedDeltaStepGraph);
    std::vector<mx::array> outputs;
    mx::array current = state;
    for (int row = 0; row < rows; ++row) {
      const auto at = [&](const mx::array &x) {
        mx::Shape start(x.ndim(), 0);
        mx::Shape stop = x.shape();
        start[1] = row;
        stop[1] = row + 1;
        return mx::squeeze(mx::slice(x, start, stop), 1);
      };
      const std::vector<mx::array> stepped = step(
          {at(queries), at(keys), at(values), at(gates), at(beta), current});
      outputs.push_back(stepped[0]);
      current = stepped[1];
    }
    return {mx::stack(outputs, 1), current};
  } else {
  }
  static const auto kernel = mx::fast::metal_kernel(
      "zonos3_gated_delta_step",
      {"q", "k", "v", "g", "beta", "state_in", "T"}, {"y", "state_out"},
      kGatedDeltaSource);
  const std::vector<mx::array> outputs = kernel(
      {queries, keys, values, gates, beta, state, mx::array(rows)},
      {{batch, rows, value_heads, value_size}, state.shape()},
      {queries.dtype(), state.dtype()}, {32, value_size, batch * value_heads},
      {32, 4, 1},
      {{"InT", queries.dtype()},
       {"StT", state.dtype()},
       {"Dk", key_size},
       {"Dv", value_size},
       {"Hk", key_heads},
       {"Hv", value_heads}},
      std::nullopt, false, {});
  return {outputs[0], outputs[1]};
}

ModelConfig ParseConfig(const nlohmann::json &backbone) {
  ModelConfig config;
  config.layer_count = backbone.at("n_layers").get<int>();
  config.max_rows = backbone.at("max_seqlen").get<int>();
  config.hidden_size = backbone.at("dim").get<int>();
  config.head_dimension = backbone.at("head_dim").get<int>();
  config.head_count = backbone.at("n_heads").get<int>();
  config.key_value_head_count = backbone.at("n_kv_heads").get<int>();
  config.expert_count = backbone.at("n_experts").get<int>();
  config.experts_per_row = backbone.at("moe_topk").get<int>();
  config.codebook_count = backbone.at("n_codebooks").get<int>();
  config.codebook_size = backbone.at("codebook_size").get<int>();
  config.audio_vocabulary_size = backbone.at("audio_vocab").get<int>();
  config.end_of_audio_id = backbone.at("eoa_id").get<int>();
  config.audio_pad_id = backbone.at("audio_pad_id").get<int>();
  config.text_pad_id = backbone.at("text_vocab").get<int>();
  config.delta_head_count = backbone.at("gdn_heads").get<int>();
  config.delta_head_dimension = backbone.at("gdn_head_dim").get<int>();
  config.delta_kernel_size = backbone.at("gdn_kernel").get<int>();
  config.speaker_embedding_size = backbone.at("speaker_embedding_dim").get<int>();
  config.condition_bucket_counts =
      backbone.at("condition_bucket_counts").get<std::vector<int>>();
  config.condition_names =
      backbone.at("condition_names").get<std::vector<std::string>>();
  for (const auto &[name, edges] : backbone.at("condition_edges").items()) {
    config.condition_edges[name] = edges.get<std::vector<double>>();
  }
  config.rope_theta = backbone.at("rope_theta").get<float>();
  config.norm_epsilon = backbone.at("norm_eps").get<float>();
  config.query_key_norm_epsilon = backbone.at("qk_eps").get<float>();
  config.logit_softcap = backbone.at("loss_softcap").get<float>();
  config.gated_delta_period = backbone.at("gdn_period").get<int>();
  config.silence_frame = backbone.at("silence_frame").get<std::vector<int32_t>>();
  if (backbone.value("text_extra_tokens", 0) != 0 ||
      !backbone.value("span_end_id", nlohmann::json()).is_null()) {
    throw std::runtime_error(
        "this backbone uses extra text tokens or span ends, which the "
        "runtime does not support");
  } else if (config.audio_vocabulary_size != config.codebook_size + 2 ||
             config.end_of_audio_id != config.codebook_size ||
             config.audio_pad_id != config.codebook_size + 1 ||
             static_cast<int>(config.silence_frame.size()) !=
                 config.codebook_count ||
             config.condition_names.size() !=
                 config.condition_bucket_counts.size() ||
             backbone.at("condition_count").get<size_t>() !=
                 config.condition_names.size() ||
             config.gated_delta_period < 1) {
    throw std::runtime_error("inconsistent ZONOS3 backbone config");
  } else {
  }
  return config;
}

std::string Sha256(const std::vector<float> &samples) {
  std::string digest(CC_SHA256_DIGEST_LENGTH, '\0');
  CC_SHA256(samples.data(), static_cast<CC_LONG>(samples.size() * sizeof(float)),
            reinterpret_cast<unsigned char *>(digest.data()));
  return digest;
}

std::vector<float> HostFloats(const mx::array &array) {
  const mx::array as_float = mx::astype(array, mx::float32);
  mx::eval(as_float);
  const float *data = as_float.data<float>();
  return std::vector<float>(data, data + as_float.size());
}

// The reference's output limiter: gain, then a tanh knee from -1 dBFS.
std::vector<float> OutputAudio(const mx::array &audio, float gain) {
  const double knee = std::pow(10.0, -1.0 / 20.0);
  const mx::array scaled = mx::multiply(mx::astype(audio, mx::float32),
                                        Scalar(gain));
  const mx::array magnitude = mx::abs(scaled);
  const mx::array limited = mx::add(
      Scalar(static_cast<float>(knee)),
      mx::multiply(Scalar(static_cast<float>(1.0 - knee)),
                   mx::tanh(mx::divide(
                       mx::subtract(magnitude, Scalar(static_cast<float>(knee))),
                       Scalar(static_cast<float>(1.0 - knee))))));
  return HostFloats(mx::where(
      mx::greater(magnitude, Scalar(static_cast<float>(knee))),
      mx::multiply(mx::sign(scaled), limited), scaled));
}

// The aligned frames [frames - start, codebooks] of delayed rows [rows,
// codebooks]: codebook c of frame f sits in row f + c.
mx::array AlignedCodes(const mx::array &rows, int codebook_count,
                       std::optional<int> end_of_audio_frame, int start) {
  int length = std::max(0, rows.shape(0) - codebook_count + 1);
  if (end_of_audio_frame.has_value()) {
    length = std::min(length, *end_of_audio_frame);
  } else {
  }
  if (length <= start) {
    return mx::zeros({0, codebook_count}, mx::int32);
  } else {
  }
  const mx::array kept = Rows(rows, start, length + codebook_count - 1);
  std::vector<mx::array> columns;
  for (int codebook = 0; codebook < codebook_count; ++codebook) {
    columns.push_back(mx::reshape(
        mx::slice(kept, {codebook, codebook},
                  {codebook + length - start, codebook + 1}),
        {length - start}));
  }
  return mx::stack(columns, -1);
}

} // namespace

Zonos3Model::Zonos3Model(const std::filesystem::path &model_directory)
    : config_(ParseConfig(
          omni_common::ReadJson(model_directory / "config.json").at("backbone"))),
      weights_(model_directory, "backbone.params."),
      text_(model_directory,
            omni_common::ReadJson(model_directory / "config.json")
                .at("backbone")
                .at("text_tokenizer_spec"),
            config_.text_pad_id),
      codec_(model_directory / "codec"), model_directory_(model_directory) {
  const ModelConfig &c = config_;
  const auto require = [&](bool condition, const std::string &what) {
    if (!condition) {
      throw std::runtime_error("ZONOS3 checkpoint does not fit its config: " +
                               what);
    } else {
    }
  };
  for (int codebook = 0; codebook < c.codebook_count; ++codebook) {
    const mx::array &table =
        weights_.Table("audio_embd." + std::to_string(codebook) + ".weight");
    require(table.shape(0) == c.audio_vocabulary_size &&
                table.shape(1) == c.hidden_size,
            "audio embeddings");
  }
  require(weights_.Table("text_embd.weight").shape(0) == c.text_pad_id + 1,
          "text embeddings");
  require(weights_.Table("attnres_q.weight").shape(0) == 2 * c.layer_count + 1,
          "residual queries");
  for (size_t condition = 0; condition < c.condition_bucket_counts.size();
       ++condition) {
    require(weights_.Table("cond_embd." + std::to_string(condition) + ".weight")
                    .shape(0) == c.condition_bucket_counts[condition] + 1,
            "condition embeddings");
  }
  for (int layer = 0; layer < c.layer_count; ++layer) {
    const std::string prefix = "blk." + std::to_string(layer) + ".";
    require(weights_.HasMatrix(c.IsGatedDelta(layer) ? prefix + "gdn_in.weight"
                                                     : prefix + "attn_q.weight"),
            "layer " + std::to_string(layer) + " mixer");
    require(weights_.HasMatrix(prefix + "ffn_up.weight") ||
                weights_.HasMatrix(prefix + "ffn_gate_exps.weight"),
            "layer " + std::to_string(layer) + " feed-forward");
  }
  require(codec_.codebook_count() == c.codebook_count &&
              codec_.codebook_size() == c.codebook_size,
          "codec codebooks");
  const std::filesystem::path voices = model_directory / "voices.json";
  if (std::filesystem::is_regular_file(voices)) {
    const nlohmann::json voice_table = omni_common::ReadJson(voices);
    for (const auto &[name, file] : voice_table.items()) {
      if (!std::filesystem::is_regular_file(model_directory / "voices" /
                                            file.get<std::string>())) {
        throw std::runtime_error("voices.json names a missing voice " + name);
      } else {
      }
      voice_files_[name] = file.get<std::string>();
    }
  } else {
  }
  const std::filesystem::path speaker = model_directory / "speaker_encoder";
  if (std::filesystem::is_directory(speaker)) {
    speaker_encoder_.emplace(speaker);
    require(speaker_encoder_->embedding_size() == c.speaker_embedding_size,
            "speaker embedding size");
  } else {
  }
}

std::vector<std::string> Zonos3Model::VoiceNames() const {
  std::vector<std::string> names;
  for (const auto &[name, file] : voice_files_) {
    names.push_back(name);
  }
  return names;
}

bool Zonos3Model::HasVoice(const std::string &name) const {
  return voice_files_.count(name) > 0;
}

std::vector<float> Zonos3Model::VoiceEmbedding(const std::string &name) const {
  const auto found = voice_files_.find(name);
  if (found == voice_files_.end()) {
    throw std::invalid_argument("unknown voice '" + name + "'");
  } else {
  }
  const auto archive = npz::Load(model_directory_ / "voices" / found->second);
  const auto embedding = archive.find("embedding");
  if (embedding == archive.end()) {
    throw std::runtime_error(found->second + " has no embedding array");
  } else {
  }
  std::vector<float> values = HostFloats(mx::reshape(embedding->second, {-1}));
  if (static_cast<int>(values.size()) != config_.speaker_embedding_size) {
    throw std::runtime_error(found->second + " is not a speaker embedding");
  } else {
  }
  return values;
}

std::vector<float>
Zonos3Model::ReferenceEmbedding(const std::vector<float> &samples) {
  if (!speaker_encoder_.has_value()) {
    throw std::invalid_argument("this checkpoint has no speaker encoder");
  } else {
  }
  for (const float sample : samples) {
    if (!std::isfinite(sample)) {
      throw std::invalid_argument("reference audio must be finite");
    } else {
    }
  }
  const std::vector<float> kept(
      samples.begin(),
      samples.begin() + std::min<size_t>(samples.size(),
                                         kMaxReferenceSeconds * kSampleRate));
  const std::string key = Sha256(kept);
  for (auto entry = reference_embeddings_.begin();
       entry != reference_embeddings_.end(); ++entry) {
    if (entry->first == key) {
      reference_embeddings_.splice(reference_embeddings_.begin(),
                                   reference_embeddings_, entry);
      return reference_embeddings_.front().second;
    } else {
    }
  }
  std::vector<float> embedding = speaker_encoder_->Embed(kept);
  reference_embeddings_.emplace_front(key, embedding);
  if (reference_embeddings_.size() > kReferenceCacheEntries) {
    reference_embeddings_.pop_back();
  } else {
  }
  return embedding;
}

std::vector<int> Zonos3Model::ConditionIds(
    const std::optional<std::map<std::string, nlohmann::json>> &buckets,
    const std::optional<std::map<std::string, nlohmann::json>> &values) const {
  const ModelConfig &c = config_;
  if (buckets.has_value() && values.has_value()) {
    throw std::invalid_argument(
        "condition_buckets and condition_values are mutually exclusive");
  } else if (!buckets.has_value() && !values.has_value()) {
    return c.condition_bucket_counts;
  } else {
  }
  const auto &supplied = buckets.has_value() ? *buckets : *values;
  for (const auto &[name, value] : supplied) {
    if (std::find(c.condition_names.begin(), c.condition_names.end(), name) ==
        c.condition_names.end()) {
      throw std::invalid_argument("unknown condition " + name);
    } else {
    }
  }
  std::vector<int> ids;
  for (size_t index = 0; index < c.condition_names.size(); ++index) {
    const std::string &name = c.condition_names[index];
    const int count = c.condition_bucket_counts[index];
    const auto found = supplied.find(name);
    if (found == supplied.end() || found->second.is_null()) {
      ids.push_back(count);
    } else if (!found->second.is_number() || found->second.is_boolean()) {
      throw std::invalid_argument("condition " + name + " must be a number");
    } else if (values.has_value()) {
      const double value = found->second.get<double>();
      if (!std::isfinite(value)) {
        throw std::invalid_argument("condition values must be finite");
      } else {
      }
      const std::vector<double> &edges = c.condition_edges.at(name);
      ids.push_back(static_cast<int>(
          std::upper_bound(edges.begin(), edges.end(), value) - edges.begin()));
    } else {
      const double value = found->second.get<double>();
      if (std::floor(value) != value || value < 0 || value > count) {
        throw std::invalid_argument(name + " bucket must be in [0, " +
                                    std::to_string(count) + "]");
      } else {
      }
      ids.push_back(static_cast<int>(value));
    }
  }
  return ids;
}

std::vector<int> Zonos3Model::SpeedConditionIds(double speed) const {
  std::map<std::string, nlohmann::json> values;
  values["speaking_rate"] = kUnitSpeakingRate * speed;
  return ConditionIds(std::nullopt, values);
}

Prompt Zonos3Model::BuildPrompt(const SynthesisRequest &request) const {
  const ModelConfig &c = config_;
  const std::string text =
      request.normalize_text
          ? TextFrontend::Normalize(request.text, request.language)
          : request.text;
  const std::vector<int> tokens = text_.Encode(text);
  const int offset = request.speaker_embedding.has_value() ? 1 : 0;
  const int condition_count = static_cast<int>(c.condition_bucket_counts.size());
  Prompt prompt;
  const std::vector<int32_t> pad_row = [&] {
    std::vector<int32_t> row(c.codebook_count + 1, c.audio_pad_id);
    row.back() = c.text_pad_id;
    return row;
  }();
  for (int row = 0; row < offset + condition_count; ++row) {
    prompt.rows.push_back(pad_row);
  }
  for (const int token : tokens) {
    std::vector<int32_t> row = pad_row;
    row.back() = token;
    prompt.rows.push_back(row);
  }
  if (offset == 1) {
    prompt.speaker_row = 0;
  } else {
  }
  for (int condition = 0; condition < condition_count; ++condition) {
    prompt.condition_rows.emplace_back(offset + condition,
                                       request.condition_buckets.at(condition));
  }
  // The silence frame in the delay pattern without the flush: codebook c of
  // frame f in row f + c, pad elsewhere.
  const int silence_frames = request.silence_prefill_frames;
  for (int row = 0; row < silence_frames; ++row) {
    std::vector<int32_t> values = pad_row;
    for (int codebook = 0; codebook < c.codebook_count; ++codebook) {
      if (row >= codebook) {
        values[codebook] = c.silence_frame[codebook];
      } else {
      }
    }
    prompt.rows.push_back(values);
  }
  prompt.silence_row_count = silence_frames;
  if (c.max_rows - static_cast<int>(prompt.rows.size()) <= 0 ||
      request.max_rows <= 0) {
    throw std::invalid_argument(
        "max_new_tokens must be positive and the prompt must leave context "
        "for audio");
  } else {
  }
  return prompt;
}

mx::array Zonos3Model::GlobalAttention(const mx::array &x, int layer,
                                       BackboneState &state) const {
  const ModelConfig &c = config_;
  const std::string prefix = "blk." + std::to_string(layer) + ".";
  const int rows = x.shape(0);
  const int heads = c.head_count;
  const int key_value_heads = c.key_value_head_count;
  const int head_size = c.head_dimension;
  const int low = state.position;
  const mx::array gate = mx::sigmoid(weights_.Linear(x, prefix + "attn_gate.weight"));
  mx::array queries = mx::reshape(weights_.Linear(x, prefix + "attn_q.weight"),
                                  {rows, heads, head_size});
  mx::array keys = mx::reshape(weights_.Linear(x, prefix + "attn_k.weight"),
                               {rows, key_value_heads, head_size});
  const mx::array values = mx::reshape(
      weights_.Linear(x, prefix + "attn_v.weight"),
      {rows, key_value_heads, head_size});
  queries = mx::multiply(
      mx::fast::rms_norm(queries, std::nullopt, c.query_key_norm_epsilon),
      mx::expand_dims(weights_.Table(prefix + "attn_temp"), 1));
  keys = mx::fast::rms_norm(keys, std::nullopt, c.query_key_norm_epsilon);
  queries = mx::fast::rope(mx::transpose(queries, {1, 0, 2}), head_size, true,
                           c.rope_theta, 1.0f, low);
  keys = mx::fast::rope(mx::transpose(keys, {1, 0, 2}), head_size, true,
                        c.rope_theta, 1.0f, low);
  const mx::array values_by_head = mx::transpose(values, {1, 0, 2});
  std::optional<mx::array> &cached_keys = state.keys[layer];
  std::optional<mx::array> &cached_values = state.values[layer];
  const int need = low + rows;
  if (!cached_keys.has_value() || cached_keys->shape(1) < need) {
    const int capacity =
        (need + kKeyValueBlockRows - 1) / kKeyValueBlockRows * kKeyValueBlockRows;
    const int grow = capacity - (cached_keys.has_value() ? cached_keys->shape(1) : 0);
    const mx::array pad =
        mx::zeros({key_value_heads, grow, head_size}, mx::float32);
    cached_keys = cached_keys.has_value()
                      ? mx::concatenate({*cached_keys, pad}, 1)
                      : pad;
    cached_values = cached_values.has_value()
                        ? mx::concatenate({*cached_values, pad}, 1)
                        : mx::zeros({key_value_heads, grow, head_size},
                                    mx::float32);
  } else {
  }
  cached_keys = mx::slice_update(*cached_keys, keys, {0, low, 0},
                                 {key_value_heads, need, head_size});
  cached_values = mx::slice_update(*cached_values, values_by_head, {0, low, 0},
                                   {key_value_heads, need, head_size});
  const mx::array visible_keys =
      mx::slice(*cached_keys, {0, 0, 0}, {key_value_heads, need, head_size});
  const mx::array visible_values =
      mx::slice(*cached_values, {0, 0, 0}, {key_value_heads, need, head_size});
  const float scale =
      static_cast<float>(std::pow(static_cast<double>(head_size), -0.5));
  mx::array attended = mx::array(0.0f);
  if (rows > 1) {
    const mx::array mask = mx::less_equal(
        mx::expand_dims(mx::arange(need), 0),
        mx::expand_dims(mx::add(mx::array(low), mx::arange(rows)), 1));
    attended = mx::fast::scaled_dot_product_attention(
        mx::expand_dims(queries, 0), mx::expand_dims(visible_keys, 0),
        mx::expand_dims(visible_values, 0), scale, "array", mask);
  } else {
    attended = mx::fast::scaled_dot_product_attention(
        mx::expand_dims(queries, 0), mx::expand_dims(visible_keys, 0),
        mx::expand_dims(visible_values, 0), scale);
  }
  const mx::array output = mx::multiply(
      mx::transpose(mx::squeeze(attended, 0), {1, 0, 2}),
      mx::expand_dims(gate, 2));
  return weights_.Linear(mx::reshape(output, {rows, heads * head_size}),
                         prefix + "attn_output.weight");
}

mx::array Zonos3Model::GatedDeltaNet(const mx::array &x, int layer,
                                     BackboneState &state) const {
  const ModelConfig &c = config_;
  const std::string prefix = "blk." + std::to_string(layer) + ".";
  const int heads = c.delta_head_count;
  const int head_size = c.delta_head_dimension;
  const int kernel = c.delta_kernel_size;
  const int inner = heads * head_size;
  const int rows = x.shape(0);
  const mx::array projection = weights_.Linear(x, prefix + "gdn_in.weight");
  const mx::array query_key_value = Columns(projection, 0, 3 * inner);
  const mx::array output_gate = Columns(projection, 3 * inner, 4 * inner);
  const mx::array decay_logits =
      Columns(projection, 4 * inner, 4 * inner + heads);
  const mx::array beta_logits =
      Columns(projection, 4 * inner + heads, 4 * inner + 2 * heads);
  std::optional<mx::array> &history = state.convolution[layer];
  const mx::array previous =
      history.has_value() ? *history
                          : mx::zeros({3 * inner, kernel - 1}, mx::float32);
  const mx::array sequence =
      mx::concatenate({mx::transpose(previous), query_key_value}, 0);
  const mx::array &weight = weights_.Table(prefix + "gdn_conv.weight");
  const auto tap = [&](int index) {
    return mx::squeeze(Columns(weight, index, index + 1), 1);
  };
  mx::array convolved = mx::multiply(Rows(sequence, 0, rows), tap(0));
  for (int j = 1; j < kernel; ++j) {
    convolved = mx::add(convolved, mx::multiply(Rows(sequence, j, j + rows), tap(j)));
  }
  convolved = Silu(convolved);
  history = mx::transpose(
      Rows(sequence, sequence.shape(0) - (kernel - 1), sequence.shape(0)));
  const auto split = [&](int index) {
    return mx::reshape(Columns(convolved, index * inner, (index + 1) * inner),
                       {rows, heads, head_size});
  };
  const auto l2_normalize = [&](const mx::array &v) {
    return mx::multiply(
        v, mx::rsqrt(mx::add(mx::sum(mx::multiply(v, v), -1, true),
                             Scalar(kQueryKeyL2Epsilon))));
  };
  const mx::array queries = l2_normalize(split(0));
  const mx::array keys = l2_normalize(split(1));
  const mx::array values = split(2);
  const mx::array decay = mx::exp(mx::multiply(
      mx::negative(mx::exp(weights_.Table(prefix + "gdn_A_log"))),
      mx::logaddexp(
          mx::add(decay_logits, weights_.Table(prefix + "gdn_dt_bias")),
          mx::zeros_like(decay_logits))));
  const mx::array beta = mx::sigmoid(beta_logits);
  std::optional<mx::array> &recurrent = state.recurrent[layer];
  const mx::array initial =
      recurrent.has_value()
          ? *recurrent
          : mx::zeros({heads, head_size, head_size}, mx::float32);
  const auto [output, final_state] = GatedDelta(
      mx::expand_dims(
          mx::multiply(queries, Scalar(static_cast<float>(
                                    std::pow(static_cast<double>(head_size), -0.5)))),
          0),
      mx::expand_dims(keys, 0), mx::expand_dims(values, 0),
      mx::expand_dims(decay, 0), mx::expand_dims(beta, 0),
      mx::expand_dims(initial, 0));
  recurrent = mx::squeeze(final_state, 0);
  const mx::array normalized =
      mx::fast::rms_norm(mx::squeeze(output, 0),
                         weights_.Table(prefix + "gdn_norm.weight"),
                         kGatedDeltaNormEpsilon);
  const mx::array gated = mx::reshape(
      mx::multiply(normalized,
                   Silu(mx::reshape(output_gate, {rows, heads, head_size}))),
      {rows, inner});
  return weights_.Linear(gated, prefix + "attn_output.weight");
}

mx::array Zonos3Model::FeedForward(const mx::array &x, int layer) const {
  const ModelConfig &c = config_;
  const std::string prefix = "blk." + std::to_string(layer) + ".";
  if (weights_.HasMatrix(prefix + "ffn_up.weight")) {
    return weights_.Linear(
        mx::multiply(weights_.Linear(x, prefix + "ffn_up.weight"),
                     Silu(weights_.Linear(x, prefix + "ffn_gate.weight"))),
        prefix + "ffn_down.weight");
  } else {
  }
  const mx::array logits = weights_.Linear(
      mx::fast::rms_norm(x, weights_.Table(prefix + "router_norm.weight"),
                         c.norm_epsilon),
      prefix + "router.weight");
  const int top = c.experts_per_row;
  const mx::array choices = Columns(
      mx::argpartition(
          mx::subtract(weights_.Table(prefix + "router_bias"), logits), top - 1,
          -1),
      0, top);
  const mx::array expert_weights =
      mx::softmax(mx::take_along_axis(logits, choices, -1), std::vector<int>{-1});
  const mx::array latent = weights_.Linear(x, prefix + "ffn_latent_down.weight");
  const mx::array rows = mx::expand_dims(latent, {1, 2});
  const mx::array gate =
      weights_.GatherLinear(rows, prefix + "ffn_gate_exps.weight", choices);
  const mx::array up =
      weights_.GatherLinear(rows, prefix + "ffn_up_exps.weight", choices);
  const mx::array down = weights_.GatherLinear(
      mx::multiply(Silu(gate), up), prefix + "ffn_down_exps.weight", choices);
  const mx::array combined =
      mx::sum(mx::multiply(mx::squeeze(down, 2), mx::expand_dims(expert_weights, 2)),
              1);
  return weights_.Linear(combined, prefix + "ffn_latent_up.weight");
}

mx::array Zonos3Model::Forward(
    const mx::array &rows, BackboneState &state,
    const std::optional<std::pair<int, std::vector<float>>> &speaker_row,
    const std::vector<std::pair<int, int>> &condition_rows) const {
  const ModelConfig &c = config_;
  if (rows.ndim() != 2 || rows.shape(1) != c.codebook_count + 1 ||
      rows.shape(0) == 0) {
    throw std::invalid_argument("expected nonempty rows of the frame width");
  } else {
  }
  const int count = rows.shape(0);
  const int low = state.position;
  if (low + count > c.max_rows) {
    throw std::invalid_argument("rows exceed the model context");
  } else {
  }
  if (state.keys.empty()) {
    state.keys.resize(c.layer_count);
    state.values.resize(c.layer_count);
    state.recurrent.resize(c.layer_count);
    state.convolution.resize(c.layer_count);
  } else {
  }
  const mx::array ids = mx::astype(rows, mx::int32);
  const auto column = [&](int index) {
    return mx::reshape(mx::slice(ids, {0, index}, {count, index + 1}), {count});
  };
  mx::array x = mx::take(weights_.Table("audio_embd.0.weight"), column(0), 0);
  for (int codebook = 1; codebook < c.codebook_count; ++codebook) {
    x = mx::add(x, mx::take(weights_.Table("audio_embd." +
                                           std::to_string(codebook) + ".weight"),
                            column(codebook), 0));
  }
  x = mx::add(x, mx::take(weights_.Table("text_embd.weight"),
                          column(c.codebook_count), 0));
  const auto replace_row = [&](int row, const mx::array &value) {
    x = mx::slice_update(x, mx::expand_dims(value, 0), {row - low, 0},
                         {row - low + 1, c.hidden_size});
  };
  if (speaker_row.has_value() && low <= speaker_row->first &&
      speaker_row->first < low + count) {
    const mx::array embedding = mx::array(
        speaker_row->second.data(),
        {1, static_cast<int>(speaker_row->second.size())});
    const mx::array projected = mx::squeeze(
        weights_.Linear(embedding, "speaker_proj.weight"), 0);
    replace_row(speaker_row->first,
                mx::add(projected, weights_.Table("speaker_proj.bias")));
  } else {
  }
  // The reference assigns condition tables by the rows' sorted order.
  std::vector<std::pair<int, int>> sorted_conditions = condition_rows;
  std::sort(sorted_conditions.begin(), sorted_conditions.end());
  for (size_t table = 0; table < sorted_conditions.size(); ++table) {
    const auto [row, bucket] = sorted_conditions[table];
    if (low <= row && row < low + count) {
      const mx::array &embeddings =
          weights_.Table("cond_embd." + std::to_string(table) + ".weight");
      replace_row(row, mx::squeeze(Rows(embeddings, bucket, bucket + 1), 0));
    } else {
    }
  }
  x = mx::fast::rms_norm(x, std::nullopt, c.norm_epsilon);
  const mx::array &queries = weights_.Table("attnres_q.weight");
  std::vector<mx::array> bank;
  std::vector<mx::array> scores;
  // Each sublayer input is the softmax mix of every earlier output for its
  // own query; the first input is the embedding itself.
  const auto route = [&](const mx::array &source, int query) {
    bank.push_back(source);
    scores.push_back(
        mx::matmul(mx::fast::rms_norm(source, std::nullopt, c.norm_epsilon),
                   mx::transpose(queries)));
    if (bank.size() == 1) {
      return source;
    } else {
    }
    std::vector<mx::array> query_scores;
    for (const mx::array &score : scores) {
      query_scores.push_back(
          mx::reshape(Columns(score, query, query + 1), {count}));
    }
    const mx::array weights =
        mx::softmax(mx::stack(query_scores), std::vector<int>{0});
    return mx::sum(mx::multiply(mx::stack(bank), mx::expand_dims(weights, 2)), 0);
  };
  mx::array mixed = route(x, 0);
  for (int layer = 0; layer < c.layer_count; ++layer) {
    const std::string prefix = "blk." + std::to_string(layer) + ".";
    if (layer > 0) {
      mixed = route(x, 2 * layer);
    } else {
    }
    const mx::array attention_input = mx::fast::rms_norm(
        mixed, weights_.Table(prefix + "attn_norm.weight"), c.norm_epsilon);
    const mx::array attention = c.IsGatedDelta(layer)
                                    ? GatedDeltaNet(attention_input, layer, state)
                                    : GlobalAttention(attention_input, layer, state);
    mixed = route(attention, 2 * layer + 1);
    x = FeedForward(mx::fast::rms_norm(mixed,
                                       weights_.Table(prefix + "ffn_norm.weight"),
                                       c.norm_epsilon),
                    layer);
  }
  mixed = route(x, 2 * c.layer_count);
  const mx::array normalized = mx::fast::rms_norm(
      mixed, weights_.Table("output_norm.weight"), c.norm_epsilon);
  const mx::array logits =
      mx::reshape(weights_.Linear(normalized, "output.weight"),
                  {count, c.codebook_count, c.audio_vocabulary_size});
  state.position = low + count;
  return mx::multiply(Scalar(c.logit_softcap),
                      mx::tanh(mx::divide(logits, Scalar(c.logit_softcap))));
}

mx::array Zonos3Model::SampleRow(const mx::array &logits, int step,
                                 const mx::array &history,
                                 const Sampling &sampling,
                                 bool has_audio_prefix) const {
  const ModelConfig &c = config_;
  const int vocabulary = logits.shape(-1);
  const mx::array ids = mx::expand_dims(mx::arange(vocabulary), 0);
  const mx::array codebooks = mx::expand_dims(mx::arange(c.codebook_count), 1);
  mx::array invalid = mx::logical_or(
      mx::equal(ids, mx::array(c.audio_pad_id)),
      mx::logical_and(mx::greater_equal(ids, mx::array(c.codebook_size)),
                      mx::not_equal(ids, mx::array(c.end_of_audio_id))));
  invalid = mx::logical_or(
      invalid,
      mx::logical_and(
          mx::equal(ids, mx::array(c.end_of_audio_id)),
          mx::less(mx::array(step),
                   mx::add(mx::array(sampling.min_frames), codebooks))));
  const float negative_infinity = -std::numeric_limits<float>::infinity();
  mx::array x = mx::where(invalid, Scalar(negative_infinity),
                          mx::astype(logits, mx::float32));
  const int window = std::min(std::max(sampling.repetition_window, 0),
                              history.shape(0));
  if (window > 0 && sampling.repetition_penalty > 1 &&
      sampling.repetition_codebooks != 0) {
    const mx::array recent = mx::transpose(Columns(
        Rows(history, history.shape(0) - window, history.shape(0)), 0,
        c.codebook_count));
    mx::array valid = mx::logical_and(
        mx::greater_equal(recent, mx::array(0)),
        mx::less(recent, mx::array(c.codebook_size)));
    if (sampling.repetition_codebooks >= 0) {
      valid = mx::logical_and(
          valid, mx::less(codebooks, mx::array(sampling.repetition_codebooks)));
    } else {
    }
    const mx::array row_index = mx::broadcast_to(codebooks, recent.shape());
    const mx::array column_index =
        mx::clip(recent, mx::array(0), mx::array(vocabulary - 1));
    const mx::array counts = mx::scatter_add(
        mx::zeros(logits.shape(), mx::int32), {row_index, column_index},
        mx::expand_dims(mx::astype(valid, mx::int32), {-1, -2}), {0, 1});
    const mx::array repeated = mx::greater(counts, mx::array(0));
    const mx::array penalty =
        Scalar(static_cast<float>(sampling.repetition_penalty));
    x = mx::where(repeated,
                  mx::where(mx::greater(x, Scalar(0.0f)), mx::divide(x, penalty),
                            mx::multiply(x, penalty)),
                  x);
  } else {
  }
  mx::array token = mx::array(0);
  if (sampling.temperature <= 0) {
    token = mx::argmax(x, -1);
  } else {
    x = mx::divide(x, Scalar(static_cast<float>(sampling.temperature)));
    if (0 < sampling.top_k && sampling.top_k < vocabulary) {
      const mx::array threshold =
          mx::min(mx::topk(x, sampling.top_k, -1), -1, true);
      x = mx::where(mx::less(x, threshold), Scalar(negative_infinity), x);
    } else {
    }
    mx::array probabilities = mx::softmax(x, std::vector<int>{-1});
    if (0 < sampling.top_p && sampling.top_p < 1) {
      const mx::array order = mx::argsort(mx::negative(probabilities), -1);
      mx::array ranked = mx::take_along_axis(probabilities, order, -1);
      ranked = mx::where(
          mx::greater(mx::subtract(mx::cumsum(ranked, -1), ranked),
                      Scalar(static_cast<float>(sampling.top_p))),
          Scalar(0.0f), ranked);
      probabilities = mx::put_along_axis(mx::zeros_like(probabilities), order,
                                         ranked, -1);
      probabilities = mx::divide(
          probabilities,
          mx::maximum(mx::sum(probabilities, -1, true), Scalar(1e-8f)));
    } else {
    }
    if (sampling.min_p > 0) {
      probabilities = mx::where(
          mx::less(probabilities,
                   mx::multiply(Scalar(static_cast<float>(sampling.min_p)),
                                mx::max(probabilities, -1, true))),
          Scalar(0.0f), probabilities);
    } else {
    }
    const mx::array greedy = mx::equal(ids, mx::argmax(x, -1, true));
    probabilities = mx::where(
        mx::less_equal(mx::sum(probabilities, -1, true), Scalar(0.0f)), greedy,
        probabilities);
    const std::optional<mx::array> key =
        sampling.seed.has_value()
            ? std::optional<mx::array>(mx::random::key(*sampling.seed + step))
            : std::nullopt;
    token = mx::random::categorical(mx::log(probabilities), -1, key);
  }
  if (!has_audio_prefix) {
    token = mx::where(mx::greater(mx::arange(c.codebook_count), mx::array(step)),
                      mx::array(c.audio_pad_id), token);
  } else {
  }
  return mx::astype(token, mx::int32);
}

SynthesisResult Zonos3Model::Synthesize(
    const SynthesisRequest &request, const std::atomic<bool> &cancel,
    const std::function<void(const mx::array &)> &on_rows,
    const std::function<void(std::vector<float>)> &on_audio) const {
  const ModelConfig &c = config_;
  const Prompt prompt = BuildPrompt(request);
  std::vector<int32_t> flat;
  for (const auto &row : prompt.rows) {
    flat.insert(flat.end(), row.begin(), row.end());
  }
  const mx::array rows = mx::array(
      flat.data(), {static_cast<int>(prompt.rows.size()), c.codebook_count + 1});
  const int budget =
      std::min(request.max_rows, c.max_rows - static_cast<int>(prompt.rows.size()));
  const bool guided = request.speaker_guidance_scale != 1.0;
  if (guided && !request.speaker_embedding.has_value()) {
    throw std::invalid_argument(
        "speaker guidance needs a speaker embedding or reference audio");
  } else {
  }
  std::optional<std::pair<int, std::vector<float>>> speaker_row;
  if (prompt.speaker_row.has_value()) {
    speaker_row.emplace(*prompt.speaker_row, *request.speaker_embedding);
  } else {
  }
  BackboneState state;
  mx::array logits = Rows(Forward(rows, state, speaker_row, prompt.condition_rows),
                          rows.shape(0) - 1, rows.shape(0));
  logits = mx::squeeze(logits, 0);
  BackboneState twin_state;
  std::optional<mx::array> twin_logits;
  if (guided) {
    std::vector<std::pair<int, int>> unknown_conditions;
    for (size_t condition = 0; condition < c.condition_bucket_counts.size();
         ++condition) {
      unknown_conditions.emplace_back(static_cast<int>(condition),
                                      c.condition_bucket_counts[condition]);
    }
    const mx::array twin_rows = Rows(rows, 1, rows.shape(0));
    twin_logits = mx::squeeze(
        Rows(Forward(twin_rows, twin_state, std::nullopt, unknown_conditions),
             twin_rows.shape(0) - 1, twin_rows.shape(0)),
        0);
  } else {
  }
  std::vector<mx::array> evaluated = {logits};
  const auto state_arrays = [](const BackboneState &s,
                               std::vector<mx::array> &out) {
    for (const auto *arrays : {&s.keys, &s.values, &s.recurrent, &s.convolution}) {
      for (const auto &array : *arrays) {
        if (array.has_value()) {
          out.push_back(*array);
        } else {
        }
      }
    }
  };
  state_arrays(state, evaluated);
  mx::eval(evaluated);

  const int silence_rows = prompt.silence_row_count;
  mx::array history = mx::zeros({silence_rows + budget, c.codebook_count}, mx::int32);
  if (silence_rows > 0) {
    history = mx::slice_update(
        history,
        Columns(Rows(rows, rows.shape(0) - silence_rows, rows.shape(0)), 0,
                c.codebook_count),
        {0, 0}, {silence_rows, c.codebook_count});
  } else {
  }
  const float gain = codec_.OutputGain(request.headroom_db);
  std::optional<CodecStream> codec_stream;
  if (request.stream) {
    codec_stream.emplace(codec_);
  } else {
  }
  // Python's round(): halves go to the even neighbor, as nearbyint does in
  // the default rounding mode.
  const int interval = std::max(
      1, static_cast<int>(std::nearbyint(request.streaming_interval_seconds *
                                         kFramesPerSecond)));
  std::optional<int> end_of_audio_frame;
  int decoded = 0;
  int step = -1;
  SynthesisResult result;
  result.prompt_row_count = static_cast<int>(prompt.rows.size());
  result.finish_reason = qwen3_asr::FinishReason::kLength;
  for (step = 0; step < budget; ++step) {
    if (cancel.load()) {
      throw qwen3_asr::TranscriptionCancelled();
    } else {
    }
    const int t = silence_rows + step;
    const mx::array step_logits =
        guided ? mx::add(*twin_logits,
                         mx::multiply(Scalar(static_cast<float>(
                                          request.speaker_guidance_scale)),
                                      mx::subtract(logits, *twin_logits)))
               : logits;
    const mx::array token = SampleRow(step_logits, step, Rows(history, 0, t),
                                      request.sampling, silence_rows > 0);
    history = mx::slice_update(history, mx::expand_dims(token, 0), {t, 0},
                               {t + 1, c.codebook_count});
    const mx::array last_stop = mx::max(
        mx::where(mx::equal(token, mx::array(c.end_of_audio_id)),
                  mx::arange(c.codebook_count), mx::array(-1)));
    std::vector<mx::array> step_arrays = {history, last_stop, token};
    state_arrays(state, step_arrays);
    mx::eval(step_arrays);
    on_rows(token);
    const int stop_column = last_stop.item<int>();
    if (stop_column >= 0) {
      const int aligned_stop = std::max(0, t - stop_column);
      end_of_audio_frame = end_of_audio_frame.has_value()
                               ? std::min(*end_of_audio_frame, aligned_stop)
                               : aligned_stop;
    } else {
    }
    int ready = std::max(0, t + 1 - c.codebook_count + 1);
    if (end_of_audio_frame.has_value()) {
      ready = std::min(ready, *end_of_audio_frame);
    } else {
    }
    if (codec_stream.has_value() && ready - decoded >= interval) {
      const mx::array codes = AlignedCodes(Rows(history, 0, t + 1),
                                           c.codebook_count, end_of_audio_frame,
                                           decoded);
      const std::vector<float> audio = codec_stream->Decode(codes);
      decoded = ready;
      if (!audio.empty()) {
        on_audio(OutputAudio(mx::array(audio.data(),
                                       {static_cast<int>(audio.size())}),
                             gain));
      } else {
      }
    } else {
    }
    if (end_of_audio_frame.has_value() &&
        t >= *end_of_audio_frame + c.codebook_count - 1) {
      result.finish_reason = qwen3_asr::FinishReason::kStop;
      break;
    } else {
    }
    if (step + 1 < budget) {
      const mx::array next_row = mx::expand_dims(
          mx::concatenate({token, mx::array({c.text_pad_id})}), 0);
      logits = mx::squeeze(Forward(next_row, state, std::nullopt, {}), 0);
      if (guided) {
        twin_logits =
            mx::squeeze(Forward(next_row, twin_state, std::nullopt, {}), 0);
      } else {
      }
    } else {
    }
  }
  const int generated = std::min(step + 1, budget);
  result.generated_row_count = generated;
  const mx::array codes = AlignedCodes(Rows(history, 0, silence_rows + generated),
                                       c.codebook_count, end_of_audio_frame, 0);
  result.frame_count = codes.shape(0);
  if (codec_stream.has_value()) {
    std::vector<float> tail;
    if (codes.shape(0) > decoded) {
      tail = codec_stream->Decode(Rows(codes, decoded, codes.shape(0)));
    } else {
    }
    const std::vector<float> flushed = codec_stream->Flush();
    tail.insert(tail.end(), flushed.begin(), flushed.end());
    on_audio(OutputAudio(mx::array(tail.data(), {static_cast<int>(tail.size())}),
                         gain));
  } else {
    on_audio(OutputAudio(codec_.Decode(codes), gain));
  }
  return result;
}

} // namespace zonos3
