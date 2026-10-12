// SPDX-License-Identifier: Apache-2.0
// Sortformer v2.1 streaming speaker diarization on MLX, as Voxt's Swift
// MLXAudioVAD computes it (feed + AOSC speaker-cache compression).
#pragma once

#include <filesystem>
#include <string>
#include <unordered_map>
#include <vector>

#include "mlx/mlx.h"
#include "sortformer_features.h"

namespace sortformer {

struct FastConformerConfig {
  int hidden_size = 512;
  int num_hidden_layers = 18;
  int num_attention_heads = 8;
  int intermediate_size = 2048;
  int num_mel_bins = 80;
  int conv_kernel_size = 9;
  int subsampling_factor = 8;
  int subsampling_conv_channels = 256;
  int subsampling_conv_kernel_size = 3;
  int subsampling_conv_stride = 2;
  bool attention_bias = true;
  bool scale_input = true;
};

struct TransformerConfig {
  int d_model = 192;
  int encoder_layers = 18;
  int encoder_attention_heads = 8;
  int encoder_ffn_dim = 768;
  float layer_norm_eps = 1e-5f;
  int max_source_positions = 1500;
  bool k_proj_bias = false;
};

struct ModulesConfig {
  int num_speakers = 4;
  int fc_d_model = 512;
  int tf_d_model = 192;
  int subsampling_factor = 8;
  int chunk_len = 188;
  int fifo_len = 0;
  int spkcache_len = 188;
  int spkcache_update_period = 188;
  int chunk_left_context = 1;
  int chunk_right_context = 1;
  int spkcache_sil_frames_per_spk = 5;
  float pred_score_threshold = 1e-6f;
  int max_index = 10000;
  float scores_boost_latest = 0.5f;
  float sil_threshold = 0.1f;
  float strong_boost_rate = 0.3f;
  float weak_boost_rate = 0.7f;
  float min_pos_scores_rate = 0.5f;
  bool use_aosc = false;
};

// config.json, with the Swift SortformerConfig defaults for absent keys.
struct Config {
  int num_speakers = 4;
  FastConformerConfig fc_encoder;
  TransformerConfig tf_encoder;
  ModulesConfig modules;
  ProcessorConfig processor;
};

// One stream's state between feeds (Swift StreamingState). The arrays start as
// empty float32, so after the first feed the encoder runs in float32 even for a
// float16 checkpoint, as in Swift.
struct StreamingState {
  mlx::core::array spkcache;               // (1, cache_frames, emb_dim)
  mlx::core::array spkcache_preds;         // (1, cache_frames, speakers)
  mlx::core::array fifo;                   // (1, fifo_frames, emb_dim)
  mlx::core::array fifo_preds;             // (1, fifo_frames, speakers)
  int frames_processed = 0;                // diarization frames emitted so far
  mlx::core::array mean_silence_embedding; // (1, emb_dim)
  mlx::core::array silence_frame_count;    // (1,)

  int spkcache_length() const { return spkcache.shape(1); }
  int fifo_length() const { return fifo.shape(1); }
};

struct Segment {
  float start = 0.0f;
  float end = 0.0f;
  int speaker = 0;
};

// Swift feed arguments; the defaults are the values Voxt passes.
struct FeedOptions {
  float threshold = 0.5f;
  float min_duration = 0.0f;
  float merge_gap = 0.18f;
  int spkcache_max = 188;
  int fifo_max = 188;
};

struct FeedResult {
  // Row-major (frames, speakers) probabilities of this feed's frames.
  std::vector<float> probabilities;
  int frame_count = 0;
  int speaker_count = 0;
  // Segments in seconds from the start of the stream (framesProcessed before
  // this feed times the frame duration), sorted by start.
  std::vector<Segment> segments;
};

class SortformerModel {
public:
  // Loads config.json and *.safetensors; every checkpoint key must be a model
  // parameter and every parameter must be present (num_batches_tracked is
  // dropped, hidden_to_spks is loaded but unused, as in Swift).
  explicit SortformerModel(const std::filesystem::path &model_directory);

  const Config &config() const { return config_; }
  const FeatureExtractor &features() const { return features_; }
  // Seconds per diarization frame: hop_length * subsampling / sample rate.
  float frame_duration() const;

  StreamingState InitStreamingState() const;
  // Diarizes one chunk of 16 kHz mono samples and advances state.
  FeedResult Feed(const std::vector<float> &samples, StreamingState &state,
                  const FeedOptions &options = {}) const;

private:
  const mlx::core::array &Weight(const std::string &name) const;
  mlx::core::array Linear(const mlx::core::array &x,
                          const std::string &prefix) const;
  mlx::core::array LayerNorm(const mlx::core::array &x,
                             const std::string &prefix, float eps) const;
  mlx::core::array PreEncode(const mlx::core::array &features) const;
  mlx::core::array
  RelativePositionAttention(const mlx::core::array &x,
                            const mlx::core::array &position_embedding,
                            const std::string &prefix) const;
  mlx::core::array ConformerConvolution(const mlx::core::array &x,
                                        const std::string &prefix) const;
  mlx::core::array ConformerLayer(const mlx::core::array &x,
                                  const mlx::core::array &position_embedding,
                                  int layer) const;
  mlx::core::array FastConformer(const mlx::core::array &embeddings) const;
  mlx::core::array TransformerLayer(const mlx::core::array &x,
                                    const mlx::core::array &mask,
                                    int layer) const;
  mlx::core::array
  SpeakerProbabilities(const mlx::core::array &embeddings) const;

  Config config_;
  FeatureExtractor features_;
  std::unordered_map<std::string, mlx::core::array> weights_;
};

// Per-speaker segments of frames with probability > threshold, as Swift
// predsToSegments: probabilities is row-major (frame_count, speaker_count).
std::vector<Segment> ProbabilitiesToSegments(
    const std::vector<float> &probabilities, int frame_count, int speaker_count,
    float frame_duration, float threshold, float min_duration, float merge_gap);

// Swift maybeCompressState: moves frames beyond fifo_max from the FIFO to the
// speaker cache and compresses the cache with AOSC when it exceeds
// spkcache_max.
void MaybeCompressState(StreamingState &state, int spkcache_max, int fifo_max,
                        const ModulesConfig &modules);

} // namespace sortformer
