// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <atomic>
#include <cstddef>
#include <list>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "common/layers.h"

namespace fishaudio_drama3 {

inline constexpr size_t kExpertCacheBytes = 2147483648ULL;

mlx::core::array Swiglu(const mlx::core::array &gate,
                        const mlx::core::array &x);
omni_common::WeightMap
LoadDrama3Weights(const std::filesystem::path &model_directory,
                  bool offload_experts);

bool IsRoutedExpertWeight(const std::string &name);

// Serial use only. Callers must evaluate consumers before releasing a lease.
class ExpertCache {
public:
  using ExpertWeights = omni_common::WeightMap;
  explicit ExpertCache(const std::filesystem::path &model_directory,
                       size_t capacity_bytes = kExpertCacheBytes);
  ExpertCache(const ExpertCache &) = delete;
  ExpertCache &operator=(const ExpertCache &) = delete;
  std::shared_ptr<const ExpertWeights> Acquire(const std::string &prefix,
                                               int expert_id);
  mlx::core::array Routed(const mlx::core::array &input,
                          const mlx::core::array &expert_ids,
                          const mlx::core::array &scores,
                          const std::string &prefix,
                          const omni_common::Checkpoint &checkpoint);
  // The caller must synchronize GPU consumers before releasing entries.
  void Release();
  size_t released_entries() const { return released_entries_; }
  void SetProfileContext(const char *phase, int frame_index) {
    if (profile_) {
      profile_phase_ = phase;
      profile_frame_index_ = frame_index;
    } else {
    }
  }
  size_t resident_bytes() const { return resident_bytes_; }
  size_t peak_bytes() const { return peak_bytes_; }
  size_t misses() const { return misses_; }
  size_t hits() const { return hits_; }
  size_t evictions() const { return evictions_; }
  size_t payload_read_bytes() const { return payload_read_bytes_->load(); }

private:
  struct TensorSlice {
    std::shared_ptr<mlx::core::io::Reader> shard;
    mlx::core::Shape shape;
    mlx::core::Dtype dtype;
    size_t offset_bytes;
    size_t expert_bytes;
  };
  using Recency = std::list<std::pair<std::string, int>>;
  struct Entry {
    std::shared_ptr<ExpertWeights> weights;
    size_t packed_bytes;
    Recency::iterator recency;
  };
  std::map<std::string, TensorSlice> slices_;
  std::map<std::pair<std::string, int>, Entry> entries_;
  size_t capacity_bytes_;
  size_t resident_bytes_ = 0;
  size_t peak_bytes_ = 0;
  Recency recency_;
  size_t misses_ = 0;
  size_t hits_ = 0;
  size_t evictions_ = 0;
  size_t released_entries_ = 0;
  std::shared_ptr<std::atomic<size_t>> payload_read_bytes_ =
      std::make_shared<std::atomic<size_t>>(0);
  bool profile_ = false;
  std::string profile_phase_ = "unspecified";
  int profile_frame_index_ = -1;
  double load_seconds_ = 0;
};

} // namespace fishaudio_drama3
