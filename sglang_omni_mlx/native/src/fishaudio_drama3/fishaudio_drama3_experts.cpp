// SPDX-License-Identifier: Apache-2.0
#include "fishaudio_drama3/fishaudio_drama3_experts.h"

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <iostream>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <sys/stat.h>
#include <unistd.h>

#include "qwen3_asr/qwen3_asr_transcriber.h"

namespace fishaudio_drama3 {
namespace mx = mlx::core;

namespace {

bool IsDrama3Weight(const std::string &name) {
  return name.starts_with("backbone.") || name.starts_with("semantic_head.") ||
         name.starts_with("tower.");
}

std::vector<std::filesystem::path>
DramaWeightFiles(const std::filesystem::path &model_directory) {
  std::vector<std::filesystem::path> paths;
  for (const auto &file :
       std::filesystem::directory_iterator(model_directory)) {
    if (file.is_regular_file() && file.path().extension() == ".safetensors" &&
        !file.path().filename().string().starts_with("._")) {
      paths.push_back(file.path());
    } else {
    }
  }
  if (paths.empty()) {
    throw std::runtime_error("no .safetensors files in " +
                             model_directory.string());
  } else {
  }
  std::sort(paths.begin(), paths.end());
  return paths;
}

class ShardReader : public mx::io::Reader {
public:
  explicit ShardReader(const std::filesystem::path &path)
      : path_(path.string()), descriptor_(open(path_.c_str(), O_RDONLY)) {
    if (descriptor_ < 0) {
      throw std::runtime_error("cannot open expert shard: " + path_);
    } else {
    }
  }
  ~ShardReader() override { close(descriptor_); }
  ShardReader(const ShardReader &) = delete;
  ShardReader &operator=(const ShardReader &) = delete;
  bool is_open() const override { return descriptor_ >= 0; }
  bool good() const override { return is_open(); }
  std::string label() const override { return path_; }
  size_t size_bytes() const {
    struct stat status {};
    if (fstat(descriptor_, &status) != 0 || !S_ISREG(status.st_mode) ||
        status.st_size < 0) {
      throw std::runtime_error("invalid expert shard: " + path_);
    } else {
    }
    return static_cast<size_t>(status.st_size);
  }
  size_t tell() override { return position_bytes_; }
  void seek(int64_t offset, std::ios_base::seekdir direction) override {
    const int64_t file_bytes = size_bytes();
    const int64_t base = direction == std::ios_base::end   ? file_bytes
                         : direction == std::ios_base::cur ? position_bytes_
                                                           : 0;
    if (offset < -base || offset > file_bytes - base) {
      throw std::runtime_error("invalid expert shard seek");
    } else {
    }
    position_bytes_ = base + offset;
  }
  void read(char *destination, size_t count) override {
    read(destination, count, position_bytes_);
    position_bytes_ += count;
  }
  void read(char *destination, size_t count, size_t offset) override {
    const size_t file_bytes = size_bytes();
    if (offset > file_bytes || count > file_bytes - offset) {
      throw std::runtime_error("expert payload exceeds file extent: " + path_);
    } else {
    }
    while (count != 0) {
      const ssize_t received = pread(descriptor_, destination, count, offset);
      if (received < 0 && errno == EINTR) {
        continue;
      } else if (received <= 0) {
        throw std::runtime_error("short expert shard read: " + path_);
      } else {
      }
      destination += received;
      offset += received;
      count -= received;
    }
  }

private:
  std::string path_;
  int descriptor_;
  size_t position_bytes_ = 0;
};

class ExpertSliceReader : public mx::io::Reader {
public:
  ExpertSliceReader(std::shared_ptr<mx::io::Reader> shard, size_t offset_bytes,
                    size_t payload_bytes, const mx::Shape &shape,
                    mx::Dtype dtype,
                    std::shared_ptr<std::atomic<size_t>> payload_read_bytes)
      : payload_read_bytes_(std::move(payload_read_bytes)),
        shard_(std::move(shard)), offset_bytes_(offset_bytes),
        payload_bytes_(payload_bytes) {
    const std::string dtype_name = dtype == mx::uint32     ? "U32"
                                   : dtype == mx::float16  ? "F16"
                                   : dtype == mx::bfloat16 ? "BF16"
                                                           : "F32";
    const std::string metadata =
        nlohmann::json({{"expert",
                         {{"dtype", dtype_name},
                          {"shape", shape},
                          {"data_offsets", {0, payload_bytes}}}}})
            .dump();
    const uint64_t header_bytes = metadata.size();
    header_.resize(8 + header_bytes);
    std::memcpy(header_.data(), &header_bytes, 8);
    std::memcpy(header_.data() + 8, metadata.data(), header_bytes);
  }
  bool is_open() const override { return shard_->is_open(); }
  bool good() const override { return shard_->good(); }
  std::string label() const override { return shard_->label(); }
  size_t tell() override { return position_bytes_; }
  void seek(int64_t offset, std::ios_base::seekdir direction) override {
    const int64_t base = direction == std::ios_base::end
                             ? header_.size() + payload_bytes_
                         : direction == std::ios_base::cur ? position_bytes_
                                                           : 0;
    if (offset < -base ||
        offset > static_cast<int64_t>(header_.size() + payload_bytes_) - base) {
      throw std::runtime_error("invalid expert reader seek");
    } else {
    }
    position_bytes_ = base + offset;
  }
  void read(char *destination, size_t count) override {
    read(destination, count, position_bytes_);
    position_bytes_ += count;
  }
  void read(char *destination, size_t count, size_t offset) override {
    if (offset > header_.size() + payload_bytes_ ||
        count > header_.size() + payload_bytes_ - offset) {
      throw std::runtime_error("invalid expert reader range");
    } else {
    }
    if (offset < header_.size()) {
      const size_t header_count = std::min(count, header_.size() - offset);
      std::memcpy(destination, header_.data() + offset, header_count);
      destination += header_count;
      offset += header_count;
      count -= header_count;
    } else {
    }
    shard_->read(destination, count, offset_bytes_ + offset - header_.size());
    payload_read_bytes_->fetch_add(count);
  }

private:
  std::shared_ptr<std::atomic<size_t>> payload_read_bytes_;
  std::shared_ptr<mx::io::Reader> shard_;
  std::string header_;
  size_t offset_bytes_;
  size_t payload_bytes_;
  size_t position_bytes_ = 0;
};

std::vector<mx::array> SwigluGraph(const std::vector<mx::array> &inputs) {
  const mx::array &gate = inputs[0];
  return {mx::multiply(mx::multiply(gate, mx::sigmoid(gate)), inputs[1])};
}

} // namespace

// silu(gate) * x, fused into one kernel.
mx::array Swiglu(const mx::array &gate, const mx::array &x) {
  static const auto compiled = mx::compile(SwigluGraph, true);
  return compiled({gate, x})[0];
}

omni_common::WeightMap
LoadDrama3Weights(const std::filesystem::path &model_directory,
                  bool offload_experts) {
  omni_common::WeightMap weights;
  for (const auto &path : DramaWeightFiles(model_directory)) {
    auto [loaded, metadata] = mx::load_safetensors(path.string());
    for (auto &[name, weight] : loaded) {
      weights.insert_or_assign(name, std::move(weight));
    }
  }
  std::erase_if(weights, [offload_experts](const auto &named) {
    if (offload_experts && IsRoutedExpertWeight(named.first)) {
      return true;
    } else {
    }
    return !IsDrama3Weight(named.first);
  });
  return weights;
}

bool IsRoutedExpertWeight(const std::string &name) {
  return name.starts_with("backbone.") &&
         name.find(".mlp.switch_mlp.") != std::string::npos;
}

ExpertCache::ExpertCache(const std::filesystem::path &model_directory,
                         size_t capacity_bytes)
    : ExpertCache(model_directory,
                  omni_common::ReadJson(model_directory / "config.json"),
                  capacity_bytes) {}

ExpertCache::ExpertCache(const std::filesystem::path &model_directory,
                         const nlohmann::json &config, size_t capacity_bytes)
    : capacity_bytes_(capacity_bytes),
      profile_(std::getenv("DRAMA3_EXPERT_PROFILE") != nullptr) {
  if (capacity_bytes == 0 || capacity_bytes > kExpertCacheBytes) {
    throw std::invalid_argument("expert cache capacity must be in (0, 2 GiB]");
  } else {
  }
  for (const auto &path : DramaWeightFiles(model_directory)) {
    // Note (Codex): MLX validates extents and dtypes without evaluating
    // payloads.
    auto shard = std::make_shared<ShardReader>(path);
    const auto [weights, metadata] = mx::load_safetensors(shard);
    uint64_t header_bytes = 0;
    shard->read(reinterpret_cast<char *>(&header_bytes), sizeof(header_bytes),
                0);
    if (header_bytes == 0 || header_bytes >= 100000000) {
      throw std::runtime_error("invalid safetensors header");
    } else {
    }
    std::string header(header_bytes, '\0');
    shard->read(header.data(), header.size(), sizeof(header_bytes));
    const auto tensors = nlohmann::json::parse(header);
    for (const auto &[name, weight] : weights) {
      if (!IsRoutedExpertWeight(name)) {
        if (IsDrama3Weight(name)) {
          resident_weights_.insert_or_assign(name, weight);
        } else {
        }
        continue;
      } else if (weight.ndim() != 3 || weight.shape(0) <= 0 ||
                 weight.nbytes() == 0) {
        throw std::runtime_error("invalid expert tensor shape: " + name);
      } else {
      }
      size_t expected_bytes = weight.itemsize();
      for (const int dimension : weight.shape()) {
        if (dimension <= 0 ||
            expected_bytes > std::numeric_limits<size_t>::max() / dimension) {
          throw std::runtime_error("invalid expert tensor length: " + name);
        } else {
        }
        expected_bytes *= dimension;
      }
      const auto offsets =
          tensors.at(name).at("data_offsets").get<std::vector<size_t>>();
      const size_t file_bytes = shard->size_bytes();
      if (offsets.size() != 2 || offsets[1] < offsets[0] ||
          offsets[1] - offsets[0] != expected_bytes ||
          8 + header_bytes > file_bytes ||
          offsets[1] > file_bytes - 8 - header_bytes) {
        throw std::runtime_error("invalid expert tensor extent: " + name);
      } else {
      }
      const size_t offset_bytes = offsets[0];
      if (!slices_
               .emplace(name, TensorSlice{shard, weight.shape(), weight.dtype(),
                                          8 + header_bytes + offset_bytes,
                                          weight.nbytes() / weight.shape(0)})
               .second) {
        throw std::runtime_error("duplicate expert tensor: " + name);
      } else {
      }
    }
  }
  if (slices_.empty()) {
    throw std::runtime_error("no routed expert tensors found");
  } else {
  }
  const auto &text = config.at("text_config");
  const size_t layer_count = text.at("num_hidden_layers").get<size_t>();
  const int expert_count = text.at("num_experts").get<int>();
  const auto mlp_layer_types =
      text.value("mlp_layer_types",
                 nlohmann::json(std::vector<std::string>(layer_count, "moe")));
  if (mlp_layer_types !=
      nlohmann::json(std::vector<std::string>(layer_count, "moe"))) {
    throw std::runtime_error(
        "expected a mixture-of-experts MLP in every layer");
  } else {
  }
  for (size_t layer = 0; layer < layer_count; ++layer) {
    const std::string prefix =
        "backbone.model.layers." + std::to_string(layer) + ".mlp.switch_mlp";
    for (const char *projection : {"up_proj", "gate_proj", "down_proj"}) {
      const std::string projection_prefix = prefix + "." + projection;
      for (const char *suffix : {"weight", "scales", "biases"}) {
        if (!slices_.contains(projection_prefix + "." + suffix)) {
          throw std::runtime_error("missing routed expert tensor: " +
                                   projection_prefix + "." + suffix);
        } else {
        }
      }
      const auto &weight = slices_.at(projection_prefix + ".weight");
      const auto &scales = slices_.at(projection_prefix + ".scales");
      const auto &biases = slices_.at(projection_prefix + ".biases");
      if (weight.dtype != mx::uint32 || weight.shape[0] != expert_count ||
          scales.shape != biases.shape || scales.shape[0] != weight.shape[0] ||
          scales.shape[1] != weight.shape[1] || scales.dtype != biases.dtype ||
          (scales.dtype != mx::float16 && scales.dtype != mx::bfloat16 &&
           scales.dtype != mx::float32)) {
        throw std::runtime_error("inconsistent quantized expert tensors: " +
                                 prefix);
      } else {
      }
    }
  }
}

omni_common::WeightMap ExpertCache::TakeResidentWeights() {
  if (resident_weights_taken_) {
    throw std::runtime_error(
        "resident weights already taken from expert cache");
  } else {
    resident_weights_taken_ = true;
    return std::move(resident_weights_);
  }
}

void ExpertCache::Release() {
  for (const auto &[identity, entry] : entries_) {
    if (entry.weights.use_count() != 1) {
      throw std::runtime_error("cannot release a pinned expert cache");
    } else {
    }
  }
  released_entries_ += entries_.size();
  entries_.clear();
  recency_.clear();
  resident_bytes_ = 0;
}

std::shared_ptr<const ExpertCache::ExpertWeights>
ExpertCache::Acquire(const std::string &prefix, int expert_id) {
  const auto identity = std::make_pair(prefix, expert_id);
  const auto found = entries_.find(identity);
  if (found != entries_.end()) {
    ++hits_;
    recency_.splice(recency_.end(), recency_, found->second.recency);
    return found->second.weights;
  } else {
  }
  const auto load_started = profile_ ? std::chrono::steady_clock::now()
                                     : std::chrono::steady_clock::time_point{};
  size_t packed_bytes = 0;
  for (const char *projection : {"up_proj", "gate_proj", "down_proj"}) {
    for (const char *suffix : {"weight", "scales", "biases"}) {
      const auto &tensor = slices_.at(prefix + "." + projection + "." + suffix);
      if (expert_id < 0 || expert_id >= tensor.shape[0] ||
          tensor.expert_bytes > capacity_bytes_ - packed_bytes) {
        throw std::runtime_error(
            "invalid expert id or expert exceeds cache capacity");
      } else {
      }
      if ((std::string(suffix) == "weight" && tensor.dtype != mx::uint32) ||
          (std::string(suffix) != "weight" && tensor.dtype != mx::float16 &&
           tensor.dtype != mx::bfloat16 && tensor.dtype != mx::float32)) {
        throw std::runtime_error("unsupported quantized expert dtype");
      } else {
      }
      packed_bytes += tensor.expert_bytes;
    }
  }
  while (resident_bytes_ > capacity_bytes_ - packed_bytes) {
    auto victim = entries_.end();
    for (auto recency = recency_.begin(); recency != recency_.end();
         ++recency) {
      const auto entry = entries_.find(*recency);
      if (entry->second.weights.use_count() == 1) {
        victim = entry;
        break;
      } else {
      }
    }
    if (victim == entries_.end()) {
      throw std::runtime_error("expert cache capacity is pinned");
    } else {
    }
    resident_bytes_ -= victim->second.packed_bytes;
    recency_.erase(victim->second.recency);
    entries_.erase(victim);
    ++evictions_;
  }
  peak_bytes_ = std::max(peak_bytes_, resident_bytes_ + packed_bytes);
  auto weights = std::make_shared<ExpertWeights>();
  std::vector<mx::array> parameters;
  for (const char *projection : {"up_proj", "gate_proj", "down_proj"}) {
    for (const char *suffix : {"weight", "scales", "biases"}) {
      const std::string name = std::string(projection) + "." + suffix;
      const auto &tensor = slices_.at(prefix + "." + name);
      const size_t offset_bytes =
          tensor.offset_bytes + expert_id * tensor.expert_bytes;
      mx::Shape shape(tensor.shape.begin() + 1, tensor.shape.end());
      auto [loaded, metadata] =
          mx::load_safetensors(std::make_shared<ExpertSliceReader>(
              tensor.shard, offset_bytes, tensor.expert_bytes, shape,
              tensor.dtype, payload_read_bytes_));
      mx::array weight = loaded.at("expert");
      parameters.push_back(weight);
      weights->emplace(name, std::move(weight));
    }
  }
  // Note (Codex): completed loads own MLX arrays, without a second host payload
  // cache.
  mx::eval(parameters);
  recency_.push_back(identity);
  try {
    entries_.emplace(identity,
                     Entry{weights, packed_bytes, std::prev(recency_.end())});
  } catch (...) {
    recency_.pop_back();
    throw;
  }
  resident_bytes_ += packed_bytes;
  ++misses_;
  if (profile_) {
    load_seconds_ += std::chrono::duration<double>(
                         std::chrono::steady_clock::now() - load_started)
                         .count();
  } else {
  }
  return weights;
}

mx::array ExpertCache::Routed(const mx::array &input,
                              const mx::array &expert_ids,
                              const mx::array &scores,
                              const std::string &prefix,
                              const omni_common::Checkpoint &checkpoint,
                              const std::atomic<bool> *cancel) {
  const auto routed_started = profile_
                                  ? std::chrono::steady_clock::now()
                                  : std::chrono::steady_clock::time_point{};
  if (input.ndim() < 2 || expert_ids.ndim() != input.ndim() ||
      expert_ids.shape() != scores.shape() || expert_ids.size() == 0 ||
      expert_ids.size() >
          static_cast<size_t>(std::numeric_limits<int32_t>::max()) ||
      !std::equal(input.shape().begin(), input.shape().end() - 1,
                  expert_ids.shape().begin())) {
    throw std::invalid_argument("inconsistent routed expert rows");
  } else {
  }
  const int top_k = expert_ids.shape(-1);
  const mx::array flat_input = mx::reshape(input, {-1, input.shape(-1)});
  const mx::array flat_ids =
      mx::astype(mx::reshape(expert_ids, {-1, top_k}), mx::int32);
  const mx::array flat_scores = mx::reshape(scores, {-1, top_k});
  mx::eval({flat_input, flat_ids, flat_scores});
  std::map<int, std::vector<int32_t>> positions_by_expert;
  for (int position = 0; position < static_cast<int>(flat_ids.size());
       ++position) {
    positions_by_expert[flat_ids.data<int32_t>()[position]].push_back(position);
  }
  std::vector<std::optional<mx::array>> routed(flat_ids.size());
  for (const auto &[expert_id, positions] : positions_by_expert) {
    if (cancel != nullptr && cancel->load()) {
      mx::synchronize();
      throw qwen3_asr::TranscriptionCancelled();
    } else {
    }
    const auto lease = Acquire(prefix, expert_id);
    std::vector<int32_t> rows;
    for (const int position : positions) {
      rows.push_back(position / top_k);
    }
    const int expert_count = slices_.at(prefix + ".up_proj.weight").shape[0];
    const bool use_sorted_kernel =
        flat_ids.size() >= 64 && flat_ids.size() / expert_count >= 4;
    // Note (Codex): preserve MLX 0.32.3's sorted gather kernel when splitting a
    // batch.
    if (use_sorted_kernel && rows.size() < 16) {
      rows.resize(16, rows.back());
    } else {
    }
    const auto hidden = mx::expand_dims(
        mx::take(
            flat_input,
            mx::array(rows.data(), {static_cast<int>(rows.size())}, mx::int32),
            0),
        -2);
    const auto indices = mx::zeros({static_cast<int>(rows.size())}, mx::uint32);
    const auto linear = [&](const mx::array &value,
                            const std::string &projection) {
      const auto &quantization =
          checkpoint.Quantization(prefix + "." + projection);
      return mx::gather_qmm(
          value, mx::expand_dims(lease->at(projection + ".weight"), 0),
          mx::expand_dims(lease->at(projection + ".scales"), 0),
          mx::expand_dims(lease->at(projection + ".biases"), 0), std::nullopt,
          indices, true, quantization.group_size, quantization.bits,
          quantization.mode, std::nullopt, use_sorted_kernel);
    };
    auto result =
        linear(Swiglu(linear(hidden, "gate_proj"), linear(hidden, "up_proj")),
               "down_proj");
    mx::eval(result);
    for (size_t position = 0; position < positions.size(); ++position) {
      routed[positions[position]] =
          mx::take(result, mx::array(static_cast<int>(position)), 0);
    }
  }
  std::vector<mx::array> ordered;
  for (const auto &result : routed) {
    ordered.push_back(*result);
  }
  const auto combined = mx::sum(
      mx::multiply(mx::reshape(mx::concatenate(ordered, 0),
                               {flat_input.shape(0), top_k, input.shape(-1)}),
                   mx::expand_dims(flat_scores, -1)),
      -2);
  mx::eval(combined);
  if (profile_) {
    std::cerr
        << nlohmann::
               json{{"drama3_expert_profile", true},
                    {"phase", profile_phase_},
                    {"frame_index", profile_frame_index_},
                    {"unix_seconds",
                     std::chrono::duration<double>(
                         std::chrono::system_clock::now().time_since_epoch())
                         .count()},
                    {"monotonic_seconds",
                     std::chrono::duration<double>(
                         std::chrono::steady_clock::now().time_since_epoch())
                         .count()},
                    {"load_seconds_cumulative", load_seconds_},
                    {"routed_seconds",
                     std::chrono::duration<double>(
                         std::chrono::steady_clock::now() - routed_started)
                         .count()},
                    {"prefix", prefix},
                    {"rows", flat_input.shape(0)},
                    {"hits", hits_},
                    {"misses", misses_},
                    {"evictions", evictions_},
                    {"released_entries", released_entries_},
                    {"packed_bytes", resident_bytes_},
                    {"peak_packed_bytes", peak_bytes_},
                    {"payload_read_bytes", payload_read_bytes()},
                    {"mlx_active_bytes", mx::get_active_memory()},
                    {"mlx_peak_bytes", mx::get_peak_memory()},
                    {"mlx_allocator_cache_bytes", mx::get_cache_memory()}}
                   .dump()
        << '\n';
  } else {
  }
  return mx::reshape(combined, input.shape());
}

} // namespace fishaudio_drama3
