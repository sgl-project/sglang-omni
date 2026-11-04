// SPDX-License-Identifier: Apache-2.0
#include "fishaudio_drama3/fishaudio_drama3_experts.h"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <type_traits>

namespace mx = mlx::core;
namespace drama = fishaudio_drama3;

static_assert(!std::is_copy_constructible_v<drama::ExpertCache>);
static_assert(!std::is_copy_assignable_v<drama::ExpertCache>);

namespace {

const std::string kPrefix = "backbone.model.layers.0.mlp.switch_mlp";

void Require(bool condition, const std::string &message) {
  if (!condition) {
    throw std::runtime_error(message);
  } else {
  }
}

class CountingReader : public mx::io::Reader {
public:
  explicit CountingReader(const std::string &path)
      : stream_(path, std::ios::binary) {}
  size_t payload_bytes = 0;
  bool is_open() const override { return stream_.is_open(); }
  bool good() const override { return stream_.good(); }
  std::string label() const override { return "counted fixture"; }
  size_t tell() override { return stream_.tellg(); }
  void seek(int64_t offset, std::ios_base::seekdir direction) override {
    stream_.seekg(offset, direction);
  }
  void read(char *destination, size_t count) override {
    stream_.read(destination, count);
    Require(stream_.good(), "short fixture read");
  }
  void read(char *destination, size_t count, size_t offset) override {
    const std::lock_guard<std::mutex> lock(mutex_);
    payload_bytes += count;
    stream_.seekg(offset);
    read(destination, count);
  }

private:
  std::ifstream stream_;
  std::mutex mutex_;
};

void Verify(const std::filesystem::path &directory, int bits, mx::Dtype dtype) {
  std::filesystem::create_directories(directory);
  omni_common::WeightMap weights;
  size_t expert_bytes = 0;
  for (const char *projection : {"up_proj", "gate_proj", "down_proj"}) {
    const auto values = mx::astype(
        mx::reshape(
            mx::multiply(
                mx::sin(mx::add(mx::arange(17 * 128 * 128, mx::float32),
                                mx::array(static_cast<float>(projection[0])))),
                mx::array(0.05f)),
            {17, 128, 128}),
        dtype);
    const auto quantized = mx::quantize(values, 64, bits);
    const auto &packed = quantized[0];
    const auto &scales = quantized[1];
    const auto &biases = quantized[2];
    weights.emplace(kPrefix + "." + projection + ".weight", packed);
    weights.emplace(kPrefix + "." + projection + ".scales", scales);
    weights.emplace(kPrefix + "." + projection + ".biases", biases);
    expert_bytes += (packed.nbytes() + scales.nbytes() + biases.nbytes()) / 17;
  }
  const std::string other_prefix = "backbone.model.layers.1.mlp.switch_mlp";
  for (const char *projection : {"up_proj", "gate_proj", "down_proj"}) {
    for (const char *suffix : {"weight", "scales", "biases"}) {
      const std::string name = std::string(projection) + "." + suffix;
      weights.emplace(other_prefix + "." + name,
                      weights.at(kPrefix + "." + name));
    }
  }
  weights.emplace("backbone.model.layers.0.mlp.gate.weight",
                  mx::ones({3, 128}));
  weights.emplace("backbone.model.layers.0.mlp.shared_expert.weight",
                  mx::ones({128, 128}));
  weights.emplace("tower.test", mx::ones({1}));
  weights.emplace("codec.test", mx::ones({1}));
  const auto path = directory / "model.safetensors";
  mx::save_safetensors(path.string(), weights);
  const nlohmann::json config = {
      {"quantization", {{"group_size", 64}, {"bits", bits}}}};
  auto reader = std::make_shared<CountingReader>(path.string());
  auto [lazy_weights, metadata] = mx::load_safetensors(reader);
  Require(reader->payload_bytes == 0, "MLX load read payload eagerly");
  std::erase_if(lazy_weights, [](const auto &named) {
    return drama::IsRoutedExpertWeight(named.first);
  });
  const omni_common::Checkpoint observed(std::move(lazy_weights), config);
  Require(reader->payload_bytes == 3 * 128 * 4 + 128 * 128 * 4 + 8,
          "excluded expert payload was read");
  const omni_common::Checkpoint resident(
      drama::LoadDrama3Weights(directory, false), config);
  const omni_common::Checkpoint initial(
      drama::LoadDrama3Weights(directory, true), config);
  Require(!initial.Has(kPrefix + ".up_proj.weight"),
          "initial checkpoint retained experts");
  Require(initial.Has("tower.test") && !initial.Has("codec.test") &&
              initial.Has("backbone.model.layers.0.mlp.gate.weight") &&
              initial.Has("backbone.model.layers.0.mlp.shared_expert.weight"),
          "resident selection changed");
  drama::ExpertCache empty(directory);
  Require(empty.resident_bytes() == 0 && empty.peak_bytes() == 0,
          "metadata indexing materialized experts");
  drama::ExpertCache cache(directory, expert_bytes);
  for (const int positions : {1, 8, 31, 32, 40, 128}) {
    const auto input = mx::astype(
        mx::reshape(mx::cos(mx::arange(positions * 128, mx::float32)),
                    {1, positions, 128}),
        dtype);
    const auto probabilities = mx::softmax(
        mx::reshape(mx::sin(mx::arange(positions * 17, mx::float32)),
                    {1, positions, 17}),
        -1);
    const auto ids = mx::slice(mx::argpartition(probabilities, 15, -1),
                               {0, 0, 15}, {1, positions, 17});
    auto scores = mx::take_along_axis(probabilities, ids, -1);
    scores = mx::divide(scores, mx::sum(scores, -1, true));
    const bool sorted = positions * 2 >= 64;
    auto rows = mx::expand_dims(input, {-2, -3});
    auto expert_ids = ids;
    std::optional<mx::array> inverse_order;
    if (sorted) {
      const auto flat_ids = mx::flatten(ids);
      const auto order = mx::argsort(flat_ids);
      inverse_order = mx::argsort(order);
      rows = mx::take(mx::flatten(rows, 0, -3),
                      mx::floor_divide(order, mx::array(2, order.dtype())), 0);
      expert_ids = mx::take(flat_ids, order);
    } else {
    }
    const auto linear = [&](const mx::array &value,
                            const std::string &projection) {
      return mx::gather_qmm(
          value, resident.Weight(kPrefix + "." + projection + ".weight"),
          resident.Weight(kPrefix + "." + projection + ".scales"),
          resident.Weight(kPrefix + "." + projection + ".biases"), std::nullopt,
          expert_ids, true, 64, bits, "affine", std::nullopt, sorted);
    };
    auto expected = linear(
        drama::Swiglu(linear(rows, "gate_proj"), linear(rows, "up_proj")),
        "down_proj");
    if (sorted) {
      expected = mx::reshape(mx::take(expected, *inverse_order, 0),
                             {1, positions, 2, 1, 128});
    } else {
    }
    expected = mx::sum(
        mx::multiply(mx::squeeze(expected, -2), mx::expand_dims(scores, -1)),
        -2);
    const auto actual = cache.Routed(input, ids, scores, kPrefix, initial);
    const float error =
        mx::max(mx::abs(mx::subtract(actual, expected))).item<float>();
    Require(error == 0.0f, "cached output differs");
    Require(cache.peak_bytes() <= expert_bytes, "cache exceeded capacity");
    mx::synchronize();
    const size_t misses_before_release = cache.misses();
    const size_t evictions_before_release = cache.evictions();
    const size_t released_before = cache.released_entries();
    cache.Release();
    Require(cache.resident_bytes() == 0 &&
                cache.misses() == misses_before_release &&
                cache.evictions() == evictions_before_release &&
                cache.released_entries() == released_before + 1,
            "phase release changed LRU counters or retained expert bytes");
    const auto reloaded = cache.Routed(input, ids, scores, kPrefix, initial);
    Require(mx::max(mx::abs(mx::subtract(reloaded, expected))).item<float>() ==
                    0.0f &&
                cache.misses() > misses_before_release &&
                cache.peak_bytes() <= expert_bytes,
            "release/reload changed numerical output or capacity");
    std::cout << "bits=" << bits << " dtype=" << dtype
              << " positions=" << positions << " max_absolute_error=" << error
              << " peak_bytes=" << cache.peak_bytes() << '\n';
  }
  auto pinned = cache.Acquire(kPrefix, 0);
  Require(pinned->at("up_proj.weight").dtype() == mx::uint32 &&
              pinned->at("up_proj.scales").dtype() == dtype,
          "materialization changed packed dtype");
  const size_t misses = cache.misses();
  const size_t read_bytes = cache.payload_read_bytes();
  const size_t hits = cache.hits();
  Require(cache.Acquire(kPrefix, 0) == pinned && cache.misses() == misses,
          "cache hit failed");
  Require(cache.hits() == hits + 1 &&
              cache.payload_read_bytes() == read_bytes &&
              read_bytes == cache.misses() * expert_bytes,
          "read/hit counters differ");
  bool rejected = false;
  try {
    cache.Acquire(kPrefix, 1);
  } catch (const std::runtime_error &) {
    rejected = true;
  }
  Require(rejected, "pinned expert was evicted");
  rejected = false;
  try {
    cache.Acquire(other_prefix, 0);
  } catch (const std::runtime_error &) {
    rejected = true;
  }
  Require(rejected, "cache capacity was not global across layers");
  pinned.reset();
  cache.Acquire(kPrefix, 1);
  Require(cache.misses() == misses + 1, "unpinned eviction failed");
  drama::ExpertCache lru(directory, 2 * expert_bytes);
  lru.Acquire(kPrefix, 0);
  lru.Acquire(kPrefix, 1);
  lru.Acquire(kPrefix, 0);
  lru.Acquire(other_prefix, 0);
  Require(lru.misses() == 3, "LRU hit changed miss count");
  lru.Acquire(kPrefix, 1);
  Require(lru.evictions() == 2 &&
              lru.payload_read_bytes() == 4 * expert_bytes &&
              lru.misses() == 4 && lru.peak_bytes() == 2 * expert_bytes,
          "LRU victim was incorrect");
  drama::ExpertCache pinned_oldest(directory, 2 * expert_bytes);
  auto oldest = pinned_oldest.Acquire(kPrefix, 0);
  pinned_oldest.Acquire(kPrefix, 1);
  pinned_oldest.Acquire(other_prefix, 0);
  Require(pinned_oldest.Acquire(kPrefix, 0) == oldest &&
              pinned_oldest.misses() == 3 && pinned_oldest.evictions() == 1,
          "pinned oldest was not skipped");
  oldest.reset();
  pinned_oldest.Acquire(other_prefix, 0);
  pinned_oldest.Acquire(kPrefix, 1);
  const size_t misses_after_unpin = pinned_oldest.misses();
  pinned_oldest.Acquire(kPrefix, 0);
  Require(pinned_oldest.misses() == misses_after_unpin + 1,
          "unpinned oldest lost its hit recency");
  rejected = false;
  try {
    cache.Acquire(kPrefix, -1);
  } catch (const std::runtime_error &) {
    rejected = true;
  }
  Require(rejected && cache.resident_bytes() == expert_bytes,
          "invalid id changed cache");
  drama::ExpertCache release_cache(directory, 2 * expert_bytes);
  release_cache.Acquire(kPrefix, 0);
  auto release_pinned = release_cache.Acquire(kPrefix, 1);
  rejected = false;
  try {
    release_cache.Release();
  } catch (const std::runtime_error &) {
    rejected = true;
  }
  Require(rejected && release_cache.resident_bytes() == 2 * expert_bytes &&
              release_cache.released_entries() == 0,
          "pinned release partially removed entries or lost byte accounting");
  release_pinned.reset();
  mx::synchronize();
  release_cache.Release();
  release_cache.Release();
  Require(release_cache.resident_bytes() == 0 &&
              release_cache.released_entries() == 2 &&
              release_cache.evictions() == 0,
          "release was not idempotent or counted as LRU eviction");
  release_cache.Acquire(kPrefix, 0);
  Require(release_cache.misses() == 3 &&
              release_cache.payload_read_bytes() == 3 * expert_bytes &&
              release_cache.resident_bytes() == expert_bytes,
          "released metadata could not reload an expert");
  release_cache.Acquire(kPrefix, 1);
  release_cache.Acquire(kPrefix, 2);
  Require(release_cache.evictions() == 1 && release_cache.misses() == 5,
          "release/reload retained stale recency");
  drama::ExpertCache too_small(directory, expert_bytes - 1);
  rejected = false;
  try {
    too_small.Acquire(kPrefix, 0);
  } catch (const std::runtime_error &) {
    rejected = true;
  }
  Require(rejected && too_small.resident_bytes() == 0,
          "oversized expert was loaded");
  // Note (Codex): replacing a shard must not mix cached and uncached checkpoint
  // values.
  drama::ExpertCache replaced(directory, 2 * expert_bytes);
  const auto cached_expert = replaced.Acquire(kPrefix, 0);
  auto replacement_weights = weights;
  for (auto &[name, weight] : replacement_weights) {
    if (drama::IsRoutedExpertWeight(name) && name.ends_with(".biases")) {
      weight = mx::add(weight, mx::array(1.0f, weight.dtype()));
    } else {
    }
  }
  const auto replacement_path = directory / "replacement.tmp.safetensors";
  mx::save_safetensors(replacement_path.string(), replacement_weights);
  Require(std::filesystem::file_size(path) ==
              std::filesystem::file_size(replacement_path),
          "replacement fixture changed shard size");
  auto original_reader = std::make_shared<CountingReader>(path.string());
  auto replacement_reader =
      std::make_shared<CountingReader>(replacement_path.string());
  uint64_t original_header_bytes = 0;
  uint64_t replacement_header_bytes = 0;
  original_reader->read(reinterpret_cast<char *>(&original_header_bytes), 8);
  replacement_reader->read(reinterpret_cast<char *>(&replacement_header_bytes),
                           8);
  std::string original_header(original_header_bytes, '\0');
  std::string replacement_header(replacement_header_bytes, '\0');
  original_reader->read(original_header.data(), original_header.size());
  replacement_reader->read(replacement_header.data(),
                           replacement_header.size());
  Require(original_header == replacement_header,
          "replacement fixture changed tensor layout");
  Require(
      mx::max(
          mx::abs(mx::subtract(
              mx::astype(weights.at(kPrefix + ".up_proj.biases"), mx::float32),
              mx::astype(replacement_weights.at(kPrefix + ".up_proj.biases"),
                         mx::float32))))
              .item<float>() > 0.5f,
      "replacement fixture did not change expert values");
  std::filesystem::rename(replacement_path, path);
  Require(replaced.Acquire(kPrefix, 0) == cached_expert,
          "replacement invalidated cached expert");
  const auto uncached_expert = replaced.Acquire(kPrefix, 1);
  for (const auto &[name, weight] : *uncached_expert) {
    const auto expected =
        mx::take(weights.at(kPrefix + "." + name), mx::array(1), 0);
    Require(mx::max(mx::abs(mx::subtract(mx::astype(weight, mx::float32),
                                         mx::astype(expected, mx::float32))))
                    .item<float>() == 0.0f,
            "atomic replacement mixed checkpoint values for uncached expert: " +
                name);
  }
  Require(replaced.hits() == 1 && replaced.misses() == 2 &&
              replaced.payload_read_bytes() == 2 * expert_bytes,
          "replacement changed cache hit or payload accounting");
  std::cout << "atomic replacement retained indexed shard bits=" << bits
            << " dtype=" << dtype << '\n';
  mx::save_safetensors(path.string(), weights);
  drama::ExpertCache failing(directory, expert_bytes);
  std::filesystem::resize_file(path, 0);
  rejected = false;
  try {
    failing.Acquire(kPrefix, 0);
  } catch (const std::runtime_error &) {
    rejected = true;
  }
  Require(rejected && failing.resident_bytes() == 0 && failing.misses() == 0,
          "failed read published a cache entry");
  mx::save_safetensors(std::make_shared<mx::io::FileWriter>(path.string()),
                       weights);
  Require(failing.Acquire(kPrefix, 0)->size() == 9,
          "failed read prevented retry");
  failing.Acquire(kPrefix, 1);
  Require(failing.misses() == 2 && failing.evictions() == 1 &&
              failing.resident_bytes() == expert_bytes,
          "failed read left stale recency during eviction");
}

} // namespace

int main(int argc, char **argv) {
  try {
    Require(argc == 2, "expected fixture directory");
    for (const int bits : {4, 8}) {
      for (const auto dtype : {mx::float32, mx::float16, mx::bfloat16}) {
        Verify(std::filesystem::path(argv[1]) /
                   (std::to_string(bits) + "-" +
                    std::to_string(static_cast<int>(dtype.val()))),
               bits, dtype);
      }
    }
    const auto malformed_directory =
        std::filesystem::path(argv[1]) / "malformed";
    std::filesystem::create_directories(malformed_directory);
    {
      std::ofstream malformed(malformed_directory / "broken.safetensors",
                              std::ios::binary);
      malformed << "short";
    }
    bool rejected = false;
    try {
      drama::ExpertCache malformed(malformed_directory);
    } catch (const std::exception &) {
      rejected = true;
    }
    Require(rejected, "malformed header was accepted");
    const std::string invalid_header =
        nlohmann::json({{kPrefix + ".up_proj.weight",
                         {{"dtype", "U32"},
                          {"shape", {1, 2, 2}},
                          {"data_offsets", {0, 4}}}}})
            .dump();
    {
      std::ofstream malformed(malformed_directory / "broken.safetensors",
                              std::ios::binary);
      const uint64_t header_bytes = invalid_header.size();
      malformed.write(reinterpret_cast<const char *>(&header_bytes), 8);
      malformed << invalid_header << "xxxx";
    }
    rejected = false;
    try {
      drama::ExpertCache malformed(malformed_directory);
    } catch (const std::exception &) {
      rejected = true;
    }
    Require(rejected, "invalid tensor byte length was accepted");
    for (const size_t capacity_bytes :
         {size_t(0), drama::kExpertCacheBytes + 1}) {
      rejected = false;
      try {
        drama::ExpertCache invalid(malformed_directory, capacity_bytes);
      } catch (const std::invalid_argument &) {
        rejected = true;
      }
      Require(rejected, "invalid cache budget was accepted");
    }
    std::cout << "all expert cache checks passed\n";
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
