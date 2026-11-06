// SPDX-License-Identifier: Apache-2.0
#include "fishaudio_drama3/fishaudio_drama3_experts.h"

#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <future>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <type_traits>

#include "qwen3_asr/worker.h"

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

class BlockingReader : public CountingReader {
public:
  BlockingReader(const std::string &path, std::promise<void> &reading,
                 std::shared_future<void> resume)
      : CountingReader(path), reading_(reading), resume_(std::move(resume)) {}
  void read(char *destination, size_t count, size_t offset) override {
    reading_.set_value();
    resume_.wait();
    CountingReader::read(destination, count, offset);
  }

private:
  std::promise<void> &reading_;
  std::shared_future<void> resume_;
};

void VerifyCancelledWorker(const std::filesystem::path &directory,
                           const nlohmann::json &config, size_t expert_bytes) {
  std::promise<void> reading;
  auto reading_future = reading.get_future();
  std::promise<void> resume;
  auto resume_future = resume.get_future().share();
  std::promise<std::exception_ptr> cancelled;
  auto cancelled_future = cancelled.get_future();
  std::promise<qwen3_asr::TranscriptionResult> next;
  auto next_future = next.get_future();
  std::unique_ptr<drama::ExpertCache> cache;
  std::unique_ptr<omni_common::Checkpoint> checkpoint;
  qwen3_asr::TranscriptionWorker worker(
      [&] {
        cache = std::make_unique<drama::ExpertCache>(directory, config,
                                                     expert_bytes);
        checkpoint = std::make_unique<omni_common::Checkpoint>(
            cache->TakeResidentWeights(), config);
        mx::save_safetensors(std::make_shared<mx::io::FileWriter>(
                                 (directory / "input.pending").string()),
                             {{"input", mx::ones({1, 256, 128})}});
      },
      [] {});
  const auto cancel = qwen3_asr::NewCancelFlag();
  worker.Submit(
      [&](const std::atomic<bool> &cancelled) {
        auto reader = std::make_shared<BlockingReader>(
            (directory / "input.pending").string(), reading, resume_future);
        const auto [loaded, metadata] = mx::load_safetensors(reader);
        cache->Routed(
            loaded.at("input"),
            mx::broadcast_to(mx::arange(17, mx::uint32), {1, 256, 17}),
            mx::full({1, 256, 17}, 1.0f / 17), kPrefix, *checkpoint,
            &cancelled);
        return qwen3_asr::TranscriptionResult{};
      },
      cancel,
      [&](std::optional<qwen3_asr::TranscriptionResult>,
          std::exception_ptr error) { cancelled.set_value(error); });
  Require(reading_future.wait_for(std::chrono::seconds(5)) ==
              std::future_status::ready,
          "worker never reached routed input reading");
  worker.Submit(
      [&](const std::atomic<bool> &cancelled) {
        Require(cache->misses() == 0 && cache->payload_read_bytes() == 0,
                "cancelled prefill kept loading experts before queued request");
        cache->Release();
        const auto output = cache->Routed(
            mx::ones({1, 1, 128}), mx::zeros({1, 1, 1}, mx::uint32),
            mx::ones({1, 1, 1}), kPrefix, *checkpoint, &cancelled);
        Require(mx::all(mx::isfinite(output)).item<bool>(),
                "queued request could not reuse cache after cancellation");
        qwen3_asr::TranscriptionResult result;
        result.text = "next request finished";
        return result;
      },
      qwen3_asr::NewCancelFlag(),
      [&](std::optional<qwen3_asr::TranscriptionResult> result,
          std::exception_ptr error) {
        if (error) {
          next.set_exception(error);
        } else {
          next.set_value(*result);
        }
      });
  cancel->store(true);
  resume.set_value();
  Require(next_future.wait_for(std::chrono::seconds(5)) ==
              std::future_status::ready,
          "queued worker request did not finish after cancellation");
  const auto error = cancelled_future.get();
  Require(error != nullptr, "offloaded prefill ignored cancellation");
  bool rejected = false;
  try {
    std::rethrow_exception(error);
  } catch (const qwen3_asr::TranscriptionCancelled &) {
    rejected = true;
  }
  Require(rejected && next_future.get().text == "next request finished",
          "worker cancellation protocol or queue progression changed");
  std::cout
      << "cancelled offloaded prefill yielded worker before any expert read\n";
}

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
      {"text_config", {{"num_hidden_layers", 2}, {"num_experts", 17}}},
      {"quantization", {{"group_size", 64}, {"bits", bits}}}};
  std::ofstream(directory / "config.json") << config;
  if (bits == 4 && dtype == mx::float32) {
    VerifyCancelledWorker(directory, config, expert_bytes);
  } else {
  }
  for (const bool entire_layer : {false, true}) {
    const auto malformed_directory =
        directory / (entire_layer ? "missing-layer" : "missing-up-weight");
    std::filesystem::create_directories(malformed_directory);
    auto malformed_weights = weights;
    std::erase_if(malformed_weights, [&](const auto &named) {
      return entire_layer ? named.first.starts_with(other_prefix + ".")
                          : named.first == other_prefix + ".up_proj.weight";
    });
    mx::save_safetensors((malformed_directory / "model.safetensors").string(),
                         malformed_weights);
    bool rejected = false;
    try {
      drama::ExpertCache malformed(malformed_directory, config);
    } catch (const std::runtime_error &error) {
      rejected =
          std::string(error.what()) ==
          "missing routed expert tensor: " + other_prefix + ".up_proj.weight";
    }
    Require(rejected, "startup accepted an incomplete configured expert layer");
    std::cout << "startup rejected "
              << (entire_layer ? "missing-layer" : "missing-up-weight")
              << " bits=" << bits << " dtype=" << dtype << '\n';
  }
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
  const auto replacement_path = directory / "replacement.tmp.safetensors";
  std::filesystem::copy_file(path, replacement_path,
                             std::filesystem::copy_options::overwrite_existing);
  std::ifstream original(path, std::ios::binary);
  uint64_t header_bytes = 0;
  original.read(reinterpret_cast<char *>(&header_bytes), 8);
  std::string header(header_bytes, '\0');
  original.read(header.data(), header.size());
  const auto tensors = nlohmann::json::parse(header);
  std::fstream replacement(replacement_path,
                           std::ios::binary | std::ios::in | std::ios::out);
  for (const auto &[name, tensor] : tensors.items()) {
    if (drama::IsRoutedExpertWeight(name) && name.ends_with(".biases")) {
      const auto offsets = tensor.at("data_offsets").get<std::vector<size_t>>();
      const std::string payload(offsets[1] - offsets[0], '\0');
      replacement.seekp(8 + header_bytes + offsets[0]);
      replacement.write(payload.data(), payload.size());
    } else if (name == "tower.test") {
      const float value = 2.0f;
      replacement.seekp(8 + header_bytes +
                        tensor.at("data_offsets").at(0).get<size_t>());
      replacement.write(reinterpret_cast<const char *>(&value), sizeof(value));
    } else {
    }
  }
  replacement.close();
  Require(std::filesystem::file_size(path) ==
              std::filesystem::file_size(replacement_path),
          "replacement fixture changed shard size");
  std::ifstream patched(replacement_path, std::ios::binary);
  std::string patched_header(8 + header_bytes, '\0');
  patched.read(patched_header.data(), patched_header.size());
  Require(patched_header.substr(8) == header &&
              std::memcmp(patched_header.data(), &header_bytes, 8) == 0,
          "replacement fixture changed tensor layout");
  std::filesystem::rename(replacement_path, path);
  const omni_common::Checkpoint startup(replaced.TakeResidentWeights(), config);
  Require(startup.Weight("tower.test").item<float>() == 1.0f,
          "startup mixed resident and expert shard generations");
  Require(!startup.Has(kPrefix + ".up_proj.weight") &&
              !startup.Has("codec.test") && startup.Has("tower.test") &&
              startup.Has("backbone.model.layers.0.mlp.gate.weight") &&
              startup.Has("backbone.model.layers.0.mlp.shared_expert.weight"),
          "one-shot resident extraction changed weight selection");
  bool repeated_take_rejected = false;
  try {
    replaced.TakeResidentWeights();
  } catch (const std::runtime_error &error) {
    repeated_take_rejected = std::string(error.what()) ==
                             "resident weights already taken from expert cache";
  }
  Require(repeated_take_rejected,
          "repeated resident extraction did not report consumption");
  std::cout << "one-shot resident extraction rejected repeat bits=" << bits
            << " dtype=" << dtype << '\n';
  const omni_common::Checkpoint replacement_checkpoint(
      drama::LoadDrama3Weights(directory, false), config);
  Require(replacement_checkpoint.Weight("tower.test").item<float>() == 2.0f &&
              mx::max(mx::abs(mx::subtract(
                          replacement_checkpoint.Weight(kPrefix +
                                                        ".up_proj.biases"),
                          weights.at(kPrefix + ".up_proj.biases"))))
                      .item<float>() > 0.0f,
          "replacement did not change resident and expert payloads");
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
    std::ofstream(malformed_directory / "config.json") << nlohmann::json{
        {"text_config", {{"num_hidden_layers", 2}, {"num_experts", 17}}}};
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
