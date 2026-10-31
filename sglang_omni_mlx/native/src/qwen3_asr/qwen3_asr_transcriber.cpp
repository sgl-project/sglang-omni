// SPDX-License-Identifier: Apache-2.0
#include "qwen3_asr/qwen3_asr_transcriber.h"
#include "common/text.h"

#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <deque>
#include <map>
#include <mutex>
#include <thread>
#include <unordered_set>

#include "mlx/backend/metal/device.h"

namespace qwen3_asr {

using omni_common::IsTokenLoop;
using omni_common::IsUnicodeSpace;

namespace mx = mlx::core;

namespace {

constexpr const char *kAsrTextMarker = "<asr_text>";
constexpr const char *kLanguageMarker = "language";
constexpr const char *kAudioPad = "<|audio_pad|>";
constexpr int kOutputTokensPerAudioSecond = 10;
constexpr int kMinDefaultOutputTokens = 128;

// Note (XXXXRT666): after even 0.1 s idle the CPU cluster's clock stays low
// until about 50 ms of load, and a transcription, which mostly waits on the
// GPU, never loads it enough: graph building and kernel encoding then lag the
// GPU throughout, up to a fifth slower end to end on the M4 Pro and M4 Max.
// A thread spinning beside the transcription keeps the cluster's clock up.
class ClockKeeper {
public:
  ClockKeeper()
      : thread_([this] {
          while (!stopped_.load(std::memory_order_relaxed)) {
          }
        }) {}
  ~ClockKeeper() {
    stopped_.store(true, std::memory_order_relaxed);
    thread_.join();
  }
  ClockKeeper(const ClockKeeper &) = delete;
  ClockKeeper &operator=(const ClockKeeper &) = delete;

private:
  std::atomic<bool> stopped_{false};
  std::thread thread_;
};

// Note (XXXXRT666): the GPU also clocks down in the gaps between a
// transcription's command buffers, the more after seconds idle. A
// one-simdgroup kernel kept in flight on a queue of its own keeps it busy
// beside the transcription; with the residency heartbeat, a 0.6B transcription
// after 5-30 s idle is 13-16% faster on the M4 Pro and M5 Max.
class GpuKeeper {
public:
  GpuKeeper()
      : buffers_in_flight_(std::make_shared<std::atomic<int>>(0)),
        thread_([this] {
          while (!stopped_.load(std::memory_order_relaxed)) {
            if (buffers_in_flight_->load() < kBuffersInFlight) {
              Commit(Shared(), Shared().iterations_per_buffer,
                     buffers_in_flight_)
                  ->release();
            } else {
              std::this_thread::sleep_for(kPollInterval);
            }
          }
        }) {}
  // Buffers still in flight finish on their own.
  ~GpuKeeper() {
    stopped_.store(true, std::memory_order_relaxed);
    thread_.join();
  }
  GpuKeeper(const GpuKeeper &) = delete;
  GpuKeeper &operator=(const GpuKeeper &) = delete;

  // Compiles and times the kernel, once per process.
  static void Prepare() { Shared(); }

private:
  static constexpr int kBuffersInFlight = 2;
  static constexpr double kBufferSeconds = 5e-4;
  static constexpr uint32_t kProbeIterations = 200000;
  static constexpr int kProbeCount = 3;
  static constexpr std::chrono::microseconds kPollInterval{50};
  static constexpr const char *kSource = R"(
#include <metal_stdlib>
using namespace metal;
kernel void keep_warm(device uint *sink [[buffer(0)]],
                      constant uint &iterations [[buffer(1)]],
                      uint thread_index [[thread_position_in_grid]]) {
  uint x = thread_index;
  for (uint i = 0; i < iterations; ++i) {
    x = x * 1664525u + 1013904223u;
  }
  if (x == 0xffffffffu) {
    sink[0] = x;
  } else {
  }
}
)";

  struct Pipeline {
    MTL::ComputePipelineState *state = nullptr;
    MTL::CommandQueue *queue = nullptr;
    MTL::Buffer *sink = nullptr;
    uint32_t iterations_per_buffer = 0;
  };

  // The process's one pipeline and queue, never released.
  static const Pipeline &Shared() {
    static const Pipeline shared = [] {
      MTL::Device *device = mx::metal::device(mx::Device::gpu).mtl_device();
      NS::AutoreleasePool *pool = NS::AutoreleasePool::alloc()->init();
      NS::Error *error = nullptr;
      MTL::Library *library = device->newLibrary(
          NS::String::string(kSource, NS::UTF8StringEncoding), nullptr, &error);
      if (library == nullptr) {
        throw std::runtime_error("the GPU keeper's kernel failed to compile");
      } else {
      }
      MTL::Function *function = library->newFunction(
          NS::String::string("keep_warm", NS::UTF8StringEncoding));
      Pipeline pipeline;
      pipeline.state = device->newComputePipelineState(function, &error);
      function->release();
      library->release();
      pipeline.queue = device->newCommandQueue();
      pipeline.sink =
          device->newBuffer(sizeof(uint32_t), MTL::ResourceStorageModeShared);
      pool->release();
      std::vector<double> iterations_per_second;
      for (int i = 0; i < kProbeCount; ++i) {
        MTL::CommandBuffer *probe = Commit(pipeline, kProbeIterations, nullptr);
        probe->waitUntilCompleted();
        iterations_per_second.push_back(
            kProbeIterations / (probe->GPUEndTime() - probe->GPUStartTime()));
        probe->release();
      }
      std::sort(iterations_per_second.begin(), iterations_per_second.end());
      pipeline.iterations_per_buffer = static_cast<uint32_t>(
          iterations_per_second[kProbeCount / 2] * kBufferSeconds);
      return pipeline;
    }();
    return shared;
  }

  // One retained command buffer of the kernel, committed; with a counter, it
  // counts itself in flight until it completes.
  static MTL::CommandBuffer *
  Commit(const Pipeline &pipeline, uint32_t iterations,
         const std::shared_ptr<std::atomic<int>> &buffers_in_flight) {
    NS::AutoreleasePool *pool = NS::AutoreleasePool::alloc()->init();
    MTL::CommandBuffer *buffer = pipeline.queue->commandBuffer();
    MTL::ComputeCommandEncoder *encoder = buffer->computeCommandEncoder();
    encoder->setComputePipelineState(pipeline.state);
    encoder->setBuffer(pipeline.sink, 0, 0);
    encoder->setBytes(&iterations, sizeof(iterations), 1);
    encoder->dispatchThreadgroups(MTL::Size(1, 1, 1), MTL::Size(32, 1, 1));
    encoder->endEncoding();
    if (buffers_in_flight != nullptr) {
      buffers_in_flight->fetch_add(1);
      buffer->addCompletedHandler([buffers_in_flight](MTL::CommandBuffer *) {
        buffers_in_flight->fetch_sub(1);
      });
    } else {
    }
    buffer->retain();
    buffer->commit();
    pool->release();
    return buffer;
  }

  const std::shared_ptr<std::atomic<int>> buffers_in_flight_;
  std::atomic<bool> stopped_{false};
  std::thread thread_;
};

const std::map<std::string, std::string> &LanguageCodeToName() {
  static const std::map<std::string, std::string> table = {
      {"ar", "Arabic"},     {"yue", "Cantonese"}, {"zh", "Chinese"},
      {"cs", "Czech"},      {"da", "Danish"},     {"nl", "Dutch"},
      {"en", "English"},    {"fil", "Filipino"},  {"fi", "Finnish"},
      {"fr", "French"},     {"de", "German"},     {"el", "Greek"},
      {"hi", "Hindi"},      {"hu", "Hungarian"},  {"id", "Indonesian"},
      {"it", "Italian"},    {"ja", "Japanese"},   {"ko", "Korean"},
      {"mk", "Macedonian"}, {"ms", "Malay"},      {"fa", "Persian"},
      {"pl", "Polish"},     {"pt", "Portuguese"}, {"ro", "Romanian"},
      {"ru", "Russian"},    {"es", "Spanish"},    {"sv", "Swedish"},
      {"th", "Thai"},       {"tr", "Turkish"},    {"vi", "Vietnamese"},
  };
  return table;
}

std::string Strip(const std::string &text) {
  const auto is_space = [](unsigned char c) { return std::isspace(c) != 0; };
  size_t begin = 0;
  size_t end = text.size();
  while (begin < end && is_space(text[begin]))
    ++begin;
  while (end > begin && is_space(text[end - 1]))
    --end;
  return text.substr(begin, end - begin);
}

std::string Lowercase(std::string text) {
  std::transform(text.begin(), text.end(), text.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });
  return text;
}

std::optional<size_t> FindSubsequence(const std::vector<int> &values,
                                      const std::vector<int> &pattern) {
  if (pattern.size() > values.size()) {
    return std::nullopt;
  } else {
  }
  for (size_t start = 0; start + pattern.size() <= values.size(); ++start) {
    if (std::equal(pattern.begin(), pattern.end(), values.begin() + start)) {
      return start;
    } else {
    }
  }
  return std::nullopt;
}

} // namespace

const char *FinishReasonName(FinishReason reason) {
  return reason == FinishReason::kStop ? "stop" : "length";
}

nlohmann::ordered_json
SpeakerSegmentsJson(const std::vector<SpeakerSegment> &segments) {
  nlohmann::ordered_json array = nlohmann::ordered_json::array();
  for (const SpeakerSegment &segment : segments) {
    nlohmann::ordered_json json = {{"start", segment.start_seconds},
                                   {"end", segment.end_seconds},
                                   {"speaker", segment.speaker},
                                   {"text", segment.text}};
    for (const auto &[name, value] : {std::pair{"language", &segment.language},
                                      std::pair{"emotion", &segment.emotion},
                                      std::pair{"event", &segment.event}}) {
      if (value->has_value()) {
        json[name] = **value;
      } else {
      }
    }
    array.push_back(std::move(json));
  }
  return array;
}

std::optional<std::string> NormalizeLanguage(const std::string &language) {
  const std::string stripped = Strip(language);
  const std::string normalized = Lowercase(stripped);
  if (stripped.empty()) {
    return std::nullopt;
  } else if (normalized == "cn" || normalized.rfind("zh-", 0) == 0 ||
             normalized.rfind("zh_", 0) == 0) {
    return std::string("Chinese");
  } else {
  }
  const auto code = LanguageCodeToName().find(normalized);
  if (code != LanguageCodeToName().end()) {
    return code->second;
  } else {
  }
  for (const auto &[unused_code, name] : LanguageCodeToName()) {
    if (Lowercase(name) == normalized) {
      return name;
    } else {
    }
  }
  return stripped;
}

// Note (XXXXRT666): macOS lets the weights' residency lapse after seconds of
// GPU idle despite a standing request, and the first transcription after idle
// pays 3-9% to restore it on the M4 Pro and M5 Max. Requesting residency again
// every 500 ms keeps them resident.
class ResidencyHeartbeat {
public:
  explicit ResidencyHeartbeat(const omni_common::WeightMap &weights) {
    MTL::Device *device = mx::metal::device(mx::Device::gpu).mtl_device();
    NS::AutoreleasePool *pool = NS::AutoreleasePool::alloc()->init();
    MTL::ResidencySetDescriptor *descriptor =
        MTL::ResidencySetDescriptor::alloc()->init();
    descriptor->setInitialCapacity(weights.size());
    NS::Error *error = nullptr;
    residency_set_ = device->newResidencySet(descriptor, &error);
    descriptor->release();
    if (residency_set_ == nullptr) {
      throw std::runtime_error("cannot create the weights' residency set");
    } else {
    }
    std::unordered_set<const void *> buffers;
    for (const auto &[name, weight] : weights) {
      const void *buffer = weight.buffer().ptr();
      if (buffers.insert(buffer).second) {
        residency_set_->addAllocation(
            static_cast<const MTL::Allocation *>(buffer));
      } else {
      }
    }
    residency_set_->commit();
    pool->release();
    thread_ = std::thread([this] {
      std::unique_lock<std::mutex> lock(mutex_);
      do {
        residency_set_->requestResidency();
      } while (!stop_requested_.wait_for(lock, kRequestInterval,
                                         [this] { return stopped_; }));
    });
  }
  ~ResidencyHeartbeat() {
    {
      const std::lock_guard<std::mutex> lock(mutex_);
      stopped_ = true;
    }
    stop_requested_.notify_one();
    thread_.join();
    residency_set_->endResidency();
    residency_set_->release();
  }
  ResidencyHeartbeat(const ResidencyHeartbeat &) = delete;
  ResidencyHeartbeat &operator=(const ResidencyHeartbeat &) = delete;

private:
  static constexpr std::chrono::milliseconds kRequestInterval{500};

  MTL::ResidencySet *residency_set_ = nullptr;
  std::mutex mutex_;
  std::condition_variable stop_requested_;
  bool stopped_ = false;
  std::thread thread_;
};

Qwen3ASRTranscriber::Qwen3ASRTranscriber(
    const std::filesystem::path &model_directory)
    : model_(model_directory), tokenizer_(model_directory),
      audio_pad_id_(tokenizer_.AddedTokenId(kAudioPad)),
      end_of_text_id_(tokenizer_.AddedTokenId("<|endoftext|>")),
      im_end_id_(tokenizer_.AddedTokenId("<|im_end|>")),
      asr_text_ids_(tokenizer_.Encode(kAsrTextMarker)),
      language_ids_(tokenizer_.Encode(kLanguageMarker)),
      residency_heartbeat_(
          std::make_unique<ResidencyHeartbeat>(model_.Weights())) {
  GpuKeeper::Prepare();
}

Qwen3ASRTranscriber::~Qwen3ASRTranscriber() = default;

std::vector<int>
Qwen3ASRTranscriber::PromptIds(int audio_token_count,
                               const TranscriptionOptions &options) const {
  std::string prompt = "<|im_start|>system\n" +
                       Strip(options.context.value_or("")) +
                       "<|im_end|>\n<|im_start|>user\n<|audio_start|>";
  for (int i = 0; i < audio_token_count; ++i)
    prompt += kAudioPad;
  prompt += "<|audio_end|><|im_end|>\n<|im_start|>assistant\n";
  const std::optional<std::string> language =
      NormalizeLanguage(options.language.value_or(""));
  if (language.has_value()) {
    prompt += "language " + *language + kAsrTextMarker;
  } else {
  }
  std::vector<int> ids = tokenizer_.Encode(prompt);
  ids.insert(ids.end(), options.prefix_token_ids.begin(),
             options.prefix_token_ids.end());
  return ids;
}

std::pair<std::vector<int>, std::string>
Qwen3ASRTranscriber::RetainedPrefix(const std::string &text,
                                    int rollback_token_count) const {
  const std::vector<int> token_ids = tokenizer_.Encode(text);
  const size_t keep =
      token_ids.size() > static_cast<size_t>(rollback_token_count)
          ? token_ids.size() - rollback_token_count
          : 0;
  std::vector<int> retained(token_ids.begin(), token_ids.begin() + keep);
  static const std::string replacement = "\xEF\xBF\xBD";
  while (!retained.empty()) {
    const std::string decoded = tokenizer_.Decode(retained, false);
    if (decoded.size() >= replacement.size() &&
        decoded.compare(decoded.size() - replacement.size(), replacement.size(),
                        replacement) == 0) {
      retained.pop_back();
    } else {
      return {retained, decoded};
    }
  }
  return {{}, ""};
}

TranscriptionResult Qwen3ASRTranscriber::Transcribe(
    const std::vector<float> &samples, const TranscriptionOptions &options,
    const std::atomic<bool> &cancel, TranscriptionProfile *profile) const {
  if (cancel.load()) {
    throw TranscriptionCancelled();
  } else {
  }
  const ClockKeeper clock_keeper;
  const GpuKeeper gpu_keeper;
  auto stage_started = std::chrono::steady_clock::now();
  const auto record_stage = [&](const mx::array &output,
                                double TranscriptionProfile::*seconds) {
    if (profile != nullptr) {
      mx::eval(output);
      const auto finished = std::chrono::steady_clock::now();
      profile->*seconds =
          std::chrono::duration<double>(finished - stage_started).count();
      stage_started = finished;
    } else {
    }
  };
  const mx::array mel = LogMel(samples, options.layout);
  record_stage(mel, &TranscriptionProfile::mel_seconds);
  const int audio_token_count = TokenCount(mel.shape(-1), options.layout);
  const mx::array audio_features = model_.EncodeAudio(mel, options.layout);
  record_stage(audio_features, &TranscriptionProfile::encoder_seconds);
  const std::vector<int> prompt_ids = PromptIds(audio_token_count, options);
  const mx::array input_ids(
      prompt_ids.data(), {1, static_cast<int>(prompt_ids.size())}, mx::int32);
  mx::array embeddings = model_.EmbedTokens(input_ids);
  const int audio_start = static_cast<int>(
      std::find(prompt_ids.begin(), prompt_ids.end(), audio_pad_id_) -
      prompt_ids.begin());
  // Note (Jiaxin Deng): the Swift layout can reserve more placeholders than
  // encoder rows; those keep the audio_pad embedding, surplus rows are dropped.
  const int filled = std::min(audio_token_count, audio_features.shape(0));
  const int hidden = embeddings.shape(2);
  embeddings = mx::slice_update(
      embeddings,
      mx::expand_dims(mx::astype(mx::slice(audio_features, {0, 0},
                                           {filled, audio_features.shape(1)}),
                                 embeddings.dtype()),
                      0),
      {0, audio_start, 0}, {1, audio_start + filled, hidden});

  // Note (Jiaxin Deng): an unset or zero budget means the default.
  const int max_new_tokens =
      options.max_new_tokens.value_or(0) != 0
          ? *options.max_new_tokens
          : std::max(
                kMinDefaultOutputTokens,
                static_cast<int>(std::ceil(static_cast<double>(samples.size()) /
                                           omni_common::kSampleRate *
                                           kOutputTokensPerAudioSecond)));
  const auto is_stop = [&](int token_id) {
    return token_id == im_end_id_ ||
           (options.stop_at_end_of_text && token_id == end_of_text_id_);
  };
  // Note (XXXXRT666): with no language the reply opens with "language", the
  // language name and <asr_text>. The two template tokens ride along as extra
  // rows of the forward before them and count only where greedy decoding
  // picks them; a miss rewinds the caches past every row fed after it.
  std::deque<int> guesses;
  if (!NormalizeLanguage(options.language.value_or("")).has_value() &&
      options.prefix_token_ids.empty() && language_ids_.size() == 1 &&
      asr_text_ids_.size() == 1) {
    guesses = {language_ids_[0], asr_text_ids_[0]};
  } else {
  }
  std::vector<Qwen3ASRKVCache> caches = model_.NewCaches();
  // One forward: the greedy token after each of its last fed.size() + 1 rows,
  // fed holding the guesses that filled the rows after the first.
  struct Step {
    mx::array tokens;
    std::vector<int> fed;
  };
  const auto forward = [&](const mx::array &inputs, std::vector<int> fed) {
    mx::array tokens = GreedyTokens(
        model_.Decode(inputs, caches, static_cast<int>(fed.size()) + 1));
    mx::async_eval({tokens});
    return Step{tokens, std::move(fed)};
  };
  // The forward after this step's, assuming every guess in it holds.
  const auto queue_after = [&](const Step &step) {
    const int rows = static_cast<int>(step.fed.size()) + 1;
    mx::array ids =
        mx::reshape(mx::slice(step.tokens, {rows - 1}, {rows}), {1, 1});
    std::vector<int> fed;
    if (!guesses.empty()) {
      const uint32_t guess = guesses.front();
      guesses.pop_front();
      ids = mx::concatenate({ids, mx::array(&guess, {1, 1}, mx::uint32)}, 1);
      fed.push_back(static_cast<int>(guess));
    } else {
    }
    return forward(model_.EmbedTokens(ids), std::move(fed));
  };
  std::vector<int> prefill_fed;
  if (!guesses.empty()) {
    const int guess = guesses.front();
    guesses.pop_front();
    embeddings = mx::concatenate(
        {embeddings, model_.EmbedTokens(mx::array(&guess, {1, 1}, mx::int32))},
        1);
    prefill_fed.push_back(guess);
  } else {
  }
  Step step = forward(embeddings, std::move(prefill_fed));
  record_stage(step.tokens, &TranscriptionProfile::prefill_seconds);
  std::vector<int> output_ids;
  FinishReason finish_reason = FinishReason::kLength;
  // Whether decoding ends at this token, which is kept unless over budget.
  const auto emit = [&](int token_id) {
    if (static_cast<int>(output_ids.size()) >= max_new_tokens) {
      return true;
    } else {
    }
    output_ids.push_back(token_id);
    if (is_stop(token_id) ||
        (options.stop_on_token_loop && IsTokenLoop(output_ids))) {
      finish_reason = FinishReason::kStop;
      return true;
    } else {
      return false;
    }
  };
  bool finished = false;
  while (!finished) {
    if (cancel.load()) {
      throw TranscriptionCancelled();
    } else {
    }
    // Note (Jiaxin Deng): queue the next step before reading this token, so
    // the GPU decodes while the stop rules are checked.
    Step next = queue_after(step);
    mx::eval(step.tokens);
    const uint32_t *tokens = step.tokens.data<uint32_t>();
    const int fed_count = static_cast<int>(step.fed.size());
    int row = 0;
    while (true) {
      finished = emit(static_cast<int>(tokens[row]));
      if (finished || row == fed_count ||
          static_cast<int>(tokens[row]) != step.fed[row]) {
        break;
      } else {
        ++row;
      }
    }
    if (!finished && row < fed_count) {
      for (Qwen3ASRKVCache &cache : caches) {
        cache.Rewind(fed_count - row + static_cast<int>(next.fed.size()) + 1);
      }
      guesses.clear();
      const uint32_t token = tokens[row];
      next = forward(model_.EmbedTokens(mx::array(&token, {1, 1}, mx::uint32)),
                     {});
    } else {
    }
    step = std::move(next);
  }
  record_stage(step.tokens, &TranscriptionProfile::decode_seconds);
  if (profile != nullptr) {
    profile->prompt_tokens = static_cast<int>(prompt_ids.size());
  } else {
  }
  auto [text, language] = SplitOutput(output_ids, options);
  const bool ended_on_stop = !output_ids.empty() && is_stop(output_ids.back());
  return {options.prefix_text + text,
          language,
          static_cast<int>(output_ids.size()) - (ended_on_stop ? 1 : 0),
          finish_reason,
          {}};
}

std::pair<std::string, std::optional<std::string>>
Qwen3ASRTranscriber::SplitOutput(const std::vector<int> &output_ids,
                                 const TranscriptionOptions &options) const {
  const std::optional<size_t> marker =
      FindSubsequence(output_ids, asr_text_ids_);
  std::optional<std::string> detected;
  if (!options.language.has_value() && marker.has_value()) {
    const std::string prefix = Strip(tokenizer_.Decode(
        std::vector<int>(output_ids.begin(), output_ids.begin() + *marker),
        true));
    const size_t separator = prefix.find(' ');
    if (separator != std::string::npos &&
        Lowercase(prefix.substr(0, separator)) == "language") {
      const std::string value = Strip(prefix.substr(separator + 1));
      if (!value.empty() && Lowercase(value) != "none") {
        detected = value;
      } else {
      }
    } else if (!prefix.empty()) {
      detected = prefix;
    } else {
    }
  } else {
  }
  const std::vector<int> transcript_ids =
      marker.has_value() ? std::vector<int>(output_ids.begin() + *marker +
                                                asr_text_ids_.size(),
                                            output_ids.end())
                         : output_ids;
  const std::string text = tokenizer_.Decode(transcript_ids, true);
  std::optional<std::string> language =
      NormalizeLanguage(options.language.value_or(""));
  if (!language.has_value() && detected.has_value()) {
    language = NormalizeLanguage(*detected);
  } else {
  }
  return {text, language};
}

} // namespace qwen3_asr
