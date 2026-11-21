// SPDX-License-Identifier: Apache-2.0
#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <new>
#include <string>
#include <vector>

#include "common/audio.h"
#include "common/transcription.h"
#include "moss/moss_realtime.h"

bool Check(bool condition, const char *expression) {
  if (condition) {
    return true;
  }
  std::fprintf(stderr, "check failed: %s\n", expression);
  return false;
}

// Records every allocation's size on the calling thread while armed, so the
// test can tell whether an append allocated a preview timing object without
// reading the session's private state. thread_local keeps the worker thread's
// own allocations out of the counts and the measurement windows race-free.
thread_local bool t_record_allocs = false;
thread_local size_t t_alloc_count = 0;
constexpr size_t kMaxRecordedAllocs = 8192;
thread_local size_t t_alloc_sizes[kMaxRecordedAllocs];

void RecordAlloc(std::size_t size) {
  if (!t_record_allocs) {
    return;
  }
  if (t_alloc_count < kMaxRecordedAllocs) {
    t_alloc_sizes[t_alloc_count] = size;
  }
  ++t_alloc_count;
}

void *operator new(std::size_t size) {
  RecordAlloc(size);
  void *pointer = std::malloc(size == 0 ? 1 : size);
  if (pointer == nullptr) {
    throw std::bad_alloc();
  }
  return pointer;
}

void *operator new[](std::size_t size) { return operator new(size); }
void operator delete(void *pointer) noexcept { std::free(pointer); }
void operator delete[](void *pointer) noexcept { std::free(pointer); }
void operator delete(void *pointer, std::size_t) noexcept {
  std::free(pointer);
}
void operator delete[](void *pointer, std::size_t) noexcept {
  std::free(pointer);
}

struct AllocWatch {
  AllocWatch() {
    t_alloc_count = 0;
    t_record_allocs = true;
  }
  ~AllocWatch() { t_record_allocs = false; }
};

size_t RecordedAllocCountOfSize(std::size_t size) {
  const size_t count = std::min(t_alloc_count, kMaxRecordedAllocs);
  size_t matches = 0;
  for (size_t i = 0; i < count; ++i) {
    if (t_alloc_sizes[i] == size) {
      ++matches;
    }
  }
  return matches;
}

// Silent PCM16 audio as the base64 an append event carries.
std::string ZeroAudioBase64(int sample_count) {
  const size_t byte_count = static_cast<size_t>(sample_count) * 2;
  std::string encoded;
  encoded.reserve((byte_count + 2) / 3 * 4);
  const size_t whole_triples = byte_count / 3 * 3;
  for (size_t i = 0; i < whole_triples; i += 3) {
    encoded += "AAAA";
  }
  if (byte_count - whole_triples == 1) {
    encoded += "AA==";
  } else if (byte_count - whole_triples == 2) {
    encoded += "AAA=";
  }
  return encoded;
}

int main() {
  omni_common::MossPreviewTiming timing;
  timing.window_id = 7;
  timing.kind = "preview";
  timing.audio_sample_count = 40000;
  timing.append_at_ms = 1000;
  timing.decision_at_ms = 1002;
  timing.submit_at_ms = 1004;
  timing.transcribe_started_at_ms = 1200;
  timing.transcribe_at_ms = 1250;
  timing.publish_at_ms = 1252;
  timing.min_preview_samples = 20000;
  timing.verify_round_count = 4;
  timing.proposed_draft_token_count = 28;
  timing.accepted_draft_token_count = 17;

  const auto json = omni_common::MossPreviewTimingJson(timing);
  const std::vector<std::string> schema = {"event",
                                           "window_id",
                                           "kind",
                                           "audio_sample_count",
                                           "append_at_ms",
                                           "decision_at_ms",
                                           "submit_at_ms",
                                           "transcribe_started_at_ms",
                                           "transcribe_at_ms",
                                           "publish_at_ms",
                                           "min_preview_samples",
                                           "verify_round_count",
                                           "proposed_draft_token_count",
                                           "accepted_draft_token_count"};
  if (!Check(json.size() == schema.size(), "exact schema size") ||
      !Check(json["event"] == "moss_preview_timing", "event") ||
      !Check(json["window_id"] == 7, "window_id") ||
      !Check(json["kind"] == "preview", "kind") ||
      !Check(json["audio_sample_count"] == 40000, "audio_sample_count") ||
      !Check(json["append_at_ms"] == 1000, "append_at_ms") ||
      !Check(json["decision_at_ms"] == 1002, "decision_at_ms") ||
      !Check(json["submit_at_ms"] == 1004, "submit_at_ms") ||
      !Check(json["transcribe_started_at_ms"] == 1200,
             "transcribe_started_at_ms") ||
      !Check(json["transcribe_at_ms"] == 1250, "transcribe_at_ms") ||
      !Check(json["publish_at_ms"] == 1252, "publish_at_ms") ||
      !Check(json["min_preview_samples"] == 20000, "min_preview_samples") ||
      !Check(json["verify_round_count"] == 4, "verify_round_count") ||
      !Check(json["proposed_draft_token_count"] == 28,
             "proposed_draft_token_count") ||
      !Check(json["accepted_draft_token_count"] == 17,
             "accepted_draft_token_count")) {
    return 1;
  }
  for (const std::string &key : schema) {
    if (!Check(json.contains(key), key.c_str())) {
      return 1;
    }
  }
  for (const std::string &key :
       std::vector<std::string>{"text", "prompt", "speaker", "segments"}) {
    if (!Check(!json.contains(key), key.c_str())) {
      return 1;
    }
  }

  // Regression: the timing object belongs to a decode, so an append that
  // produces none must not allocate one. Declared so teardown joins the
  // worker before anything its job references is destroyed.
  moss::MossTranscriber transcriber("", "", 0);
  std::vector<std::string> events;
  omni_server::TranscriptionWorker worker([] {}, [] {}, 1);
  const auto session = std::make_shared<moss::MossRealtimeSession>(
      worker, transcriber, [&events](const std::string &event) {
        events.push_back(event);
        return true;
      });

  std::size_t timing_alloc_size = 0;
  {
    AllocWatch watch;
    const auto probe = std::make_shared<omni_common::MossPreviewTiming>();
    (void)probe;
    if (!Check(t_alloc_count == 1, "timing allocation is observable")) {
      return 1;
    }
    timing_alloc_size = t_alloc_sizes[0];
  }

  // 30 samples stay below the 1.25 s preview minimum, so nothing decodes.
  const nlohmann::json short_append = {{"type", "input_audio_buffer.append"},
                                       {"audio", ZeroAudioBase64(30)}};
  {
    AllocWatch watch;
    session->Handle(short_append);
    if (!Check(RecordedAllocCountOfSize(timing_alloc_size) == 0,
               "append without a decode allocates no timing")) {
      return 1;
    }
  }

  // 1.5 s more brings the pending audio past the preview minimum, so this
  // append decodes and allocates the timing; the double's gate holds that
  // decode in flight, so the identical append right after decodes nothing and
  // shows the incidental allocations of the same work. Their counts of the
  // timing block's size must differ by exactly one.
  const nlohmann::json longer_append = {
      {"type", "input_audio_buffer.append"},
      {"audio", ZeroAudioBase64(omni_common::kSampleRate * 3 / 2)}};
  size_t decode_allocs = 0;
  size_t held_allocs = 0;
  moss::g_transcribe_gate.store(false, std::memory_order_release);
  {
    AllocWatch watch;
    session->Handle(longer_append);
    decode_allocs = RecordedAllocCountOfSize(timing_alloc_size);
  }
  {
    AllocWatch watch;
    session->Handle(longer_append);
    held_allocs = RecordedAllocCountOfSize(timing_alloc_size);
  }
  moss::g_transcribe_gate.store(true, std::memory_order_release);
  if (!Check(decode_allocs == held_allocs + 1,
             "append with a decode allocates the timing exactly once")) {
    return 1;
  }
  return 0;
}
