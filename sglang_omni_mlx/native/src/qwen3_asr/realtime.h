// SPDX-License-Identifier: Apache-2.0
// Realtime transcription over one socket: segments refreshed on a cadence,
// then finalized.
#pragma once

#include <condition_variable>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "common/realtime_connection.h"
#include "common/transcription.h"
#include "common/worker.h"
#include "nlohmann/json.hpp"
#include "qwen3_asr/qwen3_asr_transcriber.h"

namespace qwen3_asr {

struct RealtimeSettings {
  int decode_interval_samples = 0;
  int first_decode_samples = 0;
  int max_segment_samples = 0;
};

RealtimeSettings MakeRealtimeSettings(int decode_interval_ms,
                                      int first_decode_ms,
                                      double max_segment_seconds);

// note (chenye): a preview's detected language and text are guesses, not
// a language hint or a committed transcript prefix for the final decode.
struct FinalPreviewGuess {
  std::string language;
  std::string text;
};

// What a session holds of a Final's audio: taken, and no longer extended by
// the session, or shared with its refreshes.
struct FinalReuse {
  std::shared_ptr<RefreshReuse> reuse;
  bool is_taken = false;
  std::optional<FinalPreviewGuess> auto_language_guess = std::nullopt;
};

// Qwen3-ASR manual-turn session: audio becomes one segment, cut every max
// segment length, with preview decodes on a cadence. A Final's prompt, sent as
// final_prompt, is prefilled in the background over the recording's complete
// windows, from the first preview on, so it does not delay it; with an empty
// one that starts with the second segment, the first's previews having the
// keys and values already. input_audio_buffer.final_start moves the Final's
// start on, when the audio before it was recognised otherwise.
class RealtimeSession : public omni_server::RealtimeConnection {
public:
  RealtimeSession(omni_server::TranscriptionWorker &worker,
                  const Qwen3ASRTranscriber &transcriber,
                  RealtimeSettings settings, Sender sender);

  bool Handle(const nlohmann::json &message) override;
  // What the session holds for a Final of its recording's start with this
  // prompt: the one prefilled for it, else what its refreshes keep; none is
  // null.
  //
  // note (chenye): only snapshot a preview guess when the caller opts in.
  FinalReuse TakeFinalReuse(const std::optional<std::string> &context,
                            bool guess_auto_language);

private:
  struct Segment {
    int segment_id = 0;
    long start_sample = 0;
    long next_refresh_sample = 0;
    std::optional<std::string> language;
    int decode_count = 0;
    std::string transcript;
    std::string last_text;
    // The last decode ran out of tokens, a loop the guard missed, likely on
    // the prefix: its text was not kept, and the next decode has no prefix.
    bool ran_out_of_tokens = false;
    // Where the audio of the decode the transcript came from ended.
    long transcript_end_sample = 0;
    // What the segment's refreshes reuse.
    std::shared_ptr<RefreshReuse> reuse;
  };

  void Append(const nlohmann::json &audio);
  // The Final starts at this sample of the session's audio, not at the first:
  // the windows prefilled for it are dropped, its prompt's head kept.
  void StartFinalAt(const nlohmann::json &start_sample);
  void StartSegment(long start_sample);
  long EndSample() const {
    return buffer_start_sample_ + static_cast<long>(samples_.size());
  }
  long LockedEndSample();
  std::vector<float> SegmentSamples(const Segment &segment,
                                    long end_sample) const;
  // note (chenye): Builds refresh or final options through end_sample and
  // increments the segment decode count.
  TranscriptionOptions DecodeOptions(Segment &segment, long end_sample);
  // note (chenye): Finals snapshot shared reuse on the worker; refreshes update
  // it.
  omni_server::Transcription Decode(std::vector<float> samples,
                                    TranscriptionOptions options,
                                    std::shared_ptr<RefreshReuse> reuse,
                                    bool snapshot_reuse) const;
  void ApplyResult(Segment &segment,
                   const omni_common::TranscriptionResult &result,
                   long end_sample);
  void MaybeStartRefresh();
  void MaybeStartFinalPrefill();
  void FinalizeThrough(long end_sample);

  omni_server::TranscriptionWorker &worker_;
  const Qwen3ASRTranscriber &transcriber_;
  const RealtimeSettings settings_;

  // Session state; refreshes finish on the worker thread.
  std::mutex mutex_;
  std::condition_variable refresh_done_;
  std::optional<std::string> language_;
  // A Final with the context final_context_ is to be prefilled, until one is
  // taken.
  bool should_prefill_final_ = false;
  // The context the Final will have, and what is prefilled for it.
  std::string final_context_;
  std::shared_ptr<RefreshReuse> final_reuse_;
  // A preview with text was sent; the Final's prefill waits for it.
  bool has_preview_ = false;
  bool is_prefilling_final_ = false;
  std::vector<float> samples_;
  long buffer_start_sample_ = 0;
  // The audio from where the Final starts, what it is prefilled from;
  // samples_ drops what is committed.
  std::vector<float> recording_samples_;
  long final_start_sample_ = 0;
  // The audio was longer than is kept, so none is prefilled from here on.
  bool is_recording_cut_off_ = false;
  std::optional<Segment> segment_;
  // The first segment's reuse.
  std::shared_ptr<RefreshReuse> recording_reuse_;
  int next_segment_id_ = 0;
  std::vector<std::pair<int, std::string>> committed_;
  bool refreshing_ = false;
  bool finalizing_ = false;
};

} // namespace qwen3_asr
