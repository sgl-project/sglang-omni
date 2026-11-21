// SPDX-License-Identifier: Apache-2.0
#include "qwen3_asr/qwen3_asr_transcriber.h"

namespace qwen3_asr {

omni_server::BatchRunner BatchableTranscriptionRunner() {
  return {[](const omni_server::Transcription &transcription) {
            return transcription.target<BatchableTranscription>() != nullptr;
          },
          [](const std::vector<omni_server::BatchItem> &items,
             const omni_server::BatchCompletion &finish) {
            std::vector<BatchRequest> requests;
            std::vector<const BatchableTranscription *> batchables;
            const Qwen3ASRTranscriber *transcriber = nullptr;
            for (const omni_server::BatchItem &item : items) {
              const auto *batchable =
                  item.transcription->target<BatchableTranscription>();
              transcriber = batchable->transcriber;
              batchables.push_back(batchable);
              requests.push_back(
                  {&batchable->samples, &batchable->options, item.cancel});
            }
            const auto started = NowRealtimeMilliseconds();
            for (auto *batchable : batchables) {
              if (batchable->timing) {
                batchable->timing->started_at_ms = started;
              } else {
              }
            }
            std::vector<bool> logged(batchables.size(), false);
            const auto log_outcome = [&](size_t i, bool succeeded) {
              const auto &timing = batchables[i]->timing;
              if (!timing || logged[i]) {
                return;
              }
              timing->finished_at_ms = NowRealtimeMilliseconds();
              timing->succeeded = succeeded;
              LogRealtimeDecodeTiming(*timing);
              logged[i] = true;
            };
            try {
              transcriber->TranscribeBatch(
                  requests, [&](size_t i, BatchOutcome outcome) {
                    log_outcome(i, outcome.result.has_value());
                    finish(i, std::move(outcome.result), outcome.error);
                  });
            } catch (...) {
              for (size_t i = 0; i < batchables.size(); ++i) {
                log_outcome(i, false);
              }
              throw;
            }
          }};
}

} // namespace qwen3_asr
