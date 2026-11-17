// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <array>
#include <cassert>

namespace qwen3_asr {

// Request-local measurements; a single slow round cannot disable the draft.
class DraftBackoff {
public:
  void RecordRound(int committed_token_count, double seconds) {
    assert(committed_token_count > 0 && seconds > 0.0);
    ++round_count_;
    if (round_count_ == 1) {
      // note (ZaneZhang-67): the first round also evaluates draft prefill.
      return;
    } else {
    }
    const int position = (round_count_ - 2) % (2 * kWindowRoundCount);
    Window &window = windows_[position / kWindowRoundCount];
    if (position % kWindowRoundCount == 0) {
      window = {};
    } else {
    }
    window.token_count += committed_token_count;
    window.seconds += seconds;
  }

  bool ShouldMeasurePlain() const {
    return round_count_ >= 1 + 2 * kWindowRoundCount &&
           (round_count_ - 1) % kWindowRoundCount == 0 &&
           (plain_seconds_per_token_ == 0.0 ||
            round_count_ >= 1 + kPlainBaselineMaxRoundCount ||
            SlowerThanPlain(plain_seconds_per_token_));
  }

  bool ObservePlain(double plain_seconds_per_token) {
    assert(ShouldMeasurePlain() && plain_seconds_per_token > 0.0);
    const bool should_disable = SlowerThanPlain(plain_seconds_per_token);
    plain_seconds_per_token_ = plain_seconds_per_token;
    // note (ZaneZhang-67): resuming speculation pays deferred probe context;
    // exclude that first round just as we exclude draft prefill.
    round_count_ = 0;
    windows_ = {};
    return should_disable;
  }

private:
  // note (ZaneZhang-67): two windows and a 5% margin filter timing spikes.
  static constexpr int kWindowRoundCount = 4;
  static constexpr double kBackoffCostRatio = 1.05;
  // note (ZaneZhang-67): refresh to expose losses hidden by a slow plain probe.
  static constexpr int kPlainBaselineMaxRoundCount = 64;
  bool SlowerThanPlain(double plain_seconds_per_token) const {
    return windows_[0].seconds > windows_[0].token_count *
                                     plain_seconds_per_token *
                                     kBackoffCostRatio &&
           windows_[1].seconds > windows_[1].token_count *
                                     plain_seconds_per_token *
                                     kBackoffCostRatio;
  }
  struct Window {
    int token_count = 0;
    double seconds = 0.0;
  };
  std::array<Window, 2> windows_{};
  int round_count_ = 0;
  double plain_seconds_per_token_ = 0.0;
};

} // namespace qwen3_asr
