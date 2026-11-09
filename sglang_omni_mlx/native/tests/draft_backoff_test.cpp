// SPDX-License-Identifier: Apache-2.0
#include "qwen3_asr/draft_backoff.h"

#include <cassert>

using qwen3_asr::DraftBackoff;

void RecordWindow(DraftBackoff &backoff, int token_count, double seconds) {
  for (int i = 0; i < 4; ++i) {
    backoff.RecordRound(token_count, seconds);
  }
}

int main() {
  {
    DraftBackoff backoff;
    assert(!backoff.ShouldMeasurePlain());
    backoff.RecordRound(1, 100.0);
    RecordWindow(backoff, 4, 2.0);
    assert(!backoff.ShouldMeasurePlain());
    RecordWindow(backoff, 4, 2.0);
    assert(backoff.ShouldMeasurePlain());
    assert(!backoff.ObservePlain(1.0));

    // Deferred context on the first resumed round is not a losing window.
    backoff.RecordRound(1, 100.0);
    RecordWindow(backoff, 4, 2.0);
    RecordWindow(backoff, 4, 2.0);
    assert(!backoff.ShouldMeasurePlain());
    for (int i = 0; i < 32; ++i) {
      backoff.RecordRound(4, 2.0);
      assert(!backoff.ShouldMeasurePlain());
    }
    RecordWindow(backoff, 1, 2.0);
    assert(!backoff.ShouldMeasurePlain());
    RecordWindow(backoff, 1, 2.0);
    assert(backoff.ShouldMeasurePlain());
    // A slower current plain path invalidates the old timing baseline.
    assert(!backoff.ObservePlain(3.0));
    backoff.RecordRound(1, 100.0);
    RecordWindow(backoff, 1, 4.0);
    RecordWindow(backoff, 1, 4.0);
    assert(backoff.ShouldMeasurePlain());
    assert(backoff.ObservePlain(3.0));
  }
  {
    // note (ZaneZhang-67): an inflated first plain probe can hide a later loss.
    DraftBackoff backoff;
    backoff.RecordRound(1, 0.100);
    RecordWindow(backoff, 1, 0.003);
    RecordWindow(backoff, 1, 0.003);
    assert(backoff.ShouldMeasurePlain());
    assert(!backoff.ObservePlain(0.004));
    backoff.RecordRound(1, 0.100);
    for (int i = 0; i < 63; ++i) {
      backoff.RecordRound(1, 0.003);
      assert(!backoff.ShouldMeasurePlain());
    }
    backoff.RecordRound(1, 0.003);
    assert(backoff.ShouldMeasurePlain());
    assert(backoff.ObservePlain(0.002));
  }
  {
    // note (ZaneZhang-67): a winning refresh must reset the probe interval.
    DraftBackoff backoff;
    backoff.RecordRound(1, 0.100);
    RecordWindow(backoff, 1, 0.003);
    RecordWindow(backoff, 1, 0.003);
    assert(backoff.ShouldMeasurePlain());
    assert(!backoff.ObservePlain(0.004));
    for (int i = 0; i < 2; ++i) {
      assert(!backoff.ShouldMeasurePlain());
      backoff.RecordRound(1, 0.100);
      for (int j = 0; j < 63; ++j) {
        backoff.RecordRound(1, 0.003);
        assert(!backoff.ShouldMeasurePlain());
      }
      backoff.RecordRound(1, 0.003);
      assert(backoff.ShouldMeasurePlain());
      assert(!backoff.ObservePlain(0.0035));
      assert(!backoff.ShouldMeasurePlain());
    }
  }
  {
    // One slow window is not enough to permanently disable speculation.
    DraftBackoff backoff;
    backoff.RecordRound(1, 1.0);
    RecordWindow(backoff, 1, 10.0);
    RecordWindow(backoff, 4, 2.0);
    assert(backoff.ShouldMeasurePlain());
    assert(!backoff.ObservePlain(1.0));
  }
  {
    // The decision weighs useful tokens, including the target's bonus.
    DraftBackoff backoff;
    backoff.RecordRound(1, 1.0);
    for (int i = 0; i < 2; ++i) {
      backoff.RecordRound(1, 2.0);
      backoff.RecordRound(7, 2.0);
      backoff.RecordRound(7, 2.0);
      backoff.RecordRound(7, 2.0);
    }
    assert(backoff.ShouldMeasurePlain());
    assert(!backoff.ObservePlain(0.5));
  }
  {
    DraftBackoff backoff;
    backoff.RecordRound(1, 1.0);
    RecordWindow(backoff, 2, 2.08);
    RecordWindow(backoff, 2, 2.08);
    assert(backoff.ShouldMeasurePlain());
    assert(!backoff.ObservePlain(1.0));
  }
  {
    // A fresh request does not inherit a preceding request's backoff.
    DraftBackoff backoff;
    backoff.RecordRound(1, 1.0);
    RecordWindow(backoff, 1, 2.0);
    RecordWindow(backoff, 1, 2.0);
    assert(backoff.ShouldMeasurePlain());
    assert(backoff.ObservePlain(1.0));
    DraftBackoff next_request;
    assert(!next_request.ShouldMeasurePlain());
    next_request.RecordRound(1, 1.0);
    RecordWindow(next_request, 4, 2.0);
    RecordWindow(next_request, 4, 2.0);
    assert(next_request.ShouldMeasurePlain());
    assert(!next_request.ObservePlain(1.0));
  }
}
