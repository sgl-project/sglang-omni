// SPDX-License-Identifier: Apache-2.0
// Keeps the GPU's clock up while a live session takes audio.
#pragma once

namespace omni_common {

// Note (XXXXRT666): between a live session's decodes the GPU clocks down,
// and the Final after the user stops then runs, its encoder about 1.8x slower
// (Qwen3-ASR 1.7B on an M4 Air). While a hold lives and audio came within the
// last seconds, a one-simdgroup kernel stays queued on a command queue of its
// own, which keeps the clock up: the Final takes 12% less time.
class GpuClockHold {
public:
  GpuClockHold();
  ~GpuClockHold();
  GpuClockHold(const GpuClockHold &) = delete;
  GpuClockHold &operator=(const GpuClockHold &) = delete;

  // Audio came: the clock is kept up for a while longer.
  void Touch();
};

} // namespace omni_common
