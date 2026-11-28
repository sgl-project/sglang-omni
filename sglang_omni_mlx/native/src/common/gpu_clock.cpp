// SPDX-License-Identifier: Apache-2.0
#include "common/gpu_clock.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <mutex>
#include <stdexcept>
#include <vector>

#include "mlx/backend/metal/device.h"

namespace omni_common {

namespace mx = mlx::core;

namespace {

constexpr int kBuffersInFlight = 2;
constexpr double kBufferSeconds = 5e-4;
constexpr uint32_t kProbeIterations = 200000;
constexpr int kProbeCount = 3;
// A session that has had no audio for this long lets the clock go down.
constexpr std::chrono::seconds kIdleTimeout{3};

constexpr const char *kKernelSource = R"(
#include <metal_stdlib>
using namespace metal;
kernel void keep_clock(device uint *sink [[buffer(0)]],
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

// What every hold shares.
struct Keeper {
  std::mutex mutex;
  int hold_count = 0;
  int buffers_in_flight = 0;
  std::chrono::steady_clock::time_point last_touch;
};

Keeper &SharedKeeper() {
  static Keeper keeper;
  return keeper;
}

// One command buffer of the kernel, autoreleased.
MTL::CommandBuffer *EncodeBuffer(const Pipeline &pipeline,
                                 uint32_t iterations) {
  MTL::CommandBuffer *buffer = pipeline.queue->commandBuffer();
  MTL::ComputeCommandEncoder *encoder = buffer->computeCommandEncoder();
  encoder->setComputePipelineState(pipeline.state);
  encoder->setBuffer(pipeline.sink, 0, 0);
  encoder->setBytes(&iterations, sizeof(iterations), 1);
  encoder->dispatchThreadgroups(MTL::Size(1, 1, 1), MTL::Size(32, 1, 1));
  encoder->endEncoding();
  return buffer;
}

// The process's pipeline and queue, never released; the kernel's iterations
// are timed to last kBufferSeconds.
const Pipeline &SharedPipeline() {
  static const Pipeline shared = [] {
    MTL::Device *device = mx::metal::device(mx::Device::gpu).mtl_device();
    NS::AutoreleasePool *pool = NS::AutoreleasePool::alloc()->init();
    NS::Error *error = nullptr;
    MTL::Library *library = device->newLibrary(
        NS::String::string(kKernelSource, NS::UTF8StringEncoding), nullptr,
        &error);
    if (library == nullptr) {
      pool->release();
      throw std::runtime_error("the GPU clock kernel failed to compile");
    } else {
    }
    MTL::Function *function = library->newFunction(
        NS::String::string("keep_clock", NS::UTF8StringEncoding));
    Pipeline pipeline;
    pipeline.state = device->newComputePipelineState(function, &error);
    function->release();
    library->release();
    pipeline.queue = device->newCommandQueue();
    pipeline.sink =
        device->newBuffer(sizeof(uint32_t), MTL::ResourceStorageModeShared);
    std::vector<double> iterations_per_second;
    for (int i = 0; i < kProbeCount; ++i) {
      MTL::CommandBuffer *probe = EncodeBuffer(pipeline, kProbeIterations);
      probe->commit();
      probe->waitUntilCompleted();
      iterations_per_second.push_back(
          kProbeIterations / (probe->GPUEndTime() - probe->GPUStartTime()));
    }
    pool->release();
    std::sort(iterations_per_second.begin(), iterations_per_second.end());
    pipeline.iterations_per_buffer = static_cast<uint32_t>(
        iterations_per_second[kProbeCount / 2] * kBufferSeconds);
    return pipeline;
  }();
  return shared;
}

// Commits a buffer that, once complete, commits the next one while a hold
// has had audio recently, so the kernel stays queued without a thread.
void CommitBuffer() {
  NS::AutoreleasePool *pool = NS::AutoreleasePool::alloc()->init();
  const Pipeline &pipeline = SharedPipeline();
  MTL::CommandBuffer *buffer =
      EncodeBuffer(pipeline, pipeline.iterations_per_buffer);
  buffer->addCompletedHandler([](MTL::CommandBuffer *) {
    Keeper &keeper = SharedKeeper();
    bool keeps_clock;
    {
      const std::lock_guard<std::mutex> lock(keeper.mutex);
      keeps_clock =
          keeper.hold_count > 0 &&
          std::chrono::steady_clock::now() - keeper.last_touch < kIdleTimeout;
      if (!keeps_clock) {
        --keeper.buffers_in_flight;
      } else {
      }
    }
    if (keeps_clock) {
      CommitBuffer();
    } else {
    }
  });
  buffer->commit();
  pool->release();
}

// Marks audio now and fills the queue up to kBuffersInFlight.
void TouchKeeper(bool adds_hold) {
  SharedPipeline();
  Keeper &keeper = SharedKeeper();
  int missing_count;
  {
    const std::lock_guard<std::mutex> lock(keeper.mutex);
    keeper.hold_count += adds_hold ? 1 : 0;
    keeper.last_touch = std::chrono::steady_clock::now();
    missing_count =
        keeper.hold_count > 0 ? kBuffersInFlight - keeper.buffers_in_flight : 0;
    keeper.buffers_in_flight += missing_count;
  }
  for (int i = 0; i < missing_count; ++i) {
    CommitBuffer();
  }
}

} // namespace

GpuClockHold::GpuClockHold() { TouchKeeper(true); }

GpuClockHold::~GpuClockHold() {
  Keeper &keeper = SharedKeeper();
  const std::lock_guard<std::mutex> lock(keeper.mutex);
  --keeper.hold_count;
}

void GpuClockHold::Touch() { TouchKeeper(false); }

} // namespace omni_common
