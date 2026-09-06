// Preloaded bootstrap that caps a stage process to a subset of the device's
// SMs using a CUDA Green Context.
//
// Build:
//   make -C tools/green_ctx
//
// Use: set GREEN_CTX_SM (and LD_PRELOAD to this library) on a stage process,
// normally by setting `sm_cap` on that stage. The constructor runs before CUDA
// is initialized, so every allocation and launch in the process lands in the
// capped context.
//
// Note (Jiaxin Deng): a green context is current per thread, and the CUDA
// runtime rebinds the calling thread to the primary context on every
// cudaSetDevice. Interposing only one of pthread_create / cudaSetDevice leaves
// part of the process on the full device and splits it across two contexts,
// which breaks stream ordering between them; both interposers are required.
//
// Note (Jiaxin Deng): every failure here exits the process. A stage that
// silently ran uncapped would be indistinguishable from a working one in every
// metric except throughput, which is the failure this library exists to catch.

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <dlfcn.h>
#include <pthread.h>

#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

extern char *program_invocation_short_name;

namespace {

CUcontext capped_context = nullptr;
unsigned int requested_sm = 0;
unsigned int actual_sm = 0;
int device_ordinal = 0;

[[noreturn]] void fail_message(const char *format, ...) {
  va_list args;
  va_start(args, format);
  std::fprintf(stderr, "green-ctx: ");
  std::vfprintf(stderr, format, args);
  va_end(args);
  std::fputc('\n', stderr);
  std::fflush(stderr);
  std::_Exit(125);
}

void check(CUresult result, const char *expression) {
  if (result == CUDA_SUCCESS) {
    return;
  }
  const char *name = nullptr;
  const char *message = nullptr;
  cuGetErrorName(result, &name);
  cuGetErrorString(result, &message);
  fail_message("%s failed: %s (%s)", expression, name ? name : "unknown",
               message ? message : "unknown");
}

#define CU_CHECK(expression) check((expression), #expression)

unsigned int env_uint(const char *name, unsigned int fallback) {
  const char *text = std::getenv(name);
  if (!text || !*text) {
    return fallback;
  }
  return static_cast<unsigned int>(std::strtoul(text, nullptr, 10));
}

__attribute__((constructor)) void initialize_capped_context() {
  const char *requested_text = std::getenv("GREEN_CTX_SM");
  if (!requested_text || !*requested_text) {
    return;
  }
  // Note (Jiaxin Deng): helper binaries a stage spawns (ldconfig via
  // torch.inductor, compilers, shells) inherit LD_PRELOAD. Initializing CUDA
  // inside them fails and their non-zero exit breaks the caller, so only the
  // Python stage process takes a partition.
  const char *invoked = program_invocation_short_name;
  if (!invoked || std::strncmp(invoked, "python", 6) != 0) {
    return;
  }

  requested_sm =
      static_cast<unsigned int>(std::strtoul(requested_text, nullptr, 10));
  unsigned int split = env_uint("GREEN_CTX_SPLIT", 0);
  unsigned int group_count = env_uint("GREEN_CTX_GROUP_COUNT", 0);
  if (requested_sm == 0 || split == 0 || group_count == 0) {
    fail_message("GREEN_CTX_SM, GREEN_CTX_SPLIT and GREEN_CTX_GROUP_COUNT must "
                 "all be positive");
  }

  CU_CHECK(cuInit(0));
  int device_total = 0;
  CU_CHECK(cuDeviceGetCount(&device_total));
  // Note (Jiaxin Deng): a cap names one device, but a stage placed on a
  // different ordinal would be capped on the wrong one and run unrestricted
  // while still passing verification. Requiring a single visible device makes
  // ordinal 0 unambiguous; narrow CUDA_VISIBLE_DEVICES per worker instead.
  if (device_total != 1) {
    fail_message("a capped process must see exactly one CUDA device, sees %d; "
                 "narrow CUDA_VISIBLE_DEVICES for this stage",
                 device_total);
  }
  device_ordinal = 0;
  CUdevice device;
  CU_CHECK(cuDeviceGet(&device, device_ordinal));

  CUdevResource whole_device{};
  CU_CHECK(
      cuDeviceGetDevResource(device, &whole_device, CU_DEV_RESOURCE_TYPE_SM));
  unsigned int available = 0;
  CU_CHECK(cuDevSmResourceSplitByCount(nullptr, &available, &whole_device,
                                       nullptr, 0, split));
  if (available < group_count) {
    fail_message("device splits into %u groups of %u SMs, need %u", available,
                 split, group_count);
  }

  std::vector<CUdevResource> groups(available);
  CUdevResource remainder{};
  CU_CHECK(cuDevSmResourceSplitByCount(groups.data(), &available, &whole_device,
                                       &remainder, 0, split));
  actual_sm = 0;
  for (unsigned int index = 0; index < group_count; ++index) {
    actual_sm += groups[index].sm.smCount;
  }

  CUdevResourceDesc descriptor = nullptr;
  CU_CHECK(cuDevResourceGenerateDesc(&descriptor, groups.data(), group_count));
  CUgreenCtx green_context = nullptr;
  CU_CHECK(cuGreenCtxCreate(&green_context, descriptor, device,
                            CU_GREEN_CTX_DEFAULT_STREAM));
  CU_CHECK(cuCtxFromGreenCtx(&capped_context, green_context));
  CU_CHECK(cuCtxSetCurrent(capped_context));

  std::fprintf(stderr, "green-ctx: device=%d requested_sm=%u actual_sm=%u\n",
               device_ordinal, requested_sm, actual_sm);
  std::fflush(stderr);
}

void bind_or_die(const char *where) {
  CUresult result = cuCtxSetCurrent(capped_context);
  if (result != CUDA_SUCCESS) {
    const char *name = nullptr;
    cuGetErrorName(result, &name);
    fail_message("%s could not bind the capped context: %s", where,
                 name ? name : "unknown");
  }
}

struct ThreadStart {
  void *(*entry)(void *);
  void *argument;
};

void *bind_then_run(void *raw) {
  ThreadStart *boxed = static_cast<ThreadStart *>(raw);
  ThreadStart call = *boxed;
  delete boxed;
  bind_or_die("new thread");
  return call.entry(call.argument);
}

using pthread_create_fn = int (*)(pthread_t *, const pthread_attr_t *,
                                  void *(*)(void *), void *);
using cuda_set_device_fn = cudaError_t (*)(int);

void *next_symbol(const char *name) {
  void *symbol = dlsym(RTLD_NEXT, name);
  if (symbol == nullptr) {
    fail_message("could not resolve the real %s: %s", name, dlerror());
  }
  return symbol;
}

} // namespace

extern "C" int pthread_create(pthread_t *thread,
                              const pthread_attr_t *attributes,
                              void *(*entry)(void *), void *argument) {
  static pthread_create_fn real =
      reinterpret_cast<pthread_create_fn>(next_symbol("pthread_create"));
  if (capped_context == nullptr) {
    return real(thread, attributes, entry, argument);
  }
  ThreadStart *boxed = new ThreadStart{entry, argument};
  int status = real(thread, attributes, bind_then_run, boxed);
  if (status != 0) {
    delete boxed;
  }
  return status;
}

extern "C" cudaError_t cudaSetDevice(int device) {
  static cuda_set_device_fn real =
      reinterpret_cast<cuda_set_device_fn>(next_symbol("cudaSetDevice"));
  cudaError_t status = real(device);
  if (capped_context == nullptr || status != cudaSuccess) {
    return status;
  }
  // A capped process sees one device, so any successful cudaSetDevice selected
  // the capped one and left the primary context current.
  bind_or_die("cudaSetDevice");
  return status;
}

// Read back by the stage process to verify its cap took effect.
extern "C" unsigned int green_ctx_requested_sm() { return requested_sm; }

extern "C" unsigned int green_ctx_actual_sm() { return actual_sm; }

// 1 when the calling thread is bound to the capped context itself, not merely
// to some context reporting the same SM count.
extern "C" int green_ctx_current_is_capped() {
  CUcontext current = nullptr;
  if (capped_context == nullptr || cuCtxGetCurrent(&current) != CUDA_SUCCESS) {
    return 0;
  }
  return current == capped_context ? 1 : 0;
}

extern "C" unsigned int green_ctx_current_sm() {
  CUcontext current = nullptr;
  if (cuCtxGetCurrent(&current) != CUDA_SUCCESS || current == nullptr) {
    return 0;
  }
  CUdevResource resource{};
  if (cuCtxGetDevResource(current, &resource, CU_DEV_RESOURCE_TYPE_SM) !=
      CUDA_SUCCESS) {
    return 0;
  }
  return resource.sm.smCount;
}
