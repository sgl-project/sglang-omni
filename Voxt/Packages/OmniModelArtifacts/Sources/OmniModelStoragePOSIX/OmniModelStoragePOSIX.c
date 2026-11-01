#define _GNU_SOURCE
#include "OmniModelStoragePOSIX.h"
#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <sys/file.h>
#include <unistd.h>
#if defined(__linux__)
#include <linux/fs.h>
#include <sys/syscall.h>
#endif

int omni_model_storage_flock(int descriptor, int operation) {
    return flock(descriptor, operation) == 0 ? 0 : errno;
}

int omni_model_storage_exchange(const char *staging, const char *destination) {
#if defined(__APPLE__)
    return renameatx_np(AT_FDCWD, staging, AT_FDCWD, destination, RENAME_SWAP) == 0 ? 0 : errno;
#elif defined(__linux__) && defined(SYS_renameat2)
    return syscall(SYS_renameat2, AT_FDCWD, staging, AT_FDCWD, destination, RENAME_EXCHANGE) == 0 ? 0 : errno;
#else
    return ENOTSUP;
#endif
}
