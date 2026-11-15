// SPDX-License-Identifier: Apache-2.0
// Keeps a GPU session's stdout free of the Metal source that MLX prints the
// first time it compiles a custom kernel.
#pragma once

#include <fcntl.h>
#include <unistd.h>

#include <cstdio>

namespace haru {

class QuietStdout {
public:
  QuietStdout()
      : saved_(dup(STDOUT_FILENO)), null_(open("/dev/null", O_WRONLY)) {
    if (saved_ >= 0 && null_ >= 0) {
      std::fflush(stdout);
      dup2(null_, STDOUT_FILENO);
      // Unbuffered while silenced: whatever is written here must reach the
      // null device and not resurface through a later flush.
      setvbuf(stdout, nullptr, _IONBF, 0);
    }
  }
  ~QuietStdout() {
    if (saved_ < 0) {
      if (null_ >= 0) {
        close(null_);
      }
      return;
    }
    dup2(saved_, STDOUT_FILENO);
    close(saved_);
    if (null_ >= 0) {
      close(null_);
    }
    setvbuf(stdout, nullptr, _IOLBF, BUFSIZ);
  }
  QuietStdout(const QuietStdout &) = delete;
  QuietStdout &operator=(const QuietStdout &) = delete;

private:
  int saved_ = -1;
  int null_ = -1;
};

} // namespace haru
