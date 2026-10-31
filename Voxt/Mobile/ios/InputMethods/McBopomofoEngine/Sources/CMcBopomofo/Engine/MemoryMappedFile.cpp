// Copyright (c) 2024 and onwards The McBopomofo Authors.
//
// Permission is hereby granted, free of charge, to any person
// obtaining a copy of this software and associated documentation
// files (the "Software"), to deal in the Software without
// restriction, including without limitation the rights to use,
// copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following
// conditions:
//
// The above copyright notice and this permission notice shall be
// included in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
// EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
// OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
// NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
// HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
// WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
// OTHER DEALINGS IN THE SOFTWARE.

#include "MemoryMappedFile.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <utility>

namespace McBopomofo {

MemoryMappedFile::MemoryMappedFile(MemoryMappedFile&& other) noexcept
    : fd_(std::exchange(other.fd_, -1)),
      data_(std::exchange(other.data_, nullptr)),
      length_(std::exchange(other.length_, 0)) {}

MemoryMappedFile& MemoryMappedFile::operator=(
    MemoryMappedFile&& other) noexcept {
  close();
  fd_ = std::exchange(other.fd_, -1);
  data_ = std::exchange(other.data_, nullptr);
  length_ = std::exchange(other.length_, 0);
  return *this;
}

MemoryMappedFile::~MemoryMappedFile() { close(); }

bool MemoryMappedFile::open(const char* path) {
  if (fd_ != -1) {
    return false;
  }

  fd_ = ::open(path, O_RDONLY);
  if (fd_ == -1) {
    return false;
  }

  struct stat sb;
  if (fstat(fd_, &sb) == -1) {
    ::close(fd_);
    fd_ = -1;
    return false;
  }

  length_ = static_cast<size_t>(sb.st_size);

  // No need to check if length_ is 0; mmmap fails on empty files.
  data_ = mmap(nullptr, length_, PROT_READ, MAP_SHARED, fd_, 0);
  if (data_ == MAP_FAILED) {
    ::close(fd_);
    fd_ = -1;
    length_ = 0;
    data_ = nullptr;
    return false;
  }

  return true;
}

void MemoryMappedFile::close() {
  if (fd_ == -1) {
    return;
  }
  munmap(data_, length_);
  ::close(fd_);
  fd_ = -1;
  length_ = 0;
  data_ = nullptr;
}

}  // namespace McBopomofo
