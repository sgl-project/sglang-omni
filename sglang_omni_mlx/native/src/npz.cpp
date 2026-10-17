// SPDX-License-Identifier: Apache-2.0
#include "npz.h"

#include <fcntl.h>
#include <unistd.h>

#include <cstring>
#include <memory>
#include <stdexcept>
#include <vector>

namespace npz {

namespace mx = mlx::core;

namespace {

// ZIP record signatures and sizes, and the field value that defers to the
// entry's ZIP64 extra field.
constexpr uint32_t kEndOfCentralDirectorySignature = 0x06054b50;
constexpr uint32_t kCentralDirectoryEntrySignature = 0x02014b50;
constexpr uint32_t kLocalFileHeaderSignature = 0x04034b50;
constexpr size_t kEndOfCentralDirectorySize = 22;
constexpr size_t kCentralDirectoryEntrySize = 46;
constexpr size_t kLocalFileHeaderSize = 30;
constexpr uint16_t kZip64ExtraFieldId = 0x0001;
constexpr uint32_t kZip64Deferred = 0xFFFFFFFF;

// A little-endian integer inside bytes; a field past the end means the
// archive's records lie about their sizes.
template <typename Integer>
Integer ReadLittleEndian(const std::vector<char> &bytes, size_t offset) {
  if (offset > bytes.size() || bytes.size() - offset < sizeof(Integer)) {
    throw std::runtime_error(
        "corrupt .npz archive: a record runs past its end");
  } else {
  }
  Integer value = 0;
  std::memcpy(&value, bytes.data() + offset, sizeof(value));
  return value;
}

// An open archive read at absolute offsets, shared by its members' readers.
class ArchiveFile {
public:
  explicit ArchiveFile(const std::filesystem::path &path)
      : descriptor_(open(path.c_str(), O_RDONLY)), path_(path.string()) {
    if (descriptor_ < 0) {
      throw std::runtime_error("cannot read " + path_);
    } else {
    }
  }
  ~ArchiveFile() { close(descriptor_); }
  ArchiveFile(const ArchiveFile &) = delete;
  ArchiveFile &operator=(const ArchiveFile &) = delete;

  void Read(char *data, size_t byte_count, size_t offset) const {
    while (byte_count > 0) {
      const ssize_t read_count =
          pread(descriptor_, data, byte_count, static_cast<off_t>(offset));
      if (read_count <= 0) {
        throw std::runtime_error("cannot read " + path_);
      } else {
      }
      data += read_count;
      byte_count -= static_cast<size_t>(read_count);
      offset += static_cast<size_t>(read_count);
    }
  }
  std::vector<char> ReadRange(size_t offset, size_t byte_count) const {
    std::vector<char> bytes(byte_count);
    Read(bytes.data(), byte_count, offset);
    return bytes;
  }

private:
  int descriptor_;
  std::string path_;
};

// One stored member of an archive, seen as a file of its own so MLX's .npy
// loader reads it lazily from the archive. The loader reads the header in
// order and the data at an offset; it never seeks.
class ZipMemberReader : public mx::io::Reader {
public:
  ZipMemberReader(std::shared_ptr<const ArchiveFile> archive,
                  size_t member_offset, std::string label)
      : archive_(std::move(archive)), member_offset_(member_offset),
        label_(std::move(label)) {}

  bool is_open() const override { return true; }
  bool good() const override { return true; }
  size_t tell() override { return position_; }
  void seek(int64_t, std::ios_base::seekdir) override {
    throw std::logic_error(label_ + " is read in order or at offsets");
  }
  void read(char *data, size_t byte_count) override {
    archive_->Read(data, byte_count, member_offset_ + position_);
    position_ += byte_count;
  }
  void read(char *data, size_t byte_count, size_t offset) override {
    archive_->Read(data, byte_count, member_offset_ + offset);
  }
  std::string label() const override { return label_; }

private:
  std::shared_ptr<const ArchiveFile> archive_;
  size_t member_offset_;
  std::string label_;
  size_t position_ = 0;
};

} // namespace

std::unordered_map<std::string, mx::array>
Load(const std::filesystem::path &path) {
  const auto archive = std::make_shared<const ArchiveFile>(path);
  const size_t file_size = std::filesystem::file_size(path);
  if (file_size < kEndOfCentralDirectorySize) {
    throw std::runtime_error(path.string() + " is not an .npz archive");
  } else {
  }
  // Note (khazic): the end record is the archive's last 22 bytes: numpy writes
  // no comment.
  const std::vector<char> end_record = archive->ReadRange(
      file_size - kEndOfCentralDirectorySize, kEndOfCentralDirectorySize);
  const size_t directory_offset = ReadLittleEndian<uint32_t>(end_record, 16);
  if (ReadLittleEndian<uint32_t>(end_record, 0) !=
      kEndOfCentralDirectorySignature) {
    throw std::runtime_error(path.string() + " is not an .npz archive");
  } else if (directory_offset == kZip64Deferred) {
    throw std::runtime_error(path.string() + " is a ZIP64 archive");
  } else if (directory_offset > file_size - kEndOfCentralDirectorySize) {
    throw std::runtime_error(path.string() + " has a corrupt directory");
  } else {
  }
  const uint16_t entry_count = ReadLittleEndian<uint16_t>(end_record, 10);
  const std::vector<char> directory = archive->ReadRange(
      directory_offset,
      file_size - kEndOfCentralDirectorySize - directory_offset);
  std::unordered_map<std::string, mx::array> arrays;
  size_t entry_offset = 0;
  for (uint16_t entry = 0; entry < entry_count; ++entry) {
    // Note (khazic): also keeps the name check below from underflowing.
    if (entry_offset + kCentralDirectoryEntrySize > directory.size() ||
        ReadLittleEndian<uint32_t>(directory, entry_offset) !=
            kCentralDirectoryEntrySignature) {
      throw std::runtime_error(path.string() + " has a corrupt directory");
    } else if (ReadLittleEndian<uint16_t>(directory, entry_offset + 10) != 0) {
      throw std::runtime_error(path.string() +
                               " is compressed; only stored .npz is read");
    } else {
    }
    uint64_t local_header_offset =
        ReadLittleEndian<uint32_t>(directory, entry_offset + 42);
    const uint16_t name_length =
        ReadLittleEndian<uint16_t>(directory, entry_offset + 28);
    if (directory.size() - entry_offset - kCentralDirectoryEntrySize <
        name_length) {
      throw std::runtime_error(path.string() + " has a corrupt directory");
    } else {
    }
    const std::string name(directory.data() + entry_offset +
                               kCentralDirectoryEntrySize,
                           name_length);
    // Note (khazic): a ZIP64 extra field holds, in order, only the values
    // deferred to it: the uncompressed size, the compressed size, then the
    // offset.
    size_t extra_field_offset =
        entry_offset + kCentralDirectoryEntrySize + name_length;
    const size_t extra_fields_end =
        extra_field_offset +
        ReadLittleEndian<uint16_t>(directory, entry_offset + 30);
    while (extra_field_offset + 4 <= extra_fields_end) {
      const uint16_t header_id =
          ReadLittleEndian<uint16_t>(directory, extra_field_offset);
      const uint16_t field_size =
          ReadLittleEndian<uint16_t>(directory, extra_field_offset + 2);
      if (header_id == kZip64ExtraFieldId &&
          local_header_offset == kZip64Deferred) {
        size_t value_offset = extra_field_offset + 4;
        if (ReadLittleEndian<uint32_t>(directory, entry_offset + 24) ==
            kZip64Deferred) {
          value_offset += 8;
        } else {
        }
        if (ReadLittleEndian<uint32_t>(directory, entry_offset + 20) ==
            kZip64Deferred) {
          value_offset += 8;
        } else {
        }
        if (value_offset + 8 > extra_field_offset + 4 + field_size) {
          throw std::runtime_error(path.string() + " has a corrupt directory");
        } else {
        }
        local_header_offset =
            ReadLittleEndian<uint64_t>(directory, value_offset);
      } else {
      }
      extra_field_offset += 4 + field_size;
    }
    entry_offset = extra_fields_end +
                   ReadLittleEndian<uint16_t>(directory, entry_offset + 32);
    const std::vector<char> local_header =
        archive->ReadRange(local_header_offset, kLocalFileHeaderSize);
    if (ReadLittleEndian<uint32_t>(local_header, 0) !=
        kLocalFileHeaderSignature) {
      throw std::runtime_error(path.string() + " has a corrupt member " + name);
    } else {
    }
    const size_t member_offset = local_header_offset + kLocalFileHeaderSize +
                                 ReadLittleEndian<uint16_t>(local_header, 26) +
                                 ReadLittleEndian<uint16_t>(local_header, 28);
    arrays.insert_or_assign(
        name.ends_with(".npy") ? name.substr(0, name.size() - 4) : name,
        mx::load(std::make_shared<ZipMemberReader>(
            archive, member_offset, path.string() + ":" + name)));
  }
  return arrays;
}

} // namespace npz
