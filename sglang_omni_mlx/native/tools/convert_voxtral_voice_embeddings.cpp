// SPDX-License-Identifier: Apache-2.0
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

namespace {

using Json = nlohmann::json;

constexpr uint32_t kEndOfCentralDirectorySignature = 0x06054b50;
constexpr uint32_t kCentralDirectoryEntrySignature = 0x02014b50;
constexpr uint32_t kLocalFileHeaderSignature = 0x04034b50;
constexpr size_t kEndOfCentralDirectorySize = 22;
constexpr size_t kCentralDirectoryEntrySize = 46;
constexpr size_t kLocalFileHeaderSize = 30;

uint16_t ReadLittleEndian16(const std::vector<char> &bytes, size_t offset) {
  if (offset > bytes.size() || bytes.size() - offset < 2) {
    throw std::runtime_error("corrupt PyTorch archive");
  } else {
  }
  return static_cast<uint8_t>(bytes[offset]) |
         (static_cast<uint16_t>(static_cast<uint8_t>(bytes[offset + 1])) << 8);
}

uint32_t ReadLittleEndian32(const std::vector<char> &bytes, size_t offset) {
  if (offset > bytes.size() || bytes.size() - offset < 4) {
    throw std::runtime_error("corrupt PyTorch archive");
  } else {
  }
  return static_cast<uint32_t>(ReadLittleEndian16(bytes, offset)) |
         (static_cast<uint32_t>(ReadLittleEndian16(bytes, offset + 2)) << 16);
}

std::vector<char> ReadArchive(const std::filesystem::path &path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    throw std::runtime_error("cannot read " + path.string());
  } else {
  }
  return std::vector<char>(std::istreambuf_iterator<char>(input),
                           std::istreambuf_iterator<char>());
}

std::vector<char> ReadStoredMember(const std::vector<char> &archive,
                                   const std::string &suffix) {
  if (archive.size() < kEndOfCentralDirectorySize) {
    throw std::runtime_error("not a PyTorch ZIP archive");
  } else {
  }
  const size_t end_offset = archive.size() - kEndOfCentralDirectorySize;
  if (ReadLittleEndian32(archive, end_offset) !=
      kEndOfCentralDirectorySignature) {
    throw std::runtime_error("unsupported PyTorch ZIP archive");
  } else {
  }
  const uint16_t entry_count = ReadLittleEndian16(archive, end_offset + 10);
  const uint32_t directory_offset =
      ReadLittleEndian32(archive, end_offset + 16);
  size_t entry_offset = directory_offset;
  for (uint16_t entry = 0; entry < entry_count; ++entry) {
    if (ReadLittleEndian32(archive, entry_offset) !=
        kCentralDirectoryEntrySignature) {
      throw std::runtime_error("corrupt PyTorch ZIP directory");
    } else {
    }
    const uint16_t method = ReadLittleEndian16(archive, entry_offset + 10);
    const uint32_t compressed_size =
        ReadLittleEndian32(archive, entry_offset + 20);
    const uint16_t name_length = ReadLittleEndian16(archive, entry_offset + 28);
    const uint16_t extra_length =
        ReadLittleEndian16(archive, entry_offset + 30);
    const uint16_t comment_length =
        ReadLittleEndian16(archive, entry_offset + 32);
    const uint32_t local_offset =
        ReadLittleEndian32(archive, entry_offset + 42);
    const size_t name_offset = entry_offset + kCentralDirectoryEntrySize;
    if (name_offset > archive.size() ||
        archive.size() - name_offset < name_length) {
      throw std::runtime_error("corrupt PyTorch ZIP member name");
    } else {
    }
    const std::string name(archive.data() + name_offset, name_length);
    entry_offset = name_offset + name_length + extra_length + comment_length;
    if (!name.ends_with(suffix)) {
      continue;
    } else {
    }
    if (method != 0 || ReadLittleEndian32(archive, local_offset) !=
                           kLocalFileHeaderSignature) {
      throw std::runtime_error(
          "voice embedding must be a stored PyTorch tensor");
    } else {
    }
    const size_t data_offset = local_offset + kLocalFileHeaderSize +
                               ReadLittleEndian16(archive, local_offset + 26) +
                               ReadLittleEndian16(archive, local_offset + 28);
    if (data_offset > archive.size() ||
        archive.size() - data_offset < compressed_size) {
      throw std::runtime_error("corrupt PyTorch ZIP member data");
    } else {
    }
    return {archive.begin() + data_offset,
            archive.begin() + data_offset + compressed_size};
  }
  throw std::runtime_error("PyTorch archive is missing " + suffix);
}

std::pair<std::vector<char>, std::vector<int64_t>>
LoadVoiceTensor(const std::filesystem::path &path, int64_t hidden_dimension) {
  const std::vector<char> archive = ReadArchive(path);
  const std::vector<char> pickle = ReadStoredMember(archive, "/data.pkl");
  const std::string pickle_text(pickle.begin(), pickle.end());
  if (pickle_text.find("torch\nBFloat16Storage\n") == std::string::npos) {
    throw std::runtime_error("voice embedding must be a BF16 tensor: " +
                             path.string());
  } else {
  }
  const std::vector<char> byteorder = ReadStoredMember(archive, "/byteorder");
  if (std::string(byteorder.begin(), byteorder.end()) != "little") {
    throw std::runtime_error("voice embedding must use little-endian storage");
  } else {
  }
  std::vector<char> tensor = ReadStoredMember(archive, "/data/0");
  if (hidden_dimension <= 0) {
    throw std::invalid_argument("hidden dimension must be positive");
  } else {
  }
  const size_t bytes_per_frame = static_cast<size_t>(hidden_dimension) * 2;
  if (tensor.empty() || tensor.size() % bytes_per_frame != 0) {
    throw std::runtime_error("voice embedding has an unexpected shape: " +
                             path.string());
  } else {
  }
  const int64_t frame_count = tensor.size() / bytes_per_frame;
  return {std::move(tensor), {frame_count, hidden_dimension}};
}

void WriteSafetensors(const std::filesystem::path &path,
                      const std::vector<char> &tensor,
                      const std::vector<int64_t> &shape) {
  Json header = {{"voice_embedding",
                  {{"dtype", "BF16"},
                   {"shape", shape},
                   {"data_offsets", {0, tensor.size()}}}}};
  std::string header_bytes = header.dump();
  header_bytes.append((8 - header_bytes.size() % 8) % 8, ' ');
  std::ofstream output(path, std::ios::binary);
  if (!output) {
    throw std::runtime_error("cannot write " + path.string());
  } else {
  }
  const uint64_t header_size = header_bytes.size();
  output.write(reinterpret_cast<const char *>(&header_size),
               sizeof(header_size));
  output.write(header_bytes.data(),
               static_cast<std::streamsize>(header_bytes.size()));
  output.write(tensor.data(), static_cast<std::streamsize>(tensor.size()));
  if (!output) {
    throw std::runtime_error("failed to write " + path.string());
  } else {
  }
}

void ConvertVoiceEmbeddings(const std::filesystem::path &model_path) {
  const std::vector<char> parameter_bytes =
      ReadArchive(model_path / "params.json");
  const Json parameters =
      Json::parse(parameter_bytes.begin(), parameter_bytes.end());
  const int64_t hidden_dimension = parameters.at("dim").get<int64_t>();
  const std::filesystem::path voice_directory = model_path / "voice_embedding";
  if (!std::filesystem::is_directory(voice_directory)) {
    throw std::runtime_error("Voxtral voice embeddings are missing: " +
                             voice_directory.string());
  } else {
  }
  size_t converted_count = 0;
  for (const auto &entry :
       std::filesystem::directory_iterator(voice_directory)) {
    if (!entry.is_regular_file() || entry.path().extension() != ".pt") {
      continue;
    } else {
    }
    auto [tensor, shape] = LoadVoiceTensor(entry.path(), hidden_dimension);
    std::filesystem::path output_path = entry.path();
    output_path.replace_extension(".safetensors");
    WriteSafetensors(output_path, tensor, shape);
    std::cout << "Converted " << entry.path().filename().string() << " to "
              << output_path.filename().string() << "\n";
    ++converted_count;
  }
  if (converted_count == 0) {
    throw std::runtime_error("Voxtral voice directory has no .pt files: " +
                             voice_directory.string());
  } else {
  }
}

} // namespace

int main(int argument_count, char **argument_values) {
  try {
    if (argument_count != 2) {
      throw std::invalid_argument(
          "usage: voxtral_tts_convert_voices MODEL_PATH");
    } else {
    }
    ConvertVoiceEmbeddings(argument_values[1]);
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "voxtral_tts_convert_voices: " << error.what() << "\n";
    return 1;
  }
}
