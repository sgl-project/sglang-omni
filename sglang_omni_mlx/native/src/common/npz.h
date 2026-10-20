// SPDX-License-Identifier: Apache-2.0
// The arrays of a stored (uncompressed) .npz archive, each read lazily from
// the archive by MLX's .npy loader. Entries may defer their offsets to ZIP64
// extra fields; the archive's own end record is plain.
#pragma once

#include <filesystem>
#include <string>
#include <unordered_map>

#include "mlx/mlx.h"

namespace npz {

// The arrays by member name, without the .npy suffix.
std::unordered_map<std::string, mlx::core::array>
Load(const std::filesystem::path &path);

} // namespace npz
