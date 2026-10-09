# SPDX-License-Identifier: Apache-2.0
# Native Qwen3-ASR runtime on MLX: one binary, no Python.
cmake_minimum_required(VERSION 3.24)
project(sglang_omni_mlx_native LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Note (Jiaxin Deng): exactly the MLX build the Python reference server runs
# on, so both produce bit-identical outputs.
set(SGLANG_OMNI_MLX_VERSION 0.32.3)
if(NOT MLX_DIR OR NOT EXISTS "${MLX_DIR}/MLXConfig.cmake")
  message(FATAL_ERROR "Set MLX_DIR to <mlx ${SGLANG_OMNI_MLX_VERSION} wheel>/share/cmake/MLX")
endif()
find_package(MLX ${SGLANG_OMNI_MLX_VERSION} EXACT CONFIG REQUIRED PATHS "${MLX_DIR}" NO_DEFAULT_PATH)

# Note (Jiaxin Deng): civetweb and PCRE2 still declare pre-3.5 minimums.
set(CMAKE_POLICY_VERSION_MINIMUM 3.5)
include(FetchContent)
FetchContent_Declare(
  nlohmann_json
  URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz
  URL_HASH SHA256=d6c65aca6b1ed68e7a182f4757257b107ae403032760ed6ef121c9d55e81757d
  EXCLUDE_FROM_ALL)
set(PCRE2_BUILD_PCRE2GREP OFF CACHE BOOL "" FORCE)
set(PCRE2_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(PCRE2_SUPPORT_UNICODE ON CACHE BOOL "" FORCE)
set(PCRE2_SUPPORT_JIT OFF CACHE BOOL "" FORCE)
FetchContent_Declare(
  pcre2
  GIT_REPOSITORY https://github.com/PCRE2Project/pcre2.git
  GIT_TAG 6ae58beca071f13ccfed31d03b3f479ab520639b # pcre2-10.44
  EXCLUDE_FROM_ALL)
set(CIVETWEB_ENABLE_WEBSOCKETS ON CACHE BOOL "" FORCE)
set(CIVETWEB_ENABLE_SSL OFF CACHE BOOL "" FORCE)
set(CIVETWEB_BUILD_TESTING OFF CACHE BOOL "" FORCE)
set(CIVETWEB_ENABLE_SERVER_EXECUTABLE OFF CACHE BOOL "" FORCE)
set(CIVETWEB_ENABLE_CXX OFF CACHE BOOL "" FORCE)
set(CIVETWEB_ENABLE_ASAN OFF CACHE BOOL "" FORCE)
set(CIVETWEB_INSTALL_EXECUTABLE OFF CACHE BOOL "" FORCE)
FetchContent_Declare(
  civetweb
  GIT_REPOSITORY https://github.com/civetweb/civetweb.git
  GIT_TAG d7ba35bbb649209c66e582d5a0244ba988a15159 # v1.16
  EXCLUDE_FROM_ALL)
set(BUILD_SHARED_LIBS OFF)
FetchContent_MakeAvailable(nlohmann_json pcre2 civetweb)

add_library(qwen3_asr STATIC
  src/audio.cpp
  src/model.cpp
  src/tokenizer.cpp
  src/transcriber.cpp
  src/worker.cpp
  src/realtime.cpp
  src/form.cpp)
target_include_directories(qwen3_asr PUBLIC src)
target_link_libraries(qwen3_asr PUBLIC mlx nlohmann_json::nlohmann_json pcre2-8-static
  "-framework CoreFoundation")
target_compile_options(qwen3_asr PUBLIC ${MLX_CXX_FLAGS} PRIVATE -Wall -Wextra)

add_executable(qwen3_asr_server src/server.cpp)
target_link_libraries(qwen3_asr_server PRIVATE qwen3_asr civetweb-c-library)

add_executable(qwen3_asr_transcribe tools/transcribe.cpp)
target_link_libraries(qwen3_asr_transcribe PRIVATE qwen3_asr)

set_target_properties(qwen3_asr_server qwen3_asr_transcribe PROPERTIES
  BUILD_RPATH "${MLX_LIBRARY_DIRS};@loader_path"
  INSTALL_RPATH "@loader_path")

# Note (Jiaxin Deng): binaries sit next to libmlx.dylib and mlx.metallib,
# which MLX loads from beside its library.
install(TARGETS qwen3_asr_server qwen3_asr_transcribe RUNTIME DESTINATION bin)
# Note (Jiaxin Deng): libmlx.dylib loads sibling dylibs (libjaccl) via @rpath.
file(GLOB MLX_RUNTIME_FILES "${MLX_LIBRARY_DIRS}/*.dylib" "${MLX_LIBRARY_DIRS}/mlx.metallib")
foreach(runtime_file ${MLX_RUNTIME_FILES})
  file(REAL_PATH "${runtime_file}" resolved_file)
  install(FILES "${resolved_file}" DESTINATION bin)
endforeach()
