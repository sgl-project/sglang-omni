# SPDX-License-Identifier: Apache-2.0
# Native MLX runtime for Voxt (Qwen3-ASR, Silero VAD): one binary, no Python.
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
  src/layers.cpp
  src/model.cpp
  src/qwen3_decoder.cpp
  src/tokenizer.cpp
  src/transcriber.cpp
  src/worker.cpp
  src/realtime.cpp
  src/form.cpp)
target_include_directories(qwen3_asr PUBLIC src)
target_link_libraries(qwen3_asr PUBLIC mlx nlohmann_json::nlohmann_json pcre2-8-static
  "-framework CoreFoundation")
target_compile_options(qwen3_asr PUBLIC ${MLX_CXX_FLAGS} PRIVATE -Wall -Wextra)

add_library(silero_vad STATIC src/silero_vad.cpp)
target_include_directories(silero_vad PUBLIC src)
target_link_libraries(silero_vad PUBLIC mlx nlohmann_json::nlohmann_json)
target_compile_options(silero_vad PUBLIC ${MLX_CXX_FLAGS} PRIVATE -Wall -Wextra)

add_library(sortformer STATIC
  src/sortformer.cpp
  src/sortformer_checkpoint.cpp
  src/sortformer_features.cpp
  src/sortformer_state.cpp)
target_include_directories(sortformer PUBLIC src)
target_link_libraries(sortformer PUBLIC mlx nlohmann_json::nlohmann_json)
target_compile_options(sortformer PUBLIC ${MLX_CXX_FLAGS} PRIVATE -Wall -Wextra)

add_library(asr_service STATIC src/asr_service.cpp src/http.cpp)
target_link_libraries(asr_service PUBLIC qwen3_asr civetweb-c-library)
target_compile_options(asr_service PRIVATE -Wall -Wextra)

add_executable(qwen3_asr_server src/server.cpp src/vad_service.cpp
  src/sortformer_service.cpp)
target_link_libraries(qwen3_asr_server PRIVATE asr_service silero_vad sortformer)

add_executable(silero_vad_probe tools/silero_vad_probe.cpp)
target_link_libraries(silero_vad_probe PRIVATE silero_vad qwen3_asr)

add_executable(sortformer_probe tools/sortformer_probe.cpp)
target_link_libraries(sortformer_probe PRIVATE sortformer qwen3_asr)

add_executable(qwen3_asr_transcribe tools/transcribe.cpp)
target_link_libraries(qwen3_asr_transcribe PRIVATE qwen3_asr)

set_target_properties(qwen3_asr_server qwen3_asr_transcribe silero_vad_probe
  sortformer_probe PROPERTIES
  BUILD_RPATH "${MLX_LIBRARY_DIRS};@loader_path"
  INSTALL_RPATH "@loader_path")

# Note (Jiaxin Deng): binaries sit next to libmlx.dylib and mlx.metallib,
# which MLX loads from beside its library.
install(TARGETS qwen3_asr_server qwen3_asr_transcribe silero_vad_probe
  sortformer_probe RUNTIME DESTINATION bin)
# Note (Jiaxin Deng): libmlx.dylib loads sibling dylibs (libjaccl) via @rpath.
file(GLOB MLX_RUNTIME_FILES "${MLX_LIBRARY_DIRS}/*.dylib" "${MLX_LIBRARY_DIRS}/mlx.metallib")
foreach(runtime_file ${MLX_RUNTIME_FILES})
  file(REAL_PATH "${runtime_file}" resolved_file)
  install(FILES "${resolved_file}" DESTINATION bin)
endforeach()

# Note (khazic): Whisper: whisper_server, with qwen3_asr_server's API without realtime, and
# its parity tool. Both reuse the WAV decoder, form parser and byte-level
# tokenizer above.
add_library(swift_port STATIC src/swift_port.cpp)
target_link_libraries(swift_port PUBLIC mlx)
add_library(whisper_asr STATIC src/whisper_model.cpp src/whisper_transcriber.cpp
  src/npz.cpp)
target_link_libraries(whisper_asr PUBLIC qwen3_asr swift_port)
# Note (khazic): CPU float math rounds each operation, as Swift does: no fused multiply-add.
target_compile_options(swift_port PRIVATE -Wall -Wextra -ffp-contract=off)
target_compile_options(whisper_asr PRIVATE -Wall -Wextra -ffp-contract=off)
add_executable(whisper_server src/whisper_server.cpp)
target_link_libraries(whisper_server PRIVATE whisper_asr asr_service)
add_executable(whisper_transcribe tools/whisper_transcribe.cpp)
target_link_libraries(whisper_transcribe PRIVATE whisper_asr)
set_target_properties(whisper_server whisper_transcribe PROPERTIES
  BUILD_RPATH "${MLX_LIBRARY_DIRS};@loader_path"
  INSTALL_RPATH "@loader_path")
install(TARGETS whisper_server whisper_transcribe RUNTIME DESTINATION bin)

# Note (khazic): Cohere Transcribe: cohere_transcribe_server, with qwen3_asr_server's API
# without realtime, and its parity tool; Silero VAD cuts long audio at speech.
add_library(cohere_asr STATIC src/cohere_model.cpp src/cohere_transcriber.cpp
  src/speech_segments.cpp)
target_link_libraries(cohere_asr PUBLIC qwen3_asr swift_port silero_vad)
# Note (khazic): CPU float math rounds each operation, as Swift does: no fused multiply-add.
target_compile_options(cohere_asr PRIVATE -Wall -Wextra -ffp-contract=off)
add_executable(cohere_transcribe_server src/cohere_transcribe_server.cpp)
target_link_libraries(cohere_transcribe_server PRIVATE cohere_asr asr_service)
add_executable(cohere_transcribe tools/cohere_transcribe.cpp)
target_link_libraries(cohere_transcribe PRIVATE cohere_asr)
set_target_properties(cohere_transcribe_server cohere_transcribe PROPERTIES
  BUILD_RPATH "${MLX_LIBRARY_DIRS};@loader_path"
  INSTALL_RPATH "@loader_path")
install(TARGETS cohere_transcribe_server cohere_transcribe RUNTIME DESTINATION bin)
# Note (Dayuxiaoshui): MOSS-Transcribe-Diarize on Qwen3-ASR's decoder, WAV and
# form handling, with its own server and parity tool.
add_library(moss_asr STATIC src/moss_model.cpp src/moss_transcriber.cpp src/moss_realtime.cpp)
target_link_libraries(moss_asr PUBLIC qwen3_asr)
target_compile_options(moss_asr PRIVATE -Wall -Wextra)
# Note (Dayuxiaoshui): a long recording's cut is the minimum of energy sums,
# which fused multiply-adds would round differently from Voxt's Swift.
set_source_files_properties(src/moss_transcriber.cpp PROPERTIES COMPILE_OPTIONS -ffp-contract=off)
add_executable(moss_transcribe_diarize_server src/moss_transcribe_diarize_server.cpp)
target_link_libraries(moss_transcribe_diarize_server PRIVATE moss_asr asr_service)
add_executable(moss_transcribe tools/moss_transcribe.cpp)
target_link_libraries(moss_transcribe PRIVATE moss_asr)
set_target_properties(moss_transcribe_diarize_server moss_transcribe PROPERTIES
  BUILD_RPATH "${MLX_LIBRARY_DIRS};@loader_path"
  INSTALL_RPATH "@loader_path")
install(TARGETS moss_transcribe_diarize_server moss_transcribe RUNTIME DESTINATION bin)

# Note (Dayuxiaoshui): Parakeet TDT: parakeet_server, with qwen3_asr_server's API without
# realtime, and its parity tool; the front end is Sortformer's NeMo log-mel.
add_library(parakeet_asr STATIC src/parakeet_model.cpp src/parakeet_transcriber.cpp)
target_link_libraries(parakeet_asr PUBLIC qwen3_asr sortformer swift_port)
target_compile_options(parakeet_asr PRIVATE -Wall -Wextra -ffp-contract=off)
add_executable(parakeet_server src/parakeet_server.cpp)
target_link_libraries(parakeet_server PRIVATE parakeet_asr asr_service)
add_executable(parakeet_transcribe tools/parakeet_transcribe.cpp)
target_link_libraries(parakeet_transcribe PRIVATE parakeet_asr)
set_target_properties(parakeet_server parakeet_transcribe PROPERTIES
  BUILD_RPATH "${MLX_LIBRARY_DIRS};@loader_path"
  INSTALL_RPATH "@loader_path")
install(TARGETS parakeet_server parakeet_transcribe RUNTIME DESTINATION bin)
