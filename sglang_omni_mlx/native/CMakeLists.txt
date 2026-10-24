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

add_library(common STATIC
  src/common/audio.cpp
  src/common/form.cpp
  src/common/layers.cpp
  src/common/npz.cpp
  src/common/tokenizer.cpp)
target_include_directories(common PUBLIC src)
target_link_libraries(common PUBLIC mlx nlohmann_json::nlohmann_json pcre2-8-static
  "-framework CoreFoundation")
target_compile_options(common PUBLIC ${MLX_CXX_FLAGS} PRIVATE -Wall -Wextra)
# Note (Jiaxin Deng): npz keeps the no-fused-multiply-add rounding it was built
# with as part of whisper_asr.
set_source_files_properties(src/common/npz.cpp PROPERTIES COMPILE_OPTIONS -ffp-contract=off)

add_library(qwen3_asr STATIC
  src/qwen3_asr/qwen3_asr_model.cpp
  src/qwen3_asr/qwen3_decoder.cpp
  src/qwen3_asr/qwen3_asr_transcriber.cpp
  src/qwen3_asr/worker.cpp
  src/qwen3_asr/realtime.cpp)
target_link_libraries(qwen3_asr PUBLIC common)
target_compile_options(qwen3_asr PRIVATE -Wall -Wextra)

add_library(silero_vad STATIC src/silero_vad/silero_vad.cpp src/silero_vad/speech_segments.cpp)
# Note (Jiaxin Deng): speech_segments keeps the no-fused-multiply-add rounding
# it was built with as part of cohere_asr.
set_source_files_properties(src/silero_vad/speech_segments.cpp PROPERTIES COMPILE_OPTIONS -ffp-contract=off)
target_include_directories(silero_vad PUBLIC src)
target_link_libraries(silero_vad PUBLIC qwen3_asr)
target_compile_options(silero_vad PUBLIC ${MLX_CXX_FLAGS} PRIVATE -Wall -Wextra)

add_library(sortformer STATIC
  src/sortformer/sortformer.cpp
  src/sortformer/sortformer_checkpoint.cpp
  src/sortformer/sortformer_features.cpp
  src/sortformer/sortformer_state.cpp)
target_include_directories(sortformer PUBLIC src)
target_link_libraries(sortformer PUBLIC qwen3_asr)
target_compile_options(sortformer PUBLIC ${MLX_CXX_FLAGS} PRIVATE -Wall -Wextra)

add_library(asr_service STATIC src/qwen3_asr/asr_service.cpp src/common/http.cpp)
target_link_libraries(asr_service PUBLIC qwen3_asr civetweb-c-library)
target_compile_options(asr_service PRIVATE -Wall -Wextra)

add_executable(qwen3_asr_server src/qwen3_asr/qwen3_asr_server.cpp src/silero_vad/vad_service.cpp
  src/sortformer/sortformer_service.cpp)
target_link_libraries(qwen3_asr_server PRIVATE asr_service silero_vad sortformer)

add_executable(silero_vad_probe tools/silero_vad_probe.cpp)
target_link_libraries(silero_vad_probe PRIVATE silero_vad qwen3_asr)

add_executable(sortformer_probe tools/sortformer_probe.cpp)
target_link_libraries(sortformer_probe PRIVATE sortformer qwen3_asr)

add_executable(qwen3_asr_transcribe tools/qwen3_asr_transcribe.cpp)
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
add_library(swift_port STATIC src/common/swift_port.cpp)
target_include_directories(swift_port PUBLIC src)
target_link_libraries(swift_port PUBLIC mlx)
add_library(whisper_asr STATIC src/whisper/whisper_model.cpp src/whisper/whisper_transcriber.cpp)
target_link_libraries(whisper_asr PUBLIC qwen3_asr swift_port)
# Note (khazic): CPU float math rounds each operation, as Swift does: no fused multiply-add.
target_compile_options(swift_port PRIVATE -Wall -Wextra -ffp-contract=off)
target_compile_options(whisper_asr PRIVATE -Wall -Wextra -ffp-contract=off)
add_executable(whisper_server src/whisper/whisper_server.cpp)
target_link_libraries(whisper_server PRIVATE whisper_asr asr_service)
add_executable(whisper_transcribe tools/whisper_transcribe.cpp)
target_link_libraries(whisper_transcribe PRIVATE whisper_asr)
set_target_properties(whisper_server whisper_transcribe PROPERTIES
  BUILD_RPATH "${MLX_LIBRARY_DIRS};@loader_path"
  INSTALL_RPATH "@loader_path")
install(TARGETS whisper_server whisper_transcribe RUNTIME DESTINATION bin)

# Note (khazic): Cohere Transcribe: cohere_transcribe_server, with qwen3_asr_server's API
# without realtime, and its parity tool; Silero VAD cuts long audio at speech.
add_library(cohere_asr STATIC src/cohere/cohere_model.cpp src/cohere/cohere_transcriber.cpp)
target_link_libraries(cohere_asr PUBLIC qwen3_asr swift_port silero_vad)
# Note (khazic): CPU float math rounds each operation, as Swift does: no fused multiply-add.
target_compile_options(cohere_asr PRIVATE -Wall -Wextra -ffp-contract=off)
add_executable(cohere_transcribe_server src/cohere/cohere_transcribe_server.cpp)
target_link_libraries(cohere_transcribe_server PRIVATE cohere_asr asr_service)
add_executable(cohere_transcribe tools/cohere_transcribe.cpp)
target_link_libraries(cohere_transcribe PRIVATE cohere_asr)
set_target_properties(cohere_transcribe_server cohere_transcribe PROPERTIES
  BUILD_RPATH "${MLX_LIBRARY_DIRS};@loader_path"
  INSTALL_RPATH "@loader_path")
install(TARGETS cohere_transcribe_server cohere_transcribe RUNTIME DESTINATION bin)
# Note (Dayuxiaoshui): MOSS-Transcribe-Diarize on Qwen3-ASR's decoder, WAV and
# form handling, with its own server and parity tool.
add_library(moss_asr STATIC src/moss/moss_model.cpp src/moss/moss_transcriber.cpp src/moss/moss_realtime.cpp)
target_link_libraries(moss_asr PUBLIC qwen3_asr)
target_compile_options(moss_asr PRIVATE -Wall -Wextra)
# Note (Dayuxiaoshui): a long recording's cut is the minimum of energy sums,
# which fused multiply-adds would round differently from Voxt's Swift.
set_source_files_properties(src/moss/moss_transcriber.cpp PROPERTIES COMPILE_OPTIONS -ffp-contract=off)
add_executable(moss_transcribe_diarize_server src/moss/moss_transcribe_diarize_server.cpp)
target_link_libraries(moss_transcribe_diarize_server PRIVATE moss_asr asr_service)
add_executable(moss_transcribe tools/moss_transcribe.cpp)
target_link_libraries(moss_transcribe PRIVATE moss_asr)
set_target_properties(moss_transcribe_diarize_server moss_transcribe PROPERTIES
  BUILD_RPATH "${MLX_LIBRARY_DIRS};@loader_path"
  INSTALL_RPATH "@loader_path")
install(TARGETS moss_transcribe_diarize_server moss_transcribe RUNTIME DESTINATION bin)
# Note (Dayuxiaoshui): Nemotron 3.5 ASR streaming on Sortformer's NeMo front end
# and Qwen3-ASR's checkpoint loading, with its own server and parity tool.
add_library(nemotron_asr STATIC src/nemotron/nemotron_model.cpp src/nemotron/nemotron_transcriber.cpp
  src/nemotron/nemotron_realtime.cpp)
target_link_libraries(nemotron_asr PUBLIC qwen3_asr sortformer swift_port)
# Note (Dayuxiaoshui): the positional table is float math that must round as
# Swift's does, so no fused multiply-add.
target_compile_options(nemotron_asr PRIVATE -Wall -Wextra -ffp-contract=off)
add_executable(nemotron_asr_server src/nemotron/nemotron_asr_server.cpp)
target_link_libraries(nemotron_asr_server PRIVATE nemotron_asr asr_service)
add_executable(nemotron_transcribe tools/nemotron_transcribe.cpp)
target_link_libraries(nemotron_transcribe PRIVATE nemotron_asr)
set_target_properties(nemotron_asr_server nemotron_transcribe PROPERTIES
  BUILD_RPATH "${MLX_LIBRARY_DIRS};@loader_path"
  INSTALL_RPATH "@loader_path")
install(TARGETS nemotron_asr_server nemotron_transcribe RUNTIME DESTINATION bin)

# Note (Dayuxiaoshui): Parakeet TDT: parakeet_server, with qwen3_asr_server's API without
# realtime, and its parity tool; the front end is Sortformer's NeMo log-mel.
add_library(parakeet_asr STATIC src/parakeet/parakeet_model.cpp src/parakeet/parakeet_transcriber.cpp)
target_link_libraries(parakeet_asr PUBLIC qwen3_asr sortformer swift_port)
target_compile_options(parakeet_asr PRIVATE -Wall -Wextra -ffp-contract=off)
add_executable(parakeet_server src/parakeet/parakeet_server.cpp)
target_link_libraries(parakeet_server PRIVATE parakeet_asr asr_service)
add_executable(parakeet_transcribe tools/parakeet_transcribe.cpp)
target_link_libraries(parakeet_transcribe PRIVATE parakeet_asr)
set_target_properties(parakeet_server parakeet_transcribe PROPERTIES
  BUILD_RPATH "${MLX_LIBRARY_DIRS};@loader_path"
  INSTALL_RPATH "@loader_path")
install(TARGETS parakeet_server parakeet_transcribe RUNTIME DESTINATION bin)

# Note (Dayuxiaoshui): SenseVoice Small: sensevoice_server, with qwen3_asr_server's API
# without realtime, and its parity tool; Silero VAD cuts long audio at speech.
add_library(sensevoice_asr STATIC src/sensevoice/sensevoice_model.cpp src/sensevoice/sensevoice_transcriber.cpp)
target_link_libraries(sensevoice_asr PUBLIC qwen3_asr swift_port silero_vad)
target_compile_options(sensevoice_asr PRIVATE -Wall -Wextra -ffp-contract=off)
add_executable(sensevoice_server src/sensevoice/sensevoice_server.cpp)
target_link_libraries(sensevoice_server PRIVATE sensevoice_asr asr_service)
add_executable(sensevoice_transcribe tools/sensevoice_transcribe.cpp)
target_link_libraries(sensevoice_transcribe PRIVATE sensevoice_asr)
set_target_properties(sensevoice_server sensevoice_transcribe PROPERTIES
  BUILD_RPATH "${MLX_LIBRARY_DIRS};@loader_path"
  INSTALL_RPATH "@loader_path")
install(TARGETS sensevoice_server sensevoice_transcribe RUNTIME DESTINATION bin)

# Note (Minyi Tang): Voxtral 4B TTS: voxtral_tts_server, with the CUDA server's
# /v1/audio/speech API, and its parity tool; the text decoder reuses Qwen3's.
add_library(voxtral_tts STATIC src/voxtral_tts/voxtral_tts_tokenizer.cpp
  src/voxtral_tts/voxtral_tts_codec.cpp src/voxtral_tts/voxtral_tts_model.cpp)
target_link_libraries(voxtral_tts PUBLIC qwen3_asr)
target_compile_options(voxtral_tts PRIVATE -Wall -Wextra -ffp-contract=off)
add_executable(voxtral_tts_server src/voxtral_tts/voxtral_tts_server.cpp)
target_link_libraries(voxtral_tts_server PRIVATE voxtral_tts asr_service)
target_compile_options(voxtral_tts_server PRIVATE -Wall -Wextra)
add_executable(voxtral_tts_synthesize tools/voxtral_tts_synthesize.cpp)
target_link_libraries(voxtral_tts_synthesize PRIVATE voxtral_tts)
set_target_properties(voxtral_tts_server voxtral_tts_synthesize PROPERTIES
  BUILD_RPATH "${MLX_LIBRARY_DIRS};@loader_path"
  INSTALL_RPATH "@loader_path")
install(TARGETS voxtral_tts_server voxtral_tts_synthesize RUNTIME DESTINATION bin)
