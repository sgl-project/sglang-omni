// SPDX-License-Identifier: Apache-2.0
#include <algorithm>
#include <fstream>
#include <functional>
#include <iostream>

#include "qwen3_tts/qwen3_tts_model.h"

int main(int argc, char **argv) {
  try {
    std::filesystem::path directory, request_path, output_path, codes_path,
        decode_path;
    int decode_chunk_frames = qwen3_tts::kMaximumFrames;
    for (int index = 1; index < argc; ++index) {
      const std::string flag = argv[index];
      if (index + 1 == argc) {
        throw std::invalid_argument(flag + " needs a value");
      } else if (flag == "--model-path") {
        directory = argv[++index];
      } else if (flag == "--request") {
        request_path = argv[++index];
      } else if (flag == "--output") {
        output_path = argv[++index];
      } else if (flag == "--codes-output") {
        codes_path = argv[++index];
      } else if (flag == "--decode-chunk-frames") {
        decode_chunk_frames = std::stoi(argv[++index]);
      } else if (flag == "--decode-codes") {
        decode_path = argv[++index];
      } else {
        throw std::invalid_argument("unknown argument " + flag);
      }
    }
    if (directory.empty() || output_path.empty() ||
        (request_path.empty() == decode_path.empty())) {
      throw std::invalid_argument(
          "usage: qwen3_tts_synthesize --model-path DIR (--request JSON | "
          "--decode-codes JSON) --output WAV [--codes-output JSON]");
    } else {
    }
    if (decode_chunk_frames < 1 ||
        decode_chunk_frames > qwen3_tts::kMaximumFrames) {
      throw std::invalid_argument("decode-chunk-frames must be in [1,2048]");
    } else {
    }
    mlx::core::set_cache_limit(1024ULL * 1024 * 1024);
    const std::function<void()> check_cancelled = [] {};
    qwen3_tts::Result result;
    nlohmann::json diagnostics;
    if (!decode_path.empty()) {
      const auto frames =
          qwen3_asr::ReadJson(decode_path).get<std::vector<std::vector<int>>>();
      if (frames.empty() || frames.size() > qwen3_tts::kMaximumFrames) {
        throw std::invalid_argument(
            "fixed codes must contain 1 to 2048 frames");
      } else {
      }
      std::vector<int> codes;
      for (const auto &frame : frames) {
        if (frame.size() != qwen3_tts::kCodeGroups) {
          throw std::invalid_argument("fixed code frame must contain 16 codes");
        } else {
        }
        for (const auto code : frame) {
          if (code < 0 || code >= qwen3_tts::kCodebookSize) {
            throw std::invalid_argument("fixed code is outside [0,2048)");
          } else {
            codes.push_back(code);
          }
        }
      }
      qwen3_tts::AudioCodec codec(directory / "speech_tokenizer", false);
      qwen3_tts::AudioCodec::State state;
      const auto code_array = mlx::core::array(
          codes.data(),
          {static_cast<int>(frames.size()), qwen3_tts::kCodeGroups},
          mlx::core::int32);
      for (int start = 0; start < static_cast<int>(frames.size());
           start += decode_chunk_frames) {
        const int end = std::min(start + decode_chunk_frames,
                                 static_cast<int>(frames.size()));
        auto waveform = mlx::core::astype(
            codec.Decode(mlx::core::slice(code_array, {start, 0},
                                          {end, qwen3_tts::kCodeGroups}),
                         state, check_cancelled),
            mlx::core::float32);
        mlx::core::eval(waveform);
        result.samples.insert(result.samples.end(), waveform.data<float>(),
                              waveform.data<float>() + waveform.size());
      }
      result.codes = frames;
      result.finish_reason = "fixed_codes";
    } else {
      qwen3_tts::Model model(directory);
      auto body = qwen3_asr::ReadJson(request_path);
      body["stream"] = false;
      const auto request = model.ParseRequest(body, "");
      result = model.Synthesize(request, check_cancelled);
      diagnostics["prompt_ids"] = request.text_ids;
      diagnostics["control_ids"] = request.control_ids;
      diagnostics["instruction_ids"] = request.instruction_ids;
      diagnostics["reference_text_ids"] = request.reference_text_ids;
    }
    diagnostics["codes"] = result.codes;
    diagnostics["finish_reason"] = result.finish_reason;
    diagnostics["sample_count"] = result.samples.size();
    diagnostics["sample_rate"] = qwen3_tts::kSampleRate;
    const auto audio = qwen3_tts::EncodeAudio(result.samples, false);
    std::ofstream output(output_path, std::ios::binary);
    output.write(audio.data(), audio.size());
    output.close();
    if (!output) {
      throw std::runtime_error("cannot write " + output_path.string());
    } else {
    }
    if (!codes_path.empty()) {
      std::ofstream codes(codes_path);
      codes << diagnostics.dump(2) << '\n';
      codes.close();
      if (!codes) {
        throw std::runtime_error("cannot write " + codes_path.string());
      } else {
      }
    } else {
    }
    std::cout << diagnostics.dump() << '\n';
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "qwen3_tts_synthesize: " << error.what() << '\n';
    return 1;
  }
}
