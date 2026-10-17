// SPDX-License-Identifier: Apache-2.0
// AuK component outputs from explicit inputs for checkpoint parity checks.
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "auk/auk_conditioner.h"
#include "auk/auk_dit.h"
#include "auk/auk_vae.h"

namespace {

namespace mx = mlx::core;

std::vector<int> IntegerVector(const mx::array &input) {
  if (input.ndim() != 1 || input.size() == 0) {
    throw std::invalid_argument(
        "token ids and attention mask must be nonempty vectors");
  } else {
  }
  const auto integers = mx::contiguous(mx::astype(input, mx::int32));
  mx::eval(integers);
  const auto *values = integers.data<int32_t>();
  return std::vector<int>(values, values + integers.size());
}

} // namespace

int main(int argc, char **argv) {
  try {
    std::filesystem::path model_path, input_path, output_path, options_path;
    std::string component, dtype_name;
    for (int i = 1; i < argc; ++i) {
      const std::string argument = argv[i];
      if (argument == "--help") {
        std::cout
            << "auk_probe --model-path DIR --component vae|dit|conditioner "
               "--input FILE.safetensors --output FILE.safetensors "
               "[--options FILE.json] [--dtype float32|bfloat16]\n";
        return 0;
      } else if (i + 1 >= argc) {
        throw std::invalid_argument("missing value for " + argument);
      } else if (argument == "--model-path") {
        model_path = argv[++i];
      } else if (argument == "--component") {
        component = argv[++i];
      } else if (argument == "--input") {
        input_path = argv[++i];
      } else if (argument == "--output") {
        output_path = argv[++i];
      } else if (argument == "--options") {
        options_path = argv[++i];
      } else if (argument == "--dtype") {
        dtype_name = argv[++i];
      } else {
        throw std::invalid_argument("unknown argument " + argument);
      }
    }
    if (model_path.empty() || input_path.empty() || output_path.empty() ||
        (component != "vae" && component != "dit" &&
         component != "conditioner")) {
      throw std::invalid_argument(
          "model-path, component, input and output are required");
    } else {
    }
    const auto options = options_path.empty()
                             ? nlohmann::json::object()
                             : qwen3_asr::ReadJson(options_path);
    if (dtype_name.empty()) {
      dtype_name = options.value("dtype", std::string("float32"));
    } else {
    }
    if (dtype_name != "float32" && dtype_name != "bfloat16") {
      throw std::invalid_argument("dtype must be float32 or bfloat16");
    } else {
    }
    const mx::Dtype dtype =
        dtype_name == "float32" ? mx::float32 : mx::bfloat16;
    const auto [inputs, metadata] = mx::load_safetensors(input_path.string());
    qwen3_asr::WeightMap outputs;
    if (component == "vae") {
      const auk::VAE model(model_path / "vae");
      const auto posterior = model.Posterior(inputs.at("waveform"));
      outputs.emplace("mean", posterior.mean);
      outputs.emplace("log_standard_deviation",
                      posterior.log_standard_deviation);
      outputs.emplace("encoded",
                      model.Encode(posterior, inputs.at("posterior_noise")));
      if (inputs.contains("latents")) {
        outputs.emplace("decoded", model.Decode(inputs.at("latents")));
      } else {
      }
    } else if (component == "dit") {
      const auk::DiT model(model_path / "dit", dtype);
      const auto &conditioning = inputs.at("conditioning");
      const auto &noise = inputs.at("noise");
      const auto reference = inputs.contains("reference")
                                 ? std::optional(inputs.at("reference"))
                                 : std::nullopt;
      const int reference_length = options.value(
          "reference_length", reference ? reference->shape(0) : 0);
      const float guidance_strength = options.at("cfg_strength").get<float>();
      const auto sway =
          options.contains("sway_sampling_coef") &&
                  !options.at("sway_sampling_coef").is_null()
              ? std::optional(options.at("sway_sampling_coef").get<float>())
              : std::nullopt;
      const auto time_grid = options.value("time_grid", std::vector<float>{});
      outputs.emplace("forward",
                      model.Forward(noise, conditioning,
                                    options.at("timestep").get<float>(),
                                    reference, reference_length,
                                    guidance_strength >= 1e-5f));
      outputs.emplace("sampled",
                      model.Sample(conditioning, reference, reference_length,
                                   noise, options.at("steps").get<int>(),
                                   guidance_strength, sway, time_grid, [] {}));
      if (inputs.contains("hidden_states")) {
        outputs.emplace("fused",
                        model.FuseHiddenStates(inputs.at("hidden_states")));
      } else {
      }
    } else {
      const auk::Conditioner model(model_path / "conditioner", dtype);
      const auto features = inputs.contains("features")
                                ? std::optional(inputs.at("features"))
                                : std::nullopt;
      outputs.emplace(
          "hidden_states",
          model.EncodeInputs(IntegerVector(inputs.at("input_ids")),
                             IntegerVector(inputs.at("attention_mask")),
                             features));
      if (features) {
        outputs.emplace("audio_embeddings", model.EncodeAudio(*features));
      } else {
      }
      if (options.contains("instruction")) {
        const auto token_ids =
            model.Tokenize(options.at("instruction").get<std::string>(),
                           features ? features->shape(1) : 0);
        outputs.emplace("token_ids",
                        mx::array(token_ids.data(),
                                  {static_cast<int>(token_ids.size())},
                                  mx::int32));
      } else {
      }
      if (inputs.contains("reference_audio_16k") ||
          options.value("evaluate_frontend", false)) {
        std::optional<std::vector<float>> reference_audio;
        if (inputs.contains("reference_audio_16k")) {
          const auto &input_audio = inputs.at("reference_audio_16k");
          if (input_audio.ndim() != 1 || input_audio.size() == 0) {
            throw std::invalid_argument(
                "reference_audio_16k must be a nonempty vector");
          } else {
          }
          const auto waveform =
              mx::contiguous(mx::astype(input_audio, mx::float32));
          mx::eval(waveform);
          reference_audio = std::vector<float>(
              waveform.data<float>(), waveform.data<float>() + waveform.size());
        } else {
        }
        outputs.emplace(
            "frontend_hidden_states",
            model.Encode(options.at("instruction").get<std::string>(),
                         reference_audio));
      } else {
      }
    }
    mx::save_safetensors(output_path.string(), std::move(outputs));
    std::cout << output_path.string() << '\n';
    return 0;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
