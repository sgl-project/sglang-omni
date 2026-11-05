// SPDX-License-Identifier: Apache-2.0
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <sys/stat.h>

#include "common/audio.h"
#include "mlx/random.h"
#include "qwen3_omni/generation.h"
#include "qwen3_omni/media.h"
#include "qwen3_omni/model.h"
#include "qwen3_omni/request.h"

namespace mx = mlx::core;
using namespace qwen3_omni;

void check(bool condition, const char *message) {
  if (!condition) {
    throw std::runtime_error(message);
  } else {
  }
}

void check_configuration_rejections() {
  const Json text = {{"hidden_size", 128},
                     {"head_dim", 128},
                     {"num_attention_heads", 2},
                     {"num_key_value_heads", 1},
                     {"num_hidden_layers", 1},
                     {"intermediate_size", 128},
                     {"hidden_act", "silu"},
                     {"rms_norm_eps", 1e-6},
                     {"rope_theta", 10000},
                     {"vocab_size", 4096},
                     {"num_experts", 2},
                     {"num_experts_per_tok", 1},
                     {"moe_intermediate_size", 128},
                     {"norm_topk_prob", true},
                     {"rope_scaling", {{"mrope_section", {24, 20, 20}}}}};
  Json config = {
      {"model_type", "qwen3_omni_moe"},
      {"enable_audio_output", true},
      {"quantization", {{"bits", 4}, {"group_size", 64}, {"mode", "affine"}}},
      {"thinker_config",
       {{"text_config", text},
        {"image_token_id", 151655},
        {"video_token_id", 151656},
        {"audio_token_id", 151675},
        {"audio_config",
         {{"d_model", 128},
          {"encoder_attention_heads", 2},
          {"encoder_ffn_dim", 128},
          {"encoder_layers", 1},
          {"downsample_hidden_size", 64},
          {"num_mel_bins", 128},
          {"output_dim", 128},
          {"n_window", 50},
          {"n_window_infer", 800}}},
        {"vision_config",
         {{"hidden_size", 128},
          {"intermediate_size", 128},
          {"depth", 1},
          {"num_heads", 2},
          {"out_hidden_size", 128},
          {"patch_size", 16},
          {"temporal_patch_size", 2},
          {"spatial_merge_size", 2},
          {"deepstack_visual_indexes", {0}}}}}},
      {"talker_config",
       {{"text_config", text},
        {"code_predictor_config", text},
        {"num_code_groups", 16},
        {"thinker_hidden_size", 128},
        {"speaker_id", {{"chelsie", 2301}}}}},
      {"code2wav_config",
       {{"hidden_size", 128},
        {"intermediate_size", 128},
        {"num_hidden_layers", 1},
        {"num_attention_heads", 2},
        {"num_key_value_heads", 2},
        {"sliding_window", 72},
        {"decoder_dim", 128},
        {"codebook_size", 2048},
        {"num_quantizers", 16},
        {"rms_norm_eps", 1e-6},
        {"rope_theta", 10000},
        {"upsample_rates", {8, 5, 4, 3}},
        {"upsampling_ratios", {2, 2}}}}};
  check(validate_configuration(config) == config,
        "valid architecture must remain supported");
  auto directory =
      std::filesystem::current_path() / ".local-artifacts/omni-invalid-config";
  std::filesystem::create_directories(directory);
  for (const auto &[pointer, value] : std::vector<std::pair<std::string, Json>>{
           {"/thinker_config/audio_config/n_window", 0},
           {"/thinker_config/audio_config/n_window", 49},
           {"/thinker_config/audio_config/n_window_infer", 0},
           {"/thinker_config/audio_config/encoder_attention_heads", 0},
           {"/thinker_config/text_config/norm_topk_prob", false},
           {"/thinker_config/text_config/num_key_value_heads", 0},
           {"/talker_config/text_config/num_attention_heads", 0},
           {"/talker_config/code_predictor_config/num_hidden_layers", 0},
           {"/thinker_config/text_config/num_experts_per_tok", 3},
           {"/thinker_config/text_config/decoder_sparse_step", 2},
           {"/thinker_config/vision_config/num_heads", 0},
           {"/code2wav_config/upsample_rates/0", 0},
           {"/code2wav_config/num_key_value_heads", 0}}) {
    auto invalid = config;
    invalid[Json::json_pointer(pointer)] = value;
    std::ofstream(directory / "config.json") << invalid;
    bool rejected = false;
    try {
      OmniModel model(directory);
    } catch (const std::invalid_argument &error) {
      rejected = std::string(error.what())
                     .starts_with("unsupported Omni configuration:");
    }
    check(rejected, "invalid architecture must fail before tokenizer, shards "
                    "or component construction");
  }
  std::filesystem::remove_all(directory);
}

void check_readonly_video() {
  const auto original = std::filesystem::current_path();
  const auto root = original / ".local-artifacts/omni-video-regression";
  const auto readonly = root / "readonly";
  const auto temporary = root / "writable-temp";
  std::filesystem::create_directories(readonly);
  std::filesystem::create_directories(temporary);
  const char *previous = std::getenv("TMPDIR");
  const std::optional<std::string> previous_temp =
      previous ? std::optional<std::string>(previous) : std::nullopt;
  struct Restore {
    std::filesystem::path original, root, readonly;
    std::optional<std::string> previous_temp;
    ~Restore() {
      std::filesystem::current_path(original);
      std::filesystem::permissions(readonly, std::filesystem::perms::owner_all);
      std::filesystem::remove_all(root);
      if (previous_temp) {
        setenv("TMPDIR", previous_temp->c_str(), 1);
      } else {
        unsetenv("TMPDIR");
      }
    }
  } restore{original, root, readonly, previous_temp};
  setenv("TMPDIR", temporary.c_str(), 1);
  const auto bytes =
      media_bytes(std::string(OMNI_TEST_DATA_DIR) + "/draw.mp4", {});
  const auto source = "data:video/mp4;base64," + encode_base64(bytes);
  auto request = parse_request(
      {{"messages", {{{"role", "user"}, {"content", "describe"}}}},
       {"video_max_frames", 2},
       {"video_max_pixels", 200704}},
      "qwen3-omni-mlx", {});
  std::filesystem::permissions(readonly,
                               std::filesystem::perms::owner_read |
                                   std::filesystem::perms::owner_exec);
  std::filesystem::current_path(readonly);
  auto result = video_pixels(source, request, {});
  check(result.patches.shape(0) > 0 && std::filesystem::is_empty(temporary),
        "inline video must work from read-only cwd and remove temporary input");
  bool cancelled = false;
  try {
    video_pixels(source, request, [&] {
      for (const auto &entry : std::filesystem::directory_iterator(temporary)) {
        struct stat info {};
        check(stat(entry.path().c_str(), &info) == 0 &&
                  (info.st_mode & 0777) == 0600,
              "materialized media must have private permissions");
        return true;
      }
      return false;
    });
  } catch (const Cancelled &) {
    cancelled = true;
  }
  check(cancelled && std::filesystem::is_empty(temporary),
        "cancelled video preprocessing must remove its exclusive file");
  bool invalid = false;
  try {
    video_pixels("data:video/mp4;base64,AA==", request, {});
  } catch (const std::invalid_argument &) {
    invalid = true;
  }
  check(invalid && std::filesystem::is_empty(temporary),
        "invalid video preprocessing must remove its exclusive file");
}

int main() {
  try {
    mx::set_default_device(mx::Device::cpu);
    check_configuration_rejections();
    check_readonly_video();
    auto large_sample =
        sample(mx::reshape(mx::arange(152064, mx::float32), {1, -1}), {},
               Sampling{1, 1, 100000, 1});
    std::set<std::uintptr_t> visited;
    std::function<void(const mx::array &)> check_graph =
        [&](const mx::array &node) {
          if (!visited.insert(node.id()).second) {
            return;
          } else {
          }
          check(node.size() <= 4 * 152064,
                "top-k intermediates must scale with vocabulary, not "
                "vocabulary times k");
          for (const auto &input : node.inputs()) {
            check_graph(input);
          }
        };
    check_graph(large_sample);
    check(large_sample.item<uint32_t>() >= 52064,
          "large-vocabulary sample must remain in the selected top-k");
    for (int seed = 0; seed < 8; ++seed) {
      mx::random::seed(seed);
      auto actual = sample(mx::array({1.f, 4.f, 2.f, 3.f}, {1, 4}), {},
                           Sampling{1, 1, 2, 1})
                        .item<uint32_t>();
      mx::random::seed(seed);
      auto expected = mx::random::categorical(
                          mx::array({-INFINITY, 4.f, -INFINITY, 3.f}, {1, 4}))
                          .item<uint32_t>();
      check(actual == expected,
            "top-k must preserve seeded vocabulary sampling");
    }
    auto tokenizer_directory = std::filesystem::current_path() /
                               ".local-artifacts/omni-component-tokenizer";
    std::filesystem::create_directories(tokenizer_directory);
    std::ofstream(tokenizer_directory / "vocab.json")
        << R"({"a":0,"b":1,"ab":2})";
    std::ofstream(tokenizer_directory / "tokenizer_config.json")
        << R"({"added_tokens_decoder":{}})";
    std::ofstream(tokenizer_directory / "tokenizer.json")
        << R"({"model":{"vocab":{"a":0,"b":1,"ab":2},"merges":[["a","b"]]},"added_tokens":[]})";
    omni_common::Tokenizer tokenizer(tokenizer_directory);
    check(tokenizer.Encode("ab") == std::vector<int>{2},
          "tokenizer.json merge pairs must work without merges.txt");
    Json chat_vocabulary = {{"a", 0}, {"b", 1}, {"ab", 2}, {"u", 3},
                            {"s", 4}, {"e", 5}, {"r", 6},  {"i", 7},
                            {"t", 8}, {"n", 9}, {"Ċ", 10}};
    std::ofstream(tokenizer_directory / "vocab.json") << chat_vocabulary;
    std::ofstream(tokenizer_directory / "tokenizer_config.json")
        << R"({"added_tokens_decoder":{"11":{"content":"<|im_start|>","special":true},"12":{"content":"<|im_end|>","special":true},"13":{"content":"<think>","special":true},"14":{"content":"</think>","special":true}}})";
    omni_common::Tokenizer chat_tokenizer(tokenizer_directory);
    auto split_text =
        parse_request({{"messages",
                        {{{"role", "user"},
                          {"content",
                           {{{"type", "text"}, {"text", "a"}},
                            {{"type", "text"}, {"text", "b"}}}}}}}},
                      "qwen3-omni-mlx", {});
    auto prepared_text = prepare(split_text, chat_tokenizer, {});
    check(std::count(prepared_text.tokens.begin(), prepared_text.tokens.end(),
                     2) == 1,
          "adjacent text content parts must share BPE segmentation");
    auto assistant_history =
        parse_request({{"messages",
                        {{{"role", "user"}, {"content", "a"}},
                         {{"role", "assistant"}, {"content", "b"}}}}},
                      "qwen3-omni-mlx", {});
    auto prepared_history = prepare(assistant_history, chat_tokenizer, {});
    check(
        chat_tokenizer.Decode(prepared_history.tokens, false) ==
            "<|im_start|>user\na<|im_end|>\n<|im_start|>assistant\n<think>\n\n<"
            "/think>\n\nb<|im_end|>\n<|im_start|>assistant\n",
        "trailing assistant history must follow checkpoint reasoning template");
    std::filesystem::remove_all(tokenizer_directory);
    auto packed = mx::quantize(mx::ones({2, 64}), 64, 4);
    omni_common::WeightMap tensors;
    tensors.emplace("packed.weight", packed[0]);
    tensors.emplace("packed.scales", packed[1]);
    tensors.emplace("packed.biases", packed[2]);
    tensors.emplace("dense.weight", mx::full({2, 64}, 2.0f));
    auto expert_packed = mx::quantize(
        mx::stack({mx::ones({2, 64}), mx::full({2, 64}, 2.f)}), 64, 4);
    tensors.emplace("experts.weight", expert_packed[0]);
    tensors.emplace("experts.scales", expert_packed[1]);
    tensors.emplace("experts.biases", expert_packed[2]);
    Weights weights(std::move(tensors), 64, 4);
    weights.require_linear("packed", 64, 2);
    weights.require_linear("dense", 64, 2);
    weights.require_linear("experts", 64, 2, false, 2);
    weights.finish_validation();
    check(mx::sum(weights.linear("packed", mx::ones({1, 64}))).item<float>() ==
              128,
          "packed linear must use affine scales and offsets");
    check(mx::sum(weights.linear("dense", mx::ones({1, 64}))).item<float>() ==
              256,
          "dense weights must not be treated as packed");
    auto embedded = weights.embedding("packed", mx::array({1, 0}, {2}));
    check(mx::sum(embedded).item<float>() == 128,
          "packed embedding must dequantize selected rows");
    auto selected_experts = weights.linear(
        "experts", mx::ones({1, 1, 1, 1, 64}), mx::array({1, 0}, {1, 1, 2}));
    mx::eval(selected_experts);
    check(selected_experts.shape() == mx::Shape({1, 1, 2, 1, 2}) &&
              selected_experts.data<float>()[0] == 128 &&
              selected_experts.data<float>()[2] == 64,
          "quantized expert gather must preserve routing order and "
          "expert-specific offsets");
    omni_common::WeightMap expert_tensors;
    std::string layer = "tiny.layers.0";
    for (const auto &name : {"input_layernorm", "post_attention_layernorm"}) {
      expert_tensors.emplace(layer + "." + name + ".weight", mx::ones({64}));
    }
    for (const auto &name : {"q_norm", "k_norm"}) {
      expert_tensors.emplace(layer + ".self_attn." + name + ".weight",
                             mx::ones({4}));
    }
    for (const auto &name : {"q_proj", "k_proj", "v_proj"}) {
      expert_tensors.emplace(layer + ".self_attn." + name + ".weight",
                             mx::zeros({4, 64}));
    }
    expert_tensors.emplace(layer + ".self_attn.o_proj.weight",
                           mx::zeros({64, 4}));
    expert_tensors.emplace(layer + ".mlp.gate.weight", mx::zeros({2, 64}));
    for (const auto &name : {"gate_proj", "up_proj", "down_proj"}) {
      expert_tensors.emplace(layer + ".mlp.switch_mlp." + name + ".weight",
                             mx::zeros({2, 64, 64}));
    }
    expert_tensors.emplace("tiny.norm.weight", mx::ones({64}));
    Weights expert_weights(std::move(expert_tensors), 64, 4);
    TextModel expert_model(expert_weights,
                           {{"hidden_size", 64},
                            {"intermediate_size", 64},
                            {"head_dim", 4},
                            {"hidden_act", "silu"},
                            {"num_attention_heads", 1},
                            {"num_key_value_heads", 1},
                            {"num_hidden_layers", 1},
                            {"num_experts", 2},
                            {"num_experts_per_tok", 1},
                            {"moe_intermediate_size", 64},
                            {"rope_theta", 10000},
                            {"rms_norm_eps", 1e-6}},
                           "tiny", false);
    expert_weights.finish_validation();
    auto expert_cache = expert_model.make_cache();
    auto expert_output =
        expert_model
            .forward(mx::ones({1, 2, 64}), mx::array({0, 1}), expert_cache, {})
            .first;
    check(std::abs(mx::sum(expert_output).item<float>() - 128) < .001f &&
              expert_cache[0].offset == 2,
          "zero expert projections preserve residual and cache both prompt "
          "tokens");

    auto rotated = rotary(mx::array({1.f, 2.f, 3.f, 4.f}, {1, 1, 1, 4}),
                          mx::array({1.f}, {1}), 10000.f, false);
    mx::eval(rotated);
    check(std::abs(rotated.data<float>()[0] -
                   (std::cos(1.f) - 3 * std::sin(1.f))) < 1e-5,
          "rotary must pair split halves rather than adjacent channels");
    std::vector<float> coordinates(3);
    coordinates = {1, 2, 3};
    auto factors = rotary_factors(mx::array(coordinates.data(), {3, 1, 1}), 128,
                                  10000.f, true);
    mx::eval(factors.first);
    check(std::abs(factors.first.data<float>()[1] -
                   std::cos(2 * std::pow(10000.f, -2.f / 128))) < 1e-5,
          "M-RoPE must select height for channel one");
    std::vector<float> pixels(2 * 32 * 32 * 3);
    for (int frame = 0; frame < 2; ++frame) {
      for (int y = 0; y < 32; ++y) {
        for (int x = 0; x < 32; ++x) {
          for (int channel = 0; channel < 3; ++channel) {
            pixels[((frame * 32 + y) * 32 + x) * 3 + channel] =
                frame * 10000 + channel * 1000 + y * 32 + x;
          }
        }
      }
    }
    auto patches = patchify(pixels, 2, 32, 32);
    mx::eval(patches);
    check(patches.shape() == mx::Shape({4, 1536}), "image patch shape");
    check(patches.data<float>()[256] == 10000 &&
              patches.data<float>()[512] == 1000 &&
              patches.data<float>()[1536] == 16 &&
              patches.data<float>()[3072] == 512,
          "patches must be block-major then channel/time/row/column");
    MediaEntry positioned{
        true,
        {151652, 151656, 151675, 151670, 151653},
        {{0, 0, 0}, {2.5f, 99, 99}, {3, 3, 3}, {4, 4, 4}, {5, 5, 5}},
        {},
        {},
        {},
        151656};
    auto [media_positions, media_delta] = position_ids(
        {1, 151652, 151656, 151675, 151670, 151653, 2}, {positioned});
    mx::eval(media_positions);
    check(media_delta == 94 && media_positions.data<float>()[6] == 7,
          "decode position delta must use global spatial maximum while "
          "subsequent text uses media tail");
    for (float temporal_stride : {3.25f, 13.25f}) {
      MediaEntry video;
      video.user_owned = true;
      video.tokens = {151652};
      video.positions = {{0, 0, 0}};
      for (int frame = 0; frame < 2; ++frame) {
        for (int row = 0; row < 2; ++row) {
          for (int column = 0; column < 2; ++column) {
            video.tokens.push_back(151656);
            video.positions.push_back({frame * temporal_stride + 1,
                                       float(row + 1), float(column + 1)});
          }
        }
      }
      video.tokens.push_back(151653);
      video.positions.push_back(
          {temporal_stride + 2, temporal_stride + 2, temporal_stride + 2});
      auto tokens = video.tokens;
      tokens.push_back(7);
      auto [positions, delta] = position_ids(tokens, {video});
      mx::eval(positions);
      check(positions.data<float>()[tokens.size() - 1] == temporal_stride + 3,
            "fractional video position must survive trailing prompt text");
      for (int decoded = 0; decoded < 3; ++decoded) {
        const float position = int(tokens.size()) + decoded + delta;
        check(position == temporal_stride + 4 + decoded,
              "thinker and talker decode offsets must retain fractional "
              "positive and negative deltas");
        const auto factors =
            rotary_factors(mx::full({3, 1, 1}, position), 128, 10000.f, true);
        mx::eval(factors.first);
        check(std::abs(factors.first.data<float>()[0] -
                       std::cos(temporal_stride + 4 + decoded)) < 1e-6f,
              "decode rotary factors must use the continuous video position");
      }
    }
    for (int attempt = 0; attempt < 4; ++attempt) {
      bool rejected = false;
      try {
        image_pixels(
            "data:image/png;base64,"
            "iVBORw0KGgoAAAANSUhEUgAAAgAAAAABCAIAAAAn2YEhAAAAFElEQVR4nGOoHwWj"
            "YBSMglEwIgEACRH6H2A/SCcAAAAASUVORK5CYII=",
            {});
      } catch (const std::invalid_argument &error) {
        rejected = std::string(error.what()) == "unsupported image geometry";
      }
      check(rejected, "skinny images must remain rejected after decoding");
    }
    auto silence_mel = audio_mel(std::vector<float>(3200, 0));
    const float swift_pi = std::nextafter(float(M_PI), 0.f);
    auto legacy_whisper_window =
        .5f *
        (1.f - mx::cos(2.f * swift_pi * mx::arange(400, mx::float32) / 400.f));
    check(
        mx::array_equal(legacy_whisper_window,
                        omni_common::SwiftPeriodicHannWindow())
            .item<bool>(),
        "shared Hann extraction must preserve existing Whisper window values");
    std::vector<float> nonsilent(3200);
    for (size_t i = 0; i < nonsilent.size(); ++i) {
      nonsilent[i] = (int(i * 17 % 257) - 128) / 128.f;
    }
    auto nonsilent_mel = mx::reshape(audio_mel(nonsilent), {-1});
    mx::eval(nonsilent_mel);
    // Swift's Float.pi has bits 0x40490fda; the next float changes these bins.
    for (const auto &[index, expected] :
         std::vector<std::pair<int, float>>{{639, .07988613843917847f},
                                            {659, .08100616931915283f},
                                            {26, .5071178674697876f},
                                            {6, .4095534086227417f}}) {
      check(std::abs(nonsilent_mel.data<float>()[index] - expected) < 2e-7f,
            "nonsilent mel must match the native Swift Hann window");
    }
    check(silence_mel.shape() == mx::Shape({128, 20}) &&
              mx::max(silence_mel).item<float>() == -1.5f &&
              mx::min(silence_mel).item<float>() == -1.5f,
          "silent audio mel must drop final centered frame and preserve log "
          "floor");
    auto wave = encode_wav(std::vector<float>(24000, .25f));
    auto decoded_audio = decode_audio(wave, {});
    check(decoded_audio.size() == 16000 &&
              std::abs(decoded_audio[8000] - .25f) < .0001f,
          "native WAV decode/resampling must preserve duration and amplitude");

    nlohmann::json request = {
        {"messages", {{{"role", "user"}, {"content", "hello"}}}},
        {"max_tokens", 1}};
    auto parsed = parse_request(request, "qwen3-omni-mlx", {});
    check(parsed.max_tokens == 1, "explicit output bound must survive parsing");
    for (const auto &invalid :
         std::vector<nlohmann::json>{{{"max_tokens", true}},
                                     {{"temperature", -1}},
                                     {{"unknown", 1}},
                                     {{"modalities", {"audio"}}},
                                     {{"model", "wrong"}}}) {
      auto changed = request;
      changed.update(invalid);
      bool rejected = false;
      try {
        parse_request(changed, "qwen3-omni-mlx", {});
      } catch (const std::invalid_argument &) {
        rejected = true;
      }
      check(rejected, "invalid request must be rejected before model work");
    }
    bool malformed_path_rejected = false;
    try {
      validate_media("bad%xx.wav", {std::filesystem::current_path()});
    } catch (const std::invalid_argument &) {
      malformed_path_rejected = true;
    }
    check(malformed_path_rejected,
          "malformed percent-encoded media paths must be request errors");
    int consumed = 0, advanced = 0;
    auto generation = generate_text_tokens(
        mx::array({0.f, 10.f}, {1, 2}), 4, {1}, Sampling{0, 1, -1, 1}, {},
        [&](const mx::array &token) {
          ++advanced;
          return std::vector<mx::array>{
              mx::astype(mx::reshape(token, {1, 1}), mx::float32)};
        },
        [&](const auto &) { ++consumed; });
    check(generation.stopped && generation.tokens.empty() &&
              generation.completion_tokens == 1 && consumed == 0 &&
              advanced == 1,
          "EOS must discard bounded lookahead without emitting EOS");
    generation = generate_text_tokens(
        mx::array({10.f, 0.f}, {1, 2}), 1, {1}, Sampling{0, 1, -1, 1}, {},
        [&](const mx::array &) -> std::vector<mx::array> {
          throw std::runtime_error("must not advance at output limit");
        },
        [&](const auto &) { ++consumed; });
    check(!generation.stopped && generation.tokens == std::vector<int>{0} &&
              consumed == 1,
          "length bound must stop before another forward pass");
    bool cancel_after_advance = false;
    bool cancellation_observed = false;
    try {
      generate_text_tokens(
          mx::array({10.f, 0.f}, {1, 2}), 4, {1}, Sampling{0, 1, -1, 1},
          [&] { return cancel_after_advance; },
          [&](const mx::array &token) {
            cancel_after_advance = true;
            return std::vector<mx::array>{
                mx::astype(mx::reshape(token, {1, 1}), mx::float32)};
          },
          [&](const auto &) {
            throw std::runtime_error("must not emit after cancellation");
          });
    } catch (const Cancelled &) {
      cancellation_observed = true;
    }
    check(cancellation_observed,
          "cancellation after async lookahead must suppress pending emission");
    std::cout << "qwen3_omni component properties passed\n";
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
