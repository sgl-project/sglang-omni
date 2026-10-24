// SPDX-License-Identifier: Apache-2.0
#include "qwen3_tts/qwen3_tts_model.h"

namespace qwen3_tts {
namespace mx = mlx::core;

Model::Prompt
Model::BuildPrompt(const Request &request,
                   const std::function<void()> &check_cancelled) const {
  const auto special =
      Text({config_.at("tts_bos_token_id"), config_.at("tts_eos_token_id"),
            config_.at("tts_pad_token_id")});
  const auto bos = TimeSlice(special, 0, 1);
  const auto eos = TimeSlice(special, 1, 2);
  const auto pad = TimeSlice(special, 2, 3);
  auto control = CodecEmbedding(request.control_ids);
  std::optional<mx::array> reference_codes;
  if (!request.reference_samples.empty()) {
    check_cancelled();
    const auto speaker = mx::astype(SpeakerEmbedding(request.reference_samples),
                                    control.dtype());
    check_cancelled();
    control = mx::concatenate({control, speaker}, 1);
    reference_codes = codec_.Encode(request.reference_samples, check_cancelled);
  } else {
  }
  const int codec_pad = talker_config_.at("codec_pad_id");
  const int codec_bos = talker_config_.at("codec_bos_id");
  control =
      mx::concatenate({control, CodecEmbedding({codec_pad, codec_bos})}, 1);
  const auto role = Text(
      std::vector<int>(request.text_ids.begin(), request.text_ids.begin() + 3));
  const auto prefix = mx::concatenate(
      {role, mx::concatenate({mx::broadcast_to(
                                  pad, {1, control.shape(1) - 2, pad.shape(2)}),
                              bos},
                             1) +
                 TimeSlice(control, 0, control.shape(1) - 1)},
      1);
  const std::vector<int> body(request.text_ids.begin() + 3,
                              request.text_ids.end() - 5);
  mx::array prefill = prefix;
  mx::array trailing = pad;
  if (reference_codes.has_value()) {
    std::vector<int> full_text = request.reference_text_ids;
    full_text.insert(full_text.end(), body.begin(), body.end());
    const int frames = reference_codes->shape(0);
    auto reference = checkpoint_.Embed(
        mx::reshape(mx::slice(*reference_codes, {0, 0}, {frames, 1}),
                    {1, frames}),
        "talker.model.codec_embedding");
    for (int group = 1; group < kCodeGroups; ++group) {
      reference =
          reference +
          checkpoint_.Embed(mx::reshape(mx::slice(*reference_codes, {0, group},
                                                  {frames, group + 1}),
                                        {1, frames}),
                            "talker.code_predictor.model.codec_embedding." +
                                std::to_string(group - 1));
    }
    prefill = mx::concatenate(
        {prefix,
         mx::concatenate({Text(full_text), eos}, 1) +
             CodecEmbedding({codec_pad}),
         mx::concatenate({CodecEmbedding({codec_bos}), reference}, 1) + pad},
        1);
  } else if (request.non_streaming_mode) {
    prefill = mx::concatenate(
        {prefix,
         mx::concatenate({Text(body), eos}, 1) + CodecEmbedding({codec_pad}),
         pad + CodecEmbedding({codec_bos})},
        1);
  } else {
    prefill = mx::concatenate(
        {prefix, Text({body.front()}) + CodecEmbedding({codec_bos})}, 1);
    trailing =
        body.size() > 1
            ? mx::concatenate(
                  {Text(std::vector<int>(body.begin() + 1, body.end())), eos},
                  1)
            : eos;
  }
  if (!request.instruction_ids.empty()) {
    prefill = mx::concatenate({Text(request.instruction_ids), prefill}, 1);
  } else {
  }
  return {prefill, trailing, pad, reference_codes};
}

} // namespace qwen3_tts
