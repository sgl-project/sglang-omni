// SPDX-License-Identifier: Apache-2.0
#include "higgs_tts/higgs_tts_codec.h"

#include <algorithm>
#include <string>

namespace higgs_tts {

namespace mx = mlx::core;

namespace {

constexpr const char *kDecoderPrefix = "acoustic_decoder";
constexpr int kSampleRate = 24000;
// The decoder blocks' upsampling strides, 8 * 5 * 4 * 2 * 3 = 960 samples a
// frame.
constexpr int kBlockStrides[] = {8, 5, 4, 2, 3};
constexpr int kResidualDilations[] = {1, 3, 9};
constexpr int kConvolutionKernel = 7;
constexpr float kSnakeEpsilon = 1e-9f;
constexpr int kInitialChunkFrames = 20;
constexpr int kStreamStrideRows = 75;
constexpr int kStreamFollowupStrideRows = 75;
constexpr int kStreamOverlapFrames = 8;
constexpr int kStreamHoldbackFrames = 4;

// DAC's snake activation, x + sin^2(alpha x) / alpha.
mx::array Snake(const mx::array &x, const mx::array &alpha) {
  return mx::add(
      x, mx::multiply(mx::reciprocal(mx::add(alpha, mx::array(kSnakeEpsilon))),
                      mx::square(mx::sin(mx::multiply(alpha, x)))));
}

} // namespace

AudioCodec::AudioCodec(omni_common::WeightMap weights)
    : checkpoint_(PrepareWeights(std::move(weights)),
                  nlohmann::json::object()) {}

int AudioCodec::samples_per_frame() const {
  int samples = 1;
  for (const int stride : kBlockStrides) {
    samples *= stride;
  }
  return samples;
}

int AudioCodec::sample_rate() const { return kSampleRate; }

omni_common::WeightMap
AudioCodec::PrepareWeights(omni_common::WeightMap weights) {
  omni_common::WeightMap prepared;
  for (auto &[name, weight] : weights) {
    const bool is_decoder =
        name.rfind(std::string(kDecoderPrefix) + ".", 0) == 0;
    const bool is_quantizer_decode =
        name.rfind("quantizer.", 0) == 0 &&
        (name.find(".codebook.embed") != std::string::npos ||
         name.find(".project_out.") != std::string::npos) &&
        name.find("embed_avg") == std::string::npos;
    if (!is_decoder && !is_quantizer_decode && name.rfind("fc2.", 0) != 0) {
      continue;
    } else {
    }
    // Note (Jiaxin Deng): float32 throughout, as the CUDA vocoder decodes.
    mx::array value = mx::astype(weight, mx::float32);
    if (name.size() > 6 && name.compare(name.size() - 6, 6, ".alpha") == 0) {
      value = mx::transpose(value, {0, 2, 1});
    } else if (value.ndim() == 3 &&
               name.find(".conv_t1.") != std::string::npos) {
      value = mx::transpose(value, {1, 2, 0});
    } else if (value.ndim() == 3) {
      value = mx::transpose(value, {0, 2, 1});
    } else {
    }
    prepared.insert_or_assign(name, value);
  }
  return prepared;
}

mx::array AudioCodec::ResidualUnit(const mx::array &x,
                                   const std::string &prefix,
                                   int dilation) const {
  mx::array y = Snake(x, checkpoint_.Weight(prefix + ".snake1.alpha"));
  y = mx::add(mx::conv1d(y, checkpoint_.Weight(prefix + ".conv1.weight"), 1,
                         (kConvolutionKernel - 1) * dilation / 2, dilation),
              checkpoint_.Weight(prefix + ".conv1.bias"));
  y = Snake(y, checkpoint_.Weight(prefix + ".snake2.alpha"));
  y = mx::add(mx::conv1d(y, checkpoint_.Weight(prefix + ".conv2.weight")),
              checkpoint_.Weight(prefix + ".conv2.bias"));
  return mx::add(x, y);
}

mx::array AudioCodec::Decode(const mx::array &frames) const {
  const int frame_count = frames.shape(0);
  mx::array latents = mx::zeros({1, frame_count, 1}, mx::float32);
  for (int codebook = 0; codebook < frames.shape(1); ++codebook) {
    const std::string prefix =
        "quantizer.quantizers." + std::to_string(codebook);
    const mx::array codes = mx::reshape(
        mx::slice(frames, {0, codebook}, {frame_count, codebook + 1}),
        {1, frame_count});
    latents = mx::add(
        latents,
        checkpoint_.Linear(
            mx::take(checkpoint_.Weight(prefix + ".codebook.embed"), codes, 0),
            prefix + ".project_out"));
  }
  mx::array x = checkpoint_.Linear(latents, "fc2");
  const std::string decoder = kDecoderPrefix;
  x = mx::add(mx::conv1d(x, checkpoint_.Weight(decoder + ".conv1.weight"), 1,
                         kConvolutionKernel / 2),
              checkpoint_.Weight(decoder + ".conv1.bias"));
  for (int block = 0; block < static_cast<int>(std::size(kBlockStrides));
       ++block) {
    const std::string prefix = decoder + ".block." + std::to_string(block);
    const int stride = kBlockStrides[block];
    x = Snake(x, checkpoint_.Weight(prefix + ".snake1.alpha"));
    // Note (Jiaxin Deng): the codec's transposed convolutions pad ceil(s / 2)
    // with an output padding of s % 2, so each yields exactly s samples a
    // step.
    x = mx::add(
        mx::conv_transpose1d(x, checkpoint_.Weight(prefix + ".conv_t1.weight"),
                             stride, (stride + 1) / 2, 1, stride % 2),
        checkpoint_.Weight(prefix + ".conv_t1.bias"));
    for (int unit = 0; unit < static_cast<int>(std::size(kResidualDilations));
         ++unit) {
      x = ResidualUnit(x, prefix + ".res_unit" + std::to_string(unit + 1),
                       kResidualDilations[unit]);
    }
  }
  x = Snake(x, checkpoint_.Weight(decoder + ".snake1.alpha"));
  x = mx::add(mx::conv1d(x, checkpoint_.Weight(decoder + ".conv2.weight"), 1,
                         kConvolutionKernel / 2),
              checkpoint_.Weight(decoder + ".conv2.bias"));
  return mx::reshape(x, {-1});
}

mx::array RawFrames(const std::vector<int32_t> &delayed_codes, int first_row,
                    int row_end) {
  const int frame_count = row_end - first_row - kCodebookCount + 1;
  std::vector<int32_t> frames(static_cast<size_t>(frame_count) *
                              kCodebookCount);
  for (int frame = 0; frame < frame_count; ++frame) {
    for (int codebook = 0; codebook < kCodebookCount; ++codebook) {
      const int32_t code =
          delayed_codes[static_cast<size_t>(first_row + frame + codebook) *
                            kCodebookCount +
                        codebook];
      frames[static_cast<size_t>(frame) * kCodebookCount + codebook] =
          code >= kCodecEntryCount ? 0 : code;
    }
  }
  return mx::array(frames.begin(), {frame_count, kCodebookCount}, mx::int32);
}

int CodecStream::row_count() const {
  return static_cast<int>(delayed_codes_.size()) / kCodebookCount;
}

std::vector<float> CodecStream::Push(const mx::array &row_codes) {
  const int32_t *codes = row_codes.data<int32_t>();
  delayed_codes_.insert(delayed_codes_.end(), codes, codes + kCodebookCount);
  const int rows = row_count();
  const bool first_window = emitted_frames_ == 0;
  const int decode_rows =
      next_decode_rows_ > 0
          ? next_decode_rows_
          : (first_window ? kInitialChunkFrames + kCodebookCount - 1
                          : kStreamStrideRows);
  if (rows < decode_rows) {
    next_decode_rows_ = decode_rows;
    return {};
  } else {
  }
  const int frame_total = rows - kCodebookCount + 1;
  const int frame_end = first_window
                            ? std::min(frame_total, kInitialChunkFrames)
                            : std::max(0, frame_total - kStreamHoldbackFrames);
  if (frame_end <= emitted_frames_) {
    next_decode_rows_ = rows + kStreamFollowupStrideRows;
    return {};
  } else {
  }
  std::vector<float> samples = DecodeWindow(frame_end, false);
  next_decode_rows_ = first_window
                          ? kStreamStrideRows + kStreamFollowupStrideRows
                          : rows + kStreamFollowupStrideRows;
  return samples;
}

std::vector<float> CodecStream::Finish() {
  const int frame_total = row_count() - kCodebookCount + 1;
  if (frame_total <= emitted_frames_) {
    return {};
  } else {
    return DecodeWindow(frame_total, true);
  }
}

std::vector<float> CodecStream::DecodeWindow(int frame_end, bool is_final) {
  const int window_start = std::max(0, emitted_frames_ - kStreamOverlapFrames);
  const int samples_per_frame = codec_.samples_per_frame();
  const int first_sample = (emitted_frames_ - window_start) * samples_per_frame;
  // Note (Jiaxin Deng): a window before the last ends on its new frames,
  // since the decoder's right edge still awaits frames to come; the last
  // keeps everything it decodes.
  const mx::array decoded = codec_.Decode(
      RawFrames(delayed_codes_, window_start, frame_end + kCodebookCount - 1));
  const int sample_end =
      is_final
          ? decoded.shape(0)
          : first_sample + (frame_end - emitted_frames_) * samples_per_frame;
  const mx::array audio = mx::slice(decoded, {first_sample}, {sample_end});
  mx::eval(audio);
  emitted_frames_ = frame_end;
  return std::vector<float>(audio.data<float>(),
                            audio.data<float>() + audio.size());
}

} // namespace higgs_tts
