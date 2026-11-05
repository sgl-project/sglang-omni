// SPDX-License-Identifier: Apache-2.0
#include "qwen3_omni/audio_decoder.h"
#include "qwen3_omni/media.h"
#import <AVFoundation/AVFoundation.h>

namespace qwen3_omni {
std::vector<float> decode_audio(const std::vector<uint8_t> &bytes,
                                const Cancellation &cancelled) {
  @autoreleasepool {
    auto decoded = decode_audio_bytes(bytes, cancelled);
    if (decoded.sample_rate == 16000) {
      return decoded.samples;
    } else {
    }
    AVAudioFormat *input_format =
        [[AVAudioFormat alloc] initWithCommonFormat:AVAudioPCMFormatFloat32
                                         sampleRate:decoded.sample_rate
                                           channels:1
                                        interleaved:NO];
    AVAudioFormat *output_format =
        [[AVAudioFormat alloc] initWithCommonFormat:AVAudioPCMFormatFloat32
                                         sampleRate:16000
                                           channels:1
                                        interleaved:NO];
    AVAudioConverter *converter =
        [[AVAudioConverter alloc] initFromFormat:input_format
                                        toFormat:output_format];
    AVAudioPCMBuffer *input =
        [[AVAudioPCMBuffer alloc] initWithPCMFormat:input_format
                                      frameCapacity:decoded.samples.size()];
    AVAudioPCMBuffer *output = [[AVAudioPCMBuffer alloc]
        initWithPCMFormat:output_format
            frameCapacity:std::ceil(double(decoded.samples.size()) * 16000 /
                                    decoded.sample_rate) +
                          512];
    if (!input || !output || !converter) {
      throw std::invalid_argument("cannot initialize native audio resampler");
    } else {
    }
    input.frameLength = decoded.samples.size();
    std::copy(decoded.samples.begin(), decoded.samples.end(),
              input.floatChannelData[0]);
    converter.sampleRateConverterQuality = AVAudioQualityMax;
    __block bool supplied = false;
    NSError *error = nil;
    auto status = [converter
           convertToBuffer:output
                     error:&error
        withInputFromBlock:^AVAudioBuffer *(
            AVAudioPacketCount, AVAudioConverterInputStatus *input_status) {
          if (supplied) {
            *input_status = AVAudioConverterInputStatus_EndOfStream;
            return nil;
          } else {
            supplied = true;
            *input_status = AVAudioConverterInputStatus_HaveData;
            return input;
          }
        }];
    if (status == AVAudioConverterOutputStatus_Error) {
      throw std::invalid_argument("native audio resampling failed");
    } else {
      return {output.floatChannelData[0],
              output.floatChannelData[0] + output.frameLength};
    }
  }
}
std::vector<float> audio_samples(const std::string &source,
                                 const Cancellation &cancelled) {
  return decode_audio(media_bytes(source, cancelled), cancelled);
}
} // namespace qwen3_omni
