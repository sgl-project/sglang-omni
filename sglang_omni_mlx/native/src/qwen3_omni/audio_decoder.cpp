// SPDX-License-Identifier: Apache-2.0
#include "qwen3_omni/audio_decoder.h"
#include "qwen3_omni/media.h"

#include <chrono>
#include <cmath>
#include <cstring>
extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/samplefmt.h>
}
static_assert(LIBAVCODEC_VERSION_MAJOR == 61 &&
                  LIBAVFORMAT_VERSION_MAJOR == 61 &&
                  LIBAVUTIL_VERSION_MAJOR == 59,
              "Qwen3-Omni requires FFmpeg 7 APIs");

namespace qwen3_omni {
namespace {
struct AudioInput {
  const std::vector<uint8_t> &bytes;
  const Cancellation &cancelled;
  size_t offset = 0;
  std::chrono::steady_clock::time_point deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(30);
  bool interrupted() const {
    return (cancelled && cancelled()) ||
           std::chrono::steady_clock::now() >= deadline;
  }
};
struct AudioResources {
  AVIOContext *io = nullptr;
  AVFormatContext *format = nullptr;
  AVCodecContext *codec = nullptr;
  AVPacket *packet = nullptr;
  AVFrame *frame = nullptr;
  AVDictionary *options = nullptr;
  ~AudioResources() {
    av_packet_free(&packet);
    av_frame_free(&frame);
    avcodec_free_context(&codec);
    avformat_close_input(&format);
    if (io) {
      av_freep(&io->buffer);
      avio_context_free(&io);
    } else {
    }
    av_dict_free(&options);
  }
};
} // namespace
DecodedAudio decode_audio_bytes(const std::vector<uint8_t> &bytes,
                                const Cancellation &cancelled) {
  AudioInput input{bytes, cancelled};
  AudioResources resources;
  auto checked = [&](int status) {
    check_cancelled(cancelled);
    if (input.interrupted()) {
      throw std::invalid_argument("audio decoding exceeded 30 seconds");
    } else if (status < 0) {
      char message[256];
      av_strerror(status, message, sizeof(message));
      throw std::invalid_argument(
          std::string("native audio decoding failed: ") + message);
    } else {
    }
  };
  auto storage = static_cast<uint8_t *>(av_malloc(32768));
  if (!storage) {
    throw std::bad_alloc();
  } else {
  }
  resources.io = avio_alloc_context(
      storage, 32768, 0, &input,
      [](void *opaque, uint8_t *destination, int count) {
        auto &input = *static_cast<AudioInput *>(opaque);
        if (input.interrupted()) {
          return AVERROR_EXIT;
        } else if (count <= 0) {
          return AVERROR(EINVAL);
        } else {
          int available =
              int(std::min(size_t(count), input.bytes.size() - input.offset));
          if (available == 0) {
            return AVERROR_EOF;
          } else {
            std::memcpy(destination, input.bytes.data() + input.offset,
                        available);
            input.offset += available;
            return available;
          }
        }
      },
      nullptr,
      [](void *opaque, int64_t displacement, int whence) -> int64_t {
        auto &input = *static_cast<AudioInput *>(opaque);
        if (input.interrupted()) {
          return AVERROR_EXIT;
        } else if (whence == AVSEEK_SIZE) {
          return input.bytes.size();
        } else {
          int origin = whence & ~AVSEEK_FORCE;
          int64_t base = origin == SEEK_SET   ? 0
                         : origin == SEEK_CUR ? input.offset
                                              : input.bytes.size();
          if ((origin != SEEK_SET && origin != SEEK_CUR &&
               origin != SEEK_END) ||
              displacement < -base ||
              displacement > int64_t(input.bytes.size()) - base) {
            return AVERROR(EINVAL);
          } else {
            input.offset = base + displacement;
            return input.offset;
          }
        }
      });
  if (!resources.io) {
    av_free(storage);
    throw std::bad_alloc();
  } else {
  }
  resources.format = avformat_alloc_context();
  if (!resources.format) {
    throw std::bad_alloc();
  } else {
  }
  resources.format->pb = resources.io;
  resources.format->flags |= AVFMT_FLAG_CUSTOM_IO;
  resources.format->max_streams = 64;
  resources.format->probesize = 1048576;
  resources.format->io_open = [](AVFormatContext *, AVIOContext **,
                                 const char *, int,
                                 AVDictionary **) { return AVERROR(EACCES); };
  resources.format->interrupt_callback = {
      [](void *opaque) {
        return int(static_cast<AudioInput *>(opaque)->interrupted());
      },
      &input};
  checked(av_dict_set(&resources.options, "format_whitelist",
                      "wav,flac,ogg,mp3,aac,mov,matroska,aiff,caf,au", 0));
  checked(avformat_open_input(&resources.format, nullptr, nullptr,
                              &resources.options));
  int stream_index = -1;
  for (unsigned i = 0; i < resources.format->nb_streams; ++i) {
    if (resources.format->streams[i]->codecpar->codec_type ==
        AVMEDIA_TYPE_AUDIO) {
      stream_index = int(i);
      break;
    } else {
    }
  }
  if (stream_index < 0) {
    throw std::invalid_argument("media has no supported audio track");
  } else {
  }
  auto stream = resources.format->streams[stream_index];
  auto parameters = stream->codecpar;
  auto decoder = avcodec_find_decoder(parameters->codec_id);
  if (!decoder || parameters->sample_rate > 192000 ||
      parameters->ch_layout.nb_channels > 32) {
    throw std::invalid_argument("unsupported audio codec, rate or channels");
  } else {
  }
  resources.codec = avcodec_alloc_context3(decoder);
  if (!resources.codec) {
    throw std::bad_alloc();
  } else {
  }
  checked(avcodec_parameters_to_context(resources.codec, parameters));
  resources.codec->thread_count = 1;
  resources.codec->max_samples = 65536;
  resources.codec->pkt_timebase = stream->time_base;
  resources.codec->err_recognition = AV_EF_CAREFUL | AV_EF_EXPLODE;
  checked(avcodec_open2(resources.codec, decoder, nullptr));
  resources.packet = av_packet_alloc();
  resources.frame = av_frame_alloc();
  if (!resources.packet || !resources.frame) {
    throw std::bad_alloc();
  } else {
  }
  DecodedAudio decoded{{}, 0};
  int channels = 0;
  int frames_received = 0;
  auto receive = [&]() {
    while (true) {
      int status = avcodec_receive_frame(resources.codec, resources.frame);
      if (status == AVERROR_EOF || status == AVERROR(EAGAIN)) {
        return status;
      } else {
        checked(status);
      }
      auto frame = resources.frame;
      ++frames_received;
      if (frame->sample_rate < 1 || frame->sample_rate > 192000 ||
          frame->ch_layout.nb_channels < 1 ||
          frame->ch_layout.nb_channels > 32 || frame->nb_samples < 1 ||
          frame->nb_samples > 65536 || frame->flags & AV_FRAME_FLAG_CORRUPT) {
        throw std::invalid_argument("invalid audio frame");
      } else if (decoded.sample_rate == 0) {
        decoded.sample_rate = frame->sample_rate;
        channels = frame->ch_layout.nb_channels;
      } else if (decoded.sample_rate != frame->sample_rate ||
                 channels != frame->ch_layout.nb_channels) {
        throw std::invalid_argument("changing audio sample rate or channels");
      } else {
      }
      int first = 0, end = frame->nb_samples;
      if (stream->duration > 0 && stream->time_base.num > 0 &&
          stream->time_base.den > 0) {
        AVRational units{1, decoded.sample_rate};
        int64_t duration =
            av_rescale_q(stream->duration, stream->time_base, units);
        if (duration <= 0 || duration > int64_t(decoded.sample_rate) * 120) {
          throw std::invalid_argument("audio exceeds 120 seconds");
        } else {
        }
        int64_t position = decoded.samples.size();
        if (frame->pts != AV_NOPTS_VALUE &&
            stream->start_time != AV_NOPTS_VALUE) {
          int64_t relative;
          if (__builtin_sub_overflow(frame->pts, stream->start_time,
                                     &relative)) {
            throw std::invalid_argument("invalid audio timestamp");
          } else {
            position = av_rescale_q(relative, stream->time_base, units);
          }
        } else {
        }
        if (position <= -int64_t(decoded.sample_rate) * 120 ||
            position >= int64_t(decoded.sample_rate) * 240) {
          throw std::invalid_argument(
              "audio timestamp outside supported duration");
        } else {
        }
        first = int(std::min<int64_t>(end, std::max<int64_t>(0, -position)));
        end = int(
            std::min<int64_t>(end, std::max<int64_t>(0, duration - position)));
      } else {
      }
      if (decoded.samples.size() + std::max(0, end - first) >
          size_t(decoded.sample_rate) * 120) {
        throw std::invalid_argument("audio exceeds 120 seconds");
      } else {
      }
      auto format = AVSampleFormat(frame->format);
      bool planar = av_sample_fmt_is_planar(format);
      auto packed = av_get_packed_sample_fmt(format);
      for (int sample = first; sample < end; ++sample) {
        float sum = 0;
        for (int channel = 0; channel < channels; ++channel) {
          auto plane = frame->extended_data[planar ? channel : 0];
          int index = planar ? sample : sample * channels + channel;
          float value;
          switch (packed) {
          case AV_SAMPLE_FMT_U8:
            value = (float(plane[index]) - 128) / 128;
            break;
          case AV_SAMPLE_FMT_S16:
            value = float(reinterpret_cast<int16_t *>(plane)[index]) / 32768;
            break;
          case AV_SAMPLE_FMT_S32:
            value =
                float(reinterpret_cast<int32_t *>(plane)[index]) / 2147483648.f;
            break;
          case AV_SAMPLE_FMT_S64:
            value = float(double(reinterpret_cast<int64_t *>(plane)[index]) /
                          9223372036854775808.0);
            break;
          case AV_SAMPLE_FMT_FLT:
            value = reinterpret_cast<float *>(plane)[index];
            break;
          case AV_SAMPLE_FMT_DBL:
            value = float(reinterpret_cast<double *>(plane)[index]);
            break;
          default:
            throw std::invalid_argument("unsupported audio sample format");
          }
          if (!std::isfinite(value)) {
            throw std::invalid_argument("non-finite decoded audio");
          } else {
            sum += value;
          }
        }
        if (!std::isfinite(sum / channels)) {
          throw std::invalid_argument("non-finite audio downmix");
        } else {
          decoded.samples.push_back(sum / channels);
        }
      }
      av_frame_unref(frame);
    }
  };
  for (int packet_count = 0;; ++packet_count) {
    if (packet_count >= 1000000) {
      throw std::invalid_argument("too many audio packets");
    } else {
    }
    int status = av_read_frame(resources.format, resources.packet);
    if (status == AVERROR_EOF) {
      break;
    } else {
      checked(status);
    }
    if (resources.packet->stream_index == stream_index) {
      if (resources.packet->flags & AV_PKT_FLAG_CORRUPT) {
        throw std::invalid_argument("corrupt audio packet");
      } else {
      }
      int sent = avcodec_send_packet(resources.codec, resources.packet);
      if (sent == AVERROR(EAGAIN)) {
        int previous = frames_received;
        receive();
        if (frames_received == previous) {
          throw std::invalid_argument("audio decoder made no progress");
        } else {
          sent = avcodec_send_packet(resources.codec, resources.packet);
        }
      } else {
      }
      checked(sent);
      receive();
    } else {
    }
    av_packet_unref(resources.packet);
  }
  checked(avcodec_send_packet(resources.codec, nullptr));
  int final_status = receive();
  if (final_status != AVERROR_EOF || decoded.samples.empty()) {
    throw std::invalid_argument("incomplete or empty decoded audio");
  } else {
    return decoded;
  }
}
} // namespace qwen3_omni
