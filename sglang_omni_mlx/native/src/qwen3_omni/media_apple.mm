// SPDX-License-Identifier: Apache-2.0
#include "qwen3_omni/media.h"

#import <AVFoundation/AVFoundation.h>
#import <CoreGraphics/CoreGraphics.h>
#import <ImageIO/ImageIO.h>
#include <cerrno>
#include <curl/curl.h>
#include <fcntl.h>
#include <fstream>
#include <memory>
#include <numeric>
#include <unistd.h>

namespace qwen3_omni {
namespace {
using OwnedImage = std::unique_ptr<CGImage, decltype(&CGImageRelease)>;

NSURL *local_url(const std::string &source) {
  NSString *text = [NSString stringWithUTF8String:source.c_str()];
  NSString *decoded = [text stringByRemovingPercentEncoding];
  if (!decoded || source.find('\0') != std::string::npos) {
    throw std::invalid_argument("invalid local media path encoding");
  } else {
  }
  NSURL *url =
      source.starts_with("file:")
          ? [NSURL URLWithString:text]
          : [NSURL fileURLWithPath:[decoded stringByExpandingTildeInPath]];
  if (!url || !url.path.length) {
    throw std::invalid_argument("invalid local media URL");
  } else {
  }
  return [[url URLByResolvingSymlinksInPath] URLByStandardizingPath];
}
std::pair<int, int> resize_dimensions(int height, int width, int factor,
                                      int minimum_pixels, int maximum_pixels) {
  if (height <= 0 || width <= 0 ||
      double(std::max(height, width)) / std::min(height, width) > 200) {
    throw std::invalid_argument("unsupported image geometry");
  } else {
  }
  int resized_height =
      std::max(1, int(std::nearbyint(double(height) / factor))) * factor;
  int resized_width =
      std::max(1, int(std::nearbyint(double(width) / factor))) * factor;
  if (int64_t(resized_height) * resized_width > maximum_pixels) {
    double ratio = std::sqrt(double(height) * width / maximum_pixels);
    resized_height =
        std::max(factor, int(std::floor(height / ratio / factor)) * factor);
    resized_width =
        std::max(factor, int(std::floor(width / ratio / factor)) * factor);
  } else if (resized_height * resized_width < minimum_pixels) {
    double ratio = std::sqrt(double(minimum_pixels) / (double(height) * width));
    resized_height = int(std::ceil(height * ratio / factor)) * factor;
    resized_width = int(std::ceil(width * ratio / factor)) * factor;
  } else {
  }
  return {resized_height, resized_width};
}
std::vector<float> pixels(CGImageRef image, int height, int width) {
  std::vector<uint8_t> bytes(size_t(height) * width * 4);
  CGColorSpaceRef color_space = CGColorSpaceCreateDeviceRGB();
  CGContextRef context =
      CGBitmapContextCreate(bytes.data(), width, height, 8, width * 4,
                            color_space, kCGImageAlphaNoneSkipLast);
  CGColorSpaceRelease(color_space);
  if (!context) {
    throw std::invalid_argument("cannot create image conversion context");
  } else {
  }
  CGContextSetInterpolationQuality(context, kCGInterpolationHigh);
  CGContextDrawImage(context, CGRectMake(0, 0, width, height), image);
  CGContextRelease(context);
  std::vector<float> normalized;
  normalized.reserve(size_t(height) * width * 3);
  for (size_t pixel = 0; pixel < bytes.size(); pixel += 4) {
    for (int channel = 0; channel < 3; ++channel) {
      normalized.push_back(float(bytes[pixel + channel]) / 127.5f - 1);
    }
  }
  return normalized;
}
} // namespace
std::vector<uint8_t> decode_base64(const std::string &encoded) {
  @autoreleasepool {
    NSData *decoded = [[NSData alloc]
        initWithBase64EncodedString:[NSString
                                        stringWithUTF8String:encoded.c_str()]
                            options:0];
    if (!decoded || decoded.length == 0 ||
        decoded.length > MAXIMUM_MEDIA_BYTES) {
      throw std::invalid_argument("invalid or oversized base64 media");
    } else {
      auto bytes = static_cast<const uint8_t *>(decoded.bytes);
      return {bytes, bytes + decoded.length};
    }
  }
}
std::string encode_base64(const std::vector<uint8_t> &bytes) {
  @autoreleasepool {
    NSData *content = [NSData dataWithBytes:bytes.data() length:bytes.size()];
    return [[content base64EncodedStringWithOptions:0] UTF8String];
  }
}
void validate_media(const std::string &source,
                    const std::vector<std::filesystem::path> &allowed_paths) {
  @autoreleasepool {
    if (source.empty()) {
      throw std::invalid_argument("empty media URL");
    } else if (source.starts_with("data:")) {
      auto comma = source.find(',');
      if (source.size() >= 48000000 || comma == std::string::npos ||
          !source.substr(0, comma).ends_with(";base64")) {
        throw std::invalid_argument("invalid media data URL");
      } else {
        decode_base64(source.substr(comma + 1));
      }
    } else if (source.starts_with("http://") ||
               source.starts_with("https://")) {
      return;
    } else {
      NSURL *url =
          [NSURL URLWithString:[NSString stringWithUTF8String:source.c_str()]];
      if ((url.scheme && ![url.scheme isEqualToString:@"file"]) ||
          url.host.length > 0) {
        throw std::invalid_argument("unsupported media URL scheme");
      } else {
      }
      auto path = std::filesystem::path(local_url(source).path.UTF8String);
      bool allowed = false;
      for (const auto &root : allowed_paths) {
        auto canonical = std::filesystem::weakly_canonical(root).string() + "/";
        allowed |= path.string().starts_with(canonical);
      }
      std::error_code error;
      if (!allowed || !std::filesystem::is_regular_file(path, error) ||
          std::filesystem::file_size(path, error) > MAXIMUM_MEDIA_BYTES ||
          error) {
        throw std::invalid_argument(
            "local media outside allowed roots, unavailable, or oversized");
      } else {
      }
    }
  }
}
std::vector<uint8_t> media_bytes(const std::string &source,
                                 const Cancellation &cancelled) {
  @autoreleasepool {
    check_cancelled(cancelled);
    std::vector<uint8_t> content;
    if (source.starts_with("data:")) {
      content = decode_base64(source.substr(source.find(',') + 1));
    } else if (source.starts_with("http://") ||
               source.starts_with("https://")) {
      struct Transfer {
        std::vector<uint8_t> &content;
        const Cancellation &cancelled;
      } transfer{content, cancelled};
      auto curl = std::unique_ptr<CURL, decltype(&curl_easy_cleanup)>(
          curl_easy_init(), curl_easy_cleanup);
      if (!curl) {
        throw std::runtime_error("cannot initialize media HTTP client");
      } else {
      }
      curl_easy_setopt(curl.get(), CURLOPT_URL, source.c_str());
      curl_easy_setopt(curl.get(), CURLOPT_TIMEOUT, 30L);
      curl_easy_setopt(curl.get(), CURLOPT_FOLLOWLOCATION, 1L);
      curl_easy_setopt(curl.get(), CURLOPT_MAXREDIRS, 5L);
      curl_easy_setopt(curl.get(), CURLOPT_PROTOCOLS_STR, "http,https");
      curl_easy_setopt(curl.get(), CURLOPT_REDIR_PROTOCOLS_STR, "http,https");
      curl_easy_setopt(
          curl.get(), CURLOPT_WRITEFUNCTION,
          +[](char *bytes, size_t size, size_t count, void *opaque) -> size_t {
            auto &transfer = *static_cast<Transfer *>(opaque);
            if (transfer.content.size() + size * count > MAXIMUM_MEDIA_BYTES ||
                (transfer.cancelled && transfer.cancelled())) {
              return 0;
            } else {
              transfer.content.insert(transfer.content.end(), bytes,
                                      bytes + size * count);
              return size * count;
            }
          });
      curl_easy_setopt(curl.get(), CURLOPT_WRITEDATA, &transfer);
      curl_easy_setopt(curl.get(), CURLOPT_NOPROGRESS, 0L);
      curl_easy_setopt(
          curl.get(), CURLOPT_XFERINFOFUNCTION,
          +[](void *opaque, curl_off_t, curl_off_t, curl_off_t, curl_off_t) {
            auto &transfer = *static_cast<Transfer *>(opaque);
            return int(transfer.cancelled && transfer.cancelled());
          });
      curl_easy_setopt(curl.get(), CURLOPT_XFERINFODATA, &transfer);
      auto status = curl_easy_perform(curl.get());
      check_cancelled(cancelled);
      long response = 0;
      curl_easy_getinfo(curl.get(), CURLINFO_RESPONSE_CODE, &response);
      if (status != CURLE_OK || response < 200 || response >= 300) {
        throw std::invalid_argument("media download failed or exceeded bounds");
      } else {
      }
    } else {
      std::ifstream stream(local_url(source).path.UTF8String, std::ios::binary);
      content.resize(MAXIMUM_MEDIA_BYTES + 1);
      stream.read(reinterpret_cast<char *>(content.data()), content.size());
      content.resize(stream.gcount());
    }
    if (content.empty() || content.size() > MAXIMUM_MEDIA_BYTES) {
      throw std::invalid_argument("empty media or media exceeds 32 MiB");
    } else {
      return content;
    }
  }
}
VisualInput image_pixels(const std::string &source,
                         const Cancellation &cancelled) {
  @autoreleasepool {
    auto bytes = media_bytes(source, cancelled);
    NSData *content = [NSData dataWithBytes:bytes.data() length:bytes.size()];
    CGImageSourceRef decoder =
        CGImageSourceCreateWithData((__bridge CFDataRef)content, nullptr);
    if (!decoder) {
      throw std::invalid_argument("invalid image");
    } else {
    }
    NSDictionary *metadata = CFBridgingRelease(
        CGImageSourceCopyPropertiesAtIndex(decoder, 0, nullptr));
    int64_t width = [metadata[(__bridge NSString *)kCGImagePropertyPixelWidth]
        longLongValue];
    int64_t height = [metadata[(__bridge NSString *)kCGImagePropertyPixelHeight]
        longLongValue];
    if (width <= 0 || height <= 0 || width * height > 16777216) {
      CFRelease(decoder);
      throw std::invalid_argument(
          "invalid image dimensions or image exceeds 16 megapixels");
    } else {
    }
    NSDictionary *options = @{
      (__bridge NSString *)kCGImageSourceCreateThumbnailFromImageAlways : @YES,
      (__bridge NSString *)kCGImageSourceThumbnailMaxPixelSize : @2048,
      (__bridge NSString *)kCGImageSourceCreateThumbnailWithTransform : @YES
    };
    OwnedImage image(CGImageSourceCreateThumbnailAtIndex(
                         decoder, 0, (__bridge CFDictionaryRef)options),
                     CGImageRelease);
    CFRelease(decoder);
    if (!image) {
      throw std::invalid_argument("cannot decode image");
    } else {
    }
    auto [resized_height, resized_width] =
        resize_dimensions(CGImageGetHeight(image.get()),
                          CGImageGetWidth(image.get()), 32, 3136, 12845056);
    auto frame = pixels(image.get(), resized_height, resized_width);
    image.reset();
    auto duplicated = frame;
    duplicated.insert(duplicated.end(), frame.begin(), frame.end());
    return {patchify(duplicated, 2, resized_height, resized_width),
            {1, resized_height / 16, resized_width / 16},
            1};
  }
}
VisualInput video_pixels(const std::string &source, const ChatRequest &request,
                         const Cancellation &cancelled) {
  @autoreleasepool {
    struct OwnedFile {
      std::filesystem::path path;
      int descriptor = -1;
      ~OwnedFile() {
        if (descriptor >= 0) {
          close(descriptor);
        } else {
        }
        if (!path.empty()) {
          std::error_code error;
          std::filesystem::remove(path, error);
        } else {
        }
      }
    } owned;
    NSURL *url;
    if (source.starts_with("data:") || source.starts_with("http://") ||
        source.starts_with("https://")) {
      auto content = media_bytes(source, cancelled);
      auto path = std::filesystem::temp_directory_path() /
                  (std::string([NSUUID UUID].UUIDString.UTF8String) + ".mp4");
      owned.descriptor =
          open(path.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0600);
      if (owned.descriptor < 0) {
        throw std::invalid_argument("cannot materialize video input");
      } else {
        owned.path = path;
      }
      size_t offset = 0;
      while (offset < content.size()) {
        check_cancelled(cancelled);
        const auto written = write(owned.descriptor, content.data() + offset,
                                   content.size() - offset);
        if (written < 0 && errno == EINTR) {
          continue;
        } else if (written <= 0) {
          throw std::invalid_argument("cannot write materialized video input");
        } else {
          offset += size_t(written);
        }
      }
      const int status = close(owned.descriptor);
      owned.descriptor = -1;
      if (status != 0) {
        throw std::invalid_argument("cannot close materialized video input");
      } else {
      }
      url = [NSURL
          fileURLWithPath:[NSString stringWithUTF8String:owned.path.c_str()]];
    } else {
      url = local_url(source);
    }
    AVURLAsset *asset = [AVURLAsset URLAssetWithURL:url options:nil];
    AVAssetTrack *track =
        [[asset tracksWithMediaType:AVMediaTypeVideo] firstObject];
    double frame_rate = track.nominalFrameRate;
    double duration = CMTimeGetSeconds(asset.duration);
    if (!track || !std::isfinite(frame_rate) || frame_rate <= 0 ||
        !std::isfinite(duration) || duration <= 0 || duration > 3600) {
      throw std::invalid_argument("invalid video track or duration");
    } else {
    }
    NSError *error = nil;
    AVAssetReader *reader = [[AVAssetReader alloc] initWithAsset:asset
                                                           error:&error];
    AVAssetReaderTrackOutput *output = [[AVAssetReaderTrackOutput alloc]
         initWithTrack:track
        outputSettings:@{
          (NSString *)
          kCVPixelBufferPixelFormatTypeKey : @(kCVPixelFormatType_32BGRA)
        }];
    [reader addOutput:output];
    if (![reader startReading]) {
      throw std::invalid_argument("cannot read video frames");
    } else {
    }
    struct ReaderGuard {
      AVAssetReader *__strong reader;
      ~ReaderGuard() { [reader cancelReading]; }
    } reader_guard{reader};
    std::vector<CMTime> timestamps;
    double last_frame_seconds = 1 / frame_rate;
    while (CMSampleBufferRef sample = [output copyNextSampleBuffer]) {
      CMTime timestamp = CMSampleBufferGetPresentationTimeStamp(sample);
      double sample_seconds =
          CMTimeGetSeconds(CMSampleBufferGetDuration(sample));
      CFRelease(sample);
      check_cancelled(cancelled);
      if (!std::isfinite(CMTimeGetSeconds(timestamp)) ||
          timestamps.size() >= 216000 ||
          (!timestamps.empty() &&
           CMTimeCompare(timestamp, timestamps.back()) <= 0)) {
        throw std::invalid_argument("invalid or excessive video timestamps");
      } else {
        timestamps.push_back(timestamp);
      }
      if (std::isfinite(sample_seconds) && sample_seconds > 0) {
        last_frame_seconds = sample_seconds;
      } else {
      }
    }
    if (reader.status != AVAssetReaderStatusCompleted || timestamps.empty()) {
      throw std::invalid_argument("video decoding failed");
    } else {
    }
    double sampled_duration = CMTimeGetSeconds(timestamps.back()) -
                              CMTimeGetSeconds(timestamps.front()) +
                              last_frame_seconds;
    int total_frames = timestamps.size();
    double target = std::min(
        {std::max(double(total_frames) / frame_rate * request.video_fps, 4.0),
         double(request.video_max_frames / 2 * 2), double(total_frames)});
    int frame_count =
        total_frames == 1 ? 2 : std::max(2, int(std::floor(target / 2)) * 2);
    int maximum_pixels = std::min(request.video_max_pixels,
                                  request.video_total_pixels / frame_count);
    if (maximum_pixels < request.video_min_pixels) {
      throw std::invalid_argument("insufficient video pixel budget");
    } else {
    }
    AVAssetImageGenerator *generator =
        [[AVAssetImageGenerator alloc] initWithAsset:asset];
    generator.appliesPreferredTrackTransform = YES;
    generator.requestedTimeToleranceBefore = kCMTimeZero;
    generator.requestedTimeToleranceAfter = kCMTimeZero;
    struct GeneratorGuard {
      AVAssetImageGenerator *__strong generator;
      ~GeneratorGuard() { [generator cancelAllCGImageGeneration]; }
    } generator_guard{generator};
    std::vector<float> frames;
    int output_height = 0, output_width = 0;
    for (int i = 0; i < frame_count; ++i) {
      check_cancelled(cancelled);
      int frame_index = int(
          std::nearbyint(double(i) * (total_frames - 1) / (frame_count - 1)));
      OwnedImage image([generator copyCGImageAtTime:timestamps[frame_index]
                                         actualTime:nullptr
                                              error:&error],
                       CGImageRelease);
      if (!image) {
        throw std::invalid_argument("cannot decode sampled video frame");
      } else {
      }
      auto [smart_height, smart_width] = resize_dimensions(
          CGImageGetHeight(image.get()), CGImageGetWidth(image.get()), 28,
          request.video_min_pixels, maximum_pixels);
      auto [height, width] =
          resize_dimensions(smart_height, smart_width, 32,
                            request.video_min_pixels, request.video_max_pixels);
      if (i > 0 && (height != output_height || width != output_width)) {
        throw std::invalid_argument("video geometry changed");
      } else {
        output_height = height;
        output_width = width;
      }
      CGColorSpaceRef color_space = CGColorSpaceCreateDeviceRGB();
      CGContextRef context = CGBitmapContextCreate(
          nullptr, smart_width, smart_height, 8, smart_width * 4, color_space,
          kCGImageAlphaNoneSkipLast);
      CGColorSpaceRelease(color_space);
      if (!context) {
        throw std::invalid_argument("cannot resize video frame");
      } else {
      }
      CGContextSetInterpolationQuality(context, kCGInterpolationHigh);
      CGContextDrawImage(context, CGRectMake(0, 0, smart_width, smart_height),
                         image.get());
      image.reset();
      OwnedImage resized(CGBitmapContextCreateImage(context), CGImageRelease);
      CGContextRelease(context);
      if (!resized) {
        throw std::invalid_argument("cannot copy resized video frame");
      } else {
      }
      auto frame = pixels(resized.get(), height, width);
      frames.insert(frames.end(), frame.begin(), frame.end());
    }
    return {patchify(frames, frame_count, output_height, output_width),
            {frame_count / 2, output_height / 16, output_width / 16},
            double(frame_count) / sampled_duration};
  }
}
} // namespace qwen3_omni
