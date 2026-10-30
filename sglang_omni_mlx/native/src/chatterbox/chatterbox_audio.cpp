// SPDX-License-Identifier: Apache-2.0
#include "chatterbox/chatterbox_audio.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace chatterbox {

namespace mx = mlx::core;

namespace {

constexpr uint16_t kWavFormatPcm = 1;
constexpr uint16_t kWavFormatFloat = 3;
constexpr uint16_t kWavFormatExtensible = 0xFFFE;
// Mono or stereo: the decoder's mix of more channels wraps negative sums.
constexpr int kMaximumChannelCount = 2;
// Note (Jiaxin Deng): bounds that keep a hostile header from asking for a
// multi-gigabyte resampling filter or output; every common rate is inside.
constexpr uint32_t kMinimumFileSampleRate = 1000;
constexpr uint32_t kMaximumFileSampleRate = 384000;
// Note (Jiaxin Deng): 11025 Hz to 24 kHz, the largest reduced ratio of the
// common rates, is 320.
constexpr long kMaximumResampleFactor = 1024;
constexpr float kPcm16FullScale = 32768.0f;

// A Kaiser-windowed sinc with the kaiser_best parameters: 64 zero crossings.
constexpr long kFilterZeroCrossings = 64;
constexpr double kFilterRolloff = 0.9475937167399596;
constexpr double kFilterKaiserBeta = 14.769656459379492;

// BS.1770 K-weighting: a high shelf, then a high pass.
constexpr double kShelfFrequencyHz = 1681.974450955533;
constexpr double kShelfQuality = 0.7071752369554196;
constexpr double kShelfGainDecibels = 3.999843853973347;
constexpr double kShelfMidGainExponent = 0.4996667741545416;
constexpr double kHighPassFrequencyHz = 38.13547087602444;
constexpr double kHighPassQuality = 0.5003270373238773;
constexpr double kGatingBlockSeconds = 0.400;
constexpr double kGatingBlockOverlap = 0.75;
constexpr double kAbsoluteGateLufs = -70.0;
constexpr double kRelativeGateLu = 10.0;
constexpr double kLoudnessOffsetLufs = -0.691;

constexpr int kMelFftSize = 1920;
constexpr int kMelHopSamples = 480;
constexpr int kMelBinCount = 80;
constexpr int kMelSampleRate = 24000;
constexpr float kMelMaximumHz = 8000.0f;
constexpr float kMelMagnitudeFloor = 1e-5f;

uint16_t ReadU16(std::string_view bytes, size_t offset) {
  uint16_t value = 0;
  std::memcpy(&value, bytes.data() + offset, sizeof(value));
  return value;
}

uint32_t ReadU32(std::string_view bytes, size_t offset) {
  uint32_t value = 0;
  std::memcpy(&value, bytes.data() + offset, sizeof(value));
  return value;
}

// One stored sample as the 16-bit decoder produces it.
int16_t SampleToPcm16(const unsigned char *sample, uint16_t format,
                      int bits_per_sample) {
  if (format == kWavFormatPcm && bits_per_sample == 8) {
    return static_cast<int16_t>((static_cast<int>(sample[0]) << 8) - 32768);
  } else if (format == kWavFormatPcm && bits_per_sample == 16) {
    int16_t value = 0;
    std::memcpy(&value, sample, sizeof(value));
    return value;
  } else if (format == kWavFormatPcm && bits_per_sample == 24) {
    const int32_t value =
        static_cast<int32_t>((static_cast<uint32_t>(sample[0]) << 8) |
                             (static_cast<uint32_t>(sample[1]) << 16) |
                             (static_cast<uint32_t>(sample[2]) << 24));
    return static_cast<int16_t>(value >> 16);
  } else if (format == kWavFormatPcm && bits_per_sample == 32) {
    int32_t value = 0;
    std::memcpy(&value, sample, sizeof(value));
    return static_cast<int16_t>(value >> 16);
  } else if (bits_per_sample == 32) {
    float value = 0.0f;
    std::memcpy(&value, sample, sizeof(value));
    if (std::isnan(value)) {
      return 32767;
    } else {
    }
    // Note (Jiaxin Deng): the decoder's (x + 1) * 32767.5 is contracted into
    // one fused multiply-add; the unfused form misses one LSB now and then.
    value = std::clamp(value, -1.0f, 1.0f);
    return static_cast<int16_t>(
        static_cast<int>(std::fma(value, 32767.5f, 32767.5f)) - 32768);
  } else {
    double value = 0.0;
    std::memcpy(&value, sample, sizeof(value));
    if (std::isnan(value)) {
      return 32767;
    } else {
    }
    value = std::clamp(value, -1.0, 1.0);
    return static_cast<int16_t>(
        static_cast<int>(std::fma(value, 32767.5, 32767.5)) - 32768);
  }
}

// Chebyshev coefficients for the modified Bessel function I0.
constexpr double kBesselSmallArgument[] = {
    -4.41534164647933937950E-18, 3.33079451882223809783E-17,
    -2.43127984654795469359E-16, 1.71539128555513303061E-15,
    -1.16853328779934516808E-14, 7.67618549860493561688E-14,
    -4.85644678311192946090E-13, 2.95505266312963983461E-12,
    -1.72682629144155570723E-11, 9.67580903537323691224E-11,
    -5.18979560163526290666E-10, 2.65982372468238665035E-9,
    -1.30002500998624804212E-8,  6.04699502254191894932E-8,
    -2.67079385394061173391E-7,  1.11738753912010371815E-6,
    -4.41673835845875056359E-6,  1.64484480707288970893E-5,
    -5.75419501008210370398E-5,  1.88502885095841655729E-4,
    -5.76375574538582365885E-4,  1.63947561694133579842E-3,
    -4.32430999505057594430E-3,  1.05464603945949983183E-2,
    -2.37374148058994688156E-2,  4.93052842396707084878E-2,
    -9.49010970480476444210E-2,  1.71620901522208775349E-1,
    -3.04682672343198398683E-1,  6.76795274409476084995E-1};
constexpr double kBesselLargeArgument[] = {
    -7.23318048787475395456E-18, -4.83050448594418207126E-18,
    4.46562142029675999901E-17,  3.46122286769746109310E-17,
    -2.82762398051658348494E-16, -3.42548561967721913462E-16,
    1.77256013305652638360E-15,  3.81168066935262242075E-15,
    -9.55484669882830764870E-15, -4.15056934728722208663E-14,
    1.54008621752140982691E-14,  3.85277838274214270114E-13,
    7.18012445138366623367E-13,  -1.79417853150680611778E-12,
    -1.32158118404477131188E-11, -3.14991652796324136454E-11,
    1.18891471078464383424E-11,  4.94060238822496958910E-10,
    3.39623202570838634515E-9,   2.26666899049817806459E-8,
    2.04891858946906374183E-7,   2.89137052083475648297E-6,
    6.88975834691682398426E-5,   3.36911647825569408990E-3,
    8.04490411014108831608E-1};

template <size_t kCount>
double Chebyshev(double x, const double (&coefficients)[kCount]) {
  double b0 = coefficients[0];
  double b1 = 0.0;
  double b2 = 0.0;
  for (size_t i = 1; i < kCount; ++i) {
    b2 = b1;
    b1 = b0;
    b0 = x * b1 - b2 + coefficients[i];
  }
  return 0.5 * (b0 - b2);
}

double BesselI0(double x) {
  x = std::fabs(x);
  if (x <= 8.0) {
    return std::exp(x) * Chebyshev(x / 2.0 - 2.0, kBesselSmallArgument);
  } else {
    return std::exp(x) * Chebyshev(32.0 / x - 2.0, kBesselLargeArgument) /
           std::sqrt(x);
  }
}

// The low-pass FIR at the upsampled rate, scaled to unit DC gain.
std::vector<double> KaiserLowPass(long max_rate) {
  const long tap_count = 2 * kFilterZeroCrossings * max_rate + 1;
  const double cutoff = kFilterRolloff / static_cast<double>(max_rate);
  const double center = 0.5 * static_cast<double>(tap_count - 1);
  const double window_center = static_cast<double>(tap_count - 1) / 2.0;
  const double window_scale = BesselI0(kFilterKaiserBeta);
  std::vector<double> taps(tap_count);
  for (long i = 0; i < tap_count; ++i) {
    const double offset = static_cast<double>(i) - center;
    const double sinc_argument = cutoff * offset;
    const double phase =
        M_PI * (sinc_argument == 0.0 ? 1.0e-20 : sinc_argument);
    const double ratio =
        (static_cast<double>(i) - window_center) / window_center;
    const double window =
        BesselI0(kFilterKaiserBeta * std::sqrt(1.0 - ratio * ratio)) /
        window_scale;
    taps[i] = cutoff * (std::sin(phase) / phase) * window;
  }
  const double gain = PairwiseSum(taps.data(), taps.size());
  for (double &tap : taps) {
    tap /= gain;
  }
  return taps;
}

long PolyphaseOutputLength(long filter_length, long input_length, long up,
                           long down) {
  return ((input_length - 1) * up + filter_length - 1) / down + 1;
}

// Direct-form II transposed biquad over the whole signal, in place.
void Biquad(const double (&numerator)[3], const double (&denominator)[3],
            std::vector<double> &signal) {
  double state0 = 0.0;
  double state1 = 0.0;
  for (double &sample : signal) {
    const double input = sample;
    const double output = numerator[0] * input + state0;
    state0 = state1 + numerator[1] * input - denominator[1] * output;
    state1 = numerator[2] * input - denominator[2] * output;
    sample = output;
  }
}

double BlockLoudness(double mean_square) {
  return kLoudnessOffsetLufs + 10.0 * std::log10(mean_square);
}

// Mean of the selected blocks' mean squares; NaN when none is selected.
double GatedMeanSquare(const std::vector<double> &mean_squares,
                       const std::vector<bool> &selected) {
  std::vector<double> kept;
  for (size_t i = 0; i < mean_squares.size(); ++i) {
    if (selected[i]) {
      kept.push_back(mean_squares[i]);
    } else {
    }
  }
  if (kept.empty()) {
    return std::numeric_limits<double>::quiet_NaN();
  } else {
    return PairwiseSum(kept.data(), kept.size()) /
           static_cast<double>(kept.size());
  }
}

double IntegratedLoudness(const std::vector<float> &samples, int sample_rate) {
  std::vector<double> weighted(samples.begin(), samples.end());
  const double shelf_k = std::tan(M_PI * kShelfFrequencyHz / sample_rate);
  const double shelf_denominator =
      1.0 + shelf_k / kShelfQuality + shelf_k * shelf_k;
  const double shelf_gain = std::pow(10.0, kShelfGainDecibels / 20.0);
  const double mid_gain = std::pow(shelf_gain, kShelfMidGainExponent);
  const double shelf_numerator[3] = {
      (shelf_gain + mid_gain * shelf_k / kShelfQuality + shelf_k * shelf_k) /
          shelf_denominator,
      2.0 * (shelf_k * shelf_k - shelf_gain) / shelf_denominator,
      (shelf_gain - mid_gain * shelf_k / kShelfQuality + shelf_k * shelf_k) /
          shelf_denominator};
  const double shelf_feedback[3] = {
      1.0, 2.0 * (shelf_k * shelf_k - 1.0) / shelf_denominator,
      (1.0 - shelf_k / kShelfQuality + shelf_k * shelf_k) / shelf_denominator};
  const double pass_k = std::tan(M_PI * kHighPassFrequencyHz / sample_rate);
  const double pass_denominator =
      1.0 + pass_k / kHighPassQuality + pass_k * pass_k;
  const double pass_numerator[3] = {1.0, -2.0, 1.0};
  const double pass_feedback[3] = {
      1.0, 2.0 * (pass_k * pass_k - 1.0) / pass_denominator,
      (1.0 - pass_k / kHighPassQuality + pass_k * pass_k) / pass_denominator};
  Biquad(shelf_numerator, shelf_feedback, weighted);
  Biquad(pass_numerator, pass_feedback, weighted);

  const long block_samples =
      std::lround(std::nearbyint(kGatingBlockSeconds * sample_rate));
  const long hop_samples = std::lround(std::nearbyint(
      kGatingBlockSeconds * (1.0 - kGatingBlockOverlap) * sample_rate));
  const long sample_count = static_cast<long>(weighted.size());
  const long difference = sample_count - block_samples;
  // Floor division, as a short clip leaves no complete block.
  const long block_count =
      1 + (difference >= 0 ? difference / hop_samples
                           : -((-difference + hop_samples - 1) / hop_samples));
  if (block_count <= 0) {
    return -std::numeric_limits<double>::infinity();
  } else {
  }
  std::vector<double> mean_squares(block_count);
  std::vector<double> loudness(block_count);
  std::vector<double> squares(block_samples);
  for (long block = 0; block < block_count; ++block) {
    const double *block_start = weighted.data() + block * hop_samples;
    for (long i = 0; i < block_samples; ++i) {
      squares[i] = block_start[i] * block_start[i];
    }
    mean_squares[block] = PairwiseSum(squares.data(), squares.size()) /
                          static_cast<double>(block_samples);
    loudness[block] = BlockLoudness(mean_squares[block]);
  }
  std::vector<bool> selected(block_count);
  for (long block = 0; block < block_count; ++block) {
    selected[block] = loudness[block] > kAbsoluteGateLufs;
  }
  const double relative_gate =
      BlockLoudness(GatedMeanSquare(mean_squares, selected)) - kRelativeGateLu;
  for (long block = 0; block < block_count; ++block) {
    selected[block] =
        loudness[block] > relative_gate && loudness[block] > kAbsoluteGateLufs;
  }
  double gated_mean_square = GatedMeanSquare(mean_squares, selected);
  if (std::isnan(gated_mean_square)) {
    gated_mean_square = 0.0;
  } else {
  }
  return BlockLoudness(gated_mean_square);
}

// Slaney-normalized Slaney-scale filters [mel_bins, frequency_bins], built
// with float32 MLX operations.
std::vector<float> BuildMelFilters() {
  constexpr int frequency_bin_count = kMelFftSize / 2 + 1;
  const double linear_step_hz = 200.0 / 3;
  const double min_log_hz = 1000.0;
  const double min_log_mel = min_log_hz / linear_step_hz;
  const double log_step = std::log(6.4) / 27.0;
  const double max_mel =
      min_log_mel + std::log(kMelMaximumHz / min_log_hz) / log_step;
  const mx::array frequencies_hz =
      mx::linspace(0.0, kMelSampleRate / 2, frequency_bin_count, mx::float32);
  const mx::array mels =
      mx::linspace(0.0, max_mel, kMelBinCount + 2, mx::float32);
  const mx::array linear_hz = mx::add(
      mx::array(0.0f),
      mx::multiply(mx::array(static_cast<float>(linear_step_hz)), mels));
  const mx::array log_hz = mx::multiply(
      mx::array(static_cast<float>(min_log_hz)),
      mx::exp(mx::multiply(
          mx::array(static_cast<float>(log_step)),
          mx::subtract(mels, mx::array(static_cast<float>(min_log_mel))))));
  const mx::array edges_hz = mx::where(
      mx::greater_equal(mels, mx::array(static_cast<float>(min_log_mel))),
      log_hz, linear_hz);
  const int edge_count = kMelBinCount + 2;
  const mx::array edge_gaps =
      mx::subtract(mx::slice(edges_hz, {1}, {edge_count}),
                   mx::slice(edges_hz, {0}, {edge_count - 1}));
  const mx::array slopes = mx::subtract(mx::expand_dims(edges_hz, 0),
                                        mx::expand_dims(frequencies_hz, 1));
  const mx::array down_slopes =
      mx::divide(mx::negative(mx::slice(slopes, {0, 0},
                                        {frequency_bin_count, edge_count - 2})),
                 mx::slice(edge_gaps, {0}, {edge_count - 2}));
  const mx::array up_slopes =
      mx::divide(mx::slice(slopes, {0, 2}, {frequency_bin_count, edge_count}),
                 mx::slice(edge_gaps, {1}, {edge_count - 1}));
  mx::array filters = mx::maximum(mx::zeros_like(down_slopes),
                                  mx::minimum(down_slopes, up_slopes));
  const mx::array normalization = mx::divide(
      mx::array(2.0f), mx::subtract(mx::slice(edges_hz, {2}, {edge_count}),
                                    mx::slice(edges_hz, {0}, {kMelBinCount})));
  filters = mx::multiply(filters, mx::expand_dims(normalization, 0));
  filters = mx::contiguous(mx::moveaxis(filters, 0, 1));
  mx::eval(filters);
  const float *values = filters.data<float>();
  return std::vector<float>(values, values + filters.size());
}

// Index into a signal of size samples mirrored about its end samples.
long ReflectIndex(long index, long size) {
  if (size == 1) {
    return 0;
  } else {
  }
  const long period = 2 * (size - 1);
  long folded = index % period;
  if (folded < 0) {
    folded += period;
  } else {
  }
  return folded < size ? folded : period - folded;
}

void FftInPlace(std::complex<double> *values, const std::complex<double> *input,
                size_t input_stride, size_t size,
                const std::vector<std::complex<double>> &twiddles,
                size_t twiddle_stride) {
  if (size == 1) {
    values[0] = input[0];
    return;
  } else {
  }
  size_t radix = 2;
  while (size % radix != 0) {
    ++radix;
  }
  const size_t part_size = size / radix;
  for (size_t part = 0; part < radix; ++part) {
    FftInPlace(values + part * part_size, input + part * input_stride,
               input_stride * radix, part_size, twiddles,
               twiddle_stride * radix);
  }
  const size_t full_size = twiddles.size();
  std::vector<std::complex<double>> terms(radix);
  for (size_t bin = 0; bin < part_size; ++bin) {
    for (size_t part = 0; part < radix; ++part) {
      terms[part] = values[part * part_size + bin] *
                    twiddles[(part * bin * twiddle_stride) % full_size];
    }
    for (size_t block = 0; block < radix; ++block) {
      std::complex<double> sum = 0.0;
      for (size_t part = 0; part < radix; ++part) {
        sum +=
            terms[part] *
            twiddles[(part * block * part_size * twiddle_stride) % full_size];
      }
      values[bin + block * part_size] = sum;
    }
  }
}

} // namespace

double PairwiseSum(const double *values, size_t count) {
  if (count < 8) {
    double sum = -0.0;
    for (size_t i = 0; i < count; ++i) {
      sum += values[i];
    }
    return sum;
  } else if (count <= 128) {
    double partial[8];
    std::copy(values, values + 8, partial);
    size_t i = 8;
    for (; i < count - count % 8; i += 8) {
      for (size_t lane = 0; lane < 8; ++lane) {
        partial[lane] += values[i + lane];
      }
    }
    double sum = ((partial[0] + partial[1]) + (partial[2] + partial[3])) +
                 ((partial[4] + partial[5]) + (partial[6] + partial[7]));
    for (; i < count; ++i) {
      sum += values[i];
    }
    return sum;
  } else {
    size_t half = count / 2;
    half -= half % 8;
    return PairwiseSum(values, half) + PairwiseSum(values + half, count - half);
  }
}

std::vector<std::complex<double>> RealFft(const std::vector<double> &samples) {
  const size_t size = samples.size();
  std::vector<std::complex<double>> twiddles(size);
  for (size_t i = 0; i < size; ++i) {
    const double angle =
        -2.0 * M_PI * static_cast<double>(i) / static_cast<double>(size);
    twiddles[i] = {std::cos(angle), std::sin(angle)};
  }
  const std::vector<std::complex<double>> input(samples.begin(), samples.end());
  std::vector<std::complex<double>> spectrum(size);
  FftInPlace(spectrum.data(), input.data(), 1, size, twiddles, 1);
  spectrum.resize(size / 2 + 1);
  return spectrum;
}

std::vector<float> Resample(const std::vector<float> &samples, int from_rate,
                            int to_rate) {
  if (from_rate <= 0 || to_rate <= 0) {
    throw std::invalid_argument("sample rates must be positive");
  } else if (from_rate == to_rate) {
    return samples;
  } else {
  }
  const long divisor = std::gcd(from_rate, to_rate);
  const long up = to_rate / divisor;
  const long down = from_rate / divisor;
  const std::vector<double> lowpass = KaiserLowPass(std::max(up, down));
  const long half_length = (static_cast<long>(lowpass.size()) - 1) / 2;
  const long input_length = static_cast<long>(samples.size());
  const long output_length =
      input_length * up / down + (input_length * up % down > 0 ? 1 : 0);
  // Leading zeros put output sample 0 at the filter's center.
  const long leading_zeros = down - half_length % down;
  const long skipped_outputs = (half_length + leading_zeros) / down;
  long trailing_zeros = 0;
  while (PolyphaseOutputLength(
             static_cast<long>(lowpass.size()) + leading_zeros + trailing_zeros,
             input_length, up, down) < output_length + skipped_outputs) {
    ++trailing_zeros;
  }
  const long filter_length =
      static_cast<long>(lowpass.size()) + leading_zeros + trailing_zeros;
  const long taps_per_phase = (filter_length + up - 1) / up;
  // Each phase's taps, newest input last.
  std::vector<double> phase_taps(static_cast<size_t>(taps_per_phase) * up, 0.0);
  for (long phase = 0; phase < up; ++phase) {
    for (long j = 0; j < taps_per_phase; ++j) {
      const long tap = (taps_per_phase - 1 - j) * up + phase - leading_zeros;
      phase_taps[phase * taps_per_phase + j] =
          tap >= 0 && tap < static_cast<long>(lowpass.size())
              ? lowpass[tap] * static_cast<double>(up)
              : 0.0;
    }
  }
  std::vector<float> output(output_length);
  for (long i = 0; i < output_length; ++i) {
    const long position = (i + skipped_outputs) * down;
    const long newest_input = position / up;
    const double *taps = phase_taps.data() + (position % up) * taps_per_phase;
    const long oldest_input = newest_input - taps_per_phase + 1;
    double sum = 0.0;
    for (long j = 0; j < taps_per_phase; ++j) {
      const long input_index =
          std::clamp(oldest_input + j, 0L, input_length - 1);
      sum = sum + static_cast<double>(samples[input_index]) * taps[j];
    }
    output[i] = static_cast<float>(sum);
  }
  return output;
}

namespace {

struct WavLayout {
  uint16_t format;
  int channel_count;
  uint32_t sample_rate;
  int bits_per_sample;
  std::string_view data;
  size_t frame_count;
};

// The layout of a WAV file ReadAudio can decode at sample_rate; throws
// std::invalid_argument for one it cannot.
WavLayout ParseWav(std::string_view file_bytes, int sample_rate) {
  if (sample_rate <= 0) {
    throw std::invalid_argument("sample rate must be positive");
  } else if (file_bytes.size() < 12 || file_bytes.substr(0, 4) != "RIFF" ||
             file_bytes.substr(8, 4) != "WAVE") {
    throw std::invalid_argument("audio must be a RIFF/WAVE file");
  } else {
  }
  size_t offset = 12;
  bool has_format = false;
  uint16_t format = 0;
  int channel_count = 0;
  uint32_t file_sample_rate = 0;
  int bits_per_sample = 0;
  std::string_view data;
  bool has_data = false;
  while (!has_data && offset + 8 <= file_bytes.size()) {
    const std::string_view chunk_id = file_bytes.substr(offset, 4);
    const uint32_t chunk_size = ReadU32(file_bytes, offset + 4);
    const size_t body_offset = offset + 8;
    const size_t body_size =
        std::min<size_t>(chunk_size, file_bytes.size() - body_offset);
    if (chunk_id == "fmt ") {
      if (body_size < 16) {
        throw std::invalid_argument("WAV fmt chunk is truncated");
      } else {
      }
      format = ReadU16(file_bytes, body_offset);
      channel_count = ReadU16(file_bytes, body_offset + 2);
      file_sample_rate = ReadU32(file_bytes, body_offset + 4);
      bits_per_sample = ReadU16(file_bytes, body_offset + 14);
      if (format == kWavFormatExtensible) {
        if (body_size < 40) {
          throw std::invalid_argument("WAV extensible fmt chunk is truncated");
        } else {
        }
        // The subformat GUID starts with the plain format code.
        format = ReadU16(file_bytes, body_offset + 24);
      } else {
      }
      has_format = true;
    } else if (chunk_id == "data") {
      if (!has_format) {
        throw std::invalid_argument("WAV data chunk precedes its fmt chunk");
      } else {
      }
      data = file_bytes.substr(body_offset, body_size);
      has_data = true;
    } else {
    }
    offset = body_offset + static_cast<size_t>(chunk_size) + (chunk_size & 1u);
  }
  if (!has_data) {
    throw std::invalid_argument("WAV file has no data chunk");
  } else if (channel_count == 0) {
    throw std::invalid_argument("WAV file has no channels");
  } else if (channel_count > kMaximumChannelCount) {
    throw std::invalid_argument("WAV file has too many channels");
  } else if (!(format == kWavFormatPcm &&
               (bits_per_sample == 8 || bits_per_sample == 16 ||
                bits_per_sample == 24 || bits_per_sample == 32)) &&
             !(format == kWavFormatFloat &&
               (bits_per_sample == 32 || bits_per_sample == 64))) {
    throw std::invalid_argument(
        "WAV must be 8/16/24/32-bit PCM or 32/64-bit float");
  } else if (file_sample_rate < kMinimumFileSampleRate ||
             file_sample_rate > kMaximumFileSampleRate) {
    throw std::invalid_argument("WAV sample rate is out of range");
  } else {
  }
  const long divisor = std::gcd(static_cast<long>(file_sample_rate),
                                static_cast<long>(sample_rate));
  if (std::max(static_cast<long>(file_sample_rate),
               static_cast<long>(sample_rate)) /
          divisor >
      kMaximumResampleFactor) {
    throw std::invalid_argument("WAV sample rate is out of range");
  } else {
  }
  const size_t bytes_per_sample = static_cast<size_t>(bits_per_sample) / 8;
  const size_t bytes_per_frame = bytes_per_sample * channel_count;
  const size_t frame_count = data.size() / bytes_per_frame;
  if (frame_count == 0) {
    throw std::invalid_argument("WAV file has no samples");
  } else {
  }
  return {format,          channel_count, file_sample_rate,
          bits_per_sample, data,          frame_count};
}

} // namespace

double WavDurationSeconds(std::string_view file_bytes, int sample_rate) {
  const WavLayout layout = ParseWav(file_bytes, sample_rate);
  const long divisor = std::gcd(static_cast<long>(layout.sample_rate),
                                static_cast<long>(sample_rate));
  const long up = sample_rate / divisor;
  const long down = static_cast<long>(layout.sample_rate) / divisor;
  // Resample's output length.
  const long output_count =
      (static_cast<long>(layout.frame_count) * up + down - 1) / down;
  return static_cast<double>(output_count) / sample_rate;
}

std::vector<float> ReadAudio(std::string_view file_bytes, int sample_rate) {
  const WavLayout layout = ParseWav(file_bytes, sample_rate);
  const uint16_t format = layout.format;
  const int channel_count = layout.channel_count;
  const int bits_per_sample = layout.bits_per_sample;
  const size_t frame_count = layout.frame_count;
  const std::string_view data = layout.data;
  const size_t bytes_per_sample = static_cast<size_t>(bits_per_sample) / 8;
  const size_t bytes_per_frame = bytes_per_sample * channel_count;
  const auto *bytes = reinterpret_cast<const unsigned char *>(data.data());
  std::vector<float> samples(frame_count);
  for (size_t frame = 0; frame < frame_count; ++frame) {
    const unsigned char *frame_bytes = bytes + frame * bytes_per_frame;
    int16_t mono = 0;
    if (channel_count == 1) {
      mono = SampleToPcm16(frame_bytes, format, bits_per_sample);
    } else {
      int32_t sum = 0;
      for (int channel = 0; channel < channel_count; ++channel) {
        sum += SampleToPcm16(frame_bytes + channel * bytes_per_sample, format,
                             bits_per_sample);
      }
      // Note (Jiaxin Deng): the decoder divides the sum as unsigned and
      // keeps the low 16 bits, which for two channels is the floor of the
      // mean.
      mono = static_cast<int16_t>(static_cast<uint16_t>(
          static_cast<uint32_t>(sum) / static_cast<uint32_t>(channel_count)));
    }
    samples[frame] = static_cast<float>(mono) / kPcm16FullScale;
  }
  return Resample(samples, static_cast<int>(layout.sample_rate), sample_rate);
}

std::vector<float> NormalizeLoudness(std::vector<float> samples,
                                     int sample_rate, double target_lufs) {
  if (sample_rate <= 0 ||
      static_cast<double>(samples.size()) < kGatingBlockSeconds * sample_rate) {
    return samples;
  } else {
  }
  const double loudness = IntegratedLoudness(samples, sample_rate);
  const double gain = std::pow(10.0, (target_lufs - loudness) / 20.0);
  if (std::isfinite(gain) && gain > 0.0) {
    const float gain_float = static_cast<float>(gain);
    for (float &sample : samples) {
      sample *= gain_float;
    }
  } else {
  }
  return samples;
}

mx::array S3GenMel(const std::vector<float> &samples_24k) {
  const long sample_count = static_cast<long>(samples_24k.size());
  if (sample_count < kMelHopSamples) {
    throw std::invalid_argument("audio is too short for a mel frame");
  } else {
  }
  static const std::vector<float> filters = BuildMelFilters();
  static const std::vector<float> window = [] {
    std::vector<float> values(kMelFftSize);
    for (int i = 0; i < kMelFftSize; ++i) {
      const double position = static_cast<double>(1 - kMelFftSize + 2 * i);
      values[i] = static_cast<float>(
          0.5 + 0.5 * std::cos(M_PI * position / (kMelFftSize - 1)));
    }
    return values;
  }();
  constexpr int frequency_bin_count = kMelFftSize / 2 + 1;
  constexpr long padding = (kMelFftSize - kMelHopSamples) / 2;
  const long frame_count =
      1 + (sample_count + 2 * padding - kMelFftSize) / kMelHopSamples;
  std::vector<float> mel(static_cast<size_t>(frame_count) * kMelBinCount);
  std::vector<double> windowed(kMelFftSize);
  std::vector<double> magnitudes(frequency_bin_count);
  for (long frame = 0; frame < frame_count; ++frame) {
    for (int i = 0; i < kMelFftSize; ++i) {
      const long index =
          ReflectIndex(frame * kMelHopSamples + i - padding, sample_count);
      windowed[i] = static_cast<double>(samples_24k[index] * window[i]);
    }
    const std::vector<std::complex<double>> spectrum = RealFft(windowed);
    for (int bin = 0; bin < frequency_bin_count; ++bin) {
      magnitudes[bin] = static_cast<double>(static_cast<float>(
          std::hypot(spectrum[bin].real(), spectrum[bin].imag())));
    }
    for (int mel_bin = 0; mel_bin < kMelBinCount; ++mel_bin) {
      const float *row = filters.data() + mel_bin * frequency_bin_count;
      double energy = 0.0;
      for (int bin = 0; bin < frequency_bin_count; ++bin) {
        energy += static_cast<double>(row[bin]) * magnitudes[bin];
      }
      const float floored =
          std::max(static_cast<float>(energy), kMelMagnitudeFloor);
      mel[frame * kMelBinCount + mel_bin] =
          static_cast<float>(std::log(static_cast<double>(floored)));
    }
  }
  return mx::array(mel.data(), {1, static_cast<int>(frame_count), kMelBinCount},
                   mx::float32);
}

} // namespace chatterbox
