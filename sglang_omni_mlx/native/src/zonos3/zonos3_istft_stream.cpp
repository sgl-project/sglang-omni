// SPDX-License-Identifier: Apache-2.0
// Line-for-line ports of mlx-audio's zonos3/streaming.py IstftHeadStream and
// ResamplerStream. The inverse real FFT is a mixed-radix transform in float64
// where the reference calls NumPy's; their roundings differ below float32's.
#include "zonos3/zonos3_istft_stream.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <map>
#include <mutex>
#include <numbers>
#include <stdexcept>
#include <utility>

namespace zonos3 {

namespace {

using Complex = std::complex<double>;

// e^(2 pi i j / n) for j in [0, n), built once per size.
const std::vector<Complex> &Twiddles(int n) {
  static std::mutex mutex;
  static std::map<int, std::vector<Complex>> tables;
  std::lock_guard<std::mutex> lock(mutex);
  auto found = tables.find(n);
  if (found == tables.end()) {
    std::vector<Complex> table(n);
    for (int j = 0; j < n; ++j) {
      table[j] = std::polar(1.0, 2.0 * std::numbers::pi * j / n);
    }
    found = tables.emplace(n, std::move(table)).first;
  } else {
  }
  return found->second;
}

int SmallestFactor(int n) {
  for (int factor = 2; factor * factor <= n; ++factor) {
    if (n % factor == 0) {
      return factor;
    } else {
    }
  }
  return n;
}

// The unnormalized DFT of input[offset + k * stride], k < n, into output [n],
// by decimation in time over n's smallest prime factor; inverse turns the
// exponent's sign positive.
void Dft(const std::vector<Complex> &input, int offset, int stride, int n,
         int full_size, bool inverse, Complex *output) {
  if (n == 1) {
    output[0] = input[offset];
    return;
  } else {
  }
  const int factor = SmallestFactor(n);
  const int part = n / factor;
  std::vector<Complex> parts(n);
  for (int residue = 0; residue < factor; ++residue) {
    Dft(input, offset + residue * stride, stride * factor, part, full_size,
        inverse, parts.data() + residue * part);
  }
  // A twiddle of this size n is the full table's entry times full_size / n.
  const std::vector<Complex> &twiddles = Twiddles(full_size);
  const int step = full_size / n;
  for (int quotient = 0; quotient < factor; ++quotient) {
    for (int k = 0; k < part; ++k) {
      const int bin = quotient * part + k;
      Complex sum = 0;
      for (int residue = 0; residue < factor; ++residue) {
        const long long index =
            static_cast<long long>(residue) * bin % n * step;
        const Complex twiddle =
            inverse ? twiddles[index] : std::conj(twiddles[index]);
        sum += twiddle * parts[residue * part + k];
      }
      output[bin] = sum;
    }
  }
}

// numpy.fft.irfft(spectrum, n): the Hermitian extension's inverse DFT over
// n; the imaginary parts of the DC and Nyquist bins are ignored.
std::vector<double> InverseRealFft(const std::vector<Complex> &spectrum,
                                   int n) {
  std::vector<Complex> full(n);
  const int bins = n / 2 + 1;
  for (int k = 0; k < n; ++k) {
    if (k < bins) {
      full[k] = spectrum[k];
    } else {
      full[k] = std::conj(spectrum[n - k]);
    }
  }
  full[0] = Complex(full[0].real(), 0);
  if (n % 2 == 0) {
    full[n / 2] = Complex(full[n / 2].real(), 0);
  } else {
  }
  std::vector<Complex> transformed(n);
  Dft(full, 0, 1, n, n, true, transformed.data());
  std::vector<double> samples(n);
  for (int t = 0; t < n; ++t) {
    samples[t] = transformed[t].real() / n;
  }
  return samples;
}

std::vector<float> ToFloat(const std::vector<double> &samples) {
  return std::vector<float>(samples.begin(), samples.end());
}

long long CeilDivide(long long numerator, long long denominator) {
  return (numerator + denominator - 1) / denominator;
}

} // namespace

std::vector<Complex> RealFft(const std::vector<double> &samples, int n) {
  std::vector<Complex> padded(n, 0.0);
  for (size_t t = 0; t < samples.size() && static_cast<int>(t) < n; ++t) {
    padded[t] = samples[t];
  }
  std::vector<Complex> transformed(n);
  Dft(padded, 0, 1, n, n, false, transformed.data());
  transformed.resize(n / 2 + 1);
  return transformed;
}

IstftHeadStream::IstftHeadStream(IstftHeadConfig config)
    : config_(std::move(config)) {
  if (config_.mixer.size() != 2 || config_.mixer[0].size() % 2 != 1 ||
      config_.mixer[1].size() != config_.mixer[0].size()) {
    throw std::runtime_error("istft: mixer kernel must be odd");
  } else {
  }
  for (size_t branch = 0; branch < config_.fft_sizes.size(); ++branch) {
    const int fft_size = config_.fft_sizes[branch];
    if (static_cast<int>(config_.windows[branch].size()) != fft_size ||
        fft_size % config_.hop != 0) {
      throw std::runtime_error("istft: window does not fit its fft size");
    } else {
    }
  }
  half_taps_ = (static_cast<int>(config_.mixer[0].size()) - 1) / 2;
  overlap_add_.assign(2, {});
  envelope_.assign(2, {});
  overlap_add_base_.assign(2, 0);
  mix_input_.assign(2, std::vector<double>(half_taps_, 0.0));
}

std::vector<float> IstftHeadStream::Push(
    const std::vector<std::vector<std::vector<float>>> &branch_spectra) {
  if (done_) {
    throw std::logic_error("istft: push after flush");
  } else {
  }
  const long long columns = branch_spectra[0].size();
  if (columns == 0) {
    return {};
  } else {
  }
  const int hop = config_.hop;
  for (int branch = 0; branch < 2; ++branch) {
    const int fft_size = config_.fft_sizes[branch];
    const int bins = fft_size / 2 + 1;
    const std::vector<double> &window = config_.windows[branch];
    const int overlap = fft_size / hop;
    const long long span = (columns + overlap - 1) * hop;
    std::vector<std::vector<double>> frames(columns);
    for (long long column = 0; column < columns; ++column) {
      const std::vector<float> &spectrum = branch_spectra[branch][column];
      std::vector<Complex> bins_complex(bins);
      for (int bin = 0; bin < bins; ++bin) {
        const double magnitude = std::exp(
            std::min<double>(spectrum[bin], config_.max_log_magnitude));
        const double phase = spectrum[bins + bin];
        const double imaginary =
            bin == 0 || bin == bins - 1 ? 0.0 : magnitude * std::sin(phase);
        bins_complex[bin] = Complex(magnitude * std::cos(phase), imaginary);
      }
      frames[column] = InverseRealFft(bins_complex, fft_size);
      for (int t = 0; t < fft_size; ++t) {
        frames[column][t] *= window[t];
      }
    }
    std::vector<double> added(span, 0.0);
    std::vector<double> envelope(span, 0.0);
    for (int slot = 0; slot < overlap; ++slot) {
      for (long long column = 0; column < columns; ++column) {
        for (int sample = 0; sample < hop; ++sample) {
          const double tap = window[slot * hop + sample];
          added[(slot + column) * hop + sample] +=
              frames[column][slot * hop + sample];
          envelope[(slot + column) * hop + sample] += tap * tap;
        }
      }
    }
    const long long offset = frames_ * hop - overlap_add_base_[branch];
    const long long need = offset + span;
    if (static_cast<long long>(overlap_add_[branch].size()) < need) {
      overlap_add_[branch].resize(need, 0.0);
      envelope_[branch].resize(need, 0.0);
    } else {
    }
    for (long long i = 0; i < span; ++i) {
      overlap_add_[branch][offset + i] += added[i];
      envelope_[branch][offset + i] += envelope[i];
    }
  }
  frames_ += columns;
  for (int branch = 0; branch < 2; ++branch) {
    Finalize(branch, frames_ * hop);
  }
  return MixEmit();
}

std::vector<float> IstftHeadStream::Flush() {
  if (done_ || frames_ == 0) {
    return {};
  } else {
  }
  done_ = true;
  const long long length = frames_ * config_.hop;
  for (int branch = 0; branch < 2; ++branch) {
    Finalize(branch, overlap_add_base_[branch] +
                         static_cast<long long>(overlap_add_[branch].size()));
    const long long keep = half_taps_ + length - mix_base_;
    std::vector<double> &input = mix_input_[branch];
    input.resize(keep, 0.0);
    input.insert(input.end(), half_taps_, 0.0);
  }
  return MixEmit();
}

void IstftHeadStream::Finalize(int branch, long long final_position) {
  const long long count = final_position - overlap_add_base_[branch];
  if (count <= 0) {
    return;
  } else {
  }
  const long long skip = std::min<long long>(
      count, std::max<long long>(0, config_.fft_sizes[branch] / 2 -
                                        overlap_add_base_[branch]));
  std::vector<double> &overlap_add = overlap_add_[branch];
  std::vector<double> &envelope = envelope_[branch];
  for (long long i = skip; i < count; ++i) {
    mix_input_[branch].push_back(overlap_add[i] / envelope[i]);
  }
  overlap_add.erase(overlap_add.begin(), overlap_add.begin() + count);
  envelope.erase(envelope.begin(), envelope.begin() + count);
  overlap_add_base_[branch] = final_position;
}

std::vector<float> IstftHeadStream::MixEmit() {
  const long long taps = config_.mixer[0].size();
  const long long available = std::min(mix_input_[0].size(), mix_input_[1].size());
  const long long last = available + mix_base_ - taps + 1;
  if (last <= mix_base_) {
    return {};
  } else {
  }
  const long long count = last - mix_base_;
  std::vector<double> mixed(count, 0.0);
  for (long long m = 0; m < count; ++m) {
    double first = 0;
    double second = 0;
    for (long long k = 0; k < taps; ++k) {
      first += mix_input_[0][m + k] * config_.mixer[0][k];
      second += mix_input_[1][m + k] * config_.mixer[1][k];
    }
    mixed[m] = first + second;
  }
  for (std::vector<double> &input : mix_input_) {
    input.erase(input.begin(), input.begin() + count);
  }
  mix_base_ = last;
  return ToFloat(mixed);
}

ResamplerStream::ResamplerStream(std::vector<std::vector<double>> kernel,
                                 int from, int to, int width)
    : kernel_(std::move(kernel)), from_(from), to_(to), width_(width),
      taps_(2 * width + from) {
  if (static_cast<int>(kernel_.size()) != to_ ||
      static_cast<int>(kernel_[0].size()) != taps_) {
    throw std::runtime_error("resampler: kernel does not fit its rates");
  } else {
  }
}

std::vector<float> ResamplerStream::Push(const std::vector<float> &samples) {
  buffer_.insert(buffer_.end(), samples.begin(), samples.end());
  pushed_ += samples.size();
  return Run(false);
}

std::vector<float> ResamplerStream::Flush() { return Run(true); }

std::vector<float> ResamplerStream::Run(bool flush) {
  long long total = 0;
  long long end = 0;
  if (flush) {
    total = CeilDivide(to_ * pushed_, from_);
    end = CeilDivide(total, to_);
  } else {
    const long long reach = pushed_ + width_ - taps_;
    end = reach >= 0 ? reach / from_ + 1 : 0;
  }
  std::vector<double> resampled;
  if (end > next_block_) {
    const long long low = next_block_ * from_ - width_;
    const long long high = (end - 1) * from_ - width_ + taps_;
    std::vector<double> segment(high - low, 0.0);
    const long long first = std::max<long long>(low, 0);
    const long long stop = std::min(high, pushed_);
    for (long long i = first; i < stop; ++i) {
      segment[i - low] = buffer_[i - base_];
    }
    const long long blocks = end - next_block_;
    resampled.reserve(blocks * to_);
    for (long long block = 0; block < blocks; ++block) {
      const double *window = segment.data() + block * from_;
      for (int phase = 0; phase < to_; ++phase) {
        double sum = 0;
        for (int k = 0; k < taps_; ++k) {
          sum += window[k] * kernel_[phase][k];
        }
        resampled.push_back(sum);
      }
    }
    if (flush) {
      resampled.resize(total - next_block_ * to_);
    } else {
    }
    next_block_ = end;
  } else {
  }
  const long long keep_from =
      std::min(pushed_, std::max<long long>(0, next_block_ * from_ - width_));
  if (keep_from > base_) {
    buffer_.erase(buffer_.begin(), buffer_.begin() + (keep_from - base_));
    base_ = keep_from;
  } else {
  }
  return ToFloat(resampled);
}

} // namespace zonos3
