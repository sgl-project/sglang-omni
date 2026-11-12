// SPDX-License-Identifier: Apache-2.0
// The host half of streaming codec decode, in float64 as the reference runs
// it: the two-branch centered iSTFT overlap-add and FIR mixer, then the
// streaming 44160 to 44100 Hz sinc resampler. Each returns only samples no
// later input can change; float32 crosses between them.
#pragma once

#include <complex>
#include <vector>

namespace zonos3 {

// numpy.fft.rfft(samples, n): bins [n / 2 + 1] of samples zero-padded to n.
std::vector<std::complex<double>> RealFft(const std::vector<double> &samples,
                                          int n);

struct IstftHeadConfig {
  std::vector<int> fft_sizes;
  int hop = 0;
  // Per branch, its analysis window [fft size].
  std::vector<std::vector<double>> windows;
  // [branches][taps], odd taps.
  std::vector<std::vector<double>> mixer;
  double max_log_magnitude = 0;
};

class IstftHeadStream {
public:
  explicit IstftHeadStream(IstftHeadConfig config);

  // Spectra per branch [columns][fft size + 2]: log magnitudes of the
  // fft size / 2 + 1 bins, then their phases. Returns the internal-rate
  // samples now final.
  std::vector<float>
  Push(const std::vector<std::vector<std::vector<float>>> &branch_spectra);
  // Every remaining sample; the mixer sees zeros past the end.
  std::vector<float> Flush();

private:
  void Finalize(int branch, long long final_position);
  std::vector<float> MixEmit();

  IstftHeadConfig config_;
  int half_taps_;
  std::vector<std::vector<double>> overlap_add_;
  std::vector<std::vector<double>> envelope_;
  std::vector<long long> overlap_add_base_;
  // Per branch, mixer input from absolute sample mix_base_ on.
  std::vector<std::vector<double>> mix_input_;
  long long mix_base_ = 0;
  long long frames_ = 0;
  bool done_ = false;
};

class ResamplerStream {
public:
  // kernel [to][2 * width + from].
  ResamplerStream(std::vector<std::vector<double>> kernel, int from, int to,
                  int width);

  std::vector<float> Push(const std::vector<float> &samples);
  std::vector<float> Flush();

private:
  std::vector<float> Run(bool flush);

  std::vector<std::vector<double>> kernel_;
  int from_;
  int to_;
  int width_;
  int taps_;
  std::vector<double> buffer_;
  // The absolute index of buffer_[0], samples pushed, and the next output
  // block.
  long long base_ = 0;
  long long pushed_ = 0;
  long long next_block_ = 0;
};

} // namespace zonos3
