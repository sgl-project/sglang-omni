// SPDX-License-Identifier: Apache-2.0
// The CAM++ speaker x-vector encoder: a Kaldi-style log-mel fbank front end
// (Povey window, pre-emphasis, DC removal, snip_edges, a fixed 512-FFT via a
// precomputed DFT, 80-bin mel bank, log floor, per-bin CMN) followed by the
// 3D-Speaker CAM++ conv trunk. Conv weights are pre-transposed to MLX layout
// and every BatchNorm folded to a frozen affine at load.
#pragma once

#include <optional>
#include <utility>
#include <vector>

#include "dots_tts/dots_tts_checkpoint.h"
#include "mlx/mlx.h"

namespace dots_tts {

namespace mx = mlx::core;

class KaldiFbank {
public:
  KaldiFbank();

  std::vector<float> operator()(const std::vector<float> &wav) const;

private:
  std::vector<float> window_;
  std::vector<float> dft_cos_; // note (guozhihao-224): [257, 512] real part
  std::vector<float> dft_sin_; // [257, 512] imaginary part
  std::vector<float> mel_;     // [80, 257]
};

struct BN {
  mx::array scale;
  mx::array shift;
  mx::array operator()(const mx::array &x) const {
    return mx::add(mx::multiply(x, scale), shift);
  }
};

struct SpeakerConv2d {
  mx::array weight;
  std::optional<mx::array> bias;
  std::pair<int, int> stride{1, 1};
  std::pair<int, int> padding{1, 1};
  mx::array operator()(const mx::array &x) const;
};

struct SpeakerConv1d {
  mx::array weight;
  std::optional<mx::array> bias;
  int stride = 1;
  int padding = 0;
  int dilation = 1;
  mx::array operator()(const mx::array &x) const;
};

struct BasicResBlock {
  SpeakerConv2d conv1;
  BN bn1;
  SpeakerConv2d conv2;
  BN bn2;
  std::optional<SpeakerConv2d> shortcut_conv;
  std::optional<BN> shortcut_bn;

  mx::array operator()(const mx::array &x) const;
};

struct FCM {
  SpeakerConv2d conv1;
  BN bn1;
  std::vector<BasicResBlock> layer1;
  std::vector<BasicResBlock> layer2;
  SpeakerConv2d conv2;
  BN bn2;

  mx::array operator()(const mx::array &x) const;
};

struct TDNNLayer {
  SpeakerConv1d conv;
  BN bn;
  mx::array operator()(const mx::array &x) const;
};

mx::array SegPooling(const mx::array &x, int seg_len = 100);

struct CAMLayer {
  SpeakerConv1d linear_local;
  SpeakerConv1d linear1;
  SpeakerConv1d linear2;
  mx::array operator()(const mx::array &x) const;
};

struct CamDenseTDNNLayer {
  BN bn1;
  SpeakerConv1d linear1;
  BN bn2;
  CAMLayer cam_layer;
  mx::array operator()(const mx::array &x) const;
};

struct CamDenseTDNNBlock {
  std::vector<CamDenseTDNNLayer> layers;
  mx::array operator()(const mx::array &x) const;
};

struct TransitLayer {
  BN bn;
  SpeakerConv1d linear;
  mx::array operator()(const mx::array &x) const;
};

mx::array StatsPooling(const mx::array &x);

struct CAMPPlus {
  FCM head;
  TDNNLayer tdnn;
  CamDenseTDNNBlock block1;
  TransitLayer transit1;
  CamDenseTDNNBlock block2;
  TransitLayer transit2;
  CamDenseTDNNBlock block3;
  TransitLayer transit3;
  BN out_nonlinear;
  SpeakerConv1d dense_linear;
  BN dense_bn;

  mx::array operator()(const mx::array &x) const;
};

CAMPPlus LoadSpeaker(const DotsCheckpoint &checkpoint);

inline constexpr int kSpeakerFbankSampleRate = 16000;
inline constexpr int kSpeakerMelBins = 80;

} // namespace dots_tts
