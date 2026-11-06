# YuE2 implementation attribution

The model equations, checkpoint tensor names, text tokenizer protocol, sampling arithmetic, fixed-capacity AR graph attention and acoustic solver are adapted from [YuE2](https://github.com/multimodal-art-projection/YuE) at commit `1116bea76032bee5668e5f097294d082fa0b84f0`, licensed under Apache-2.0. The Omni request handling, scheduler and model integration are implemented separately.

Packed request batching and bounded graph reuse were also informed by [SGLang-Omni PR #2540](https://github.com/sgl-project/sglang-omni/pull/2540) at commit `85ae4b733144fdc8746c5cbc131bdd1a92a8d5a9`. This integration retains independent request sampling, completion and cancellation state.

The Oobleck decoder and SnakeBeta equations derive from stable-audio-tools commit `a6ae0cdf8b2eb1567a4b42ceadddec3712d99d45` and BigVGAN, as identified by YuE2's third-party notices. Their MIT license texts are retained in `licenses/stable-audio-tools-MIT.txt` and `licenses/SnakeBeta-NVIDIA-MIT.txt`.

The model weights have separate terms in the model repositories; this code license does not relicense those weights.
