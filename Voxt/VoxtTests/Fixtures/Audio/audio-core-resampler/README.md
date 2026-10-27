# Pinned AudioCore resampling baseline

`baseline.bin` was captured **before package unlink**, by calling the actual
`resampleAudio([Float],from:to:)` in the compiled MLXAudioCore product pinned to
`2a6e75d28ae6a399ba7c7aec842384ef7a3142b5`. Expected samples were not produced by
Voxt's replacement converter. The prior direct old/new comparison and the capture
XCTest passed on the same Mac.

Capture platform: Apple M4 arm64, macOS 26.7 (25G229), Xcode 26.6 (17F113),
SDK 26.5, Apple Swift 6.3.3. AVAudioConverter DSP output can be platform-specific;
this is a genuine platform baseline, not certification of every OS/SDK version.
Do not regenerate expected samples from the implementation under test to silence
failures. Keep original capture provenance when deliberately adding a platform.

- 42 cases: 5 rate pairs × 7 frame boundaries, silence/constant/impulse, and
  4 integer-truncated source rates with the modulo-17 signal.
- Inputs and comparison retain the original test formulas and Float32 bits.
- Capture JSON: 527,931 bytes, SHA256
  `1a06f9f061077263e21bdc068d6eb27e1d66a0c7dda5ee4fec167d716752f2ab`.
- Binary: 201,024 bytes, SHA256
  `b8c7bd65ad8e83d31ca26dd21a99919f6bf965e19867551df3909211479c7362`.

Binary encoding is little-endian: ASCII `VOXTAC01`, UInt32 case count, then each
case's five UInt32 fields (signal, source Hz, target Hz, input sample count, output
sample count), followed by UInt32 output Float32 bit patterns. Signals are
1=tone, 2=silence, 3=constant, 4=impulse, 5=modulo17. There is no user audio.

Capture/source/result evidence remains under the task's independent Mac validation
workspace and `/tmp/voxt-omni-migration/audio-unlink-wave`; those paths are evidence,
not a test dependency. The committed fixture is the test input.
