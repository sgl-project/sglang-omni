# Speech sample of the end-to-end tests

`sample-zh-en.wav`: 10.5 seconds of speech mixing Chinese and English, as
a 16 kHz, mono, 16-bit PCM WAV file (44-byte header, no extra chunks). The
Mac tests play it through the helper's file input (`TestMacHelperCaptureFile`,
`TestMacE2ESessionLocal`, `TestMacE2EOmni`); `internal/vad` and
`internal/dsp` analyse it. They only check that a transcript is not
empty; the golden text below is for reading the logs.

## Golden text

```
今天下午三点开会，讨论 VoxtNew 的发布计划。Please send the meeting notes to the team.
```

"VoxtNew" is a made-up word, and this voice says it close to "Fox New":
both Qwen3-ASR models write "Fox New", even with "VoxtNew" as their
context (the dictionary). The eval set's `dict-*` clips measure the
dictionary (docs/eval.md).

## Where it came from

Synthetic speech, no human voice (re-synthesised 2026-10-17: the first
take, from before the rename, said "Auralink"):

- Model: [Qwen3-TTS-12Hz-0.6B-CustomVoice](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice)
  (Alibaba Qwen, **Apache-2.0**), in its MLX 8-bit conversion
  `mlx-community/Qwen3-TTS-12Hz-0.6B-CustomVoice-8bit`. Built-in speaker
  `Uncle_Fu`, seed 5, the whole golden text in one call.
- Engine: AudioSmith's `audiosmith-engine` 0.3.0 (mlx-audio-swift), its
  one-shot mode, driven by `internal/tools/evalsynth`, which also does the
  post-processing (in Go, as the first take's SoX recipe): leading and
  trailing silence under -60 dBFS trimmed, 0.3 s of silence before, the
  end padded to 10.5 s, the peak normalised to -3 dBFS, resampled to 16 kHz
  (windowed sinc), 16-bit with a fixed-seed triangular dither:

  ```
  go run ./internal/tools/evalsynth -engine …/audiosmith-engine -voice Uncle_Fu -seed 5 -length 10.5 \
    -text "今天下午三点开会，讨论 VoxtNew 的发布计划。Please send the meeting notes to the team." \
    -o testdata/speech/sample-zh-en.wav
  ```

  The speech starts at 0.34 s and ends at 9.94 s (`internal/vad`; Silero:
  0.35–9.98 s); its
  speech is at -20.5 dBFS (`internal/dsp`, `TestCompareVariants`); the
  tests that depend on the level and the onset measure them.
- AudioSmith adds no watermark on this path. The eval set's clips
  (`testdata/eval`) are made the same way.

Apache-2.0 puts no condition on the model's outputs; the file is shipped
under VoxtNew's licence (Apache-2.0; NOTICE).

Checked with the two curated recognisers on mac-air (`TestMacE2EOmni`, the
dictionary's "VoxtNew" as context):

| Recogniser | Transcript |
|---|---|
| Qwen3-ASR 1.7B (8-bit, `asr.accurate`) | 今天下午三点开会，讨论 Fox New 的发布计划。Please send the meeting notes to the team. |
| Qwen3-ASR 0.6B (4-bit, `asr.fast`) | 今天下午三点开会，讨论Fox New的发布计划。Please send the meeting notes to the team. |

Do not replace this file with macOS `say` output: its voices' terms do
not clearly allow redistribution.
