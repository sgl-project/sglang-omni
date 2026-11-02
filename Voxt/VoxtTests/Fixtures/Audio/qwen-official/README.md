## Qwen Official Audio Fixtures

These fixtures are for local-only transcription replay and real-model regression checks.
They are intentionally excluded from CI because they exercise installed local models and
larger audio assets.

### Sources

- `qwen3_asr_long_en.wav`
  - Source: `https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-ASR-Repo/asr_en.wav`
  - Origin: official `Qwen3-ASR` demo asset
  - Notes: official demo audio; the repo does not publish a canonical reference transcript in the README

- `qwen3_asr_long_zh.wav`
  - Source: `https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-ASR-Repo/asr_zh.wav`
  - Origin: official `Qwen3-ASR` demo asset
  - Notes: official demo audio; the repo does not publish a canonical reference transcript in the README

- `qwen_audio_short_en.wav`
  - Source audio: `https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Audio/1272-128104-0000.flac`
  - Converted locally to WAV for easier replay
  - Origin: official `Qwen-Audio` sample
  - Reference transcript:
    - `mister quilter is the apostle of the middle classes and we are glad to welcome his gospel`

- `qwen_audio_short_zh_chongqing.wav`
  - Source: `https://raw.githubusercontent.com/QwenLM/Qwen-Audio/main/assets/audio/example-%E9%87%8D%E5%BA%86%E8%AF%9D.wav`
  - Origin: official `Qwen-Audio` sample
  - Reference transcript:
    - `对了我还想提议我们可以租一些自行车骑行一下既锻炼身体又心情愉悦`

- `qwen_audio_short_zh_relaxed.wav`
  - Source: `https://raw.githubusercontent.com/QwenLM/Qwen-Audio/main/assets/audio/%E4%BD%A0%E6%B2%A1%E4%BA%8B%E5%90%A7-%E8%BD%BB%E6%9D%BE.wav`
  - Origin: official `Qwen-Audio` sample
  - Reference transcript:
    - `你没事吧`

- `qwen_audio_short_zh_negative.wav`
  - Source: `https://raw.githubusercontent.com/QwenLM/Qwen-Audio/main/assets/audio/%E4%BD%A0%E6%B2%A1%E4%BA%8B%E5%90%A7-%E6%B6%88%E6%9E%81.wav`
  - Origin: official `Qwen-Audio` sample
  - Reference transcript:
    - `你没事吧`

### Composite Long-Form Fixtures

These are built from official Qwen short samples so long-form replay tests can use fully
public, referenceable material instead of private local history clips.

- `qwen_audio_long_en_composite.wav`
  - Built from: `qwen_audio_short_en.wav` repeated 6 times
  - Duration: about 35.13s

- `qwen_audio_long_zh_composite.wav`
  - Built from:
    - `qwen_audio_short_zh_chongqing.wav`
    - `qwen_audio_short_zh_relaxed.wav`
    - `qwen_audio_short_zh_negative.wav`
    - repeated twice
  - Duration: about 35.76s

### CI Policy

Fixture-backed real-model tests are skipped by default. Run them explicitly with
`VOXT_RUN_MODEL_TESTS=1` plus the desired `-only-testing:` filter.
