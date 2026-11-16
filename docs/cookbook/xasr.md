# X-ASR

X-ASR is a Zipformer/RNN-T speech recognition model supporting Chinese and English. The 480 ms configuration uses a shared FP32 safetensors checkpoint and greedy decoding.

```bash
sgl-omni serve --model-path Chal1ce/X-ASR-zh-en-native-480ms
```

Send an audio file of at most 30 seconds to `/v1/audio/transcriptions` with the served model name. The original checkpoint emits unpunctuated text. Uploaded audio is decoded into 16 kHz filterbank features and processed in 480 ms model steps; multiple requests may be batched. This initial path returns a final transcription for an uploaded file and does not implement live streaming audio sessions or timestamps.

The model checkpoint is derived from [X-ASR](https://github.com/Gilgamesh-J/X-ASR) under Apache-2.0. The source repository and its model card describe training data and supported languages.
