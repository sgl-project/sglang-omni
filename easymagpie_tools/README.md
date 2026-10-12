# EasyMagpie tools

Tools for serving EasyMagpie (Nemotron-TTS) with SGLang-Omni. The model support
itself lives in `sglang_omni/models/easymagpie_tts`; see
`docs/cookbook/easymagpie_tts.md` for the request API.

| Directory | Purpose |
|---|---|
| [`easymagpie_converter/`](easymagpie_converter/README.md) | Convert a NeMo talker checkpoint, its spectral codec and a voice CSV into the model directory the server loads |
| [`deploy/`](deploy/README.md) | Build the server image and start it with one command: `./deploy/serve.sh /path/to/model` |

A typical setup converts once, then serves the output directory:

```bash
python -m easymagpie_converter --nemo-checkpoint ... --output /models/easymagpie
./deploy/serve.sh /models/easymagpie
```
