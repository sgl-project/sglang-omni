# EasyMagpie converter

Converts an EasyMagpie (Nemotron-TTS) NeMo talker checkpoint and its spectral
codec into a model directory that SGLang-Omni serves with
`--model-path <dir>`.

## Output layout

| Path | Contents |
| --- | --- |
| `config.json` | Nemotron-H backbone fields and EasyMagpie scalars |
| `model.safetensors`, `model.safetensors.index.json` | Talker weights; the text embedding path is baked into `text_embedding.weight` |
| `tokenizer.json`, `tokenizer_config.json` | Text tokenizer for target and context text |
| `phoneme_text_tokenizer/tokenizer.json` | IPA BPE for `<bop>...<eop>` spans (pronunciation-control checkpoints only) |
| `codec_native/config.json`, `codec_native/model.safetensors` | Causal codec decoder, weight norm folded |
| `speaker_embeddings/contexts.safetensors` | One context tensor per context id |
| `speaker_embeddings/contexts.json` | Voice registry (schema 1): contexts, `voice_contexts`, `langs_supported`, `default_speaker_id` |
| `conversion.json` | Names and sizes of the inputs |

The voice registry uses the Nemotron-Speech repository format. Each CSV row
`speaker_name, languages, context_text` becomes one context per speaker and
primary language. A speaker's first context uses the speaker name as its id.
Text-only voices share NeMo's omitted-audio context tensor and differ only by
their context text.

At serving time a request picks a context as follows:

| Request | Context |
| --- | --- |
| `voice` and `language` | `voice_contexts[voice][language]` |
| `voice` only | The voice's first context |
| `language` only | The first voice listed for the language |
| Neither | `default_speaker_id` |

The paired `context_text` is used unless the request sends its own.

## Running

The converter needs NeMo (`nemo.collections.tts`) and a GPU. One way to get
both is the server image from `../deploy`, with a NeMo source checkout
(NVIDIA-NeMo/Speech) and NeMo's extra dependencies installed into a separate
`--target` directory, so the image stays unchanged:

```bash
NEMO=/path/to/NeMo-Speech        # source checkout
TOOLS=/path/to/easymagpie_tools  # this directory's parent
A=/path/to/artifacts             # .nemo files, tokenizer, CSV
mkdir -p nemo_env
docker run --rm --entrypoint "" -v $PWD/nemo_env:/env easymagpie-tts:<tag> \
  pip install --target /env/site lhotse==2.0.0a6 opentelemetry-api==1.44.0 \
    onnx peft strenum overrides toml

docker run --rm --gpus all --entrypoint "" -u $(id -u):$(id -g) \
  -e HOME=/tmp -e HF_HOME=/hf -e PYTHONDONTWRITEBYTECODE=1 \
  -e PYTHONPATH=/conv:/nemo:/env/site \
  -v $HOME/.cache/huggingface:/hf -v $PWD/nemo_env:/env \
  -v $NEMO:/nemo:ro -v $TOOLS:/conv:ro \
  -v $A:/art -w /tmp easymagpie-tts:<tag> \
  python -m easymagpie_converter \
    --nemo-checkpoint /art/NemotronTTS_GRPO_ChallengingText_P15_ContextShuffle_P05_Prefix510_Step120.nemo \
    --codec-checkpoint /art/spectral_codec/25fps_spectral_codec_with_bandwidth_extension.nemo \
    --phoneme-tokenizer /art/bpe_ipa_tokenizer_3072_en_de_es_fr_hi_it_vi_zh_ko-KR_pt-BR_ar_full_byte_espeak152_v2610_2026-08.json \
    --context-texts-csv /art/context_texts.csv \
    --output /art/sglang_model
```

The output directory must not exist. The converter builds in a hidden sibling
directory and renames it into place only after every file is written.

## Differences from the NeMo converter

`upstream/` holds unmodified copies of the NeMo sources this package was
adapted from. `upstream/SOURCE.txt` records their paths and commit. Compared with
`convert_to_vllm.py`:

- No Perth watermark download and no codec encoder or reference-speaker
  bundle, because SGLang-Omni serves preset voices and does not watermark.
- No dependency on the `easymagpie_vllm_omni` package; the codec config is
  plain JSON.
- Text voices from a CSV are built in the same NeMo load as the talker,
  instead of a separate context conversion run.
- `config.json` also records `ignore_phoneme_languages` and
  `add_language_to_context_text`.
