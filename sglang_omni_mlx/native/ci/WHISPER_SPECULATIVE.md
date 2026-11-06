# Whisper distil decoding

Convert the pinned `wilsonzheng0327/Whisper-large-v3-distil-draft` checkpoint with `python ci/convert_whisper_draft.py HF_DIRECTORY NATIVE_DIRECTORY`. Use the existing pinned `mlx-community/whisper-large-v3-mlx` target and tokenizer.

Both `whisper_server` and `whisper_transcribe` accept `--draft-model-path NATIVE_DIRECTORY --draft-steps 7`. Startup checks the shared encoder weights and releases the draft encoder. Greedy requests verify draft blocks; positive-temperature requests use ordinary sampling. Omitting the draft preserves ordinary decoding. This is an independent distil decoder, not MTP.

Run `NATIVE_RUNTIME_BIN=BUILD_DIRECTORY python -m pytest tests/test_whisper_speculative.py` from the native directory. Full-corpus accuracy, long-audio performance and Mac CI remain required before this draft is ready.
