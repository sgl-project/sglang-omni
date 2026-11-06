# Ming3 audio MTP decoding

Both `ming3_audio_server` and `ming3_audio_chat` accept `--mtp 1` to use the pinned Ming3-audio-tiny checkpoint's single prediction layer. The default remains ordinary greedy decoding; no separate draft download is needed.

Each round verifies one proposal. Rejection restores attention, convolution and recurrent state before replaying the anchor. The MTP cache advances only from committed target states. Low acceptance can make this implementation slower; full-model performance has not yet been measured.

Build with `-DSGLANG_OMNI_MLX_BUILD_TESTS=ON` and target `ming3_mtp_probe`, then run `NATIVE_RUNTIME_BIN=BUILD_DIRECTORY python -m pytest tests/test_ming3_mtp.py` from the native directory. These reduced-checkpoint tests do not replace full-corpus accuracy, long-audio performance or Mac CI. Keep this PR in draft until those experiments are complete.
