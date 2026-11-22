# Turn a recording into a subtitle file

[中文](../file_to_srt.md)

You have a recording of a meeting or a talk and want a subtitle file with timestamps, or a plain transcript. Voxt's own command line does it with the same speech model you chose for **Transcribe**.

1. Keep Voxt running and make sure a speech model is chosen on the **Transcribe** page.
2. Open Terminal and run:

   ```sh
   VOXT=/Applications/Voxt.app/Contents/MacOS/Voxt
   $VOXT transcribe talk.m4a -o talk.srt
   ```

3. The result, `talk.srt`, is a subtitle file with start and end times. For plain text without timestamps, add `--format txt`; when the recording is not in your main language, add a language hint such as `--lang zh`.

Voxt splits the recording at pauses into cues of at most about seven seconds and leaves out cues where nothing was heard. Common formats (m4a, mp3, wav, mp4, mov and similar) work directly; mkv and webm need a conversion first, for example with ffmpeg. The full options, exit codes and privacy notes are in the [command-line transcription doc](https://github.com/sgl-project/sglang-omni/blob/main/Voxt/docs/transcribe.md).

## For agents

> Following this page, check that Voxt is running and Transcribe has a speech model chosen. Run the command on the recording I name, writing to the location I name. Tell me the output file and how the command exited.
