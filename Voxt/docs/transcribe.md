# Transcribing a file from the command line

`Voxt transcribe` turns a recording on your Mac (a meeting, a voice memo, a screen recording or another video) into SRT subtitles or plain text with the speech model you chose for Transcribe on the Speech models page. Use it to archive recordings, or to give a transcript to a language model or a script that summarizes it.

## The command

The executable is the app itself. Voxt must be running:

```sh
VOXT=/Applications/Voxt.app/Contents/MacOS/Voxt
$VOXT transcribe meeting.m4a -o meeting.srt
$VOXT transcribe talk.mp4 --format txt > talk.txt
$VOXT transcribe interview.mov --lang en -o interview.srt
```

- `FILE` is an audio file or a video that macOS reads without extra software: m4a, mp3, wav, aiff, caf, mp4, mov and similar. For a video, Voxt uses its sound. Other containers, such as mkv and webm, fail with exit 5: convert them first, for example with `ffmpeg -i in.mkv -vn out.m4a`.
- `--format` is `srt` (default: numbered cues with start and end times) or `txt` (one line per cue, no times).
- `-o OUT` writes to a file; without it, or with `-o -`, the result goes to stdout. Progress goes to stderr. The file is written only when the transcription finishes, so a failed run leaves no partial file.
- `--lang CODE` is a language hint for the speech model, such as `zh` or `en`. Without it, Voxt uses the same hint as dictation (your main language in Settings → General).

Voxt splits the recording at pauses into cues of at most about seven seconds, the same way it transcribes meetings, and leaves out cues where the model heard nothing. A recording without speech gives an empty result and the message "No speech was found."

## Exit codes

| Code | Meaning |
|---|---|
| 0 | Done. |
| 2 | Wrong arguments. |
| 3 | Voxt is not running. |
| 4 | The transcription stopped: no speech model is set up for Transcribe, the model failed, or the output file cannot be written. stderr says which. |
| 5 | The file does not exist, or macOS cannot read its sound. |

## Privacy

The command reads the file and converts its sound with macOS's `afconvert`. Then it sends the audio to the running Voxt through a socket in `~/Library/Application Support/VoxtNew` that only your account can open (mode 0600, no network port). Voxt reads none of your folders. The audio goes where Transcribe's audio goes: to the on-device speech model, or to the remote speech service you set up for Transcribe. Voxt keeps nothing of it: no history entry and no recording. Its log notes the length of the audio and the number of cues, not the text.
