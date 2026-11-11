package main

// `Voxt transcribe FILE` (docs/transcribe.md): an audio file, or the
// sound of a video, to SRT or text with Transcribe's speech model. The
// command decodes the file itself (macOS's afconvert), so the app needs
// no access to the user's folders, and sends the audio to the running app
// over internal/fileasr's socket. Its messages are English, as the app's
// other developer commands.

import (
	"bytes"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/asr"
	"github.com/sgl-project/sglang-omni/Voxt/internal/fileasr"
)

// exitUnreadable: the file has no sound macOS can read (the other codes
// are history_cli.go's).
const exitUnreadable = 5

const transcribeUsage = `usage:
  Voxt transcribe [--format srt|txt] [-o OUT] [--lang CODE] FILE
FILE is an audio file or a video (m4a, mp3, wav, aiff, caf, mp4, mov…);
OUT "-" or none writes to standard output.`

func runTranscribeCLI(args []string) int {
	return transcribeCLI(args, os.Stdout, os.Stderr, filepath.Join(filepath.Dir(historySocketPath()), fileasr.SocketName), decodeAudio)
}

func transcribeCLI(args []string, stdout, stderr io.Writer, socket string, decode func(string) ([]byte, error)) int {
	fs := flag.NewFlagSet("transcribe", flag.ContinueOnError)
	fs.SetOutput(io.Discard)
	format := fs.String("format", fileasr.FormatSRT, "")
	out := fs.String("o", "-", "")
	lang := fs.String("lang", "", "")
	// Options may come after FILE too: flag stops at the first argument.
	var files []string
	for {
		if err := fs.Parse(args); err != nil {
			fmt.Fprintln(stderr, transcribeUsage)
			return exitUsage
		}
		if fs.NArg() == 0 {
			break
		}
		files, args = append(files, fs.Arg(0)), fs.Args()[1:]
	}
	if len(files) != 1 {
		fmt.Fprintln(stderr, transcribeUsage)
		return exitUsage
	}
	if *format != fileasr.FormatSRT && *format != fileasr.FormatText {
		fmt.Fprintf(stderr, "unknown format %q\n%s\n", *format, transcribeUsage)
		return exitUsage
	}
	in := files[0]
	pcm, err := decode(in)
	if err != nil {
		fmt.Fprintf(stderr, "Cannot read the sound of %s: %v\n", in, err)
		return exitUnreadable
	}
	total := time.Duration(len(pcm)/2) * time.Second / asr.SampleRate
	var segs []fileasr.Segment
	err = fileasr.Transcribe(socket, *lang, pcm, func(s fileasr.Segment) error {
		segs = append(segs, s)
		fmt.Fprintf(stderr, "\r%s of %s", hms(s.End), hms(total))
		return nil
	})
	if len(segs) > 0 {
		fmt.Fprintln(stderr)
	}
	switch {
	case errors.Is(err, fileasr.ErrUnavailable):
		fmt.Fprintln(stderr, "Voxt is not running. Open Voxt, then run the command again.")
		return exitUnavailable
	case err != nil:
		fmt.Fprintf(stderr, "Transcription stopped: %v\n", err)
		return exitInterrupted
	case len(segs) == 0:
		fmt.Fprintln(stderr, "No speech was found.")
	}
	// Written only now: a failed run leaves no half file behind.
	var b bytes.Buffer
	fileasr.Write(&b, segs, *format)
	if *out == "-" {
		stdout.Write(b.Bytes())
		return exitOK
	}
	if err := os.WriteFile(*out, b.Bytes(), 0o644); err != nil {
		fmt.Fprintf(stderr, "Cannot write %s: %v\n", *out, err)
		return exitInterrupted
	}
	return exitOK
}

// hms is d as H:MM:SS.
func hms(d time.Duration) string {
	s := int(d.Seconds())
	return fmt.Sprintf("%d:%02d:%02d", s/3600, s/60%60, s%60)
}

// decodeAudio is the file's sound as Voxt's PCM (16-bit, mono, 16 kHz),
// by macOS's afconvert, which reads the sound of mp4 and mov videos too.
func decodeAudio(path string) ([]byte, error) {
	if _, err := os.Stat(path); err != nil {
		return nil, err
	}
	dir, err := os.MkdirTemp("", "voxt-transcribe-")
	if err != nil {
		return nil, err
	}
	defer os.RemoveAll(dir)
	wav := filepath.Join(dir, "audio.wav")
	msg, err := exec.Command("/usr/bin/afconvert", "-f", "WAVE", "-d", fmt.Sprintf("LEI16@%d", asr.SampleRate), "-c", "1", path, wav).CombinedOutput()
	if err != nil {
		if m := strings.TrimSpace(string(msg)); m != "" {
			return nil, fmt.Errorf("not an audio or video file macOS can read (%s)", m)
		}
		return nil, err
	}
	data, err := os.ReadFile(wav)
	if err != nil {
		return nil, err
	}
	pcm, _, _, err := asr.ReadWAV(data)
	return pcm, err
}
