// Package fileasr transcribes a recording given as a file, for
// `Voxt transcribe` (docs/transcribe.md): the command decodes the file
// and sends its PCM over a socket to the running app, which cuts it into
// utterances the way a meeting's file-mode recognition does
// (meeting.Cutter) and recognises each with Transcribe's speech model.
// The segments come back with their times and the command writes them
// as SRT or plain text.
//
// The socket's wire format is internal to Voxt: one JSON request line,
// the PCM (16-bit little-endian, mono, asr.SampleRate), then the reply's
// JSON lines (a segment each, then the end line or an error line).
//
// Imports: asr, meeting.
package fileasr

import (
	"context"
	"fmt"
	"io"
	"strings"
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/meeting"
)

// Segment is one recognised utterance and its place in the recording.
type Segment struct {
	Start time.Duration `json:"start"`
	End   time.Duration `json:"end"`
	Text  string        `json:"text"`
}

// Recognize turns one utterance's PCM into text.
type Recognize func(ctx context.Context, pcm []byte) (string, error)

// readSize is how much PCM Run reads at a time: 2 s.
const readSize = 64000

// Run reads PCM from r to its end, cuts it into utterances and gives
// each one's text to each, in order; utterances without text are left
// out. It stops at the first error of r, rec or each.
func Run(ctx context.Context, r io.Reader, rec Recognize, each func(Segment) error) error {
	var c meeting.Cutter
	send := func(us []meeting.Utterance) error {
		for _, u := range us {
			text, err := rec(ctx, u.PCM)
			if err != nil {
				return err
			}
			if text = strings.TrimSpace(text); text == "" {
				continue
			}
			if err := each(Segment{Start: u.Start, End: u.End, Text: text}); err != nil {
				return err
			}
		}
		return nil
	}
	buf := make([]byte, readSize)
	for {
		n, err := r.Read(buf)
		if n > 0 {
			if err := send(c.Push(buf[:n])); err != nil {
				return err
			}
		}
		if err == io.EOF {
			return send(c.Flush())
		}
		if err != nil {
			return err
		}
	}
}

// Formats of the output.
const (
	FormatSRT  = "srt"
	FormatText = "txt"
)

// Write writes segs in format: SRT (numbered cues with their times) or
// plain text (one line per segment).
func Write(w io.Writer, segs []Segment, format string) error {
	var b strings.Builder
	for i, s := range segs {
		text := strings.TrimSpace(s.Text)
		if format == FormatText {
			fmt.Fprintln(&b, text)
			continue
		}
		if i > 0 {
			b.WriteByte('\n')
		}
		fmt.Fprintf(&b, "%d\n%s --> %s\n%s\n", i+1, srtTime(s.Start), srtTime(s.End), text)
	}
	_, err := io.WriteString(w, b.String())
	return err
}

// srtTime is d as SRT writes it: HH:MM:SS,mmm.
func srtTime(d time.Duration) string {
	ms := max(d.Milliseconds(), 0)
	return fmt.Sprintf("%02d:%02d:%02d,%03d", ms/3_600_000, ms/60_000%60, ms/1000%60, ms%1000)
}
