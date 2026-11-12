package fileasr

import (
	"bytes"
	"context"
	"encoding/binary"
	"errors"
	"fmt"
	"math"
	"net"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

// speech is PCM of tone bursts (heard as speech by the cutter's level)
// of on, each followed by off of silence, n times.
func speech(n int, on, off time.Duration) []byte {
	var b bytes.Buffer
	for range n {
		for i := range int(on.Seconds() * 16000) {
			binary.Write(&b, binary.LittleEndian, int16(3000*math.Sin(float64(i)*2*math.Pi*440/16000)))
		}
		b.Write(make([]byte, int(off.Seconds()*16000)*2))
	}
	return b.Bytes()
}

// numbered recognises each utterance as "line N".
func numbered() Recognize {
	n := 0
	return func(context.Context, []byte) (string, error) {
		n++
		return fmt.Sprintf(" line %d ", n), nil
	}
}

func TestRunCutsAtPauses(t *testing.T) {
	var got []Segment
	err := Run(context.Background(), bytes.NewReader(speech(2, time.Second, time.Second)), numbered(), func(s Segment) error {
		got = append(got, s)
		return nil
	})
	if err != nil || len(got) != 2 {
		t.Fatalf("segments %+v, err %v", got, err)
	}
	if got[0].Text != "line 1" || got[0].Start > 0 || got[0].End < time.Second || got[1].Start < 1700*time.Millisecond || got[1].End > 4*time.Second {
		t.Fatalf("segments %+v", got)
	}
}

func TestRunSilenceAndEmptyTextGiveNothing(t *testing.T) {
	for name, pcm := range map[string][]byte{"empty": nil, "silent": make([]byte, 64000), "odd": {1}} {
		err := Run(context.Background(), bytes.NewReader(pcm), numbered(), func(s Segment) error {
			t.Fatalf("%s: segment %+v", name, s)
			return nil
		})
		if err != nil {
			t.Fatalf("%s: %v", name, err)
		}
	}
	blank := func(context.Context, []byte) (string, error) { return "  ", nil }
	if err := Run(context.Background(), bytes.NewReader(speech(1, time.Second, 0)), blank, func(s Segment) error {
		t.Fatalf("blank text gave %+v", s)
		return nil
	}); err != nil {
		t.Fatal(err)
	}
}

func TestRunStopsAtTheModelsError(t *testing.T) {
	boom := errors.New("boom")
	fail := func(context.Context, []byte) (string, error) { return "", boom }
	if err := Run(context.Background(), bytes.NewReader(speech(2, time.Second, time.Second)), fail, func(Segment) error { return nil }); !errors.Is(err, boom) {
		t.Fatalf("err %v", err)
	}
}

func TestWriteSRTAndText(t *testing.T) {
	segs := []Segment{
		{Start: 0, End: 1500 * time.Millisecond, Text: "Hello 世界"},
		{Start: time.Hour + 2*time.Minute + 3*time.Second + 4*time.Millisecond, End: 26 * time.Hour, Text: " 用 Voxt 转录 "},
	}
	var b strings.Builder
	if err := Write(&b, segs, FormatSRT); err != nil {
		t.Fatal(err)
	}
	want := "1\n00:00:00,000 --> 00:00:01,500\nHello 世界\n\n2\n01:02:03,004 --> 26:00:00,000\n用 Voxt 转录\n"
	if b.String() != want {
		t.Fatalf("srt %q", b.String())
	}
	b.Reset()
	Write(&b, segs, FormatText)
	if b.String() != "Hello 世界\n用 Voxt 转录\n" {
		t.Fatalf("text %q", b.String())
	}
	b.Reset()
	Write(&b, nil, FormatSRT)
	if b.String() != "" {
		t.Fatalf("empty %q", b.String())
	}
}

// serve starts a server at a short socket path (a Unix socket's path is
// at most 104 bytes on macOS) and returns the path.
func serve(t *testing.T, s *Server) string {
	t.Helper()
	dir, err := os.MkdirTemp("", "fa")
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { os.RemoveAll(dir) })
	path := filepath.Join(dir, SocketName)
	ln, err := net.Listen("unix", path)
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	t.Cleanup(cancel)
	go s.Serve(ctx, ln)
	return path
}

func TestTranscribeOverTheSocket(t *testing.T) {
	var gotLang string
	var done []string
	path := serve(t, &Server{
		Open: func(_ context.Context, lang string) (Recognize, error) {
			gotLang = lang
			return numbered(), nil
		},
		Done: func(n int, audio time.Duration, err error) { done = append(done, fmt.Sprint(n, audio, err)) },
	})
	// 300 utterances, 10 MB: more replies than a socket buffers while the
	// command is still sending.
	var got []Segment
	err := Transcribe(path, "zh", speech(300, 500*time.Millisecond, 500*time.Millisecond), func(s Segment) error {
		got = append(got, s)
		return nil
	})
	if err != nil || len(got) != 300 || got[299].Text != "line 300" || gotLang != "zh" {
		t.Fatalf("got %d segments, err %v, lang %q", len(got), err, gotLang)
	}
	if err := Transcribe(path, "", nil, func(s Segment) error { t.Fatalf("segment %+v", s); return nil }); err != nil {
		t.Fatal(err)
	}
	if len(done) != 2 || done[0] != "300 5m0s <nil>" {
		t.Fatalf("done %q", done)
	}
}

func TestTranscribeSaysWhyItFailed(t *testing.T) {
	path := serve(t, &Server{Open: func(context.Context, string) (Recognize, error) {
		return nil, errors.New("no speech model is set up for Transcribe")
	}})
	err := Transcribe(path, "", speech(1, time.Second, 0), func(Segment) error { return nil })
	if err == nil || err.Error() != "no speech model is set up for Transcribe" {
		t.Fatalf("err %v", err)
	}
	if err := Transcribe(filepath.Join(t.TempDir(), "none.sock"), "", nil, nil); !errors.Is(err, ErrUnavailable) {
		t.Fatalf("no socket: %v", err)
	}
}

func TestServerRefusesBadOrShortUploads(t *testing.T) {
	path := serve(t, &Server{Open: func(context.Context, string) (Recognize, error) { return numbered(), nil }})
	for req, want := range map[string]string{
		`{"op":"export","bytes":2}`:       "bad request",
		`{"op":"transcribe","bytes":3}`:   "bad request",
		`{"op":"transcribe","bytes":-2}`:  "bad request",
		`{"op":"transcribe","bytes":100}`: "the audio ended early",
	} {
		conn, err := net.Dial("unix", path)
		if err != nil {
			t.Fatal(err)
		}
		conn.Write([]byte(req + "\n" + "0123456789"))
		conn.(*net.UnixConn).CloseWrite()
		var b bytes.Buffer
		b.ReadFrom(conn)
		conn.Close()
		if !strings.Contains(b.String(), want) {
			t.Fatalf("%s: reply %q", req, b.String())
		}
	}
}
