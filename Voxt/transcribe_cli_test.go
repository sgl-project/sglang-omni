package main

import (
	"context"
	"encoding/binary"
	"errors"
	"math"
	"net"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"

	"github.com/sgl-project/sglang-omni/Voxt/internal/asr"
	"github.com/sgl-project/sglang-omni/Voxt/internal/fileasr"
	"github.com/sgl-project/sglang-omni/Voxt/internal/local"
)

// tone is s seconds of a 440 Hz tone (speech to the cutter's level).
func tone(s float64) []byte {
	pcm := make([]byte, int(s*asr.SampleRate)*2)
	for i := range len(pcm) / 2 {
		binary.LittleEndian.PutUint16(pcm[2*i:], uint16(int16(3000*math.Sin(float64(i)*2*math.Pi*440/asr.SampleRate))))
	}
	return pcm
}

// transcribeServer serves rec at a short socket path and returns it.
func transcribeServer(t *testing.T, rec fileasr.Recognize) string {
	t.Helper()
	return transcribeServerOpen(t, func(context.Context, string) (fileasr.Recognize, error) { return rec, nil })
}

func transcribeServerOpen(t *testing.T, open func(context.Context, string) (fileasr.Recognize, error)) string {
	t.Helper()
	dir, err := os.MkdirTemp("", "tcli")
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { os.RemoveAll(dir) })
	path := filepath.Join(dir, fileasr.SocketName)
	ln, err := net.Listen("unix", path)
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	t.Cleanup(cancel)
	srv := &fileasr.Server{Open: open}
	go srv.Serve(ctx, ln)
	return path
}

func runTranscribe(socket string, decode func(string) ([]byte, error), args ...string) (code int, stdout, stderr string) {
	var o, e strings.Builder
	code = transcribeCLI(args, &o, &e, socket, decode)
	return code, o.String(), e.String()
}

func TestTranscribeCLIWritesSRTAndText(t *testing.T) {
	say := func(context.Context, []byte) (string, error) { return "你好 Voxt", nil }
	sock := transcribeServer(t, say)
	decode := func(string) ([]byte, error) { return tone(1.5), nil }
	out := filepath.Join(t.TempDir(), "talk.srt")
	code, _, stderr := runTranscribe(sock, decode, "-o", out, "talk.mp4")
	got, _ := os.ReadFile(out)
	if code != exitOK || string(got) != "1\n00:00:00,000 --> 00:00:01,500\n你好 Voxt\n" {
		t.Fatalf("code %d, srt %q, stderr %q", code, got, stderr)
	}
	os.Remove(out)
	code, _, stderr = runTranscribe(sock, decode, "talk.mp4", "--format", "srt", "-o", out) // the issue's order
	if got, _ := os.ReadFile(out); code != exitOK || !strings.HasPrefix(string(got), "1\n00:00:00,000 --> ") {
		t.Fatalf("options after FILE: code %d, srt %q, stderr %q", code, got, stderr)
	}
	code, stdout, _ := runTranscribe(sock, decode, "--format", "txt", "talk.mp4")
	if code != exitOK || stdout != "你好 Voxt\n" {
		t.Fatalf("code %d, text %q", code, stdout)
	}
}

func TestTranscribeCLIFailures(t *testing.T) {
	good := func(string) ([]byte, error) { return tone(1), nil }
	sock := transcribeServer(t, func(context.Context, []byte) (string, error) { return "", errors.New("the model stopped") })
	silent := transcribeServer(t, func(context.Context, []byte) (string, error) { return "never", nil })
	out := filepath.Join(t.TempDir(), "out.srt")
	for _, c := range []struct {
		name   string
		socket string
		decode func(string) ([]byte, error)
		args   []string
		code   int
		says   string
	}{
		{"no file", sock, good, nil, exitUsage, "usage:"},
		{"two files", sock, good, []string{"a.mp4", "b.mp4"}, exitUsage, "usage:"},
		{"two files around an option", sock, good, []string{"a.mp4", "-o", out, "b.mp4"}, exitUsage, "usage:"},
		{"bad format", sock, good, []string{"--format", "vtt", "a.mp4"}, exitUsage, `unknown format "vtt"`},
		{"unreadable", sock, func(string) ([]byte, error) { return nil, errors.New("no sound") }, []string{"-o", out, "a.mkv"}, exitUnreadable, "Cannot read the sound of a.mkv: no sound"},
		{"not running", filepath.Join(t.TempDir(), "none.sock"), good, []string{"-o", out, "a.mp4"}, exitUnavailable, "Voxt is not running"},
		{"model fails", sock, good, []string{"-o", out, "a.mp4"}, exitInterrupted, "Transcription stopped: the model stopped"},
	} {
		code, _, stderr := runTranscribe(c.socket, c.decode, c.args...)
		if code != c.code || !strings.Contains(stderr, c.says) {
			t.Errorf("%s: code %d, stderr %q", c.name, code, stderr)
		}
		if _, err := os.Stat(out); err == nil {
			t.Fatalf("%s: a failed run wrote %s", c.name, out)
		}
	}
	// Silence: no speech, an empty file, success.
	code, _, stderr := runTranscribe(silent, func(string) ([]byte, error) { return make([]byte, 32000), nil }, "-o", out, "quiet.m4a")
	got, err := os.ReadFile(out)
	if code != exitOK || err != nil || len(got) != 0 || !strings.Contains(stderr, "No speech was found.") {
		t.Fatalf("silence: code %d, file %q %v, stderr %q", code, got, err, stderr)
	}
}

func TestDecodeAudioReadsWAVAndRefusesText(t *testing.T) {
	if runtime.GOOS != "darwin" {
		t.Skip("afconvert is macOS's")
	}
	dir := t.TempDir()
	in := filepath.Join(dir, "a.wav")
	os.WriteFile(in, asr.WAV(tone(1), 48000), 0o644) // 1/3 s at 48 kHz
	pcm, err := decodeAudio(in)
	if err != nil || len(pcm) < 10000 || len(pcm) > 11500 {
		t.Fatalf("pcm %d bytes, err %v", len(pcm), err)
	}
	txt := filepath.Join(dir, "notes.txt")
	os.WriteFile(txt, []byte("hello"), 0o644)
	if _, err := decodeAudio(txt); err == nil || !strings.Contains(err.Error(), "not an audio or video file") {
		t.Fatalf("text file: %v", err)
	}
	if _, err := decodeAudio(filepath.Join(dir, "missing.mp4")); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("missing file: %v", err)
	}
}

// TestTranscribeUsesTranscribesSpeechModel: the app's server recognises
// with Transcribe's speech model; without one, the command says so.
func TestTranscribeUsesTranscribesSpeechModel(t *testing.T) {
	a, _, tr := retryApp(t, "Hello from the file")
	sock := transcribeServerOpen(t, a.fileRecognizer)
	code, stdout, stderr := runTranscribe(sock, func(string) ([]byte, error) { return tone(1), nil }, "--format", "txt", "a.m4a")
	if code != exitOK || stdout != "Hello from the file\n" || tr.count() != 1 {
		t.Fatalf("code %d, stdout %q, stderr %q, calls %d", code, stdout, stderr, tr.count())
	}

	old := newLocalServices
	newLocalServices = func(*app) local.Services { return &fakeLocal{} }
	t.Cleanup(func() { newLocalServices = old })
	b, _, _ := newTestApp(t)
	sock = transcribeServerOpen(t, b.fileRecognizer)
	code, _, stderr = runTranscribe(sock, func(string) ([]byte, error) { return tone(1), nil }, "a.m4a")
	if code != exitInterrupted || !strings.Contains(stderr, "no speech model is installed for Transcribe") {
		t.Fatalf("no model: code %d, stderr %q", code, stderr)
	}
}
