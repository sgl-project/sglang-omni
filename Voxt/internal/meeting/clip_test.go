package meeting

import (
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/asr"
)

func TestClipWAVHeaderAndLength(t *testing.T) {
	dir := t.TempDir()
	src := filepath.Join(dir, "full.wav")
	pcm := make([]byte, asr.BytesPerSecond*2) // 2 s
	for i := range pcm {
		pcm[i] = byte(i % 200)
	}
	if err := os.WriteFile(src, asr.WAV(pcm, asr.SampleRate), 0o644); err != nil {
		t.Fatal(err)
	}
	dst := filepath.Join(dir, "clip.wav")
	start := 250 * time.Millisecond
	end := 750 * time.Millisecond
	if err := ClipWAV(src, dst, start, end); err != nil {
		t.Fatal(err)
	}
	raw, err := os.ReadFile(dst)
	if err != nil {
		t.Fatal(err)
	}
	out, rate, ch, err := asr.ReadWAV(raw)
	if err != nil {
		t.Fatal(err)
	}
	if rate != asr.SampleRate || ch != 1 {
		t.Fatalf("got rate %d ch %d", rate, ch)
	}
	wantDur := end - start
	if d := asr.Duration(out, rate); d != wantDur {
		t.Fatalf("duration %v want %v", d, wantDur)
	}
	// sample-aligned start: 250 ms = 4000 samples = byte 8000 in source
	wantStart := int(start.Seconds() * float64(asr.SampleRate) * 2)
	snip := pcm[wantStart : wantStart+len(out)]
	if len(snip) != len(out) {
		t.Fatalf("clip len %d want %d", len(out), len(snip))
	}
	for i := range out {
		if out[i] != snip[i] {
			t.Fatalf("pcm mismatch at %d", i)
		}
	}
}

func TestClipWAVEndBeforeStart(t *testing.T) {
	dir := t.TempDir()
	src := filepath.Join(dir, "x.wav")
	if err := os.WriteFile(src, asr.WAV([]byte{0, 1, 0, 1}, asr.SampleRate), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := ClipWAV(src, filepath.Join(dir, "y.wav"), time.Second, 0); err == nil {
		t.Fatal("expected error")
	}
}
