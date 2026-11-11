package meeting

import (
	"encoding/binary"
	"testing"
	"time"
)

func synthPCM(ms int, loud bool) []byte {
	n := cutterRate * ms / 1000
	out := make([]byte, n*2)
	amp := int16(0)
	if loud {
		amp = 12000
	}
	for i := 0; i < n; i++ {
		binary.LittleEndian.PutUint16(out[i*2:], uint16(amp))
	}
	return out
}

func concat(parts ...[]byte) []byte {
	var out []byte
	for _, p := range parts {
		out = append(out, p...)
	}
	return out
}

func TestCutterSilenceCut(t *testing.T) {
	var c Cutter
	out := c.Push(concat(synthPCM(1200, true), synthPCM(500, false)))
	if len(out) != 1 {
		t.Fatalf("want 1 utterance, got %d", len(out))
	}
	if out[0].End-out[0].Start < cutterMinUtterance {
		t.Fatalf("too short %v", out[0].End-out[0].Start)
	}
	rest := c.Push(synthPCM(800, true))
	rest = append(rest, c.Push(synthPCM(500, false))...)
	if len(rest) != 1 {
		t.Fatalf("want second utterance, got %d", len(rest))
	}
}

func TestCutterDiscardsShortCough(t *testing.T) {
	var c Cutter
	out := c.Push(concat(synthPCM(300, true), synthPCM(500, false)))
	if len(out) != 0 {
		t.Fatalf("short cough should be discarded, got %d", len(out))
	}
}

func TestCutterMaxSegmentQuietCut(t *testing.T) {
	var c Cutter
	// 8 s loud with a 100 ms dip in the last second.
	loud := synthPCM(7900, true)
	quiet := synthPCM(100, false)
	louder := synthPCM(1000, true)
	out := c.Push(concat(loud, quiet, louder))
	var forced Utterance
	for _, u := range out {
		if u.End-u.Start >= cutterMaxSegment-time.Second {
			forced = u
		}
	}
	if forced.End == 0 {
		t.Fatal("expected forced cut before 8 s")
	}
	if forced.End-forced.Start > cutterMaxSegment+cutterFrame {
		t.Fatalf("forced utterance too long: %v", forced.End-forced.Start)
	}
}

func TestCutterFlush(t *testing.T) {
	var c Cutter
	_ = c.Push(synthPCM(600, true))
	fl := c.Flush()
	if len(fl) != 1 {
		t.Fatalf("flush want 1, got %d", len(fl))
	}
	if fl[0].End-fl[0].Start < 600*time.Millisecond {
		t.Fatalf("flush duration %v", fl[0].End-fl[0].Start)
	}
}
