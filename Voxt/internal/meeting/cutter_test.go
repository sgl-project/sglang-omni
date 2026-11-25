package meeting

import (
	"encoding/binary"
	"math"
	"math/rand/v2"
	"os"
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

// cutterNoise is white noise at db dBFS, the room a file-only speech model
// hears between words.
func cutterNoise(ms int, db float64, seed uint64) []byte {
	rng := rand.New(rand.NewPCG(seed, 1))
	n := cutterRate * ms / 1000
	out := make([]byte, n*2)
	a := math.Pow(10, db/20) * 32767
	for i := range n {
		v := int16(max(-32768, min(32767, rng.NormFloat64()*a)))
		binary.LittleEndian.PutUint16(out[i*2:], uint16(v))
	}
	return out
}

// A steady noise louder than the cutter's level gate sends nothing to the
// model once the meeting's detector takes it for the floor (11-25: Whisper
// wrote "Thank you" and the like for a room's noise).
func TestCutterDropsSteadyNoise(t *testing.T) {
	var c Cutter
	var out []Utterance
	for i := range 30 {
		out = append(out, c.Push(cutterNoise(1000, -28, uint64(i)))...)
	}
	out = append(out, c.Flush()...)
	late := 0
	for _, u := range out {
		if u.Start > 3*time.Second {
			late++
		}
	}
	t.Logf("30 s of -28 dBFS noise: %d utterances, %d after the first 3 s", len(out), late)
	if late != 0 || len(out) > 1 {
		t.Fatalf("noise sent to the model: %d utterances, %d after 3 s", len(out), late)
	}
}

// Speech over that noise is still cut and sent, as before.
func TestCutterKeepsSpeechOverNoise(t *testing.T) {
	raw, err := os.ReadFile("../../testdata/speech/sample-zh-en.wav")
	if err != nil {
		t.Fatal(err)
	}
	speech := raw[44:]
	noise := cutterNoise(30000, -35, 9)
	mixed := append([]byte(nil), noise...)
	at := 10 * cutterRate * 2 // speech from 10 s
	for i := 0; i+1 < len(speech) && at+i+1 < len(mixed); i += 2 {
		v := int32(int16(binary.LittleEndian.Uint16(mixed[at+i:]))) + int32(int16(binary.LittleEndian.Uint16(speech[i:])))
		binary.LittleEndian.PutUint16(mixed[at+i:], uint16(int16(max(-32768, min(32767, v)))))
	}
	var c Cutter
	out := c.Push(mixed)
	out = append(out, c.Flush()...)
	var sent time.Duration
	for _, u := range out {
		sent += u.End - u.Start
	}
	t.Logf("10.5 s of speech over -35 dBFS noise: %d utterances, %v sent", len(out), sent)
	if len(out) == 0 || sent < 8*time.Second {
		t.Fatalf("speech over noise not sent: %d utterances, %v", len(out), sent)
	}
}
