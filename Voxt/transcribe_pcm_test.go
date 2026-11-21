package main

import (
	"encoding/binary"
	"testing"

	"github.com/sgl-project/sglang-omni/Voxt/internal/asr"
)

func samples(vals ...int16) []byte {
	out := make([]byte, len(vals)*2)
	for i, v := range vals {
		binary.LittleEndian.PutUint16(out[i*2:], uint16(v))
	}
	return out
}

func TestPcm16Mono(t *testing.T) {
	// Stereo: the channels average.
	got := pcm16Mono(asr.SampleRate, samples(100, 300, -100, 100), asr.SampleRate, 2)
	if len(got) != 4 {
		t.Fatalf("%d bytes", len(got))
	}
	if v := int16(binary.LittleEndian.Uint16(got)); v != 200 {
		t.Fatalf("first frame %d", v)
	}
	if v := int16(binary.LittleEndian.Uint16(got[2:])); v != 0 {
		t.Fatalf("second frame %d", v)
	}
	// Rate conversion: 32 kHz halves to every other frame-ish.
	got = pcm16Mono(asr.SampleRate, samples(0, 1000, 2000, 3000, 4000, 5000, 6000, 7000), 32000, 1)
	if len(got) != 8 {
		t.Fatalf("%d bytes", len(got))
	}
	if v := int16(binary.LittleEndian.Uint16(got[2:])); v != 2000 {
		t.Fatalf("resampled frame %d", v)
	}
	// Edge cases: empty and one frame.
	if got := pcm16Mono(asr.SampleRate, nil, 48000, 1); len(got) != 0 {
		t.Fatalf("empty: %d", len(got))
	}
	if got := pcm16Mono(asr.SampleRate, samples(42), 48000, 1); len(got) != 2 || int16(binary.LittleEndian.Uint16(got)) != 42 {
		t.Fatalf("one frame: %v", got)
	}
}
