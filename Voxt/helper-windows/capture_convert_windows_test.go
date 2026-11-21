//go:build windows

package main

import (
	"bufio"
	"bytes"
	"encoding/binary"
	"math"
	"slices"
	"testing"
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/helper/proto"
)

func TestLinearResamplerChunkInvariant(t *testing.T) {
	input := make([]float64, 961)
	for index := range input {
		input[index] = math.Sin(float64(index) / 17)
	}
	for _, rates := range [][2]int{{48000, 16000}, {16000, 48000}} {
		var whole linearResampler
		whole.reset(rates[0], rates[1])
		want := whole.add(input)
		var chunked linearResampler
		chunked.reset(rates[0], rates[1])
		var got []float64
		for position := 0; position < len(input); {
			length := min(1+position%37, len(input)-position)
			got = append(got, chunked.add(input[position:position+length])...)
			position += length
		}
		if len(want) == 0 || !slices.Equal(got, want) {
			t.Fatalf("rates %v: chunking changed samples: got %d want %d", rates, len(got), len(want))
		}
	}
}

func TestCaptureFramesGaplessWithPartialLastFrame(t *testing.T) {
	var wire bytes.Buffer
	server := &helperServer{output: bufio.NewWriter(&wire)}
	session := &captureSession{
		server: server, id: "j1", startedAt: time.Now(), rate: 16000,
		frameSamples: 1600, levelInterval: 50 * time.Millisecond, peak: -120,
	}
	session.startSource(proto.CaptureDevice{ID: "test", Name: "Test", Channels: 2, SampleRate: 16000})
	session.accept(make([]float64, 4000))
	result := session.result(proto.CaptureStopped)
	if result.Frames != 3 || result.Samples != 4000 || result.Reason != proto.CaptureStopped {
		t.Fatalf("capture result %+v", result)
	}
	var sequences, offsets, lengths []int
	scanner := bufio.NewScanner(&wire)
	for scanner.Scan() {
		message, err := proto.Decode(scanner.Bytes())
		if err != nil || message.Method != proto.MethodCapturePCM {
			continue
		}
		var frame proto.PCMParams
		if err := message.DecodeParams(&frame); err != nil {
			t.Fatal(err)
		}
		sequences = append(sequences, frame.Seq)
		offsets = append(offsets, frame.Offset)
		lengths = append(lengths, len(frame.Data)/2)
	}
	if !slices.Equal(sequences, []int{0, 1, 2}) || !slices.Equal(offsets, []int{0, 1600, 3200}) || !slices.Equal(lengths, []int{1600, 1600, 800}) {
		t.Fatalf("frames seq=%v offsets=%v lengths=%v", sequences, offsets, lengths)
	}
}

func TestWaveFormatDownmix(t *testing.T) {
	format := waveFormat{tag: waveFormatPCM, channels: 2, sampleRate: 48000, bitsPerSample: 16, blockAlign: 4}
	data := make([]byte, 8)
	binary.LittleEndian.PutUint16(data[0:], uint16(32767))
	binary.LittleEndian.PutUint16(data[2:], uint16(32768))
	binary.LittleEndian.PutUint16(data[4:], uint16(16384))
	binary.LittleEndian.PutUint16(data[6:], uint16(16384))
	samples, err := format.decodeBytes(data)
	if err != nil {
		t.Fatal(err)
	}
	if len(samples) != 2 || math.Abs(samples[0]) > 0.0001 || math.Abs(samples[1]-0.5) > 0.0001 {
		t.Fatalf("downmixed samples %v", samples)
	}
}
