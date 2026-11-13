//go:build windows

package main

import (
	"encoding/binary"
	"fmt"
	"math"
	"os"
)

type linearResampler struct {
	sourceRate        int
	targetRate        int
	pending           []float64
	positionNumerator int64
}

func (resampler *linearResampler) reset(sourceRate, targetRate int) {
	resampler.sourceRate = sourceRate
	resampler.targetRate = targetRate
	resampler.pending = nil
	resampler.positionNumerator = 0
}

func (resampler *linearResampler) add(input []float64) []float64 {
	if len(input) == 0 {
		return nil
	}
	if resampler.sourceRate == resampler.targetRate {
		return input
	}
	resampler.pending = append(resampler.pending, input...)
	targetRate := int64(resampler.targetRate)
	output := make([]float64, 0, int(float64(len(input))*float64(resampler.targetRate)/float64(resampler.sourceRate))+1)
	for resampler.positionNumerator/targetRate+1 < int64(len(resampler.pending)) {
		index := int(resampler.positionNumerator / targetRate)
		fraction := float64(resampler.positionNumerator%targetRate) / float64(targetRate)
		first := resampler.pending[index]
		second := resampler.pending[index+1]
		output = append(output, first+(second-first)*fraction)
		resampler.positionNumerator += int64(resampler.sourceRate)
	}
	drop := min(int(resampler.positionNumerator/targetRate), len(resampler.pending))
	copy(resampler.pending, resampler.pending[drop:])
	resampler.pending = resampler.pending[:len(resampler.pending)-drop]
	resampler.positionNumerator -= int64(drop) * targetRate
	return output
}

func pcm16Bytes(samples []float64) []byte {
	data := make([]byte, 2*len(samples))
	for index, sample := range samples {
		scaled := math.Round(sample * 32768)
		scaled = min(max(scaled, -32768), 32767)
		binary.LittleEndian.PutUint16(data[2*index:], uint16(int16(scaled)))
	}
	return data
}

func audioLevels(samples []float64) (rms, peak float64) {
	if len(samples) == 0 {
		return -120, -120
	}
	var squareSum, maximum float64
	for _, sample := range samples {
		absolute := math.Abs(sample)
		squareSum += sample * sample
		maximum = max(maximum, absolute)
	}
	return decibels(math.Sqrt(squareSum / float64(len(samples)))), decibels(maximum)
}

func decibels(amplitude float64) float64 {
	if amplitude <= 0.000001 {
		return -120
	}
	return math.Max(-120, math.Round(200*math.Log10(amplitude))/10)
}

func readCaptureFile(path string, rawSampleRate int) ([]float64, waveFormat, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, waveFormat{}, err
	}
	if len(data) < 12 || string(data[:4]) != "RIFF" || string(data[8:12]) != "WAVE" {
		format := waveFormat{tag: waveFormatPCM, channels: 1, sampleRate: rawSampleRate, bitsPerSample: 16, blockAlign: 2}
		samples, err := format.decodeBytes(data)
		return samples, format, err
	}
	var format waveFormat
	var audioData []byte
	formatFound := false
	for offset := 12; offset+8 <= len(data); {
		chunkName := string(data[offset : offset+4])
		chunkSize := int(binary.LittleEndian.Uint32(data[offset+4:]))
		bodyStart := offset + 8
		bodyEnd := min(bodyStart+chunkSize, len(data))
		body := data[bodyStart:bodyEnd]
		switch chunkName {
		case "fmt ":
			format, err = parseFileWaveFormat(body)
			formatFound = err == nil
		case "data":
			audioData = body
		}
		offset = bodyStart + chunkSize + chunkSize%2
	}
	if !formatFound || len(audioData) == 0 {
		if err != nil {
			return nil, waveFormat{}, err
		}
		return nil, waveFormat{}, fmt.Errorf("%s: no readable WAV format or audio", path)
	}
	samples, err := format.decodeBytes(audioData)
	return samples, format, err
}

func parseFileWaveFormat(data []byte) (waveFormat, error) {
	if len(data) < 16 {
		return waveFormat{}, fmt.Errorf("short WAV format")
	}
	format := waveFormat{
		tag:           binary.LittleEndian.Uint16(data),
		channels:      int(binary.LittleEndian.Uint16(data[2:])),
		sampleRate:    int(binary.LittleEndian.Uint32(data[4:])),
		blockAlign:    int(binary.LittleEndian.Uint16(data[12:])),
		bitsPerSample: int(binary.LittleEndian.Uint16(data[14:])),
	}
	if format.tag == waveFormatExtensible {
		if len(data) < 40 {
			return waveFormat{}, fmt.Errorf("short extensible WAV format")
		}
		format.tag = uint16(binary.LittleEndian.Uint32(data[24:]))
	}
	if format.channels <= 0 || format.sampleRate <= 0 || format.blockAlign <= 0 {
		return waveFormat{}, fmt.Errorf("invalid WAV format")
	}
	return format, nil
}

func (format waveFormat) decodeBytes(data []byte) ([]float64, error) {
	if len(data) == 0 {
		return nil, nil
	}
	frameCount := len(data) / format.blockAlign
	return format.decode(&data[0], frameCount)
}
