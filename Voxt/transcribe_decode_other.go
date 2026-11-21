//go:build !darwin

package main

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/sgl-project/sglang-omni/Voxt/internal/asr"
)

// decodeAudio is the file's sound as Voxt's PCM (16-bit, mono, 16 kHz).
// Without macOS's afconvert only WAV files are read; compressed audio
// and video files need an external decoder first.
func decodeAudio(path string) ([]byte, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	pcm, rate, channels, err := asr.ReadWAV(data)
	if err != nil {
		ext := strings.TrimPrefix(filepath.Ext(path), ".")
		if ext == "" {
			ext = "this"
		}
		return nil, fmt.Errorf("Voxt reads WAV files on this platform; convert %s to WAV first (%v)", ext, err)
	}
	return pcm16Mono(asr.SampleRate, pcm, rate, channels), nil
}
