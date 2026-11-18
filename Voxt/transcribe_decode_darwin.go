//go:build darwin

package main

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"

	"github.com/sgl-project/sglang-omni/Voxt/internal/asr"
)

// decodeAudio is the file's sound as Voxt's PCM (16-bit, mono, 16 kHz),
// by macOS's afconvert, which reads the sound of mp4 and mov videos too.
func decodeAudio(path string) ([]byte, error) {
	if _, err := os.Stat(path); err != nil {
		return nil, err
	}
	path, err := filepath.Abs(path) // a name starting with "-" is not an option to afconvert
	if err != nil {
		return nil, err
	}
	dir, err := os.MkdirTemp("", "voxt-transcribe-")
	if err != nil {
		return nil, err
	}
	defer os.RemoveAll(dir)
	wav := filepath.Join(dir, "audio.wav")
	msg, err := exec.Command("/usr/bin/afconvert", "-f", "WAVE", "-d", fmt.Sprintf("LEI16@%d", asr.SampleRate), "-c", "1", path, wav).CombinedOutput()
	if err != nil {
		if m := strings.TrimSpace(string(msg)); m != "" {
			return nil, fmt.Errorf("not an audio or video file macOS can read (%s)", m)
		}
		return nil, err
	}
	data, err := os.ReadFile(wav)
	if err != nil {
		return nil, err
	}
	pcm, _, _, err := asr.ReadWAV(data)
	return pcm, err
}
