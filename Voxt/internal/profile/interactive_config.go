package profile

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
)

const defaultInteractiveSegmentSeconds = 25

// InteractiveConfig is one opt-in whole-track versus chunk-boundary priority
// experiment. Both operations use the same local speech-model client.
type InteractiveConfig struct {
	Name                 string      `json:"name,omitempty"`
	Operation            string      `json:"operation"`
	LongAudio            string      `json:"long_audio"`
	InteractiveAudio     string      `json:"interactive_audio"`
	OutputDir            string      `json:"output_dir"`
	Model                ModelConfig `json:"model"`
	Draft                DraftConfig `json:"draft"`
	SegmentLengthSeconds float64     `json:"segment_length_seconds,omitempty"`
	VAD                  VADConfig   `json:"vad"`
	Language             string      `json:"language,omitempty"`
}

// LoadInteractiveConfig rejects unknown fields before a speech model starts.
func LoadInteractiveConfig(path string) (InteractiveConfig, error) {
	file, err := os.Open(path)
	if err != nil {
		return InteractiveConfig{}, err
	}
	defer file.Close()
	decoder := json.NewDecoder(file)
	decoder.DisallowUnknownFields()
	var config InteractiveConfig
	if err := decoder.Decode(&config); err != nil {
		return InteractiveConfig{}, fmt.Errorf("interactive config: %w", err)
	}
	if err := decoder.Decode(new(any)); !errors.Is(err, io.EOF) {
		return InteractiveConfig{}, fmt.Errorf("interactive config: more than one JSON value")
	}
	if err := config.Validate(); err != nil {
		return InteractiveConfig{}, err
	}
	return config, nil
}

func (c *InteractiveConfig) Validate() error {
	if c.Operation != "transcribe" && c.Operation != "diarize" {
		return fmt.Errorf("interactive: operation must be transcribe or diarize")
	}
	for _, path := range []string{c.LongAudio, c.InteractiveAudio} {
		switch strings.ToLower(filepath.Ext(path)) {
		case ".m4a", ".mp4", ".wav":
		default:
			return fmt.Errorf("interactive: long_audio and interactive_audio must be m4a, mp4 or wav files")
		}
	}
	if c.OutputDir == "" || c.Model.ID == "" || c.Model.Dir == "" {
		return fmt.Errorf("interactive: output_dir and model id/dir are required")
	}
	if c.Draft.Enabled {
		if c.Draft.Dir == "" {
			return fmt.Errorf("interactive: an enabled draft needs its dir")
		}
		if c.Draft.Bits == 0 {
			c.Draft.Bits = 8
		}
		if c.Draft.Bits != 4 && c.Draft.Bits != 6 && c.Draft.Bits != 8 {
			return fmt.Errorf("interactive: draft bits must be 4, 6 or 8")
		}
	}
	if c.SegmentLengthSeconds == 0 {
		c.SegmentLengthSeconds = defaultInteractiveSegmentSeconds
	}
	if c.SegmentLengthSeconds < 20 || c.SegmentLengthSeconds > 30 {
		return fmt.Errorf("interactive: segment_length_seconds must be between 20 and 30 seconds")
	}
	if !c.VAD.Enabled || c.VAD.ModelDir == "" {
		return fmt.Errorf("interactive: priority_vad requires vad.enabled=true and vad.model_dir")
	}
	return nil
}
