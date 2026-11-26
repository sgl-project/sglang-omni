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

// OverlapConfig is one three-way scheduling experiment. There is deliberately
// no remote language-model configuration: every run uses SimulatedLLMConfig.
type OverlapConfig struct {
	Name                 string             `json:"name,omitempty"`
	Mode                 string             `json:"mode"`
	Audio                string             `json:"audio"`
	OutputDir            string             `json:"output_dir"`
	Model                ModelConfig        `json:"model"`
	Draft                DraftConfig        `json:"draft"`
	SegmentLengthSeconds float64            `json:"segment_length_seconds,omitempty"`
	VAD                  VADConfig          `json:"vad"`
	Language             string             `json:"language,omitempty"`
	SimulatedLLM         SimulatedLLMConfig `json:"simulated_llm"`
}

type SimulatedLLMConfig struct {
	LatencyMS int `json:"latency_ms"`
}

// LoadOverlapConfig rejects unknown fields before a speech model can start.
func LoadOverlapConfig(path string) (OverlapConfig, error) {
	file, err := os.Open(path)
	if err != nil {
		return OverlapConfig{}, err
	}
	defer file.Close()
	decoder := json.NewDecoder(file)
	decoder.DisallowUnknownFields()
	var config OverlapConfig
	if err := decoder.Decode(&config); err != nil {
		return OverlapConfig{}, fmt.Errorf("overlap config: %w", err)
	}
	if err := decoder.Decode(new(any)); !errors.Is(err, io.EOF) {
		return OverlapConfig{}, fmt.Errorf("overlap config: more than one JSON value")
	}
	if err := config.Validate(); err != nil {
		return OverlapConfig{}, err
	}
	return config, nil
}

func (c *OverlapConfig) Validate() error {
	if c.Mode != "dictation" && c.Mode != "meeting" {
		return fmt.Errorf("overlap: mode must be dictation or meeting")
	}
	switch strings.ToLower(filepath.Ext(c.Audio)) {
	case ".m4a", ".mp4", ".wav":
	default:
		return fmt.Errorf("overlap: audio must be an m4a, mp4 or wav file")
	}
	if c.OutputDir == "" || c.Model.ID == "" || c.Model.Dir == "" {
		return fmt.Errorf("overlap: output_dir and model id/dir are required")
	}
	if c.Draft.Enabled {
		if c.Draft.Dir == "" {
			return fmt.Errorf("overlap: an enabled draft needs its dir")
		}
		if c.Draft.Bits == 0 {
			c.Draft.Bits = 8
		}
		if c.Draft.Bits != 4 && c.Draft.Bits != 6 && c.Draft.Bits != 8 {
			return fmt.Errorf("overlap: draft bits must be 4, 6 or 8")
		}
	}
	if c.SegmentLengthSeconds < 0 || c.SegmentLengthSeconds > 0 && c.SegmentLengthSeconds <= 1 {
		return fmt.Errorf("overlap: segment_length_seconds must be zero or exceed the one-second chunk overlap")
	}
	if c.VAD.Enabled && c.VAD.ModelDir == "" {
		return fmt.Errorf("overlap: enabled VAD needs vad.model_dir")
	}
	if c.SimulatedLLM.LatencyMS < 0 {
		return fmt.Errorf("overlap: simulated_llm.latency_ms cannot be negative")
	}
	return nil
}
