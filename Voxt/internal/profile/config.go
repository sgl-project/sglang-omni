// Package profile contains the deterministic reporting and instrumentation
// used by the opt-in long-recording profiling test.
package profile

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/url"
	"os"
	"path/filepath"
	"regexp"
	"strings"
)

// Config is one profiling run. API keys are deliberately absent: KeyEnv
// names an environment variable whose value is read only by the runner.
type Config struct {
	Name                 string      `json:"name,omitempty"`
	Mode                 string      `json:"mode"`
	Audio                string      `json:"audio"`
	MicAudio             string      `json:"mic_audio,omitempty"`
	Reference            string      `json:"reference,omitempty"`
	OutputDir            string      `json:"output_dir"`
	Model                ModelConfig `json:"model"`
	Draft                DraftConfig `json:"draft"`
	DecodeIntervalMS     int         `json:"decode_interval_ms,omitempty"`
	SegmentLengthSeconds float64     `json:"segment_length_seconds,omitempty"`
	CacheLimitMB         *int        `json:"cache_limit_mb,omitempty"`
	IdleTimeoutMS        int         `json:"idle_timeout_ms,omitempty"`
	VADStrategy          string      `json:"vad_strategy,omitempty"`
	LeaseStrategy        string      `json:"lease_strategy,omitempty"`
	FastProfile          bool        `json:"fast_profile,omitempty"`
	DeferEvaluation      bool        `json:"-"`
	VAD                  VADConfig   `json:"vad"`
	Language             string      `json:"language,omitempty"`
	Cleanup              bool        `json:"cleanup"`
	Summary              bool        `json:"summary"`
	LanguageModel        LLMConfig   `json:"language_model"`
}

type ModelConfig struct {
	ID  string `json:"id"`
	Dir string `json:"dir"`
}

type DraftConfig struct {
	Enabled bool   `json:"enabled"`
	Dir     string `json:"dir,omitempty"`
	Bits    int    `json:"bits,omitempty"`
}

type VADConfig struct {
	Enabled  bool   `json:"enabled"`
	ModelDir string `json:"model_dir,omitempty"`
}

type LLMConfig struct {
	Provider string `json:"provider"`
	Endpoint string `json:"endpoint,omitempty"`
	Model    string `json:"model,omitempty"`
	KeyEnv   string `json:"key_env,omitempty"`
}

var environmentName = regexp.MustCompile(`^[A-Za-z_][A-Za-z0-9_]*$`)

// LoadConfig reads exactly one JSON object and rejects unknown fields,
// including a key accidentally pasted into the configuration.
func LoadConfig(path string) (Config, error) {
	file, err := os.Open(path)
	if err != nil {
		return Config{}, err
	}
	defer file.Close()
	decoder := json.NewDecoder(file)
	decoder.DisallowUnknownFields()
	var config Config
	if err := decoder.Decode(&config); err != nil {
		return Config{}, fmt.Errorf("profile config: %w", err)
	}
	if err := decoder.Decode(new(any)); !errors.Is(err, io.EOF) {
		return Config{}, fmt.Errorf("profile config: more than one JSON value")
	}
	if err := config.Validate(); err != nil {
		return Config{}, err
	}
	return config, nil
}

// Validate rejects unsupported combinations before a model starts.
func (c *Config) Validate() error {
	if c.Mode != "import" && c.Mode != "subtitles" {
		return fmt.Errorf("profile: mode must be import or subtitles")
	}
	switch strings.ToLower(filepath.Ext(c.Audio)) {
	case ".m4a", ".mp4", ".wav":
	default:
		return fmt.Errorf("profile: audio must be an m4a, mp4 or wav file")
	}
	if c.MicAudio != "" {
		switch strings.ToLower(filepath.Ext(c.MicAudio)) {
		case ".m4a", ".mp4", ".wav":
		default:
			return fmt.Errorf("profile: mic_audio must be an m4a, mp4 or wav file")
		}
	}
	if c.OutputDir == "" || c.Model.ID == "" || c.Model.Dir == "" {
		return fmt.Errorf("profile: output_dir and model id/dir are required")
	}
	if c.Draft.Enabled {
		if c.Draft.Dir == "" {
			return fmt.Errorf("profile: an enabled draft needs its dir")
		}
		if c.Draft.Bits == 0 {
			c.Draft.Bits = 8
		}
		if c.Draft.Bits != 4 && c.Draft.Bits != 6 && c.Draft.Bits != 8 {
			return fmt.Errorf("profile: draft bits must be 4, 6 or 8")
		}
	}
	if c.DecodeIntervalMS < 0 || c.SegmentLengthSeconds < 0 {
		return fmt.Errorf("profile: decode interval and segment length cannot be negative")
	}
	if c.SegmentLengthSeconds > 0 && c.SegmentLengthSeconds <= 1 {
		return fmt.Errorf("profile: segment_length_seconds must exceed the one-second chunk overlap")
	}
	if c.CacheLimitMB != nil && *c.CacheLimitMB < 0 {
		return fmt.Errorf("profile: cache_limit_mb cannot be negative")
	}
	if c.IdleTimeoutMS < 0 {
		return fmt.Errorf("profile: idle_timeout_ms cannot be negative")
	}
	if c.VADStrategy != "" && c.VADStrategy != "whole" && c.VADStrategy != "batched" {
		return fmt.Errorf("profile: vad_strategy must be whole or batched")
	}
	if c.LeaseStrategy != "" && c.LeaseStrategy != "per-track" {
		return fmt.Errorf("profile: lease_strategy must be per-track")
	}
	if c.VAD.Enabled && c.VAD.ModelDir == "" {
		return fmt.Errorf("profile: enabled VAD needs vad.model_dir")
	}
	if c.Cleanup || c.Summary {
		if c.LanguageModel.Provider == "" {
			return fmt.Errorf("profile: clean-up and summary need language_model.provider")
		}
		if c.LanguageModel.KeyEnv != "" && !environmentName.MatchString(c.LanguageModel.KeyEnv) {
			return fmt.Errorf("profile: language_model.key_env is not an environment variable name")
		}
		if c.LanguageModel.Endpoint != "" {
			endpoint, err := url.Parse(c.LanguageModel.Endpoint)
			if err != nil || endpoint.User != nil || endpoint.RawQuery != "" || endpoint.Fragment != "" {
				return fmt.Errorf("profile: language_model.endpoint cannot contain credentials, query or fragment")
			}
		}
	}
	return nil
}
