package profile

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func validConfig() Config {
	return Config{
		Mode: "import", Audio: "meeting.wav", OutputDir: "out",
		Model:         ModelConfig{ID: "asr.moss", Dir: "model"},
		Draft:         DraftConfig{Enabled: true, Dir: "draft"},
		VAD:           VADConfig{Enabled: true, ModelDir: "vad"},
		LanguageModel: LLMConfig{Provider: "openai", KeyEnv: "PROFILE_TEST_KEY"},
	}
}

func TestConfigValidation(t *testing.T) {
	config := validConfig()
	config.Cleanup = true
	if err := config.Validate(); err != nil {
		t.Fatal(err)
	}
	if config.Draft.Bits != 8 {
		t.Errorf("draft bits default %d", config.Draft.Bits)
	}
	negative := -1
	tests := []struct {
		name   string
		change func(*Config)
	}{
		{"mode", func(c *Config) { c.Mode = "meeting" }},
		{"audio type", func(c *Config) { c.Audio = "meeting.mp3" }},
		{"model", func(c *Config) { c.Model.Dir = "" }},
		{"draft dir", func(c *Config) { c.Draft.Dir = "" }},
		{"draft bits", func(c *Config) { c.Draft.Bits = 16 }},
		{"decode", func(c *Config) { c.DecodeIntervalMS = -1 }},
		{"segment", func(c *Config) { c.SegmentLengthSeconds = -1 }},
		{"segment overlap", func(c *Config) { c.SegmentLengthSeconds = 1 }},
		{"cache", func(c *Config) { c.CacheLimitMB = &negative }},
		{"VAD", func(c *Config) { c.VAD.ModelDir = "" }},
		{"provider", func(c *Config) { c.LanguageModel.Provider = "" }},
		{"key environment", func(c *Config) { c.LanguageModel.KeyEnv = "BAD;env" }},
		{"endpoint credentials", func(c *Config) { c.LanguageModel.Endpoint = "https://user:secret@example.com/v1" }},
		{"endpoint query", func(c *Config) { c.LanguageModel.Endpoint = "https://example.com/v1?api_key=secret" }},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			config := validConfig()
			config.Cleanup = true
			test.change(&config)
			if err := config.Validate(); err == nil {
				t.Fatal("invalid config was accepted")
			}
		})
	}
}

func TestOverlapConfigIsLocalFakeOnly(t *testing.T) {
	config := fakeOverlapConfig("dictation")
	config.Draft = DraftConfig{Enabled: true, Dir: "draft"}
	if err := config.Validate(); err != nil {
		t.Fatal(err)
	}
	if config.Draft.Bits != 8 {
		t.Errorf("draft bits default %d", config.Draft.Bits)
	}
	for _, change := range []func(*OverlapConfig){
		func(c *OverlapConfig) { c.Mode = "subtitles" },
		func(c *OverlapConfig) { c.SimulatedLLM.LatencyMS = -1 },
		func(c *OverlapConfig) { c.VAD = VADConfig{Enabled: true} },
	} {
		invalid := fakeOverlapConfig("meeting")
		change(&invalid)
		if err := invalid.Validate(); err == nil {
			t.Errorf("invalid overlap config was accepted: %+v", invalid)
		}
	}
	path := filepath.Join(t.TempDir(), "overlap.json")
	if err := os.WriteFile(path, []byte(`{"mode":"dictation","audio":"a.wav","output_dir":"out","model":{"id":"asr.fast","dir":"m"},"language_model":{"provider":"openai"}}`), 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := LoadOverlapConfig(path); err == nil || !strings.Contains(err.Error(), "unknown field") {
		t.Fatalf("remote language model config was accepted: %v", err)
	}
}

func TestLoadConfigRejectsUnknownFieldsAndSecrets(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "config.json")
	if err := os.WriteFile(path, []byte(`{"mode":"subtitles","audio":"a.wav","output_dir":"out","model":{"id":"asr.fast","dir":"m"},"key":"do-not-save"}`), 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := LoadConfig(path); err == nil || !strings.Contains(err.Error(), "unknown field") {
		t.Fatalf("err %v", err)
	}
	if err := os.WriteFile(path, []byte(`{"mode":"subtitles","audio":"a.wav","output_dir":"out","model":{"id":"asr.fast","dir":"m"}} {}`), 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := LoadConfig(path); err == nil {
		t.Fatal("two JSON values were accepted")
	}
}
