package main

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"

	"github.com/sgl-project/sglang-omni/Voxt/internal/omni"
	"github.com/sgl-project/sglang-omni/Voxt/internal/profile"
	"github.com/sgl-project/sglang-omni/Voxt/internal/vad"
)

// TestMeetingOverlap runs real ASR only when its dedicated config is set.
// Every language-model call is the local deterministic scheduling fake.
func TestMeetingOverlap(t *testing.T) {
	configPath := strings.TrimSpace(os.Getenv("VOXTNEW_OVERLAP_CONFIG"))
	if configPath == "" {
		t.Skip("set VOXTNEW_OVERLAP_CONFIG to run real ASR with a simulated LLM")
	}
	config, err := profile.LoadOverlapConfig(configPath)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(config.OutputDir, 0o700); err != nil {
		t.Fatal(err)
	}
	logFile, err := os.OpenFile(filepath.Join(config.OutputDir, "native.log"), os.O_CREATE|os.O_WRONLY|os.O_TRUNC, 0o600)
	if err != nil {
		t.Fatal(err)
	}
	var logMu sync.Mutex
	executable, err := os.Executable()
	if err != nil {
		_ = logFile.Close()
		t.Fatal(err)
	}
	binary, err := omni.Locate(executable)
	if err != nil {
		_ = logFile.Close()
		t.Fatal(err)
	}
	runtime := omni.NewRuntime(omni.Config{Binary: binary, IdleTimeout: -1, Log: func(level, message string) {
		logMu.Lock()
		defer logMu.Unlock()
		_, _ = fmt.Fprintf(logFile, "%s %s\n", level, message)
	}})
	defer func() {
		runtime.Close()
		if err := logFile.Close(); err != nil {
			t.Errorf("close native log: %v", err)
		}
	}()

	speech, err := omni.NewASR(runtime, config.Model.ID, config.Model.Dir)
	if err != nil {
		t.Fatal(err)
	}
	if config.Draft.Enabled {
		speech, err = speech.WithDraftBits(config.Draft.Dir, config.Draft.Bits)
		if err != nil {
			t.Fatal(err)
		}
	}
	releaseSpeech, err := speech.Hold(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	defer releaseSpeech()
	var detector vad.Model
	if config.VAD.Enabled {
		model, err := omni.NewVAD(runtime, config.VAD.ModelDir)
		if err != nil {
			t.Fatal(err)
		}
		releaseVAD, err := model.Hold(context.Background())
		if err != nil {
			t.Fatal(err)
		}
		defer releaseVAD()
		detector = model
	}
	pcm, err := decodeAudio(config.Audio)
	if err != nil {
		t.Fatal(err)
	}
	reports := profile.RunOverlap(context.Background(), config, pcm, speech, detector)
	if err := profile.WriteOverlapReports(config.OutputDir, reports); err != nil {
		t.Fatal(err)
	}
	for _, report := range reports {
		t.Logf("%s result: %s", report.Method, filepath.Join(config.OutputDir, report.Method, "result.json"))
		if report.Status != "success" {
			t.Errorf("%s is invalid: %s", report.Method, strings.Join(report.Errors, "; "))
		}
	}
}
