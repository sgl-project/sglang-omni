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
)

// TestMeetingInteractive starts a real speech model only for an explicit
// interactive-priority profiling configuration. Default test runs use fakes.
func TestMeetingInteractive(t *testing.T) {
	configPath := strings.TrimSpace(os.Getenv("VOXTNEW_INTERACTIVE_CONFIG"))
	if configPath == "" {
		t.Skip("set VOXTNEW_INTERACTIVE_CONFIG to run real interactive-priority profiling")
	}
	config, err := profile.LoadInteractiveConfig(configPath)
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
	vadModel, err := omni.NewVAD(runtime, config.VAD.ModelDir)
	if err != nil {
		t.Fatal(err)
	}
	releaseSpeech, err := speech.Hold(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	defer releaseSpeech()
	releaseVAD, err := vadModel.Hold(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	defer releaseVAD()
	longPCM, err := decodeAudio(context.Background(), config.LongAudio)
	if err != nil {
		t.Fatal(err)
	}
	interactivePCM, err := decodeAudio(context.Background(), config.InteractiveAudio)
	if err != nil {
		t.Fatal(err)
	}
	report := profile.RunInteractive(context.Background(), config, longPCM, interactivePCM, speech, vadModel)
	if err := profile.WriteInteractiveReport(config.OutputDir, report); err != nil {
		t.Fatal(err)
	}
	resultPath := filepath.Join(config.OutputDir, "result.json")
	t.Logf("interactive-priority result: %s", resultPath)
	if report.Status != "success" {
		t.Fatalf("interactive-priority profiling is invalid: %s", strings.Join(report.Errors, "; "))
	}
}
