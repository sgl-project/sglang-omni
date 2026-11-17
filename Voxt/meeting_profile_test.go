package main

import (
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"runtime/pprof"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/meeting"
	"github.com/sgl-project/sglang-omni/Voxt/internal/omni"
	"github.com/sgl-project/sglang-omni/Voxt/internal/profile"
	"github.com/sgl-project/sglang-omni/Voxt/internal/providers"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
)

// TestMeetingProfile is this harness's only entry point for real speech or
// language models. An unset environment variable keeps its default test runs
// entirely on fakes.
func TestMeetingProfile(t *testing.T) {
	configPath := strings.TrimSpace(os.Getenv("VOXTNEW_PROFILE_CONFIG"))
	if configPath == "" {
		t.Skip("set VOXTNEW_PROFILE_CONFIG to run a real profiling configuration")
	}
	config, err := profile.LoadConfig(configPath)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(config.OutputDir, 0o700); err != nil {
		t.Fatal(err)
	}
	recorder := profile.NewRecorder(config)
	nativeLog, err := profile.OpenNativeLog(filepath.Join(config.OutputDir, "native.log"), recorder)
	if err != nil {
		t.Fatal(err)
	}
	stopArtifacts, err := startMeetingProfiles(config.OutputDir)
	if err != nil {
		_ = nativeLog.Close()
		t.Fatal(err)
	}
	dependencies, runErr := realProfileDependencies(config, nativeLog, recorder)
	var report profile.Report
	if runErr == nil {
		report, runErr = runMeetingProfile(context.Background(), config, dependencies, recorder, nativeLog, stopArtifacts)
	} else {
		runErr = recorder.CleanError(errors.Join(runErr, stopArtifacts(), nativeLog.Close()))
		report = recorder.Finish(runErr)
	}
	resultPath := filepath.Join(config.OutputDir, "result.json")
	if err := profile.WriteReport(resultPath, report); err != nil {
		t.Fatal(recorder.CleanError(err))
	}
	t.Logf("profiling result: %s", recorder.Clean(resultPath))
	if runErr != nil {
		t.Fatal(recorder.CleanError(runErr))
	}
	if report.Status != "success" {
		t.Fatalf("profiling run is invalid: %s", strings.Join(report.Warnings, "; "))
	}
}

func realProfileDependencies(config profile.Config, nativeLog *profile.NativeLog, recorder *profile.Recorder) (profileDependencies, error) {
	dependencies := defaultProfileDependencies()
	executable, err := os.Executable()
	if err != nil {
		return dependencies, err
	}
	binary, err := omni.Locate(executable)
	if err != nil {
		return dependencies, err
	}
	runtime := omni.NewRuntime(omni.Config{Binary: binary, IdleTimeout: -1, Log: nativeLog.Write})
	dependencies.nativePIDs = func() []int {
		processes := runtime.Processes()
		pids := make([]int, 0, len(processes))
		for _, process := range processes {
			pids = append(pids, process.PID)
		}
		return pids
	}
	dependencies.closeRuntime = func() error {
		runtime.Close()
		return nil
	}
	dependencies.newASR = func(config profile.Config) (profileASR, error) {
		if config.Mode == "import" && config.Model.ID != omni.ASRMoss {
			return nil, fmt.Errorf("profile: import mode requires %s", omni.ASRMoss)
		}
		speech, err := omni.NewASR(runtime, config.Model.ID, config.Model.Dir)
		if err != nil {
			return nil, err
		}
		speech, err = speech.WithTuning(omni.Tuning{
			DecodeInterval: durationMilliseconds(config.DecodeIntervalMS),
			CacheLimitMB:   config.CacheLimitMB,
		})
		if err != nil {
			return nil, err
		}
		if config.Draft.Enabled {
			speech, err = speech.WithDraftBits(config.Draft.Dir, config.Draft.Bits)
			if err != nil {
				return nil, err
			}
		}
		return speech, nil
	}
	dependencies.newVAD = func(config profile.Config) (profileVAD, error) {
		return omni.NewVAD(runtime, config.VAD.ModelDir)
	}
	dependencies.newLLM = func(_ context.Context, config profile.LLMConfig) (meeting.SummaryChat, error) {
		return openProfileLLM(config, recorder)
	}
	return dependencies, nil
}

func openProfileLLM(config profile.LLMConfig, recorder *profile.Recorder) (meeting.SummaryChat, error) {
	provider := settings.Provider{ID: config.Provider, Endpoint: config.Endpoint, Model: config.Model}
	key := ""
	if config.KeyEnv != "" {
		key = os.Getenv(config.KeyEnv)
	}
	resolved, resolveErr := providers.Config(provider).Resolve()
	endpoint, model := config.Endpoint, providers.Model(provider)
	if resolveErr == nil {
		endpoint, model = resolved.BaseURL, resolved.DefaultModel
	}
	recorder.RegisterSecrets(endpoint, model, config.Endpoint, config.Model, key)
	if missing := providers.Missing(provider, key != ""); missing != "" {
		return nil, fmt.Errorf("profile: language model is missing %s", missing)
	}
	return providers.Open(provider, key, nil)
}

func durationMilliseconds(milliseconds int) time.Duration {
	return time.Duration(milliseconds) * time.Millisecond
}

func startMeetingProfiles(outputDir string) (func() error, error) {
	cpuFile, err := os.OpenFile(filepath.Join(outputDir, "cpu.pprof"), os.O_CREATE|os.O_WRONLY|os.O_TRUNC, 0o600)
	if err != nil {
		return nil, err
	}
	if err := pprof.StartCPUProfile(cpuFile); err != nil {
		_ = cpuFile.Close()
		return nil, err
	}
	var once sync.Once
	var stopErr error
	return func() error {
		once.Do(func() {
			pprof.StopCPUProfile()
			runtime.GC()
			heapFile, err := os.OpenFile(filepath.Join(outputDir, "heap.pprof"), os.O_CREATE|os.O_WRONLY|os.O_TRUNC, 0o600)
			if err != nil {
				stopErr = errors.Join(stopErr, err, cpuFile.Close())
				return
			}
			stopErr = errors.Join(pprof.WriteHeapProfile(heapFile), heapFile.Close(), cpuFile.Close())
		})
		return stopErr
	}, nil
}
