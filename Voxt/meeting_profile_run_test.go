package main

import (
	"context"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/asr"
	"github.com/sgl-project/sglang-omni/Voxt/internal/meeting"
	"github.com/sgl-project/sglang-omni/Voxt/internal/profile"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
	"github.com/sgl-project/sglang-omni/Voxt/internal/store"
	"github.com/sgl-project/sglang-omni/Voxt/internal/store/sqlitestore"
	"github.com/sgl-project/sglang-omni/Voxt/internal/vad"
)

type profileASR interface {
	asr.FileTranscriber
	asr.Diarizer
	Hold(context.Context) (func(), error)
}

type profileVAD interface {
	vad.Model
	Hold(context.Context) (func(), error)
}

type profileDependencies struct {
	decode       func(context.Context, string) ([]byte, error)
	openStore    func(context.Context, string) (store.Store, error)
	newASR       func(profile.Config) (profileASR, error)
	newVAD       func(profile.Config) (profileVAD, error)
	newLLM       func(context.Context, profile.LLMConfig) (meeting.SummaryChat, error)
	nativePIDs   func() []int
	closeRuntime func() error
}

func defaultProfileDependencies() profileDependencies {
	return profileDependencies{
		decode: decodeAudio,
		openStore: func(ctx context.Context, dir string) (store.Store, error) {
			return sqlitestore.Open(ctx, dir)
		},
		nativePIDs:   func() []int { return nil },
		closeRuntime: func() error { return nil },
	}
}

func profileMeetingText(segments []store.MeetingSegment) string {
	lines := make([]string, 0, len(segments))
	for _, segment := range segments {
		if text := strings.TrimSpace(segment.Text); text != "" {
			lines = append(lines, text)
		}
	}
	return strings.Join(lines, "\n")
}

func runMeetingProfile(
	ctx context.Context,
	config profile.Config,
	dependencies profileDependencies,
	recorder *profile.Recorder,
	nativeLog *profile.NativeLog,
	stopArtifacts func() error,
) (report profile.Report, runErr error) {
	monitor := profile.StartResourceMonitor(ctx, profile.SelfAndNative(dependencies.nativePIDs))
	var database store.Store
	var releases []func()
	var temporaryDir string
	defer func() {
		teardownErr := recorder.Measure("teardown", 0, func() error {
			var errs []error
			for index := len(releases) - 1; index >= 0; index-- {
				releases[index]()
			}
			if dependencies.closeRuntime != nil {
				if err := dependencies.closeRuntime(); err != nil {
					errs = append(errs, err)
				}
			}
			if database != nil {
				if err := database.Close(); err != nil {
					errs = append(errs, err)
				}
			}
			if nativeLog != nil {
				if err := nativeLog.Close(); err != nil {
					errs = append(errs, err)
				}
			}
			if stopArtifacts != nil {
				if err := stopArtifacts(); err != nil {
					errs = append(errs, err)
				}
			}
			if temporaryDir != "" {
				if err := os.RemoveAll(temporaryDir); err != nil {
					errs = append(errs, err)
				}
			}
			return errors.Join(errs...)
		})
		recorder.SetResources(monitor.Stop(context.WithoutCancel(ctx)))
		runErr = recorder.CleanError(errors.Join(runErr, teardownErr))
		report = recorder.Finish(runErr)
	}()

	temporaryDir, runErr = os.MkdirTemp("", "voxt-meeting-profile-")
	if runErr != nil {
		return
	}
	var pcm, micPCM []byte
	runErr = recorder.Measure("decode", 0, func() error {
		var err error
		pcm, err = dependencies.decode(ctx, config.Audio)
		if err != nil {
			return err
		}
		if config.MicAudio != "" {
			micPCM, err = dependencies.decode(ctx, config.MicAudio)
		}
		return err
	})
	audioDuration := asr.Duration(pcm, asr.SampleRate)
	if micDuration := asr.Duration(micPCM, asr.SampleRate); micDuration > audioDuration {
		audioDuration = micDuration
	}
	recorder.SetLastAudio(audioDuration)
	recorder.SetAudioDuration(audioDuration)
	if runErr != nil {
		return
	}

	var meetingRow store.Meeting
	var reference string
	runErr = recorder.Measure("setup", 0, func() error {
		var err error
		database, err = dependencies.openStore(ctx, filepath.Join(temporaryDir, "db"))
		if err != nil {
			return err
		}
		if config.Reference != "" {
			raw, err := os.ReadFile(config.Reference)
			if err != nil {
				return err
			}
			reference = string(raw)
		}
		meetingDir := filepath.Join(temporaryDir, "meetings", "profile")
		if err := os.MkdirAll(meetingDir, 0o700); err != nil {
			return err
		}
		if err := os.WriteFile(filepath.Join(meetingDir, "system.wav"), asr.WAV(pcm, asr.SampleRate), 0o600); err != nil {
			return err
		}
		if len(micPCM) > 0 {
			if err := os.WriteFile(filepath.Join(meetingDir, "mic.wav"), asr.WAV(micPCM, asr.SampleRate), 0o600); err != nil {
				return err
			}
		}
		mode := store.MeetingModeSubtitles
		if config.Mode == "import" {
			mode = store.MeetingModeImport
		}
		if len(micPCM) > 0 {
			// A second track makes it a real two-track meeting: the final
			// pass processes mic and system, not the import's system only.
			mode = store.MeetingModeMeeting
		}
		now := time.Now()
		meetingRow, err = database.Meetings().CreateMeeting(ctx, store.Meeting{
			ID: "profile", CreatedAt: now, EndedAt: now.Add(audioDuration), Status: store.MeetingFinishing,
			Mode: mode, Title: filepath.Base(config.Audio), DurationMS: audioDuration.Milliseconds(), Audio: "profile",
		})
		return err
	})
	if runErr != nil {
		return
	}

	var speech profileASR
	var detector profileVAD
	finalCtx, stopFinal := context.WithCancel(ctx)
	defer stopFinal()
	var firstRequestCritical bool
	monitor.Snapshot("pre_load")
	recorder.SetFirstASRHook(func() {
		if sample := monitor.Snapshot("first_request"); sample.Pressure == "critical" {
			// Same early-archive gate as ready: the first request already
			// hit critical memory pressure, so stop the final pass instead
			// of spending the whole input on a rejected sample.
			firstRequestCritical = true
			stopFinal()
		}
	})
	runErr = recorder.Measure("load", 0, func() error {
		var err error
		speech, err = dependencies.newASR(config)
		if err != nil {
			return err
		}
		release, err := speech.Hold(ctx)
		if err != nil {
			return err
		}
		releases = append(releases, release)
		if config.VAD.Enabled {
			detector, err = dependencies.newVAD(config)
			if err != nil {
				return err
			}
			release, err = detector.Hold(ctx)
			if err != nil {
				return err
			}
			releases = append(releases, release)
		}
		return nil
	})
	if runErr != nil {
		return
	}
	if ready := monitor.Snapshot("ready"); ready.Pressure == "critical" || monitor.PeakPressure() == "critical" {
		// Loading already pushed the machine to critical memory pressure:
		// archive the run instead of spending the whole input on a sample
		// the validator would reject anyway.
		runErr = errors.New("memory pressure critical before the model was ready; archiving the run")
		return
	}

	chatKind := "correction"
	meetingConfig := meeting.Config{
		Store: database.Meetings(), Dir: filepath.Join(temporaryDir, "meetings"),
		Settings: func() settings.Meeting {
			return settings.Meeting{
				Feature: settings.Feature{LLM: config.LanguageModel.Provider},
				Lang:    config.Language, NoAutoSummary: true,
			}
		},
		Privacy: func() settings.Privacy { return settings.Privacy{KeepHistory: true, KeepAudio: true} },
		ASR: func(_ context.Context, source meeting.Source) (meeting.TrackASR, error) {
			return meeting.TrackASR{Caps: speech.Caps(), File: recorder.WrapFile(string(source), speech), Lang: config.Language}, nil
		},
		Diarizer: func(context.Context) (asr.Diarizer, error) {
			return recorder.WrapDiarizer("system", speech), nil
		},
		DisableVAD:   !config.VAD.Enabled,
		VADFallback:  func() { recorder.Warn("VAD used energy fallback") },
		MainLanguage: func() string { return config.Language },
		Warn:         recorder.Warn,
		LLM: func(ctx context.Context, _ string) (meeting.SummaryChat, error) {
			chat, err := dependencies.newLLM(ctx, config.LanguageModel)
			if err != nil {
				return nil, err
			}
			return recorder.WrapChat(chatKind, chat), nil
		},
	}
	if config.VAD.Enabled {
		meetingConfig.VAD = recorder.WrapVAD(detector)
	}
	if config.SegmentLengthSeconds > 0 {
		chunks := meeting.DefaultChunkParams()
		chunks.MaxDur = time.Duration(config.SegmentLengthSeconds * float64(time.Second))
		meetingConfig.FinalizeChunkParams = &chunks
	}
	manager := meeting.NewManager(meetingConfig)
	stopToSummaryStart := time.Now()
	runErr = recorder.Measure("final_pass", audioDuration, func() error {
		return meeting.FinalizePass(finalCtx, meetingConfig, meetingRow, nil)
	})
	if firstRequestCritical {
		runErr = errors.New("memory pressure critical at the first request; archiving the run")
		return
	}
	if runErr != nil {
		return
	}
	_, rawSegments, err := database.Meetings().Meeting(ctx, meetingRow.ID)
	if err != nil {
		runErr = err
		return
	}
	rawText := profileMeetingText(rawSegments)
	finalText := rawText
	summaryText := ""
	if !filepath.IsAbs(config.OutputDir) {
		// Fake configurations use a relative placeholder; real runs pass
		// an absolute output directory.
	} else if segmentsErr := os.MkdirAll(config.OutputDir, 0o700); segmentsErr != nil {
		recorder.Warn("segments.jsonl: " + segmentsErr.Error())
	} else if segmentsFile, segmentsErr := os.Create(filepath.Join(config.OutputDir, "segments.jsonl")); segmentsErr == nil {
		encoder := json.NewEncoder(segmentsFile)
		for _, segment := range rawSegments {
			if encodeErr := encoder.Encode(map[string]any{
				"source": segment.Source, "speaker": segment.Speaker,
				"start_ms": segment.StartMS, "end_ms": segment.EndMS, "text": segment.Text,
			}); encodeErr != nil {
				recorder.Warn("segments.jsonl: " + encodeErr.Error())
				break
			}
		}
		if closeErr := segmentsFile.Close(); closeErr != nil {
			recorder.Warn("segments.jsonl: " + closeErr.Error())
		}
	} else {
		recorder.Warn("segments.jsonl: " + segmentsErr.Error())
	}

	if config.Cleanup {
		chatKind = "correction"
		err := recorder.Measure("correction_pass", 0, func() error {
			return manager.CorrectTranscript(ctx, meetingRow.ID)
		})
		if err != nil {
			runErr = errors.Join(runErr, err)
		}
	}
	if config.Summary {
		chatKind = "summary"
		err := recorder.Measure("summary_pass", 0, func() error {
			return manager.GenerateMeetingSummary(ctx, meetingRow.ID)
		})
		if err != nil {
			runErr = errors.Join(runErr, err)
		}
	}
	recorder.SetStopToSummary(time.Since(stopToSummaryStart))
	storedMeeting, finalSegments, err := database.Meetings().Meeting(ctx, meetingRow.ID)
	if err != nil {
		runErr = errors.Join(runErr, err)
		return
	}
	finalText = profileMeetingText(finalSegments)
	if storedMeeting.Summary != "" {
		summaryText = storedMeeting.Summary
	}
	recorder.SetTranscripts(rawText, finalText, summaryText)
	if config.Reference != "" {
		recorder.SetEvaluation(reference, rawText, finalText)
	}
	return
}
