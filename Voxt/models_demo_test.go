package main

import (
	"slices"
	"testing"

	"github.com/sgl-project/sglang-omni/Voxt/internal/actnow"
	"github.com/sgl-project/sglang-omni/Voxt/internal/models"
)

func TestModelsDemoActionControls(t *testing.T) {
	for _, test := range []struct {
		name    string
		fixture string
		steps   []historyDemoStep
		want    []string
		check   func(*testing.T, *meetingDemoHost)
	}{
		{name: "network open", fixture: "en/models-page-normal", steps: historySteps("Network…"), want: []string{"openNetwork"}},
		{name: "network cancel", fixture: "en/models-network-automatic", steps: historySteps("Cancel"), want: []string{"closeNetwork"}},
		{name: "network save", fixture: "en/models-network-no-proxy", steps: historySteps("Save"), want: []string{"saveNetwork", "closeNetwork"}},
		{name: "network custom", fixture: "en/models-network-automatic", steps: historySteps("Custom"), check: func(t *testing.T, host *meetingDemoHost) {
			if host.modelsUI.netForm.proxyMode != proxyCustom {
				t.Fatalf("proxy mode %d", host.modelsUI.netForm.proxyMode)
			}
		}},
		{name: "network source", fixture: "en/models-network-automatic", steps: historySteps("hf-mirror.com only"), check: func(t *testing.T, host *meetingDemoHost) {
			if host.modelsUI.netForm.source != 2 {
				t.Fatalf("source %d", host.modelsUI.netForm.source)
			}
		}},
		{name: "network use theirs", fixture: "en/models-network-conflict", steps: historySteps("Use theirs"), want: []string{"followNetwork " + actnow.KeyProxy + " true"}},
		{name: "network own", fixture: "en/models-network-follows", steps: historySteps("Use a different setting for this app"), want: []string{"followNetwork " + actnow.KeyProxy + " false"}},
		{name: "network follow", fixture: "en/models-network-own", steps: historySteps("Use the same setting as the other apps that share this model library"), want: []string{"followNetwork " + actnow.KeyProxy + " true"}},
		{name: "network proxy input", fixture: "en/models-network-custom-validation", steps: []historyDemoStep{{label: "Proxy address", text: "fixture-proxy"}}, check: func(t *testing.T, host *meetingDemoHost) {
			if host.modelsUI.netForm.proxyURL != "fixture-proxy" {
				t.Fatalf("proxy input %q", host.modelsUI.netForm.proxyURL)
			}
		}},
		{name: "download", fixture: "en/models-download-fresh", steps: historySteps("Download Qwen3-ASR 0.6B (4bit)"), want: []string{"downloadModel " + models.ASRFast}},
		{name: "download again", fixture: "en/models-download-deleted", steps: historySteps("Download Qwen3-ASR 0.6B (4bit)"), want: []string{"downloadModel " + models.ASRFast}},
		{name: "resume", fixture: "en/models-download-stopped", steps: historySteps("Download Qwen3-ASR 0.6B (4bit)"), want: []string{"downloadModel " + models.ASRFast}},
		{name: "retry", fixture: "en/models-error-not-found", steps: historySteps("Try again"), want: []string{"downloadModel " + models.ASRFast}},
		{name: "cancel download", fixture: "en/models-download-progress", steps: historySteps("Cancel downloading Qwen3-ASR 0.6B (4bit)"), want: []string{"cancelModel " + models.ASRFast}},
		{name: "discard download", fixture: "en/models-download-stopped", steps: historySteps("Delete the partial download of Qwen3-ASR 0.6B (4bit)"), want: []string{"discardModelDownload " + models.ASRFast}},
		{name: "discard retired", fixture: "en/models-download-partial-retired", steps: historySteps("Delete the partial download of Qwen3.5 2B (4bit)"), want: []string{"discardModelDownload " + models.LLMFast}},
		{name: "use on disk", fixture: "en/models-download-on-disk", steps: historySteps("Use Qwen3-ASR 0.6B (4bit)"), want: []string{"useModel " + models.ASRFast}},
		{name: "ask delete", fixture: "en/models-page-normal", steps: historySteps("Delete Qwen3-ASR 1.7B (8bit)…"), want: []string{"askDeleteModel " + models.ASRAccurate}},
		{name: "delete plain", fixture: "en/models-delete-plain", steps: historySteps("Delete Qwen3-ASR 1.7B (8bit)"), want: []string{"confirmModelDelete " + models.ASRAccurate + " false"}},
		{name: "delete plain cancel", fixture: "en/models-delete-plain", steps: historySteps("Cancel"), check: func(t *testing.T, host *meetingDemoHost) {
			if host.modelsUI.modelAsk != "" {
				t.Fatal("plain deletion stayed open")
			}
		}},
		{name: "delete shared", fixture: "en/models-delete-shared", steps: historySteps("Remove from Voxt"), want: []string{"confirmModelDelete " + models.ASRAccurate + " false"}},
		{name: "delete all", fixture: "en/models-delete-shared", steps: historySteps("Delete for all apps…", "Delete Qwen3-ASR 1.7B (8bit) for all apps"), want: []string{"confirmModelDelete " + models.ASRAccurate + " true"}},
		{name: "delete shared cancel", fixture: "en/models-delete-shared", steps: historySteps("Cancel"), check: func(t *testing.T, host *meetingDemoHost) {
			if host.modelsUI.modelShared != nil {
				t.Fatal("shared deletion stayed open")
			}
		}},
		{name: "delete all cancel", fixture: "en/models-delete-all", steps: historySteps("Cancel"), check: func(t *testing.T, host *meetingDemoHost) {
			if host.modelsUI.modelShared != nil {
				t.Fatal("all-app deletion stayed open")
			}
		}},
		{name: "hide", fixture: "en/models-download-fresh", steps: historySteps("Hide Qwen3-ASR 0.6B (4bit)"), want: []string{"setModelHidden " + models.ASRFast + " true"}},
		{name: "show", fixture: "en/models-hidden-open", steps: historySteps("Show Qwen3-ASR 0.6B (4bit)"), want: []string{"setModelHidden " + models.ASRFast + " false"}},
		{name: "faster on", fixture: "en/models-page-normal", steps: historySteps("Faster transcription for Qwen3-ASR 1.7B (8bit)"), want: []string{"setModelFaster " + models.ASRAccurate + " true"}},
		{name: "faster off", fixture: "en/models-draft-on-missing", steps: historySteps("Faster transcription for Qwen3-ASR 1.7B (8bit)"), want: []string{"setModelFaster " + models.ASRAccurate + " false"}},
		{name: "draft download", fixture: "en/models-draft-on-missing", steps: historySteps("Download Qwen3-ASR 1.7B DFlash draft"), want: []string{"downloadModel " + models.ASRAccurateDraft}},
		{name: "keep loaded", fixture: "en/models-keep-auto", steps: []historyDemoStep{{label: "Automatic", click: true}, {label: "Until Voxt quits", click: true}}, want: []string{`setModelsKeepLoaded "quit"`}},
		{name: "advanced toggle", fixture: "en/models-page-normal", steps: historySteps("Advanced"), check: func(t *testing.T, host *meetingDemoHost) {
			if !host.modelsUI.modelAdvanced {
				t.Fatal("Advanced did not open")
			}
		}},
		{name: "hidden toggle", fixture: "en/models-hidden-open", steps: historySteps("Hidden models"), check: func(t *testing.T, host *meetingDemoHost) {
			if host.modelsUI.modelHiddenOpen {
				t.Fatal("Hidden models did not close")
			}
		}},
		{name: "speech context", fixture: "en/models-page-normal", steps: []historyDemoStep{{label: "Speech context", text: "Fixture context"}}, want: []string{"autosaveSpeechContext true true"}},
		{name: "speech context focus loss", fixture: "en/models-page-normal", steps: []historyDemoStep{{label: "Speech context", text: "Fixture context"}, {label: "Advanced"}}, want: []string{"autosaveSpeechContext true true", "autosaveSpeechContext false false"}},
	} {
		t.Run(test.name, func(t *testing.T) {
			host, err := loadMeetingDemoHost("testdata/meeting")
			if err != nil {
				t.Fatal(err)
			}
			selectMeetingDemoFixture(t, host, test.fixture)
			tester := newMeetingDemoTester(t, host)
			tester.SetSize(1000, 6000)
			frames(tester)
			for _, step := range test.steps {
				var err error
				switch {
				case step.text != "":
					if step.label == "Speech context" {
						caption, ok := tester.Find("Tell the speech model who is speaking and about what.")
						if !ok {
							t.Fatal("no Speech context caption")
						}
						tester.ClickAt(caption.X+10, caption.Y+caption.H+30)
					} else {
						err = tester.Click(step.label)
					}
					if err == nil {
						tester.Type(step.text)
					}
				case step.click:
					err = tester.Click(step.label)
				default:
					err = tester.ClickControl(step.label)
				}
				if err != nil {
					t.Fatalf("%s: %v (controls %q)", step.label, err, tester.AccessLabels())
				}
				frames(tester)
			}
			if !slices.Equal(host.calls, test.want) {
				t.Fatalf("calls %q, want %q", host.calls, test.want)
			}
			if test.check != nil {
				test.check(t, host)
			}
		})
	}
}

func TestModelsDemoDisabledControls(t *testing.T) {
	for _, test := range []struct {
		fixture string
		label   string
	}{
		{fixture: "en/models-download-elsewhere", label: "Download Qwen3-ASR 0.6B (4bit)"},
		{fixture: "en/models-deciding-delete", label: "Delete Qwen3-ASR 1.7B (8bit)…"},
	} {
		t.Run(test.fixture, func(t *testing.T) {
			host, err := loadMeetingDemoHost("testdata/meeting")
			if err != nil {
				t.Fatal(err)
			}
			selectMeetingDemoFixture(t, host, test.fixture)
			tester := newMeetingDemoTester(t, host)
			tester.SetSize(1000, 6000)
			frames(tester)
			if err := tester.ClickControl(test.label); err == nil {
				t.Fatalf("disabled %q accepted a click", test.label)
			}
			if len(host.calls) != 0 {
				t.Fatalf("disabled control recorded %v", host.calls)
			}
		})
	}
}
