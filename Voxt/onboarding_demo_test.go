package main

import (
	"slices"
	"testing"

	"github.com/egoist/mygo/ui"
	"github.com/sgl-project/sglang-omni/Voxt/internal/helper/proto"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/models"
)

func TestOnboardingDemoActionControls(t *testing.T) {
	for _, test := range []struct {
		name    string
		fixture string
		steps   []historyDemoStep
		want    []string
	}{
		{name: "get started", fixture: "en/step-welcome", steps: historySteps("Get started"), want: []string{"obGo 1"}},
		{name: "set up later", fixture: "en/step-welcome", steps: historySteps("Set up later"), want: []string{"closeOnboarding"}},
		{name: "close", fixture: "en/step-welcome", steps: historySteps("Close guide"), want: []string{"closeOnboarding"}},
		{name: "legacy import", fixture: "en/legacy-available", steps: historySteps("Import from Voxt 1.x"), want: []string{"importLegacy"}},
		{name: "legacy import again", fixture: "en/legacy-imported", steps: historySteps("Import again"), want: []string{"importLegacy"}},
		{name: "legacy complete", fixture: "en/legacy-imported", steps: historySteps("Complete setup of " + models.ASRAccurate), want: []string{"completeOnboardingLegacyTodo asr " + models.ASRAccurate}},
		{name: "permissions required", fixture: "en/step-permissions", steps: historySteps("Ask for the required permissions"), want: []string{"askRequired"}},
		{name: "permissions recheck", fixture: "en/step-permissions", steps: historySteps("Check again"), want: []string{"recheckPermissions"}},
		{name: "models choose", fixture: "en/models-loading", steps: historySteps("Choose another model…"), want: []string{"obLeave " + modelsPath}},
		{name: "models language setup", fixture: "en/models-loading", steps: historySteps("Set up a language model"), want: []string{"obLeave " + remotePath}},
		{name: "models download", fixture: "en/models-loading", steps: historySteps("Download and continue (2.5 GB)"), want: []string{"onboardingDownloadAndContinue"}},
		{name: "models skip", fixture: "en/models-loading", steps: historySteps("Skip this step"), want: []string{"obGo 3"}},
		{name: "models reuse continue", fixture: "en/models-reuse", steps: historySteps("Continue"), want: []string{"obGo 3"}},
		{name: "models reuse use", fixture: "en/models-reuse", steps: historySteps("Use them"), want: []string{"obUse " + models.ASRAccurate + "," + models.VADSilero}},
		{name: "models reuse choose", fixture: "en/models-reuse", steps: historySteps("Choose models…"), want: []string{"obLeave " + modelsPath}},
		{name: "models vad download", fixture: "en/models-vad-only", steps: historySteps("Download (2.5 GB)"), want: []string{"obDownload"}},
		{name: "models alternative download", fixture: "en/models-alternative", steps: historySteps("Download " + onboardingModelName(models.ASRAccurate)), want: []string{"obDownload"}},
		{name: "models alternative use", fixture: "en/models-alternative", steps: historySteps("Use " + onboardingModelName(models.ASRFast)), want: []string{"obUse " + models.ASRFast}},
		{name: "models downloading continue", fixture: "en/models-downloading", steps: historySteps("Continue"), want: []string{"obGo 3"}},
		{name: "try start", fixture: "en/step-transcribe", steps: historySteps("Start a test"), want: []string{`obTry ""`}},
		{name: "try translate sample", fixture: "en/step-translate", steps: historySteps("Translate this text"), want: []string{`obTry "Let's go camping in the mountains this weekend; remember to bring a warm jacket and a torch."`}},
		{name: "try stop", fixture: "en/session-recording", steps: historySteps("Stop recording"), want: []string{"stopSession"}},
		{name: "try recording cancel", fixture: "en/session-recording", steps: historySteps("Cancel"), want: []string{"cancelSession"}},
		{name: "try working cancel", fixture: "en/session-composing", steps: historySteps("Cancel"), want: []string{"cancelSession"}},
		{name: "try again", fixture: "en/session-done", steps: historySteps("Start a test"), want: []string{`obTry ""`}},
		{name: "try copy", fixture: "en/session-done", steps: historySteps("Copy result"), want: []string{`copyOnboardingResult "Fixture cleaned result"`}},
		{name: "try blocked action", fixture: "en/try-blocked", steps: historySteps("Open Speech models"), want: []string{"onboardingNoticeAction models"}},
		{name: "try blocked dismiss", fixture: "en/try-blocked", steps: historySteps("Dismiss"), want: []string{"clearOnboardingBlock"}},
		{name: "try language setup", fixture: "en/try-language-model", steps: historySteps("Set up a language model"), want: []string{"onboardingTryLanguage"}},
		{name: "try failure action", fixture: "en/problem-noASR", steps: historySteps("Open Speech models"), want: []string{"onboardingNoticeAction models"}},
		{name: "try microphone action", fixture: "en/problem-microphone", steps: historySteps("Open Permissions"), want: []string{"onboardingNoticeAction permissions"}},
		{name: "try helper action", fixture: "en/problem-helper", steps: historySteps("Restart helper"), want: []string{"onboardingNoticeAction helper"}},
		{name: "try language action", fixture: "en/problem-noLLM", steps: historySteps("Open Language models"), want: []string{"onboardingNoticeAction language"}},
		{name: "try speech wait action", fixture: "en/problem-speechWait", steps: historySteps("Open Speech models"), want: []string{"onboardingNoticeAction models"}},
		{name: "try skip", fixture: "en/step-transcribe", steps: historySteps("Skip this step"), want: []string{"obGo 4"}},
		{name: "more app enhancement", fixture: "en/step-more", steps: historySteps("Open App enhancement"), want: []string{"finishOnboarding /app-enhancement"}},
		{name: "more dictionary", fixture: "en/step-more", steps: historySteps("Open Dictionary"), want: []string{"finishOnboarding /dictionary"}},
		{name: "more preset", fixture: "en/step-more", steps: []historyDemoStep{{label: "Shortcut preset", click: true}, {label: "fn (Auralink's keys)", click: true}}, want: []string{"applyShortcutPreset fn"}},
		{name: "more finish", fixture: "en/step-more", steps: historySteps("Start using Voxt"), want: []string{"finishOnboarding /home"}},
	} {
		t.Run(test.name, func(t *testing.T) {
			host, tester := onboardingDemoTester(t, test.fixture)
			for _, step := range test.steps {
				ensureOnboardingDemoVisible(t, tester, step.label)
				var err error
				if step.click {
					err = tester.Click(step.label)
				} else {
					err = tester.Click(step.label)
				}
				if err != nil {
					t.Fatalf("%s: %v (controls %q)", step.label, err, tester.AccessLabels())
				}
				frames(tester)
			}
			if !slices.Equal(host.calls, test.want) {
				t.Fatalf("calls %q, want %q", host.calls, test.want)
			}
		})
	}
}

func TestOnboardingDemoNavigationControls(t *testing.T) {
	for _, test := range []struct {
		name, label, want string
		backBar           bool
	}{
		{name: "back to guide", label: "Back to the setup guide", want: "openOnboarding", backBar: true},
		{name: "dismiss back bar", label: "Dismiss", want: "dismissOnboardingBack", backBar: true},
		{name: "settings row", label: "Open setup guide", want: "openOnboarding"},
	} {
		t.Run(test.name, func(t *testing.T) {
			host, err := loadMeetingDemoHost("testdata/meeting")
			if err != nil {
				t.Fatal(err)
			}
			tester := ui.NewTester(func(c *ui.Context) {
				if test.backBar {
					state := buildOnboardingBackViewState(i18n.NewTranslator("en"), false, remotePath, remotePath, obModels)
					onboardingBackBar(c, state, host)
					return
				}
				onboardingRow(c, host)
			}, 800, 300)
			tester.SetLocale("en")
			frames(tester)
			if err := tester.Click(test.label); err != nil {
				t.Fatal(err)
			}
			if !slices.Equal(host.calls, []string{test.want}) {
				t.Fatalf("calls %q, want %q", host.calls, test.want)
			}
		})
	}
}

func TestOnboardingDemoPermissionControls(t *testing.T) {
	for _, test := range []struct {
		name, row, control string
		want               string
	}{
		{name: "microphone ask", row: "Microphone", control: "Ask", want: "askPermission " + proto.PermissionMicrophone},
		{name: "microphone open", row: "Microphone", control: "Open System Settings", want: "onboardingOpenPrivacy " + privacyMicrophone},
		{name: "accessibility ask", row: "Accessibility", control: "Ask", want: "askPermission " + proto.PermissionAccessibility},
		{name: "accessibility open", row: "Accessibility", control: "Open System Settings", want: "onboardingOpenPrivacy " + privacyAccessibility},
		{name: "system audio ask", row: "System audio recording", control: "Ask", want: "askPermission " + proto.PermissionSystemAudio},
		{name: "system audio open", row: "System audio recording", control: "Open System Settings", want: "onboardingOpenPrivacy " + privacyAudioCapture},
		{name: "speech ask", row: "Speech Recognition", control: "Ask", want: "askPermission " + proto.PermissionSpeechRecognition},
		{name: "speech open", row: "Speech Recognition", control: "Open System Settings", want: "onboardingOpenPrivacy " + privacySpeech},
	} {
		t.Run(test.name, func(t *testing.T) {
			host, tester := onboardingDemoTester(t, "en/step-permissions")
			clickOnboardingRowControl(t, tester, test.row, test.control)
			if !slices.Equal(host.calls, []string{test.want}) {
				t.Fatalf("calls %q, want %q", host.calls, test.want)
			}
		})
	}
}

func TestOnboardingDemoBackControl(t *testing.T) {
	host, tester := onboardingDemoTester(t, "en/step-transcribe")
	ensureOnboardingDemoVisible(t, tester, "Back")
	if err := tester.Click("Back"); err != nil {
		t.Fatal(err)
	}
	frames(tester)
	if !slices.Equal(host.calls, []string{"obGo 2"}) {
		t.Fatalf("calls %q, want obGo 2", host.calls)
	}
}

func onboardingDemoTester(t *testing.T, fixture string) (*meetingDemoHost, *ui.Tester) {
	t.Helper()
	host, err := loadMeetingDemoHost("testdata/meeting")
	if err != nil {
		t.Fatal(err)
	}
	selectMeetingDemoFixture(t, host, fixture)
	tester := ui.NewTester(func(c *ui.Context) {
		onboardingView(c, *host.current().state.Onboarding, host, &host.onboardingUI)
	}, 1000, 700)
	tester.SetLocale(host.current().locale)
	frames(tester)
	return host, tester
}

func ensureOnboardingDemoVisible(t *testing.T, tester *ui.Tester, label string) {
	t.Helper()
	for range 8 {
		if bounds, ok := tester.Find(label); ok && bounds.W > 0 && bounds.H > 0 && bounds.Y >= 0 && bounds.Y+bounds.H <= 570 {
			return
		}
		tester.Scroll(500, 400, 0, 300)
		frames(tester)
	}
	t.Fatalf("%q is not visible: %q", label, tester.Texts())
}

func clickOnboardingRowControl(t *testing.T, tester *ui.Tester, row, control string) {
	t.Helper()
	for range 5 {
		if _, ok := tester.Find(row); ok {
			break
		}
		tester.Scroll(500, 400, 0, 400)
		frames(tester)
	}
	title, ok := tester.Find(row)
	if !ok {
		t.Fatalf("row %q is not visible: %q", row, tester.Texts())
	}
	if title.Y > 450 {
		tester.Scroll(500, 400, 0, 200)
		frames(tester)
		title, ok = tester.Find(row)
		if !ok {
			t.Fatalf("row %q disappeared after scrolling: %q", row, tester.Texts())
		}
	}
	button, ok := tester.Find(control)
	if !ok {
		t.Fatalf("control %q is not visible: %q", control, tester.AccessLabels())
	}
	baseTitle, ok := tester.Find("Microphone")
	if !ok {
		t.Fatal("Microphone is not visible")
	}
	tester.ClickAt(button.X+button.W/2, button.Y+button.H/2+title.Y-baseTitle.Y)
	frames(tester)
}
