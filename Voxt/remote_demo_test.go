package main

import (
	"slices"
	"testing"

	"github.com/egoist/mygo/ui"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/local"
)

type remoteDemoStep struct {
	label string
	text  string
	click bool
	at    func(*ui.Tester)
	mods  ui.Modifiers
	key   ui.Key
}

func TestRemoteDemoActionControls(t *testing.T) {
	for _, test := range []struct {
		name    string
		fixture string
		steps   []remoteDemoStep
		want    []string
		check   func(*testing.T, *meetingDemoHost, *ui.Tester)
	}{
		{name: "open provider", fixture: "en/remote-page-normal", steps: remoteSteps(remoteDemoStep{label: "Fixture service", click: true}), want: []string{"openRemoteProvider llm fixture"}},
		{name: "test provider", fixture: "en/remote-page-normal", steps: remoteSteps(remoteDemoStep{label: "Test Fixture service"}), want: []string{"testRemoteProvider llm fixture"}},
		{name: "complete provider", fixture: "en/remote-missing-key", steps: remoteSteps(remoteDemoStep{label: "Complete setup of Fixture service"}), want: []string{"openRemoteProvider llm fixture"}},
		{name: "open debug", fixture: "en/remote-page-normal", steps: remoteSteps(remoteDemoStep{label: "Try prompts on a model…"}), want: []string{"openLLMDebug"}},
		{name: "open onboarding", fixture: "en/remote-onboarding-back", steps: remoteSteps(remoteDemoStep{label: "Back to the setup guide"}), want: []string{"openOnboarding"}},
		{name: "dismiss onboarding", fixture: "en/remote-onboarding-back", steps: remoteSteps(remoteDemoStep{label: "Dismiss"}), want: []string{"dismissOnboardingBack"}},
		{name: "wait preset", fixture: "en/remote-wait-automatic", steps: remoteSteps(remoteDemoStep{label: "Automatic", click: true}, remoteDemoStep{label: "5 minutes", click: true}), want: []string{`chooseLLMWait "5m" false 0`}},
		{name: "wait custom", fixture: "en/remote-wait-automatic", steps: remoteSteps(remoteDemoStep{label: "Automatic", click: true}, remoteDemoStep{label: "Custom…", click: true}), want: []string{`chooseLLMWait "" true 1`}},
		{name: "wait minutes", fixture: "en/remote-wait-custom", steps: remoteSteps(remoteDemoStep{label: "Wait for the language model in minutes", click: true}, remoteDemoStep{key: ui.KeyUp}), want: []string{"changeLLMWaitMinutes 46"}},
		{name: "test dialog", fixture: "en/remote-dialog-normal", steps: remoteSteps(remoteDemoStep{label: "Test connection"}), want: []string{"testDialog"}},
		{name: "save dialog", fixture: "en/remote-dialog-normal", steps: remoteSteps(remoteDemoStep{label: "Save"}), want: []string{"saveProviderDialog"}},
		{name: "remove dialog", fixture: "en/remote-dialog-normal", steps: remoteSteps(remoteDemoStep{label: "Remove Fixture service"}), want: []string{"removeProvider"}},
		{name: "cancel dialog", fixture: "en/remote-dialog-normal", steps: remoteSteps(remoteDemoStep{label: "Cancel"}), check: remoteDialogClosed},
		{name: "spaced key dismissal", fixture: "en/remote-dialog-key-spaced", steps: remoteSteps(remoteDemoStep{key: ui.KeyEscape}), check: remoteDialogOpen},
		{name: "remove key dismissal", fixture: "en/remote-dialog-remove-key", steps: remoteSteps(remoteDemoStep{key: ui.KeyEscape}), check: remoteDialogOpen},
		{name: "show key", fixture: "en/remote-dialog-key-spaced", steps: remoteSteps(remoteDemoStep{label: "Show or hide the key"}), check: func(t *testing.T, host *meetingDemoHost, tester *ui.Tester) {
			if !host.remoteDialogUI.showKey {
				t.Fatal("key stayed hidden")
			}
		}},
		{name: "remove saved key", fixture: "en/remote-dialog-normal", steps: remoteSteps(remoteDemoStep{label: "Remove saved key"}), check: func(t *testing.T, host *meetingDemoHost, tester *ui.Tester) {
			if !host.remoteDialogUI.removeKey {
				t.Fatal("saved key was not marked for removal")
			}
		}},
		{name: "key url", fixture: "en/remote-dialog-normal", steps: remoteSteps(remoteDemoStep{label: "Get a key"}), want: []string{"openRemoteKeyURL"}},
		{name: "generation thinking", fixture: "en/remote-generation-think-on", steps: remoteSteps(remoteDemoStep{label: "Thinking on", click: true}, remoteDemoStep{label: "Effort", click: true}), check: func(t *testing.T, host *meetingDemoHost, tester *ui.Tester) {
			if host.remoteDialogUI.gen.think != "Effort" {
				t.Fatalf("thinking selection %q", host.remoteDialogUI.gen.think)
			}
		}},
		{name: "generation responses", fixture: "en/remote-generation-api-chat-completions", steps: remoteSteps(remoteDemoStep{label: "API format"}, remoteDemoStep{label: "Responses", click: true}), check: func(t *testing.T, host *meetingDemoHost, tester *ui.Tester) {
			if host.remoteDialogUI.gen.api != "Responses" || tester.HasText("Stop sequences") {
				t.Fatal("Responses did not hide its unsupported fields")
			}
		}},
		{name: "generation collapse", fixture: "en/remote-dialog-normal", steps: remoteSteps(remoteDemoStep{label: "Advanced", click: true}), check: func(t *testing.T, host *meetingDemoHost, tester *ui.Tester) {
			if host.remoteDialogUI.gen.advanced {
				t.Fatal("Advanced stayed open")
			}
		}},
		{name: "dismiss dialog error", fixture: "en/remote-dialog-error", steps: remoteSteps(remoteDemoStep{at: func(tester *ui.Tester) {
			row, rowOK := tester.Find("The remote model's address is not allowed. Use https://, or http:// for an address on this Mac.")
			close, closeOK := tester.Find("Dismiss")
			if rowOK && closeOK {
				tester.ClickAt(close.X+close.W/2, row.Y+row.H/2)
			}
		}}), check: func(t *testing.T, host *meetingDemoHost, tester *ui.Tester) {
			if host.remoteDialogUI.err != "" {
				t.Fatal("dialog error stayed visible")
			}
		}},
		{name: "disabled running test", fixture: "en/remote-dialog-loading", steps: remoteSteps(remoteDemoStep{label: "Test connection", click: true}), check: func(t *testing.T, host *meetingDemoHost, tester *ui.Tester) {
			if !tester.HasText("Testing…") {
				t.Fatal("running test state is not visible")
			}
		}},
		{name: "disabled saving", fixture: "en/remote-dialog-saving", steps: remoteSteps(remoteDemoStep{label: "Save", click: true}, remoteDemoStep{label: "Remove Fixture service", click: true}), check: func(t *testing.T, host *meetingDemoHost, tester *ui.Tester) {
			if !tester.HasText("Save") {
				t.Fatal("saving state is not visible")
			}
		}},
		{name: "disabled signed out test", fixture: "en/remote-chatgpt-signin", steps: remoteSteps(remoteDemoStep{label: "Test connection", click: true})},
		{name: "chat sign in", fixture: "en/remote-chatgpt-signin", steps: remoteSteps(remoteDemoStep{label: "Sign in with ChatGPT"}), want: []string{"signInChatGPT"}},
		{name: "chat cancel", fixture: "en/remote-chatgpt-running", steps: remoteSteps(remoteDemoStep{label: "Cancel sign-in"}), want: []string{"cancelChatGPTLogin"}},
		{name: "chat usage", fixture: "en/remote-chatgpt-signed-in", steps: remoteSteps(remoteDemoStep{label: "Manage usage"}), want: []string{"openChatGPTUsage"}},
		{name: "chat sign out", fixture: "en/remote-chatgpt-signed-in", steps: remoteSteps(remoteDemoStep{label: "Sign out of ChatGPT"}), want: []string{"signOutChatGPT"}},
		{name: "chat dismiss error", fixture: "en/remote-chatgpt-problem-denied", steps: remoteSteps(remoteDemoStep{label: "Dismiss"}), want: []string{"dismissChatGPTError"}},
		{name: "debug close", fixture: "en/remote-debug-empty", steps: remoteSteps(remoteDemoStep{label: "Close"}), want: []string{"openRemoteLanguageModels"}},
		{name: "debug open models", fixture: "en/remote-debug-empty", steps: remoteSteps(remoteDemoStep{label: "Open Language models"}), want: []string{"openRemoteLanguageModels"}},
		{name: "debug preset", fixture: "en/remote-debug-transcribe", steps: remoteSteps(remoteDemoStep{label: "Request", click: true}, remoteDemoStep{label: "Rewrite", click: true}), check: func(t *testing.T, host *meetingDemoHost, tester *ui.Tester) {
			saved := ""
			for _, preset := range host.current().state.RemoteDebug.PresetStates {
				if preset.Title == "Rewrite" {
					saved = preset.SavedPrompt
				}
			}
			if host.remoteDebugUI.filled != "Rewrite" || saved == "" || host.remoteDebugUI.prompt != saved || !tester.HasText("Spoken instruction") {
				t.Fatalf("preset did not rebuild the debug state: %+v", host.remoteDebugUI)
			}
		}},
		{name: "debug run", fixture: "en/remote-debug-transcribe", steps: remoteSteps(remoteDemoStep{label: "Run"}), want: []string{"runRemoteDebug"}},
		{name: "debug use prompt", fixture: "en/remote-debug-transcribe", steps: remoteSteps(remoteDemoStep{label: "Use this prompt"}), want: []string{"useRemoteDebugPrompt"}},
		{name: "debug reset prompt", fixture: "en/remote-debug-transcribe", steps: remoteSteps(remoteDemoStep{label: "Back to the saved prompt"}), check: func(t *testing.T, host *meetingDemoHost, tester *ui.Tester) {
			if host.remoteDebugUI.prompt != remoteDebugSavedPrompt(host) {
				t.Fatal("prompt was not reset")
			}
		}},
		{name: "debug dismiss error", fixture: "en/remote-debug-error", steps: remoteSteps(remoteDemoStep{label: "Dismiss"}), want: []string{"dismissLLMDebugResult"}},
	} {
		t.Run(test.name, func(t *testing.T) {
			host, err := loadMeetingDemoHost("testdata/meeting")
			if err != nil {
				t.Fatal(err)
			}
			selectMeetingDemoFixture(t, host, test.fixture)
			tester := newMeetingDemoTester(t, host)
			for _, step := range test.steps {
				var err error
				switch {
				case step.at != nil:
					step.at(tester)
				case step.text != "":
					err = tester.Click(step.label)
					if err == nil {
						tester.Type(step.text)
					}
				case step.click:
					err = tester.Click(step.label)
				case step.key != 0:
					tester.Key(step.mods, step.key)
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
				test.check(t, host, tester)
			}
		})
	}
}

func TestRemoteDemoDismissalUsesSavedSetup(t *testing.T) {
	for _, test := range []struct {
		name      string
		speech    bool
		showAppID bool
		edit      func(*providerDialogLocalUI)
	}{
		{name: "App ID", showAppID: true, edit: func(local *providerDialogLocalUI) { local.appID = "fixture-app" }},
		{name: "model", edit: func(local *providerDialogLocalUI) { local.model = "other-model" }},
		{name: "custom model", edit: func(local *providerDialogLocalUI) {
			local.model, local.customModel = "Other model…", "custom-model"
		}},
		{name: "recognition prompt", speech: true, edit: func(local *providerDialogLocalUI) { local.recognitionPrompt = "changed prompt" }},
		{name: "generation", edit: func(local *providerDialogLocalUI) { local.gen.temperature = "0.7" }},
	} {
		for _, revert := range []bool{false, true} {
			name := test.name + "/edit"
			if revert {
				name = test.name + "/revert"
			}
			t.Run(name, func(t *testing.T) {
				kind := local.LLM
				if test.speech {
					kind = local.ASR
				}
				input := remoteFixtureDialogInput(kind)
				if test.showAppID {
					input.Service.appID = true
				}
				original := input.Dialog.providerDialogLocalUI
				if revert {
					test.edit(&input.Dialog.providerDialogLocalUI)
					input.Unsaved = input.Dialog.unsaved()
				}
				state := buildRemoteDialogViewState(i18n.NewTranslator("en"), *input)
				host := &meetingDemoHost{fixtures: []meetingDemoFixture{{state: meetingFixtureState{RemoteDialog: &state}}}}
				host.remoteDialogUI = remoteDialogLocalState(state)
				tester := ui.NewTester(func(c *ui.Context) {
					if revert {
						open := host.remoteDialogUI.open
						host.remoteDialogUI = original
						host.remoteDialogUI.open = open
					} else {
						test.edit(&host.remoteDialogUI)
					}
					remoteProviderDialogView(c, &state, host, &host.remoteDialogUI)
				}, 1000, 1000)
				tester.Key(0, ui.KeyEscape)
				frames(tester)
				wantOpen := !revert
				if host.remoteDialogUI.open != wantOpen {
					t.Fatalf("dialog open %t, want %t", host.remoteDialogUI.open, wantOpen)
				}
			})
		}
	}
}

func TestRemoteDemoDismissalNormalizesSavedGeneration(t *testing.T) {
	host, err := loadMeetingDemoHost("testdata/meeting")
	if err != nil {
		t.Fatal(err)
	}
	selectMeetingDemoFixture(t, host, "en/remote-dialog-unsaved")
	state := host.current().state.RemoteDialog
	if state.SavedForm == nil || state.SavedForm.Generation.Think != "on" || state.Form.Generation.Think != "Thinking on" {
		t.Fatalf("fixture generation domains changed: saved %+v, live %+v", state.SavedForm, state.Form)
	}
	host.remoteDialogUI.endpoint = state.SavedForm.Endpoint
	tester := newMeetingDemoTester(t, host)
	tester.Key(0, ui.KeyEscape)
	frames(tester)
	remoteDialogClosed(t, host, tester)
}

func remoteSteps(steps ...remoteDemoStep) []remoteDemoStep { return steps }

func remoteDebugSavedPrompt(host *meetingDemoHost) string {
	for _, preset := range host.current().state.RemoteDebug.PresetStates {
		if preset.Title == host.remoteDebugUI.preset {
			return preset.SavedPrompt
		}
	}
	return ""
}

func remoteDialogOpen(t *testing.T, host *meetingDemoHost, tester *ui.Tester) {
	t.Helper()
	if !host.remoteDialogUI.open {
		t.Fatal("Escape dismissed a dialog with an unsaved key edit")
	}
	if !host.current().state.RemoteDialog.Unsaved {
		t.Fatal("fixture dialog state disagrees with its key edit")
	}
}

func remoteDialogClosed(t *testing.T, host *meetingDemoHost, tester *ui.Tester) {
	if host.remoteDialogUI.open {
		t.Fatal("remote dialog stayed open")
	}
}
