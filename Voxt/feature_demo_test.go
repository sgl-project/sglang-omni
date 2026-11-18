package main

import (
	"slices"
	"testing"

	"github.com/egoist/mygo/ui"
)

func TestFeatureDemoActionControls(t *testing.T) {
	for _, test := range []struct {
		name    string
		fixture string
		prepare func(*meetingDemoHost)
		steps   []historyDemoStep
		want    []string
		check   func(*testing.T, *meetingDemoHost)
	}{
		{name: "add shortcut", fixture: "en/feature-normal", steps: historySteps("Add a shortcut"), want: []string{"addFeatureShortcut transcribe"}},
		{name: "permissions", fixture: "en/feature-no-accessibility", steps: historySteps("Open Permissions"), want: []string{"openFeaturePermissions"}},
		{name: "open shortcut editor", fixture: "en/feature-normal", steps: []historyDemoStep{{label: "Keys of shortcut 1"}, {key: ui.KeyK}}, want: []string{`openShortcutEditor "RightOption"`, "beginHotkeyRecord", "markFeatureKeysHeard", "recordFeatureKeys"}},
		{name: "type shortcut", fixture: "en/feature-normal", steps: []historyDemoStep{{label: "Keys of shortcut 1"}, {label: "Type instead"}, {label: "Keys of shortcut 1", click: true}, {command: "replace", text: "Control+Option+K"}, {key: ui.KeyEnter}}, want: []string{`openShortcutEditor "RightOption"`, "beginHotkeyRecord", "markFeatureKeysHeard", "cancelHotkeyRecord", `saveFeatureShortcutKeys transcribe 0 "Control+Option+K"`, "closeShortcutEditor"}},
		{name: "cancel shortcut editor", fixture: "en/feature-normal", steps: []historyDemoStep{{label: "Keys of shortcut 1"}, {label: "Cancel"}}, want: []string{`openShortcutEditor "RightOption"`, "beginHotkeyRecord", "markFeatureKeysHeard", "cancelHotkeyRecord", "closeShortcutEditor"}},
		{name: "custom shortcut", fixture: "en/feature-shortcut-custom-error", prepare: func(host *meetingDemoHost) {
			host.feature.local.rows[0].text = ""
		}, steps: []historyDemoStep{{label: "Keys of shortcut 1", text: "Control+Option+K"}, {key: ui.KeyEnter}}, want: []string{`saveFeatureShortcutKeys transcribe 0 "Control+Option+K"`}},
		{name: "record shortcut", fixture: "en/feature-shortcut-custom-error", steps: []historyDemoStep{{label: "Record instead"}, {key: ui.KeyK}, {label: "Your rules"}}, want: []string{"beginHotkeyRecord", "markFeatureKeysHeard", "recordFeatureKeys", "cancelHotkeyRecord"}},
		{name: "helper record shortcut", fixture: "en/feature-record-unknown", steps: []historyDemoStep{{key: ui.KeyK}}, want: []string{"beginHotkeyRecord", "markFeatureKeysHeard"}},
		{name: "record permissions", fixture: "en/feature-record-denied", prepare: func(host *meetingDemoHost) {
			host.feature.keysHeard = true
		}, steps: historySteps("Open Permissions"), want: []string{"cancelHotkeyRecord", "openFeaturePermissions"}},
		{name: "trigger", fixture: "en/feature-normal", steps: historySteps("Double tap"), want: []string{"setFeatureShortcutTrigger transcribe 0 1"}},
		{name: "remove shortcut", fixture: "en/feature-normal", steps: historySteps("Remove shortcut 1"), want: []string{"removeFeatureShortcut transcribe 0"}},
		{name: "paste off", fixture: "en/feature-normal", steps: historySteps("Clear shortcut"), want: []string{`saveFeaturePasteLast ""`}},
		{name: "custom paste", fixture: "en/feature-normal", prepare: func(host *meetingDemoHost) {
			host.feature.local.paste.custom, host.feature.local.paste.typing = true, true
			host.feature.local.paste.text = ""
		}, steps: []historyDemoStep{{label: "Keys to paste the last result", text: "Control+Option+B"}, {key: ui.KeyEnter}}, want: []string{`saveFeaturePasteLast "Control+Option+B"`}},
		{name: "cancel keys", fixture: "en/feature-normal", steps: []historyDemoStep{{label: "Keys to cancel recording"}, {key: ui.KeyEscape}}, want: []string{`openShortcutEditor "Escape"`, "beginHotkeyRecord", "markFeatureKeysHeard", "recordFeatureKeys"}},
		{name: "custom cancel", fixture: "en/feature-normal", prepare: func(host *meetingDemoHost) {
			host.feature.local.cancel.custom, host.feature.local.cancel.typing = true, true
			host.feature.local.cancel.text = ""
		}, steps: []historyDemoStep{{label: "Keys to cancel recording", text: "Control+Option+B"}, {key: ui.KeyEnter}}, want: []string{`saveFeatureCancelKeys "Control+Option+B"`}},
		{name: "end word", fixture: "en/feature-normal", steps: []historyDemoStep{{label: "Word that ends the recording"}, {label: "over", click: true}}, want: []string{`setFeatureEndWord "over"`}},
		{name: "custom end word", fixture: "en/feature-end-word-custom", prepare: func(host *meetingDemoHost) {
			host.feature.local.endWord.text = ""
			host.feature.local.endWord.edited = false
		}, steps: []historyDemoStep{{label: "Your word that ends the recording", text: "finish"}, {key: ui.KeyEnter}}, want: []string{`setFeatureEndWord "finish"`}},
		{name: "speech models", fixture: "en/feature-models-none", steps: historySteps("Open Speech models"), want: []string{"openFeatureSpeechModels"}},
		{name: "language models", fixture: "en/feature-models-none", steps: historySteps("Set up a language model"), want: []string{"openFeatureLanguageModels"}},
		{name: "speech model", fixture: "en/feature-normal", steps: []historyDemoStep{{label: "Speech model for Transcribe"}, {label: "Fixture speech model", click: true}}, want: []string{`selectFeatureModel transcribe asr "fixture-speech"`}},
		{name: "language model", fixture: "en/feature-normal", steps: []historyDemoStep{{label: "Language model for Transcribe"}, {label: "Fixture language model", click: true}}, want: []string{`selectFeatureModel transcribe llm "fixture-language"`}},
		{name: "fallback", fixture: "en/feature-models-fallback", steps: []historyDemoStep{{label: "Fallback language model for Translate"}, {label: "Fixture language model", click: true}}, want: []string{`selectFeatureFallback "fixture-language"`}},
		{name: "enhance", fixture: "en/feature-normal", steps: historySteps("Clean up with a language model"), want: []string{"setFeatureEnhance false"}},
		{name: "structure", fixture: "en/feature-normal", steps: historySteps("Structure"), want: []string{"setFeatureStructure true"}},
		{name: "transcribe cleanup", fixture: "en/feature-normal", steps: historySteps("Basic clean-up"), want: []string{"setFeatureCleanup false"}},
		{name: "translate cleanup", fixture: "en/feature-translate-normal", steps: historySteps("Basic clean-up"), want: []string{"setFeatureTranslateCleanup false"}},
		{name: "rewrite cleanup", fixture: "en/feature-rewrite-normal", steps: historySteps("Clean up the instruction"), want: []string{"setFeatureRewriteCleanup false"}},
		{name: "target", fixture: "en/feature-translate-normal", steps: []historyDemoStep{{label: "Target language for Translate"}, {label: "Japanese", click: true}}, want: []string{`selectFeatureTarget "ja"`}},
		{name: "target warning", fixture: "en/feature-translate-target-warning", steps: historySteps("Add {{TARGET_LANGUAGE}} to the prompt"), want: []string{"addFeatureTargetVariable"}},
		{name: "show card", fixture: "en/feature-translate-normal", steps: historySteps("Show the result in a card"), want: []string{"setFeatureShowCard translate true"}},
		{name: "card size", fixture: "en/feature-rewrite-normal", steps: historySteps("Medium"), want: []string{"setCardSize rewrite 1"}},
		{name: "schedule prompt", fixture: "en/feature-rules-preview", prepare: func(host *meetingDemoHost) {
			host.feature.local.prompt = ""
		}, steps: []historyDemoStep{{label: "System prompt for Transcribe", text: "Be brief."}}, want: []string{"scheduleFeaturePromptSave transcribe 1"}},
		{name: "save prompt", fixture: "en/feature-rules-preview", prepare: func(host *meetingDemoHost) {
			host.feature.local.promptDirty = true
		}, steps: []historyDemoStep{{label: "System prompt for Transcribe", click: true}, {label: "Your rules", click: true}}, want: []string{"savePrompt transcribe"}},
		{name: "reset prompt", fixture: "en/feature-translate-target-warning", steps: []historyDemoStep{{label: "Reset to default prompt", click: true}}, want: []string{"resetFeaturePrompt translate"}},
		{name: "copy variable", fixture: "en/feature-rules-preview", steps: historySteps("{{USER_MAIN_LANGUAGE}}"), want: []string{`copyFeatureText "{{USER_MAIN_LANGUAGE}}"`}},
		{name: "start", fixture: "en/feature-normal", steps: historySteps("Start a test"), want: []string{"startTry"}},
		{name: "stop", fixture: "en/feature-try-state-recording", steps: historySteps("Stop recording"), want: []string{"stopTry"}},
		{name: "cancel recording", fixture: "en/feature-try-state-recording", steps: historySteps("Cancel"), want: []string{"cancelTry"}},
		{name: "cancel working", fixture: "en/feature-loading", steps: historySteps("Cancel"), want: []string{"cancelTry"}},
		{name: "copy result", fixture: "en/feature-try-state-done", steps: historySteps("Copy result"), want: []string{`copyTryResult "The meeting moves to Thursday at four."`}},
		{name: "blocked action", fixture: "en/feature-blocked-start", steps: historySteps("Open Speech models"), want: []string{`runTryNoticeAction models`}},
		{name: "blocked dismiss", fixture: "en/feature-blocked-start", steps: historySteps("Dismiss"), want: []string{"clearTryNotice"}, check: func(t *testing.T, host *meetingDemoHost) {
			if host.feature.block != nil {
				t.Fatal("notice stayed open")
			}
		}},
		{name: "failed action", fixture: "en/feature-problem-noASR", steps: historySteps("Open Speech models"), want: []string{`runTryNoticeAction models`}},
	} {
		t.Run(test.name, func(t *testing.T) {
			host, err := loadMeetingDemoHost("testdata/meeting")
			if err != nil {
				t.Fatal(err)
			}
			selectMeetingDemoFixture(t, host, test.fixture)
			if test.prepare != nil {
				test.prepare(host)
			}
			tester := newMeetingDemoTester(t, host)
			for _, step := range test.steps {
				var err error
				switch {
				case step.command == "replace":
					tester.Command("selectAll")
					tester.Type(step.text)
				case step.context:
					err = tester.RightClick(step.label)
				case step.menu:
					err = tester.ChooseMenuItem(step.label)
				case step.text != "":
					err = tester.Click(step.label)
					if err == nil {
						tester.Type(step.text)
					}
				case step.command != "":
					tester.Command(step.command)
				case step.key != 0:
					tester.Key(step.mods, step.key)
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
