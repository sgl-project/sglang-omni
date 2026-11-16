package main

import (
	"testing"

	"github.com/egoist/mygo/ui"
)

func settingsDemoKeyRecorderCases() []settingsDemoCase {
	return []settingsDemoCase{
		{
			name: "key recorder focus and window chord", fixture: "en/key-recorder-window-recording",
			steps: []historyDemoStep{{key: ui.KeyTab}, {mods: ui.Shift, key: ui.KeyTab}, {mods: ui.Alt, key: ui.KeySpace}},
			want:  []string{"beginHotkeyRecord", "saveShortcutRecorderKeys Option+Space", "saveKeys Option+Space"},
			check: func(t *testing.T, host *meetingDemoHost) {
				if host.keyRow.text != "Option+Space" || host.keyRow.edited {
					t.Fatalf("window chord left recorder state %+v", host.keyRow)
				}
			},
		},
		{
			name: "key recorder helper focus and cancel", fixture: "en/key-recorder-helper-recording",
			steps: []historyDemoStep{{key: ui.KeyTab}, {mods: ui.Shift, key: ui.KeyTab}, {key: ui.KeyTab}},
			want:  []string{"beginHotkeyRecord", "cancelHotkeyRecord"},
			check: func(t *testing.T, host *meetingDemoHost) {
				if host.recorderActive {
					t.Fatal("helper recording stayed active after cancellation")
				}
			},
		},
		{
			name: "key recorder typed valid canonicalizes and saves", fixture: "en/key-recorder-typed-valid",
			prepare: func(host *meetingDemoHost) { host.keyRow.text = "" },
			steps:   []historyDemoStep{{label: "Custom keys of shortcut 1", text: "ctrl+opt+K"}, {key: ui.KeyEnter}},
			want:    []string{"typeShortcutKeys", "saveTypedShortcutKeys", "saveKeys Control+Option+K"},
			check: func(t *testing.T, host *meetingDemoHost) {
				if host.keyRow.text != "Control+Option+K" || host.keyRow.edited {
					t.Fatalf("valid typed keys left recorder state %+v", host.keyRow)
				}
			},
		},
		{
			name: "key recorder typed invalid shows error without saving", fixture: "en/key-recorder-typed-invalid",
			prepare:  func(host *meetingDemoHost) { host.keyRow.text = "" },
			steps:    []historyDemoStep{{label: "Custom keys of shortcut 1", text: "ctrl+opt+K+J"}, {key: ui.KeyEnter}},
			want:     []string{"typeShortcutKeys"},
			wantText: "Only one key besides the modifiers.",
			check: func(t *testing.T, host *meetingDemoHost) {
				if host.keyRow.text != "ctrl+opt+K+J" || !host.keyRow.edited {
					t.Fatalf("invalid typed keys left recorder state %+v", host.keyRow)
				}
			},
		},
		{
			name: "key recorder permission denied opens permissions", fixture: "en/key-recorder-permission-denied",
			steps: historySteps("Open Permissions"), want: []string{"openShortcutPermissions"},
			wantText: "Recording a shortcut needs Accessibility access.",
		},
		{
			name: "key recorder unknown key does not save", fixture: "en/key-recorder-window-recording",
			steps:    []historyDemoStep{{key: ui.KeyContextMenu}},
			wantText: "Voxt cannot use that key. Press another, or type the keys instead.",
			check: func(t *testing.T, host *meetingDemoHost) {
				if host.keyRow.text != "Control+Option+Space" || !host.keyRow.unknown {
					t.Fatalf("unknown key left recorder state %+v", host.keyRow)
				}
			},
		},
		{
			name: "key recorder local press and type transitions", fixture: "en/key-recorder-typed-valid",
			steps: historySteps("Press the keys instead", "Type the keys instead"),
			check: func(t *testing.T, host *meetingDemoHost) {
				if !host.keyRow.typing || host.keyRow.edited {
					t.Fatalf("local transitions left recorder state %+v", host.keyRow)
				}
			},
		},
	}
}

func settingsDemoControlCases() []settingsDemoCase {
	return []settingsDemoCase{
		{name: "overlay position", fixture: "en/settings-overlay-normal", steps: historySteps("Top centre"), want: []string{"overlayEdited"}},
		{name: "overlay margin slider", fixture: "en/settings-overlay-normal", scroll: "Distance from the edge", slider: "96", steps: []historyDemoStep{{key: ui.KeyRight}}, want: []string{"overlayEdited", "overlayEdited"}},
		{name: "overlay opacity slider", fixture: "en/settings-overlay-normal", scroll: "Opacity", slider: "80%", steps: []historyDemoStep{{key: ui.KeyRight}}, want: []string{"overlayEdited", "overlayEdited"}},
		{name: "overlay radius slider", fixture: "en/settings-overlay-normal", scroll: "Corner radius", slider: "24", steps: []historyDemoStep{{key: ui.KeyRight}}, want: []string{"overlayEdited", "overlayEdited"}},
		{name: "overlay recording time", fixture: "en/settings-overlay-normal", scroll: "Show recording time", steps: historySteps("Show recording time"), want: []string{"overlayEdited"}},
		{name: "overlay live text", fixture: "en/settings-overlay-normal", scroll: "Show live text", steps: historySteps("Show live text"), want: []string{"overlayEdited"}},
		{name: "overlay live layout", fixture: "en/settings-overlay-normal", scroll: "Live text layout", steps: historySteps("Lines"), want: []string{"overlayEdited"}},
		{name: "shortcut permissions", fixture: "en/settings-shortcuts-duplicate-problem-strict-fn-no-accessibility", steps: historySteps("Open Permissions"), want: []string{"openShortcutPermissions"}},
		{name: "shortcut preset", fixture: "en/settings-shortcuts-normal", steps: settingsDemoSelect("Shortcut preset", ui.KeyDown), want: []string{"applyShortcutPreset fn"}},
		{name: "strict shortcuts", fixture: "en/settings-shortcuts-normal", steps: historySteps("Strict shortcuts"), want: []string{"setStrictShortcuts true"}},
		{name: "transcribe shortcut", fixture: "en/settings-shortcuts-normal", scroll: "Transcribe Right ⌥", steps: historySteps("Transcribe Right ⌥"), want: []string{"openShortcutFeature /transcribe"}},
		{name: "translate shortcut", fixture: "en/settings-shortcuts-normal", scroll: "Translate Right ⌥ ⇧", steps: historySteps("Translate Right ⌥ ⇧"), want: []string{"openShortcutFeature /translate"}},
		{name: "rewrite shortcut", fixture: "en/settings-shortcuts-normal", scroll: "Rewrite ⌃ Right ⌥", steps: historySteps("Rewrite ⌃ Right ⌥"), want: []string{"openShortcutFeature /rewrite"}},
		{name: "paste shortcut", fixture: "en/settings-shortcuts-normal", scroll: "Paste the last result ⌃⌥V", steps: historySteps("Paste the last result ⌃⌥V"), want: []string{"openShortcutFeature /transcribe"}},
		{name: "cancel shortcut", fixture: "en/settings-shortcuts-normal", scroll: "Cancel recording Esc", steps: historySteps("Cancel recording Esc"), want: []string{"openShortcutFeature /transcribe"}},
		{name: "notes shortcut", fixture: "en/settings-shortcuts-normal", scroll: "Notes Right ⌘", steps: historySteps("Notes Right ⌘"), want: []string{"openShortcutFeature /notes"}},
		{name: "send window", fixture: "en/settings-privacy-normal", scroll: "Send a screenshot", steps: historySteps("Send the window to the language model"), want: []string{"setSendWindow false"}},
		{name: "send clipboard", fixture: "en/settings-privacy-normal", scroll: "Send a copied image", steps: historySteps("Send the copied image with Rewrite"), want: []string{"setSendClipboardImage false"}},
		{name: "keep history", fixture: "en/settings-privacy-normal", scroll: "Keep a history", steps: historySteps("Save each result"), want: []string{"setKeepHistory false"}},
		{name: "keep audio", fixture: "en/settings-privacy-normal", scroll: "Keep recordings", steps: historySteps("Save each recording"), want: []string{"setKeepAudio false"}},
		{name: "privacy history settings", fixture: "en/settings-privacy-normal", scroll: "History settings", steps: historySteps("Open…"), want: []string{"showHistorySettings"}},
		{name: "clear history", fixture: "en/settings-privacy-normal", scroll: "Clear history", steps: historySteps("Clear…", "Delete history"), want: []string{"clearHistory"}, check: func(t *testing.T, host *meetingDemoHost) {
			if host.settingsForm.clearHistoryAsk {
				t.Fatal("clear confirmation stayed open")
			}
		}},
		{name: "clear seeded confirmation", fixture: "en/settings-privacy-clear-confirmation", steps: historySteps("Delete history"), want: []string{"clearHistory"}},
		{name: "clear history cancel", fixture: "en/settings-privacy-normal", scroll: "Clear history", steps: historySteps("Clear…", "Cancel"), check: func(t *testing.T, host *meetingDemoHost) {
			if host.settingsForm.clearHistoryAsk {
				t.Fatal("clear confirmation stayed open")
			}
		}},
		{name: "log text", fixture: "en/settings-privacy-normal", scroll: "Write text to the log", steps: historySteps("Log what was said"), want: []string{"setLogText true"}},
		{name: "privacy access off", fixture: "en/settings-privacy-normal", scroll: "Allow reading", steps: historySteps("Allow reading"), want: []string{"setHistoryAccess false"}},
		{name: "privacy access selection", fixture: "en/settings-privacy-normal", scroll: "Include selected text", steps: historySteps("Include selected text"), want: []string{"setHistorySelection false"}},
		{name: "privacy access log", fixture: "en/settings-privacy-normal", scroll: "Read log…", steps: historySteps("Read log…", "Done"), check: func(t *testing.T, host *meetingDemoHost) {
			if host.historyAccessUI.listOpen {
				t.Fatal("read log stayed open")
			}
		}},
		{name: "privacy access copy", fixture: "en/settings-privacy-normal", scroll: "Command line", steps: historySteps("Copy"), want: []string{"copyHistoryAccessCommand"}},
		{name: "privacy access docs", fixture: "en/settings-privacy-normal", scroll: "Command line", steps: historySteps("How to use it"), want: []string{"openHistoryAccessDocs"}},
		{name: "privacy access on", fixture: "en/settings-privacy-offline-disabled", scroll: "Allow reading", steps: historySteps("Allow reading", "Turn on"), want: []string{"setHistoryAccess true"}, check: func(t *testing.T, host *meetingDemoHost) {
			if host.historyAccessUI.ask {
				t.Fatal("access confirmation stayed open")
			}
		}},
		{name: "privacy access cancel", fixture: "en/settings-privacy-offline-disabled", scroll: "Allow reading", steps: historySteps("Allow reading", "Cancel"), check: func(t *testing.T, host *meetingDemoHost) {
			if host.historyAccessUI.ask || host.historyAccessUI.historyAccess {
				t.Fatal("access cancellation changed local state")
			}
		}},
	}
}
