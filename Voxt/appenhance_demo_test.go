package main

import (
	"slices"
	"testing"

	"github.com/egoist/mygo/ui"
)

type appEnhancementDemoStep struct {
	label string
	text  string
	menu  bool
	enter bool
	point bool
}

func TestAppEnhancementDemoActionControls(t *testing.T) {
	for _, test := range []struct {
		name       string
		fixture    string
		prepare    func(*meetingDemoHost)
		steps      []appEnhancementDemoStep
		want       []string
		wantText   string
		otherText  string
		rejectText string
		check      func(*testing.T, *meetingDemoHost)
	}{
		{name: "master switch", fixture: "en/appenhance-normal", steps: appEnhancementSteps("Use App enhancement"), want: []string{"setAppEnhancementEnabled false"}},
		{name: "new group", fixture: "en/appenhance-normal", steps: appEnhancementSteps("New group"), want: []string{"addAppEnhancementGroup"}},
		{name: "empty new group", fixture: "en/appenhance-empty", steps: appEnhancementSteps("New group"), want: []string{"addAppEnhancementGroup"}},
		{name: "open browsers", fixture: "en/appenhance-normal", steps: appEnhancementSteps("Browsers"), want: []string{"openAppEnhancementBrowsers"}, wantText: "Bundle ID"},
		{name: "group enabled", fixture: "en/appenhance-normal", steps: appEnhancementSteps("Group enabled"), want: []string{"setAppEnhancementGroupEnabled g-chat false"}},
		{name: "transcribe feature", fixture: "en/appenhance-normal", steps: appEnhancementSteps("Use the group for Transcribe"), want: []string{"setAppEnhancementGroupFeature g-chat transcribe false"}},
		{name: "translate feature", fixture: "en/appenhance-normal", steps: appEnhancementSteps("Use the group for Translate"), want: []string{"setAppEnhancementGroupFeature g-chat translate true"}},
		{name: "rewrite feature", fixture: "en/appenhance-normal", steps: appEnhancementSteps("Use the group for Rewrite"), want: []string{"setAppEnhancementGroupFeature g-chat rewrite false"}},
		{name: "group name", fixture: "en/appenhance-normal", steps: []appEnhancementDemoStep{{label: "Group name", text: " 2"}}, want: []string{`setAppEnhancementGroupName g-chat "Team chat 2"`}},
		{name: "group prompt", fixture: "en/appenhance-normal", prepare: func(host *meetingDemoHost) {
			host.appEnhancementUI.editFor = "g-chat"
			host.appEnhancementUI.name = "Team chat"
		}, steps: []appEnhancementDemoStep{{label: "Prompt for this group", text: "New prompt"}}, want: []string{`setAppEnhancementGroupPrompt g-chat "New prompt"`}},
		{name: "copy variable", fixture: "en/appenhance-normal", steps: appEnhancementSteps("{{APP_NAME}}"), want: []string{`copyAppEnhancementVariable "{{APP_NAME}}"`}},
		{name: "delete ask", fixture: "en/appenhance-normal", steps: []appEnhancementDemoStep{{label: "More actions"}, {label: "Delete group…", menu: true}}, wantText: "Delete “Team chat”?"},
		{name: "delete", fixture: "en/appenhance-delete-ask", steps: appEnhancementSteps("Delete"), want: []string{"deleteAppEnhancementGroup g-chat"}},
		{name: "delete cancel", fixture: "en/appenhance-delete-ask", steps: appEnhancementSteps("Cancel"), rejectText: "Delete “Team chat”?", check: func(t *testing.T, host *meetingDemoHost) {
			if host.appEnhancementUI.delAsk {
				t.Fatal("cancel did not close the delete dialog")
			}
		}},
		{name: "open apps", fixture: "en/appenhance-normal", steps: appEnhancementSteps("Add apps…"), want: []string{"openAppEnhancementApps"}, wantText: "Add apps to Team chat"},
		{name: "toggle app", fixture: "en/appenhance-apps-normal", steps: appEnhancementSteps("Google Chrome"), want: []string{"toggleAppEnhancementApp com.google.Chrome"}},
		{name: "add bundle", fixture: "en/appenhance-apps-normal", steps: []appEnhancementDemoStep{{label: "Bundle ID", text: "com.example.editor"}, {label: "Add bundle ID"}}, want: []string{"addAppEnhancementBundle g-chat com.example.editor"}},
		{name: "remove app", fixture: "en/appenhance-normal", steps: appEnhancementSteps("Remove Slack"), want: []string{"removeAppEnhancementApp g-chat com.tinyspeck.slackmacgap"}},
		{name: "add website", fixture: "en/appenhance-normal", steps: []appEnhancementDemoStep{{label: "New website rule", text: "example.com"}, {label: "Add website rule"}}, want: []string{`addAppEnhancementURL g-chat "example.com"`}},
		{name: "remove website", fixture: "en/appenhance-preview-url", steps: appEnhancementSteps("Remove gitlab.com/acme/*"), want: []string{"removeAppEnhancementURL g-code 1"}},
		{name: "auto key none", fixture: "en/appenhance-normal", steps: []appEnhancementDemoStep{{label: "Key after inserting"}, {label: "None", point: true}}, want: []string{`setAppEnhancementAutoKey g-chat ""`}},
		{name: "auto key custom", fixture: "en/appenhance-normal", steps: []appEnhancementDemoStep{{label: "Key after inserting"}, {label: "Custom…", point: true}}, wantText: "Custom key after inserting", check: func(t *testing.T, host *meetingDemoHost) {
			if !host.appEnhancementUI.keyCustom {
				t.Fatal("custom key editor did not open")
			}
		}},
		{name: "custom key commit", fixture: "en/appenhance-key-custom-editing", steps: []appEnhancementDemoStep{{label: "Custom key after inserting", enter: true}}, want: []string{`commitAppEnhancementCustomAutoKey g-chat "command+return"`}},
		{name: "add browser", fixture: "en/appenhance-browsers-normal", steps: []appEnhancementDemoStep{{label: "Bundle ID", text: "org.example.new"}, {label: "Name", text: "New Browser"}, {label: "Add browser"}}, want: []string{`addAppEnhancementBrowser org.example.new "New Browser"`}},
		{name: "remove browser", fixture: "en/appenhance-browsers-normal", steps: appEnhancementSteps("Remove Example Browser"), want: []string{"removeAppEnhancementBrowser org.example.browser"}},
		{name: "refresh preview", fixture: "en/appenhance-normal", steps: appEnhancementSteps("Read again"), want: []string{"refreshAppEnhancementPreview"}},
		{name: "add preview app", fixture: "en/appenhance-preview-add-actions", steps: appEnhancementSteps("Add Google Chrome to Team chat"), want: []string{"addAppEnhancementPreviewApp"}},
		{name: "add preview site", fixture: "en/appenhance-preview-add-actions", steps: appEnhancementSteps("Add example.com to Team chat"), want: []string{"addAppEnhancementPreviewSite"}},
		{name: "select group", fixture: "en/appenhance-normal", steps: appEnhancementSteps("Code review"), wantText: "gitlab.com/acme/*", otherText: "Add Slack to Code review", check: func(t *testing.T, host *meetingDemoHost) {
			if host.appEnhancementUI.sel != "g-code" {
				t.Fatalf("selected group %q", host.appEnhancementUI.sel)
			}
		}},
		{name: "selection updates preview actions", fixture: "en/appenhance-preview-add-actions", steps: appEnhancementSteps("Code review"), wantText: "Add Google Chrome to Code review", otherText: "Add example.com to Code review"},
		{name: "selection updates app membership", fixture: "en/appenhance-normal", steps: appEnhancementSteps("Code review", "Add apps…"), want: []string{"openAppEnhancementApps"}, wantText: "Add apps to Code review", check: func(t *testing.T, host *meetingDemoHost) {
			for _, row := range host.appEnhancementState().Apps.Rows {
				if row.Ref.BundleID == "com.tinyspeck.slackmacgap" && row.Member {
					t.Fatal("Slack stayed checked after Code review was selected")
				}
			}
		}},
		{name: "search apps", fixture: "en/appenhance-apps-normal", steps: []appEnhancementDemoStep{{label: "Search apps", text: "missing"}}, wantText: "No app matches."},
		{name: "empty catalog stays empty", fixture: "en/appenhance-apps-empty", steps: []appEnhancementDemoStep{{label: "Search apps", text: "Chrome"}}, rejectText: "Google Chrome", check: func(t *testing.T, host *meetingDemoHost) {
			if apps := host.appEnhancementState().Apps; apps.HasCatalog || len(apps.Rows) != 0 {
				t.Fatalf("search populated the empty catalog: %+v", apps)
			}
		}},
		{name: "apps done", fixture: "en/appenhance-apps-normal", steps: appEnhancementSteps("Done"), rejectText: "Add apps to Team chat", check: func(t *testing.T, host *meetingDemoHost) {
			if host.appEnhancementUI.appsOpen {
				t.Fatal("Done did not close Add apps")
			}
		}},
		{name: "browsers done", fixture: "en/appenhance-browsers-normal", steps: appEnhancementSteps("Done"), check: func(t *testing.T, host *meetingDemoHost) {
			if host.appEnhancementUI.browsersOpen {
				t.Fatal("Done did not close Browsers")
			}
		}},
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
				switch {
				case step.point:
					if err := tester.Click(step.label); err != nil {
						t.Fatalf("%s: %v (texts %q)", step.label, err, tester.Texts())
					}
				case step.menu:
					if err := tester.ChooseMenuItem(step.label); err != nil {
						t.Fatalf("%s: %v", step.label, err)
					}
				case step.text != "":
					if err := tester.Click(step.label); err != nil {
						t.Fatalf("%s: %v (texts %q)", step.label, err, tester.Texts())
					}
					tester.Type(step.text)
				case step.enter:
					if err := tester.Click(step.label); err != nil {
						t.Fatalf("%s: %v", step.label, err)
					}
					tester.Key(0, ui.KeyEnter)
				default:
					if err := tester.ClickControl(step.label); err != nil {
						t.Fatalf("%s: %v (controls %q)", step.label, err, tester.AccessLabels())
					}
				}
				frames(tester)
			}
			if !slices.Equal(host.calls, test.want) {
				t.Fatalf("calls %q, want %q", host.calls, test.want)
			}
			if test.wantText != "" && !tester.HasText(test.wantText) {
				t.Fatalf("control did not show %q; texts %q", test.wantText, tester.Texts())
			}
			if test.otherText != "" && !tester.HasText(test.otherText) {
				t.Fatalf("control did not show %q; texts %q", test.otherText, tester.Texts())
			}
			if test.rejectText != "" && tester.HasText(test.rejectText) {
				t.Fatalf("control still shows %q; texts %q", test.rejectText, tester.Texts())
			}
			if test.check != nil {
				test.check(t, host)
			}
		})
	}
}

func appEnhancementSteps(labels ...string) []appEnhancementDemoStep {
	steps := make([]appEnhancementDemoStep, len(labels))
	for index, label := range labels {
		steps[index].label = label
	}
	return steps
}
