package main

import (
	"slices"
	"testing"

	"github.com/egoist/mygo/ui"
)

func TestUpdatesDemoActionControls(t *testing.T) {
	for _, test := range []struct {
		name    string
		fixture string
		steps   []historyDemoStep
		want    []string
		check   func(*testing.T, *meetingDemoHost)
	}{
		{name: "check", fixture: "en/updates-about-normal", steps: historySteps("Check for Updates…"), want: []string{"checkForUpdates"}},
		{name: "restart", fixture: "en/updates-about-installed", steps: historySteps("Restart Voxt"), want: []string{"relaunch"}},
		{name: "automatic checks", fixture: "en/updates-about-normal", steps: historySteps("Check for updates automatically"), want: []string{"setAutoCheck false"}},
		{name: "automatic install", fixture: "en/updates-about-normal", steps: historySteps("Install updates without asking"), want: []string{"setAutoInstall true"}},
		{name: "automatic install disabled", fixture: "en/updates-about-autocheck-off", steps: []historyDemoStep{{label: "Install updates without asking", click: true}}, check: func(t *testing.T, host *meetingDemoHost) {
			if !host.updatesUI.autoInstall {
				t.Fatal("disabled install switch changed value")
			}
		}},
		{name: "beta on", fixture: "en/updates-about-normal", steps: historySteps("Get beta versions"), want: []string{"setBeta true"}},
		{name: "beta off", fixture: "en/updates-about-beta", steps: historySteps("Get beta versions"), want: []string{"setBeta false"}},
		{name: "checking cancel", fixture: "en/updates-release-checking", steps: historySteps("Cancel"), check: updatesReleaseClosed},
		{name: "available later", fixture: "en/updates-release-available", steps: historySteps("Later"), check: updatesReleaseClosed},
		{name: "available close", fixture: "en/updates-release-available", steps: historySteps("Close"), check: updatesReleaseClosed},
		{name: "download", fixture: "en/updates-release-available", steps: historySteps("Download"), want: []string{`downloadRelease "https://fixture.invalid/Voxt-2.0.4.dmg"`}, check: updatesReleaseClosed},
		{name: "up to date done", fixture: "en/updates-release-up-to-date", steps: historySteps("Done"), check: updatesReleaseClosed},
		{name: "up to date escape", fixture: "en/updates-release-up-to-date", steps: []historyDemoStep{{key: ui.KeyEscape}}, check: updatesReleaseClosed},
		{name: "error retry", fixture: "en/updates-release-error-generic", steps: historySteps("Try again"), want: []string{"checkReleasesNow"}},
		{name: "error cancel", fixture: "en/updates-release-error-rate-limit", steps: historySteps("Cancel"), check: updatesReleaseClosed},
	} {
		t.Run(test.name, func(t *testing.T) {
			host, err := loadMeetingDemoHost("testdata/meeting")
			if err != nil {
				t.Fatal(err)
			}
			selectMeetingDemoFixture(t, host, test.fixture)
			tester := newMeetingDemoTester(t, host)
			for _, step := range test.steps {
				step.label = platformText(step.label)
				var err error
				if step.key != 0 {
					tester.Key(step.mods, step.key)
				} else if step.click {
					err = tester.Click(step.label)
				} else {
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

func updatesReleaseClosed(t *testing.T, host *meetingDemoHost) {
	t.Helper()
	if host.updatesUI.relOpen {
		t.Fatal("release dialog stayed open")
	}
}
