package main

// The shell's fixtures: the sidebar, the notification cards, the logs,
// the palette, the "?" sheet, the announcement dialog and the whole
// frame, built from a real test app through the shell's model builders.

import (
	"testing"
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/core"
	"github.com/sgl-project/sglang-omni/Voxt/internal/helper"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/learn"
	"github.com/sgl-project/sglang-omni/Voxt/internal/local"
	"github.com/sgl-project/sglang-omni/Voxt/internal/logs"
	"github.com/sgl-project/sglang-omni/Voxt/internal/providers"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
)

// shellFixtureScenario is the app configuration a shell fixture is
// built from.
type shellFixtureScenario struct {
	readyLocal   bool     // speech and language models installed
	hidden       []string // features hidden (their pages leave the sidebar)
	closed       []string // sidebar sections closed
	rail         bool     // the user's layout choice: icons only
	sidebarWidth float64  // DIPs at 100 %
	suggestions  int      // dictionary suggestions (the Dictionary item's badge)
	settingsOpen bool
	current      string // the router's page ("/home" when "")
	toasts       []string
	logsPanel    bool
	logEntries   []core.LogEntry
	logsSource   string
	logsQuery    string
	paletteOpen  bool
	paletteQuery string
	sheetOpen    bool
	notices      []core.Notification
	problem      string // "models-both", "models-asr", "models-llm", "models-pending", "helper-<reason>"
	announcement *core.Notification
	noHotkeys    bool // the features have no shortcuts (the sheet's "—")
}

// shellFixtureScenarios are the shell's fixtures, one per state the
// shell shows.
func shellFixtureScenarios() []meetingFixtureScenario {
	var scenarios []meetingFixtureScenario
	add := func(kind, name string, in shellFixtureScenario) {
		scenarios = append(scenarios, meetingFixtureScenario{name: name, kind: kind, shell: in})
	}

	add("shell-nav", "nav-sidebar", shellFixtureScenario{readyLocal: true})
	add("shell-nav", "nav-rail", shellFixtureScenario{readyLocal: true, rail: true})
	add("shell-nav", "nav-badge-sections", shellFixtureScenario{readyLocal: true, suggestions: 2, hidden: []string{"notes"}, closed: []string{"Resources"}, sidebarWidth: 240})
	add("shell-nav", "nav-notices", shellFixtureScenario{readyLocal: true, notices: shellFixtureNotices(2)})
	add("shell-nav", "nav-settings-open", shellFixtureScenario{readyLocal: true, settingsOpen: true, current: "/dictionary"})

	add("shell-notices", "notices-info", shellFixtureScenario{notices: shellFixtureNotices(1)})
	add("shell-notices", "notices-warning", shellFixtureScenario{notices: []core.Notification{shellFixtureNotice("n1", core.SeverityWarning, "Watch out", "Something needs a look.", "/history")}})
	add("shell-notices", "notices-error", shellFixtureScenario{notices: []core.Notification{shellFixtureNotice("n1", core.SeverityError, "It broke", "The thing did not work.", "")}})
	add("shell-notices", "notices-success", shellFixtureScenario{notices: []core.Notification{shellFixtureNotice("n1", core.SeveritySuccess, "Done", "It worked.", "")}})
	add("shell-notices", "notices-stack", shellFixtureScenario{notices: shellFixtureNotices(3)})
	add("shell-notices", "notices-actions", shellFixtureScenario{notices: []core.Notification{
		shellFixtureNotice("n1", core.SeverityInfo, "Update ready", "Restart to use it.", linkRelaunch),
		shellFixtureNotice("n2", core.SeverityInfo, "Voxt 2.1 is out", "What is new.", linkAnnouncement),
		shellFixtureNotice("n3", core.SeverityWarning, "Old data found", "Bring it over.", linkImportLegacy),
		shellFixtureNotice("n4", core.SeverityInfo, "Download paused", "Get the rest.", linkDownload+"https://example.invalid/voxt.dmg"),
		shellFixtureNotice("n5", core.SeverityInfo, "History grew", "100 entries.", "/history"),
	}})
	add("shell-notices", "notices-models-both", shellFixtureScenario{problem: "models-both"})
	add("shell-notices", "notices-models-asr", shellFixtureScenario{problem: "models-asr"})
	add("shell-notices", "notices-models-llm", shellFixtureScenario{problem: "models-llm"})
	add("shell-notices", "notices-models-pending", shellFixtureScenario{problem: "models-pending"})
	for _, reason := range []helper.FailReason{
		helper.FailMissing, helper.FailStart, helper.FailProtocol,
		helper.FailHandshake, helper.FailCrashed, helper.FailStuck,
	} {
		add("shell-notices", "notices-helper-"+string(reason), shellFixtureScenario{problem: "helper-" + string(reason)})
	}

	add("shell-logs", "logs-empty", shellFixtureScenario{readyLocal: true, logsSource: "app"})
	add("shell-logs", "logs-lines", shellFixtureScenario{readyLocal: true, logEntries: shellFixtureLogEntries()})
	add("shell-logs", "logs-filter", shellFixtureScenario{readyLocal: true, logEntries: shellFixtureLogEntries(), logsQuery: "line 2"})
	add("shell-logs", "logs-filter-empty", shellFixtureScenario{readyLocal: true, logEntries: shellFixtureLogEntries(), logsQuery: "zzqx"})
	add("shell-logs", "logs-source-helper", shellFixtureScenario{readyLocal: true, logEntries: shellFixtureLogEntries(), logsSource: helper.LogSource})

	add("shell-palette", "palette-all", shellFixtureScenario{readyLocal: true, paletteOpen: true})
	add("shell-palette", "palette-filter", shellFixtureScenario{readyLocal: true, paletteOpen: true, paletteQuery: "mod"})
	add("shell-palette", "palette-no-match", shellFixtureScenario{readyLocal: true, paletteOpen: true, paletteQuery: "zzqx"})
	add("shell-palette", "palette-hidden-page", shellFixtureScenario{readyLocal: true, paletteOpen: true, hidden: []string{"notes"}, paletteQuery: "notes"})

	add("shell-sheet", "sheet-hotkeys", shellFixtureScenario{readyLocal: true, sheetOpen: true})
	add("shell-sheet", "sheet-none", shellFixtureScenario{readyLocal: true, sheetOpen: true, noHotkeys: true})

	add("shell-announcement", "shell-announcement-version", shellFixtureScenario{readyLocal: true, announcement: &core.Notification{
		ID: "announcement/v2-1", Title: "Voxt 2.1 is out", Body: "Faster transcription and a new look for the cards.",
		Time: meetingFixtureNow, Version: "2.1.0", Link: linkAnnouncement,
	}})
	add("shell-announcement", "shell-announcement-plain", shellFixtureScenario{readyLocal: true, announcement: &core.Notification{
		ID: "announcement/hello", Title: "Hello", Body: "Welcome to Voxt.",
		Time: meetingFixtureNow, Link: linkAnnouncement,
	}})

	add("shell", "shell-normal", shellFixtureScenario{readyLocal: true})
	add("shell", "shell-busy", shellFixtureScenario{
		readyLocal: true, notices: shellFixtureNotices(3), logsPanel: true, logEntries: shellFixtureLogEntries(),
		toasts: []string{"Copied 2 lines"}, suggestions: 2,
	})
	add("shell", "shell-palette", shellFixtureScenario{readyLocal: true, paletteOpen: true, paletteQuery: "mod"})
	add("shell", "shell-sheet", shellFixtureScenario{readyLocal: true, sheetOpen: true})
	add("shell", "shell-announcement", shellFixtureScenario{readyLocal: true, announcement: &core.Notification{
		ID: "announcement/v2-1", Title: "Voxt 2.1 is out", Body: "Faster transcription and a new look for the cards.",
		Time: meetingFixtureNow, Version: "2.1.0", Link: linkAnnouncement,
	}})
	add("shell", "shell-rail-notices", shellFixtureScenario{readyLocal: true, rail: true, notices: shellFixtureNotices(2)})
	return scenarios
}

// shellFixtureNotice is a notification at the fixtures' moment.
func shellFixtureNotice(id string, severity core.Severity, title, body, link string) core.Notification {
	return core.Notification{ID: id, Severity: severity, Title: title, Body: body, Link: link, Time: meetingFixtureNow}
}

// shellFixtureNotices are n cards, the newest first.
func shellFixtureNotices(n int) []core.Notification {
	titles := []string{"First card", "Second card", "Third card"}
	bodies := []string{"The newest thing that happened.", "Before that.", "Before both."}
	severities := []core.Severity{core.SeverityInfo, core.SeverityWarning, core.SeverityError}
	var out []core.Notification
	for i := 0; i < n; i++ {
		out = append(out, shellFixtureNotice("fixture-"+itoa(i+1), severities[i%len(severities)], titles[i%len(titles)], bodies[i%len(bodies)], "/history"))
	}
	return out
}

// shellFixtureLogEntries are lines of every level and both sources at
// fixed times.
func shellFixtureLogEntries() []core.LogEntry {
	at := time.Date(2026, time.January, 2, 9, 30, 0, 0, time.UTC)
	return []core.LogEntry{
		{Time: at, Source: "app", Level: core.SeverityInfo, Message: "line 1: the app started"},
		{Time: at.Add(time.Second), Source: helper.LogSource, Level: core.SeverityInfo, Message: "line 2: the helper said hello"},
		{Time: at.Add(2 * time.Second), Source: "app", Level: core.SeverityWarning, Message: "line 3: a warning"},
		{Time: at.Add(3 * time.Second), Source: helper.LogSource, Level: core.SeverityError, Message: "line 4: an error"},
		{Time: at.Add(4 * time.Second), Source: "app", Level: core.SeverityDebug, Message: "line 5: a debug line"},
	}
}

// newShellFixtureApp is a test app for the shell's fixtures, in the
// language the golden test set. The start-up's own lines and cards go:
// the fixtures say what the log and the notifications hold.
func newShellFixtureApp(t *testing.T) *app {
	t.Helper()
	locale := i18n.Locale()
	a, _, _ := newTestAppIn(t, testDir(t))
	i18n.SetLocale(locale)
	a.logs = logs.New(logs.Options{Bus: a.bus})
	a.notices = nil
	a.problemCode, a.problemDismissed = nil, nil
	a.noticesGen++
	return a
}

// shellFixtureApply configures the test app as the scenario says.
func shellFixtureApply(t *testing.T, a *app, in shellFixtureScenario) {
	t.Helper()
	if in.readyLocal {
		a.local = readyLocal()
	}
	a.updateSettings(func(s *settings.Settings) {
		if len(in.hidden) > 0 {
			s.Features.Hidden = in.hidden
		}
		if in.rail {
			s.Appearance.Rail, s.Appearance.RailChosen = true, true
		}
		if in.sidebarWidth != 0 {
			s.Appearance.SidebarWidth = in.sidebarWidth
		}
		if len(in.closed) > 0 {
			s.Appearance.ClosedSections = in.closed
		}
		if in.noHotkeys {
			s.Features.Transcribe.Hotkeys = nil
			s.Features.Translate.Hotkeys = nil
			s.Features.Rewrite.Hotkeys = nil
		}
	})
	for i := 0; i < in.suggestions; i++ {
		a.suggestions = append(a.suggestions, learn.Suggestion{})
	}
	a.settingsOpen = in.settingsOpen
	if in.current != "" {
		a.router.Push(in.current)
	}
	a.toasts = append([]string{}, in.toasts...)
	a.logsPanel = in.logsPanel
	for _, e := range in.logEntries {
		a.logs.Append(e)
	}
	a.logsSource, a.logsQuery = in.logsSource, in.logsQuery
	a.paletteOpen, a.paletteQuery = in.paletteOpen, in.paletteQuery
	a.sheetOpen = in.sheetOpen
	for _, n := range in.notices {
		a.noticePosted(n)
	}
	switch {
	case in.problem == "models-both":
		a.checkModels()
	case in.problem == "models-asr":
		a.local = withChat(&fakeLocal{}, &fakeChat{})
		a.checkModels()
	case in.problem == "models-llm":
		a.local = &fakeLocal{ASRP: &fakeTranscriptions{}}
		a.checkModels()
	case in.problem == "models-pending":
		a.updateSettings(func(s *settings.Settings) {
			s.Providers.LLM = []settings.Provider{{ID: "openai"}}
		})
		a.keyMasks[providers.Account(local.LLM, "openai")] = ""
		a.checkModels()
	case in.problem != "":
		reason := helper.FailReason(in.problem[len("helper-"):])
		a.helperChanged(helper.Status{State: helper.Failed, Reason: reason, Err: "fixture failure"})
	}
	if in.announcement != nil {
		a.announceShown, a.announceShownOpen = *in.announcement, true
	}
}

// buildShellFixture drives a scenario through the shell's model
// builders, on a real test app.
func buildShellFixture(t *testing.T, scenario meetingFixtureScenario, locale string) meetingFixtureState {
	t.Helper()
	fixture := meetingFixtureState{Name: scenario.name, Kind: scenario.kind}
	a := newShellFixtureApp(t)
	shellFixtureApply(t, a, scenario.shell)
	// Cards posted by the app's checks carry the wall clock; the
	// fixtures' moment is the golden one.
	for i := range a.notices {
		a.notices[i].Time = meetingFixtureNow
	}
	switch scenario.kind {
	case "shell":
		fixture.Shell = a.shellModel()
		fixture.Announcement = a.announcementModel(meetingFixtureNow)
		fixture.ShellUI = &shellFixtureUI{
			PaletteOpen: a.paletteOpen, PaletteQuery: a.paletteQuery, PaletteSel: a.paletteSel,
			SheetOpen: a.sheetOpen, AnnouncementOpen: fixture.Announcement != nil,
		}
	case "shell-nav":
		fixture.Nav = a.navModel()
		fixture.Notices = a.noticesModel()
	case "shell-notices":
		fixture.Notices = a.noticesModel()
	case "shell-logs":
		fixture.Logs = a.logsModel()
	case "shell-palette":
		fixture.Palette = a.paletteModel()
		fixture.ShellUI = &shellFixtureUI{PaletteOpen: true, PaletteQuery: a.paletteQuery, PaletteSel: a.paletteSel}
	case "shell-sheet":
		fixture.Sheet = a.sheetModel()
		fixture.ShellUI = &shellFixtureUI{SheetOpen: true}
	case "shell-announcement":
		fixture.Announcement = a.announcementModel(meetingFixtureNow)
		fixture.ShellUI = &shellFixtureUI{AnnouncementOpen: fixture.Announcement != nil}
	default:
		t.Fatalf("unknown shell fixture kind %q", scenario.kind)
	}
	return fixture
}

// TestShellFixtureCoverage: every state and problem form the shell shows
// has a fixture.
func TestShellFixtureCoverage(t *testing.T) {
	scenarios := meetingFixtureScenarios()
	var shell []meetingFixtureScenario
	for _, scenario := range scenarios {
		if scenario.kind == "shell" || scenario.kind == "shell-nav" || scenario.kind == "shell-notices" ||
			scenario.kind == "shell-logs" || scenario.kind == "shell-palette" ||
			scenario.kind == "shell-sheet" || scenario.kind == "shell-announcement" {
			shell = append(shell, scenario)
		}
	}
	if len(shell) == 0 {
		t.Fatal("no shell scenarios")
	}
	kinds := map[string]bool{}
	severities := map[string]bool{}
	levels := map[string]bool{}
	problems := map[string]bool{}
	for _, locale := range i18n.Supported() {
		i18n.SetLocale(locale)
		for _, scenario := range shell {
			fixture := buildShellFixture(t, scenario, locale)
			kinds[scenario.kind] = true
			for _, n := range fixture.Notices {
				severities[n.Severity] = true
			}
			if fixture.Shell != nil {
				for _, n := range fixture.Shell.Notices {
					severities[n.Severity] = true
				}
			}
			if fixture.Logs != nil {
				for _, line := range fixture.Logs.Lines {
					levels[line.Level] = true
				}
			}
			if scenario.shell.problem != "" {
				problems[scenario.shell.problem] = true
			}
		}
	}
	i18n.SetLocale("en")
	for _, kind := range []string{"shell", "shell-nav", "shell-notices", "shell-logs", "shell-palette", "shell-sheet", "shell-announcement"} {
		if !kinds[kind] {
			t.Errorf("no fixture of kind %s", kind)
		}
	}
	for _, severity := range []string{noticeLevelInfo, noticeLevelWarning, noticeLevelError, noticeLevelSuccess} {
		if !severities[severity] {
			t.Errorf("no notice fixture of severity %s", severity)
		}
	}
	for _, level := range []string{logLevelDebug, logLevelInfo, logLevelWarning, logLevelError} {
		if !levels[level] {
			t.Errorf("no log fixture of level %s", level)
		}
	}
	for _, problem := range []string{"models-both", "models-asr", "models-llm", "models-pending"} {
		if !problems[problem] {
			t.Errorf("no models problem fixture %s", problem)
		}
	}
	// Every failure of the helper but FailUpdated (the update's own card
	// covers it, updates.go).
	for reason := range meetingTypedConstants(t, "internal/helper/helper.go", "FailReason") {
		if reason == string(helper.FailUpdated) {
			continue
		}
		if !problems["helper-"+reason] {
			t.Errorf("no helper failure fixture helper-%s", reason)
		}
	}
}
