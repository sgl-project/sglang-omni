package main

import (
	"slices"
	"testing"
	"time"

	"github.com/egoist/mygo"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/overlay/layout"
)

// The card's menu: "Ask again in 2 minutes" hides it and asks again while
// the app still records; "Start recording automatically" starts this
// meeting and the next ones without a card.
func TestMeetingAskMenu(t *testing.T) {
	a, _, _ := newTestApp(t)
	timers := detectClock(t, a)
	micUsers(t, a, "us.zoom.xos")
	closed := meetingAskLayoutModel(a.ask.view)
	a.toggleMeetingAskMenu()
	if open := meetingAskLayoutModel(a.ask.view); !a.ask.view.MenuOpen || open.Height <= closed.Height || open.Width != closed.Width {
		t.Fatalf("the menu: %+v, card %v then %v", a.ask.view, closed, open)
	}
	a.chooseMeetingAskMenu(meetingAskSnooze)
	if a.ask.view.Visible {
		t.Fatal("asking later left the card")
	}
	timers.advance(meetingAskLater - time.Second)
	if a.ask.view.Visible {
		t.Fatal("asked again before two minutes")
	}
	timers.advance(time.Second)
	if !a.ask.view.Visible || a.ask.view.App != "Zoom" || a.ask.view.MenuOpen {
		t.Fatalf("two minutes later, Zoom still records: %+v", a.ask.view)
	}

	a.chooseMeetingAskMenu(meetingAskAuto)
	if !a.prefs.Features.Meeting.AutoStart || a.ask.view.Visible || !a.meetingActive() && len(a.toasts) != 1 {
		t.Fatalf("automatic: setting %+v, card %+v, toasts %q", a.prefs.Features.Meeting, a.ask.view, a.toasts)
	}
	micUsers(t, a, "us.zoom.xos", "com.microsoft.teams2")
	if a.ask.view.Visible || !a.meetingActive() && len(a.toasts) != 2 {
		t.Fatalf("the next meeting, automatic: card %+v, toasts %q", a.ask.view, a.toasts)
	}

	b, _, _ := newTestApp(t)
	micUsers(t, b, "us.zoom.xos")
	b.toggleMeetingAskMenu()
	b.chooseMeetingAskMenu(meetingAskSettings)
	if !b.ask.view.Visible || b.ask.view.MenuOpen || !b.mvUI.setOpen {
		t.Fatalf("settings: card %+v, dialog open %v", b.ask.view, b.mvUI.setOpen)
	}
	// Start recording automatically, switched on while the card asks,
	// answers it: this meeting starts too.
	b.setMeetingAutoStart(true)
	if b.ask.view.Visible || !b.meetingActive() && len(b.toasts) != 1 {
		t.Fatalf("automatic, chosen in the settings: card %+v, toasts %q", b.ask.view, b.toasts)
	}
}

// The card turns compact when left alone, opens under the pointer, and
// stays open while its menu is.
func TestMeetingAskCompact(t *testing.T) {
	a, _, _ := newTestApp(t)
	timers := detectClock(t, a)
	micUsers(t, a, "us.zoom.xos")
	open := meetingAskLayoutModel(a.ask.view)
	timers.advance(meetingAskRest)
	if compact := meetingAskLayoutModel(a.ask.view); !a.ask.view.Compact || compact.Height >= open.Height {
		t.Fatalf("left alone: %+v, card %v then %v", a.ask.view, open, compact)
	}
	a.meetingAskHovered()
	if a.ask.view.Compact {
		t.Fatal("the pointer did not open the compact card")
	}
	a.toggleMeetingAskMenu()
	timers.advance(3 * meetingAskRest)
	if a.ask.view.Compact || !a.ask.view.MenuOpen {
		t.Fatalf("the card closed under its open menu: %+v", a.ask.view)
	}
	a.toggleMeetingAskMenu()
	timers.advance(meetingAskRest)
	if !a.ask.view.Compact {
		t.Fatal("the card stayed open after its menu closed")
	}
	a.dismissMeetingAsk()
	timers.advance(time.Hour)
	if a.ask.view.Visible {
		t.Fatal("a timer brought a dismissed card back")
	}
}

// The card starts on the left edge and grows upwards from where it is.
func TestMeetingAskFrame(t *testing.T) {
	area := mygo.Rectangle{X: 0, Y: 25, Width: 1440, Height: 875}
	row := meetingAskLayoutModel(buildMeetingAskViewState(i18n.NewTranslator("en"), meetingAskInput{App: "zoom"}))
	menu := meetingAskLayoutModel(buildMeetingAskViewState(i18n.NewTranslator("en"), meetingAskInput{App: "zoom", MenuOpen: true}))
	first := meetingAskFrame(mygo.Rectangle{}, area, row.Width, row.Height)
	if first.X > 20 || first.Y < area.Y+area.Height/3 || first.Y+first.Height > area.Y+area.Height*2/3 || first.Width != meetingAskFaceW+2*layout.Pad {
		t.Fatalf("the first frame %+v in %+v", first, area)
	}
	grown := meetingAskFrame(first, area, menu.Width, menu.Height)
	if grown.X != first.X || grown.Y+grown.Height != first.Y+first.Height || grown.Height <= first.Height {
		t.Fatalf("the menu's frame %+v from %+v", grown, first)
	}
	dragged := mygo.Rectangle{X: 5000, Y: -400, Width: row.Width, Height: row.Height}
	if kept := meetingAskFrame(dragged, area, row.Width, row.Height); kept.X+kept.Width > area.X+area.Width+layout.Pad || kept.Y < area.Y-layout.Pad {
		t.Fatalf("a card off the screen stays there: %+v", kept)
	}
}

// The card's controls call its actions, in English and in Chinese.
func TestMeetingAskDemoControls(t *testing.T) {
	for _, test := range []struct {
		name, fixture string
		steps         []string
		want          []string
	}{
		{"start", "en/ask-normal", []string{"Start recording"}, []string{"startDetectedMeeting"}},
		{"start zh", "zh-Hans/ask-normal", []string{"开始记录"}, []string{"startDetectedMeeting"}},
		{"later", "en/ask-normal", []string{"More options", "Ask again in 2 minutes"}, []string{"toggleMeetingAskMenu", "chooseMeetingAskMenu snooze"}},
		{"automatic", "en/ask-menu", []string{"Start recording automatically"}, []string{"chooseMeetingAskMenu auto"}},
		{"settings", "zh-Hans/ask-menu", []string{"检测设置"}, []string{"chooseMeetingAskMenu settings"}},
		{"close menu", "en/ask-menu", []string{"More options"}, []string{"toggleMeetingAskMenu"}},
		{"dismiss", "en/ask-menu", []string{"Dismiss"}, []string{"dismissMeetingAsk"}},
		{"end", "en/ask-ended", []string{"End recording"}, []string{"startDetectedMeeting"}},
		{"end zh", "zh-Hans/ask-ended", []string{"结束记录"}, []string{"startDetectedMeeting"}},
		{"not ended", "en/ask-ended", []string{"Dismiss"}, []string{"dismissMeetingAsk"}},
	} {
		t.Run(test.name, func(t *testing.T) {
			host, err := loadMeetingDemoHost("testdata/meeting")
			if err != nil {
				t.Fatal(err)
			}
			selectMeetingDemoFixture(t, host, test.fixture)
			tester := newMeetingDemoTester(t, host)
			for _, label := range test.steps {
				if err := tester.ClickControl(label); err != nil {
					t.Fatalf("%s: %v (controls %q)", label, err, tester.AccessLabels())
				}
				frames(tester)
			}
			if !slices.Equal(host.calls, test.want) {
				t.Fatalf("calls %q, want %q", host.calls, test.want)
			}
		})
	}
	host, err := loadMeetingDemoHost("testdata/meeting")
	if err != nil {
		t.Fatal(err)
	}
	selectMeetingDemoFixture(t, host, "en/ask-compact")
	tester := newMeetingDemoTester(t, host)
	frames(tester)
	if tester.HasText("Start recording") || !tester.HasText("Meeting detected") || !tester.HasText("Tencent Meeting") {
		t.Fatalf("the compact card: %q", tester.Texts())
	}
}
