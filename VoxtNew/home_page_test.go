package main

import (
	"context"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/egoist/mygo/ui"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/helper"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/helper/proto"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/i18n"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/permission"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/session"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/settings"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/store"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/uitext"
)

// homeTester is the test app on Home, in a window tall enough to show all
// of it.
func homeTester(t *testing.T) (*app, *ui.Tester) {
	t.Helper()
	a, _, tt := newTestApp(t)
	tt.SetSize(windowWidth, 1600)
	a.navigate("/home")
	frames(tt)
	return a, tt
}

// hasSuffix reports whether a text on screen ends with s.
func hasSuffix(tt *ui.Tester, s string) bool {
	frames(tt)
	for _, x := range tt.Texts() {
		if strings.HasSuffix(x, s) {
			return true
		}
	}
	return false
}

// allowAll makes the Microphone and Accessibility allowed, as the helper
// would say.
func allowAll(a *app) {
	a.perms.mic, a.perms.access, a.perms.checked = permission.Granted, true, true
	a.hotkeys.Installed = true
}

func TestHomeReadiness(t *testing.T) {
	a, tt := homeTester(t)
	for _, s := range []string{
		"No speech model or language model is installed: install one of each to start.",
		"Without Accessibility results are copied for you to paste, and shortcuts such as Right ⌥ do not work.",
		"Right ⌥ (once Accessibility is allowed)",
		"Try it", "Today", "Recent results", "Results show here once you use a shortcut.",
	} {
		if !has(tt, s) {
			t.Errorf("no %q in %q", s, tt.Texts())
		}
	}
	if !hasSuffix(tt, " steps to go") {
		t.Errorf("no steps to go: %q", tt.Texts())
	}
	// The whole line is the way to fix it.
	click(t, tt, "Models: Open Local models")
	if a.shownPath() != modelsPath {
		t.Fatalf("went to %s", a.shownPath())
	}
	a.navigate("/home")

	// A speech model only: the language model is missing.
	a.local = &fakeLocal{ASRP: &fakeTranscriptions{}}
	if !has(tt, "No language model is installed: Translate and Rewrite need one, and Transcribe types the text as recognised.") || !has(tt, fakeASRModel.Name) {
		t.Errorf("no language model: %q", tt.Texts())
	}

	// All ready: no list at all, neither its title nor what is ready.
	readyModels(a)
	allowAll(a)
	for _, s := range []string{fakeASRModel.Name, fakeLLMModel.Name, "Speech recognition", "Microphone"} {
		if hasExact(tt, s) {
			t.Errorf("ready, still %q: %q", s, tt.Texts())
		}
	}
	if hasSuffix(tt, " to go") {
		t.Errorf("ready, still the list: %q", tt.Texts())
	}

	// A shortcut macOS takes says so, and opens Transcribe.
	a.updateSettings(func(s *settings.Settings) {
		s.Features.Transcribe.Hotkeys = []settings.Hotkey{{Keys: "Command+Space", Trigger: settings.TriggerTap}}
	})
	if !has(tt, "1 step to go") || !has(tt, "Transcribe's shortcut ⌘ Space: macOS uses these keys for Spotlight.") {
		t.Errorf("no clash: %q", tt.Texts())
	}
	// What is ready shows again, quieter, under what is missing.
	if !hasExact(tt, fakeLLMModel.Name) {
		t.Errorf("not ready, no ready line: %q", tt.Texts())
	}
	click(t, tt, "Shortcut: Open Transcribe")
	if a.router.Path() != "/transcribe" {
		t.Errorf("went to %s", a.router.Path())
	}
}

// TestHomeReadinessLive: the list comes and goes as things stop or
// start being ready, while Home shows.
func TestHomeReadinessLive(t *testing.T) {
	a, tt := homeTester(t)
	readyModels(a)
	allowAll(a)
	if hasSuffix(tt, " to go") {
		t.Fatalf("ready, a list: %q", tt.Texts())
	}
	// Accessibility taken away.
	a.perms.access = false
	if !has(tt, "1 step to go") || !has(tt, "Without Accessibility results are copied for you to paste, and shortcuts such as Right ⌥ do not work.") {
		t.Errorf("access revoked: %q", tt.Texts())
	}
	a.perms.access = true
	if hasSuffix(tt, " to go") {
		t.Errorf("access back, still a list: %q", tt.Texts())
	}
	// The language model removed.
	a.local.(*fakeLocal).Chat = nil
	if !has(tt, "1 step to go") || !has(tt, "No language model is installed: Translate and Rewrite need one, and Transcribe types the text as recognised.") {
		t.Errorf("model deleted: %q", tt.Texts())
	}
	readyModels(a)
	if hasSuffix(tt, " to go") {
		t.Errorf("model back, still a list: %q", tt.Texts())
	}
	// The Microphone denied; the helper failed.
	a.perms.mic = permission.Denied
	a.host.sup.(*fakeSupervisor).set(helper.Status{State: helper.Failed, Reason: helper.FailCrashed, Err: "exit status 2"})
	if !has(tt, "2 steps to go") {
		t.Errorf("mic denied, helper failed: %q", tt.Texts())
	}
	// A helper that is starting is not a step.
	a.host.sup.(*fakeSupervisor).set(helper.Status{State: helper.Starting})
	if !has(tt, "1 step to go") {
		t.Errorf("helper starting: %q", tt.Texts())
	}
}

// TestHomeTryFirstUse: Home offers Try it until VoxtNew has given a
// result once; then never again, the history cleared or not.
func TestHomeTryFirstUse(t *testing.T) {
	a, tt := homeTester(t)
	readyModels(a)
	allowAll(a)
	if !hasExact(tt, "Try it") || a.prefs.Setup.Tried {
		t.Fatalf("first use, no Try it: %q", tt.Texts())
	}
	// A try that is done keeps its result in view on this visit.
	a.sessionChanged(trySnapshot(a, session.Done))
	frames(tt)
	if !a.prefs.Setup.Tried {
		t.Error("a try's result did not mark Tried")
	}
	if !hasExact(tt, "Try it") || !has(tt, "The meeting moves to Thursday at four.") {
		t.Errorf("the try's result went: %q", tt.Texts())
	}
	// Another visit: no Try it.
	a.navigate("/history")
	a.navigate("/home")
	if hasExact(tt, "Try it") {
		t.Errorf("tried, Try it on the next visit: %q", tt.Texts())
	}
	// The history cleared: still none.
	a.clearHistory()
	frames(tt)
	if hasExact(tt, "Try it") {
		t.Errorf("history cleared, Try it again: %q", tt.Texts())
	}
}

// TestHomeTryGoesAfterASession: a real session's result (any feature)
// counts as the first use too; a failure or nothing heard does not.
func TestHomeTryGoesAfterASession(t *testing.T) {
	a, tt := homeTester(t)
	readyModels(a)
	allowAll(a)
	rev := func() uint64 { return a.sess.Rev + 1 }
	a.sessionChanged(session.Snapshot{Rev: rev(), ID: "s1", Kind: store.KindTranslate, State: session.Recording})
	a.sessionChanged(session.Snapshot{Rev: rev(), ID: "s1", Kind: store.KindTranslate, State: session.Done, Result: "  "})
	a.sessionChanged(session.Snapshot{Rev: rev(), ID: "s2", Kind: store.KindRewrite, State: session.Recording})
	a.sessionChanged(session.Snapshot{Rev: rev(), ID: "s2", Kind: store.KindRewrite, State: session.Failed})
	if a.prefs.Setup.Tried || !hasExact(tt, "Try it") {
		t.Fatalf("no result yet, but tried: %v %q", a.prefs.Setup.Tried, tt.Texts())
	}
	a.sessionChanged(session.Snapshot{Rev: rev(), ID: "s3", Kind: store.KindTranslate, State: session.Recording})
	a.sessionChanged(session.Snapshot{Rev: rev(), ID: "s3", Kind: store.KindTranslate, State: session.Done, Result: "Hello."})
	frames(tt)
	if !a.prefs.Setup.Tried {
		t.Error("a translation's result did not mark Tried")
	}
	// Not Home's try: the area goes at once.
	if hasExact(tt, "Try it") {
		t.Errorf("Try it after a session's result: %q", tt.Texts())
	}
}

// TestHomeTryHistoryBeforeTheFlag: a history from before Tried existed
// (an upgrade) hides Try it and sets the flag.
func TestHomeTryHistoryBeforeTheFlag(t *testing.T) {
	a, tt := homeTester(t)
	readyModels(a)
	allowAll(a)
	seedHistory(t, a)
	a.historyChanged()
	frames(tt)
	if hasExact(tt, "Try it") || !a.prefs.Setup.Tried {
		t.Errorf("history, Try it %v tried %v: %q", hasExact(tt, "Try it"), a.prefs.Setup.Tried, tt.Texts())
	}
}

func TestSloganHint(t *testing.T) {
	join := func(h *settings.Hotkey) string {
		b, k, a := sloganHint(h)
		return b + "[" + k + "]" + a
	}
	for _, tc := range []struct {
		h      *settings.Hotkey
		en, zh string
	}{
		{&settings.Hotkey{Keys: "Fn", Trigger: settings.TriggerTap}, "Press [fn] to start", "按 [fn] 开始"},
		{&settings.Hotkey{Keys: "Fn"}, "Press [fn] to start", "按 [fn] 开始"},
		{&settings.Hotkey{Keys: "RightCommand", Trigger: settings.TriggerDouble}, "Double-press [Right ⌘] to start", "双击 [右 ⌘] 开始"},
		{&settings.Hotkey{Keys: "Option+Space", Trigger: settings.TriggerLong}, "Hold [⌥ Space] and speak", "按住 [⌥ 空格] 说话"},
		{nil, "Add a shortcut to start[]", "先设置一个快捷键[]"},
	} {
		i18n.SetLocale("en")
		if got := join(tc.h); got != tc.en {
			t.Errorf("en %+v: %q, want %q", tc.h, got, tc.en)
		}
		i18n.SetLocale("zh-Hans")
		if got := join(tc.h); got != tc.zh {
			t.Errorf("zh %+v: %q, want %q", tc.h, got, tc.zh)
		}
	}
	i18n.SetLocale("en")
}

// TestHomeSlogan: the slogan shows Transcribe's first shortcut in a key
// cap, and without one, a link to add it.
func TestHomeSlogan(t *testing.T) {
	a, tt := homeTester(t)
	for _, s := range []string{"Speak anywhere, it types at your cursor", "Press", "Right ⌥", "to start"} {
		if !hasExact(tt, s) {
			t.Errorf("no %q in %q", s, tt.Texts())
		}
	}
	a.updateSettings(func(s *settings.Settings) {
		s.Features.Transcribe.Hotkeys = []settings.Hotkey{{Keys: "Option+Space", Trigger: settings.TriggerLong}, {Keys: "Fn"}}
	})
	if !hasExact(tt, "Hold") || !hasExact(tt, "⌥ Space") || !hasExact(tt, "and speak") {
		t.Errorf("long: %q", tt.Texts())
	}
	a.updateSettings(func(s *settings.Settings) { s.Features.Transcribe.Hotkeys = nil })
	if !hasExact(tt, "Add a shortcut to start") {
		t.Errorf("no shortcut: %q", tt.Texts())
	}
	click(t, tt, "Add a shortcut to start")
	if a.router.Path() != "/transcribe" {
		t.Errorf("went to %s", a.router.Path())
	}
}

func TestReadHome(t *testing.T) {
	a, _, _ := newTestApp(t)
	h := a.lib.store.History()
	now := time.Date(2026, 10, 15, 15, 0, 0, 0, time.Local)
	put := func(id string, kind store.Kind, at time.Time, status store.Status, result string) {
		t.Helper()
		if err := h.Put(context.Background(), store.Entry{ID: id, Kind: kind, CreatedAt: at, Status: status, Result: result}); err != nil {
			t.Fatal(err)
		}
	}
	put("01A", store.KindTranscribe, now.Add(-time.Hour), store.StatusDone, "Hello world.")
	put("01B", store.KindTranslate, now.Add(-2*time.Hour), store.StatusDone, "你好，世界。")
	put("01C", store.KindRewrite, now.Add(-3*time.Hour), store.StatusFailed, "")
	put("01D", store.KindTranscribe, now.Add(-16*time.Hour), store.StatusDone, "Late yesterday.")
	put("01E", store.KindRewrite, now.Add(-30*time.Hour), store.StatusDone, "Older.")
	st, recent, err := readHome(context.Background(), h, now)
	if err != nil {
		t.Fatal(err)
	}
	if st.count != 2 || st.chars != len("Helloworld.")+6 || st.byKind[store.KindTranscribe] != 1 || st.byKind[store.KindTranslate] != 1 {
		t.Errorf("today %+v", st)
	}
	var ids []string
	for _, e := range recent {
		ids = append(ids, e.ID)
	}
	if strings.Join(ids, ",") != "01A,01B,01D,01E" {
		t.Errorf("recent %v", ids)
	}
	if groupDigits(1234567) != "1,234,567" || groupDigits(999) != "999" {
		t.Errorf("groupDigits %s %s", groupDigits(1234567), groupDigits(999))
	}
}

func TestHomeRecent(t *testing.T) {
	a, tt := homeTester(t)
	seedHistory(t, a)
	seedWriteHistory(t, a)
	a.historyChanged()
	frames(tt)
	if !has(tt, "Let's move the design review to Thursday afternoon.") || !has(tt, "By feature") {
		t.Fatalf("recent: %q", tt.Texts())
	}
	// The newest seeded entry is 5 minutes old: just after midnight it is yesterday's.
	if now := time.Now(); a.hm.today.count == 0 && sameDay(now, now.Add(-5*time.Minute)) {
		t.Errorf("nothing today: %+v", a.hm.today)
	}
	if len(a.hm.recent) != homeRecent {
		t.Errorf("%d recent", len(a.hm.recent))
	}
	e := a.hm.recent[0]
	click(t, tt, "Copy: "+uitext.FirstLine(e.Result))
	if tt.Clipboard() != e.Result {
		t.Errorf("clipboard %q", tt.Clipboard())
	}
	title, _ := uitext.EntryTitle(e)
	click(t, tt, title)
	if a.router.Path() != "/history" || a.hist.sel != e.ID || a.hist.kind() != e.Kind {
		t.Errorf("opened %s sel %q tab %s", a.router.Path(), a.hist.sel, a.hist.kind())
	}
	// History off: nothing is counted.
	a.navigate("/home")
	a.updateSettings(func(s *settings.Settings) { s.Privacy.KeepHistory = false })
	if !has(tt, "History is off, so nothing is counted.") {
		t.Errorf("history off: %q", tt.Texts())
	}
}

// selfFront is VoxtNew as the app in front.
func selfFront() *proto.App {
	return &proto.App{PID: os.Getpid(), BundleID: "dev.hehehai.voxtnew", Name: "VoxtNew"}
}

func TestTryPlace(t *testing.T) {
	a, _, tt := newTestApp(t)
	other := proto.App{PID: 77, BundleID: "com.apple.Notes"}
	cases := []struct {
		path   string
		target proto.App
		kind   store.Kind
		want   tryPlace
	}{
		{"/home", *selfFront(), store.KindTranscribe, placeHome},
		{"/home", *selfFront(), store.KindRewrite, placeHome},
		{"/home", other, store.KindTranscribe, placeNone},
		{"/translate", *selfFront(), store.KindTranslate, placeFeature},
		{"/translate", *selfFront(), store.KindTranscribe, placeNone},
		{"/dictionary", *selfFront(), store.KindTranscribe, placeNone},
	}
	for _, tc := range cases {
		a.navigate(tc.path)
		if got := a.tryPlace(tc.kind, tc.target); got != tc.want {
			t.Errorf("%s %s in %s: %v, want %v", tc.path, tc.kind, tc.target.BundleID, got, tc.want)
		}
	}
	a.navigate("/settings/general")
	if a.tryPlace(store.KindTranscribe, *selfFront()) != placeNone {
		t.Error("a try under the settings")
	}
	a.openOnboarding()
	a.ob.step = obTranslate
	if a.tryPlace(store.KindTranslate, *selfFront()) != placeOnboarding || a.tryPlace(store.KindTranscribe, *selfFront()) != placeIgnore {
		t.Error("the guide's places")
	}
	a.ob.step = obModels
	if a.tryPlace(store.KindTranscribe, *selfFront()) != placeIgnore {
		t.Error("a shortcut on a step without a try")
	}
	_ = tt
}

// TestHomeTryByShortcut: the shortcut pressed while Home shows (VoxtNew
// in front) runs a try on the page; in another app, a session as usual.
func TestHomeTryByShortcut(t *testing.T) {
	a, tt := homeTester(t)
	drain := queuePosts(a)
	a.hotkeyFired(proto.HotkeyEventParams{Binding: "transcribe/0", Phase: proto.PhaseTrigger, Behavior: proto.BehaviorTap, Front: selfFront()})
	frames(tt)
	if a.sess.State.Live() || a.hm.block == nil || !has(tt, "No speech model is installed.") {
		t.Fatalf("blocked try: %+v %v %q", a.sess, a.hm.block, tt.Texts())
	}
	if a.notice != nil {
		t.Error("a try's problem went to the overlay")
	}
	readyModels(a)
	drain()
	a.hotkeyFired(proto.HotkeyEventParams{Binding: "transcribe/0", Phase: proto.PhaseTrigger, Behavior: proto.BehaviorTap, Front: selfFront()})
	drain()
	frames(tt)
	if a.sess.ID == "" || !a.sess.Try || a.sess.Kind != store.KindTranscribe {
		t.Fatalf("no try: %+v", a.sess)
	}
	a.cancelSession()
	drain()
	frames(tt)
	// Done: the result shows on Home.
	s := trySnapshot(a, session.Done)
	a.sessionChanged(s)
	if !has(tt, "The meeting moves to Thursday at four.") {
		t.Errorf("result: %q", tt.Texts())
	}
}

// homeShots saves Home with everything ready, the history seeded, and a
// try that is done (TestSnapshots), at the top and scrolled to its end.
func homeShots(a *app, tt *ui.Tester, save func(string)) {
	a.host.sup.Start(context.Background()) // ready again (the shots before stopped it)
	readyModels(a)
	allowAll(a)
	a.clearNotices() // the earlier shots' cards
	a.navigate("/home")
	a.historyChanged()
	a.sessionChanged(trySnapshot(a, session.Done))
	frames(tt)
	save("page-home-ready")
	w, h := snapshotSize()
	for range 5 {
		tt.Scroll(float32(w)-100, float32(h)/2, 0, 2000)
		frames(tt)
	}
	save("page-home-ready-end")
	a.sessionChanged(session.Snapshot{Rev: a.sess.Rev + 1})
}

// sameDay reports whether a and b fall on the same local calendar day.
func sameDay(a, b time.Time) bool {
	ay, am, ad := a.Date()
	by, bm, bd := b.Date()
	return ay == by && am == bm && ad == bd
}
