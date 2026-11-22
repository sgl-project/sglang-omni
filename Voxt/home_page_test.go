package main

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/egoist/mygo/ui"
	"github.com/sgl-project/sglang-omni/Voxt/internal/dictation"
	"github.com/sgl-project/sglang-omni/Voxt/internal/helper/helpertest"
	"github.com/sgl-project/sglang-omni/Voxt/internal/helper/proto"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/local"
	"github.com/sgl-project/sglang-omni/Voxt/internal/permission"
	"github.com/sgl-project/sglang-omni/Voxt/internal/session"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
	"github.com/sgl-project/sglang-omni/Voxt/internal/store"
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

func TestSloganHint(t *testing.T) {
	join := func(h *settings.Hotkey) string {
		b, k, a := sloganHintFor(i18n.CurrentTranslator(), h)
		return b + "[" + k + "]" + a
	}
	for _, tc := range []struct {
		h      *settings.Hotkey
		en, zh string
	}{
		{&settings.Hotkey{Keys: "Fn", Trigger: settings.TriggerTap}, "Press [fn] to start", "按 [fn] 开始"},
		{&settings.Hotkey{Keys: "Fn"}, "Press [fn] to start", "按 [fn] 开始"},
		{&settings.Hotkey{Keys: "RightCommand", Trigger: settings.TriggerDouble}, "Double-press [Right ⌘] to start", "双击 [右 ⌘] 开始"},
		{&settings.Hotkey{Keys: "Option+Space", Trigger: settings.TriggerLong}, "Hold [⌥Space] and speak", "按住 [⌥空格] 说话"},
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
	for _, s := range []string{"Speak anywhere and Voxt types at your cursor", "Press", "Right ⌥", "to start"} {
		if !hasExact(tt, s) {
			t.Errorf("no %q in %q", s, tt.Texts())
		}
	}
	a.updateSettings(func(s *settings.Settings) {
		s.Features.Transcribe.Hotkeys = []settings.Hotkey{{Keys: "Option+Space", Trigger: settings.TriggerLong}, {Keys: "Fn"}}
	})
	if !hasExact(tt, "Hold") || !hasExact(tt, "⌥Space") || !hasExact(tt, "and speak") {
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

func TestHomeLastSpeechModel(t *testing.T) {
	a, tt := homeTester(t)
	if err := a.lib.store.History().Put(context.Background(), store.Entry{
		ID: "spoken", Kind: store.KindTranscribe, CreatedAt: time.Now(), Status: store.StatusDone,
		Result: "Hello.", ASRModel: "fixture-speech-model",
	}); err != nil {
		t.Fatal(err)
	}
	a.historyChanged()
	if !hasExact(tt, "fixture-speech-model") {
		t.Fatalf("the last speech model is missing: %q", tt.Texts())
	}
	a.updateSettings(func(s *settings.Settings) { s.Features.Transcribe.Hotkeys = nil })
	if !hasExact(tt, "fixture-speech-model") {
		t.Fatalf("no shortcut hid the last speech model: %q", tt.Texts())
	}
}

type homeUsageLocal struct {
	fakeLocal
	remoteName string
}

func (localServices homeUsageLocal) Installed(kind local.Kind) []local.Model {
	installed := localServices.fakeLocal.Installed(kind)
	if kind == local.ASR {
		name := localServices.remoteName
		if name == "" {
			name = "OpenAI Transcribe"
		}
		return append(installed, local.Model{Kind: local.ASR, ID: "openai", Name: name})
	}
	return installed
}

func TestHomeRemoteSpeechModel(t *testing.T) {
	a, tester := homeTester(t)
	a.local = &homeUsageLocal{fakeLocal: fakeLocal{ASRP: &fakeTranscriptions{}}, remoteName: "OpenAI Transcribe · gpt-4o-mini-transcribe"}
	entry := store.Entry{ID: "remote", Kind: store.KindTranscribe, CreatedAt: time.Now(), Status: store.StatusDone, Result: "hello", ASRProvider: "openai", ASRModel: "gpt-4o-mini-transcribe"}
	if err := a.lib.store.History().Put(context.Background(), entry); err != nil {
		t.Fatal(err)
	}
	a.historyChanged()
	if !hasExact(tester, "OpenAI Transcribe · gpt-4o-mini-transcribe") {
		t.Fatalf("remote speech model: %q", tester.Texts())
	}
	a.local.(*homeUsageLocal).remoteName = "OpenAI Transcribe · gpt-new"
	if !hasExact(tester, "OpenAI Transcribe · gpt-new · gpt-4o-mini-transcribe") {
		t.Fatalf("updated remote speech model: %q", tester.Texts())
	}
}

func TestHomeModelNameLocale(t *testing.T) {
	i18n.SetLocale("en")
	t.Cleanup(func() { i18n.SetLocale("en") })
	a, tester := homeTester(t)
	entry := store.Entry{ID: "dictation", Kind: store.KindTranscribe, CreatedAt: time.Now(), Status: store.StatusDone, Result: "hello", ASRModel: dictation.ModelID}
	if err := a.lib.store.History().Put(context.Background(), entry); err != nil {
		t.Fatal(err)
	}
	a.historyChanged()
	if !has(tester, "System dictation") {
		t.Fatalf("English model name: %q", tester.Texts())
	}
	i18n.SetLocale("zh-Hans")
	a.updateSettings(func(s *settings.Settings) { s.General.Language = "zh-Hans" })
	if !has(tester, "系统听写") {
		t.Fatalf("Chinese model name: %q", tester.Texts())
	}
}

func TestReadHomeUsageModelIdentity(t *testing.T) {
	now := time.Date(2026, 11, 20, 15, 0, 0, 0, time.Local)
	for _, test := range []struct {
		name         string
		entries      []store.Entry
		wantProvider string
		wantModel    string
		wantASRCount int
	}{
		{name: "empty new entry keeps looking", entries: []store.Entry{
			{ID: "empty", Kind: store.KindTranscribe, CreatedAt: now, Status: store.StatusDone, Result: "new"},
			{ID: "old", Kind: store.KindTranscribe, CreatedAt: now.AddDate(0, 0, -200), Status: store.StatusDone, Result: "old", ASRProvider: "openai", ASRModel: "gpt-old"},
		}, wantProvider: "openai", wantModel: "gpt-old"},
		{name: "provider-only is not overwritten", entries: []store.Entry{
			{ID: "provider", Kind: store.KindTranscribe, CreatedAt: now, Status: store.StatusDone, Result: "new", ASRProvider: "openai"},
			{ID: "model", Kind: store.KindTranscribe, CreatedAt: now.AddDate(0, 0, -10), Status: store.StatusDone, Result: "old", ASRModel: "local-model"},
		}, wantProvider: "openai"},
		{name: "provider outside the window", entries: []store.Entry{
			{ID: "old", Kind: store.KindTranscribe, CreatedAt: now.AddDate(0, 0, -200), Status: store.StatusDone, Result: "old", ASRProvider: "openai", ASRMS: 9000},
		}, wantProvider: "openai"},
	} {
		t.Run(test.name, func(t *testing.T) {
			a, _, _ := newTestApp(t)
			for _, entry := range test.entries {
				if err := a.lib.store.History().Put(context.Background(), entry); err != nil {
					t.Fatal(err)
				}
			}
			usage, err := readHomeUsage(context.Background(), a.lib.store.History(), now)
			if err != nil {
				t.Fatal(err)
			}
			if usage.lastProvider != test.wantProvider || usage.lastModel != test.wantModel || usage.asrCount != test.wantASRCount {
				t.Fatalf("usage %+v, want provider %q model %q and %d timed results", usage, test.wantProvider, test.wantModel, test.wantASRCount)
			}
		})
	}
}

func TestReadHomeUsage(t *testing.T) {
	now := time.Date(2026, 11, 20, 15, 0, 0, 0, time.Local)
	a, _, _ := newTestApp(t)
	h := a.lib.store.History()
	for _, entry := range []store.Entry{
		{ID: "failed", CreatedAt: now, Status: store.StatusFailed, Result: "", ASRModel: "failed-model"},
		{ID: "blank", CreatedAt: now.Add(-time.Hour), Status: store.StatusDone, Result: "  ", ASRModel: "blank-model"},
		{ID: "new", CreatedAt: now.Add(-2 * time.Hour), Status: store.StatusDone, Result: "new", ASRModel: "new-model", ASRMS: 1500},
		{ID: "older", CreatedAt: now.AddDate(0, 0, -10), Status: store.StatusDone, Result: "older", ASRMS: 2500},
		{ID: "ancient", CreatedAt: now.AddDate(0, 0, -200), Status: store.StatusDone, Result: "ancient", ASRModel: "ancient-model", ASRMS: 9000},
	} {
		entry.Kind = store.KindTranscribe
		if err := h.Put(context.Background(), entry); err != nil {
			t.Fatal(err)
		}
	}
	usage, err := readHomeUsage(context.Background(), h, now)
	if err != nil {
		t.Fatal(err)
	}
	if usage.lastModel != "new-model" || usage.asrMS != 4000 || usage.asrCount != 2 {
		t.Fatalf("usage %+v, want new-model and 2 results averaging 2 seconds", usage)
	}

	empty, _, _ := newTestApp(t)
	usage, err = readHomeUsage(context.Background(), empty.lib.store.History(), now)
	if err != nil || usage != (homeUsage{}) {
		t.Fatalf("empty history: %+v, %v", usage, err)
	}

	old, _, _ := newTestApp(t)
	if err := old.lib.store.History().Put(context.Background(), store.Entry{
		ID: "old", Kind: store.KindTranscribe, CreatedAt: now.AddDate(0, 0, -200), Status: store.StatusDone, Result: "old", ASRModel: "old-model", ASRMS: 9000,
	}); err != nil {
		t.Fatal(err)
	}
	usage, err = readHomeUsage(context.Background(), old.lib.store.History(), now)
	if err != nil || usage.lastModel != "old-model" || usage.asrCount != 0 {
		t.Fatalf("old history: %+v, %v", usage, err)
	}
	if groupDigits(1234567) != "1,234,567" || groupDigits(999) != "999" {
		t.Errorf("groupDigits %s %s", groupDigits(1234567), groupDigits(999))
	}
}

type countingHistory struct {
	store.History
	listCalls int
}

func (h *countingHistory) List(ctx context.Context, query store.HistoryQuery) ([]store.Entry, error) {
	h.listCalls++
	return h.History.List(ctx, query)
}

func TestReadHomeUsagePagesAndMidnight(t *testing.T) {
	now := time.Date(2026, 11, 20, 15, 0, 0, 0, time.Local)
	a, _, _ := newTestApp(t)
	history := &countingHistory{History: a.lib.store.History()}
	put := func(id, model string, at time.Time, asrMS int64) {
		t.Helper()
		if err := history.Put(context.Background(), store.Entry{
			ID: id, Kind: store.KindTranscribe, CreatedAt: at, Status: store.StatusDone,
			Result: "spoken", ASRModel: model, ASRMS: asrMS,
		}); err != nil {
			t.Fatal(err)
		}
	}
	for index := range 500 {
		put(fmt.Sprintf("recent-%03d", index), "new-model", now, 1000)
	}
	for index := range 501 {
		put(fmt.Sprintf("old-%03d", index), "old-model", now.AddDate(0, 0, -200), 9000)
	}
	usage, err := readHomeUsage(context.Background(), history, now)
	if err != nil {
		t.Fatal(err)
	}
	if history.listCalls != 2 || usage.lastModel != "new-model" || usage.asrCount != 500 {
		t.Fatalf("paged read made %d calls and returned %+v, want 2 calls and the recent window", history.listCalls, usage)
	}

	boundary, _, tt := newTestApp(t)
	boundaryTime := now.AddDate(0, 0, -182)
	if err := boundary.lib.store.History().Put(context.Background(), store.Entry{
		ID: "boundary", Kind: store.KindTranscribe, CreatedAt: boundaryTime, Status: store.StatusDone,
		Result: "boundary", ASRModel: "boundary-model", ASRMS: 1000,
	}); err != nil {
		t.Fatal(err)
	}
	usage, err = readHomeUsage(context.Background(), boundary.lib.store.History(), now)
	if err != nil || usage.asrCount != 1 {
		t.Fatalf("the window's first day is excluded: %+v, %v", usage, err)
	}
	usage, err = readHomeUsage(context.Background(), boundary.lib.store.History(), now.AddDate(0, 0, 1))
	if err != nil || usage.asrCount != 0 || usage.lastModel != "boundary-model" {
		t.Fatalf("the window did not move at midnight: %+v, %v", usage, err)
	}
	boundary.navigate("/home")
	frames(tt)
	tt.Advance(time.Second) // finish the calendar's entrance animation first
	boundary.hm.read, boundary.hm.usageDay = true, "2000-01-01"
	gen := boundary.hm.gen
	tt.Advance(time.Minute)
	if boundary.hm.gen <= gen || boundary.hm.usageDay != time.Now().Format("2006-01-02") {
		t.Fatalf("Home did not refresh on its minute frame: gen %d -> %d, day %q", gen, boundary.hm.gen, boundary.hm.usageDay)
	}
}

// selfFront is Voxt as the app in front.
func selfFront() *proto.App {
	return &proto.App{PID: os.Getpid(), BundleID: "dev.hehehai.voxtnew", Name: "Voxt"}
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
		{"/home", *selfFront(), store.KindTranscribe, placeNone},
		{"/home", *selfFront(), store.KindRewrite, placeNone},
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
	a.navigate("/translate")
	if got := a.tryPlace(store.KindTranslate, *selfFront()); got != placeFeature {
		t.Errorf("Translate's page: %v, want its try", got)
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

// TestHomeShortcutStartsNormalSession (#4199): Home has no Try area, so a
// shortcut there always starts a normal, visible, persisted session.
func TestHomeShortcutStartsNormalSession(t *testing.T) {
	for _, tried := range []bool{false, true} {
		t.Run(map[bool]string{false: "first use", true: "returning"}[tried], func(t *testing.T) {
			set, keys, _ := fileSetup(t, "hello aura link", &fakeChat{text: "Hello, Voxt."})
			self := func(afterMS int) string {
				b, _ := json.Marshal(helpertest.Script{AfterMS: afterMS, Method: proto.MethodHotkeyEvent,
					Params: mustJSON(proto.HotkeyEventParams{Binding: "transcribe/0", Phase: proto.PhaseTrigger, Behavior: proto.BehaviorTap, T: 1, Front: selfFront()})})
				return string(b)
			}
			a := newSessionApp(t, script(t, self(500), self(400)), func(s *settings.Settings) {
				set(s)
				s.Setup.Tried = tried
			}, keys)
			a.navigate("/home")
			a.pump("the end", func() bool { return a.sess.State.Ended() })
			if a.sess.Try || a.sess.State != session.Done || a.sess.Result != "Hello, Voxt." {
				t.Fatalf("end %+v", a.sess)
			}
			shown := a.sess.Inserted || a.answer.open || a.notice != nil && a.notice.text == copiedNotice
			if !shown {
				t.Errorf("result shown nowhere: inserted %v, card %v, notice %+v", a.sess.Inserted, a.answer.open, a.notice)
			}
			if es := a.history(); len(es) != 1 || es[0].Result != "Hello, Voxt." {
				t.Errorf("history %+v", es)
			}
		})
	}
}

// TestShortcutInOwnField (FB10): the shortcut pressed while a field of
// the window has the focus starts a session, with the overlay, that types
// into the field as in another app. With a button focused, a try as before.
func TestShortcutInOwnField(t *testing.T) {
	a, tt := pageTester(t, "/transcribe")
	readyModels(a)
	drain := queuePosts(a)
	click(t, tt, "Your rules")
	click(t, tt, "System prompt for Transcribe")
	if got := a.tryPlace(store.KindTranscribe, *selfFront()); got != placeNone {
		t.Fatalf("rules field focused: %v, want no try", got)
	}
	a.hotkeyFired(proto.HotkeyEventParams{Binding: "transcribe/0", Phase: proto.PhaseTrigger, Behavior: proto.BehaviorTap, Front: selfFront()})
	drain()
	frames(tt)
	if a.sess.ID == "" || a.sess.Try || !a.ov.shown {
		t.Fatalf("not a session with the overlay: %+v, overlay %v", a.sess, a.ov.shown)
	}
	a.cancelSession()
	drain()
	frames(tt)
	click(t, tt, "What the language model receives")
	if !tt.Focused("What the language model receives") {
		t.Fatal("the button did not take the focus")
	}
	if got := a.tryPlace(store.KindTranscribe, *selfFront()); got != placeFeature {
		t.Errorf("button focused: %v, want the page's try", got)
	}
}

// homeShots saves Home with its usage populated, at the top and scrolled
// to its end.
func homeShots(a *app, tt *ui.Tester, save func(string)) {
	a.host.sup.Start(context.Background())
	readyModels(a)
	allowAll(a)
	a.clearNotices()
	a.navigate("/home")
	a.historyChanged()
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

// TestWidePageCentered: in a window much wider than a page's column (full
// screen), the column sits in the middle of the page and the header's
// title and action line up with it too.
func TestWidePageCentered(t *testing.T) {
	a, _, tt := newTestApp(t)
	const w = 1700
	tt.SetSize(w, 1100)
	if err := a.lib.store.History().Put(context.Background(), store.Entry{
		ID: "wide", Kind: store.KindTranscribe, CreatedAt: time.Now(), Status: store.StatusDone,
		Result: "wide", ASRModel: "fixture-speech-model",
	}); err != nil {
		t.Fatal(err)
	}
	a.navigate("/home")
	a.historyChanged()
	frames(tt)
	slogan, ok1 := tt.Find("Speak anywhere and Voxt types at your cursor")
	guide, ok2 := tt.Find("Setup guide")
	model, ok3 := tt.Find("fixture-speech-model")
	if !ok1 || !ok2 || !ok3 {
		t.Fatalf("found slogan %v, Setup guide %v, model %v", ok1, ok2, ok3)
	}
	left := slogan.X - sidebarWidth
	right := w - (model.X + model.W)
	if math.Abs(float64(left-right)) > 20 || left < 300 {
		t.Errorf("the column: %.0f from the sidebar, %.0f from the window's edge", left, right)
	}
	if math.Abs(float64(guide.X+guide.W-(model.X+model.W))) > 20 {
		t.Errorf("Setup guide ends at %.0f, the column's model at %.0f", guide.X+guide.W, model.X+model.W)
	}
	// A page of lists keeps a wider body in the middle, its header lined
	// up with it too.
	a.navigate("/dictionary")
	frames(tt)
	search, ok1 := tt.Find("Search dictionary")
	add, ok2 := tt.Find("Add term")
	if !ok1 || !ok2 {
		t.Fatalf("Dictionary: found search %v, Add term %v", ok1, ok2)
	}
	left, right = search.X-sidebarWidth, w-(add.X+add.W)
	if math.Abs(float64(left-right)) > 30 || left < 150 {
		t.Errorf("Dictionary: %.0f from the sidebar to the search, %.0f from Add term to the window's edge", left, right)
	}
}
