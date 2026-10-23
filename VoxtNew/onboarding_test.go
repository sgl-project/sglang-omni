package main

import (
	"math"
	"testing"

	"github.com/egoist/mygo/ui"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/helper/proto"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/i18n"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/models"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/session"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/settings"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/store"
)

// obTester is the test app with the setup guide open.
func obTester(t *testing.T) (*app, *ui.Tester) {
	t.Helper()
	a, _, tt := newTestApp(t)
	a.maybeOnboard()
	frames(tt)
	if !a.ob.open {
		t.Fatal("a fresh install did not open the guide")
	}
	return a, tt
}

// obSnap is a try's snapshot of kind in state st.
func obSnap(a *app, id string, kind store.Kind, st session.State) session.Snapshot {
	s := trySnapshot(a, st)
	s.ID, s.Kind = id, kind
	return s
}

func TestOnboardingSteps(t *testing.T) {
	a, tt := obTester(t)
	for _, s := range []string{"Welcome to VoxtNew", "Your audio and text stay on this Mac", "History stays on this Mac", "Get started"} {
		if !has(tt, s) {
			t.Errorf("welcome: no %q in %q", s, tt.Texts())
		}
	}
	click(t, tt, "Get started")
	if a.ob.step != obPermissions || !has(tt, "Microphone") || !has(tt, "Accessibility") {
		t.Fatalf("permissions: step %d %q", a.ob.step, tt.Texts())
	}
	click(t, tt, "Continue")
	if a.ob.step != obModels || !has(tt, "Local models") || !has(tt, "Download both (5.8 GB)") {
		t.Fatalf("models: step %d %q", a.ob.step, tt.Texts())
	}
	click(t, tt, "Continue")
	if a.ob.step != obTranscribe || !has(tt, "Try Transcribe") || !has(tt, "Skip this step") {
		t.Fatalf("transcribe: step %d %q", a.ob.step, tt.Texts())
	}
	// Continue waits for a try that passes.
	click(t, tt, "Continue")
	if a.ob.step != obTranscribe {
		t.Fatal("Continue moved on without a try")
	}
	// Start without a service: why, on the step.
	a.ob.block = a.obTry("") // the guide's Start (Home's, under it, has the same label)
	if !has(tt, "No speech model is installed.") {
		t.Errorf("blocked: %q", tt.Texts())
	}
	click(t, tt, "Skip this step")
	if a.ob.step != obTranslate || !has(tt, "Translate this text") {
		t.Fatalf("translate: step %d %q", a.ob.step, tt.Texts())
	}
	click(t, tt, "Back")
	if a.ob.step != obTranscribe {
		t.Fatal("Back")
	}
	a.obGo(obMore)
	frames(tt)
	for _, s := range []string{"More to explore", "App enhancement", "Dictionary", "Your shortcuts", "Right ⌥ ⇧", "⌃ Right ⌥", "⌃⌥V", "Start using VoxtNew"} {
		if !has(tt, s) {
			t.Errorf("more: no %q in %q", s, tt.Texts())
		}
	}
	click(t, tt, "Open Dictionary")
	if a.ob.open || a.router.Path() != "/dictionary" {
		t.Errorf("open dictionary: %v %s", a.ob.open, a.router.Path())
	}
	if !a.prefs.Setup.Done || !a.lib.settings.Get().Setup.Done {
		t.Error("the guide is not marked done")
	}
	// Done: it does not open again by itself; Settings → General opens it.
	a.ob.open = false
	a.maybeOnboard()
	if a.ob.open {
		t.Error("it opened again")
	}
	a.navigate("/settings/general")
	frames(tt)
	tt.Scroll(windowWidth*0.7, windowHeight*0.6, 0, 600) // the row is under the fold
	click(t, tt, "Open the setup guide")
	if !a.ob.open || a.settingsOpen || a.ob.step != obWelcome {
		t.Errorf("reopen: %v %v %d", a.ob.open, a.settingsOpen, a.ob.step)
	}
	tt.Key(0, ui.KeyEscape)
	frames(tt)
	if a.ob.open {
		t.Error("Escape left it open")
	}
}

// TestOnboardingTryPasses: only the step's own session passes it, done
// with a result; another session, an older one or a failure does not.
func TestOnboardingTryPasses(t *testing.T) {
	a, tt := obTester(t)
	drain := queuePosts(a)
	readyModels(a)
	drain()
	a.obGo(obTranscribe)
	frames(tt)
	// The shortcut, VoxtNew in front: a try of this step.
	a.hotkeyFired(proto.HotkeyEventParams{Binding: "transcribe/0", Phase: proto.PhaseTrigger, Behavior: proto.BehaviorTap, Front: selfFront()})
	drain()
	frames(tt)
	id := a.ob.tries[obTranscribe]
	if id == "" || a.sess.ID != id || !a.sess.Try {
		t.Fatalf("no try: %q %+v", id, a.sess)
	}
	// While it runs, nothing moves.
	rec := obSnap(a, id, store.KindTranscribe, session.Recording)
	rec.Rev = a.sess.Rev + 1
	a.sessionChanged(rec)
	if !a.obBusy() {
		t.Fatal("not busy")
	}
	a.obGo(obTranslate)
	if a.ob.step != obTranscribe {
		t.Fatal("moved while the try runs")
	}
	end := obSnap(a, id, store.KindTranscribe, session.Cancelled)
	end.Rev = a.sess.Rev + 1
	a.sessionChanged(end)
	drain()
	frames(tt)

	// Another feature's shortcut does not start anything here.
	a.hotkeyFired(proto.HotkeyEventParams{Binding: "translate/0", Phase: proto.PhaseTrigger, Behavior: proto.BehaviorTap, Front: selfFront()})
	frames(tt)
	if a.sess.Kind == store.KindTranslate || !hasPrefix(tt, "That was the Translate shortcut: this step tries Transcribe (Right ⌥)") {
		t.Errorf("translate's shortcut: %+v %q", a.sess, tt.Texts())
	}

	// Some other session that ends done does not pass the step.
	a.sessionChanged(obSnap(a, "someone-else", store.KindTranscribe, session.Done))
	if a.ob.passed[obTranscribe] {
		t.Fatal("another session passed the step")
	}
	// Its own, failed: not passed; done with a result: passed.
	a.ob.tries[obTranscribe] = "mine"
	a.sessionChanged(obSnap(a, "mine", store.KindTranscribe, session.Failed))
	if a.ob.passed[obTranscribe] {
		t.Fatal("a failure passed the step")
	}
	a.ob.tries[obTranscribe] = "mine-2"
	a.sessionChanged(obSnap(a, "mine-2", store.KindTranscribe, session.Done))
	if !a.ob.passed[obTranscribe] || !has(tt, "It works. Continue, or try again.") || !has(tt, "The meeting moves to Thursday at four.") {
		t.Fatalf("not passed: %v %q", a.ob.passed, tt.Texts())
	}
	click(t, tt, "Continue")
	if a.ob.step != obTranslate {
		t.Fatalf("step %d", a.ob.step)
	}
	// The last step's session, late, does not show or pass this one.
	a.sessionChanged(obSnap(a, "mine-2", store.KindTranscribe, session.Done))
	if a.ob.passed[obTranslate] || has(tt, "It works. Continue, or try again.") {
		t.Error("a late session passed Translate")
	}
	// Translate the sample: a try with the text, no recording.
	click(t, tt, "Translate this text")
	drain()
	// (The test's own snapshots ran ahead of the manager's revisions: ask
	// the manager.)
	if cur := a.mgr.Current(); cur.ID == "" || cur.ID != a.ob.tries[obTranslate] || cur.Kind != store.KindTranslate || !cur.Try {
		t.Fatalf("sample: %+v %v", cur, a.ob.tries)
	}
	// Closing the guide cancels its try.
	live := obSnap(a, a.ob.tries[obTranslate], store.KindTranslate, session.Composing)
	live.Rev = a.sess.Rev + 1
	a.sessionChanged(live)
	a.closeOnboarding()
	drain()
	frames(tt)
	if a.ob.open {
		t.Error("still open")
	}
}

// onboardingShots saves each step of the setup guide (TestSnapshots):
// the welcome, permissions and the models (a speech model only), the try
// steps with a try that passed (Transcribe) or that runs (Rewrite).
func onboardingShots(a *app, tt *ui.Tester, save func(string)) {
	names := []string{"welcome", "permissions", "models", "transcribe", "translate", "rewrite", "more"}
	a.openOnboarding()
	for step, name := range names {
		a.ob.step = step
		switch step {
		case obModels: // a speech model installed, the language model downloading
			a.local = &fakeLocal{ASRP: &fakeTranscriptions{}}
			m, _ := models.Lookup(models.LLMBalanced)
			a.modelStatus[models.LLMBalanced] = models.Status{Downloading: true, Done: m.Size() / 4, Total: m.Size()}
		case obTranscribe:
			readyModels(a)
			a.ob.tries[step] = "ob-try"
			s := obSnap(a, "ob-try", store.KindTranscribe, session.Done)
			a.sessionChanged(s)
		case obTranslate:
			a.sessionChanged(session.Snapshot{Rev: a.sess.Rev + 1})
		case obRewrite:
			a.ob.tries[step] = "ob-rw"
			s := obSnap(a, "ob-rw", store.KindRewrite, session.Recording)
			s.Partial = "write a short note to the team saying"
			a.sessionChanged(s)
		case obMore:
			a.sessionChanged(session.Snapshot{Rev: a.sess.Rev + 1})
		}
		save("onboarding-" + itoa(step+1) + "-" + name)
	}
	// Last, as they change the models' statuses (the page behind scrolls).
	a.ob.step = obModels
	obFamilyShots(a, save) // family_ui_test.go
	a.ob.open = false
}

// TestOnboardingDialogSize: 600×440 in the default window, smaller than
// the settings dialog with a margin all round in the smallest, and
// obMaxW × obMaxH in a large one.
func TestOnboardingDialogSize(t *testing.T) {
	th := ui.LightTheme()
	for _, c := range []struct{ w, h, dw, dh float32 }{
		{windowWidth, windowHeight, obMaxW, obMaxH},
		{windowMinWidth, windowMinHeight, obMaxW, 374},
		{1600, 1000, obMaxW, obMaxH},
	} {
		dw, dh := onboardingDialogSize(th, c.w, c.h)
		if math.Abs(float64(dw-c.dw)) > 0.5 || math.Abs(float64(dh-c.dh)) > 0.5 {
			t.Errorf("%v×%v: %v×%v, want %v×%v", c.w, c.h, dw, dh, c.dw, c.dh)
		}
		sw, sh := settingsDialogSize(th, c.w, c.h)
		if dw > sw || dh > sh {
			t.Errorf("%v×%v: %v×%v, larger than the settings dialog's %v×%v", c.w, c.h, dw, dh, sw, sh)
		}
		if mx, my := (c.w-dw)/2, (c.h-dh)/2; mx < 24 || my < 24 {
			t.Errorf("%v×%v: margins %v, %v", c.w, c.h, mx, my)
		}
	}
}

// TestOnboardingStepsOneLine: the breadcrumbs keep to one line, clear of
// ×, with every try step checked, in the default and smallest windows.
func TestOnboardingStepsOneLine(t *testing.T) {
	for _, lang := range []string{"en", "zh-Hans"} {
		for _, size := range [][2]int{{windowWidth, windowHeight}, {windowMinWidth, windowMinHeight}} {
			a, tt := obTester(t)
			if lang != "en" {
				a.updateSettings(func(s *settings.Settings) { s.General.Language = lang })
			}
			tt.SetSize(size[0], size[1])
			for _, step := range []int{obTranscribe, obTranslate, obRewrite} {
				a.ob.passed[step] = true
			}
			for _, step := range []int{obWelcome, obMore} {
				a.ob.step = step
				frames(tt)
				titles := obStepTitles()
				first, ok1 := tt.Find(titles[0])
				last, ok2 := tt.Find(titles[len(titles)-1])
				x, ok3 := tt.Find(i18n.T("Close the guide"))
				if !ok1 || !ok2 || !ok3 {
					t.Fatalf("%s %v: steps or × not found in %q", lang, size, tt.Texts())
				}
				if math.Abs(float64(first.Y-last.Y)) > 1 || last.X+last.W > x.X {
					t.Errorf("%s %v step %d: %q at %+v, %q at %+v, × at %+v", lang, size, step, titles[0], first, titles[len(titles)-1], last, x)
				}
			}
		}
	}
}
