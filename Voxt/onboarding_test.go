package main

import (
	"math"
	"reflect"
	"slices"
	"testing"
	"time"

	"github.com/egoist/mygo/ui"
	"github.com/hehehai/actnowkit/shortcuts"
	"github.com/sgl-project/sglang-omni/Voxt/internal/helper/proto"
	"github.com/sgl-project/sglang-omni/Voxt/internal/hotkey"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/llm"
	"github.com/sgl-project/sglang-omni/Voxt/internal/local"
	"github.com/sgl-project/sglang-omni/Voxt/internal/models"
	"github.com/sgl-project/sglang-omni/Voxt/internal/session"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
	"github.com/sgl-project/sglang-omni/Voxt/internal/store"
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

// readyASROnly installs a speech model without a language model.
func readyASROnly(a *app) {
	a.local = &fakeLocal{ASRP: &fakeTranscriptions{}}
	a.checkModels()
}

func TestOnboardingLanguageModelGuide(t *testing.T) {
	a, tt := obTester(t)
	readyASROnly(a)
	a.obGo(obTranslate)
	frames(tt)
	for _, s := range []string{
		"No language model is set up. Set one up to translate.",
		"Set up a language model",
		"Set one up on the Language models page",
		"Skip this step",
	} {
		if !has(tt, s) {
			t.Errorf("translate without LLM: no %q in %q", s, tt.Texts())
		}
	}
	if n := a.obTry(""); n == nil || n.action != actionLanguage {
		t.Fatalf("Start should still need a language model: %+v", n)
	}
	readyModels(a)
	frames(tt)
	if has(tt, "set one up on the Language models page") {
		t.Errorf("guide still shown with LLM: %q", tt.Texts())
	}
	if n := a.obTry(""); n != nil {
		t.Fatalf("Start blocked with LLM: %+v", n)
	}
	a2, tt2 := obTester(t)
	readyASROnly(a2)
	a2.obGo(obRewrite)
	frames(tt2)
	if !has(tt2, "No language model is set up. Set one up to rewrite.") {
		t.Errorf("rewrite without LLM: %q", tt2.Texts())
	}
}

func TestOnboardingSpeechDownloadGuide(t *testing.T) {
	a, tt := obTester(t)
	a.obGo(obModels)
	m, _ := models.Lookup(a.obSpeech())
	a.modelStatus[a.obSpeech()] = models.Status{Downloading: true, Done: m.Size() / 3, Total: m.Size()}
	a.obGo(obTranscribe)
	frames(tt)
	if !has(tt, "still downloading (33%)") {
		t.Errorf("download guide: %q", tt.Texts())
	}
}

func TestOnboardingSteps(t *testing.T) {
	a, tt := obTester(t)
	for _, s := range []string{"Welcome to Voxt", "Your audio stays on this Mac", "History stays on this Mac", "Get started"} {
		if !has(tt, s) {
			t.Errorf("welcome: no %q in %q", s, tt.Texts())
		}
	}
	click(t, tt, "Get started")
	if a.ob.step != obPermissions || !has(tt, "Microphone") || !has(tt, "Accessibility") {
		t.Fatalf("permissions: step %d %q", a.ob.step, tt.Texts())
	}
	click(t, tt, "Continue")
	if a.ob.step != obModels || !has(tt, "Models") || !has(tt, "Download and continue (2.5 GB)") || !has(tt, "Set up a language model") {
		t.Fatalf("models: step %d %q", a.ob.step, tt.Texts())
	}
	click(t, tt, "Skip this step")
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
	for _, s := range []string{"More to explore", "App enhancement", "Dictionary", "Your shortcuts", "Right ⌥ ⇧", "⌃ Right ⌥", "⌃⌥V", "Start using Voxt"} {
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
	click(t, tt, "Open setup guide")
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
	if a.sess.Kind == store.KindTranslate || !hasPrefix(tt, "That was the Translate shortcut. This step tests Transcribe (Right ⌥)") {
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
	obFamilyShots(a, save)  // family_ui_test.go
	obDefaultShots(a, save) // nothing on this Mac: the easy default
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
				x, ok3 := tt.Find(i18n.T("Close guide"))
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

// Each step of the guide first says what the feature is.
func TestOnboardingStepsExplainTheFeature(t *testing.T) {
	a, tt := obTester(t)
	for step, want := range map[int]string{
		obPermissions: "Voxt needs the microphone to hear you",
		obTranscribe:  "Transcribe writes down what you say",
		obTranslate:   "Translate turns the selected text",
		obRewrite:     "Rewrite writes what you ask for",
	} {
		a.obGo(step)
		if !has(tt, want) {
			t.Errorf("step %d: no %q in %q", step, want, tt.Texts())
		}
	}
}

// TestOnboardingComesBack: the bug was a guide left from its models step
// for the Language models page, which then opened again at Welcome. That
// page offers the way back to the models step; a guide closed before its
// end opens again at its step, from Settings too; one finished starts
// over.
func TestOnboardingComesBack(t *testing.T) {
	a, tt := obTester(t)
	click(t, tt, "Get started")
	click(t, tt, "Continue")
	click(t, tt, "Set up a language model")
	if a.ob.open || a.shownPath() != remotePath || !has(tt, "The setup guide is paused at the Models step.") {
		t.Fatalf("left: open %v, %s, %q", a.ob.open, a.shownPath(), tt.Texts())
	}
	click(t, tt, "Back to the setup guide")
	if !a.ob.open || a.ob.step != obModels || !has(tt, "Set up a language model") {
		t.Fatalf("back: open %v, step %d", a.ob.open, a.ob.step)
	}
	a.closeOnboarding() // its ×
	a.navigate("/settings/general")
	frames(tt)
	tt.Scroll(windowWidth*0.7, windowHeight*0.6, 0, 600) // the row is under the fold
	click(t, tt, "Open setup guide")
	if !a.ob.open || a.ob.step != obModels {
		t.Fatalf("from Settings: open %v, step %d", a.ob.open, a.ob.step)
	}
	a.obGo(obMore)
	click(t, tt, "Start using Voxt")
	a.openOnboarding()
	if a.ob.step != obWelcome {
		t.Errorf("after finishing: step %d", a.ob.step)
	}
}

// TestOnboardingTryStepLeaves: #3917, Set up a language model on the
// Translate and Rewrite try steps navigated behind the guide, so it
// seemed to do nothing. It leaves like the Models step's button.
func TestOnboardingTryStepLeaves(t *testing.T) {
	for _, c := range []struct {
		step  int
		title string
	}{{obTranslate, "Translate"}, {obRewrite, "Rewrite"}} {
		a, tt := obTester(t)
		readyASROnly(a)
		a.obGo(c.step)
		frames(tt)
		tt.Scroll(windowWidth*0.6, windowHeight*0.5, 0, 600) // the notice is under the fold
		click(t, tt, "Set up a language model")
		if a.ob.open || a.shownPath() != remotePath || !has(tt, "The setup guide is paused at the "+c.title+" step.") {
			t.Fatalf("%s: open %v, %s, %q", c.title, a.ob.open, a.shownPath(), tt.Texts())
		}
		click(t, tt, "Back to the setup guide")
		if !a.ob.open || a.ob.step != c.step {
			t.Fatalf("%s back: open %v, step %d", c.title, a.ob.open, a.ob.step)
		}
	}
}

// TestKeyboardLit: the setup guide's keyboard lights the keys of the
// shortcut: a modifier on its side, on both sides when either does, fn,
// the space bar (#3644: "Right ⌥" did not say which physical key).
func TestOnboardingViewStateAdapterUsesExplicitLocale(t *testing.T) {
	previousLocale := i18n.Locale()
	t.Cleanup(func() { i18n.SetLocale(previousLocale) })
	a, _ := obTester(t)
	a.ob.legacyAvailable, a.ob.legacyDone = true, true
	a.ob.legacy.todo = []legacyTodo{{text: "Finish custom setup", kind: local.LLM, id: llm.Custom}}
	a.famConflicts = []shortcuts.Conflict{{Mine: shortcuts.Shortcut{ID: "transcribe"}, Other: shortcuts.Holder{Name: "AudioSmith"}, Theirs: shortcuts.Shortcut{ID: "transcribe"}}}
	translator := i18n.NewTranslator("en")
	for _, step := range []int{obWelcome, obTranscribe, obMore} {
		a.ob.step = step
		for _, globalLocale := range []string{"zh-Hans", "en"} {
			i18n.SetLocale(globalLocale)
			state := a.onboardingViewStateForTranslator(translator)
			switch step {
			case obWelcome:
				if got := state.Welcome.Legacy.Todos[0].ServiceName; got != "Custom (OpenAI-compatible)" {
					t.Errorf("global locale %s changed service name to %q", globalLocale, got)
				}
			case obTranscribe:
				if got := state.Try.Conflicts[0]; got != "Also used by AudioSmith (Transcribe)" {
					t.Errorf("global locale %s changed conflict to %q", globalLocale, got)
				}
			case obMore:
				if got := state.More.Shortcuts[0].Problems[0]; got != "Also used by AudioSmith (Transcribe)" {
					t.Errorf("global locale %s changed conflict to %q", globalLocale, got)
				}
			}
		}
	}
}

func TestOnboardingWelcomeViewStateCache(t *testing.T) {
	a, _ := obTester(t)
	first := a.onboardingViewState()
	if second := a.onboardingViewState(); !reflect.DeepEqual(second, first) {
		t.Fatal("unchanged welcome state differs")
	}
	a.ob.tries[obTranscribe] = "mine"
	snapshot := obSnap(a, "mine", store.KindTranscribe, session.Done)
	snapshot.Rev = a.sess.Rev + 1
	a.sessionChanged(snapshot)
	if third := a.onboardingViewState(); !third.Passed[obTranscribe] {
		t.Fatal("cached welcome kept stale breadcrumbs")
	}
	a.updateSettings(func(s *settings.Settings) { s.General.Language = "zh-Hans" })
	if fourth := a.onboardingViewState(); fourth.StepTitles[obWelcome] != "欢迎" {
		t.Fatalf("cached welcome kept the old locale: %q", fourth.StepTitles[obWelcome])
	}
}

func TestOnboardingTimingsText(t *testing.T) {
	snapshot := session.Snapshot{ASR: session.Model{Name: "Speech One"}, LLM: session.Model{Name: "Language Two"}}
	if got := onboardingTimingsText(i18n.NewTranslator("en"), snapshot); got != "Speech One, Language Two" {
		t.Fatalf("model names %q", got)
	}
}

func TestNoticeActionLabels(t *testing.T) {
	for _, test := range []struct {
		action, english, chinese string
	}{
		{actionSignIn, "Sign in", "登录"},
		{actionUsage, "Manage usage", "管理用量"},
		{actionModels, "Open Speech models", "打开语音识别模型页"},
		{actionLanguage, "Open Language models", "打开语言模型页"},
		{actionPermissions, "Open Permissions", "打开权限页"},
		{actionHelper, "Restart helper", "重启辅助进程"},
		{actionRetry, "Retry", "重试"},
		{"unknown", "", ""},
	} {
		t.Run(test.action, func(t *testing.T) {
			for _, locale := range []string{"en", "zh-Hans"} {
				want := test.english
				if locale == "zh-Hans" {
					want = test.chinese
				}
				translator := i18n.NewTranslator(locale)
				onboardingState := buildOnboardingTry(translator, onboardingViewStateInput{
					Step: obTranscribe, Passed: map[int]bool{}, Tries: map[int]string{}, Settings: settings.Defaults(),
					Block: &sessionNotice{action: test.action},
				})
				if got := onboardingState.Card.Block.ActionLabel; got != want {
					t.Errorf("%s onboarding action label %q, want %q", locale, got, want)
				}
				tryState := buildTryCardViewState(translator, tryCardStateInput{Block: &tryNoticeStateInput{Action: test.action}})
				if got := tryState.Block.ActionLabel; got != want {
					t.Errorf("%s try action label %q, want %q", locale, got, want)
				}
			}
		})
	}
}

func TestOnboardingMoreFeatureTitles(t *testing.T) {
	for _, test := range []struct {
		locale string
		want   []string
	}{
		{"en", []string{"Transcribe", "Translate", "Rewrite", "Paste the last result"}},
		{"zh-Hans", []string{"转录", "翻译", "改写", "粘贴最近结果"}},
	} {
		state := buildOnboardingMore(i18n.NewTranslator(test.locale), onboardingViewStateInput{Settings: settings.Defaults()})
		got := make([]string, len(state.Shortcuts))
		for index, shortcut := range state.Shortcuts {
			got[index] = shortcut.Title
		}
		if !slices.Equal(got, test.want) {
			t.Errorf("%s feature titles %q, want %q", test.locale, got, test.want)
		}
	}
}

func TestRecordingElapsedText(t *testing.T) {
	now := time.Date(2026, time.January, 2, 3, 4, 5, 0, time.UTC)
	for _, test := range []struct {
		elapsed time.Duration
		want    string
	}{
		{0, "0:00"},
		{500 * time.Millisecond, "0:00"},
		{1500 * time.Millisecond, "0:01"},
		{time.Minute + 500*time.Millisecond, "1:00"},
		{time.Minute + 1500*time.Millisecond, "1:01"},
	} {
		snapshot := session.Snapshot{ID: "mine", Kind: store.KindTranscribe, State: session.Recording, HeardAt: now.Add(-test.elapsed)}
		tryState := buildTryCardViewState(i18n.NewTranslator("en"), tryCardStateInput{Mine: true, Snapshot: snapshot, Now: now})
		if got := tryState.Recording.ElapsedText; got != test.want {
			t.Errorf("%s try elapsed %q, want %q", test.elapsed, got, test.want)
		}
		onboardingState := buildOnboardingTry(i18n.NewTranslator("en"), onboardingViewStateInput{
			Step: obTranscribe, Passed: map[int]bool{}, Tries: map[int]string{obTranscribe: snapshot.ID},
			Settings: settings.Defaults(), Session: snapshot, Now: now,
		})
		if got := onboardingState.Card.ElapsedText; got != test.want {
			t.Errorf("%s onboarding elapsed %q, want %q", test.elapsed, got, test.want)
		}
	}
}

func TestKeyboardLit(t *testing.T) {
	for _, c := range []struct {
		keys string
		want []string
	}{
		{"RightOption", []string{"option-right"}},
		{"Option+Space", []string{"option-left", "option-right", "space"}},
		{"Fn+Shift", []string{"fn", "shift-left", "shift-right"}},
		{"LeftCommand+C", []string{"command-left"}},
	} {
		chord, err := hotkey.Parse(c.keys)
		if err != nil {
			t.Fatalf("%s: %v", c.keys, err)
		}
		lit := keyboardLit(chord)
		var got []string
		for id := range lit {
			got = append(got, id)
		}
		slices.Sort(got)
		if !slices.Equal(got, c.want) {
			t.Errorf("%s: lit %v, want %v", c.keys, got, c.want)
		}
		for _, id := range c.want {
			if !slices.ContainsFunc(keyboardRows, func(row []keyboardKey) bool {
				return slices.ContainsFunc(row, func(k keyboardKey) bool { return k.id == id })
			}) {
				t.Errorf("%s: no key %q in the drawing", c.keys, id)
			}
		}
	}
}
