package main

// Sessions as the UI sees them: the session manager
// (internal/session) wired to the helper, the local models and the store;
// the shortcuts (internal/hotkey) sent to the helper's event tap, with
// MyGo's Carbon shortcuts as the fallback without Accessibility; the
// snapshot the overlay, the Transcribe page and the menu bar read; and
// what their buttons do (stop, cancel, insert, copy, try again, another
// target language, paste the last result). The answer card also takes a
// Rewrite follow-up: the Rewrite shortcut while it shows a Rewrite
// answer records the next instruction for it.
//
// The manager runs sessions on goroutines; its snapshots come back
// through a.post (sessionChanged), newest revision wins. Starting a
// session checks on the main thread what can be checked without I/O (the
// helper, a speech model), so that a missing model shows a notice
// instead of switching the microphone on.

import (
	"context"
	"errors"
	"os"
	"time"

	"github.com/egoist/mygo"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/compose"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/core"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/helper/proto"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/hotkey"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/i18n"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/llm"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/local"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/session"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/settings"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/store"
)

// sessionSource is the logs' source of session lines.
const sessionSource = "session"

// liveFeatures are the features whose shortcuts are bound.
var liveFeatures = []string{hotkey.Transcribe, hotkey.Translate, hotkey.Rewrite}

// sessionState is the sessions' view state.
type sessionState struct {
	mgr *session.Manager
	// sess is the current (or last) session, as its newest snapshot said.
	sess session.Snapshot
	// answer is the answer card: open while a result waits there.
	answer answerCard
	// notice is the overlay's one-line message, when it shows one;
	// noticeSeq numbers them, so that a notice's timer hides only itself.
	notice    *sessionNotice
	noticeSeq int
	// hotkeysSent is the hotkey.configure last sent (as JSON), so the
	// same one is not sent again; carbon the accelerators registered
	// with MyGo while the event tap is missing.
	hotkeysSent string
	carbon      []string
	// lastLoaded: the last result was read from the history at start.
	lastLoaded bool
	// followUp is the answer card's session a running Rewrite follows up
	// on ("" for none): its answer joins the card's conversation.
	followUp string
}

// answerCard is the answer card's state (overlay_card.go draws it).
type answerCard struct {
	open      bool
	id        string // the session whose result it shows
	kind      store.Kind
	text      string
	title     string // Rewrite's answer title ("" for none)
	target    proto.App
	inserting bool   // Insert pastes: the card is hidden meanwhile (Voxt's)
	msg       string // a line under the text: why Insert, or another language, did not work
	// why says why the result is in the card and not in place (Insert's
	// tip: Voxt says nothing; one line of orange was too loud, 10-15).
	why string
	// problem: the session failed (the card shows why, and Try again);
	// input is what it worked on, for Try again and another language.
	problem  *session.Problem
	input    session.TextInput
	lang     string // Translate's target
	langOpen bool   // the language menu shows (overlay_lang.go)
	// turns are a Rewrite conversation so far, the shown answer last
	// (a follow-up adds to them); conversation: a follow-up began, and
	// the card shows them as one (Voxt's conversation mode), from then on.
	turns        []compose.Turn
	conversation bool
	// redo is the session translating the text into redoLang, another
	// language chosen in the card, while it runs: the card keeps the text
	// it shows, with a spinner, until the new one replaces it (Voxt's).
	redo, redoLang string
	// copied: Copy's mark shows (2 s, Voxt's); copySeq numbers the
	// copies, so that a copy's timer clears only its own mark.
	copied  bool
	copySeq int
}

// Notice actions: where the notice's button goes.
const (
	actionNone        = ""
	actionModels      = "models"
	actionPermissions = "permissions"
	actionHelper      = "helper"
	actionRetry       = "retry" // the session's entry, in History (history_retry.go)
)

// sessionNotice is a one-line message in the overlay, where
// the pill's waveform goes: an error and a way forward (a click), hidden
// after a few seconds.
type sessionNotice struct {
	text, more string
	action     string
	entry      string // actionRetry's history entry
	warn       bool   // a warning (the session went on), not an error
	seq        int    // which notice: its timer hides only itself
	// then runs when its time is up (not when a click or the next
	// session hides it): the notice that follows it.
	then func()
}

// noticeTime is how long a notice shows.
const noticeTime = 5 * time.Second

// openSessions makes the session manager; connectHelper calls it once the
// supervisor exists.
func (a *app) openSessions() {
	sup := a.host.sup
	lm := a.local
	lib := a.lib
	logs := a.logs
	st := a.settings
	a.mgr = session.New(session.Config{
		Audio:   session.HelperAudio{H: sup},
		Text:    a.textIO(), // family_runtime.go: the family's paste rule
		Select:  session.HelperText{H: sup},
		Resolve: resolver(lm),
		History: lib.store.History(),
		Glossary: func(ctx context.Context) ([]store.Term, error) {
			return lib.store.Dictionary().List(ctx, "")
		},
		Settings: st.Get,
		Locale:   i18n.Locale,
		Scene:    a.sceneConfig(), // appenhance.go
		Self:     os.Getpid(),
		Activate: func(pid int) bool { ok, _ := activateTarget(pid); return ok },
		OnChange: func(s session.Snapshot) { a.post(func() { a.sessionChanged(s) }) },
		Log: func(level, msg string) {
			logs.Append(core.LogEntry{Time: time.Now(), Level: core.Severity(level), Source: sessionSource, Message: msg})
		},
		NewID: func() string { return core.NewID(time.Now()) },
	})
	a.sess = a.mgr.Current()
}

// resolver opens a session's local models (off the main thread: opening
// one may start its runtime). A model that is not installed stays nil:
// the session says what it lacks when it needs it (a selection to
// translate needs no speech model).
func resolver(lm local.Services) session.Resolver {
	return func(ctx context.Context, kind session.Kind, s settings.Settings) (session.Services, error) {
		f := featureOf(s, string(kind))
		var svc session.Services
		p, m, err := lm.ASR(ctx, f.ASR)
		switch {
		case err == nil:
			svc.ASR, svc.ASRInfo = p, sessionModel(m)
		case !errors.Is(err, local.ErrNotInstalled):
			return svc, err
		}
		if va, ok := svc.ASR.(local.VoiceActivity); ok {
			// No model: speech is found by energy.
			svc.VAD = va.VAD()
		}
		if kind == store.KindTranscribe && !s.Features.Transcribe.Enhance {
			return svc, nil
		}
		// A model that serves the feature (the translation model only
		// translates).
		m, ok := local.Choose(local.ForFeature(lm.Installed(local.LLM), string(kind)), f.LLM)
		if !ok {
			return svc, nil
		}
		c, ml, err := lm.LLM(ctx, m.ID)
		if err != nil {
			return svc, nil // no language model: the text goes on as recognised
		}
		svc.LLM, svc.LLMInfo = modelClient{c: c}, sessionModel(ml)
		return svc, nil
	}
}

// sessionModel is how a session (and its history entry) names a model.
func sessionModel(m local.Model) session.Model {
	return session.Model{ID: m.ID, Name: m.Name, Model: m.ID}
}

// modelClient sends a session's chat to its model at the skill's
// temperature.
type modelClient struct{ c local.Chat }

func (m modelClient) Chat(ctx context.Context, req llm.Request) (llm.Response, error) {
	if req.Temperature == nil {
		req.Temperature = llm.F64(llm.DefaultTemperature(req.Skill))
	}
	return m.c.Chat(ctx, req)
}

// Prefill warms the model's prompt cache up (llm.Prefiller), when it has
// one.
func (m modelClient) Prefill(ctx context.Context, req llm.Request) error {
	if p, ok := m.c.(llm.Prefiller); ok {
		return p.Prefill(ctx, req)
	}
	return nil
}

// sessionChanged takes a session's snapshot (the newest wins) and
// updates the overlay, the shortcuts and the menu bar.
func (a *app) sessionChanged(s session.Snapshot) {
	if s.Rev <= a.sess.Rev {
		return
	}
	was := a.sess
	a.sess = s
	for _, fn := range a.onSession { // events.go: the setup guide, the sounds
		fn(was, s)
	}
	if s.ID != was.ID {
		a.notice = nil
	}
	if s.State.Ended() && !was.State.Ended() {
		a.sessionEnded(s)
	}
	if s.State.Live() != was.State.Live() {
		a.syncHotkeys()
	}
	for _, fn := range a.afterSession { // events.go: the menu bar, the overlay
		fn(was, s)
	}
}

// sessionEnded opens the answer card or shows a notice, and refreshes the
// history.
func (a *app) sessionEnded(s session.Snapshot) {
	for _, fn := range a.onSessionEnd { // events.go: Home
		fn(s)
	}
	if s.Try {
		a.historyChanged() // a try keeps nothing, but the pages may show it
		return
	}
	switch {
	case a.answer.open && a.answer.redo != "" && a.answer.redo == s.ID:
		a.endRedo(s)
	case s.Card && (s.Result != "" || s.State == session.Failed):
		a.openAnswer(s)
		a.syncHotkeys() // Escape closes the card
	}
	if s.ID != "" && a.followUp != "" && !s.Card {
		a.followUp = "" // a follow-up that ended without an answer: the card keeps the last one
	}
	if p := s.Problem; p != nil && s.State != session.Cancelled && !(s.Card && s.State == session.Failed) {
		text, more, action := sessionProblemText(p)
		if text != "" {
			// Nothing heard is not a failure: a warning.
			n := sessionNotice{text: text, more: more, action: action, warn: p.Degraded || p.Code == session.ProblemNoSpeech}
			if p.Code == session.ProblemASR && s.Recording {
				n.action, n.entry = actionRetry, s.ID // the entry kept the recording: Retry
			}
			a.showNotice(n)
		}
	}
	a.memoryNoticeAfter(s) // memory_notice.go
	if s.State != session.Cancelled {
		a.historyChanged()
	}
}

// showNotice shows a notice in the overlay; it hides by itself.
func (a *app) showNotice(n sessionNotice) {
	a.noticeSeq++
	seq := a.noticeSeq
	n.seq = seq
	a.notice = &n
	a.overlaySync()
	a.later(noticeTime, func() {
		if a.notice != nil && a.notice.seq == seq {
			if then := a.notice.then; then != nil {
				then() // in its place, without hiding the overlay between
			} else {
				a.closeNotice()
			}
		}
	})
}

// closeNotice hides the notice.
func (a *app) closeNotice() {
	a.notice = nil
	a.overlaySync()
}

// noticeAction runs a notice's button.
func (a *app) noticeAction(action string) {
	entry := ""
	if a.notice != nil {
		entry = a.notice.entry
	}
	a.closeNotice()
	switch action {
	case actionRetry:
		a.retryFromNotice(entry)
	case actionModels:
		a.showMainWindow()
		a.navigate(modelsPath)
	case actionPermissions:
		a.showMainWindow()
		a.navigate("/settings/permissions")
	case actionHelper:
		a.restartHelper()
	}
}

// closeSessions cancels a running session and removes the Carbon
// shortcuts; it runs on quit.
func (a *app) closeSessions() {
	a.cancelSession()
	if a.hasMenu {
		for _, acc := range a.carbon {
			mygo.GlobalShortcut.Unregister(acc)
		}
	}
	a.carbon = nil
}
