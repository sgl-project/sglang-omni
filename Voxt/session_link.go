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
	"encoding/json"
	"errors"
	"os"
	"slices"
	"strings"
	"time"

	"github.com/egoist/mygo"
	"github.com/sgl-project/sglang-omni/Voxt/internal/compose"
	"github.com/sgl-project/sglang-omni/Voxt/internal/core"
	"github.com/sgl-project/sglang-omni/Voxt/internal/dict"
	"github.com/sgl-project/sglang-omni/Voxt/internal/helper/proto"
	"github.com/sgl-project/sglang-omni/Voxt/internal/hotkey"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/llm"
	"github.com/sgl-project/sglang-omni/Voxt/internal/local"
	"github.com/sgl-project/sglang-omni/Voxt/internal/providers"
	"github.com/sgl-project/sglang-omni/Voxt/internal/session"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
	"github.com/sgl-project/sglang-omni/Voxt/internal/store"
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
	// bgID is the backgrounded answer's session ("" for none) and bgSess
	// its newest snapshot: an answer that kept streaming into its card
	// while a new session started (session_background.go, wall 116).
	bgID   string
	bgSess session.Snapshot
	// answer is the answer card: open while a result waits there.
	answer answerCard
	// box is Rewrite's card with the input box (rewrite_box.go): the
	// Rewrite shortcut opens it with the microphone. boxShare is its
	// content as the sending session reads it (Config.BoxInput).
	box      rewriteBox
	boxShare boxShare
	// boxGen numbers the box's lives: a picture conversion posted for an
	// older life is dropped. boxPending counts the conversions not yet
	// landed in order, including held ones; boxSendQueued: Enter came
	// while one converted.
	boxGen, boxPending int
	boxSendQueued      bool
	// boxSeq numbers the picture conversions in drop (or paste) order;
	// one lands only after every earlier number (boxNext is the next
	// due), boxHeld keeping one whose earlier numbers still convert.
	boxSeq, boxNext int
	boxHeld         map[int]boxLanding
	// under is the card a new session took the place of: it comes back
	// when that session leaves no card (resurface, #4013).
	under answerCard
	// reopen is the history's continuable conversations, newest first.
	reopen     []reopenConversation
	restoreSeq int
	// notice is the overlay's one-line message, when it shows one;
	// noticeSeq numbers them, so that a notice's timer hides only itself.
	notice    *sessionNotice
	noticeSeq int
	// hotkeysSent is the newest hotkey.configure requested, so the same
	// one is not sent again. Sequence numbers let a newer close supersede
	// an older open even after a timeout. carbon is the accelerators
	// registered with MyGo while the event tap is missing.
	hotkeysSent             string
	hotkeyConfigureSequence uint64
	hotkeyStatusSequence    uint64
	hotkeyCardIdentity      string
	hotkeyCardGeneration    uint64
	carbon                  []string
	// hotkeyGen numbers the hotkey.configure requests sent (1 on the
	// first). The helper applies only a newer one and reports it on the
	// events it decides under it; hotkeyFired drops the rest.
	hotkeyGen int64
	// keysListening: a shortcut field takes pressed keys (key_recorder.go),
	// and the shortcuts are off meanwhile; keysHeard is this frame's.
	keysListening, keysHeard bool
	recorderActive           bool // helper hotkey.record is in flight (key_recorder_helper.go)
	// lastLoaded: the last result was read from the history at start.
	lastLoaded bool
	// followUp is the answer card's session a running Rewrite follows up
	// on ("" for none): its answer joins the card's conversation.
	followUp string
}

// answerCard is the answer card's state (overlay_card.go draws it).
type answerCard struct {
	open       bool
	id         string // the session whose result it shows
	conv       string // the first session of its conversation (store.Entry.Conversation)
	kind       store.Kind
	text       string
	highlights []dict.Span
	title      string // Rewrite's answer title ("" for none)
	target     proto.App
	inserting  bool   // Insert pastes: the card is hidden meanwhile (Voxt's)
	msg        string // a line under the text: why Insert, or another language, did not work
	// why says why the result is in the card and not in place (Insert's
	// tip: Voxt says nothing; one line of orange was too loud, 10-15).
	why string
	// problem: the session failed (the card shows why, and Try again);
	// input is what it worked on, for Try again and another language.
	problem *session.Problem
	input   session.TextInput
	lang    string // Translate's target
	// turns are a Rewrite conversation so far, the shown answer last
	// (a follow-up adds to them); conversation: a follow-up began, and
	// the card shows them as one (Voxt's conversation mode), from then on.
	turns        []compose.Turn
	conversation bool
	// attach says which picture went to the language model
	// (Snapshot.ImageHint); images are the input box's pictures.
	attach   string
	images   [][]byte
	previews []answerInputPreview
	// redo is the session translating the text into redoLang, another
	// language chosen in the card, while it runs: the card keeps the text
	// it shows, with a spinner, until the new one replaces it (Voxt's).
	redo, redoLang string
	// llmID is the language model service that answered (its settings'
	// Provider.ID); think the thinking level the answer was asked with,
	// nil the service's; redoThink the redo's.
	llmID            string
	think, redoThink *llm.Thinking
	ui               answerCardUI
	// copied: Copy's mark shows (2 s, Voxt's); copySeq numbers the
	// copies, so that a copy's timer clears only its own mark.
	copied  bool
	copySeq int
}

// Notice actions: where the notice's button goes.
const (
	actionNone        = ""
	actionModels      = "models"
	actionLanguage    = "language" // the Language models page (remote_page.go)
	actionPermissions = "permissions"
	actionHelper      = "helper"
	actionRetry       = "retry"  // the session's entry, in History (history_retry.go)
	actionSignIn      = "signin" // the ChatGPT (Codex) dialog (remote_chatgpt.go)
	actionUsage       = "usage"  // the ChatGPT usage page, in the browser
)

// sessionNotice is a one-line message in the overlay, where
// the pill's waveform goes: an error and a way forward (a click), hidden
// after a few seconds.
type sessionNotice struct {
	text, more string
	action     string
	entry      string // actionRetry's history entry
	warn       bool   // a warning (the session went on), not an error
	ok         bool   // done as asked (a term added, a note saved), not a problem
	seq        int    // which notice: its timer hides only itself
	over       bool   // shows before the card that came back as its session ended (#4013)
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
		Audio:  session.HelperAudio{H: sup},
		Text:   a.textIO(), // family_runtime.go: the family's paste rule
		Select: session.HelperText{H: sup},
		Window: func(ctx context.Context, pid int) ([]byte, error) {
			var r proto.ScreenWindowResult
			raw, err := sup.Call(ctx, "", proto.MethodScreenWindow, proto.ScreenWindowParams{PID: pid}, nil)
			if err == nil {
				err = json.Unmarshal(raw, &r)
			}
			return r.JPEG, err
		},
		ClipboardImage: func(ctx context.Context) ([]byte, error) {
			var r proto.ClipboardImageResult
			raw, err := sup.Call(ctx, "", proto.MethodClipboardImage, proto.ClipboardImageParams{}, nil)
			if err == nil {
				err = json.Unmarshal(raw, &r)
			}
			return r.JPEG, err
		},
		BoxInput:          a.boxShare.load,
		BoxAutoSuppressed: func() (bool, bool) { return a.boxShare.automaticSuppressed() },
		BoxReview:         a.boxShare.waitForReview,
		Resolve:           resolver(lm),
		History:           lib.store.History(),
		Glossary: func(ctx context.Context) ([]store.Term, error) {
			return lib.store.Dictionary().List(ctx, "")
		},
		Notes:    lib.store.Notes(),
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
// the session says what it lacks when it needs it. Without speech the
// speech model is not opened at all: loading it again after it went
// idle cost a selection Translate seconds (#3838).
func resolver(lm local.Services) session.Resolver {
	return func(ctx context.Context, kind session.Kind, s settings.Settings, speech bool) (session.Services, error) {
		f := featureOf(s, string(kind))
		var svc session.Services
		if speech {
			id := strings.TrimSpace(f.ASR)
			skip := false
			if id == "" {
				// Automatic: only a model that can recognise the main
				// language (automaticSpeech). Models installed but none
				// for it is as without one; none installed opens "" and
				// reports the lack, as before.
				all := lm.Installed(local.ASR)
				if m, ok := local.Choose(automaticSpeech(s, all), ""); ok {
					id = m.ID
				} else {
					skip = len(all) > 0
				}
			}
			if !skip {
				p, m, err := lm.ASR(ctx, id)
				switch {
				case err == nil:
					svc.ASR, svc.ASRInfo = p, sessionModel(m)
				case !errors.Is(err, local.ErrNotInstalled):
					return svc, err
				}
			}
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
		ready := local.ForFeature(lm.Installed(local.LLM), string(kind))
		want := strings.TrimSpace(f.LLM)
		m, ok := local.Choose(ready, want)
		if !ok {
			return svc, nil
		}
		// An unavailable explicit choice stays unavailable; only Translate's
		// own fallback may stand in.
		if want != "" && m.ID != want {
			m = local.Model{}
		}
		// When it does not open, there is no language model: the text goes
		// on as recognised.
		if m.ID != "" {
			if c, ml, err := lm.LLM(ctx, m.ID); err == nil && ml.ID == m.ID {
				svc.LLM, svc.LLMInfo = c, sessionModel(ml)
			}
		}
		// The fallback model, or the model when that one did not open: the
		// first candidate that opens.
		for _, id := range fallbackCandidates(kind, s, ready, m.ID) {
			c, ml, err := lm.LLM(ctx, id)
			if err != nil || ml.ID != id {
				continue
			}
			if svc.LLM == nil {
				svc.LLM, svc.LLMInfo = c, sessionModel(ml)
			} else {
				svc.Fallback, svc.FallbackInfo = c, sessionModel(ml)
			}
			break
		}
		return svc, nil
	}
}

// fallbackCandidates are the models a session may fall back to, in order:
// Translate's fallback model, then, with Models.LLMSwitch, every other
// ready model on this Mac in the Language models page's order. The switch
// never picks an external API: the text would go to a provider the user
// did not choose for the feature.
func fallbackCandidates(kind session.Kind, s settings.Settings, ready []local.Model, chosen string) []string {
	var ids []string
	if fb := s.Features.Translate.Fallback; kind == store.KindTranslate && fb != "" && fb != chosen && slices.ContainsFunc(ready, func(x local.Model) bool { return x.ID == fb }) {
		ids = append(ids, fb)
	}
	if !s.Models.LLMSwitch {
		return ids
	}
	for _, x := range ready {
		setup, _ := providers.Setup(s, x.ID)
		preset, err := providers.Config(setup).Resolve()
		if x.ID != chosen && !slices.Contains(ids, x.ID) && err == nil && onThisMac(preset.BaseURL) {
			ids = append(ids, x.ID)
		}
	}
	return ids
}

// sessionModel is how a session (and its history entry) names a model.
func sessionModel(m local.Model) session.Model {
	return session.Model{ID: m.ID, Name: m.Name, Model: m.ID}
}

// sessionChanged takes a session's snapshot (the newest wins) and
// updates the overlay, the shortcuts and the menu bar. The backgrounded
// answer's snapshots go to its card (backgroundChanged).
func (a *app) sessionChanged(s session.Snapshot) {
	if a.bgID != "" && s.ID == a.bgID {
		a.backgroundChanged(s)
		return
	}
	if s.Rev <= a.sess.Rev {
		return
	}
	was := a.sess
	a.sess = s
	if a.box.open {
		if s.ID == a.box.session {
			if s.Selection != "" {
				a.box.selection = s.Selection
			}
			if s.Spoken != "" {
				a.box.instruction = s.Spoken
			}
		}
		if s.ID == a.box.sending && s.ImagePreviewKind == "window" && len(s.ImagePreview) > 0 &&
			(was.ID != s.ID || was.ImagePreviewKind != "window" || len(was.ImagePreview) == 0) {
			a.box.shown = true
		}
		a.boxSync() // the box's lines follow its session
	}
	for _, fn := range a.onSession { // events.go: the setup guide, the sounds
		fn(was, s)
	}
	if s.ID != was.ID {
		a.notice = nil
	}
	if s.State.Ended() && !was.State.Ended() {
		a.sessionEnded(s)
	}
	if s.State.Live() != was.State.Live() || (s.State == session.Recording) != (was.State == session.Recording) {
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
	if a.prefs.General.CopyResult && s.Card && s.Result != "" && s.State != session.Failed {
		a.copyRecent(s.Result) // tray.go: the clipboard keeps each result
	}
	switch {
	case a.answer.open && a.answer.redo != "" && a.answer.redo == s.ID:
		a.endRedo(s)
	case s.Card && (s.Result != "" || s.State == session.Failed):
		// The box's send failed: the box shows the error and keeps its
		// text and pictures, no failed card.
		if s.State != session.Failed || s.ID != a.box.sending && s.ID != a.box.session {
			switch {
			case s.ID == a.bgID:
				// The backgrounded answer's own end: its card opens (and its
				// conversation merges) as the foreground path's would. A card
				// that is not its own waits under it: setAside holds only
				// Rewrite and Translate cards, so a Transcribe card is held
				// directly.
				if a.answer.open && a.followUp == "" {
					if c := a.answer; c.kind == store.KindTranscribe && c.problem == nil && !c.inserting {
						a.under, a.answer = c, answerCard{}
					} else {
						a.setAside(false)
					}
				}
				a.openAnswer(s)
			case a.answer.open && a.answer.id != s.ID && a.bgID != "" &&
				(a.followUp != "" && a.followUp == a.answer.id || a.answer.redo == a.bgID):
				// The open card's own answer still streams in the background:
				// this card waits under it, conversation and redo untouched.
				saved, followUp := a.answer, a.followUp
				a.followUp = ""
				a.openAnswer(s)
				a.under, a.answer = a.answer, saved
				a.followUp = followUp
			case a.answer.open && a.answer.id != s.ID && !(a.followUp != "" && a.followUp == a.answer.id):
				// A card that is not this session's own waits under the new
				// one and comes back when it closes, as setAside would have
				// held it. A follow-up is the card's own: it merges instead.
				a.setAside(false)
				a.openAnswer(s)
			default:
				a.openAnswer(s)
			}
			a.syncHotkeys() // the cancel shortcut closes the card
		}
	}
	a.boxEnded(s)
	back := a.resurface()
	var copied *sessionNotice
	if !s.Card && !s.Inserted && s.Delivery != "" && s.Result != "" && s.State == session.Done {
		// Not inserted and the card is off: the clipboard has it (Voxt 1.x).
		a.copyRecent(s.Result)
		// One short line: a notice is one line, cut past NoticeW2.
		copied = &sessionNotice{text: i18n.T("Not inserted. Copied: press ⌘V to paste."), warn: true, over: back}
	}
	if s.ID != "" && a.followUp != "" && !s.Card {
		a.followUp = "" // a follow-up that ended without an answer: the card keeps the last one
	}
	if s.Note && s.State == session.Done && s.Result != "" {
		a.loadNotes() // notes_page.go: the page shows it
	}
	if p := s.Problem; p != nil && s.State != session.Cancelled && !(s.Card && s.State == session.Failed) {
		text, more, action := sessionProblemText(p)
		switch {
		case s.Note && s.Card:
			text = i18n.T("Could not save the note, so the card shows it.")
		case s.Note && p.Code == session.ProblemLLM && p.Degraded:
			text = degradedLLMText(p, true) // the recognised text went to Notes
		}
		if text != "" {
			// Nothing heard is not a failure: a warning.
			n := sessionNotice{text: text, more: more, action: action, warn: p.Degraded || p.Code == session.ProblemNoSpeech, over: back}
			if p.Code == session.ProblemASR && s.Recording {
				n.action, n.entry = actionRetry, s.ID // the entry kept the recording: Retry
			}
			if copied != nil {
				n.then = func() { a.showNotice(*copied) }
			}
			a.showNotice(n)
		} else if copied != nil {
			a.showNotice(*copied)
		}
	} else if s.Note && s.State == session.Done && s.Result != "" {
		a.showNotice(sessionNotice{text: i18n.T("Saved to Notes"), ok: true})
	} else if copied != nil {
		a.showNotice(*copied)
	}
	a.historyChanged()
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
	case actionLanguage:
		a.showMainWindow()
		a.navigate(remotePath)
	case actionPermissions:
		a.showMainWindow()
		a.navigate("/settings/permissions")
	case actionHelper:
		a.restartHelper()
	case actionSignIn:
		a.showMainWindow()
		a.completeSetup(local.LLM, providers.ChatGPT)
	case actionUsage:
		go openBrowser(chatgptUsageURL)
	}
}

// closeSessions cancels the running sessions and removes the Carbon
// shortcuts; it runs on quit.
func (a *app) closeSessions() {
	a.cancelSession()
	if a.mgr != nil && a.bgID != "" {
		a.mgr.Cancel(a.bgID)
	}
	if a.mgr != nil {
		a.mgr.Close()
	}
	if a.hasMenu {
		for _, acc := range a.carbon {
			mygo.GlobalShortcut.Unregister(acc)
		}
	}
	a.carbon = nil
}
