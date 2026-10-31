package main

// What the answer card's buttons do (session_link.go; the card is
// overlay_card.go): opening it on a session's result, trying again (in
// another language), a Rewrite follow-up, copying, inserting, closing.

import (
	"context"
	"os"
	"time"

	"github.com/sgl-project/sglang-omni/VoxtNew/internal/compose"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/core"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/deliver"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/helper/proto"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/i18n"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/session"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/store"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/uitext"
)

// openAnswer opens the answer card on an ended session: its result (or
// why it failed), what it worked on, a Rewrite conversation's turns.
func (a *app) openAnswer(s session.Snapshot) {
	card := answerCard{
		open: true, id: s.ID, kind: s.Kind, text: s.Result, title: s.Title, target: s.Target,
		input: session.TextInput{Raw: s.Raw, Selection: s.Selection}, lang: s.TargetLanguage,
		why: uitext.WhyInCard(s.Delivery, s.Selection != "", s.Target.Name),
	}
	if s.State == session.Failed {
		card.problem = s.Problem
	}
	switch s.Delivery {
	case deliver.ReasonNoAccess:
		card.msg = i18n.T("Allow Accessibility to insert results where you type.")
	case deliver.ReasonUnconfirmed, deliver.ReasonSecureInput:
		// Pasted, but it did not show: say so where the user looks.
		card.msg = uitext.DeliveryText(deliver.Result{Reason: s.Delivery}, s.Target)
	}
	if s.Kind == store.KindRewrite {
		// A follow-up continues the card's conversation.
		if a.answer.kind == store.KindRewrite && a.answer.id == a.followUp {
			card.turns = append(card.turns, a.answer.turns...)
			card.conversation = a.answer.conversation
			if card.why == "" {
				card.why = a.answer.why
			}
		}
		if card.problem == nil {
			// Any answer takes follow-ups, a rewritten selection's too
			// (Voxt's): the selection goes with its turn.
			card.turns = append(card.turns, compose.Turn{Instruction: s.Raw, Selection: s.Selection, Answer: compose.Answer{Title: s.Title, Content: s.Result}})
		} else if card.problem != nil && len(card.turns) > 0 {
			card.input.Raw = s.Raw // Try again asks the same follow-up
		}
	}
	a.followUp = ""
	if card.target.PID == 0 && a.answer.open {
		card.target = a.answer.target
	}
	a.answer = card
}

// endRedo ends the card's translation into another language: its text
// replaces the card's; a failure (or nothing) leaves the card as it was,
// in the language it was, and says so under it (Voxt goes back to the
// language without a word); a cancel just goes back.
func (a *app) endRedo(s session.Snapshot) {
	lang := a.answer.redoLang
	a.answer.redo, a.answer.redoLang = "", ""
	switch {
	case s.State == session.Done && s.Result != "":
		a.answer.id, a.answer.text, a.answer.lang = s.ID, s.Result, s.TargetLanguage
		a.answer.msg, a.answer.copied = "", false
	case s.State == session.Cancelled:
	default:
		why := i18n.T("the language model answered nothing")
		if s.Problem != nil && s.Problem.Code != session.ProblemEmpty && s.Problem.Code != session.ProblemNoSpeech {
			why, _, _ = sessionProblemText(s.Problem)
		}
		a.answer.msg = i18n.T("Not translated into %s: %s", uitext.TargetLanguageLabel(lang), why)
	}
}

// retryAnswer starts the card's session again with the text it worked on
// (Try again after a failure), or into the language lang (Translate's
// target select). The card stays until the new session's own result.
func (a *app) retryAnswer(lang string) {
	card := a.answer
	if !card.open || a.sess.State.Live() {
		return
	}
	in := card.input
	req := session.StartRequest{Kind: card.kind, Target: card.target, Text: &in, TargetLanguage: lang}
	if req.TargetLanguage == "" {
		req.TargetLanguage = card.lang
	}
	if lang != "" && card.problem == nil {
		// Another language for the card's text: it stays in the card,
		// which shows its text until the new one comes (Voxt's).
		req.Card = true
		a.answer.msg = ""
		if n := a.sessionBlocker(card.kind); n != nil {
			a.answer.msg = n.text
			return
		}
		id, err := a.mgr.Start(req)
		if err == nil {
			a.answer.redo, a.answer.redoLang = id, lang
		}
		a.syncHotkeys()
		a.overlaySync()
		return
	}
	if card.kind == store.KindRewrite && len(card.turns) > 0 && in.Selection == "" {
		req.Previous = card.turns
		if card.problem == nil {
			return // nothing to ask again
		}
		a.followUp = card.id
	}
	a.notice = nil
	if n := a.sessionBlocker(card.kind); n != nil {
		a.answer.msg = n.text
		return
	}
	if _, err := a.mgr.Start(req); err == nil && len(req.Previous) == 0 {
		a.answer = answerCard{}
	}
	a.syncHotkeys()
	a.overlaySync()
}

// followUpAnswer records a follow-up to the card's Rewrite answer (the
// Rewrite shortcut while the card shows one): the next instruction
// rewrites that answer, in the card.
func (a *app) followUpAnswer(hold string) bool {
	card := a.answer
	if !a.canFollowUp() {
		return false
	}
	a.followUp = card.id
	if n := a.startSession(session.StartRequest{Kind: store.KindRewrite, Target: card.target, HoldBinding: hold, Previous: card.turns}); n != nil {
		a.followUp = ""
	} else {
		// The card turns into a conversation, where the follow-up records
		// (Voxt's).
		a.answer.conversation, a.answer.msg = true, ""
		a.overlaySync()
	}
	return true
}

// canFollowUp says whether the card takes a follow-up now: a Rewrite
// answer written from an instruction, nothing running.
func (a *app) canFollowUp() bool {
	card := a.answer
	return card.open && card.kind == store.KindRewrite && card.problem == nil && len(card.turns) > 0 && !a.sess.State.Live()
}

// closeAnswer closes the answer card (and cancels a follow-up, or another
// language, on its way).
func (a *app) closeAnswer() {
	if a.cardSessionLive() {
		a.cancelSession()
	}
	a.answer, a.followUp = answerCard{}, ""
	a.syncHotkeys()
	a.overlaySync()
}

// cardSessionLive says whether the session running is the card's own: a
// follow-up, or the text into another language.
func (a *app) cardSessionLive() bool {
	s := a.sess
	return a.answer.open && s.State.Live() && !s.Try && (a.followUp != "" && a.followUp == a.answer.id || s.ID == a.answer.redo)
}

// copyMark is how long Copy shows its mark (Voxt's).
const copyMark = 2 * time.Second

// copyAnswer copies the card's text (the button's handler gives the
// window's clipboard); the card stays, Copy showing a mark for a moment
// (Voxt's).
func (a *app) copyAnswer(copyText func(string)) {
	if !a.answer.open || a.answer.text == "" {
		return
	}
	copyText(a.answer.text)
	a.setDelivery(a.answer.id, "copy", proto.App{})
	a.answer.copySeq++
	seq, id := a.answer.copySeq, a.answer.id
	a.answer.copied = true
	a.later(copyMark, func() {
		if a.answer.open && a.answer.copySeq == seq && a.answer.id == id {
			a.answer.copied = false
			if a.ov.win != nil {
				a.ov.win.Invalidate()
			}
		}
	})
}

// insertAnswer inserts the card's text where the user is typing: the
// app the session started in (the user chose to, so any focused element
// there will do; when VoxtNew is in front, that app comes back first), or
// a text field in the app in front now. The card hides while it pastes
// and comes back if that did not work (Voxt's).
func (a *app) insertAnswer() {
	card := a.answer
	if !card.open || card.inserting || card.text == "" || a.cardSessionLive() {
		return
	}
	a.answer.inserting, a.answer.msg = true, ""
	a.overlaySync()
	text := a.textIO() // family_runtime.go: the family's paste rule
	a.async(func() {
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		res := deliver.Insert(ctx, text, deliver.Request{Text: card.text, Target: card.target, AnyEditable: true, Self: os.Getpid(),
			Activate: func(pid int) bool { ok, _ := activateTarget(pid); return ok }, VerifyMS: deliver.DefaultVerifyMS})
		a.post(func() {
			a.logs.Append(core.LogEntry{Time: time.Now(), Level: core.SeverityInfo, Source: sessionSource, Message: "card insert: " + res.Diag()})
			if !a.answer.open || a.answer.id != card.id {
				return
			}
			a.answer.inserting = false
			if res.Inserted {
				var into proto.App
				if res.Moved(card.target) {
					into = res.To // the user clicked into another app
				}
				a.setDelivery(card.id, "insert", into)
				a.closeAnswer()
				return
			}
			a.answer.msg = uitext.DeliveryText(res, card.target)
			a.overlaySync()
		})
	})
}
