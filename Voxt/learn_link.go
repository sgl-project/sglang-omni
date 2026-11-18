package main

// Learning from the user's corrections: after a Transcribe result went
// into another app's field, internal/learn watches that field and proposes
// what the user corrects (the Dictionary page's suggestions). A new
// session ends the watch; turning learning off drops it.

import (
	"context"
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/core"
	"github.com/sgl-project/sglang-omni/Voxt/internal/helper/proto"
	"github.com/sgl-project/sglang-omni/Voxt/internal/learn"
	"github.com/sgl-project/sglang-omni/Voxt/internal/session"
	"github.com/sgl-project/sglang-omni/Voxt/internal/store"
)

// learnSource is the logs' source of learning lines.
const learnSource = "learn"

// learnState is the watch for corrections, one Learner over the app's life.
type learnState struct {
	learner      *learn.Learner
	learnCtx     context.Context
	stopLearning context.CancelFunc
}

// openLearner makes the Learner; connectHelper calls it, once the
// supervisor exists.
func (a *app) openLearner() {
	logs := a.logs
	a.learner = &learn.Learner{Read: session.HelperText{H: a.host.sup}, Book: a.book, Log: func(level, msg string) {
		logs.Append(core.LogEntry{Time: time.Now(), Level: core.Severity(level), Source: learnSource, Message: msg})
	}}
	a.learnCtx, a.stopLearning = context.WithCancel(context.Background())
}

// learnFromSession starts the watch after an insert. A real goroutine,
// not a.async: the watch lasts up to 30 s, and a.async runs inline in
// tests.
func (a *app) learnFromSession(s session.Snapshot) {
	d := a.prefs.Dictionary
	if !d.Enabled || !d.Learn || s.Kind != store.KindTranscribe || s.Try || s.Landing == nil || !a.hstatus.Has(proto.MethodTextRead) {
		return
	}
	ins := learn.Insertion{Text: s.Result, Landing: *s.Landing}
	learner, ctx := a.learner, a.learnCtx
	go func() {
		if learner.Learn(ctx, ins) > 0 {
			a.post(a.loadSuggestions)
		}
	}()
}
