package main

// The overlay's notice that the Mac is short of memory (docs/behavior.md
// 5.32): when a language model's chat in a session was slow, or a deadline
// stopped it, with the swap files over 90% full (llmrt.ShortOfMemory), the
// session's end shows it, at most once a run: after the result went in
// and after the session's own notice, as a warning. It is no setting; the
// log says when it showed, and why.
//
// VOXTNEW_DEV_SHORT_OF_MEMORY=1 makes every chat count as slow with the
// swap 95% full: the next session that used a language model shows it
// (docs/manual-test.md).

import (
	"fmt"
	"os"
	"time"

	"github.com/sgl-project/sglang-omni/VoxtNew/internal/core"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/i18n"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/llmrt"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/macx"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/session"
)

// devShortOfMemoryEnv fakes a slow chat and a full swap, to see the notice.
const devShortOfMemoryEnv = "VOXTNEW_DEV_SHORT_OF_MEMORY"

// shortMemoryState is the notice's state, main-thread only.
type shortMemoryState struct {
	shortMemory struct {
		// session is the live session whose chat called for the notice:
		// its end shows it; reason says why, for the log.
		session, reason string
		shown           bool // it showed in this run
	}
}

// onPace is llmrt.Config.OnPace: how a chat went, to the main thread.
func (a *app) onPace() func(llmrt.ChatPace, macx.SystemMemory) {
	fake := os.Getenv(devShortOfMemoryEnv) != ""
	return func(pace llmrt.ChatPace, memory macx.SystemMemory) {
		if fake {
			pace.Model += " (" + devShortOfMemoryEnv + ")"
			pace.FirstToken = max(pace.FirstToken, 4*time.Second)
			memory.SwapUsedBytes, memory.SwapTotalBytes = 19<<30, 20<<30
		}
		a.post(func() { a.chatPaced(pace, memory) })
	}
}

// chatPaced marks the live session (not a try) for the notice when its
// chat calls for it. The chat's pace comes before the session's end: its
// snapshot is posted after the chat returned.
func (a *app) chatPaced(pace llmrt.ChatPace, memory macx.SystemMemory) {
	s, m := a.sess, &a.shortMemory
	if !s.State.Live() || s.Try || m.session == s.ID || !llmrt.ShortOfMemory(pace, memory, m.shown) {
		return
	}
	how := fmt.Sprintf("first token %.1f s, %.1f tokens/s", pace.FirstToken.Seconds(), pace.TokensPerSecond)
	if pace.Stopped {
		how = fmt.Sprintf("stopped by a deadline, %d pieces streamed", pace.Tokens)
	}
	m.session, m.reason = s.ID, fmt.Sprintf("%s %s %s; %s", pace.Model, pace.Skill, how, memory)
}

// memoryNoticeAfter shows the notice at the end of the session that
// called for it, after the session's own notice when it has one. Not
// after a cancel, nor under the answer card: a later session may show it.
func (a *app) memoryNoticeAfter(s session.Snapshot) {
	m := &a.shortMemory
	if m.session == "" || m.session != s.ID {
		return
	}
	reason := m.reason
	m.session, m.reason = "", ""
	switch {
	case s.State == session.Cancelled || a.answer.open:
	case a.notice != nil:
		a.notice.then = func() { a.showMemoryNotice(reason) }
	default:
		a.showMemoryNotice(reason)
	}
}

// showMemoryNotice shows the notice, and logs why.
func (a *app) showMemoryNotice(reason string) {
	a.shortMemory.shown = true
	a.logs.Append(core.LogEntry{Time: time.Now(), Level: core.SeverityWarning, Source: sessionSource, Message: "the overlay said the Mac is short of memory: " + reason})
	a.showNotice(memoryNotice())
}

// memoryNotice is the notice: a warning, one line that says what helps.
func memoryNotice() sessionNotice {
	return sessionNotice{text: i18n.T("Low memory slows the model: quit unused apps."), warn: true}
}
