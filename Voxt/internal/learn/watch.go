package learn

import (
	"context"
	"fmt"
	"strings"
	"sync"
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/deliver"
	"github.com/sgl-project/sglang-omni/Voxt/internal/helper/proto"
)

// WatchSeconds is how long a watch lasts at most.
const WatchSeconds = 30

// Voxt 1.x's watch: the first read after watchPause, up to three tries
// watchRetry apart, then a read every watchPoll for watchWindow. Its pad
// is 2.x's: only the text around the insertion.
const (
	watchPause     = 900 * time.Millisecond
	watchRetry     = 500 * time.Millisecond
	watchTries     = 3
	watchPoll      = time.Second
	watchWindow    = WatchSeconds * time.Second
	watchIdle      = 4 * time.Second // reads failing after an edit end the watch only once it has rested this long
	watchMissLimit = 3
	watchPad       = 300 // UTF-16 units each side of the insertion
)

// Insertion is a Transcribe result pasted into another app's field.
type Insertion struct {
	Text string // the result as pasted
	deliver.Landing
}

// Reader reads a field (the helper's textio.read; session.HelperText).
type Reader interface {
	Read(ctx context.Context, p proto.ReadParams) (proto.ReadResult, error)
}

// Learner watches one insertion at a time, with Voxt 1.x's rules. Learn
// blocks for up to 30 s; a later Learn or Finish ends the watch before and
// the edit seen so far is judged; Stop ends it and judges nothing.
type Learner struct {
	Read Reader
	Book *Book
	Log  func(level, msg string) // lengths and outcomes only, never text
	// Sleep waits between reads (nil: the clock's; tests set it).
	Sleep func(ctx context.Context, d time.Duration) error

	mu      sync.Mutex
	current *watchHandle
}

// watchHandle ends a running watch: stop drops it, finish judges it.
type watchHandle struct {
	stop   context.CancelFunc
	finish context.CancelFunc
}

// Finish ends the current watch now and judges it (a new session started).
func (l *Learner) Finish() {
	l.mu.Lock()
	defer l.mu.Unlock()
	if l.current != nil {
		l.current.finish()
	}
}

// Stop ends the current watch now and drops it (the user turned learning off).
func (l *Learner) Stop() {
	l.mu.Lock()
	defer l.mu.Unlock()
	if l.current != nil {
		l.current.stop()
	}
}

func (l *Learner) pause(ctx context.Context, d time.Duration) error {
	if l.Sleep != nil {
		return l.Sleep(ctx, d)
	}
	timer := time.NewTimer(d)
	defer timer.Stop()
	select {
	case <-timer.C:
		return nil
	case <-ctx.Done():
		return ctx.Err()
	}
}

// Learn watches ins's field, judges the edit and puts what it teaches in
// the Book; it reports how many suggestions are new.
func (l *Learner) Learn(ctx context.Context, ins Insertion) int {
	if ins.At < 0 || ins.Pressed || strings.TrimSpace(ins.Text) == "" {
		l.Log("info", "watch skipped: "+skipWatch(ins))
		return 0
	}
	watchCtx, stop := context.WithCancel(ctx)
	defer stop()
	readCtx, finish := context.WithCancel(watchCtx)
	defer finish()
	handle := &watchHandle{stop: stop, finish: finish}
	l.mu.Lock()
	if l.current != nil {
		l.current.finish()
	}
	l.current = handle
	l.mu.Unlock()
	defer func() {
		l.mu.Lock()
		defer l.mu.Unlock()
		if l.current == handle {
			l.current = nil
		}
	}()

	l.Log("info", fmt.Sprintf("watch started: %d units inserted in %s", utf16Len(ins.Text), ins.Role))
	state, end, lengths := l.follow(readCtx, ins)
	if watchCtx.Err() != nil {
		end = "dropped"
	}
	l.Log("info", fmt.Sprintf("watch ended (%s): %d reads, lengths %v", end, len(lengths), lengths))
	if watchCtx.Err() != nil || state == nil {
		return 0
	}
	final, ok := state.conclude()
	if !ok {
		l.Log("info", "watch: no finished edit")
		return 0
	}
	s, why := suggestionFromEdit(ins.Text, state.baseline, final)
	if why != "" {
		l.Log("info", "watch: "+why)
		return 0
	}
	terms, err := l.Book.terms.List(watchCtx, "")
	if err != nil {
		l.Log("warning", "watch: reading the dictionary failed")
		return 0
	}
	for _, t := range terms {
		if fold(t.Term) == fold(s.Heard[0]) {
			l.Log("info", "watch: the user changed a dictionary term")
			return 0
		}
	}
	n, err := l.Book.Propose(watchCtx, []Suggestion{s})
	if err != nil {
		l.Log("warning", "watch: saving the suggestion failed")
		return 0
	}
	l.Log("info", fmt.Sprintf("watch: %d new suggestion", n))
	return n
}

// skipWatch says why an insertion is not watched, for the log.
func skipWatch(ins Insertion) string {
	switch {
	case ins.At < 0:
		return "position unknown"
	case ins.Pressed:
		return "keys were pressed after it"
	default:
		return "empty text"
	}
}

// follow runs the watch: the baseline read, then the polls until an end.
// state is nil when the watch cannot be judged: nothing read that shows
// the edit, or a password field.
func (l *Learner) follow(ctx context.Context, ins Insertion) (state *watchState, end string, lengths []int) {
	if l.pause(ctx, watchPause) != nil {
		return nil, "cancelled before the first read", nil
	}
	want := proto.TextRange{Location: max(0, ins.At-watchPad)}
	want.Length = ins.At - want.Location + utf16Len(ins.Text) + watchPad
	var base proto.ReadResult
	got := false
	for try := range watchTries {
		if try > 0 && l.pause(ctx, watchRetry) != nil {
			return nil, "cancelled before the first read", lengths
		}
		res, err := l.Read.Read(ctx, proto.ReadParams{PID: ins.App.PID, Range: want})
		if ctx.Err() != nil {
			return nil, "cancelled before the first read", lengths
		}
		lengths = append(lengths, utf16Len(res.Text))
		if res.Error == proto.ReadSecure {
			return nil, "secure", lengths
		}
		if err == nil && res.Error == "" {
			base, got = res, true
			break
		}
	}
	if !got {
		return nil, "no baseline", lengths
	}
	at := runeAtUTF16(base.Text, ins.At-base.Range.Location)
	scoped, found := scopedBaseline(ins.Text, base.Text, at)
	if !found {
		return nil, "the inserted text is not in the field", lengths
	}

	state = &watchState{baseline: scoped, latest: scoped}
	fieldLength, changedAt := base.Length, time.Duration(0)
	for elapsed := watchPoll; elapsed <= watchWindow; elapsed += watchPoll {
		if l.pause(ctx, watchPoll) != nil {
			return state, "ended early", lengths
		}
		res, err := l.Read.Read(ctx, proto.ReadParams{PID: ins.App.PID, Range: proto.TextRange{
			Location: base.Range.Location, Length: max(0, base.Range.Length+fieldLength-base.Length),
		}})
		if ctx.Err() != nil {
			return state, "ended early", lengths
		}
		lengths = append(lengths, utf16Len(res.Text))
		switch {
		case err == nil && res.Error == proto.ReadSecure:
			return nil, "secure", lengths
		case err != nil || res.Error != "" || res.Element != base.Element:
			switch state.missed(elapsed - changedAt) {
			case watchGiveUp:
				return state, "unreadable", lengths
			case watchSettle:
				return state, "left the field", lengths
			}
		default:
			fieldLength = res.Length
			if state.saw(scopedCurrent(ins.Text, scoped, res.Text)) {
				changedAt = elapsed
			}
		}
	}
	return state, "window over", lengths
}

// watchDecision is what a read that failed decides.
type watchDecision int

const (
	watchGoOn watchDecision = iota
	watchGiveUp
	watchSettle
)

// watchState is what a watch has seen so far (1.x's observation state).
type watchState struct {
	baseline, latest string
	changed          bool
	missing          int // reads that failed in a row
}

// saw takes a read of the field and reports whether it changed the text.
func (w *watchState) saw(text string) bool {
	w.missing = 0
	if text == w.latest {
		return false
	}
	w.latest, w.changed = text, true
	return true
}

// missed takes a failed read; idle is how long the last edit has rested.
func (w *watchState) missed(idle time.Duration) watchDecision {
	w.missing++
	if w.missing < watchMissLimit {
		return watchGoOn
	}
	if !w.changed {
		return watchGiveUp
	}
	if idle >= watchIdle && !incompleteEdit(w.baseline, w.latest) {
		return watchSettle
	}
	return watchGoOn
}

// conclude is the edit to judge at the end of the watch, if a finished one
// was seen: only a deletion is not one (the user may be still replacing).
func (w *watchState) conclude() (string, bool) {
	if !w.changed || incompleteEdit(w.baseline, w.latest) {
		return "", false
	}
	return w.latest, true
}
