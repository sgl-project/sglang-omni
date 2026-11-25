package session

import (
	"context"
	"fmt"
	"slices"
	"strings"
	"sync"
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/compose"
	"github.com/sgl-project/sglang-omni/Voxt/internal/llm"
)

// partsTogether is how many of a long dictation's parts a remote service
// is asked at once. One that limits the rate answers some with 429, which
// the client waits out; it is asked one by one afterwards (resting).
const partsTogether = 4

// cleanPartsTogether is cleanParts with the requests side by side: the
// draft shows the answers in order, up to the first part not answered yet.
func (r *run) cleanPartsTogether(text string, parts []compose.Part, answered []string, think *llm.Thinking, ctx context.Context) (string, string, error) {
	answers := make([]string, len(parts))
	copy(answers, answered)
	fails := make([]error, len(parts))
	done := make([]bool, len(parts))
	for i := range answered {
		done[i] = true
	}
	drafts := make([]strings.Builder, len(parts))
	var mu sync.Mutex
	var published time.Time
	first := func() int { // the first part not answered, len(parts) for none
		i := slices.Index(done, false)
		if i < 0 {
			return len(parts)
		}
		return i
	}
	publish := func() {
		n := first()
		so := answers[:n:n]
		if n < len(parts) {
			so = append(so, compose.PartialOutput(drafts[n].String(), parts[n].Text))
		}
		draft := r.draftSnippets(compose.JoinParts(parts[:min(n+1, len(parts))], so), parts[:min(n+1, len(parts))])
		r.m.update(r, func(s *Snapshot) {
			if ctx.Err() == nil && r.ctx.Err() == nil && !r.skipped.Load() && r.cleanupDecisionOpenLocked() {
				s.Draft = draft
			}
		})
	}
	slots := make(chan struct{}, partsTogether)
	var wg sync.WaitGroup
	for i := len(answered); i < len(parts); i++ {
		wg.Go(func() {
			slots <- struct{}{}
			defer func() { <-slots }()
			if r.ctx.Err() != nil || !r.cleanupMayContinue() {
				return
			}
			p := parts[i]
			resp, err := r.askPart(ctx, r.partRequest(text, p, think, func(delta string) {
				mu.Lock()
				defer mu.Unlock()
				drafts[i].WriteString(delta)
				if now := time.Now(); i == first() && now.Sub(published) >= draftEvery {
					published = now
					publish()
				}
			}), p.Text)
			if ctx.Err() != nil || r.ctx.Err() != nil || r.skipped.Load() || r.rethink.Load() {
				return
			}
			mu.Lock()
			defer mu.Unlock()
			if err != nil {
				fails[i] = fmt.Errorf("piece %d of %d: %w", i+1, len(parts), err)
				r.logf("warning", "%v: kept as recognised", fails[i])
				answers[i] = p.Text
			} else {
				answers[i] = compose.CleanOutput(resp.Text, p.Text)
			}
			done[i] = true
			publish()
		})
	}
	wg.Wait()
	if r.ctx.Err() != nil || !r.cleanupMayContinue() {
		return text, parts[0].Profile(), context.DeadlineExceeded
	}
	var failed error
	for _, err := range fails {
		if err != nil {
			failed = err
			break
		}
	}
	return compose.JoinParts(parts, answers), parts[0].Profile(), failed
}
