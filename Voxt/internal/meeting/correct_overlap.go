package meeting

// The mic track's correction hides under the system track's final pass
// (behavior.md: the transcript's correction). Once the first track's
// final segments exist, their correction batches start while the next
// track still transcribes; one request is in flight at a time, and the
// correction that follows reuses the results by line text. Echo dropping
// only removes whole mic lines, never rewrites them, so a surviving
// line's speculative correction stays valid and a dropped line's one is
// discarded with it.

import (
	"context"
	"sync"

	"github.com/sgl-project/sglang-omni/Voxt/internal/store"
)

// A correctionOverlap is the first track's speculative correction, armed
// by runFinalize for the meeting whose final pass it overlaps.
type correctionOverlap struct {
	meeting string
	cancel  context.CancelFunc
	reqMu   sync.Mutex // one correction request in flight at a time
	wg      sync.WaitGroup
	mu      sync.Mutex
	fixed   map[string]string // recognised line text -> corrected text
}

// startCorrectOverlap arms the overlap for a meeting's final pass, or
// returns nil when the correction cannot overlap: no language model, or
// one served on this Mac (it would share the GPU and memory with
// recognition, as live polish already avoids, behavior.md 5.31b).
func (m *Manager) startCorrectOverlap(id string) *correctionOverlap {
	if m.cfg.LLM == nil {
		return nil
	}
	if m.cfg.LLMIsLocal != nil && m.cfg.LLMIsLocal(m.cfg.Settings().LLM) {
		return nil
	}
	return &correctionOverlap{meeting: id, fixed: map[string]string{}}
}

// speculate corrects the first track's final segments in the background;
// a failed batch is left out and asked again by the correction after the
// final pass.
func (o *correctionOverlap) speculate(ctx context.Context, cfg Config, segs []Segment) {
	chat, err := cfg.LLM(ctx, cfg.Settings().LLM)
	if err != nil || chat == nil {
		return
	}
	var tc TranscriptContext
	if cfg.Context != nil {
		tc = cfg.Context(ctx)
	}
	rows := make([]store.MeetingSegment, 0, len(segs))
	for i, s := range segs {
		rows = append(rows, store.MeetingSegment{Idx: i, Text: s.Text})
	}
	style := cfg.Settings().CorrectionStyle
	o.wg.Add(1)
	go func() {
		defer o.wg.Done()
		for _, batch := range correctionBatches(rows) {
			got, err := o.correctBatch(ctx, chat, tc, style, batch)
			if err != nil {
				continue
			}
			o.mu.Lock()
			for _, s := range batch {
				if t, ok := got[s.Idx]; ok {
					o.fixed[s.Text] = t
				}
			}
			o.mu.Unlock()
		}
	}()
}

// take waits for the speculative correction and returns its results by
// recognised line text.
func (o *correctionOverlap) take() map[string]string {
	o.wg.Wait()
	o.mu.Lock()
	defer o.mu.Unlock()
	return o.fixed
}

// correctBatch runs one batch under the single-flight lock; a nil
// overlap means no other correction can be running.
func (o *correctionOverlap) correctBatch(ctx context.Context, chat SummaryChat, tc TranscriptContext, style string, batch []store.MeetingSegment) (map[int]string, error) {
	if o == nil {
		return correctBatch(ctx, chat, tc, style, batch)
	}
	o.reqMu.Lock()
	defer o.reqMu.Unlock()
	return correctBatch(ctx, chat, tc, style, batch)
}
