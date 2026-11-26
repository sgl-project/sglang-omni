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
	"reflect"
	"sync"

	"github.com/sgl-project/sglang-omni/Voxt/internal/store"
)

type correctionWork struct {
	track Source
	batch []store.MeetingSegment
}

// A correctionOverlap is one finalization run's speculative correction,
// armed by runFinalize for the meeting whose final pass it overlaps.
type correctionOverlap struct {
	meeting string
	cancel  context.CancelFunc
	reqMu   sync.Mutex // one correction request in flight at a time
	wg      sync.WaitGroup
	mu      sync.Mutex

	fixed     map[string]map[int]string
	completed map[string]bool
	pending   []store.MeetingSegment
	runes     int
	nextIdx   int
	queues    map[Source][]correctionWork
	next      Source
	running   bool
	closed    bool

	chat    SummaryChat
	context TranscriptContext
	style   string
	llm     string
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
	return &correctionOverlap{
		meeting: id, fixed: map[string]map[int]string{}, completed: map[string]bool{},
		queues: map[Source][]correctionWork{}, next: SourceMic,
	}
}

// speculate corrects the first track's final segments in the background;
// a failed batch is left out and asked again by the correction after the
// final pass.
func (o *correctionOverlap) speculate(ctx context.Context, cfg Config, segs []Segment) {
	o.add(ctx, cfg, SourceMic, segs)
}

// speculateSystemDelta adds a completed MTD chunk's new final lines. A
// pending batch is sealed only at the same line/rune boundary as the final
// transcript, so a chunk never creates an extra small request.
func (o *correctionOverlap) speculateSystemDelta(ctx context.Context, cfg Config, segs []Segment) {
	o.add(ctx, cfg, SourceSystem, segs)
}

func (o *correctionOverlap) add(ctx context.Context, cfg Config, track Source, segs []Segment) {
	if o == nil {
		return
	}
	o.mu.Lock()
	closed := o.closed
	o.mu.Unlock()
	if closed {
		return
	}
	chat, err := cfg.LLM(ctx, cfg.Settings().LLM)
	if err != nil || chat == nil {
		return
	}
	var tc TranscriptContext
	if cfg.Context != nil {
		tc = cfg.Context(ctx)
	}
	style := cfg.Settings().CorrectionStyle
	llmID := cfg.Settings().LLM
	o.mu.Lock()
	defer o.mu.Unlock()
	if o.closed {
		return
	}
	if o.chat == nil {
		o.chat, o.context, o.style, o.llm = chat, tc, style, llmID
	}
	if o.fixed == nil {
		o.fixed = map[string]map[int]string{}
	}
	if o.completed == nil {
		o.completed = map[string]bool{}
	}
	if o.queues == nil {
		o.queues = map[Source][]correctionWork{}
	}
	for _, s := range segs {
		n := len([]rune(s.Text))
		if len(o.pending) > 0 && (len(o.pending) == correctBatchLines || o.runes+n > correctBatchRunes) {
			o.sealLocked(track)
		}
		o.pending = append(o.pending, store.MeetingSegment{Idx: o.nextIdx, Source: toSegmentSource(track), Text: s.Text})
		o.nextIdx++
		o.runes += n
		if len(o.pending) == correctBatchLines || o.runes == correctBatchRunes {
			o.sealLocked(track)
		}
	}
	o.startWorkerLocked(ctx)
}

// flush preserves the first track's eager tail batch on paths without
// chunk deltas. A later mixed-track batch is asked again when it differs.
func (o *correctionOverlap) flush(ctx context.Context) {
	o.mu.Lock()
	defer o.mu.Unlock()
	if o.closed {
		return
	}
	o.sealLocked(SourceSystem)
	o.startWorkerLocked(ctx)
}

func (o *correctionOverlap) sealLocked(track Source) {
	if len(o.pending) == 0 {
		return
	}
	// The batch keeps the lines' store indices, so its request reads
	// exactly like the final pass's request for the same lines.
	batch := append([]store.MeetingSegment(nil), o.pending...)
	o.queues[track] = append(o.queues[track], correctionWork{track: track, batch: batch})
	o.pending, o.runes = nil, 0
}

func (o *correctionOverlap) startWorkerLocked(ctx context.Context) {
	if o.running || o.closed {
		return
	}
	if len(o.queues[SourceMic])+len(o.queues[SourceSystem]) == 0 {
		return
	}
	o.running = true
	o.wg.Add(1)
	go func() {
		defer o.wg.Done()
		o.run(ctx)
	}()
}

func (o *correctionOverlap) run(ctx context.Context) {
	for {
		o.mu.Lock()
		work, ok := o.nextWorkLocked()
		if !ok {
			o.running = false
			o.mu.Unlock()
			return
		}
		chat, tc, style := o.chat, o.context, o.style
		o.mu.Unlock()
		got, err := o.correctBatch(ctx, chat, tc, style, work.batch)
		if err != nil {
			continue
		}
		key := correctionBatchKey(work.batch)
		o.mu.Lock()
		// A batch's answers are kept by store index inside their own batch:
		// identical text on different lines can never share an answer.
		batchFixed := o.fixed[key]
		if batchFixed == nil {
			batchFixed = map[int]string{}
			o.fixed[key] = batchFixed
		}
		for _, s := range work.batch {
			if t, ok := got[s.Idx]; ok {
				batchFixed[s.Idx] = t
			}
		}
		o.completed[key] = true
		o.mu.Unlock()
	}
}

func (o *correctionOverlap) nextWorkLocked() (correctionWork, bool) {
	mic, system := len(o.queues[SourceMic]) > 0, len(o.queues[SourceSystem]) > 0
	if !mic && !system {
		return correctionWork{}, false
	}
	track := o.next
	if track != SourceMic && track != SourceSystem {
		track = SourceMic
	}
	if !mic {
		track = SourceSystem
	} else if !system {
		track = SourceMic
	}
	work := o.queues[track][0]
	o.queues[track] = o.queues[track][1:]
	if track == SourceMic {
		o.next = SourceSystem
	} else {
		o.next = SourceMic
	}
	return work, true
}

// takeFor closes submission, waits for the worker, and returns only results
// produced with the final correction's model, style and transcript context:
// per exact batch, the corrected text of each of its lines by store index.
func (o *correctionOverlap) takeFor(llmID string, style string, tc TranscriptContext) (map[string]map[int]string, map[string]bool) {
	o.mu.Lock()
	o.closed = true
	o.mu.Unlock()
	o.wg.Wait()
	o.mu.Lock()
	defer o.mu.Unlock()
	if o.llm != llmID || o.style != style || !reflect.DeepEqual(o.context, tc) {
		return nil, nil
	}
	return o.fixed, o.completed
}

// abort discards this run after cancellation and waits for late workers,
// so no result can leak into a later finalization of the same meeting.
func (o *correctionOverlap) abort() {
	o.mu.Lock()
	o.closed = true
	o.mu.Unlock()
	if o.cancel != nil {
		o.cancel()
	}
	o.wg.Wait()
	o.mu.Lock()
	o.fixed = map[string]map[int]string{}
	o.completed = map[string]bool{}
	o.pending = nil
	o.nextIdx = 0
	o.queues = map[Source][]correctionWork{}
	o.mu.Unlock()
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
