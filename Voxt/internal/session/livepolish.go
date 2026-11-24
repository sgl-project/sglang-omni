package session

import (
	"context"
	"slices"
	"strings"
	"sync"
	"time"
	"unicode"
	"unicode/utf8"

	"github.com/sgl-project/sglang-omni/Voxt/internal/compose"
	"github.com/sgl-project/sglang-omni/Voxt/internal/dict"
	"github.com/sgl-project/sglang-omni/Voxt/internal/llm"
	"github.com/sgl-project/sglang-omni/Voxt/internal/store"
)

// livePolish cleans a dictation up while the user speaks (behavior.md
// 5.31b): the whole sentences of the stream's live text go to the
// language model as they come, so that Stop leaves only what was said
// last, or nothing.
//
// The live text is the speech model's guess: at Stop each clean-up counts
// only when its text starts the final text word for word and ends at one
// of its sentence ends. The text is cleaned in blocks of at most
// compose.PieceChars characters, each block whole and with the prompt of a
// clean-up after Stop: a block is asked again whenever the live text adds
// a sentence to it, so that a list or a correction across its sentences is
// cleaned as at Stop. A block that is full is kept (committed), and the
// next one goes with the cleaned text before it (compose.Input.Before), as
// does the text left at Stop. The last settings.Transcribe.RecleanSentences
// sentences before that text are cleaned again with it, so that what was
// said last can correct them ("Tuesday. No, make it Thursday.").
//
// add is the stream's reader goroutine's, work the worker's, finish the
// session goroutine's (after the reader returned); mu guards the rest, and
// cond (on mu) wakes the worker.
type livePolish struct {
	r             *run
	ctx           context.Context
	cancel        context.CancelFunc
	workerDone    chan struct{}
	thinkOffUnder int
	reclean       int // sentences cleaned again with the text left at Stop

	mu      sync.Mutex
	cond    *sync.Cond
	asked   int          // requests while recording
	dropped int          // of them, cancelled as the live text changed
	text    string       // the newest live text, as the final text is cleaned (aliases replaced)
	blocks  []liveResult // committed, in order
	epoch   int          // bumped when blocks change: a clean-up of an older epoch has other text before it
	done    []liveResult // the clean-ups of the open block: each of a stretch that starts the text after blocks
	flight  *liveFlight  // the request under way, nil when none
	closed  bool         // the final text came, or the live clean-up gave up: no more requests
}

// liveResult is the clean-up of src, a stretch of the live text that ends
// a sentence: the parts it was asked in (compose.BuildParts) and their
// answers.
type liveResult struct {
	src      string
	parts    []compose.Part
	answers  []string
	chars    int  // the dictation's characters up to the end of src
	thinkOff bool // asked with thinking off (chars under the service's ThinkingOffUnder)
	epoch    int
	usage    llm.Usage
}

// liveFlight is the request under way: done closes when it ends, ok when
// it answered.
type liveFlight struct {
	liveResult
	cancel context.CancelFunc
	done   chan struct{}
	ok     bool
}

// hearLive takes a live text of the recording (asr.EventPartial) to the
// live clean-up. Once the speech model reports stretches of its own
// (hearPiece) the final text is made of those, and the live clean-up ends.
func (r *run) hearLive(text string) {
	if r.liveOff {
		return
	}
	r.liveOnce.Do(func() { r.live = newLivePolish(r) })
	if r.live == nil {
		return
	}
	text = strings.TrimSpace(text)
	if r.settings.Dictionary.Enabled {
		text = dict.Replace(text, r.terms)
	}
	r.live.add(text)
}

// endLive ends the live clean-up while the recording goes on (the
// stream's reader goroutine).
func (r *run) endLive() {
	r.liveOff = true
	if r.live != nil {
		r.live.stop()
	}
}

// newLivePolish starts the live clean-up of a Transcribe that cleans up
// with a language model, unless the user has it wait for Stop
// (settings.Transcribe.CleanAfterStop) or the glossary depends on the text
// (more than compose.GlossaryAllUpTo terms: only the whole text picks its
// terms); nil when it does not run.
func newLivePolish(r *run) *livePolish {
	if r.req.Kind != store.KindTranscribe || r.settings.Features.Transcribe.CleanAfterStop || !r.transcribeCleanup() ||
		r.svc.LLM == nil || len(r.glossary()) > compose.GlossaryAllUpTo {
		return nil
	}
	ctx, cancel := context.WithCancel(r.llmCtx)
	lp := &livePolish{r: r, ctx: ctx, cancel: cancel, workerDone: make(chan struct{}), thinkOffUnder: r.thinkOffUnder(),
		reclean: max(0, r.settings.Features.Transcribe.RecleanSentences)}
	lp.cond = sync.NewCond(&lp.mu)
	go func() {
		<-ctx.Done()
		lp.mu.Lock()
		lp.closed = true
		lp.cond.Broadcast()
		lp.mu.Unlock()
	}()
	go lp.work()
	return lp
}

// stop gives the live clean-up up: Stop cleans the whole text.
func (lp *livePolish) stop() {
	lp.mu.Lock()
	lp.closed = true
	lp.blocks, lp.done = nil, nil
	lp.mu.Unlock()
	lp.cancel()
}

// add takes the newest live text: the blocks it no longer starts with go,
// and so does a request for a stretch it no longer has.
func (lp *livePolish) add(text string) {
	lp.mu.Lock()
	defer lp.mu.Unlock()
	if lp.closed || text == lp.text {
		return
	}
	lp.text = text
	if k, _ := chainIn(text, lp.blocks); k < len(lp.blocks) {
		lp.blocks, lp.done = lp.blocks[:k], nil
		lp.epoch++
	}
	if f := lp.flight; f != nil && (f.epoch != lp.epoch || !endsSentence(lp.open(), f.src)) {
		f.cancel()
	}
	lp.cond.Broadcast()
}

// open is the live text after the blocks (lp.mu held; the blocks start it).
func (lp *livePolish) open() string {
	_, end := chainIn(lp.text, lp.blocks)
	return strings.TrimLeftFunc(lp.text[end:], unicode.IsSpace)
}

// next is the stretch to clean next (lp.mu held), false for none: the
// open block's newest whole sentences, unless they are cleaned already. A
// full block whose clean-up is done is committed first.
func (lp *livePolish) next() (liveResult, bool) {
	for {
		open := lp.open()
		src, full := blockOf(open)
		if src == "" {
			return liveResult{}, false
		}
		i := slices.IndexFunc(lp.done, func(d liveResult) bool { return d.src == src })
		if i < 0 {
			chars := utf8.RuneCountInString(lp.text[:len(lp.text)-len(open)+len(src)])
			return liveResult{src: src, chars: chars, thinkOff: lp.thinkOffUnder > 0 && chars < lp.thinkOffUnder, epoch: lp.epoch}, true
		}
		if !full {
			return liveResult{}, false
		}
		lp.blocks, lp.done = append(lp.blocks, lp.done[i]), nil
		lp.epoch++
	}
}

// blockOf is the stretch of open to clean: the longest one that ends a
// sentence within compose.PieceChars characters; full when open is longer
// (the stretch is a block of its own, the text after it the next block's).
func blockOf(open string) (string, bool) {
	src := ""
	for _, end := range compose.SentenceEnds(open) {
		if utf8.RuneCountInString(open[:end]) > compose.PieceChars {
			break
		}
		src = open[:end]
	}
	return src, src != "" && utf8.RuneCountInString(open) > compose.PieceChars
}

// work cleans the stretches next gives, one request at a time, until
// finish or the session ends. A request that fails ends the live clean-up:
// Stop cleans the text, as without it.
func (lp *livePolish) work() {
	defer close(lp.workerDone)
	for {
		lp.mu.Lock()
		job, ok := liveResult{}, false
		for !lp.closed {
			if job, ok = lp.next(); ok {
				break
			}
			lp.cond.Wait()
		}
		if !ok {
			lp.mu.Unlock()
			return
		}
		ctx, cancel := context.WithCancel(lp.ctx)
		f := &liveFlight{liveResult: job, cancel: cancel, done: make(chan struct{})}
		lp.flight = f
		lp.asked++
		before := joinResults(lp.blocks, lp.text)
		lp.mu.Unlock()
		start := time.Now()
		res, err := lp.clean(ctx, job, before)
		cancel()
		lp.mu.Lock()
		lp.flight = nil
		f.liveResult, f.ok = res, err == nil
		switch {
		case err == nil && res.epoch == lp.epoch:
			lp.done = append(lp.done, res)
			lp.r.logf("info", "cleaned up %d characters while recording in %v (%d prompt tokens, %d cached)",
				utf8.RuneCountInString(res.src), time.Since(start).Round(time.Millisecond), res.usage.PromptTokens, res.usage.CachedTokens)
		case ctx.Err() != nil:
			lp.dropped++
		case err != nil:
			lp.r.logf("warning", "the live clean-up stops, Stop cleans the text: %v", err)
			lp.closed = true
		}
		close(f.done)
		lp.cond.Broadcast()
		lp.mu.Unlock()
	}
}

// clean asks the language model to clean job's stretch, after before (the
// blocks' cleaned text, "" for the first block).
func (lp *livePolish) clean(ctx context.Context, job liveResult, before string) (liveResult, error) {
	r := lp.r
	in := r.transcribeInput(job.src)
	in.Before = before
	parts, err := compose.BuildParts(in)
	if err != nil {
		return job, err
	}
	job.parts = parts
	for _, p := range parts {
		req := llm.Request{Messages: llmMessages(p.Output), Skill: llm.SkillTranscribeEnhance, InputChars: job.chars}
		resp, err := r.llmChatContext(ctx, r.m.cfg.llmDeadlines(req, utf8.RuneCountInString(p.Text), r.llmFloor(r.m.cfg.LLMTimeout)))
		if err != nil {
			return job, err
		}
		job.answers = append(job.answers, compose.CleanOutput(resp.Text, p.Text))
		job.usage = job.usage.Add(resp.Usage)
	}
	return job, nil
}

// finish ends the live clean-up with the final text (as cleaned, aliases
// replaced): the clean-ups that start it, as parts and their answers, and
// the text after them. A request under way for exactly the rest is
// awaited: it is ahead of a new one for the same text; any other is
// cancelled. A clean-up asked with thinking other than the final text's
// (ThinkingOffUnder) does not count.
func (lp *livePolish) finish(text string) (parts []compose.Part, answers []string, tail string) {
	chars := utf8.RuneCountInString(text)
	thinkOff := lp.thinkOffUnder > 0 && chars < lp.thinkOffUnder
	lp.mu.Lock()
	lp.closed = true
	lp.cond.Broadcast()
	k := slices.IndexFunc(lp.blocks, func(b liveResult) bool { return b.thinkOff != thinkOff })
	if k < 0 {
		k = len(lp.blocks)
	}
	k, end := chainIn(text, lp.blocks[:k])
	blocks := slices.Clone(lp.blocks[:k])
	open := strings.TrimLeftFunc(text[end:], unicode.IsSpace)
	var best *liveResult
	if k == len(lp.blocks) {
		best = lp.base(open, thinkOff)
	}
	var wait *liveFlight
	if f := lp.flight; f != nil {
		if k == len(lp.blocks) && f.epoch == lp.epoch && f.src == open && f.thinkOff == thinkOff {
			wait = f
		} else {
			f.cancel()
		}
	}
	if best != nil {
		blocks = append(blocks, *best)
	}
	asked, dropped := lp.asked, lp.dropped
	lp.mu.Unlock()
	if wait != nil {
		<-wait.done
		if wait.ok {
			blocks = append(blocks[:k], wait.liveResult)
		}
	}
	lp.cancel()
	<-lp.workerDone
	parts, answers, tail = laid(blocks, text)
	lp.r.logf("info", "the live clean-up covered %d of %d characters (%d requests while recording, %d cancelled as the text changed; awaited the request under way: %t)",
		chars-utf8.RuneCountInString(tail), chars, asked, dropped, wait != nil)
	return parts, answers, tail
}

// base is the open block's clean-up that the text left at Stop goes after
// (lp.mu held): one of all of open when there is, else the longest that
// leaves at least lp.reclean whole sentences before the rest of open, to
// clean again with it.
func (lp *livePolish) base(open string, thinkOff bool) *liveResult {
	units := compose.SentenceEnds(open)
	if len(units) == 0 || units[len(units)-1] != len(open) {
		units = append(units, len(open)) // the unfinished last sentence
	}
	var best *liveResult
	for i, d := range lp.done {
		if d.thinkOff != thinkOff || !endsSentence(open, d.src) {
			continue
		}
		if len(d.src) == len(open) {
			return &lp.done[i]
		}
		after := len(units) - slices.Index(units, len(d.src)) - 1
		if after > lp.reclean && (best == nil || len(d.src) > len(best.src)) {
			best = &lp.done[i]
		}
	}
	return best
}

// chainIn is how many of blocks start text, in order, each at one of its
// sentence ends, and the byte of text after them.
func chainIn(text string, blocks []liveResult) (int, int) {
	end := 0
	for k, b := range blocks {
		rest := text[end:]
		trimmed := strings.TrimLeftFunc(rest, unicode.IsSpace)
		if !endsSentence(trimmed, b.src) {
			return k, end
		}
		end += len(rest) - len(trimmed) + len(b.src)
	}
	return len(blocks), end
}

// endsSentence says whether text starts with src and src ends at one of
// text's sentence ends, or is all of it.
func endsSentence(text, src string) bool {
	if src == "" || !strings.HasPrefix(text, src) {
		return false
	}
	return len(src) == len(text) || slices.Contains(compose.SentenceEnds(text), len(src))
}

// joinResults is the blocks' cleaned text, joined as text, which they
// start, joins them.
func joinResults(blocks []liveResult, text string) string {
	parts, answers, _ := laid(blocks, text)
	return compose.JoinParts(parts, answers)
}

// laid are blocks as parts of text, which they start (each block's last
// part After the white space after it in text), and the text after them.
func laid(blocks []liveResult, text string) (parts []compose.Part, answers []string, tail string) {
	rest := strings.TrimLeftFunc(text, unicode.IsSpace)
	for _, b := range blocks {
		after := rest[len(b.src):]
		rest = strings.TrimLeftFunc(after, unicode.IsSpace)
		ps := slices.Clone(b.parts)
		ps[len(ps)-1].After = after[:len(after)-len(rest)]
		parts, answers = append(parts, ps...), append(answers, b.answers...)
	}
	return parts, answers, rest
}
