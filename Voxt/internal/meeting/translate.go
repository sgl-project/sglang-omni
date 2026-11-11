package meeting

import (
	"context"
	"fmt"
	"strings"
	"sync"
	"unicode/utf8"

	"github.com/sgl-project/sglang-omni/Voxt/internal/llm"
)

const (
	maxTranslateBatchSentences = 4
	maxTranslateBatchRunes     = 240
	maxTranslateScheduledBatch = 9
)

// TranslateTargetLanguage returns the code used for live translation ("" = en).
func TranslateTargetLanguage(settingsTarget string) string {
	if strings.TrimSpace(settingsTarget) == "" {
		return "en"
	}
	return strings.TrimSpace(settingsTarget)
}

// TranslateBatches groups pending segments into LLM batches (Voxt 1.x limits).
func TranslateBatches(pending []Segment) [][]Segment {
	var batches [][]Segment
	var cur []Segment
	var runes int
	flush := func() {
		if len(cur) == 0 {
			return
		}
		batches = append(batches, cur)
		cur = nil
		runes = 0
	}
	for _, seg := range pending {
		text := strings.TrimSpace(seg.Text)
		if text == "" {
			continue
		}
		n := utf8.RuneCountInString(text)
		if len(cur) > 0 && (len(cur) >= maxTranslateBatchSentences || runes+n > maxTranslateBatchRunes) {
			flush()
		}
		cur = append(cur, seg)
		runes += n
	}
	flush()
	return batches
}

// TranslationQueue caps outstanding translation batches and drops the oldest when over limit.
type TranslationQueue struct {
	batches [][]Segment
}

// Len is the number of scheduled batches.
func (q *TranslationQueue) Len() int { return len(q.batches) }

// Push appends a batch, evicting the oldest when more than nine batches are queued.
func (q *TranslationQueue) Push(batch []Segment) {
	if len(batch) == 0 {
		return
	}
	q.batches = append(q.batches, batch)
	for len(q.batches) > maxTranslateScheduledBatch {
		q.batches = q.batches[1:]
	}
}

// Pop removes the next batch.
func (q *TranslationQueue) Pop() ([]Segment, bool) {
	if len(q.batches) == 0 {
		return nil, false
	}
	b := q.batches[0]
	q.batches = q.batches[1:]
	return b, true
}

// TranslationChat translates a batch of segments through a language model.
type TranslationChat interface {
	Chat(ctx context.Context, req llm.Request) (llm.Response, error)
}

// TranslateBatchPrompt builds the user message for a translation batch.
func TranslateBatchPrompt(targetLang string, batch []Segment) string {
	var b strings.Builder
	fmt.Fprintf(&b, "Translate each line into %s. Reply with one translated line per input line, same order, no numbering.\n\n", targetLang)
	for _, seg := range batch {
		fmt.Fprintln(&b, strings.TrimSpace(seg.Text))
	}
	return strings.TrimSpace(b.String())
}

// ParseTranslationLines splits model output into per-segment translations.
func ParseTranslationLines(batch []Segment, raw string) []string {
	lines := strings.Split(strings.TrimSpace(raw), "\n")
	out := make([]string, 0, len(batch))
	for i := range batch {
		if i >= len(lines) {
			out = append(out, "")
			continue
		}
		out = append(out, strings.TrimSpace(lines[i]))
	}
	return out
}

// LiveTranslator serializes realtime meeting translation (one batch at a time).
type LiveTranslator struct {
	Chat   TranslationChat
	Target string
	OnDone func(segIdx int, translation string)

	mu    sync.Mutex
	queue TranslationQueue
	busy  bool
}

// Reset clears queued work for a new meeting.
func (t *LiveTranslator) Reset() {
	t.mu.Lock()
	t.queue = TranslationQueue{}
	t.busy = false
	t.mu.Unlock()
}

// Enqueue schedules segments when live translation is enabled.
func (t *LiveTranslator) Enqueue(pending []Segment) {
	if t == nil || t.Chat == nil || strings.TrimSpace(t.Target) == "" {
		return
	}
	for _, batch := range TranslateBatches(pending) {
		t.mu.Lock()
		t.queue.Push(batch)
		start := !t.busy
		t.busy = true
		t.mu.Unlock()
		if start {
			go t.drain()
		}
	}
}

func (t *LiveTranslator) drain() {
	for {
		t.mu.Lock()
		batch, ok := t.queue.Pop()
		if !ok {
			t.busy = false
			t.mu.Unlock()
			return
		}
		t.mu.Unlock()
		if len(batch) == 0 {
			continue
		}
		ctx := context.Background()
		prompt := TranslateBatchPrompt(t.Target, batch)
		resp, err := t.Chat.Chat(ctx, llm.Request{Messages: []llm.Message{{Role: llm.User, Content: prompt}}})
		if err != nil {
			continue
		}
		lines := ParseTranslationLines(batch, resp.Text)
		if t.OnDone == nil {
			continue
		}
		for i, seg := range batch {
			if i < len(lines) && lines[i] != "" {
				t.OnDone(seg.StoreIdx, lines[i])
			}
		}
	}
}
