package meeting_test

import (
	"testing"

	"github.com/sgl-project/sglang-omni/Voxt/internal/meeting"
)

func segs(texts ...string) []meeting.Segment {
	out := make([]meeting.Segment, len(texts))
	for i, t := range texts {
		out[i] = meeting.Segment{Text: t}
	}
	return out
}

func TestTranslateBatchesLimits(t *testing.T) {
	line := stringsRepeat("z", 60)
	long := make([]meeting.Segment, 5)
	for i := range long {
		long[i] = meeting.Segment{Text: line}
	}
	batches := meeting.TranslateBatches(long)
	if len(batches) != 2 {
		t.Fatalf("five sentences: want 2 batches got %d", len(batches))
	}
	if len(batches[0]) > 4 {
		t.Fatalf("first batch %d sentences", len(batches[0]))
	}
	var chars int
	for _, s := range batches[0] {
		chars += len(s.Text)
	}
	if chars > 240 {
		t.Fatalf("first batch %d chars", chars)
	}
	huge := []meeting.Segment{{Text: stringsRepeat("x", 300)}}
	if len(meeting.TranslateBatches(huge)) != 1 {
		t.Fatal("one long line per batch")
	}
}

func stringsRepeat(s string, n int) string {
	out := make([]byte, 0, len(s)*n)
	for i := 0; i < n; i++ {
		out = append(out, s...)
	}
	return string(out)
}

func TestTranslationQueueDropsOldest(t *testing.T) {
	var q meeting.TranslationQueue
	for i := 0; i < 10; i++ {
		q.Push(segs("batch"))
	}
	if q.Len() != 9 {
		t.Fatalf("want 9 queued batches got %d", q.Len())
	}
}

func TestMeetingStartsWithTranslationOff(t *testing.T) {
	m, ctx, _, _ := testManager(t, nil, nil, false)
	id, err := m.Start(ctx, meeting.StartRequest{Mode: meeting.ModeMeeting})
	if err != nil {
		t.Fatal(err)
	}
	if m.Current().Translate != "" {
		t.Fatalf("translation should start off, got %q", m.Current().Translate)
	}
	_ = m.Finish(ctx, id)
}
