package meeting

import (
	"testing"
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/asr"
	"github.com/sgl-project/sglang-omni/Voxt/internal/vad"
)

func rateSamples(d time.Duration) int {
	return int(d.Seconds() * float64(asr.SampleRate))
}

func TestPlanChunksMax22s(t *testing.T) {
	total := rateSamples(50 * time.Second)
	speech := []vad.Segment{{Start: 0, End: total}}
	p := DefaultChunkParams()
	ch := PlanChunks(speech, total, p)
	if len(ch) < 2 {
		t.Fatalf("want multiple chunks, got %d", len(ch))
	}
	max := p.maxSamples()
	for i, c := range ch {
		inner := c.End - c.Start
		if inner > max+2*p.padSamples() {
			t.Fatalf("chunk %d span %d > max %d", i, inner, max+2*p.padSamples())
		}
	}
}

func TestPlanChunksCutsAtLongPause(t *testing.T) {
	rate := asr.SampleRate
	pause := rateSamples(900 * time.Millisecond)
	aEnd := rateSamples(10 * time.Second)
	bStart := aEnd + pause
	bEnd := bStart + rateSamples(10*time.Second)
	speech := []vad.Segment{{Start: 0, End: aEnd}, {Start: bStart, End: bEnd}}
	p := DefaultChunkParams()
	ch := PlanChunks(speech, bEnd, p)
	if len(ch) != 2 {
		t.Fatalf("want 2 chunks at pause, got %d", len(ch))
	}
	if ch[0].End > bStart+p.padSamples()+rate/10 {
		t.Fatalf("first chunk end %d not before pause", ch[0].End)
	}
	_ = rate
}

func TestPlanChunksHardOverlap(t *testing.T) {
	total := rateSamples(45 * time.Second)
	speech := []vad.Segment{{Start: 0, End: total}}
	p := DefaultChunkParams()
	ch := PlanChunks(speech, total, p)
	if len(ch) < 2 {
		t.Fatal("want hard-cut chunks")
	}
	overlap := p.overlapSamples()
	for i := 1; i < len(ch); i++ {
		if ch[i-1].End-ch[i].Start < overlap {
			t.Fatalf("missing overlap between %d and %d", i-1, i)
		}
	}
}

func TestPlanChunksCapMaxDuration(t *testing.T) {
	total := rateSamples(40 * time.Second)
	speech := []vad.Segment{{Start: 0, End: total}}
	p := DefaultChunkParams()
	p.CapMaxDur = 15 * time.Second
	ch := PlanChunks(speech, total, p)
	cap := int(15 * float64(asr.SampleRate))
	for _, c := range ch {
		if c.End-c.Start > cap+2*p.padSamples()+1 {
			t.Fatalf("chunk %d-%d longer than cap", c.Start, c.End)
		}
	}
}

func TestPlanChunksMergesShort(t *testing.T) {
	short := rateSamples(300 * time.Millisecond)
	long := rateSamples(5 * time.Second)
	speech := []vad.Segment{
		{Start: 0, End: long},
		{Start: long + rateSamples(2*time.Second), End: long + rateSamples(2*time.Second) + short},
	}
	total := speech[len(speech)-1].End
	p := DefaultChunkParams()
	ch := PlanChunks(speech, total, p)
	for _, c := range ch {
		if c.End-c.Start < p.minSamples() && len(ch) > 1 {
			t.Fatalf("short chunk %d-%d not merged", c.Start, c.End)
		}
	}
}

func TestDedupeOverlapText(t *testing.T) {
	got := DedupeOverlapText("we discussed the plan", "the plan for next week")
	if got != "for next week" {
		t.Fatalf("got %q", got)
	}
}
