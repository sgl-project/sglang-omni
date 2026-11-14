package meeting

import (
	"testing"
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/asr"
)

func TestAcceptJobRejectsStaleJob(t *testing.T) {
	tl := newTimeline(func() time.Time { return time.Unix(0, 0) }, time.Unix(0, 0))
	tl.Begin(SourceMic, "job-1", 0)
	tl.Begin(SourceMic, "job-2", time.Second)
	if tl.acceptJob(SourceMic, "job-1") {
		t.Fatal("stale job should be ignored")
	}
	if !tl.acceptJob(SourceMic, "job-2") {
		t.Fatal("active job should be accepted")
	}
}

func TestTimelineAtMapsOffsetFromJobStart(t *testing.T) {
	tl := newTimeline(func() time.Time { return time.Unix(0, 0) }, time.Unix(0, 0))
	if got := tl.At(SourceMic, "job-1", 1000); got != 0 {
		t.Fatalf("unknown job: got %v want 0", got)
	}
	tl.Begin(SourceMic, "job-1", 3*time.Second)
	if got := tl.At(SourceMic, "job-1", asr.BytesPerSecond/2); got != 3500*time.Millisecond {
		t.Fatalf("half a second in: got %v want 3.5s", got)
	}
	if got := tl.At(SourceMic, "job-1", -10); got != 3*time.Second {
		t.Fatalf("negative offset: got %v want 3s", got)
	}
	if got := tl.At(SourceSystem, "job-1", 0); got != 0 {
		t.Fatalf("other source: got %v want 0", got)
	}
}
