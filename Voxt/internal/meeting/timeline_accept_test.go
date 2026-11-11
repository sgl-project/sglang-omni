package meeting

import (
	"testing"
	"time"
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
