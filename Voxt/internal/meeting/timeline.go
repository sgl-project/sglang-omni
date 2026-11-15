package meeting

import (
	"sync"
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/asr"
)

// jobAnchor maps helper PCM offsets to meeting elapsed time (pause-aware).
type jobAnchor struct {
	job string
	at  time.Duration
}

// Timeline maps capture job byte offsets to elapsed meeting audio time.
type Timeline struct {
	mu sync.Mutex

	activeJob map[Source]string
	anchors   map[Source][]jobAnchor

	pauseActive bool
	pauseStart  time.Time
	pauseTotal  time.Duration
	wallStarted time.Time
	now         func() time.Time
}

func newTimeline(now func() time.Time, started time.Time) *Timeline {
	if now == nil {
		now = time.Now
	}
	return &Timeline{
		activeJob:   map[Source]string{},
		anchors:     map[Source][]jobAnchor{},
		wallStarted: started,
		now:         now,
	}
}

// Begin records where a job starts on the meeting timeline.
func (t *Timeline) Begin(src Source, job string, at time.Duration) {
	t.mu.Lock()
	defer t.mu.Unlock()
	t.activeJob[src] = job
	t.anchors[src] = append(t.anchors[src], jobAnchor{job: job, at: at})
}

// At converts a PCM byte offset in job to meeting elapsed time.
func (t *Timeline) At(src Source, job string, offsetBytes int64) time.Duration {
	t.mu.Lock()
	defer t.mu.Unlock()
	anchor := t.findAnchor(src, job)
	if anchor == nil {
		return 0
	}
	if offsetBytes < 0 {
		offsetBytes = 0
	}
	return anchor.at + time.Duration(offsetBytes)*time.Second/asr.BytesPerSecond
}

// Elapsed is wall time minus pauses while not idle.
func (t *Timeline) Elapsed() time.Duration {
	t.mu.Lock()
	defer t.mu.Unlock()
	return t.elapsedLocked()
}

func (t *Timeline) elapsedLocked() time.Duration {
	if t.wallStarted.IsZero() {
		return 0
	}
	d := t.now().Sub(t.wallStarted) - t.pauseTotal
	if t.pauseActive {
		d -= t.now().Sub(t.pauseStart)
	}
	if d < 0 {
		return 0
	}
	return d
}

// NotePause stops the elapsed clock until NoteResume.
func (t *Timeline) NotePause() {
	t.mu.Lock()
	defer t.mu.Unlock()
	if t.pauseActive {
		return
	}
	t.pauseActive = true
	t.pauseStart = t.now()
}

// NoteResume resumes the elapsed clock.
func (t *Timeline) NoteResume() {
	t.mu.Lock()
	defer t.mu.Unlock()
	if !t.pauseActive {
		return
	}
	t.pauseTotal += t.now().Sub(t.pauseStart)
	t.pauseActive = false
	t.pauseStart = time.Time{}
}

func (t *Timeline) findAnchor(src Source, job string) *jobAnchor {
	list := t.anchors[src]
	for i := len(list) - 1; i >= 0; i-- {
		if list[i].job == job {
			return &list[i]
		}
	}
	return nil
}

func (t *Timeline) acceptJob(src Source, job string) bool {
	t.mu.Lock()
	defer t.mu.Unlock()
	cur, ok := t.activeJob[src]
	if !ok {
		return false
	}
	return cur == job
}
