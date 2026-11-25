package session

func (r *run) noteVisibleCleanupResult(result string) {
	if result == "" {
		return
	}
	r.m.mu.Lock()
	cleaned := r.cleanupOutcome == cleanupOutcomeCleaned
	r.m.mu.Unlock()
	if cleaned {
		r.noteVisibleCleanupText()
	}
}

func (r *run) noteVisibleCleanupText() {
	if r.firstCleanText.CompareAndSwap(false, true) {
		r.logTiming("cleanup_first_visible_text")
	}
}

func (r *run) logTiming(event string) {
	startedAt := r.m.snapshot(r).StartedAt
	r.logf("debug", "timing event=%s elapsed=%v", event, r.now().Sub(startedAt).Round(1_000_000))
}
