package session

import (
	"strings"
	"time"
)

// These publication timings end at the accepted session Snapshot update; UI rendering is not observable here.
func (r *run) noteVisibleCleanupResult(result string) {
	if strings.TrimSpace(result) == "" {
		return
	}
	r.m.mu.Lock()
	cleaned := r.cleanupOutcome == cleanupOutcomeCleaned && !r.firstCleanPublication.Load()
	generation := r.cleanupOutput.generation
	attempt := r.cleanupAttempt
	draftAt := time.Time{}
	if r.cleanDraftGeneration == generation && r.cleanDraftAttempt == attempt {
		draftAt = r.cleanDraftAt
	}
	if cleaned {
		r.firstCleanPublication.Store(true)
	}
	r.m.mu.Unlock()
	if !cleaned {
		return
	}
	resultAt := r.now()
	if !draftAt.IsZero() {
		r.logTimingAt("first_cleaned_draft_published", draftAt)
	}
	r.logTimingAt("first_cleaned_result_published", resultAt)
	publicationAt := resultAt
	if !draftAt.IsZero() && draftAt.Before(publicationAt) {
		publicationAt = draftAt
	}
	r.logTimingAt("first_cleaned_publication", publicationAt)
}

func (r *run) noteCleanDraftPublication(generation, attempt uint64, at time.Time, draft string) {
	draft = strings.TrimSpace(draft)
	if at.IsZero() || draft == "" {
		return
	}
	r.m.mu.Lock()
	defer r.m.mu.Unlock()
	if r.ended || generation != r.cleanupGeneration || attempt != r.cleanupAttempt || !r.cleanupChoiceOpenLocked() {
		return
	}
	if r.cleanDraftGeneration != generation || r.cleanDraftAttempt != attempt || r.cleanDraftAt.IsZero() || at.Before(r.cleanDraftAt) {
		r.cleanDraftGeneration, r.cleanDraftAttempt, r.cleanDraftAt = generation, attempt, at
	}
}

func (r *run) logTiming(event string) {
	r.logTimingAt(event, r.now())
}

func (r *run) logTimingAt(event string, at time.Time) {
	startedAt := r.m.snapshot(r).StartedAt
	r.logf("debug", "timing event=%s elapsed=%v at_unix_nano=%d", event, at.Sub(startedAt).Round(1_000_000), at.UnixNano())
}
