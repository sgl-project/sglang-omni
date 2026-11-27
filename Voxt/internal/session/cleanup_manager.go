package session

import "time"

func (m *Manager) setCleanupDeadline(r *run, deadline time.Time) {
	m.mu.Lock()
	if !r.ended && r.cleanupDeadline.IsZero() {
		r.cleanupDeadline = deadline
	}
	m.mu.Unlock()
}

func wakeCleanupLocked(r *run) {
	select {
	case r.cleanupWake <- struct{}{}:
	default:
	}
}

func (m *Manager) notifyCleanupDeadline(r *run) {
	m.mu.Lock()
	if r.cleanupChoice == cleanupUndecided && !r.ended {
		r.cleanupDeadlineNotified = true
		wakeCleanupLocked(r)
	}
	m.mu.Unlock()
}

func (m *Manager) publishCleanupCandidate(r *run, output cleanupOutput, generation uint64, publishedAt time.Time) {
	m.mu.Lock()
	r.cleanupWorkerRunning = false
	if publishedAt.IsZero() {
		publishedAt = r.now()
	}
	eligible := r.cleanupChoiceOpenLocked() && !r.cancelled.Load() && generation == r.cleanupGeneration
	if eligible && !r.cleanupDeadline.IsZero() {
		eligible = output.completedAt.Before(r.cleanupDeadline) && publishedAt.Before(r.cleanupDeadline)
	}
	if eligible {
		output.generation = generation
		r.cleanupCandidate = &output
	}
	wakeCleanupLocked(r)
	m.mu.Unlock()
}

// chooseCleanup is the sole terminal arbitration point. Candidate
// publication, Skip, Cancel, Rethink invalidation, and deadline wakeups
// all serialize on Manager.mu before the session owner can deliver.
func (m *Manager) chooseCleanup(r *run) (cleanupChoice, cleanupOutput) {
	m.mu.Lock()
	if r.cancelled.Load() || r.ctx.Err() != nil || r.ended {
		m.mu.Unlock()
		return cleanupCancelled, cleanupOutput{}
	}
	if r.cleanupChoice != cleanupUndecided {
		choice, output := r.cleanupChoice, r.cleanupOutput
		m.mu.Unlock()
		return choice, output
	}
	var snap *Snapshot
	if m.cur != r {
		m.mu.Unlock()
		return cleanupCancelled, cleanupOutput{}
	}
	snap = &m.snap
	choice, outcome := cleanupUndecided, cleanupOutcome("")
	var output cleanupOutput
	switch {
	case r.skipped.Load():
		choice, outcome = cleanupRaw, cleanupOutcomeManualSkip
	case r.cleanupCandidate != nil && r.cleanupCandidate.generation == r.cleanupGeneration:
		choice = cleanupComplete
		output = *r.cleanupCandidate
		outcome = cleanupOutcomeCleaned
		if output.err != nil {
			outcome = cleanupOutcomeLLMError
		}
	case !r.cleanupDeadline.IsZero() && (r.cleanupDeadlineNotified || !r.now().Before(r.cleanupDeadline)):
		choice, outcome = cleanupRaw, cleanupOutcomeSoftDeadline
	case r.rethink.Load():
		if r.cleanupWorkerRunning {
			m.mu.Unlock()
			return cleanupPending, cleanupOutput{}
		}
		m.mu.Unlock()
		return cleanupRetry, cleanupOutput{}
	default:
		m.mu.Unlock()
		return cleanupPending, cleanupOutput{}
	}
	if !r.cleanupChoiceOpenLocked() || (choice != cleanupComplete && choice != cleanupRaw) {
		m.mu.Unlock()
		return cleanupCancelled, cleanupOutput{}
	}
	chosenAt := r.now()
	r.cleanupChoice = choice
	r.cleanupOutcome, r.cleanupOutput = outcome, output
	stop := r.llmStop
	snap.CanSkip = false
	if choice == cleanupRaw {
		snap.Draft = ""
	}
	m.rev++
	snap.Rev = m.rev
	s := snap.clone()
	m.foregroundPub.enqueue(publication{snapshot: s})
	m.mu.Unlock()
	if stop != nil {
		stop()
	}
	r.finalizeEarlyCleanup()
	stopAt := r.stoppedAt()
	if stopAt.IsZero() {
		r.logf("info", "cleanup outcome=%s", outcome)
	} else {
		r.logf("info", "cleanup outcome=%s stop_to_choice=%v", outcome, chosenAt.Sub(stopAt).Round(time.Millisecond))
	}
	return choice, output
}

func (r *run) cleanupDecisionOpenLocked() bool {
	if !r.cleanupChoiceOpenLocked() {
		return false
	}
	deadline := r.cleanupDeadline
	if deadline.IsZero() && r.ordinaryTranscribe() && r.m.cfg.CleanupLatency.AutoFallbackAfter > 0 {
		if stopAt := r.stoppedAt(); !stopAt.IsZero() {
			deadline = stopAt.Add(r.m.cfg.CleanupLatency.AutoFallbackAfter)
		}
	}
	return deadline.IsZero() || r.now().Before(deadline)
}

func (r *run) cleanupChoiceOpenLocked() bool {
	return !r.ended && r.cleanupChoice == cleanupUndecided
}

// beginTranscribeDelivery atomically reserves foreground insertion while
// the Transcribe session still owns the current session slot.
func (m *Manager) beginTranscribeDelivery(r *run, result string) bool {
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.cur != r || r.ended || r.cancelled.Load() || r.ctx.Err() != nil {
		return false
	}
	m.snap.State, m.snap.Result, m.snap.CanSkip = Delivering, result, false
	m.rev++
	m.snap.Rev = m.rev
	m.foregroundPub.enqueue(publication{snapshot: m.snap.clone()})
	return true
}
