package session

func (r *run) logTiming(event string) {
	startedAt := r.m.snapshot(r).StartedAt
	r.logf("debug", "timing event=%s elapsed=%v", event, r.now().Sub(startedAt).Round(1_000_000))
}
