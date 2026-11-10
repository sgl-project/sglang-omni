package meeting

import (
	"strings"
	"time"
)

// Problem keys shown in the meeting overlay (i18n message ids in package main).
const (
	ProblemKeySystemSilent = "meeting.system.silent"
	ProblemKeyLiveBacklog  = "meeting.live.backlog"
	ProblemKeyDiskLow      = "meeting.disk.low"
	ProblemKeyHelperOld    = "meeting.helper.old"
)

const (
	silentLevelDBFS       = -120
	systemSilenceDuration = 30 * time.Second
	liveBacklogThreshold  = 60 * time.Second
)

type problemState struct {
	clock              func() time.Time
	systemQuietSince   time.Time
	systemSilentWarned bool
	micHasText         bool
	liveBacklog        time.Duration
	diskLow            bool
	helperOld          bool
	trackProblem       *Problem
}

func newProblemState(now func() time.Time) problemState {
	if now == nil {
		now = time.Now
	}
	return problemState{clock: now}
}

func (p *problemState) reset() {
	p.systemQuietSince = time.Time{}
	p.systemSilentWarned = false
	p.micHasText = false
	p.liveBacklog = 0
	p.diskLow = false
	p.helperOld = false
	p.trackProblem = nil
}

func (p *problemState) setTrackProblem(pr *Problem) { p.trackProblem = pr }

func (p *problemState) noteLevel(src Source, dbfs float64) {
	if src != SourceSystem {
		return
	}
	if dbfs > silentLevelDBFS {
		p.systemQuietSince = time.Time{}
		return
	}
	if p.systemQuietSince.IsZero() {
		p.systemQuietSince = p.clock()
	}
}

func (p *problemState) noteMicText() { p.micHasText = true }

func (p *problemState) noteBacklog(d time.Duration) { p.liveBacklog = d }

func (p *problemState) noteFreeSpace(free int64) {
	p.diskLow = free < diskMinStartBytes && free >= diskMinRecordBytes
}

func (p *problemState) noteHelperOld(old bool) { p.helperOld = old }

// pick chooses the user-visible problem for a live meeting snapshot.
func (p *problemState) pick(mode Mode, track *Problem) *Problem {
	if track != nil && strings.HasPrefix(track.Key, "meeting.track.") {
		return track
	}
	if p.trackProblem != nil && strings.HasPrefix(p.trackProblem.Key, "meeting.track.") {
		return p.trackProblem
	}
	if p.helperOld && modeNeedsSource(mode, SourceSystem) {
		return &Problem{Key: ProblemKeyHelperOld}
	}
	if p.diskLow {
		return &Problem{Key: ProblemKeyDiskLow}
	}
	if p.liveBacklog >= liveBacklogThreshold {
		return &Problem{Key: ProblemKeyLiveBacklog}
	}
	if p.systemSilentWarned || !p.micHasText || !modeNeedsSource(mode, SourceSystem) {
		return nil
	}
	if p.systemQuietSince.IsZero() {
		return nil
	}
	if p.clock().Sub(p.systemQuietSince) < systemSilenceDuration {
		return nil
	}
	p.systemSilentWarned = true
	return &Problem{Key: ProblemKeySystemSilent}
}

// systemSilenceForTest returns accumulated quiet time on the system track (tests only).
func (p *problemState) systemSilenceForTest(now time.Time) time.Duration {
	if p.systemQuietSince.IsZero() {
		return 0
	}
	return now.Sub(p.systemQuietSince)
}
