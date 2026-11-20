package main

// Home's activity calendar (home_activity_view.go): the days Voxt gave a
// result, counted in a document of their own (internal/activity) so the
// history's clean-up keeps them. The first read of a store without that
// document counts the history it has. Like Today, nothing is counted
// while the history is off.

import (
	"cmp"
	"context"
	"strings"
	"sync"
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/activity"
	"github.com/sgl-project/sglang-omni/Voxt/internal/session"
	"github.com/sgl-project/sglang-omni/Voxt/internal/store"
)

type homeActivity struct {
	log activity.Log
	// loaded: the log was read; loading: a read is on its way. pending
	// holds the results that ended before it landed.
	loaded, loading bool
	pending         []activityResult
	// gen numbers the log's changes (homeViewCacheKey); page is how many
	// calendar pages back from today the calendar shows.
	gen, page int
	// saves orders the writes: only a newer log replaces the stored one.
	saves activitySaves
}

// activityResult is a result that ended before the log was read: its
// session (its History entry's ID) and when it started.
type activityResult struct {
	id string
	at time.Time
}

type activitySaves struct {
	mu            sync.Mutex
	next, written int
}

// loadActivity reads the log once, or counts the history into a new one.
func (a *app) loadActivity() {
	act := &a.act
	if act.loaded || act.loading || a.lib == nil || a.lib.store == nil {
		return
	}
	act.loading = true
	docs, history := a.lib.store.Documents(), a.lib.store.History()
	a.async(func() {
		ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
		defer cancel()
		log, seen, err := readActivity(ctx, docs, history, time.Now())
		a.post(func() {
			act := &a.act
			act.loading = false
			if err != nil {
				return // Home shows the calendar empty; the next visit tries again
			}
			act.log, act.loaded = log, true
			// What ended meanwhile, unless the history counted it: its
			// entry is written before its end is published, but a count
			// that read that day before then did not see it.
			for _, r := range act.pending {
				if !seen[r.id] {
					act.log.Add(r.at)
				}
			}
			act.pending = nil
			act.gen++
			a.saveActivity()
		})
	})
}

// readActivity reads the stored log; without one it counts the history's
// finished results, and reports which of now's day and the day before it
// counted (seen, by entry ID).
func readActivity(ctx context.Context, docs store.Documents, history store.History, now time.Time) (log activity.Log, seen map[string]bool, err error) {
	data, ok, err := docs.Get(ctx, activity.Key)
	if err != nil {
		return log, nil, err
	}
	if ok {
		log, err = activity.Unmarshal(data)
		return log, nil, err
	}
	seen = map[string]bool{}
	recent := now.AddDate(0, 0, -2)
	const page = 500
	for off := 0; ; off += page {
		entries, err := history.List(ctx, store.HistoryQuery{Limit: page, Offset: off})
		if err != nil {
			return log, nil, err
		}
		for _, e := range entries {
			if e.Status == store.StatusDone && strings.TrimSpace(e.Result) != "" {
				log.Add(e.CreatedAt.Local())
				if e.CreatedAt.After(recent) {
					seen[e.ID] = true
				}
			}
		}
		if len(entries) < page {
			return log, seen, nil
		}
	}
}

// recordActivity counts a finished result on today's square.
func (a *app) recordActivity(s session.Snapshot) {
	if s.Try || s.State != session.Done || strings.TrimSpace(s.Result) == "" || !a.prefs.Privacy.KeepHistory {
		return
	}
	act := &a.act
	// Its day is its start's, as its History entry's (CreatedAt).
	result := activityResult{id: s.ID, at: cmp.Or(s.StartedAt, time.Now())}
	if !act.loaded {
		act.pending = append(act.pending, result)
		a.loadActivity()
		return
	}
	act.log.Add(result.at)
	act.gen++
	a.saveActivity()
}

// saveActivity stores the log in the background; a slower write of an
// older log does not replace a newer one.
func (a *app) saveActivity() {
	if a.lib == nil || a.lib.store == nil {
		return
	}
	data, err := a.act.log.Marshal()
	if err != nil {
		return
	}
	saves := &a.act.saves
	saves.mu.Lock()
	saves.next++
	seq := saves.next
	saves.mu.Unlock()
	docs := a.lib.store.Documents()
	a.async(func() {
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		saves.mu.Lock()
		defer saves.mu.Unlock()
		if seq < saves.written {
			return
		}
		if docs.Put(ctx, activity.Key, data) == nil {
			saves.written = seq
		}
	})
}

// The calendar's arrows: a page of weeks back, or forward to today.
func (a *app) activityBack() { a.act.page++ }
func (a *app) activityForward() {
	if a.act.page > 0 {
		a.act.page--
	}
}
