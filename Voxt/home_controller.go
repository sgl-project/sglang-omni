package main

import (
	"context"
	"strings"
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/local"
	"github.com/sgl-project/sglang-omni/Voxt/internal/store"
	"github.com/sgl-project/sglang-omni/Voxt/internal/uitext"
)

type homeState struct {
	hm struct {
		// read: the history was read once; loading a read is on its way
		// (seq numbers them: only the last lands); stale an entry was kept
		// since.
		read, loading, stale bool
		seq                  int
		usage                homeUsage
		usageDay             string
		err                  string
		// gen numbers the landed reads; view is the last built state,
		// viewKey its inputs (homePageModel).
		gen     int
		view    homeViewState
		viewKey homeViewCacheKey
	}
	act homeActivity // home_activity_controller.go
}

// homeViewCacheKey is every input of a homeViewState: when it matches the
// last build's, homePageModel reuses the state instead of rebuilding it
// every frame.
type homeViewCacheKey struct {
	settings      uint64
	gen           int
	minute        int64
	location      string
	zoneOffset    int
	locale        string
	lastModelName string
	activityGen   int
	activityPage  int
}

// readHomeUsage reads the newest speech model and the recognition times of
// the last 26 weeks. Entries arrive newest first; typed results have no model.
func readHomeUsage(ctx context.Context, h store.History, now time.Time) (homeUsage, error) {
	y, m, d := now.Date()
	since := time.Date(y, m, d, 0, 0, 0, 0, now.Location()).AddDate(0, 0, -7*26)
	var usage homeUsage
	const page = 500
	for off := 0; ; off += page {
		entries, err := h.List(ctx, store.HistoryQuery{Limit: page, Offset: off})
		if err != nil {
			return usage, err
		}
		for _, entry := range entries {
			if (usage.lastModel != "" || usage.lastProvider != "") && entry.CreatedAt.Before(since) {
				return usage, nil
			}
			if entry.Status != store.StatusDone || strings.TrimSpace(entry.Result) == "" {
				continue
			}
			if usage.lastModel == "" && usage.lastProvider == "" {
				usage.lastProvider, usage.lastModel = entry.ASRProvider, entry.ASRModel
			}
			if !entry.CreatedAt.Before(since) && entry.ASRMS > 0 {
				usage.asrMS += entry.ASRMS
				usage.asrCount++
			}
		}
		if len(entries) < page {
			return usage, nil
		}
	}
}

// loadHome reads Home's history-derived usage in the background.
func (a *app) loadHome() {
	hm := &a.hm
	hm.seq++
	hm.loading, hm.stale = true, false
	seq := hm.seq
	h := a.lib.store.History()
	now := time.Now()
	hm.usageDay = now.Format("2006-01-02")
	a.async(func() {
		ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()
		usage, err := readHomeUsage(ctx, h, now)
		a.post(func() {
			if seq != a.hm.seq {
				return
			}
			hm := &a.hm
			hm.gen++
			hm.loading, hm.read = false, true
			if err != nil {
				hm.err = err.Error()
				return
			}
			hm.usage, hm.err = usage, ""
		})
	})
}

// homeChanged refreshes the history-derived model and latency, now when
// Home shows, else on its next visit.
func (a *app) homeChanged() {
	if a.currentPage() == "home" && !a.settingsOpen {
		a.loadHome()
		return
	}
	a.hm.stale = true
}

func (a *app) prepareHomePage() {
	if a.hm.read && a.hm.usageDay != time.Now().Format("2006-01-02") {
		a.hm.stale = true // the latency's rolling window moved at local midnight
	}
	if !a.hm.loading && (!a.hm.read || a.hm.stale) {
		a.loadHome()
	}
	a.loadActivity()
	if !a.perms.checked {
		a.recheckPermissions()
	}
}

func (a *app) homePageModel(now time.Time) homeViewState {
	usage := a.hm.usage
	if usage.lastProvider != "" {
		usage.lastModelName = a.entryModel(local.ASR, usage.lastProvider, usage.lastModel)
		if suffix := " · " + usage.lastModel; strings.HasSuffix(usage.lastModelName, suffix+suffix) {
			usage.lastModelName = strings.TrimSuffix(usage.lastModelName, suffix)
		}
	} else {
		usage.lastModelName = uitext.ModelNameFor(i18n.CurrentTranslator(), usage.lastModel)
	}
	_, zoneOffset := now.Zone()
	key := homeViewCacheKey{
		settings: a.settingsRevision, gen: a.hm.gen,
		minute: now.Unix() / 60, location: now.Location().String(), zoneOffset: zoneOffset, locale: i18n.Locale(),
		lastModelName: usage.lastModelName,
		activityGen:   a.act.gen, activityPage: a.act.page,
	}
	if key != a.hm.viewKey {
		a.hm.view = buildHomeViewState(i18n.CurrentTranslator(), homeViewStateInput{
			KeepHistory: a.prefs.Privacy.KeepHistory, Features: a.prefs.Features, Now: now,
			Activity: a.act.log, ActivityRead: a.act.loaded, ActivityPage: a.act.page, Usage: usage,
		})
		a.hm.viewKey = key
	}
	return a.hm.view
}

func (a *app) openHomeGuide() { a.openOnboarding() }
func (a *app) addHomeShortcut() {
	a.navigate("/transcribe")
}
