package main

// Voxt's announcements (internal/announce), as Voxt 1.x read them: when
// Voxt starts and each time it becomes the active app, at most every 10
// minutes, through the proxy the downloads use. The running version's
// announcement becomes one short card after an update, whose What's new
// shows it whole. The "announcements" document remembers the version shown
// and keeps the full notes for Settings → About → Updates. A failure shows
// no new card: the log says it.

import (
	"context"
	"encoding/json"
	"log"
	"slices"
	"strings"
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/announce"
	"github.com/sgl-project/sglang-omni/Voxt/internal/core"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
)

// announceEvery is the least time between two checks (Voxt 1.x's).
const announceEvery = 10 * time.Minute

// announceTimeout bounds a check (announce.Timeout); tests shorten it.
var announceTimeout = announce.Timeout

// announceDoc is the document of the announcements already seen.
const announceDoc = "announcements"

// announcePrefix starts an announcement's card's id; linkAnnouncement is
// its action, What's new (not a deep link).
const (
	announcePrefix   = "announcement/"
	linkAnnouncement = "action:announcement"
)

// announcementsState is the announcements' part of the app (main thread).
type announcementsState struct {
	// announceURL is where they are read; "" never (tests, unless they
	// set their server's).
	announceURL string
	// announceAt is the last check's start, announceBusy one runs.
	announceAt   time.Time
	announceBusy bool
	// announceShown is the announcement whose full notes are open;
	// announceCurrent is the running version's notes for Settings.
	announceShown     core.Notification
	announceShownOpen bool
	announceCurrent   core.Notification

	// announceView is the last built dialog state, announceViewKey its
	// inputs (announcementModel).
	announceView    *announcementViewState
	announceViewKey announcementCacheKey
}

// announcementCacheKey is every input of the dialog's view state: when
// it matches the last build's, announcementModel reuses the state.
type announcementCacheKey struct {
	open       bool
	id         string
	year       int
	location   string
	zoneOffset int
	locale     string
}

// announceSeen is the announcements document. IDs is the old document's
// list, read once so an announcement already shown does not come back.
type announceSeen struct {
	Revision     string             `json:"revision"`
	IDs          []string           `json:"ids,omitempty"`
	AppVersion   string             `json:"appVersion,omitempty"`
	ShownVersion string             `json:"shownVersion,omitempty"`
	Current      *core.Notification `json:"current,omitempty"`
}

// checkAnnouncements reads the announcements in the background, unless
// one check runs or the last began less than 10 minutes ago. Only the
// running version's announcement gets a card, once; its full notes stay
// available in Settings. A version change reads the list again even when
// its revision is unchanged.
func (a *app) checkAnnouncements() {
	if a.announceURL == "" || a.announceBusy || !a.announceAt.IsZero() && time.Since(a.announceAt) < announceEvery {
		return
	}
	a.announceAt, a.announceBusy = time.Now(), true
	version := a.version
	base, client, docs, notes := a.announceURL, a.networkFor(a.prefs).Client, a.lib.store.Documents(), a.notes
	stored := slices.Clone(a.notices)
	chinese := i18n.Locale() == "zh-Hans"
	a.async(func() {
		defer a.post(func() { a.announceBusy = false })
		defer client.CloseIdleConnections() // a client of its own each time, whose idle connections never time out
		ctx, cancel := context.WithTimeout(context.Background(), announceTimeout)
		defer cancel()
		var seen announceSeen
		if raw, ok, err := docs.Get(ctx, announceDoc); err == nil && ok {
			json.Unmarshal(raw, &seen) // a document that does not decode: nothing seen
		}
		current := core.Notification{}
		if seen.Current != nil && seen.Current.Version == version {
			current = *seen.Current
		} else {
			for _, n := range stored {
				if n.Link == linkAnnouncement && n.Version == version {
					current = n
					break
				}
			}
		}
		postCurrent := func(n core.Notification) { a.post(func() { a.announceCurrent = n }) }
		postCurrent(current)
		requestRevision := seen.Revision
		if seen.AppVersion != version {
			requestRevision = ""
		}
		revision, list, err := announce.Fetch(ctx, client, base, requestRevision)
		if err != nil {
			log.Printf("announcements: %v", err)
			return
		}
		if list == nil && revision == seen.Revision && seen.AppVersion == version {
			return
		}
		next := announceSeen{Revision: revision, AppVersion: version, ShownVersion: seen.ShownVersion, Current: seen.Current}
		var show *core.Notification
		for _, n := range announce.For(list, chinese) {
			if strings.TrimSpace(n.Version) != version {
				continue
			}
			current = core.Notification{ID: announcePrefix + n.ID, Severity: core.SeverityInfo, Title: n.Title,
				Body: strings.TrimSpace(n.Content), Link: linkAnnouncement, Time: n.PublishedAt, Version: version}
			next.Current = &current
			next.ShownVersion = version
			if seen.ShownVersion != version && !slices.Contains(seen.IDs, n.ID) {
				show = &current
			}
			break
		}
		raw, _ := json.Marshal(next)
		if err := docs.Put(ctx, announceDoc, raw); err != nil {
			log.Printf("announcements: %v", err)
		}
		if next.Current == nil || next.Current.Version != version {
			current = core.Notification{}
		}
		postCurrent(current)
		if show != nil {
			notes.Post(*show)
		}
	})
}

// announcementModel is the open announcement's view state, nil when its
// dialog is closed; rebuilt when it opens, the year or the language
// changes.
func (a *app) announcementModel(now time.Time) *announcementViewState {
	_, zoneOffset := now.Zone()
	key := announcementCacheKey{open: a.announceShownOpen, id: a.announceShown.ID, year: now.Year(), location: now.Location().String(), zoneOffset: zoneOffset, locale: i18n.Locale()}
	if key != a.announceViewKey {
		a.announceView = nil
		if a.announceShownOpen {
			a.announceView = buildAnnouncementViewState(i18n.CurrentTranslator(), a.announceShown, now)
		}
		a.announceViewKey = key
	}
	return a.announceView
}
