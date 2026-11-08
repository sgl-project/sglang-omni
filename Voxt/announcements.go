package main

// Voxt's announcements (internal/announce), as Voxt 1.x read them: when
// Voxt starts and each time it becomes the active app, at most every 10
// minutes, through the proxy the downloads use. Voxt 1.x lit a dot for
// its newest announcement and listed them in a dialog; Voxt has no such
// list (behavior.md 2.7), so the newest announcement not shown yet
// becomes a card in the sidebar (shell_notices.go), whose Read shows it
// whole. Each is shown once: the ids of the list read are kept (the
// "announcements" document), so a card dismissed, or older ones, do not
// come back. A failure shows nothing: the log says it.

import (
	"context"
	"encoding/json"
	"log"
	"slices"
	"strings"
	"time"

	"github.com/egoist/mygo/ui"
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
// its action, Read (not a deep link).
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
	// announceShown is the announcement whose Read is open.
	announceShown     core.Notification
	announceShownOpen bool
}

// announceSeen is the announcements document: the revision read last and
// the ids of its list, shown or passed over.
type announceSeen struct {
	Revision string   `json:"revision"`
	IDs      []string `json:"ids"`
}

// checkAnnouncements reads the announcements in the background, unless
// one check runs or the last began less than 10 minutes ago, and posts a
// card for the newest one not seen yet.
func (a *app) checkAnnouncements() {
	if a.announceURL == "" || a.announceBusy || !a.announceAt.IsZero() && time.Since(a.announceAt) < announceEvery {
		return
	}
	a.announceAt, a.announceBusy = time.Now(), true
	base, client, docs, notes := a.announceURL, a.networkFor(a.prefs).Client, a.lib.store.Documents(), a.notes
	a.async(func() {
		defer a.post(func() { a.announceBusy = false })
		defer client.CloseIdleConnections() // a client of its own each time, whose idle connections never time out
		ctx, cancel := context.WithTimeout(context.Background(), announceTimeout)
		defer cancel()
		var seen announceSeen
		if raw, ok, err := docs.Get(ctx, announceDoc); err == nil && ok {
			json.Unmarshal(raw, &seen) // a document that does not decode: nothing seen
		}
		revision, list, err := announce.Fetch(ctx, client, base, seen.Revision)
		if err != nil {
			log.Printf("announcements: %v", err)
			return
		}
		if list == nil && revision == seen.Revision {
			return
		}
		if len(list) > 0 && !slices.Contains(seen.IDs, list[0].ID) {
			n := list[0]
			notes.Post(core.Notification{ID: announcePrefix + n.ID, Severity: core.SeverityInfo, Title: n.Title,
				Body: strings.TrimSpace(n.Content), Link: linkAnnouncement, Time: n.PublishedAt, Version: strings.TrimSpace(n.Version)})
		}
		next := announceSeen{Revision: revision, IDs: []string{}}
		for _, n := range list {
			next.IDs = append(next.IDs, n.ID)
		}
		raw, _ := json.Marshal(next)
		if err := docs.Put(ctx, announceDoc, raw); err != nil {
			log.Printf("announcements: %v", err)
		}
	})
}

// announcementDialog shows an announcement whole, under its date and the
// version it is about, as Voxt 1.x's dialog: its card's Read.
func (a *app) announcementDialog(c *ui.Context) {
	if !a.announceShownOpen {
		return
	}
	n := a.announceShown
	sub := i18n.DateTime(n.Time)
	if n.Version != "" {
		sub += " · " + i18n.T("Version %s", n.Version)
	}
	dialogFrame(c, "announcement", &a.announceShownOpen, n.Title, sub, 520, func() {
		ui.Text(c, n.Body).Selectable().Grow(1)
	}, func() {
		ui.Spacer(c)
		if primaryButton(c.Key("announcement-done"), i18n.T("Done")).Clicked() {
			a.announceShownOpen = false
		}
	})
}
