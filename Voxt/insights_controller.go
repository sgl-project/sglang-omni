package main

// The insights cards' controller (insights_deck.go): open on the first
// card, turn to another, another color for the one shown, Copy and Save.
// Their numbers are Home's Insights, from the activity log.

import (
	"bytes"
	"image/png"
	"log"
	"math/rand/v2"
	"os"
	"path/filepath"
	"time"

	"github.com/egoist/mygo"
	"github.com/sgl-project/sglang-omni/Voxt/internal/activity"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
)

type insightsShare struct {
	open bool
	// card is the one shown, themes each card's theme, shownAt when card
	// came up and slide the side it came from (insightsDeckViewState).
	card     int
	themes   []int
	shownAt  time.Time
	slide    int
	copiedAt time.Time
	// postedTo is the site the card went to (insightsSites), at postedAt.
	postedTo string
	postedAt time.Time
}

// chooseInsightsFile asks where to save a picture named name, then calls
// then on the main thread; tests replace it.
var chooseInsightsFile = func(a *app, name string, then func(string)) {
	go func() {
		destination, err := mygo.Dialog.Save(mygo.SaveDialogOptions{Parent: a.win, Title: i18n.T("Save picture"), DefaultPath: name,
			Filters: []mygo.FileFilter{{Name: "PNG", Extensions: []string{"png"}}}})
		if err != nil || destination == "" {
			return
		}
		a.post(func() { then(destination) })
	}()
}

// insightsSites are where a card can be posted: the page with a new post
// open on each. Nothing of the user's goes in the address: the picture
// travels on the clipboard, for the user to paste, since neither site
// takes a picture from a link.
var insightsSites = map[string]string{
	insightsSiteLinkedIn: "https://www.linkedin.com/feed/?shareActive=true",
	insightsSiteX:        "https://x.com/intent/post",
}

const (
	insightsSiteLinkedIn = "linkedin"
	insightsSiteX        = "x"
)

// insightsPostFor is how long the window says the picture waits to be
// pasted, and insightsPostAfter how long it says so before the site
// opens and takes the front.
const (
	insightsPostFor   = 8 * time.Second
	insightsPostAfter = 1200 * time.Millisecond
)

// openInsightsLink opens an address in the user's browser after a
// while; tests replace it.
var openInsightsLink = func(address string, after time.Duration) {
	time.AfterFunc(after, func() { _ = mygo.Shell.OpenExternal(address) }) // the browser's own error is the user's to see
}

// openInsightsShare opens the cards on the first, each in its own theme.
func (a *app) openInsightsShare() {
	themes := make([]int, len(insightsCardKinds))
	for i := range themes {
		themes[i] = i % len(insightsThemes)
	}
	a.share.insights = insightsShare{open: true, themes: themes, shownAt: a.now()}
}

func (a *app) closeInsightsShare() { a.share.insights.open = false }

// setInsightsClosed folds Looking back to its title row, or opens it:
// the settings keep the choice.
func (a *app) setInsightsClosed(closed bool) {
	if closed != a.prefs.Appearance.LookingBackClosed {
		a.updateSettings(func(s *settings.Settings) { s.Appearance.LookingBackClosed = closed })
	}
}

// turnInsightsShare shows the card by cards after the one shown (-1: the
// one before), around the ends.
func (a *app) turnInsightsShare(by int) {
	s := &a.share.insights
	n := len(insightsCardKinds)
	s.card, s.slide, s.shownAt = ((s.card+by)%n+n)%n, by, a.now()
}

// showInsightsShare shows card (a click on its dot).
func (a *app) showInsightsShare(card int) {
	s := &a.share.insights
	if card == s.card || card < 0 || card >= len(insightsCardKinds) {
		return
	}
	slide := 1
	if card < s.card {
		slide = -1
	}
	s.card, s.slide, s.shownAt = card, slide, a.now()
}

// shuffleInsightsShare gives the card shown another theme.
func (a *app) shuffleInsightsShare() {
	s := &a.share.insights
	if n := len(insightsThemes); n > 1 && s.card < len(s.themes) {
		s.themes[s.card] = (s.themes[s.card] + 1 + rand.IntN(n-1)) % n
	}
}

// insightsFace is the card shown, as of now.
func (a *app) insightsFace(now time.Time) insightsFaceViewState {
	s := a.share.insights
	theme := 0
	if s.card < len(s.themes) {
		theme = s.themes[s.card]
	}
	return buildInsightsFace(i18n.CurrentTranslator(), a.act.log, now, insightsCardKinds[s.card%len(insightsCardKinds)], theme)
}

// insightsModel is the cards' view state, as of now; still shows the
// card settled (Reduce motion).
func (a *app) insightsModel(now time.Time, still bool) insightsDeckViewState {
	s := a.share.insights
	state := insightsDeckViewState{
		Card: s.card, Count: len(insightsCardKinds), Face: a.insightsFace(now), Elapsed: now.Sub(s.shownAt), Slide: s.slide,
		Copied: !s.copiedAt.IsZero() && now.Sub(s.copiedAt) < shareCopiedFor,
	}
	if s.postedTo != "" && now.Sub(s.postedAt) < insightsPostFor {
		state.Posted = s.postedTo
	}
	if still || state.Elapsed > insightsSettled {
		state.Elapsed = insightsSettled
	}
	return state
}

func (a *app) insightsDialog(c *uiContext) {
	if !a.share.insights.open {
		return
	}
	state := a.insightsModel(a.now(), a.prefs.Appearance.ReduceMotion || c.Preferences().ReduceMotion)
	insightsDeckDialog(c, &state, a)
}

// copyInsightsShare renders the card shown, settled, and copies the
// picture.
func (a *app) copyInsightsShare() {
	face := a.insightsFace(a.now())
	if err := copyPicture(renderInsightsCard(&face)); err != nil {
		log.Printf("share: copy the insights: %v", err)
		a.toast(i18n.T("Could not copy the picture."))
		return
	}
	a.share.insights.copiedAt = a.now()
}

// saveInsightsShare asks for a file and saves the card shown there as a
// PNG.
func (a *app) saveInsightsShare() {
	now := a.now()
	face := a.insightsFace(now)
	name := "Voxt-" + face.Kind + "-" + activity.DayKey(now) + ".png"
	chooseInsightsFile(a, name, func(destination string) {
		picture := renderInsightsCard(&face)
		a.async(func() {
			var buf bytes.Buffer
			err := png.Encode(&buf, picture)
			if err == nil {
				err = os.WriteFile(destination, buf.Bytes(), 0o644)
			}
			a.post(func() {
				if err != nil {
					log.Printf("share: save the insights: %v", err)
					a.toast(i18n.T("Could not save the picture."))
					return
				}
				a.toast(i18n.T("Saved to %s", filepath.Base(destination)))
			})
		})
	})
}

// postInsightsShare copies the card shown and, once the window has said
// to paste it, opens site's new post.
func (a *app) postInsightsShare(site string) {
	address, ok := insightsSites[site]
	if !ok {
		return
	}
	face := a.insightsFace(a.now())
	if err := copyPicture(renderInsightsCard(&face)); err != nil {
		log.Printf("share: copy the insights for %s: %v", site, err)
		a.toast(i18n.T("Could not copy the picture."))
		return
	}
	s := &a.share.insights
	s.postedTo, s.postedAt = site, a.now()
	openInsightsLink(address, insightsPostAfter)
}
