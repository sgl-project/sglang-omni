package main

// The insights card's controller (insights_view.go): open, another
// background, Copy. Its numbers are Home's Insights, from the activity
// log.

import (
	"log"
	"math/rand/v2"
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
)

type insightsShare struct {
	open       bool
	background int
	copiedAt   time.Time
}

func (a *app) openInsightsShare() {
	a.share.insights = insightsShare{open: true, background: rand.IntN(len(shareBackgrounds))}
}

func (a *app) closeInsightsShare() { a.share.insights.open = false }

func (a *app) shuffleInsightsShare() {
	s := &a.share.insights
	if n := len(shareBackgrounds); n > 1 {
		s.background = (s.background + 1 + rand.IntN(n-1)) % n
	}
}

// insightsModel is the card's view state, as of now.
func (a *app) insightsModel(now time.Time) insightsCardViewState {
	s := a.share.insights
	index := s.background % len(shareBackgrounds)
	return insightsCardViewState{
		Background: index, Dark: shareBackgrounds[index].Dark,
		Insights: buildInsightsViewState(i18n.CurrentTranslator(), a.act.log, now),
		Copied:   !s.copiedAt.IsZero() && now.Sub(s.copiedAt) < shareCopiedFor,
	}
}

func (a *app) insightsDialog(c *uiContext) {
	if !a.share.insights.open {
		return
	}
	state := a.insightsModel(a.now())
	insightsCardDialog(c, &state, a)
}

// copyInsightsShare renders the card without its buttons and copies the
// picture.
func (a *app) copyInsightsShare() {
	state := a.insightsModel(a.now())
	if err := copyPicture(renderInsightsCard(&state)); err != nil {
		log.Printf("share: copy the insights: %v", err)
		a.toast(i18n.T("Could not copy the picture."))
		return
	}
	a.share.insights.copiedAt = a.now()
}
