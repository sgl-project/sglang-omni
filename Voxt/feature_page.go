package main

import (
	"time"

	"github.com/egoist/mygo/ui"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
)

// featureUI is a feature page's widget-local state.
type featureUI struct {
	rows   []keyRow
	paste  keyRow
	cancel keyRow
	seen   string

	asr, llm string
	svcSeen  string
	fallback string
	lang     string

	prompt        string
	promptSeen    string
	promptDirty   bool
	promptSeq     int
	rules         bool
	preview       bool
	find          string
	finder        finder
	findFocus     bool
	block         *sessionNotice
	endWord       keyRow
	transcript    transcriptRenderFunc
	customPalette bool
}

func featurePage(c *ui.Context, state *featureViewState, actions featureActions, local *featureUI, tryActions tryCardActions) {
	t := c.Theme()
	columnPageHeader(c, state.Title, nil)
	if state.RefreshRecording {
		c.After(250 * time.Millisecond)
	}
	left := func() {
		caption(c, state.Caption)
		featureShortcuts(c, state.Feature, &state.Shortcuts, actions, local)
		featureModels(c, state.Feature, &state.Models, actions, local)
	}
	right := func() {
		switch state.Feature {
		case "translate":
			translateSettings(c, &state.Prompt, actions, local)
		case "rewrite":
			rewriteSettings(c, &state.Prompt, actions)
		default:
			trEnhance(c, &state.Models.Cleanup, actions)
		}
		cardSetting(c, state.Prefix, state.Feature, &state.Prompt.Card, actions)
		featureRules(c, state.Prefix, state.Feature, &state.Prompt.Rules, actions, local)
		sectionTitle(c, i18n.T("Try it"))
		tryCard(c, &state.Try, tryActions, &tryLocalUI{transcript: local.transcript, customPalette: local.customPalette})
	}
	ui.Scroll(c.Key(state.Prefix + "-scroll")).Grow(1).MinHeight(0).Children(func() {
		pageColumn(c).Gap(px(t, sp3)).Children(func() { left(); right() })
	})
}
