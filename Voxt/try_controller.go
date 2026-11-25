package main

import (
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/uitext"
)

type tryOpts struct {
	prefix  string
	feature string
	mine    bool
	intro   string
	block   **sessionNotice
	start   func() *sessionNotice
}

type tryController struct {
	app     *app
	options tryOpts
}

func (a *app) trySessionOf(feature string) bool {
	snapshot := a.sess
	return snapshot.ID != "" && snapshot.Try && snapshot.Kind == featureKind(feature)
}

func (a *app) tryIntro(feature string) string {
	return buildTryIntro(i18n.CurrentTranslator(), feature, a.featureKeys(feature))
}

func (a *app) featureTryOptions(feature string) tryOpts {
	local := a.fui(feature)
	return tryOpts{
		prefix: uitext.FeaturePrefix(feature), feature: feature, mine: a.trySessionOf(feature),
		intro: a.tryIntro(feature), block: &local.block,
		start: func() *sessionNotice { _, notice := a.startTry(featureKind(feature)); return notice },
	}
}

func (a *app) tryCardState(options tryOpts) (tryCardViewState, tryCardActions) {
	return a.tryCardStateAt(options, time.Now())
}

func (a *app) tryCardStateAt(options tryOpts, now time.Time) (tryCardViewState, tryCardActions) {
	input := newTryCardStateInput(options.prefix, options.feature, options.mine, options.intro, *options.block, a.sess, now)
	input.Palette = a.prefs.Appearance.Palette
	state := buildTryCardViewState(i18n.CurrentTranslator(), input)
	return state, &tryController{app: a, options: options}
}

func (controller *tryController) startTry() {
	*controller.options.block = controller.options.start()
}

func (controller *tryController) stopTry() {
	controller.app.stopSession()
}

func (controller *tryController) cancelTry() {
	controller.app.cancelSession()
}

func (controller *tryController) copyTryResult(text string) {
	controller.app.copyFeatureText(text)
}

func (controller *tryController) clearTryNotice() {
	*controller.options.block = nil
}

func (controller *tryController) runTryNoticeAction(action string) {
	controller.clearTryNotice()
	controller.app.noticeAction(action)
}

func (controller *tryController) transcriptResult(c *uiContext, key, source string, transcript, streaming bool) bool {
	return controller.app.transcriptResult(c, key, source, transcript, streaming)
}
