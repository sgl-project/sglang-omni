package main

import (
	"time"

	"github.com/egoist/mygo"
	"github.com/egoist/mygo/transfer"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
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

func (a *app) tryCardState(options tryOpts) (tryCardViewState, tryCardActions) {
	input := newTryCardStateInput(options.prefix, options.feature, options.mine, options.intro, *options.block, a.sess, time.Now())
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
	if controller.app.copyText != nil {
		_ = controller.app.copyText(text)
	} else {
		_ = mygo.Clipboard.Write(transfer.TextData(text))
	}
	controller.app.toast(i18n.T("Copied"))
}

func (controller *tryController) clearTryNotice() {
	*controller.options.block = nil
}

func (controller *tryController) runTryNoticeAction(action string) {
	controller.clearTryNotice()
	controller.app.noticeAction(action)
}
