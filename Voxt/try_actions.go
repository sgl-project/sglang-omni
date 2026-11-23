package main

type tryCardActions interface {
	startTry()
	stopTry()
	cancelTry()
	copyTryResult(text string)
	clearTryNotice()
	runTryNoticeAction(action string)
	// transcriptResult shows a result with its Markdown, code and math
	// (#4105, transcript_preview_controller.go) and says whether it did;
	// the view draws the plain text when not.
	transcriptResult(c *uiContext, key, source string, transcript, streaming bool) bool
}
