package main

type tryCardActions interface {
	startTry()
	stopTry()
	cancelTry()
	copyTryResult(text string)
	clearTryNotice()
	runTryNoticeAction(action string)
}
