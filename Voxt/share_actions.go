package main

// shareActions are the share card's and its hat studio's intents
// (share_controller.go, share_studio_controller.go).
type shareActions interface {
	closeShare()
	shuffleShare()
	copyShare()
	// startShareTry records a sentence for the card (a click on its
	// prompt or its mic); stopShareTry ends the recording.
	startShareTry()
	stopShareTry()
	// shareBitmap is a drawn hat ("hat:<name>"), nil until it is read
	// and when it cannot be.
	shareBitmap(ref string) *sharePicture

	openStudio()
	studioPoint(x, y float32, begin bool)
	studioEnd()
	studioColor(color string)
	studioBrush(brush int)
	studioEraser()
	studioUndo()
	studioClear()
	studioPeek()
	studioCloset(open bool)
	studioLoad(item string)
	studioWear()
	studioCancel()
}

// insightsActions are Home's Insights intents (insights_controller.go).
type insightsActions interface {
	openInsightsShare()
	closeInsightsShare()
	shuffleInsightsShare()
	copyInsightsShare()
}
