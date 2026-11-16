package main

import (
	"github.com/egoist/mygo"
	"github.com/egoist/mygo/transfer"
	"github.com/sgl-project/sglang-omni/Voxt/internal/meeting"
)

func (a *app) copyLiveMeetingExport() {
	s := a.meetSnap
	segs := meeting.SegmentsForExport(s.Segments)
	_ = mygo.Clipboard.Write(transfer.TextData(meeting.ExportTxt(segs, nil)))
}

func (a *app) toggleMeetingTranslation(id string) {
	if a.meetMgr == nil || id == "" {
		return
	}
	target := meeting.TranslateTargetLanguage(a.prefs.Features.Meeting.TranslateTarget)
	cur := a.meetSnap.Translate
	if cur != "" {
		target = ""
	}
	_ = a.meetMgr.SetTranslate(id, target)
}
