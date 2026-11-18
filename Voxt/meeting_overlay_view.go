package main

import (
	"math"

	"github.com/egoist/mygo"
	"github.com/sgl-project/sglang-omni/Voxt/internal/overlay/layout"
)

const (
	meetingOvProblemH   = 42
	meetingOvCollapsedW = 400 + 2*layout.Pad
	meetingOvCollapsedH = layout.PillH + 2*layout.Pad
	meetingOvMaxFaceH   = 340
	meetingOvWhoH       = 15
	meetingOvRowGap     = 8
	meetingOvMenuRowH   = 40
)

var meetingOvWidths = [3]int{520, 600, 760}

const meetingOvChrome = layout.CardPadTop + layout.CardHeadH + layout.CardRuleTop + 1 + layout.CardRuleGap + layout.CardPadY

type meetingOverlayLayout struct {
	Width, Height int
	Collapsed     bool
}

type meetingOverlayModel struct {
	Layout        meetingOverlayLayout
	ProblemKey    string
	ShowBar       bool
	ShowExpanded  bool
	ShowAsking    bool
	ShowFinishing bool
}

// meetingOverlayLayoutModel keeps all dimensions on the UI side.
func meetingOverlayLayoutModel(state meetingOverlayViewState, menu meetingMenu) meetingOverlayModel {
	asking, finishing := state.ShowAsking, state.ShowFinishing
	pill := state.Collapsed || asking || finishing
	model := meetingOverlayModel{
		Layout:        meetingOverlayLayout{Width: meetingOvCollapsedW, Height: meetingOvCollapsedH, Collapsed: state.Collapsed},
		ProblemKey:    state.ProblemKey,
		ShowBar:       state.ShowBar,
		ShowExpanded:  state.ShowExpanded,
		ShowAsking:    asking,
		ShowFinishing: finishing,
	}
	problem := state.ProblemText != "" && !asking && !finishing
	if !pill {
		width := meetingOvWidths[min(max(state.CardSize, 0), len(meetingOvWidths)-1)]
		body := meetingOverlayBodyHeight(state.Rows, width-2*layout.CardPadX-layout.CardInset)
		if menu != meetingMenuNone {
			body = max(body, menuHeight(menu, state))
		}
		height := meetingOvChrome + body
		if problem {
			height += meetingOvProblemH
		}
		model.Layout.Width, model.Layout.Height = width+2*layout.Pad, min(height, meetingOvMaxFaceH)+2*layout.Pad
	} else if problem {
		model.Layout.Height += meetingOvProblemH
	}
	return model
}

func meetingOverlayBodyHeight(rows []meetingOverlayTranscriptRow, textWidth int) int {
	lineHeight := int(math.Ceil(layout.CardTextSize * 1.3))
	if len(rows) == 0 {
		return lineHeight
	}
	height := 0
	for index, row := range rows {
		if index > 0 {
			height += meetingOvRowGap
		}
		if row.First {
			height += meetingOvWhoH + 2
		}
		_, textHeight := guessText(ovSized{row.Text, layout.CardTextSize, layout.CardTextWeight, float32(textWidth)})
		height += int(math.Ceil(float64(textHeight)))
	}
	return height
}

// meetingMenu is the local menu shown in place of the transcript.
type meetingMenu int

const (
	meetingMenuNone meetingMenu = iota
	meetingMenuMode
	meetingMenuMore
)

func menuHeight(menu meetingMenu, state meetingOverlayViewState) int {
	switch menu {
	case meetingMenuMode:
		return 3*meetingOvMenuRowH + 2*4
	case meetingMenuMore:
		count := len(state.MoreItems)
		return count*30 + (count-1)*4
	default:
		return 0
	}
}

func meetingOverlayFrame(at, area mygo.Rectangle, width, height int) mygo.Rectangle {
	if at.Width == 0 {
		return mygo.Rectangle{X: area.X + (area.Width-width)/2, Y: area.Y + area.Height - height - 28, Width: width, Height: height}
	}
	x := max(min(at.X, area.X+area.Width-width+layout.Pad), area.X-layout.Pad)
	y := at.Y
	if at.Y+at.Height/2 > area.Y+area.Height/2 {
		y = at.Y + at.Height - height
	}
	y = max(min(y, area.Y+area.Height-height+layout.Pad), area.Y-layout.Pad)
	return mygo.Rectangle{X: x, Y: y, Width: width, Height: height}
}

func meetingOverlayFace(model meetingOverlayModel) (overlayMode, int) {
	mode := ovRecording
	if model.ShowExpanded {
		mode = ovAnswer
	}
	return mode, model.Layout.Height - 2*layout.Pad
}
