package main

import (
	"math"
	"strings"
	"unicode"

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
		_, textHeight := meetingOverlayGuessText(ovSized{row.Text, layout.CardTextSize, layout.CardTextWeight, float32(textWidth)})
		height += int(math.Ceil(float64(textHeight)))
	}
	return height
}

func meetingOverlayGuessText(text ovSized) (float32, float32) {
	width := float32(meetingOverlayTextWidth(text.text, float64(text.size)))
	lineHeight := float32(math.Ceil(float64(text.size) * 1.3))
	if text.wrap <= 0 {
		return width, lineHeight
	}
	lines := 0
	for _, paragraph := range strings.Split(text.text, "\n") {
		paragraphWidth := float32(meetingOverlayTextWidth(paragraph, float64(text.size)))
		lines += max(1, int(math.Ceil(float64(paragraphWidth/text.wrap))))
	}
	return min(width, text.wrap), float32(lines) * lineHeight
}

func meetingOverlayTextWidth(text string, size float64) int {
	width := 0.0
	for _, r := range text {
		switch {
		case meetingOverlayWideRune(r):
			width += size
		case r == ' ' || r == 'i' || r == 'l' || r == '.' || r == ',' || r == ':':
			width += 0.32 * size
		default:
			width += 0.58 * size
		}
	}
	return int(math.Ceil(width))
}

func meetingOverlayWideRune(r rune) bool {
	return unicode.Is(unicode.Han, r) || unicode.In(r, unicode.Hiragana, unicode.Katakana, unicode.Hangul) ||
		r >= 0xFF00 && r <= 0xFFEF || r >= 0x3000 && r <= 0x303F
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

func meetingOverlayFace(model meetingOverlayModel) (overlayMode, int) {
	mode := ovRecording
	if model.ShowExpanded {
		mode = ovAnswer
	}
	return mode, model.Layout.Height - 2*layout.Pad
}
