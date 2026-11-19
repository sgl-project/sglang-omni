package main

// Overlay text measurement keys and deferred view measurement. The session
// owner stores posted measurements and computes the native face size.

import (
	"math"
	"slices"

	"github.com/egoist/mygo"
	"github.com/egoist/mygo/ui"
	"github.com/sgl-project/sglang-omni/Voxt/internal/overlay/layout"
)

type ovSized struct {
	text   string
	size   float32
	weight int
	wrap   float32
}

func (key ovSized) span() ui.Span {
	return ui.Span{Text: key.text, Size: key.size, Weight: key.weight}
}

type ovMeasured struct{ w, h float32 }

func liveTextKey(text string) ovSized {
	return ovSized{text, layout.LiveSize, layout.LiveWeight, 0}
}

func liveLinesKey(text string) ovSized {
	return ovSized{text, layout.LiveSize, layout.LiveWeight, layout.LinesTextW}
}

func sceneText(name string) ovSized { return ovSized{name, layout.SceneSize, 0, 0} }

func kindTitleText(title string) ovSized { return ovSized{title, 12, 500, 0} }

func capsuleText(text string) ovSized { return ovSized{text, 11, 600, 0} }

func recordingPillRow(pill *sessionPillViewState, label, showTime bool, scene float32, measure func(ovSized) (float32, bool)) (int, bool) {
	modeLabel := float32(0)
	exact := true
	if label && !pill.Starting {
		width, ok := measure(kindTitleText(pill.Title))
		modeLabel = width + float32(layout.Gap)
		exact = ok
	}
	return layout.RowWidth(showTime, scene) + int(math.Ceil(float64(modeLabel))), exact
}

func noticeText(text string) ovSized { return ovSized{text, layout.NoticeSize, 0, 0} }

type sessionOverlayLayoutStyle struct {
	Position  string  `json:"position"`
	Opacity   float64 `json:"opacity"`
	Radius    float64 `json:"radius"`
	Margin    float64 `json:"margin"`
	ShowTime  bool    `json:"show_time"`
	LiveLines bool    `json:"live_lines"`
}

type sessionOverlayLayoutMemory struct {
	LiveID       string
	LiveWidth    int
	StreamHeight float32
}

type sessionOverlayLayoutInput struct {
	Measure        func(ovSized) (ovMeasured, bool)
	WorkArea       mygo.Rectangle
	Overlay        sessionOverlayLayoutStyle
	Memory         sessionOverlayLayoutMemory
	Measured       map[ovSized]ovMeasured
	PreviewHeights map[string]int
	Reveal         bool
}

const streamGrowStep = 4 * 19

// streamHeadroom is the room a card keeps under a streaming answer's
// text: a line of it (14 pt Markdown at 1.5 leading) and a little. The
// view measures a new text in the frame that first draws it, and the card
// grows to it a turn of the main loop later, so the frame between draws
// the next line in this room instead of past the body (#3836 feedback).
const streamHeadroom = 26

// sessionOverlayLayoutFor computes the native face model without owning an app or a window.
func sessionOverlayLayoutFor(state *sessionOverlayViewState, input sessionOverlayLayoutInput) sessionOverlayLayout {
	model := sessionOverlayLayout{
		Radius: float32(input.Overlay.Radius), Alpha: float32(min(max(input.Overlay.Opacity, 0.5), 1)),
		ShowTime: input.Overlay.ShowTime, LiveLines: input.Overlay.LiveLines, Top: input.Overlay.Position == "top",
		Reveal: input.Reveal, Measured: input.Measured, Exact: true,
		LiveID: input.Memory.LiveID, LiveWidth: input.Memory.LiveWidth, StreamHeight: input.Memory.StreamHeight,
	}
	measureWidth := func(key ovSized) (float32, bool) {
		measured, exact := input.Measure(key)
		return measured.w, exact
	}
	measureHeight := func(key ovSized) (float32, bool) {
		measured, exact := input.Measure(key)
		return measured.h, exact
	}
	switch state.Mode {
	case ovAnswer:
		card := state.Card
		if card == nil {
			model.Width, model.Height = layout.CardSizes[0].W, layout.CardChrome
			break
		}
		bodyMax := sessionOverlayCardBodyMax(card, input.WorkArea.Height, input.Overlay.Margin)
		body, streamHeight, exact := sessionOverlayCardBodyHeight(state.SessionID, card, measureHeight, input.PreviewHeights, bodyMax, input.Memory.StreamHeight)
		model.Width, model.Height, model.Exact, model.StreamHeight = layout.CardSizes[card.SizeStep].W, layout.CardChrome+body, exact, streamHeight
		model.CardScrolls = body >= bodyMax
		if card.Message != "" {
			model.Height += layout.CardMsgGap + layout.CardMsgH
		}
	case ovNotice:
		model.Width, model.Height = layout.NoticeW, layout.PillH
		if notice := state.Notice; notice != nil {
			textWidth, exact := measureWidth(noticeText(notice.Text))
			model.Width, model.Exact = layout.NoticeWidth(textWidth, notice.Action != actionNone), exact
		}
	case ovRecording, ovWorking:
		pill := &state.Pill
		scene := float32(0)
		if pill.SceneText != "" {
			scene, model.Exact = measureWidth(sceneText(pill.SceneText))
		}
		row, rowExact := recordingPillRow(pill, state.Mode == ovRecording, state.Mode == ovRecording && input.Overlay.ShowTime, scene, measureWidth)
		model.Exact = model.Exact && rowExact
		if pill.CanSkip {
			width, exact := measureWidth(capsuleText(pill.SkipLabel))
			row += layout.Gap + 2*10 + int(math.Ceil(float64(width)))
			model.Exact = model.Exact && exact
		}
		if pill.CanThink {
			width, exact := measureWidth(capsuleText(pill.ThinkingLabel))
			row += layout.Gap + 2*10 + int(math.Ceil(float64(width)))
			model.Exact = model.Exact && exact
		}
		text, hint := pill.LiveText, pill.LiveHint
		if input.Overlay.LiveLines {
			model.Width = layout.LinesW
			if text == "" || hint {
				model.Height, model.Exact = layout.LinesHeight(0), true
				break
			}
			textHeight, exact := measureHeight(liveLinesKey(text))
			model.Height, model.Exact = layout.LinesHeight(textHeight), exact
			break
		}
		model.Height = layout.PillH
		if text == "" {
			model.Width = row + 2*layout.PillPadX
			break
		}
		model.Height = layout.LiveH
		textWidth, exact := measureWidth(liveTextKey(text))
		model.Width = layout.LiveWidth(row, textWidth)
		if model.LiveID == state.SessionID {
			model.Width = max(model.Width, model.LiveWidth)
		}
		model.Exact = model.Exact && exact
	default:
		model.Width, model.Height = layout.PillW, layout.PillH
	}
	if model.Height == layout.LiveH && state.Mode != ovAnswer {
		if model.LiveID != state.SessionID {
			model.LiveID, model.LiveWidth = state.SessionID, 0
		}
		if model.Exact {
			model.LiveWidth = max(model.LiveWidth, model.Width)
		}
	} else if model.LiveID != state.SessionID {
		model.LiveID, model.LiveWidth = "", 0
	}
	if state.Mode != ovAnswer {
		model.Radius = min(model.Radius, float32(model.Height)/2)
	}
	return model
}

func sessionOverlayBlockHeight(card *sessionCardViewState, block sessionCardBlock, measureHeight func(ovSized) (float32, bool), previewHeight int) (float32, bool) {
	text, title, quote := sessionCardBlockText(card, block)
	height, exact := measureHeight(text)
	if previewHeight > 0 {
		height, exact = float32(previewHeight), true
	}
	if block.Kind == sessionCardProblem {
		height = max(height, layout.ProblemIcon)
	}
	if block.Kind == sessionCardBubble {
		height += 2 * layout.BubblePadY
		if title.text != "" {
			titleHeight, ok := measureHeight(title)
			height, exact = height+titleHeight+layout.BubbleTitleGap, exact && ok
		}
		if quote.text != "" {
			quoteHeight, ok := measureHeight(quote)
			height, exact = height+quoteGap+quoteHeight, exact && ok
			if block.Cut {
				height += quoteGap + quoteToggleH
			}
		}
	}
	return height, exact
}

func sessionOverlayCardBodyMax(card *sessionCardViewState, workAreaHeight int, margin float64) int {
	room := workAreaHeight - int(math.Round(margin)) - layout.Pad - layout.CardChrome
	if card != nil && card.Message != "" {
		room -= layout.CardMsgGap + layout.CardMsgH
	}
	return max(room, layout.CardBodyMax)
}

func sessionOverlayCardBodyHeight(sessionID string, card *sessionCardViewState, measureHeight func(ovSized) (float32, bool), previewHeights map[string]int, bodyMax int, previousStreamHeight float32) (int, float32, bool) {
	var height float32
	exact := true
	streamHeight := previousStreamHeight
	for index := 0; index < sessionCardBlockCount(card); index++ {
		block := sessionCardBlockAt(card, index)
		blockIndex := index
		if !card.Conversation {
			blockIndex = -1
		}
		previewHeight := previewHeights[sessionCardPreviewKey(sessionID, card, blockIndex)]
		blockHeight, ok := sessionOverlayBlockHeight(card, block, measureHeight, previewHeight)
		height += sessionCardBlockGap(block, index) + blockHeight
		exact = exact && ok
	}
	if card.Writing || card.Conversation && card.Busy && !card.Recording {
		height = float32(math.Ceil(float64((height+streamHeadroom)/streamGrowStep))) * streamGrowStep
		if exact {
			streamHeight = height
		} else if streamHeight > 0 {
			height = streamHeight
		}
	}
	return min(int(math.Ceil(float64(height))), bodyMax), streamHeight, exact
}

func sessionOverlayTexts(state *sessionOverlayViewState, layoutState sessionOverlayLayout) []ovSized {
	var texts []ovSized
	switch state.Mode {
	case ovNotice:
		if state.Notice != nil {
			texts = append(texts, noticeText(state.Notice.Text))
		}
	case ovRecording, ovWorking:
		pill := &state.Pill
		if pill.SceneText != "" {
			texts = append(texts, sceneText(pill.SceneText))
		}
		if state.Mode == ovRecording && !pill.Starting {
			texts = append(texts, kindTitleText(pill.Title))
		}
		if pill.CanSkip {
			texts = append(texts, capsuleText(pill.SkipLabel))
		}
		if pill.CanThink {
			texts = append(texts, capsuleText(pill.ThinkingLabel))
		}
		if pill.LiveText != "" && layoutState.LiveLines && !pill.LiveHint {
			texts = append(texts, liveLinesKey(pill.LiveText))
		} else if pill.LiveText != "" {
			texts = append(texts, liveTextKey(pill.LiveText))
		}
	case ovAnswer:
		for index := 0; index < sessionCardBlockCount(state.Card); index++ {
			text, title, quote := sessionCardBlockText(state.Card, sessionCardBlockAt(state.Card, index))
			texts = append(texts, text)
			if title.text != "" {
				texts = append(texts, title)
			}
			if quote.text != "" {
				texts = append(texts, quote)
			}
		}
	}
	return texts
}

func sessionOverlayMeasure(context *ui.Context, state *sessionOverlayViewState, layoutState sessionOverlayLayout, onMeasured func([]sessionOverlayMeasurement)) {
	var measurements []sessionOverlayMeasurement
	var keys []ovSized
	for _, key := range sessionOverlayTexts(state, layoutState) {
		if _, ok := layoutState.Measured[key]; ok || slices.Contains(keys, key) {
			continue
		}
		width, height := context.MeasureText(key.wrap, key.span())
		keys = append(keys, key)
		measurements = append(measurements, sessionOverlayMeasurement{Key: key, Width: width, Height: height})
	}
	if len(measurements) > 0 && onMeasured != nil {
		onMeasured(measurements)
	}
}
