package main

// The session overlay's view and its UI-local state. Native window lifetime,
// session projection, measurement ownership, and actions live in session files.

import (
	"time"

	"github.com/egoist/mygo/ui"
	"github.com/egoist/mygo/ui/markdown"
	"github.com/sgl-project/sglang-omni/Voxt/internal/overlay/layout"
)

// overlayMode is what the overlay shows.
type overlayMode string

const (
	ovHidden    overlayMode = ""
	ovRecording overlayMode = "recording"
	ovWorking   overlayMode = "working"
	ovNotice    overlayMode = "notice"
	ovAnswer    overlayMode = "answer"
)

func (mode overlayMode) String() string {
	if mode == ovHidden {
		return "hidden"
	}
	return string(mode)
}

// sessionOverlayLocalUI owns widget state that is not part of the session DTO.
type sessionOverlayLocalUI struct {
	lines         linesScroll
	cardScroll    ui.ScrollState
	cardScrollKey string
	bubbleCopied  string
	bubbleSeq     int
	langScroll    ui.ScrollState
	// mdCard and mdBubble are how the card's and its bubbles' Markdown
	// renders (overlay_card_markdown.go).
	mdCard, mdBubble markdown.Options
}

type sessionOverlayLocal struct {
	Overlay *sessionOverlayLocalUI
	Card    *answerCardUI
}

type sessionOverlayLayout struct {
	Width, Height int
	Radius, Alpha float32
	ShowTime      bool
	LiveLines     bool
	Top           bool
	Reveal        bool
	Measured      map[ovSized]ovMeasured
	Exact         bool
	LiveID        string
	LiveWidth     int
	StreamHeight  float32
}

type sessionOverlayAnimation struct {
	Motion bool
	Phase  float64
}

type sessionOverlayMeasurement struct {
	Key           ovSized
	Width, Height float32
}

func sessionOverlayContent(
	state func() *sessionOverlayViewState,
	layoutState func() sessionOverlayLayout,
	actions sessionOverlayActions,
	local sessionOverlayLocal,
	animation func() sessionOverlayAnimation,
	onMeasured func([]sessionOverlayMeasurement),
	onReveal func(),
) func(*ui.Context) {
	return func(context *ui.Context) {
		sessionOverlayView(context, state(), layoutState(), actions, local, animation(), onMeasured, onReveal)
	}
}

func sessionOverlayView(
	context *ui.Context,
	state *sessionOverlayViewState,
	layoutState sessionOverlayLayout,
	actions sessionOverlayActions,
	local sessionOverlayLocal,
	animation sessionOverlayAnimation,
	onMeasured func([]sessionOverlayMeasurement),
	onReveal func(),
) {
	context.SetTheme(overlayTheme())
	defer context.Root().Background(ui.Transparent)
	sessionOverlayMeasure(context, state, layoutState, onMeasured)
	if layoutState.Reveal && onReveal != nil {
		onReveal()
	}
	if !animation.Motion {
		context.DisableAnimations()
	}
	switch {
	case animation.Motion && (state.Mode == ovRecording || state.Mode == ovWorking || state.Mode == ovAnswer && state.Card != nil && (state.Card.Recording || state.Card.Busy && !state.Card.Conversation)):
		context.After(layout.Tick)
	case state.Mode == ovRecording && layoutState.ShowTime:
		context.After(250 * time.Millisecond)
	}
	look := pillLook{time: layoutState.ShowTime, lines: layoutState.LiveLines, scroll: &local.Overlay.lines,
		motion: animation.Motion, phase: animation.Phase, w: layoutState.Width}
	ui.Column(context).Fill().Padding(layout.Pad).Children(func() {
		switch state.Mode {
		case ovAnswer:
			faceStyle(ui.Column(context.Key("ov-face")), layoutState.Radius, layoutState.Alpha).Children(func() {
				overlayCard(context, state, layoutState, actions, local, animation, onMeasured)
			})
		case ovNotice:
			overlayNotice(context, state.Notice, layoutState.Radius, layoutState.Alpha, actions)
		default:
			faceStyle(ui.Column(context.Key("ov-face")), layoutState.Radius, layoutState.Alpha).DragWindow().Children(func() {
				overlayPill(context, state, look, actions)
			})
		}
	})
}

func overlayTheme() *ui.Theme {
	theme := *ui.DarkTheme()
	styleTheme(&theme)
	return &theme
}
