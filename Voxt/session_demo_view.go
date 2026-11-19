package main

import (
	"github.com/egoist/mygo"
	"github.com/egoist/mygo/ui"
	"github.com/sgl-project/sglang-omni/Voxt/internal/overlay/layout"
)

type sessionDemoPlacement struct {
	width, height int
	overlay       mygo.Rectangle
	menu          mygo.Rectangle
	menuState     *sessionMenuViewState
}

func (host *meetingDemoHost) sessionDemoPlacement() sessionDemoPlacement {
	model := host.rebuildSessionOverlayLayout()
	overlay := mygo.Rectangle{Width: model.Width + 2*layout.Pad, Height: model.Height + 2*layout.Pad}
	placement := sessionDemoPlacement{width: overlay.Width, height: overlay.Height, overlay: overlay}
	if host.sessionState.Menu == nil {
		return placement
	}
	menu := sessionLanguageMenuFrame(overlay, model.Top, len(host.sessionState.Menu.Options))
	switch host.sessionMenuKind() {
	case sessionMenuPillThinking:
		menu = pillMenuFrame(overlay, model.Top, len(host.sessionState.Menu.Options))
	case sessionMenuCardThinking:
		menu = thinkMenuFrame(menu, model.Top, len(host.sessionState.Menu.Options))
	default:
		if host.sessionState.Card != nil && host.sessionState.Card.ShowThinking {
			menu.X -= layout.CardBtn + layout.CardBtnGap
		}
	}
	left, top := min(0, menu.X), min(0, menu.Y)
	right, bottom := max(overlay.Width, menu.X+menu.Width), max(overlay.Height, menu.Y+menu.Height)
	placement.width, placement.height = right-left, bottom-top
	placement.overlay.X, placement.overlay.Y = -left, -top
	menu.X, menu.Y = menu.X-left, menu.Y-top
	placement.menu, placement.menuState = menu, host.sessionState.Menu
	return placement
}

func sessionDemoOverlayView(context *ui.Context, host *meetingDemoHost) {
	if host.sessionState.Mode == ovHidden {
		return
	}
	placement := host.sessionDemoPlacement()
	ui.Box(context).Absolute().Left(float32(placement.overlay.X)).Top(float32(placement.overlay.Y)).
		Width(float32(placement.overlay.Width)).Height(float32(placement.overlay.Height)).Children(func() {
		sessionOverlayView(context, &host.sessionState, host.sessionLayout, host, host.sessionLocal(),
			sessionOverlayAnimation{}, host.applySessionOverlayMeasurements, nil)
	})
	if placement.menuState != nil {
		ui.Box(context).Absolute().Left(float32(placement.menu.X)).Top(float32(placement.menu.Y)).
			Width(float32(placement.menu.Width)).Height(float32(placement.menu.Height)).Children(func() {
			sessionLanguageMenuView(context, *placement.menuState, "", host, &host.sessionUI)
		})
	}
}
