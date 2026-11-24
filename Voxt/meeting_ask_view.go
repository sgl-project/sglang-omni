package main

// The card that offers to record a detected meeting: what was found, a
// button that starts a meeting, a menu beside it (ask later, start by
// itself from now on, the settings) and a close button on its corner
// while the pointer is over it. Its window is meeting_ask_window.go's.

import (
	"github.com/egoist/mygo"
	"github.com/egoist/mygo/ui"
	"github.com/sgl-project/sglang-omni/Voxt/internal/icons"
	"github.com/sgl-project/sglang-omni/Voxt/internal/overlay/layout"
)

const (
	meetingAskFaceW    = 392
	meetingAskFaceH    = 64
	meetingAskCompactH = 40
	meetingAskRadius   = 16
	meetingAskPadX     = 14
	meetingAskBadge    = 30
	meetingAskButtonH  = 36
	meetingAskMenuRowH = 30
	meetingAskMenuGap  = 2
	meetingAskMenuPad  = 6
	meetingAskClose    = 22
)

var (
	meetingAskLive       = ui.RGB(48, 209, 88)
	meetingAskButton     = ui.RGB(245, 245, 247)
	meetingAskButtonOver = ui.RGB(255, 255, 255)
	meetingAskButtonInk  = ui.RGB(22, 22, 26)
)

type meetingAskActions interface {
	startDetectedMeeting()
	dismissMeetingAsk()
	toggleMeetingAskMenu()
	chooseMeetingAskMenu(id string)
	meetingAskHovered()
}

// meetingAskLayout is the window's size for a state: the face and the
// shadow's margin around it.
type meetingAskLayout struct {
	Width, Height int
}

func meetingAskLayoutModel(state meetingAskViewState) meetingAskLayout {
	height := meetingAskFaceH
	switch {
	case state.Compact:
		height = meetingAskCompactH
	case state.MenuOpen:
		height += meetingAskMenuHeight(len(state.Menu))
	}
	return meetingAskLayout{Width: meetingAskFaceW + 2*layout.Pad, Height: height + 2*layout.Pad}
}

// meetingAskMenuHeight is the menu's rows, its padding and the line under it.
func meetingAskMenuHeight(rows int) int {
	if rows == 0 {
		return 0
	}
	rules := 2 // under the first item, and between the menu and the card's row
	return rows*meetingAskMenuRowH + (rows-1)*meetingAskMenuGap + 2*meetingAskMenuPad + rules + meetingAskMenuGap
}

// meetingAskFrame is where the card goes in area: at first on the left
// edge, a little below the middle; after that where it is (at: the user
// may have dragged it), growing upwards, as its menu opens above its row.
func meetingAskFrame(at, area mygo.Rectangle, width, height int) mygo.Rectangle {
	if at.Width == 0 {
		return mygo.Rectangle{X: area.X + 8, Y: area.Y + area.Height*56/100 - height, Width: width, Height: height}
	}
	x := max(min(at.X, area.X+area.Width-width+layout.Pad), area.X-layout.Pad)
	y := max(min(at.Y+at.Height-height, area.Y+area.Height-height+layout.Pad), area.Y-layout.Pad)
	return mygo.Rectangle{X: x, Y: y, Width: width, Height: height}
}

func meetingAskContent(state func() meetingAskViewState, actions meetingAskActions, style func() meetingOverlayFaceStyle) mygo.Content {
	return hudContent(func(c *ui.Context) {
		meetingAskView(c, state(), actions, style())
	})
}

func meetingAskView(c *ui.Context, state meetingAskViewState, actions meetingAskActions, style meetingOverlayFaceStyle) {
	c.SetTheme(overlayTheme(style.palette))
	defer c.Root().Background(ui.Transparent)
	if !state.Visible {
		return
	}
	root := ui.Box(c.Key("ask")).Fill().Padding(layout.Pad)
	over := root.Hovered()
	if over {
		actions.meetingAskHovered()
	}
	root.Children(func() {
		face := faceStyle(c, ui.Column(c.Key("ask-face")), meetingAskRadius, style.alpha).DragWindow().Label(state.Title + " · " + state.App)
		face.Children(func() {
			if state.Compact {
				meetingAskCompactRow(c, state)
				return
			}
			if state.MenuOpen {
				meetingAskMenu(c, state, actions)
			}
			meetingAskRow(c, state, actions)
		})
		if (over || state.MenuOpen) && !state.Compact {
			meetingAskCloseButton(c, state, actions)
		}
	})
}

func meetingAskBadgeIcon(c *ui.Context, size float32) {
	ui.Box(c).Size(size, size).Radius(size / 2).Background(tok(c.Theme()).Accent).Center().Shrink(0).Children(func() {
		ui.Icon(c, meetingIcon.Bold).Size(size/2, size/2).TextColor(ui.RGB(255, 255, 255)).Shrink(0)
	})
}

func meetingAskLiveLine(c *ui.Context, app string, size float32) {
	ui.Row(c).Gap(5).AlignItems(ui.Center).MinWidth(0).Shrink(1).Children(func() {
		ui.Box(c).Size(6, 6).Radius(3).Background(meetingAskLive).Shrink(0)
		ui.Text(c, app).FontSize(size).TextColor(ovText2).SingleLine().Ellipsis("…").MinWidth(0).Shrink(1)
	})
}

func meetingAskRow(c *ui.Context, state meetingAskViewState, actions meetingAskActions) {
	ui.Row(c).Height(meetingAskFaceH-2*layout.Border).Shrink(0).Padding(0, meetingAskPadX-layout.Border).Gap(10).AlignItems(ui.Center).Children(func() {
		meetingAskBadgeIcon(c, meetingAskBadge)
		ui.Column(c).Grow(1).MinWidth(0).Gap(2).Children(func() {
			ui.Text(c, state.Title).FontSize(13).FontWeight(600).TextColor(ovText).SingleLine().Ellipsis("…")
			meetingAskLiveLine(c, state.App, 11)
		})
		ui.Row(c).Gap(1).AlignItems(ui.Center).Shrink(0).Children(func() {
			start := ui.ButtonBase(c.Key("ask-start")).Height(meetingAskButtonH).Padding(0, 12).Radius(9).Center().Shrink(0).Label(state.StartLabel)
			start.Background(meetingAskButtonFill(start.Hovered()))
			if start.Children(func() {
				ui.Row(c).Gap(6).AlignItems(ui.Center).Children(func() {
					ui.Icon(c, icons.VoxtMicrophone.Bold).Size(14, 14).TextColor(meetingAskButtonInk).Shrink(0)
					ui.Text(c, state.StartLabel).FontSize(12.5).FontWeight(600).TextColor(meetingAskButtonInk).SingleLine()
				})
			}).Clicked() {
				actions.startDetectedMeeting()
			}
			chevron := icons.AltArrowUp
			if state.MenuOpen {
				chevron = icons.AltArrowDown
			}
			menu := ui.ButtonBase(c.Key("ask-menu")).Size(26, meetingAskButtonH).Radius(9).Center().Shrink(0).Label(state.MenuLabel).Tooltip(state.MenuLabel).Checked(state.MenuOpen)
			menu.Background(meetingAskButtonFill(menu.Hovered() || state.MenuOpen))
			if menu.Children(func() {
				ui.Icon(c, chevron.Linear).Size(12, 12).TextColor(meetingAskButtonInk).Shrink(0)
			}).Clicked() {
				actions.toggleMeetingAskMenu()
			}
		})
	})
}

func meetingAskButtonFill(over bool) ui.Color {
	if over {
		return meetingAskButtonOver
	}
	return meetingAskButton
}

func meetingAskCompactRow(c *ui.Context, state meetingAskViewState) {
	ui.Row(c).Height(meetingAskCompactH-2*layout.Border).Shrink(0).Padding(0, meetingAskPadX-layout.Border).Gap(10).AlignItems(ui.Center).Children(func() {
		meetingAskBadgeIcon(c, 20)
		ui.Text(c, state.Title).FontSize(12.5).FontWeight(600).TextColor(ovText).SingleLine().Ellipsis("…").Grow(1).MinWidth(0)
		meetingAskLiveLine(c, state.App, 11)
	})
}

func meetingAskMenu(c *ui.Context, state meetingAskViewState, actions meetingAskActions) {
	line := tok(c.Theme()).OverlayLine
	ui.Column(c.Key("ask-menu-rows")).Shrink(0).Padding(meetingAskMenuPad, meetingAskMenuPad, 0).Gap(meetingAskMenuGap).Children(func() {
		for _, item := range state.Menu {
			if meetingMenuRow(c.Key("ask-item-"+item.ID), item.Label, "", false, false) {
				actions.chooseMeetingAskMenu(item.ID)
			}
			if item.Rule {
				ui.Box(c).Height(1).Shrink(0).Background(line)
			}
		}
	})
	ui.Box(c).Height(1).Shrink(0).Margin(meetingAskMenuPad, 0, 0).Background(line)
}

// meetingAskCloseButton sits on the card's top left corner, half outside
// it, in the shadow's margin.
func meetingAskCloseButton(c *ui.Context, state meetingAskViewState, actions meetingAskActions) {
	k := tok(c.Theme())
	offset := float32(layout.Pad) - meetingAskClose/2 + 2
	button := ui.ButtonBase(c.Key("ask-close")).Absolute().Left(offset).Top(offset).Size(meetingAskClose, meetingAskClose).
		Radius(meetingAskClose/2).Center().Label(state.CloseLabel).Tooltip(state.CloseLabel).Border(1, k.OverlayLine)
	fill := k.OverlayFace
	if button.Hovered() {
		fill = k.OverlayFace.Mix(ui.RGB(255, 255, 255), 0.16)
	}
	if button.Background(fill).Children(func() {
		ui.Icon(c, icons.Close.Linear).Size(11, 11).TextColor(ovText).Shrink(0)
	}).Clicked() {
		actions.dismissMeetingAsk()
	}
}
