package main

// Meeting overlay rendering. Native window ownership lives in meeting_overlay_window.go.

import (
	"github.com/egoist/mygo"
	"github.com/egoist/mygo/ui"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/icons"
	"github.com/sgl-project/sglang-omni/Voxt/internal/overlay/layout"
)

type meetingOverlayLocalUI struct {
	collapsed bool
	list      ui.ListState
	menu      meetingMenu
	abandon   bool
}

type meetingOverlayAnimation struct {
	motion bool
	phase  float64
}

type meetingOverlayFaceStyle struct {
	radius  float32
	alpha   float32
	palette string
}

func meetingOverlayContent(
	state func() meetingOverlayViewState,
	model func() meetingOverlayModel,
	actions meetingOverlayActions,
	local *meetingOverlayLocalUI,
	animation func() meetingOverlayAnimation,
	style func() meetingOverlayFaceStyle,
) mygo.Content {
	return hudContent(func(c *ui.Context) {
		meetingOverlayView(c, state(), model(), actions, local, animation(), style())
	})
}

func meetingOverlayView(
	c *ui.Context,
	state meetingOverlayViewState,
	model meetingOverlayModel,
	actions meetingOverlayActions,
	local *meetingOverlayLocalUI,
	animation meetingOverlayAnimation,
	style meetingOverlayFaceStyle,
) {
	c.SetTheme(overlayTheme(style.palette))
	defer c.Root().Background(ui.Transparent)
	if !animation.motion {
		c.DisableAnimations()
	} else if model.ShowFinishing {
		c.After(layout.Tick)
	}
	ui.Column(c).Fill().Padding(layout.Pad).Children(func() {
		face := faceStyle(c, ui.Column(c.Key("mt-face")), style.radius, style.alpha).Padding(layout.PillPadY-layout.Border, layout.PillPadX-layout.Border)
		if model.ShowExpanded {
			face.Padding(layout.CardPadTop-layout.Border, layout.CardPadX-layout.Border, layout.CardPadY-layout.Border)
		}
		face.DragWindow().Children(func() {
			switch {
			case model.ShowAsking:
				meetingOverlayAsking(c, actions)
			case model.ShowFinishing:
				meetingOverlayFinishing(c, state.FinishingText, pillLook{motion: animation.motion, phase: animation.phase})
			default:
				meetingOverlayBar(c, state, model, actions, local)
				meetingOverlayProblem(c, state.ProblemText, !model.ShowExpanded)
				if model.ShowExpanded {
					meetingOverlayExpanded(c, state, actions, local)
				}
			}
		})
	})
	meetingAbandonDialog(c, actions, local)
}

var (
	mtRecDot = ui.RGB(255, 69, 58)
	mtWarn   = ui.RGB(255, 196, 92)
)

func meetingOverlayBar(c *ui.Context, state meetingOverlayViewState, model meetingOverlayModel, actions meetingOverlayActions, local *meetingOverlayLocalUI) {
	rowHeight, gap := float32(layout.RowH), float32(layout.Gap)
	if model.ShowExpanded {
		rowHeight, gap = layout.CardHeadH, layout.CardHeadGap
	}
	ui.Row(c).Height(rowHeight).Shrink(0).Gap(gap).AlignItems(ui.Center).Children(func() {
		color := mtRecDot
		if state.Paused {
			color = ovText2
		}
		ui.Box(c).Size(layout.IconSlot, rowHeight).Shrink(0).Center().Children(func() {
			ui.Icon(c, meetingIcon.Bold).Size(layout.IconSize, layout.IconSize).TextColor(color).Shrink(0).Label(state.StatusText)
		})
		if model.ShowExpanded && state.Live {
			meetingModeCapsule(c, state, actions, local)
		} else {
			ui.Text(c, state.ModeTitle).FontSize(12).FontWeight(500).TextColor(ovText).SingleLine().Shrink(0)
		}
		if state.Paused {
			ui.Text(c, i18n.T("Paused")).FontSize(12).TextColor(ovText2).SingleLine().Shrink(0)
		} else {
			ui.Row(c).Gap(8).AlignItems(ui.Center).Shrink(0).Children(func() {
				if state.ShowMicrophone {
					meetingWave(c.Key("mt-wave-mic"), icons.MicrophoneLarge, i18n.T("Microphone"), state.MicrophoneLevels)
				}
				if state.ShowSystem {
					meetingWave(c.Key("mt-wave-system"), icons.VolumeLoud, i18n.T("System audio"), state.SystemLevels)
				}
			})
		}
		ui.Text(c, state.ElapsedText).FontSize(12).Font("monospace").TextColor(ovText2).SingleLine().Shrink(0)
		ui.Box(c).Grow(1)
		if state.Live {
			label, icon := i18n.T("Pause meeting"), icons.Pause
			if state.Paused {
				label, icon = i18n.T("Resume meeting"), icons.Play
			}
			if cardIconButton(c.Key("mt-pause"), label, label, icon, true) {
				local.menu = meetingMenuNone
				actions.meetingPauseResume()
			}
		}
		if state.Live || state.Starting {
			if cardIconButton(c.Key("mt-end"), i18n.T("End meeting"), i18n.T("End meeting"), icons.Stop, true) {
				local.menu = meetingMenuNone
				actions.meetingFinish()
			}
		}
		if model.ShowExpanded && state.Live && cardIconButton(c.Key("mt-more"), i18n.T("More"), i18n.T("More"), icons.MenuDots, true) {
			actions.toggleMeetingMenu(meetingMenuMore)
		}
		label, icon := i18n.T("Collapse"), icons.MinimizeSquareMinimalistic
		if model.Layout.Collapsed {
			label, icon = i18n.T("Expand"), icons.MaximizeSquareMinimalistic
		}
		if cardIconButton(c.Key("mt-ov-collapse"), label, label, icon, true) {
			actions.setMeetingOverlayCollapsed(!model.Layout.Collapsed)
		}
	})
}

func meetingModeCapsule(c *ui.Context, state meetingOverlayViewState, actions meetingOverlayActions, local *meetingOverlayLocalUI) {
	open := local.menu == meetingMenuMode
	button := ui.ButtonBase(c.Key("mt-mode")).Height(layout.CardBtn).Padding(0, 10-layout.Border).Radius(layout.CardBtn / 2).Shrink(0).
		Label(i18n.T("Meeting mode")).Checked(open).Tooltip(state.ModeHint)
	background, line := ui.RGBA(255, 255, 255, 0.08), ui.RGBA(255, 255, 255, 0.1)
	if open || button.Hovered() {
		background, line = ui.RGBA(255, 255, 255, 0.16), ui.RGBA(255, 255, 255, 0.18)
	}
	button.Background(background).Border(1, line)
	chevron := icons.AltArrowDown
	if open {
		chevron = icons.AltArrowUp
	}
	if button.Children(func() {
		ui.Row(c).Gap(6).AlignItems(ui.Center).Children(func() {
			ui.Text(c, state.ModeTitle).FontSize(11).FontWeight(600).TextColor(ui.RGBA(255, 255, 255, 0.92)).SingleLine()
			ui.Icon(c, chevron.Bold).Size(10, 10).TextColor(ui.RGBA(255, 255, 255, 0.72)).Shrink(0)
		})
	}).Clicked() {
		actions.toggleMeetingMenu(meetingMenuMode)
	}
}

func meetingWave(c *ui.Context, icon icons.Icon, label string, levels []float64) {
	const scaleX, scaleY = 0.6, 0.7
	bars := layout.WaveBars(levels)
	ui.Row(c).Gap(4).AlignItems(ui.Center).Shrink(0).Label(label).Children(func() {
		ui.Icon(c, icon.Linear).Size(12, 12).TextColor(ovText2).Shrink(0)
		ui.Row(c).Height(layout.WaveBarMaxH * scaleY).Gap(layout.WaveBarGap * scaleX).AlignItems(ui.Center).Shrink(0).Children(func() {
			for _, value := range bars {
				height := (layout.WaveBarMinH + (layout.WaveBarMaxH-layout.WaveBarMinH)*float32(value)) * scaleY
				ui.Box(c).Width(layout.WaveBarW*scaleX).Height(height).Radius(layout.WaveBarR).Gradient(ui.RGBA(255, 255, 255, 0.98), ui.RGBA(255, 255, 255, 0.8), 180).Shrink(0)
			}
		})
	})
}

func meetingOverlayProblem(c *ui.Context, message string, short bool) {
	if message == "" {
		return
	}
	ui.Row(c.Key("mt-problem")).Gap(6).AlignItems(ui.Start).Shrink(0).Margin(layout.LiveGap, 0, 0, 0).Children(func() {
		ui.Icon(c, icons.DangerTriangle.Linear).Size(14, 14).TextColor(mtWarn).Shrink(0)
		text := ui.Text(c, message).FontSize(12).TextColor(mtWarn).Grow(1).Shrink(1).MinWidth(0)
		if short {
			text.MaxLines(2).Ellipsis("…").Tooltip(message)
		}
	})
}

func meetingOverlaySizeButton(c *ui.Context, step int) (next int, clicked bool) {
	step = min(max(step, 0), len(meetingOvWidths)-1)
	label, icon, next := i18n.T("Make the card larger"), icons.MaximizeSquareMinimalistic, step+1
	if next == len(meetingOvWidths) {
		label, icon, next = i18n.T("Make the card smaller"), icons.MinimizeSquareMinimalistic, 0
	}
	button := ui.ButtonBase(c).Absolute().Right(layout.CardSizeBtnInset).Bottom(layout.CardSizeBtnInset).
		Size(layout.CardSizeBtn, layout.CardSizeBtn).Center().Label(label).Tooltip(label)
	alpha := float32(0.6)
	if button.Hovered() {
		alpha = 0.95
	}
	return next, button.Children(func() {
		ui.Icon(c, icon.Linear).Size(layout.CardSizeBtnIcon, layout.CardSizeBtnIcon).TextColor(ui.RGBA(255, 255, 255, alpha))
	}).Clicked()
}

func meetingOverlayExpanded(c *ui.Context, state meetingOverlayViewState, actions meetingOverlayActions, local *meetingOverlayLocalUI) {
	ui.Box(c).Height(1).Shrink(0).Margin(layout.CardRuleTop, 0, 0, 0).Background(ovCardRule)
	if local.menu != meetingMenuNone && state.Live {
		meetingOverlayMenu(c, state, actions, local)
	} else {
		meetingOverlayTranscript(c, state.Rows, local)
	}
	if next, ok := meetingOverlaySizeButton(c.Key("mt-size"), state.CardSize); ok {
		actions.setMeetingOverlayCardSize(next)
	}
}

func meetingOverlayTranscript(c *ui.Context, rows []meetingOverlayTranscriptRow, local *meetingOverlayLocalUI) {
	count := max(len(rows), 1)
	local.list.FollowEnd = true
	ui.List(c.Key("mt-live"), &local.list, count).Grow(1).MinHeight(0).Margin(layout.CardRuleGap, 0, 0, 0).Gap(meetingOvRowGap).Padding(0, layout.CardInset, 0, 0).
		ItemKey(func(index int) any {
			if index < len(rows) {
				return rows[index].Key
			}
			return nil
		}).Rows(func(row ui.ListRow) {
		if row.Index >= len(rows) {
			ui.Text(c, i18n.T("Listening…")).FontSize(layout.CardTextSize).TextColor(ovText2)
			return
		}
		meetingOverlayLine(c, rows[row.Index])
	})
}

func meetingOverlayLine(c *ui.Context, row meetingOverlayTranscriptRow) {
	ui.Column(c).Gap(2).Shrink(0).Children(func() {
		if row.First {
			whoColor := ovText2
			if row.Me {
				whoColor = ovText
			}
			ui.Text(c, row.Who).FontSize(11).FontWeight(600).TextColor(whoColor)
		}
		color := ovText
		if row.Live {
			color = ovLiveColor
		}
		ui.Text(c, row.Text).FontSize(layout.CardTextSize).FontWeight(layout.CardTextWeight).TextColor(color)
	})
}

func meetingOverlayMenu(c *ui.Context, state meetingOverlayViewState, actions meetingOverlayActions, local *meetingOverlayLocalUI) {
	ui.Column(c.Key("mt-menu")).Gap(4).Shrink(0).Margin(layout.CardRuleGap, 0, 0, 0).Children(func() {
		if local.menu == meetingMenuMode {
			for _, option := range state.Modes {
				if meetingMenuRow(c.Key("mt-mode-"+option.ID), option.Title, option.Hint, option.Selected, false) {
					actions.meetingSetModeID(option.ID)
				}
			}
			return
		}
		for _, option := range state.MoreItems {
			clicked := meetingMenuRow(c.Key("mt-"+option.ID), option.Label, "", option.Selected, option.Danger)
			if !clicked {
				continue
			}
			switch option.ID {
			case "translate":
				actions.toggleMeetingTranslation(state.ID)
			case "copy":
				actions.copyLiveMeetingTranscript()
			case "cancel":
				actions.requestMeetingDiscard()
			}
		}
	})
}

func meetingMenuRow(c *ui.Context, label, hint string, on, danger bool) bool {
	height := float32(30)
	if hint != "" {
		height = meetingOvMenuRowH
	}
	button := ui.ButtonBase(c).Height(height).Padding(0, 10).Radius(8).Shrink(0).Label(label).Checked(on)
	if button.Hovered() {
		button.Background(ui.RGBA(255, 255, 255, 0.1))
	}
	textColor := ovText
	if danger {
		textColor = mtRecDot
	}
	return button.Children(func() {
		ui.Row(c).Fill().Gap(8).AlignItems(ui.Center).Children(func() {
			ui.Column(c).Grow(1).MinWidth(0).Gap(1).Children(func() {
				ui.Text(c, label).FontSize(13).FontWeight(500).TextColor(textColor).SingleLine().Ellipsis("…")
				if hint != "" {
					ui.Text(c, hint).FontSize(11).TextColor(ovText2).SingleLine().Ellipsis("…")
				}
			})
			if on {
				ui.Icon(c, icons.Check.Linear).Size(14, 14).TextColor(ovText).Shrink(0)
			}
		})
	}).Clicked()
}

func meetingOverlayAsking(c *ui.Context, actions meetingOverlayActions) {
	ui.Row(c).Height(layout.RowH).Shrink(0).Gap(8).AlignItems(ui.Center).Children(func() {
		ui.Text(c, i18n.T("Save this meeting?")).FontSize(layout.NoticeSize).FontWeight(500).TextColor(ovText).SingleLine().Grow(1).MinWidth(0)
		if cardCapsule(c.Key("mt-save"), i18n.T("Save"), "") {
			actions.meetingSave()
		}
		if cardCapsule(c.Key("mt-discard"), i18n.T("Discard"), "") {
			actions.meetingDiscard()
		}
	})
}

func meetingOverlayFinishing(c *ui.Context, message string, look pillLook) {
	ui.Row(c).Height(layout.RowH).Shrink(0).Gap(8).AlignItems(ui.Center).Justify(ui.Center).Children(func() {
		overlayLoader(c, message, look, 0)
		ui.Text(c, message).FontSize(layout.NoticeSize).TextColor(ovText).SingleLine().Ellipsis("…").MinWidth(0).Shrink(1)
	})
}

func meetingAbandonDialog(c *ui.Context, actions meetingOverlayActions, local *meetingOverlayLocalUI) {
	t := c.Theme()
	ui.Modal(c.Key("mt-abandon-ask"), &local.abandon, func() {
		ui.Text(c, i18n.T("Cancel this meeting?")).Bold()
		ui.Text(c, i18n.T("The recording and any unsaved text are deleted. This cannot be undone.")).Width(px(t, 380))
		ui.Row(c).Gap(px(t, sp2)).Justify(ui.End).Children(func() {
			if ui.Button(c.Key("mt-abandon-no"), i18n.T("Keep recording")).AutoFocus().Clicked() {
				local.abandon = false
			}
			if recordButton(c.Key("mt-abandon-yes"), i18n.T("Cancel meeting")).Clicked() {
				local.abandon = false
				actions.meetingDiscard()
			}
		})
	})
}
