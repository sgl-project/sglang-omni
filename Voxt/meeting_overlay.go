package main

// Meeting overlay rendering. Native window ownership lives in meeting_overlay_window.go.

import (
	"github.com/egoist/mygo"
	"github.com/egoist/mygo/ui"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/icons"
	"github.com/sgl-project/sglang-omni/Voxt/internal/overlay/layout"
)

// meetingTimeW is the least room the meeting's elapsed time keeps ("0:00"
// at 12 pt), so the controls do not move as it ticks.
const meetingTimeW = 30

type meetingOverlayLocalUI struct {
	collapsed bool
	list      ui.ListState
	menu      meetingMenu
	abandon   bool
	// endAsk: the overlay's End button asks first; it sits next to Pause
	// (11-25: the user asked for it). Meetings and the menu bar end at
	// once, as the end card's End recording does: there it was chosen.
	endAsk bool
	// endAbove: End's bubble goes above the face (an overlay low on the
	// screen grows upwards), else below it.
	endAbove bool
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
		if model.EndAsk && local.endAbove {
			meetingEndBubble(c, model, actions, true)
		}
		face := faceStyle(c, ui.Column(c.Key("mt-face")), style.radius, style.alpha).Padding(layout.PillPadY-layout.Border, layout.PillPadX-layout.Border)
		if model.EndAsk {
			_, height := meetingOverlayFace(model)
			face.Height(float32(height)).Shrink(0) // the bubble has the rest
		}
		if model.ShowExpanded {
			face.Padding(layout.CardPadTop-layout.Border, layout.CardPadX-layout.Border, layout.CardPadY-layout.Border)
		}
		if model.AskingCard {
			face.Padding(0, meetingOvAskPadX-layout.Border)
		}
		face.DragWindow().Children(func() {
			switch {
			case model.ShowAsking:
				meetingOverlayAsking(c, state, actions)
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
		if model.EndAsk && !local.endAbove {
			meetingEndBubble(c, model, actions, false)
		}
	})
	meetingAbandonDialog(c, actions, local)
}

// meetingEndBubble is End's question: a small translucent bubble out of
// the overlay beside End (above the face, or below it), with Keep
// recording and End. Not a card in the card (11-25, the user: a
// dialog stacked on the overlay looked odd).
func meetingEndBubble(c *ui.Context, model meetingOverlayModel, actions meetingOverlayActions, above bool) {
	row := ui.Row(c).Height(meetingOvEndAskH).Shrink(0).Justify(ui.End).Padding(0, layout.PillPadX-6, 0, 0)
	if above {
		row.AlignItems(ui.Start)
	} else {
		row.AlignItems(ui.End)
	}
	row.Children(func() {
		bubble := ui.Row(c.Key("mt-end-ask")).Height(meetingOvEndBubbleH).Shrink(0).Gap(6).AlignItems(ui.Center).
			Padding(0, 4, 0, 12).Radius(meetingOvEndBubbleH/2).Background(ui.RGBA(22, 22, 25, 0.62)).
			Border(1, ui.RGBA(255, 255, 255, 0.14)).Shadow(0, 3, 10, 0, ui.RGBA(0, 0, 0, 0.22)).Label(i18n.T("End this meeting?"))
		bubble.Children(func() {
			ui.Text(c, i18n.T("End this meeting?")).FontSize(12).FontWeight(600).TextColor(ovText).SingleLine().Shrink(0)
			if meetingEndBubbleButton(c.Key("mt-end-no"), i18n.T("Keep recording"), false) {
				actions.keepMeetingRecording()
			}
			if meetingEndBubbleButton(c.Key("mt-end-yes"), i18n.T("End"), true) {
				actions.meetingFinish()
			}
		})
	})
}

// meetingEndBubbleButton is a capsule in End's bubble: End in red, Keep
// recording quiet.
func meetingEndBubbleButton(c *ui.Context, label string, end bool) bool {
	b := ui.ButtonBase(c).Height(meetingOvEndBubbleH-8).Padding(0, 10).Radius((meetingOvEndBubbleH - 8) / 2).Center().Shrink(0).Label(label)
	fill, ink := ui.RGBA(255, 255, 255, 0.1), ovText
	if b.Hovered() {
		fill = ui.RGBA(255, 255, 255, 0.18)
	}
	if end {
		fill, ink = mtRecDot, ui.RGB(255, 255, 255)
		if b.Hovered() {
			fill = mtRecDot.Mix(ui.RGB(255, 255, 255), 0.12)
		}
	}
	return b.Background(fill).Children(func() {
		ui.Text(c, label).FontSize(11.5).FontWeight(600).TextColor(ink).SingleLine()
	}).Clicked()
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
		color := ovIcon
		if state.Paused {
			color = ovText2
		}
		ui.Box(c).Size(layout.IconSlot, rowHeight).Shrink(0).Center().Children(func() {
			ui.Icon(c, meetingIcon.Linear).Size(layout.IconSize, layout.IconSize).TextColor(color).Shrink(0).Label(state.StatusText)
		})
		if model.ShowExpanded && state.Live {
			meetingModeCapsule(c, state, actions, local)
		} else {
			// Align the text's visible strokes with the waveform and SVG centers.
			ui.Text(c, state.ModeTitle).FontSize(12).FontWeight(500).FixedLineHeight(layout.LineH).TextColor(ovText).SingleLine().Shrink(0).Translate(0, -1.5)
		}
		if state.Paused {
			ui.Text(c, i18n.T("Paused")).FontSize(12).FixedLineHeight(layout.LineH).TextColor(ovText2).SingleLine().Shrink(0).Translate(0, -1.5)
		} else {
			ui.Row(c).Gap(8).AlignItems(ui.Center).Shrink(0).Children(func() {
				if state.ShowMicrophone {
					meetingWave(c.Key("mt-wave-mic"), icons.VoxtMicrophone, i18n.T("Microphone"), state.MicrophoneLevels)
				}
				if state.ShowSystem {
					meetingWave(c.Key("mt-wave-system"), icons.VolumeLoud, i18n.T("System audio"), state.SystemLevels)
				}
			})
		}
		ui.Box(c).Grow(1).MinWidth(meetingTimeW).Height(rowHeight).Center().Children(func() {
			ui.Text(c, state.ElapsedText).FontSize(12).FontFeatures("tnum").FixedLineHeight(layout.LineH).TextColor(ovText2).SingleLine().Shrink(0).Translate(0, -0.75)
		})
		ui.Row(c.Key("mt-actions")).Gap(2).AlignItems(ui.Center).Shrink(0).Children(func() {
			if state.Live {
				label, icon := i18n.T("Pause meeting"), icons.Pause.Bold
				if state.Paused {
					label, icon = i18n.T("Resume meeting"), icons.Play.Bold
				}
				if meetingOverlayButton(c.Key("mt-pause"), label, icon, 14) {
					local.menu = meetingMenuNone
					actions.meetingPauseResume()
				}
			}
			if state.Live || state.Starting { // a start that hangs can be ended
				if meetingOverlayButton(c.Key("mt-end"), i18n.T("End meeting"), icons.Stop.Bold, 14) {
					local.menu = meetingMenuNone
					if state.Starting {
						actions.meetingFinish() // a start that hangs: nothing recorded to keep
					} else {
						actions.requestMeetingFinish()
					}
				}
			}
			if model.ShowExpanded && state.Live && meetingOverlayButton(c.Key("mt-more"), i18n.T("More"), icons.MenuDots.Linear, 14) {
				actions.toggleMeetingMenu(meetingMenuMore)
			}
			label, icon := i18n.T("Collapse"), icons.Minimize.Linear
			if model.Layout.Collapsed {
				label, icon = i18n.T("Expand"), icons.Maximize.Linear
			}
			if meetingOverlayButton(c.Key("mt-ov-collapse"), label, icon, 12) {
				actions.setMeetingOverlayCollapsed(!model.Layout.Collapsed)
			}
		})
	})
}

func meetingOverlayButton(c *ui.Context, label string, icon *ui.SVG, iconSize float32) bool {
	b := ui.ButtonBase(c).Size(layout.CardBtn, layout.CardBtn).Radius(6).Center().Shrink(0).Label(label).Tooltip(label)
	background := ui.Transparent
	if b.Hovered() {
		background = ui.RGBA(255, 255, 255, 0.1)
	}
	b.Background(background)
	return b.Children(func() {
		ui.Icon(c, icon).Size(iconSize, iconSize).TextColor(ovText)
	}).Clicked()
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

// meetingOverlayAsking is the question a meeting ends with (history off,
// or hardly anything heard): keep it or discard it. With a line saying
// why it asks, it is a card: an icon, the question over the line, and the
// buttons centred beside both; without, one row in the pill.
func meetingOverlayAsking(c *ui.Context, state meetingOverlayViewState, actions meetingOverlayActions) {
	card := state.AskingHint != ""
	// One row either way (each ui.Row is an element): the card's fills the
	// face and centres its parts, the pill's is the pill's row.
	row, button := ui.Row(c).AlignItems(ui.Center), float32(layout.CardBtn+2)
	if card {
		row, button = row.Grow(1).Gap(10), meetingOvAskBtnH
	} else {
		row = row.Height(layout.RowH).Shrink(0).Gap(8)
	}
	row.Children(func() {
		if card {
			ui.Box(c).Size(meetingOvAskBadge, meetingOvAskBadge).Radius(meetingOvAskBadge / 2).Background(mtWarn.Alpha(0.16)).Center().Shrink(0).Children(func() {
				ui.Icon(c, icons.DangerTriangle.Bold).Size(15, 15).TextColor(mtWarn).Shrink(0)
			})
		}
		ui.Column(c).Grow(1).MinWidth(0).Gap(2).Children(func() {
			ui.Text(c, state.AskingText).FontSize(layout.NoticeSize).FontWeight(600).FixedLineHeight(18).TextColor(ovText).SingleLine().Ellipsis("…")
			if card {
				ui.Text(c.Key("mt-ask-hint"), state.AskingHint).FontSize(11).FixedLineHeight(15).TextColor(ovText2).SingleLine().Ellipsis("…")
			}
		})
		ui.Row(c).Gap(6).AlignItems(ui.Center).Shrink(0).Children(func() {
			if meetingAskingButton(c.Key("mt-discard"), state.DiscardLabel, button, false) {
				actions.meetingDiscard()
			}
			if meetingAskingButton(c.Key("mt-save"), state.KeepLabel, button, true) {
				actions.meetingSave()
			}
		})
	})
}

// meetingAskingButton is a button of the question, height high: the
// answer that loses nothing (primary) is light, the other one quiet.
func meetingAskingButton(c *ui.Context, label string, height float32, primary bool) bool {
	button := ui.ButtonBase(c).Height(height).MinWidth(2*height).Padding(0, 14-layout.Border).Radius(min(height/2, 9)).Center().Shrink(0).Label(label)
	fill, line, ink := ui.RGBA(255, 255, 255, 0.1), ui.RGBA(255, 255, 255, 0.12), ovText
	if primary {
		fill, line, ink = ui.RGB(245, 245, 247), ui.RGB(245, 245, 247), ui.RGB(22, 22, 26)
	}
	if button.Hovered() {
		fill, line = ui.RGBA(255, 255, 255, 0.18), ui.RGBA(255, 255, 255, 0.2)
		if primary {
			fill, line = ui.RGB(255, 255, 255), ui.RGB(255, 255, 255)
		}
	}
	button.Background(fill).Border(layout.Border, line)
	return button.Children(func() {
		// Half a point up: the line box leaves more room above the
		// strokes than below (measured on Chinese and English labels).
		ui.Text(c, label).FontSize(12.5).FontWeight(600).FixedLineHeight(16).TextColor(ink).SingleLine().Translate(0, -0.5)
	}).Clicked()
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
