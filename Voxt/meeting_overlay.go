package main

// The meeting overlay: timer, levels, live transcript, controls, and
// warnings while a meeting is active (design §4.12).

import (
	"fmt"
	"time"

	"github.com/egoist/mygo"
	"github.com/egoist/mygo/ui"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/icons"
	"github.com/sgl-project/sglang-omni/Voxt/internal/meeting"
	"github.com/sgl-project/sglang-omni/Voxt/internal/overlay/layout"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
	"github.com/sgl-project/sglang-omni/Voxt/internal/uitext"
)

type meetingOverlayState struct {
	meetOvWin       *mygo.Window
	meetOvShown     bool
	meetOvCollapsed bool
	meetOvList      ui.ListState // the live transcript's place
	meetOvMenu      meetingMenu  // the menu over the transcript
}

// meetingContentProtectionHook is set in tests to observe SetContentProtection.
var meetingContentProtectionHook func(bool)

func (a *app) meetingOverlayCollapsed() bool {
	if a.meetOvCollapsed {
		return true
	}
	return a.prefs.Features.Meeting.Collapsed
}

// meetingOverlayModel is the overlay's model now.
func (a *app) meetingOverlayModel() meetingOverlayModel {
	return overlayModel(a.meetSnap, a.meetingOverlayCollapsed(), a.prefs.Features.CardSize(meetingCardSize), a.meetOvMenu)
}

// meetingCardSize is the meeting window's key in Features.CardSizes.
const meetingCardSize = "meeting"

func (a *app) syncMeetingOverlay() {
	if !a.overlayNative() {
		return
	}
	show := a.meetingActive() || a.meetSnap.State == meeting.StateFinishing
	if !show {
		if a.meetOvWin != nil && a.meetOvShown {
			a.meetOvWin.Hide()
			a.meetOvShown = false
		}
		return
	}
	model := a.meetingOverlayModel()
	w, h := model.Layout.Width, model.Layout.Height
	if a.meetOvWin == nil {
		opts := floatingWindowOptions("Voxt meeting", w, h, hudContent(a.meetingOverlayView))
		opts.Vibrancy = mygo.VibrancyHUD
		opts.StateKey = "meeting-overlay"
		a.meetOvWin = mygo.NewWindow(opts)
		a.meetOvWin.SetMovable(true)
	}
	at := mygo.Rectangle{}
	if a.meetOvShown {
		at = a.meetOvWin.ContentBounds() // where the user dragged it
	}
	area := a.overlayArea()
	if at.Width > 0 {
		area = mygo.Screen.DisplayMatching(at).WorkArea
	}
	frame := meetingOverlayFrame(at, area, w, h)
	x, y := frame.X, frame.Y
	if frame != at {
		a.meetOvWin.SetContentBounds(frame)
	}
	mode, faceH := meetingOverlayFace(model)
	a.meetOvWin.SetVibrancyStyle(mygo.VibrancyStyle{Top: layout.Pad, Right: layout.Pad, Bottom: layout.Pad, Left: layout.Pad,
		Radius: float64(faceRadius(a.prefs.Overlay, mode, faceH)), Dark: true})
	a.meetOvWin.Invalidate()
	a.applyMeetingOverlayProtection()
	if !a.meetOvShown {
		showFloating(a.meetOvWin, x, y)
		a.meetOvShown = true
	}
}

// meetingOverlayFrame is the overlay's frame for w × h: where the window
// is now (at, zero before it shows), else at the bottom centre of the
// work area, the face kept inside the work area as it grows (the
// window's transparent margin, layout.Pad, may go past it, as
// overlayFrame's).
func meetingOverlayFrame(at, area mygo.Rectangle, w, h int) mygo.Rectangle {
	if at.Width == 0 {
		return mygo.Rectangle{X: area.X + (area.Width-w)/2, Y: area.Y + area.Height - h - 28, Width: w, Height: h}
	}
	x := max(min(at.X, area.X+area.Width-w+layout.Pad), area.X-layout.Pad)
	y := at.Y
	if at.Y+at.Height/2 > area.Y+area.Height/2 {
		y = at.Y + at.Height - h // in the lower half it grows upward: its bottom stays, as the text grows (11-15)
	}
	y = max(min(y, area.Y+area.Height-h+layout.Pad), area.Y-layout.Pad)
	return mygo.Rectangle{X: x, Y: y, Width: w, Height: h}
}

func (a *app) applyMeetingOverlayProtection() {
	hide := a.prefs.Features.Meeting.HideFromSharing
	if meetingContentProtectionHook != nil {
		meetingContentProtectionHook(hide)
	}
	if a.meetOvWin != nil {
		a.meetOvWin.SetContentProtection(hide)
	}
}

func (a *app) meetingOverlayView(c *ui.Context) {
	c.SetTheme(overlayTheme())
	// Only the face shows in this transparent window (overlayView's).
	defer c.Root().Background(ui.Transparent)
	s := a.meetSnap
	model := a.meetingOverlayModel()
	motion := a.overlayMotion()
	if !motion {
		c.DisableAnimations()
	} else if model.ShowFinishing {
		c.After(layout.Tick)
	}
	mode, faceH := meetingOverlayFace(model)
	ui.Column(c).Fill().Padding(layout.Pad).Children(func() {
		// The face is the handle that moves the window (syncMeetingOverlay
		// keeps it where it is dragged): it may cover the meeting's window.
		// The dictation pill's padding, or the answer card's when expanded (#3986).
		face := faceStyle(ui.Column(c.Key("mt-face")), a.prefs.Overlay, mode, faceH).Padding(layout.PillPadY-layout.Border, layout.PillPadX-layout.Border)
		if model.ShowExpanded {
			face.Padding(layout.CardPadTop-layout.Border, layout.CardPadX-layout.Border, layout.CardPadY-layout.Border)
		}
		face.DragWindow().Children(func() {
			switch {
			case model.ShowAsking:
				a.meetingOverlayAsking(c)
			case model.ShowFinishing:
				meetingOverlayFinishing(c, s, pillLook{motion: motion, phase: a.overlayPhase()})
			default:
				a.meetingOverlayBar(c, s, model)
				meetingOverlayProblem(c, model.ProblemKey, !model.ShowExpanded)
				if model.ShowExpanded {
					a.meetingOverlayExpanded(c, s)
				}
			}
		})
	})
	a.meetingAbandonDialog(c)
}

// meetingOverlayFace is the face's look and height for model: a capsule
// unless it shows the expanded card.
func meetingOverlayFace(model meetingOverlayModel) (overlayMode, int) {
	mode := ovRecording
	if model.ShowExpanded {
		mode = ovAnswer
	}
	return mode, model.Layout.Height - 2*layout.Pad
}

// Colours of the meeting face beyond overlay_face.go's.
var (
	mtRecDot = ui.RGB(255, 69, 58)
	mtWarn   = ui.RGB(255, 196, 92)
)

// meetingOverlayBar is the capsule's row, or the expanded card's header,
// in the dictation pill's order: the feature's icon and name, the
// tracks' waveforms, the time; then the actions.
func (a *app) meetingOverlayBar(c *ui.Context, s meeting.Snapshot, model meetingOverlayModel) {
	live := s.State == meeting.StateRecording || s.State == meeting.StatePaused
	paused := s.State == meeting.StatePaused
	var rowH, gap float32 = layout.RowH, layout.Gap // the pill's row, or the card's header
	if model.ShowExpanded {
		rowH, gap = layout.CardHeadH, layout.CardHeadGap
	}
	ui.Row(c).Height(rowH).Shrink(0).Gap(gap).AlignItems(ui.Center).Children(func() {
		// The icon is red while it records, as the menu bar's.
		col, status := mtRecDot, i18n.T("Meeting in progress")
		if paused {
			col, status = ovText2, i18n.T("Meeting paused")
		}
		ui.Box(c).Size(layout.IconSlot, rowH).Shrink(0).Center().Children(func() {
			ui.Icon(c, meetingIcon.Bold).Size(layout.IconSize, layout.IconSize).TextColor(col).Shrink(0).Label(status)
		})
		if model.ShowExpanded && live {
			a.meetingModeCapsule(c, s)
		} else {
			ui.Text(c, meetingModeTitle(s.Mode)).FontSize(12).FontWeight(500).TextColor(ovText).SingleLine().Shrink(0)
		}
		if paused {
			ui.Text(c, i18n.T("Paused")).FontSize(12).TextColor(ovText2).SingleLine().Shrink(0)
		} else {
			ui.Row(c).Gap(8).AlignItems(ui.Center).Shrink(0).Children(func() {
				if s.Mode != meeting.ModeSubtitles {
					meetingWave(c.Key("mt-wave-mic"), icons.MicrophoneLarge, i18n.T("Microphone"), s.Levels[meeting.SourceMic])
				}
				if s.Mode != meeting.ModeRecording {
					meetingWave(c.Key("mt-wave-system"), icons.VolumeLoud, i18n.T("System audio"), s.Levels[meeting.SourceSystem])
				}
			})
		}
		ui.Text(c, formatMeetingElapsed(s.Elapsed)).FontSize(12).Font("monospace").TextColor(ovText2).SingleLine().Shrink(0)
		ui.Box(c).Grow(1)
		if live {
			label, icon := i18n.T("Pause meeting"), icons.Pause
			if paused {
				label, icon = i18n.T("Resume meeting"), icons.Play
			}
			if cardIconButton(c.Key("mt-pause"), label, label, icon, true) {
				a.meetOvMenu = meetingMenuNone
				a.meetingPauseResume()
			}
		}
		if live || s.State == meeting.StateStarting { // a start that hangs can be ended
			if cardIconButton(c.Key("mt-end"), i18n.T("End meeting"), i18n.T("End meeting"), icons.Stop, true) {
				a.meetOvMenu = meetingMenuNone
				a.meetingFinish()
			}
		}
		if model.ShowExpanded && live && cardIconButton(c.Key("mt-more"), i18n.T("More"), i18n.T("More"), icons.MenuDots, true) {
			a.toggleMeetingMenu(meetingMenuMore)
		}
		label, icon := i18n.T("Collapse"), icons.MinimizeSquareMinimalistic
		if model.Layout.Collapsed {
			label, icon = i18n.T("Expand"), icons.MaximizeSquareMinimalistic
		}
		if cardIconButton(c.Key("mt-ov-collapse"), label, label, icon, true) {
			next := !model.Layout.Collapsed
			a.meetOvCollapsed, a.meetOvMenu = next, meetingMenuNone
			a.updateSettings(func(st *settings.Settings) { st.Features.Meeting.Collapsed = next })
			a.syncMeetingOverlay()
		}
	})
}

// meetingModeTitle is the name of a meeting's mode.
func meetingModeTitle(m meeting.Mode) string {
	switch m {
	case meeting.ModeSubtitles:
		return i18n.T("Subtitles")
	case meeting.ModeRecording:
		return i18n.T("Recording")
	default:
		return i18n.T("Meeting")
	}
}

// meetingModeCapsule is the mode in the card's header, a capsule with a
// chevron (the answer card's language capsule): it opens the modes.
func (a *app) meetingModeCapsule(c *ui.Context, s meeting.Snapshot) {
	open := a.meetOvMenu == meetingMenuMode
	b := ui.ButtonBase(c.Key("mt-mode")).Height(layout.CardBtn).Padding(0, 10-layout.Border).Radius(layout.CardBtn / 2).Shrink(0).
		Label(i18n.T("Meeting mode")).Checked(open).Tooltip(meetingModeHint(s.Mode))
	bg, line := ui.RGBA(255, 255, 255, 0.08), ui.RGBA(255, 255, 255, 0.1)
	if open || b.Hovered() {
		bg, line = ui.RGBA(255, 255, 255, 0.16), ui.RGBA(255, 255, 255, 0.18)
	}
	b.Background(bg).Border(1, line)
	chevron := icons.AltArrowDown
	if open {
		chevron = icons.AltArrowUp
	}
	if b.Children(func() {
		ui.Row(c).Gap(6).AlignItems(ui.Center).Children(func() {
			ui.Text(c, meetingModeTitle(s.Mode)).FontSize(11).FontWeight(600).TextColor(ui.RGBA(255, 255, 255, 0.92)).SingleLine()
			ui.Icon(c, chevron.Bold).Size(10, 10).TextColor(ui.RGBA(255, 255, 255, 0.72)).Shrink(0)
		})
	}).Clicked() {
		a.toggleMeetingMenu(meetingMenuMode)
	}
}

func (a *app) toggleMeetingMenu(m meetingMenu) {
	if a.meetOvMenu == m {
		m = meetingMenuNone
	}
	a.meetOvMenu = m
	a.syncMeetingOverlay()
}

// meetingWave is one track's waveform: its icon and the dictation pill's
// 16 bars, smaller, that follow the track's levels.
func meetingWave(c *ui.Context, icon icons.Icon, label string, levels []float64) {
	const sx, sy = 0.6, 0.7
	bars := layout.WaveBars(levels)
	ui.Row(c).Gap(4).AlignItems(ui.Center).Shrink(0).Label(label).Children(func() {
		ui.Icon(c, icon.Linear).Size(12, 12).TextColor(ovText2).Shrink(0)
		ui.Row(c).Height(layout.WaveBarMaxH * sy).Gap(layout.WaveBarGap * sx).AlignItems(ui.Center).Shrink(0).Children(func() {
			for _, v := range bars {
				h := (layout.WaveBarMinH + (layout.WaveBarMaxH-layout.WaveBarMinH)*float32(v)) * sy
				ui.Box(c).Width(layout.WaveBarW*sx).Height(h).Radius(layout.WaveBarR).Gradient(ui.RGBA(255, 255, 255, 0.98), ui.RGBA(255, 255, 255, 0.8), 180).Shrink(0)
			}
		})
	})
}

// meetingOverlayProblem is the meeting's warning under the bar: two lines
// at most in the capsule.
func meetingOverlayProblem(c *ui.Context, key string, short bool) {
	msg := meetingOverlayProblemText(key)
	if msg == "" {
		return
	}
	ui.Row(c.Key("mt-problem")).Gap(6).AlignItems(ui.Start).Shrink(0).Margin(layout.LiveGap, 0, 0, 0).Children(func() {
		ui.Icon(c, icons.DangerTriangle.Linear).Size(14, 14).TextColor(mtWarn).Shrink(0)
		text := ui.Text(c, msg).FontSize(12).TextColor(mtWarn).Grow(1).Shrink(1).MinWidth(0)
		if short {
			text.MaxLines(2).Ellipsis("…").Tooltip(msg)
		}
	})
}

// meetingOverlayExpanded is the card under the header: the rule, then the
// live transcript, or the menu the header opened in its place; the size
// button in the corner.
func (a *app) meetingOverlayExpanded(c *ui.Context, s meeting.Snapshot) {
	// The answer card's rule and text (#3986).
	ui.Box(c).Height(1).Shrink(0).Margin(layout.CardRuleTop, 0, 0, 0).Background(ovCardRule)
	live := s.State == meeting.StateRecording || s.State == meeting.StatePaused
	if a.meetOvMenu != meetingMenuNone && live {
		a.meetingOverlayMenu(c, s)
	} else {
		a.meetingOverlayTranscript(c, s)
	}
	if next, ok := sizeStepButton(c.Key("mt-size"), a.prefs.Features.CardSize(meetingCardSize)); ok {
		a.setCardSize(meetingCardSize, next)
		a.syncMeetingOverlay()
	}
}

// meetingOverlayTranscript is the live transcript: a List builds only the
// rows in view and follows the newest while it shows the end (#3985).
// Keyed by segment, a user reading stays on their line as the snapshot
// drops its oldest ones. One row says Voxt is listening while there is
// nothing yet.
func (a *app) meetingOverlayTranscript(c *ui.Context, s meeting.Snapshot) {
	rows := meetingOverlayRows(s)
	n := max(len(rows), 1)
	a.meetOvList.FollowEnd = true
	ui.List(c.Key("mt-live"), &a.meetOvList, n).Grow(1).MinHeight(0).Margin(layout.CardRuleGap, 0, 0, 0).Gap(meetingOvRowGap).Padding(0, layout.CardInset, 0, 0).
		ItemKey(func(i int) any {
			if i < len(rows) {
				return rows[i].Key
			}
			return nil
		}).Rows(func(r ui.ListRow) {
		if r.Index >= len(rows) {
			ui.Text(c, i18n.T("Listening…")).FontSize(layout.CardTextSize).TextColor(ovText2)
			return
		}
		meetingOverlayLine(c, rows[r.Index])
	})
}

// meetingOverlayLine is one row of the live transcript, the speaker's
// name over the first of theirs; the sentence still heard is dimmer, as
// dictation's live text, upright (CJK has no true italic).
func meetingOverlayLine(c *ui.Context, r meetingOverlayRow) {
	ui.Column(c).Gap(2).Shrink(0).Children(func() {
		if r.First {
			whoColor := ovText2
			if r.Me {
				whoColor = ovText
			}
			ui.Text(c, r.Who).FontSize(11).FontWeight(600).TextColor(whoColor)
		}
		color := ovText
		if r.Live {
			color = ovLiveColor
		}
		ui.Text(c, r.Text).FontSize(layout.CardTextSize).FontWeight(layout.CardTextWeight).TextColor(color)
	})
}

// meetingOverlayMenu is the menu the header opened, over the transcript:
// the modes with what each records, or More's actions.
func (a *app) meetingOverlayMenu(c *ui.Context, s meeting.Snapshot) {
	ui.Column(c.Key("mt-menu")).Gap(4).Shrink(0).Margin(layout.CardRuleGap, 0, 0, 0).Children(func() {
		if a.meetOvMenu == meetingMenuMode {
			for _, m := range []meeting.Mode{meeting.ModeMeeting, meeting.ModeSubtitles, meeting.ModeRecording} {
				if meetingMenuRow(c.Key("mt-mode-"+string(m)), meetingModeTitle(m), meetingModeHint(m), s.Mode == m, false) {
					a.meetOvMenu = meetingMenuNone
					if s.Mode != m {
						a.meetingSetMode(m)
					}
					a.syncMeetingOverlay()
				}
			}
			return
		}
		for _, it := range meetingMoreItems(s) {
			switch it {
			case moreTranslate:
				label := i18n.T("Live translation")
				if s.Translate != "" {
					label = i18n.T("Translating to %s", uitext.TargetLanguageLabel(s.Translate))
				}
				if meetingMenuRow(c.Key("mt-translate"), label, "", s.Translate != "", false) {
					a.meetOvMenu = meetingMenuNone
					a.toggleMeetingTranslation(s.ID)
					a.syncMeetingOverlay()
				}
			case moreCopy:
				if meetingMenuRow(c.Key("mt-copy"), i18n.T("Copy transcript"), "", false, false) {
					a.meetOvMenu = meetingMenuNone
					a.copyLiveMeetingExport()
					a.toast(i18n.T("Copied"))
					a.syncMeetingOverlay()
				}
			case moreCancel:
				if meetingMenuRow(c.Key("mt-abandon"), i18n.T("Cancel meeting"), "", false, true) {
					a.meetOvMenu = meetingMenuNone
					a.meetingAbandon = true
					a.syncMeetingOverlay()
				}
			}
		}
	})
}

// meetingMenuRow is one item of the overlay's menus: its name, a hint
// under it, a check when on; danger is red. It reports a click.
func meetingMenuRow(c *ui.Context, label, hint string, on, danger bool) bool {
	h := float32(30)
	if hint != "" {
		h = meetingOvMenuRowH
	}
	b := ui.ButtonBase(c).Height(h).Padding(0, 10).Radius(8).Shrink(0).Label(label).Checked(on)
	if b.Hovered() {
		b.Background(ui.RGBA(255, 255, 255, 0.1))
	}
	text := ovText
	if danger {
		text = mtRecDot
	}
	return b.Children(func() {
		ui.Row(c).Fill().Gap(8).AlignItems(ui.Center).Children(func() {
			ui.Column(c).Grow(1).MinWidth(0).Gap(1).Children(func() {
				ui.Text(c, label).FontSize(13).FontWeight(500).TextColor(text).SingleLine().Ellipsis("…")
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

func meetingModeHint(m meeting.Mode) string {
	switch m {
	case meeting.ModeSubtitles:
		return i18n.T("System audio only")
	case meeting.ModeRecording:
		return i18n.T("Microphone only")
	default:
		return i18n.T("Microphone and system audio")
	}
}

// meetingSpeaker is who said a line, as the meeting's details say it;
// a line without a speaker is You from the microphone, else Others.
func meetingSpeaker(name string, src meeting.Source) string {
	if name == "" && src == meeting.SourceMic {
		return i18n.T("You")
	}
	return meetingSpeakerText(name, nil)
}

func (a *app) meetingOverlayAsking(c *ui.Context) {
	ui.Row(c).Height(layout.RowH).Shrink(0).Gap(8).AlignItems(ui.Center).Children(func() {
		ui.Text(c, i18n.T("Save this meeting?")).FontSize(layout.NoticeSize).FontWeight(500).TextColor(ovText).SingleLine().Grow(1).MinWidth(0)
		if cardCapsule(c.Key("mt-save"), i18n.T("Save"), "") {
			a.meetingSave()
		}
		if cardCapsule(c.Key("mt-discard"), i18n.T("Discard"), "") {
			a.meetingDiscard()
		}
	})
}

// meetingOverlayFinishing is the capsule while the meeting is saved,
// transcribed again and summarised.
func meetingOverlayFinishing(c *ui.Context, s meeting.Snapshot, look pillLook) {
	msg := i18n.T("Finishing the meeting…")
	switch s.Stage {
	case meeting.StageTranscribing:
		msg = i18n.T("Finalising transcript…")
		if s.Progress > 0 {
			msg += fmt.Sprintf(" %.0f%%", s.Progress*100)
		}
	case meeting.StageSummary:
		msg = i18n.T("Summarising…")
	}
	ui.Row(c).Height(layout.RowH).Shrink(0).Gap(8).AlignItems(ui.Center).Justify(ui.Center).Children(func() {
		overlayLoader(c, msg, look, 0)
		ui.Text(c, msg).FontSize(layout.NoticeSize).TextColor(ovText).SingleLine().Ellipsis("…").MinWidth(0).Shrink(1)
	})
}

func (a *app) meetingAbandonDialog(c *ui.Context) {
	t := c.Theme()
	ui.Modal(c.Key("mt-abandon-ask"), &a.meetingAbandon, func() {
		ui.Text(c, i18n.T("Cancel this meeting?")).Bold()
		ui.Text(c, i18n.T("The recording and any unsaved text are deleted. This cannot be undone.")).Width(px(t, 380))
		ui.Row(c).Gap(px(t, sp2)).Justify(ui.End).Children(func() {
			if ui.Button(c.Key("mt-abandon-no"), i18n.T("Keep recording")).AutoFocus().Clicked() {
				a.meetingAbandon = false
			}
			if recordButton(c.Key("mt-abandon-yes"), i18n.T("Cancel meeting")).Clicked() {
				a.meetingAbandon = false
				a.meetingDiscard()
			}
		})
	})
}

// formatMeetingElapsed is the meeting's time as the dictation pill's
// (clock): m:ss, h:mm:ss from an hour.
func formatMeetingElapsed(d time.Duration) string {
	d = d.Round(time.Second)
	if d < time.Hour {
		return clock(d)
	}
	return fmt.Sprintf("%d:%02d:%02d", int(d.Hours()), int(d.Minutes())%60, int(d.Seconds())%60)
}

func (a *app) destroyMeetingOverlay() {
	if a.meetOvWin != nil {
		a.meetOvWin.Destroy()
		a.meetOvWin = nil
		a.meetOvShown = false
		a.meetOvCollapsed = false
	}
}
