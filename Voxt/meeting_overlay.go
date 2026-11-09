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
	"github.com/sgl-project/sglang-omni/Voxt/internal/session"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
	"github.com/sgl-project/sglang-omni/Voxt/internal/uitext"
)

type meetingOverlayState struct {
	meetOvWin       *mygo.Window
	meetOvShown     bool
	meetOvFrame     mygo.Rectangle
	meetOvCollapsed bool
}

// meetingContentProtectionHook is set in tests to observe SetContentProtection.
var meetingContentProtectionHook func(bool)

func (a *app) meetingOverlayCollapsed() bool {
	if a.meetOvCollapsed {
		return true
	}
	return a.prefs.Features.Meeting.Collapsed
}

func (a *app) syncMeetingOverlay() {
	if !a.overlayNative() || !a.meetingEnabled() {
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
	model := overlayModel(a.meetSnap, a.meetingOverlayCollapsed())
	w, h := model.Layout.Width, model.Layout.Height
	if a.meetOvWin == nil {
		opts := floatingWindowOptions("Voxt meeting", w, h, hudContent(a.meetingOverlayView))
		opts.Vibrancy = mygo.VibrancyHUD
		opts.StateKey = "meeting-overlay"
		a.meetOvWin = mygo.NewWindow(opts)
		a.meetOvWin.SetMovable(true)
	}
	area := a.overlayArea()
	x := area.X + (area.Width-w)/2
	y := area.Y + area.Height - h - 28
	if a.meetOvFrame.Width > 0 {
		x, y = a.meetOvFrame.X, a.meetOvFrame.Y
	}
	frame := mygo.Rectangle{X: x, Y: y, Width: w, Height: h}
	a.meetOvFrame = frame
	a.meetOvWin.SetContentBounds(frame)
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
	model := overlayModel(s, a.meetingOverlayCollapsed())
	motion := a.overlayMotion()
	if !motion {
		c.DisableAnimations()
	} else if model.ShowFinishing {
		c.After(layout.Tick)
	}
	mode, faceH := meetingOverlayFace(model)
	ui.Column(c).Fill().Padding(layout.Pad).Children(func() {
		faceStyle(ui.Column(c.Key("mt-face")), a.prefs.Overlay, mode, faceH).Padding(0, 14-layout.Border).Children(func() {
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

func (a *app) meetingOverlayBar(c *ui.Context, s meeting.Snapshot, model meetingOverlayModel) {
	live := s.State == meeting.StateRecording || s.State == meeting.StatePaused
	ui.Row(c).Height(meetingOvBarH).Shrink(0).Gap(10).AlignItems(ui.Center).Children(func() {
		dot, status := mtRecDot, i18n.T("Meeting in progress")
		if s.State == meeting.StatePaused {
			dot, status = ovText2, i18n.T("Meeting paused")
		}
		ui.Box(c).Size(8, 8).Radius(4).Background(dot).Shrink(0).Label(status)
		ui.Text(c, formatMeetingElapsed(s.Elapsed)).FontSize(13).Font("monospace").TextColor(ovText).SingleLine().Shrink(0)
		if s.State == meeting.StatePaused {
			ui.Text(c, i18n.T("Meeting paused")).FontSize(12).TextColor(ovText2).SingleLine().Shrink(0)
		} else {
			if s.Mode != meeting.ModeSubtitles {
				meetingLevel(c, icons.MicrophoneLarge, i18n.T("Microphone"), s.Level[meeting.SourceMic])
			}
			if s.Mode != meeting.ModeRecording {
				meetingLevel(c, icons.VolumeLoud, i18n.T("System audio"), s.Level[meeting.SourceSystem])
			}
		}
		ui.Box(c).Grow(1)
		if live {
			label, icon := i18n.T("Pause meeting"), icons.Pause
			if s.State == meeting.StatePaused {
				label, icon = i18n.T("Resume meeting"), icons.Play
			}
			if cardIconButton(c.Key("mt-pause"), label, label, icon, true) {
				a.meetingPauseResume()
			}
		}
		if live || s.State == meeting.StateStarting { // a start that hangs can be ended
			if cardIconButton(c.Key("mt-end"), i18n.T("End meeting"), i18n.T("End meeting"), icons.Stop, true) {
				a.meetingFinish()
			}
		}
		label, icon := i18n.T("Collapse"), icons.MinimizeSquareMinimalistic
		if model.Layout.Collapsed {
			label, icon = i18n.T("Expand"), icons.MaximizeSquareMinimalistic
		}
		if cardIconButton(c.Key("mt-ov-collapse"), label, label, icon, true) {
			next := !model.Layout.Collapsed
			a.meetOvCollapsed = next
			a.updateSettings(func(st *settings.Settings) { st.Features.Meeting.Collapsed = next })
			a.syncMeetingOverlay()
		}
	})
}

// meetingLevel is one track's input level: its icon and four bars that
// light up with it.
func meetingLevel(c *ui.Context, icon icons.Icon, label string, rms float64) {
	v := session.Level(rms)
	ui.Row(c).Gap(3).AlignItems(ui.Center).Shrink(0).Label(label).Children(func() {
		ui.Icon(c, icon.Linear).Size(13, 13).TextColor(ovText2).Shrink(0)
		for i := range 4 {
			alpha := float32(0.22)
			if v > float64(i)/4 {
				alpha = 0.92
			}
			ui.Box(c).Size(3, 5+float32(i)*2.5).Radius(1.5).Background(ui.RGBA(255, 255, 255, alpha)).Shrink(0)
		}
	})
}

// meetingOverlayProblem is the meeting's warning under the bar: two lines
// at most in the capsule.
func meetingOverlayProblem(c *ui.Context, key string, short bool) {
	msg := meetingOverlayProblemText(key)
	if msg == "" {
		return
	}
	ui.Row(c.Key("mt-problem")).Gap(6).AlignItems(ui.Start).Shrink(0).Padding(0, 0, 10, 0).Children(func() {
		ui.Icon(c, icons.DangerTriangle.Linear).Size(14, 14).TextColor(mtWarn).Shrink(0)
		text := ui.Text(c, msg).FontSize(12).TextColor(mtWarn).Grow(1).Shrink(1).MinWidth(0)
		if short {
			text.MaxLines(2).Ellipsis("…").Tooltip(msg)
		}
	})
}

func (a *app) meetingOverlayExpanded(c *ui.Context, s meeting.Snapshot) {
	if s.State == meeting.StateRecording || s.State == meeting.StatePaused {
		ui.Row(c).Gap(6).AlignItems(ui.Center).Shrink(0).Padding(0, 0, 10, 0).Children(func() {
			modes := []meeting.Mode{meeting.ModeMeeting, meeting.ModeSubtitles, meeting.ModeRecording}
			for i, label := range []string{i18n.T("Meeting"), i18n.T("Subtitles"), i18n.T("Recording")} {
				if meetingChip(c.Key("mt-mode-"+string(modes[i])), label, meetingModeHint(modes[i]), s.Mode == modes[i]) && s.Mode != modes[i] {
					a.meetingSetMode(modes[i])
				}
			}
			ui.Box(c).Grow(1)
			translate := i18n.T("Live translation")
			if s.Translate != "" {
				translate = i18n.T("Translating to %s", uitext.TargetLanguageLabel(s.Translate))
			}
			if meetingChip(c.Key("mt-translate"), translate, "", s.Translate != "") {
				a.toggleMeetingTranslation(s.ID)
			}
			if s.State == meeting.StatePaused && cardIconButton(c.Key("mt-copy"), i18n.T("Copy transcript"), i18n.T("Copy transcript"), icons.Copy, true) {
				a.copyLiveMeetingExport()
				a.toast(i18n.T("Copied"))
			}
			if cardIconButton(c.Key("mt-abandon"), i18n.T("Cancel meeting"), i18n.T("Cancel meeting"), icons.TrashBinMinimalistic, true) {
				a.meetingAbandon = true
			}
		})
	}
	ui.Scroll(c.Key("mt-live")).Grow(1).MinHeight(0).Padding(0, 0, 12, 0).Children(func() {
		ui.Column(c).Gap(8).Children(func() {
			if len(s.Segments) == 0 && s.Live[meeting.SourceMic] == "" && s.Live[meeting.SourceSystem] == "" {
				ui.Text(c, i18n.T("Listening…")).FontSize(13).TextColor(ovText2)
			}
			for _, seg := range s.Segments {
				meetingOverlayLine(c, meetingSpeaker(seg.Speaker, seg.Source), seg.Text, false)
			}
			for _, src := range []meeting.Source{meeting.SourceMic, meeting.SourceSystem} {
				if live := s.Live[src]; live != "" {
					meetingOverlayLine(c, meetingSpeaker("", src), live, true)
				}
			}
		})
	})
}

// meetingChip is a capsule that shows whether it is on; it reports a click.
func meetingChip(c *ui.Context, label, tip string, on bool) bool {
	b := ui.ButtonBase(c).Height(layout.CardBtn).Padding(0, 10-layout.Border).Radius(layout.CardBtn / 2).Center().Shrink(0).Label(label)
	if tip != "" {
		b.Tooltip(tip)
	}
	bg, line, text := ui.RGBA(255, 255, 255, 0.06), ui.RGBA(255, 255, 255, 0.08), ovText2
	switch {
	case on:
		bg, line, text = ui.RGBA(255, 255, 255, 0.2), ui.RGBA(255, 255, 255, 0.24), ovText
	case b.Hovered():
		bg, line = ui.RGBA(255, 255, 255, 0.14), ui.RGBA(255, 255, 255, 0.18)
	}
	b.Background(bg).Border(1, line)
	return b.Children(func() {
		ui.Text(c, label).FontSize(11).FontWeight(600).TextColor(text).SingleLine()
	}).Clicked()
}

// meetingOverlayLine is one line of the live transcript; live is the
// sentence still being heard.
func meetingOverlayLine(c *ui.Context, who, text string, live bool) {
	ui.Column(c).Gap(2).Shrink(0).Children(func() {
		ui.Text(c, who).FontSize(11).FontWeight(600).TextColor(ovText2)
		line := ui.Text(c, text).FontSize(13).TextColor(ovText)
		if live {
			line.Italic().TextColor(ovLiveColor)
		}
	})
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
// a line without a speaker is Me from the microphone, else Other.
func meetingSpeaker(name string, src meeting.Source) string {
	if name == "" && src == meeting.SourceMic {
		return i18n.T("Me")
	}
	return meetingSpeakerText(name, nil)
}

func (a *app) meetingOverlayAsking(c *ui.Context) {
	ui.Row(c).Height(meetingOvBarH).Shrink(0).Gap(8).AlignItems(ui.Center).Children(func() {
		ui.Text(c, i18n.T("Save this meeting?")).FontSize(13).FontWeight(500).TextColor(ovText).SingleLine().Grow(1).MinWidth(0)
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
		msg = i18n.T("Finalizing transcript…")
		if s.Progress > 0 {
			msg += fmt.Sprintf(" %.0f%%", s.Progress*100)
		}
	case meeting.StageSummary:
		msg = i18n.T("Summarizing…")
	}
	ui.Row(c).Height(meetingOvBarH).Shrink(0).Gap(4).AlignItems(ui.Center).Justify(ui.Center).Children(func() {
		overlayLoader(c, msg, look)
		ui.Text(c, msg).FontSize(13).TextColor(ovText).SingleLine().Ellipsis("…").MinWidth(0).Shrink(1)
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

func formatMeetingElapsed(d time.Duration) string {
	d = d.Round(time.Second)
	h := int(d.Hours())
	m := int(d.Minutes()) % 60
	sec := int(d.Seconds()) % 60
	if h > 0 {
		return fmt.Sprintf("%d:%02d:%02d", h, m, sec)
	}
	return fmt.Sprintf("%02d:%02d", m, sec)
}

func (a *app) destroyMeetingOverlay() {
	if a.meetOvWin != nil {
		a.meetOvWin.Destroy()
		a.meetOvWin = nil
		a.meetOvShown = false
		a.meetOvFrame = mygo.Rectangle{}
		a.meetOvCollapsed = false
	}
}
