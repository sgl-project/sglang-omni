package main

// The meeting overlay: timer, levels, live transcript, controls, and
// warnings while a meeting is active (design §4.12).

import (
	"fmt"
	"time"

	"github.com/egoist/mygo"
	"github.com/egoist/mygo/ui"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/meeting"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
	"github.com/sgl-project/sglang-omni/Voxt/internal/uitext"
)

type meetingOverlayState struct {
	meetOvWin      *mygo.Window
	meetOvShown    bool
	meetOvFrame    mygo.Rectangle
	meetOvCollapsed bool
	meetOvModeUI   string
	meetOvTranslate bool
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
	t := c.Theme()
	s := a.meetSnap
	model := overlayModel(s, a.meetingOverlayCollapsed())
	if model.ShowAsking {
		a.meetingOverlayAsking(c, s)
		return
	}
	ui.Column(c).Gap(px(t, 6)).Children(func() {
		a.meetingOverlayBar(c, s, model)
		if model.ShowExpanded {
			a.meetingOverlayExpanded(c, s, model)
		}
	})
	a.meetingAbandonDialog(c)
}

func (a *app) meetingOverlayBar(c *ui.Context, s meeting.Snapshot, model meetingOverlayModel) {
	t := c.Theme()
	ui.Row(c).Padding(px(t, 10), px(t, 8)).AlignItems(ui.Center).Gap(px(t, 8)).Children(func() {
		ui.Text(c, formatMeetingElapsed(s.Elapsed))
		ui.Text(c, fmt.Sprintf("mic %.0f · sys %.0f", s.Level[meeting.SourceMic], s.Level[meeting.SourceSystem])).TextColor(tok(t).Text2)
		if msg := meetingOverlayProblemText(model.ProblemKey); msg != "" {
			ui.Text(c, msg).FontSize(px(t, fsCaption)).TextColor(tok(t).Danger).Grow(1)
		} else {
			ui.Box(c).Grow(1)
		}
		if model.ShowBar {
			collapse := i18n.T("Expand")
			if !model.Layout.Collapsed {
				collapse = i18n.T("Collapse")
			}
			if ghostButton(c.Key("mt-ov-collapse"), collapse).Clicked() {
				next := !model.Layout.Collapsed
				a.meetOvCollapsed = next
				a.updateSettings(func(st *settings.Settings) { st.Features.Meeting.Collapsed = next })
				a.syncMeetingOverlay()
			}
		}
		if s.State == meeting.StateFinishing && s.Stage == meeting.StageTranscribing {
			ui.Text(c, i18n.T("Finalizing transcript…")).TextColor(tok(t).Text2)
			if s.Progress > 0 {
				ui.Text(c, fmt.Sprintf("%.0f%%", s.Progress*100)).TextColor(tok(t).Text2)
			}
			return
		}
		if s.State == meeting.StateRecording || s.State == meeting.StatePaused {
			pause := i18n.T("Pause")
			if s.State == meeting.StatePaused {
				pause = i18n.T("Resume")
			}
			if ghostButton(c.Key("mt-pause"), pause).Clicked() {
				a.meetingPauseResume()
			}
			if s.State == meeting.StatePaused {
				if ghostButton(c.Key("mt-xlate"), i18n.T("Copy transcript")).Clicked() {
					a.copyLiveMeetingExport()
				}
			}
			toggle := i18n.T("Translation off")
			if s.Translate != "" {
				toggle = i18n.T("Translation on")
			}
			if ghostButton(c.Key("mt-translate"), toggle).Clicked() {
				a.toggleMeetingTranslation(s.ID)
			}
			if ghostButton(c.Key("mt-end"), i18n.T("End")).Clicked() {
				a.meetingFinish()
			}
			if ghostButton(c.Key("mt-abandon"), i18n.T("Cancel meeting")).Clicked() {
				a.meetingAbandon = true
			}
		}
	})
}

func (a *app) meetingOverlayExpanded(c *ui.Context, s meeting.Snapshot, model meetingOverlayModel) {
	t := c.Theme()
	ui.Column(c).Padding(px(t, 10), px(t, 8)).Gap(px(t, 8)).MaxHeight(px(t, float32(model.Layout.Height-meetingOvCollapsedH))).Children(func() {
		if model.ShowFinishing {
			msg := i18n.T("Finishing the meeting…")
			if s.Stage == meeting.StageTranscribing {
				msg = i18n.T("Finalizing transcript…")
				if s.Progress > 0 {
					msg += fmt.Sprintf(" %.0f%%", s.Progress*100)
				}
			}
			ui.Text(c, msg).TextColor(tok(t).Text2)
			return
		}
		a.meetingOverlayModeRow(c, s)
		a.meetingOverlayTranslateRow(c, s)
		ui.Scroll(c.Key("mt-live")).Grow(1).MinHeight(px(t, 120)).Children(func() {
			ui.Column(c).Gap(px(t, 6)).Children(func() {
				for _, seg := range s.Segments {
					line := meetingSegmentLine(seg)
					color := tok(t).Text
					if seg.Source == meeting.SourceSystem {
						color = tok(t).Text2
					}
					ui.Text(c, line).TextColor(color)
				}
				for src, live := range s.Live {
					if live == "" {
						continue
					}
					ui.Text(c, meetingLiveLine(src, live)).Italic().TextColor(tok(t).Text2)
				}
			})
		})
	})
}

func (a *app) meetingOverlayModeRow(c *ui.Context, s meeting.Snapshot) {
	if s.State != meeting.StateRecording && s.State != meeting.StatePaused {
		return
	}
	t := c.Theme()
	labels := []string{
		i18n.T("Meeting"),
		i18n.T("Subtitles"),
		i18n.T("Recording"),
	}
	modes := []meeting.Mode{meeting.ModeMeeting, meeting.ModeSubtitles, meeting.ModeRecording}
	sel := labels[0]
	for i, m := range modes {
		if s.Mode == m {
			sel = labels[i]
			break
		}
	}
	a.meetOvModeUI = sel
	if modePick := ui.Select(c.Key("mt-ov-mode"), &a.meetOvModeUI, labels).Label(i18n.T("Mode")); modePick.Changed() {
		for i, l := range labels {
			if l == a.meetOvModeUI {
				a.meetingSetMode(modes[i])
				break
			}
		}
	}
	ui.Text(c, meetingModeHint(s.Mode)).FontSize(px(t, fsCaption)).TextColor(t.TextMuted)
}

func (a *app) meetingOverlayTranslateRow(c *ui.Context, s meeting.Snapshot) {
	if s.State != meeting.StateRecording && s.State != meeting.StatePaused {
		return
	}
	t := c.Theme()
	on := s.Translate != ""
	a.meetOvTranslate = on
	if ui.Switch(c.Key("mt-ov-translate"), &a.meetOvTranslate).Label(i18n.T("Live translation")).Changed() {
		if a.meetOvTranslate {
			target := a.prefs.Features.Meeting.TranslateTarget
			if target == "" {
				target = "en"
			}
			a.meetingSetTranslate(target)
		} else {
			a.meetingSetTranslate("")
		}
	}
	if on {
		target := s.Translate
		if target == "" {
			target = "en"
		}
		ui.Text(c, i18n.T("Translating to %s", uitext.TargetLanguageLabel(target))).
			FontSize(px(t, fsCaption)).TextColor(t.TextMuted)
	}
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

func meetingSegmentLine(seg meeting.Segment) string {
	who := seg.Speaker
	if who == "" {
		if seg.Source == meeting.SourceMic {
			who = i18n.T("Me")
		} else {
			who = i18n.T("Other")
		}
	}
	return fmt.Sprintf("%s: %s", who, seg.Text)
}

func meetingLiveLine(src meeting.Source, text string) string {
	if src == meeting.SourceMic {
		return fmt.Sprintf("%s: %s", i18n.T("Me"), text)
	}
	return fmt.Sprintf("%s: %s", i18n.T("Other"), text)
}

func (a *app) meetingOverlayAsking(c *ui.Context, s meeting.Snapshot) {
	t := c.Theme()
	ui.Column(c).Padding(px(t, 12)).Gap(px(t, 8)).Children(func() {
		ui.Text(c, i18n.T("Save this meeting?")).Bold()
		ui.Row(c).Gap(px(t, sp2)).Children(func() {
			if primaryButton(c.Key("mt-save"), i18n.T("Save")).Clicked() {
				a.meetingSave()
			}
			if ghostButton(c.Key("mt-discard"), i18n.T("Discard")).Clicked() {
				a.meetingDiscard()
			}
		})
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
