package main

// Settings → Overlay style (/settings/overlay): where the recording
// overlay shows (the bottom or the top of the display, how far from its
// edge), how opaque its face is, how round its corners are and whether it
// shows the recording's time (settings.Overlay), with a preview: the
// overlay's own pill, drawn as overlay.go draws it, over a desktop picture
// that shows through it blurred, as the desktop does through the real one,
// where the settings put it.
// A change moves the overlay at once when it shows; it is saved once the
// sliders rest.

import (
	_ "embed"
	"fmt"
	"math"
	"sync"
	"time"

	"github.com/egoist/mygo/plugins/glass"
	"github.com/egoist/mygo/ui"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/i18n"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/overlay/layout"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/session"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/settings"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/store"
)

// overlayForm is the section's controls.
type overlayForm struct {
	position                int // 0 bottom, 1 top
	opacity, radius, margin float64
	showTime                bool
	// saveAt is when the last change waits to be saved (zero: saved).
	saveAt time.Time
}

// Bounds of the sliders (settings.normalize keeps wider ones).
const (
	ovMaxRadius = 26 // the pill's half height: rounder changes nothing
	ovMaxMargin = 240
	ovSaveDelay = 400 * time.Millisecond
)

// overlayPositions are the values of Overlay.Position, by segment.
var overlayPositions = []string{"bottom", "top"}

// loadOverlayForm sets the controls from the settings, unless a change
// waits to be saved (the sliders move).
func (a *app) loadOverlayForm() {
	f := &a.settingsForm.ovStyle
	if !f.saveAt.IsZero() {
		return
	}
	o := a.prefs.Overlay
	f.position = 0
	if o.Position == "top" {
		f.position = 1
	}
	f.opacity, f.radius, f.margin, f.showTime = o.Opacity, o.Radius, o.Margin, o.ShowTime
}

// overlayEdited applies the controls: at once to the overlay and the
// preview, to the settings after ovSaveDelay without another change.
func (a *app) overlayEdited() {
	f := &a.settingsForm.ovStyle
	o := settings.Overlay{
		Position: overlayPositions[min(max(f.position, 0), 1)],
		Opacity:  math.Round(f.opacity*100) / 100,
		Radius:   math.Round(f.radius),
		Margin:   math.Round(f.margin),
		ShowTime: f.showTime,
	}
	a.prefs.Overlay = o
	a.overlayRestyle() // overlay.go
	if a.win == nil {
		a.updateSettings(func(s *settings.Settings) { s.Overlay = o })
		return
	}
	at := time.Now()
	f.saveAt = at
	time.AfterFunc(ovSaveDelay, func() {
		a.post(func() {
			if f.saveAt != at {
				return // a later change saves
			}
			f.saveAt = time.Time{}
			o := a.prefs.Overlay
			a.updateSettings(func(s *settings.Settings) { s.Overlay = o })
		})
	})
}

// overlaySettings is the Overlay style section of the settings dialog.
func (a *app) overlaySettings(c *ui.Context) {
	t := c.Theme()
	f := &a.settingsForm.ovStyle
	slider := func(key, label string, value *float64, lo, hi, step float64, text string) {
		ui.Row(c).Gap(px(t, sp3)).AlignItems(ui.Center).Children(func() {
			if ui.Slider(c.Key(key), value, lo, hi).Step(step).Label(label).Width(px(t, 160)).Shrink(1).MinWidth(px(t, 100)).Changed() {
				a.overlayEdited()
			}
			ui.Text(c, text).FontSize(px(t, fsCaption)).TextColor(tok(t).Text2).Font("monospace").Width(px(t, 40)).Shrink(0).SingleLine()
		})
	}
	settingRow(c, "ov-position", i18n.T("Position"), i18n.T("On the display under the pointer, centred."), func() {
		if ui.Segmented(c.Key("ov-position-seg"), &f.position, i18n.T("Bottom centre"), i18n.T("Top centre")).Label(i18n.T("Position")).Changed() {
			a.overlayEdited()
		}
	})
	settingRow(c, "ov-margin", i18n.T("Distance from the edge"), i18n.T("From the bottom or the top of the screen, above the Dock and below the menu bar."), func() {
		slider("ov-margin-slider", i18n.T("Distance from the edge"), &f.margin, 0, ovMaxMargin, 4, fmt.Sprintf("%.0f", f.margin))
	})
	settingRow(c, "ov-opacity", i18n.T("Opacity"), i18n.T("Of the dark face over the blurred desktop: lower lets more of what is behind show through."), func() {
		slider("ov-opacity-slider", i18n.T("Opacity"), &f.opacity, 0.5, 1, 0.05, fmt.Sprintf("%.0f%%", f.opacity*100))
	})
	settingRow(c, "ov-radius", i18n.T("Corner radius"), i18n.T("The answer card takes the same corners."), func() {
		slider("ov-radius-slider", i18n.T("Corner radius"), &f.radius, 0, ovMaxRadius, 1, fmt.Sprintf("%.0f", f.radius))
	})
	settingRow(c, "ov-time", i18n.T("Show the recording time"), i18n.T("Off, the overlay shows only the feature's icon and the waveform."), func() {
		if ui.Switch(c.Key("ov-time-switch"), &f.showTime).Label(i18n.T("Show the recording time")).Changed() {
			a.overlayEdited()
		}
	})

	sectionTitle(c, i18n.T("Preview"))
	a.overlayPreview(c)
}

// previewSession is the recording the preview shows: Transcribe, seven
// seconds in, a few words on the waveform (bars 4 to 20 high, as speech
// moves them).
func previewSession() session.Snapshot {
	t0 := time.Date(2026, 1, 1, 9, 0, 0, 0, time.UTC)
	levels := []float64{0.22, 0.38, 0.52, 0.66, 0.3, 0.2, 0.36, 0.6, 0.76, 0.42, 0.22, 0.4, 0.58, 0.32, 0.24, 0.48}
	return session.Snapshot{Kind: store.KindTranscribe, State: session.Recording, StartedAt: t0, HeardAt: t0, StoppedAt: t0.Add(7 * time.Second), Levels: levels}
}

// The preview's background: a desktop picture (Voxt's, Apache-2.0: see
// NOTICE), decoded once.
//
//go:embed assets/overlay-preview-background.png
var ovPreviewPNG []byte

var ovPreviewBackground = sync.OnceValue(func() *ui.Bitmap {
	b, _ := ui.DecodeBitmap(ovPreviewPNG)
	return b
})

// The preview card: as wide as the section, ovPreviewH high, its corners
// ovPreviewRadius (Voxt's); the pill at least ovPreviewInset from its
// top or bottom edge.
const (
	ovPreviewH      = 150
	ovPreviewRadius = 18
	ovPreviewInset  = 12
)

// overlayPreview is the overlay's pill as it draws, at its size, in its
// dark theme, over a desktop picture that shows through it blurred (the
// desktop does, on the Mac), after Voxt's preview: where the settings put
// it (the bottom or the top, the distance from the edge to scale with the
// card), as opaque, as round, with the time when it shows it. A change
// shows at once.
func (a *app) overlayPreview(c *ui.Context) {
	t := c.Theme()
	o := a.prefs.Overlay
	w := layout.RowWidth(o.ShowTime, 0) + 2*layout.PillPadX
	cardH := px(t, ovPreviewH)
	// The distance from the edge, from none to ovMaxMargin, across the
	// card's room.
	room := max(cardH-layout.PillH-2*px(t, ovPreviewInset), 0)
	off := px(t, ovPreviewInset) + room*float32(min(max(o.Margin, 0), ovMaxMargin)/ovMaxMargin)
	radius := faceRadius(o, ovRecording, layout.PillH)
	ui.Box(c.Key("ov-preview")).FillWidth().MinWidth(0).Height(cardH).Shrink(0).Radius(px(t, ovPreviewRadius)).Clip().
		Background(ui.RGB(40, 90, 170)).Label(i18n.T("Preview of the overlay")).Children(func() {
		if bg := ovPreviewBackground(); bg != nil {
			ui.Image(c, bg).Absolute().Left(0).Top(0).Right(0).Bottom(0).Fit(ui.Cover)
		}
		place := ui.Row(c).Absolute().Left(0).Right(0).Height(layout.PillH).Justify(ui.Center)
		if o.Position == "top" {
			place.Top(off)
		} else {
			place.Bottom(off)
		}
		place.Children(func() {
			c.SetTheme(overlayTheme())
			ui.Box(c.Key("ov-preview-pill")).Size(float32(w), layout.PillH).Shrink(0).Children(func() {
				// The blur of what is under it, then the face, positioned
				// too to paint over the blur.
				ui.Box(c).Absolute().Fill().Radius(radius).Material(glass.Blur{Radius: 8})
				face := ui.Column(c.Key("ov-preview-face")).Absolute().Left(0).Top(0).Right(0).Bottom(0)
				faceStyle(face, o, ovRecording, layout.PillH).Children(func() {
					overlayPill(c, previewSession(), ovRecording, pillLook{time: o.ShowTime, w: w})
				})
			})
			c.SetTheme(t)
		})
	})
}
