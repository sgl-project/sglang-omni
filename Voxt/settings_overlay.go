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
	"fmt"
	"time"

	"github.com/egoist/mygo/ui"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
)

// overlayForm is the section's controls.
type overlayForm struct {
	position                int // 0 bottom, 1 top
	opacity, radius, margin float64
	showTime                bool
	liveText                bool
	liveLayout              int // 0 one line, 1 lines (Overlay.LiveLines)
	// saveAt is when the last change waits to be saved (zero: saved).
	saveAt time.Time
}

// Bounds of the sliders (settings.normalize keeps wider ones).
const (
	ovMaxRadius = 26 // the pill's half height: rounder changes nothing
	ovMaxMargin = 240
)

// overlaySettings is the Overlay style section of the settings dialog.
func overlaySettings(c *ui.Context, state settingsOverlayViewState, actions settingsOverlayActions, local *overlayForm) {
	t := c.Theme()
	slider := func(key, label string, value *float64, lo, hi, step float64, text string) {
		ui.Row(c).Gap(px(t, sp3)).AlignItems(ui.Center).Children(func() {
			if ui.Slider(c.Key(key), value, lo, hi).Step(step).Label(label).Width(px(t, 160)).Shrink(1).MinWidth(px(t, 100)).Changed() {
				actions.overlayEdited()
			}
			ui.Text(c, text).FontSize(px(t, fsCaption)).TextColor(tok(t).Text2).Font("monospace").Width(px(t, 40)).Shrink(0).SingleLine()
		})
	}
	settingRow(c, "ov-position", i18n.T("Position"), i18n.T("On the display under the pointer, centred."), func() {
		if ui.Segmented(c.Key("ov-position-seg"), &local.position, state.PositionOptions...).Label(i18n.T("Position")).Changed() {
			actions.overlayEdited()
		}
	})
	settingRow(c, "ov-margin", i18n.T("Distance from the edge"), "", func() {
		slider("ov-margin-slider", i18n.T("Distance from the edge"), &local.margin, 0, ovMaxMargin, 4, fmt.Sprintf("%.0f", local.margin))
	})
	settingRow(c, "ov-opacity", i18n.T("Opacity"), "", func() {
		slider("ov-opacity-slider", i18n.T("Opacity"), &local.opacity, 0.5, 1, 0.05, fmt.Sprintf("%.0f%%", local.opacity*100))
	})
	settingRow(c, "ov-radius", i18n.T("Corner radius"), "", func() {
		slider("ov-radius-slider", i18n.T("Corner radius"), &local.radius, 0, ovMaxRadius, 1, fmt.Sprintf("%.0f", local.radius))
	})
	settingRow(c, "ov-time", i18n.T("Show recording time"), "", func() {
		if ui.Switch(c.Key("ov-time-switch"), &local.showTime).Label(i18n.T("Show recording time")).Changed() {
			actions.overlayEdited()
		}
	})
	settingRow(c, "ov-live", i18n.T("Show live text"), "", func() {
		if ui.Switch(c.Key("ov-live-switch"), &local.liveText).Label(i18n.T("Show live text")).Changed() {
			actions.overlayEdited()
		}
	})
	settingRow(c, "ov-lines", i18n.T("Live text layout"), "", func() {
		if ui.Segmented(c.Key("ov-lines-seg"), &local.liveLayout, state.LiveLayoutOptions...).Label(i18n.T("Live text layout")).Disabled(!local.liveText).Changed() {
			actions.overlayEdited()
		}
	})

	sectionTitle(c, i18n.T("Preview"))
	local.previewState().overlayPreview(c)
}
