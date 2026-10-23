package main

// The overlay's motion (overlay.go): the recording's waveform, the working
// loader and the recording's clock.

import (
	"time"

	"github.com/egoist/mygo/ui"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/i18n"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/overlay/layout"
)

// waveform draws the input levels as 16 rounded bars, newest on the
// right, in the waveform's slot (layout.WaveSlotW, layout.RowH), centred (Voxt's):
// smoothed (layout.WaveBars), and, where it is quiet, breathing a little about
// their least height while look moves. Dim until the first audio.
func waveform(c *ui.Context, levels []float64, heard bool, look pillLook) {
	bars := layout.WaveBars(levels)
	ui.Row(c.Key("ov-wave")).Width(layout.WaveSlotW).Height(layout.RowH).Shrink(0).Center().Children(func() {
		ui.Row(c).Width(layout.WaveW).Height(layout.RowH).Shrink(0).Gap(layout.WaveBarGap).AlignItems(ui.Center).Label(i18n.T("Input level")).Children(func() {
			for i, v := range bars {
				h := layout.WaveBarMinH + (layout.WaveBarMaxH-layout.WaveBarMinH)*float32(v)
				if look.motion {
					h += float32(layout.WaveIdle(i, v, look.phase))
				}
				// Voxt's fill: white, 98 % at the top to 80 % at the bottom.
				top, bottom := float32(0.98), float32(0.8)
				if !heard {
					top, bottom = 0.34, 0.28
				}
				bar := ui.Box(c).Width(layout.WaveBarW).Height(h).Radius(layout.WaveBarR).Gradient(ui.RGBA(255, 255, 255, top), ui.RGBA(255, 255, 255, bottom), 180).Shrink(0)
				if g := layout.WaveGlow(v); g > 0 && heard {
					bar.Shadow(0, 0, 3, 0, ui.RGBA(255, 255, 255, float32(g)))
				}
			}
		})
	})
}

// overlayLoader is the working pill's loader in the waveform's place,
// centred in its slot: five dots of 5, 4 apart, that swell and brighten
// in turn (Voxt's: 1.2 s, 0.12 s apart); still, all half bright, without
// motion.
func overlayLoader(c *ui.Context, label string, look pillLook) {
	ui.Row(c.Key("ov-loader")).Width(layout.WaveSlotW).Height(layout.RowH).Shrink(0).Gap(4).AlignItems(ui.Center).Justify(ui.Center).
		Label(label).Children(func() {
		for i := range 5 {
			scale, alpha := 1.0, 0.7
			if look.motion {
				scale, alpha = layout.LoaderDot(look.phase - float64(i)*0.12)
			}
			d := float32(5 * scale)
			ui.Box(c).Size(d, d).Radius(d / 2).Background(ui.RGBA(255, 255, 255, float32(alpha))).Shrink(0)
		}
	})
}

// clock shows a duration as m:ss.
func clock(d time.Duration) string {
	sec := int(d.Seconds())
	return itoa(sec/60) + ":" + string(rune('0'+sec%60/10)) + string(rune('0'+sec%10))
}
