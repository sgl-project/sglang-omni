package main

// The overlay's motion (overlay.go): the recording's waveform, the working
// loader and the recording's clock.

import (
	"time"

	"github.com/egoist/mygo/ui"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/overlay/layout"
)

// waveDraw is what a waveform's drawing paints from: the levels' bars
// (layout.WaveBars), or the gliding ones at a time (glide, layout.WavePlay),
// and the phase the view was built at, and when.
type waveDraw struct {
	still layout.Wave
	glide func(time.Time) layout.Wave
	phase float64
	built time.Time
}

func newWaveDraw(c *ui.Context, levels []float64, glide func(time.Time) layout.Wave, phase float64) waveDraw {
	return waveDraw{still: layout.WaveBars(levels), glide: glide, phase: phase, built: c.Now()}
}

// at are the bars and phase of the frame p paints. Gliding, they are the
// bars glide gives at its time and the phase moved on as far, and it asks
// for the next refresh's frame, which paints the drawing again without
// building the view: the bars move at the display's rate while the view
// builds no more often than without them (#4574 review).
func (w waveDraw) at(p *ui.Painter) (layout.Wave, float64) {
	if w.glide == nil {
		return w.still, w.phase
	}
	now := p.Now()
	p.AnimationFrame()
	return w.glide(now), w.phase + now.Sub(w.built).Seconds()
}

// waveform draws the input levels as 16 rounded bars, newest on the
// right, in the waveform's slot (layout.WaveSlotW, layout.RowH), centred (Voxt's):
// smoothed (layout.WaveBars, or gliding: waveDraw), and, where it is quiet,
// breathing a little about their least height while look moves. Dim until
// the first audio.
func waveform(c *ui.Context, levels []float64, heard bool, look pillLook) {
	wave := newWaveDraw(c, levels, look.wave, look.phase)
	// Voxt's fill: white, 98 % at the top to 80 % at the bottom.
	fill := ui.LinearGradient{From: ui.RGBA(255, 255, 255, 0.98), To: ui.RGBA(255, 255, 255, 0.8), Angle: 180}
	if !heard {
		fill.From, fill.To = ui.RGBA(255, 255, 255, 0.34), ui.RGBA(255, 255, 255, 0.28)
	}
	ui.Row(c.Key("ov-wave")).Width(layout.WaveSlotW).Height(layout.RowH).Shrink(0).Center().Children(func() {
		ui.Box(c).Width(layout.WaveW).Height(layout.RowH).Shrink(0).Label(i18n.T("Input level")).Draw(func(p *ui.Painter, r ui.Rect) {
			bars, phase := wave.at(p)
			for i, v := range bars {
				h := layout.WaveBarMinH + (layout.WaveBarMaxH-layout.WaveBarMinH)*float32(v)
				if look.motion {
					h += float32(layout.WaveIdle(i, v, phase))
				}
				bar := ui.Rect{X: r.X + float32(i)*(layout.WaveBarW+layout.WaveBarGap), Y: r.Y + (r.H-h)/2, W: layout.WaveBarW, H: h}
				if g := layout.WaveGlow(v); g > 0 && heard {
					p.Shadow(bar, layout.WaveBarR, 0, 0, 3, 0, ui.RGBA(255, 255, 255, float32(g)))
				}
				p.FillGradient(bar, fill, layout.WaveBarR)
			}
		})
	})
}

// overlayLoader is the working pill's loader in the waveform's place,
// centred in its slot: five dots of 5, 4 apart, that swell and brighten
// in turn (Voxt's: 1.2 s, 0.12 s apart); still, all half bright, without
// motion. w is the slot's width, 0 for just the dots.
func overlayLoader(c *ui.Context, label string, look pillLook, w float32) {
	row := ui.Row(c.Key("ov-loader")).Height(layout.RowH).Shrink(0).Gap(4).AlignItems(ui.Center).Justify(ui.Center)
	if w > 0 {
		row.Width(w)
	}
	row.Label(label).Children(func() {
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
