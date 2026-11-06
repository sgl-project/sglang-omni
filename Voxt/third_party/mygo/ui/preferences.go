package ui

// Preferences are settings of the desktop that the system's own controls
// follow. The default theme follows them, and Animate follows
// ReduceMotion; a theme of your own (Context.SetTheme) and drawing of your
// own may read them with Context.Preferences.
type Preferences struct {
	// Accent is the accent color the user chose, which the default theme
	// takes; A is 0 where the desktop has none.
	Accent Color
	// ReduceMotion asks for less motion: macOS's Reduce Motion, Windows's
	// animation effects and GNOME's animations turned off. Animate then
	// goes to its target at once; Loop, which shows that something is
	// going on, goes on.
	ReduceMotion bool
	// HighContrast asks for more contrast: macOS's Increase Contrast,
	// Windows's contrast themes, the desktop portal's higher contrast. The
	// default theme then draws borders and secondary text darker (lighter
	// in the dark), and the focus ring opaque.
	HighContrast bool
	// TextScale is how many times larger than usual text should be, as
	// Windows's and GNOME's text size settings say, 1 for usual: the
	// default theme's FontSize is that much larger.
	TextScale float32
}

// Preferences returns the settings of the desktop that controls follow.
// A frame follows their changes.
func (c *context) Preferences() Preferences { return c.rt.preferences() }

// preferences returns the desktop's settings, read once until they change.
func (rt *engine) preferences() Preferences {
	if !rt.prefsKnown {
		p := rt.host.preferences()
		rt.prefs = Preferences{
			Accent:       Color{R: p.Accent.R, G: p.Accent.G, B: p.Accent.B, A: p.Accent.A},
			ReduceMotion: p.ReduceMotion,
			HighContrast: p.HighContrast,
			TextScale:    float32(p.TextScale),
		}
		if rt.prefs.TextScale <= 0 {
			rt.prefs.TextScale = 1
		}
		rt.prefsKnown = true
	}
	return rt.prefs
}

// follow makes the theme follow the desktop's settings.
func (t *Theme) follow(p Preferences) {
	if a := p.Accent; a.A > 0 {
		a.A = 255
		black, white := Color{A: 255}, Color{R: 255, G: 255, B: 255, A: 255}
		t.Accent = a
		if t.Dark {
			t.AccentHover, t.AccentPressed = a.Mix(white, 0.2), a.Mix(black, 0.15)
			t.Selection, t.Focus = a.Alpha(0.4), a.Mix(white, 0.2).Alpha(0.6)
		} else {
			t.AccentHover, t.AccentPressed = a.Mix(black, 0.12), a.Mix(black, 0.25)
			t.Selection, t.Focus = a.Alpha(0.25), a.Alpha(0.55)
		}
		// Text on the accent is black where white would not stand out, as
		// on yellow.
		t.AccentText = white
		if l := a.gray().R; l > 165 {
			t.AccentText = Color{R: 24, G: 24, B: 27, A: 255}
		}
	}
	if p.HighContrast {
		t.Border = t.Border.Mix(t.Text, 0.45)
		t.TextMuted = t.TextMuted.Mix(t.Text, 0.4)
		t.Focus.A = 255
		t.Scrollbar = t.Scrollbar.Alpha(1.6)
	}
	if s := p.TextScale; s > 0 && s != 1 {
		t.FontSize *= s
	}
}
