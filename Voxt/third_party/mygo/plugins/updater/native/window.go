package native

import (
	"math"

	"github.com/egoist/mygo/plugins/updater/internal/frontend"
	"github.com/egoist/mygo/ui"
)

// window is the update window of a session.
type window struct {
	s     frontend.Session
	texts frontend.Texts
	icon  *ui.Bitmap // nil without an icon

	// What the view keeps from frame to frame, on the main thread: the
	// prompt of the view it shows, the prompt the user answered, and the
	// checkbox, which each prompt resets.
	prompt   int
	answered int
	checked  bool
	// built is what the last frame laid out, and fitted the size last
	// asked of the window for it.
	built  layout
	fitted [2]int
}

// layout is what the size of the window's content depends on.
type layout struct {
	prompt                    int
	title, message, detail    string
	bar, checkbox, release    bool
	windowWidth, windowHeight float32
}

// open creates the window of s, which shows at once.
func open(s frontend.Session) {
	w := newWindow(s)
	// Changes made while the window is being created redraw it too.
	_, changed := s.Current()
	win := s.NewWindow(ui.View(w.view))
	go func() {
		for {
			select {
			case <-changed:
				_, changed = s.Current()
				win.Invalidate()
			case <-s.Done():
				return
			}
		}
	}()
	frontend.Show(win, s.User())
}

func newWindow(s frontend.Session) *window {
	w := &window{s: s, texts: s.Texts(), prompt: -1, answered: -1}
	if png := s.Icon(); png != nil {
		w.icon, _ = ui.DecodeBitmap(png) // no icon when it is not an image
	}
	return w
}

// The room around the window's content and between its parts, in DIPs, as
// in the page of package updater.
const (
	padTop, padSide, padBottom = 18, 20, 20
	iconSize, iconGap          = 64, 18
	footGap                    = 12
	asideGap                   = 40
)

// view builds the update window.
func (w *window) view(c *ui.Context) {
	v, _ := w.s.Current()
	if v.Prompt != w.prompt {
		w.prompt, w.checked = v.Prompt, v.Checked
	}
	t := theme(c.Theme())
	c.SetTheme(t)
	w.keys(c, v)

	var body, aside, buttons ui.Element
	ui.Column(c).Fill().Padding(padTop, padSide, padBottom).Gap(footGap).Children(func() {
		top := w.row(c).Grow(1).Gap(iconGap).AlignItems(ui.Start)
		if v.Release {
			top.AlignItems(ui.Stretch)
		}
		top.Children(func() {
			if w.icon != nil {
				ui.Image(c, w.icon).Size(iconSize, iconSize).Shrink(0).AlignSelf(ui.Start)
			}
			body = ui.Column(c).Grow(1).Children(func() { w.body(c, t, v) })
		})
		w.row(c).Gap(10).Shrink(0).Children(func() {
			aside = w.row(c).Gap(10).Shrink(0).Children(func() { w.buttons(c, t, v, true) })
			ui.Spacer(c)
			buttons = w.row(c).Gap(10).Shrink(0).Children(func() { w.buttons(c, t, v, false) })
		})
	})
	w.fit(c, v, body, aside, buttons)
}

// row creates a row, laid out from the right in a window of a language
// written from right to left.
func (w *window) row(c *ui.Context) ui.Element {
	r := ui.Row(c)
	if w.texts.RTL {
		r.Reverse()
	}
	return r
}

// body builds the texts, progress bar, release notes and checkbox of the
// view. Texts start on the side their own language starts, as the
// page's unicode-bidi: plaintext does, since they may fall back to
// English or be errors.
func (w *window) body(c *ui.Context, t *ui.Theme, v frontend.View) {
	small := t.Rem(11.0 / 13)
	ui.Text(c.Key("title"), v.Title).Bold().Margin(0, 0, 4, 0)
	if v.Message != "" {
		ui.Text(c.Key("message"), v.Message).Margin(0, 0, 6, 0)
	}
	if v.Bar {
		bar := ui.Progress(c.Key("bar"), v.Progress).Margin(6, 0, 4, 0)
		if w.texts.RTL {
			bar.Reverse()
		}
	}
	if v.Detail != "" {
		ui.Text(c.Key("detail"), v.Detail).FontSize(small).TextColor(t.TextMuted).
			MaxLines(2).Tooltip(v.Detail).Selectable()
	}
	if len(v.Notes) > 0 {
		ui.Column(c.Key("notes")).Grow(1).Margin(6, 0, 0, 0).Children(func() {
			ui.Text(c, w.texts.ReleaseNotes).FontSize(small).Bold().Margin(0, 0, 4, 0)
			ui.Scroll(c).Grow(1).Background(panel(t)).Border(1, t.Border).Padding(8, 12).Children(func() {
				notes(c, t, v.Notes)
			})
		})
	}
	if v.Checkbox {
		w.row(c).Key("checkbox").Margin(10, 0, 0, 0).Children(func() {
			box := ui.Checkbox(c, &w.checked, w.texts.AutomaticDownloads)
			if w.texts.RTL {
				box.Reverse()
			}
		})
	}
}

// buttons builds the buttons of the view set aside, or the others.
func (w *window) buttons(c *ui.Context, t *ui.Theme, v frontend.View, aside bool) {
	height := float32(math.Round(float64(t.Rem(24.0 / 13))))
	for _, b := range v.Buttons {
		if b.Aside != aside {
			continue
		}
		// Keyed by action, so that the focus leaves with the button.
		var e ui.Element
		if b.Default {
			e = ui.PrimaryButton(c.Key(string(b.Action)), "")
		} else {
			e = ui.Button(c.Key(string(b.Action)), "").Shadow(0, 1, 1, 0, ui.RGBA(0, 0, 0, 0.08))
		}
		e.Padding(0, 14).Height(height).MinWidth(84).Disabled(w.off(v, b)).Children(func() {
			ui.Text(c, b.Label).SingleLine()
		})
		if e.Clicked() {
			w.respond(v, b)
		}
	}
}

// keys answers with the button of Escape, and that of Enter when no
// button has the focus.
func (w *window) keys(c *ui.Context, v frontend.View) {
	for _, b := range v.Buttons {
		if b.Cancel && c.Shortcut(0, ui.KeyEscape) || b.Default && c.Shortcut(0, ui.KeyEnter) {
			w.respond(v, b)
		}
	}
}

// off reports whether a button is disabled: by the view, or because the
// user answered its prompt, until the next view.
func (w *window) off(v frontend.View, b frontend.Button) bool {
	return b.Disabled || w.answered == v.Prompt
}

func (w *window) respond(v frontend.View, b frontend.Button) {
	if w.off(v, b) {
		return
	}
	w.answered = v.Prompt
	w.s.Respond(v.Prompt, b.Action, w.checked)
}

// fit asks for the size the content needs, as measured in the last frame
// when it laid out the same view at the same size: the width of the
// buttons, and for status views the height of the text.
func (w *window) fit(c *ui.Context, v frontend.View, body, aside, buttons ui.Element) {
	ww, wh := c.Size()
	l := layout{v.Prompt, v.Title, v.Message, v.Detail, v.Bar, v.Checkbox, v.Release, ww, wh}
	if l != w.built {
		// Measure the layout of this frame in the next.
		w.built = l
		c.Invalidate()
		return
	}
	b, a, f := body.Bounds(), aside.Bounds(), buttons.Bounds()
	width := padSide + a.W + f.W + padSide
	if a.W > 0 {
		width += asideGap
	}
	height := padTop + b.H + footGap + max(a.H, f.H) + padBottom
	if w.icon != nil {
		height = max(height, padTop+iconSize+footGap+max(a.H, f.H)+padBottom)
	}
	size := [2]int{int(math.Ceil(float64(width))), int(math.Ceil(float64(height)))}
	if size != w.fitted {
		w.fitted = size
		go w.s.Fit(size[0], size[1])
	}
}

// theme returns the look of the update window, that of the page of
// package updater, over base.
func theme(base *ui.Theme) *ui.Theme {
	t := *base
	light, dark := frontend.Background()
	if t.Dark {
		t.Background = ui.Hex(dark)
		t.Surface = ui.Hex("#5b5b5f")
		t.Border = ui.RGBA(255, 255, 255, 0.15)
		t.Text, t.TextMuted = ui.Hex("#f5f5f7"), ui.Hex("#98989d")
		t.Accent = ui.Hex("#0a84ff")
	} else {
		t.Background = ui.Hex(light)
		t.Surface = ui.Hex("#ffffff")
		t.Border = ui.RGBA(0, 0, 0, 0.15)
		t.Text, t.TextMuted = ui.Hex("#1d1d1f"), ui.Hex("#6e6e73")
		t.Accent = ui.Hex("#007aff")
	}
	black := ui.RGB(0, 0, 0)
	t.SurfaceHover, t.SurfacePressed = t.Surface.Mix(black, 0.05), t.Surface.Mix(black, 0.1)
	t.AccentHover, t.AccentPressed = t.Accent.Mix(black, 0.08), t.Accent.Mix(black, 0.16)
	t.Selection, t.Focus = t.Accent.Alpha(0.25), t.Accent.Alpha(0.55)
	return &t
}

// panel returns the background of the release notes.
func panel(t *ui.Theme) ui.Color {
	if t.Dark {
		return ui.Hex("#141414")
	}
	return ui.Hex("#ffffff")
}

// track returns the background of code.
func track(t *ui.Theme) ui.Color {
	if t.Dark {
		return ui.RGBA(255, 255, 255, 0.12)
	}
	return ui.RGBA(0, 0, 0, 0.1)
}
