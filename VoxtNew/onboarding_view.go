package main

// The setup guide's dialog (onboarding.go): its frame, the steps' header,
// the title and the footer with Back and Continue.

import (
	"time"

	"github.com/egoist/mygo/ui"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/i18n"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/icons"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/session"
)

// ---- The view ----

// onboarding is the guide over the window: the steps as breadcrumbs over
// a thin ink progress line, a big title, one column, Back at the bottom
// left and Continue at the bottom right.
func (a *app) onboarding(c *ui.Context) {
	if !a.ob.open {
		return
	}
	t := c.Theme()
	k := tok(t)
	w, h := c.Size()
	if a.sess.Try && a.sess.State == session.Recording {
		c.After(250 * time.Millisecond) // the try's clock
	}
	open := true
	ui.DialogBase(c.Key("onboarding"), &open, func(backdrop, panel ui.Element) {
		backdrop.Background(ui.RGBA(0, 0, 0, 0.35))
		dw, dh := onboardingDialogSize(t, w, h)
		panel.Width(dw).Height(dh).Radius(px(t, radiusCard)).Background(k.Bg).Border(1, k.Line).
			Shadow(0, 16, 40, 0, ui.RGBA(0, 0, 0, 0.3)).Label(i18n.T("Set up VoxtNew"))
		ui.Column(c).Fill().Children(func() {
			a.obHeader(c)
			ui.Scroll(c.Key("ob-scroll-" + itoa(a.ob.step))).Grow(1).MinHeight(0).Children(func() {
				ui.Column(c).Padding(px(t, sp2), px(t, obPadX), px(t, sp5)).Gap(px(t, sp3)).Children(func() {
					a.obTitle(c)
					switch a.ob.step {
					case obWelcome:
						a.obWelcomeStep(c)
					case obPermissions:
						a.permissionsSettings(c) // permissions.go
					case obModels:
						a.obModelsStep(c)
					case obTranscribe, obTranslate, obRewrite:
						a.obTryStep(c)
					default:
						a.obMoreStep(c)
					}
				})
			})
			ui.Box(c).Height(1).Background(k.Line)
			a.obFooter(c)
		})
	})
	if !open { // Escape or a click beside it
		a.closeOnboarding()
	}
}

// The setup guide's size: smaller than the settings dialog, so the
// window shows around it. obMaxW × obMaxH at most, and at least a margin
// of 5% of the window's width (sp8 at the least) and 7.5% of its height
// (sp6 at the least): 600×440 in the 800×520 window, 600×374 in the
// smallest (680×440), where the settings dialog is 612×387. Its header,
// body and footer are obPadX in from its sides.
const (
	obMaxW = 600
	obMaxH = 440
	obPadX = sp6
)

func onboardingDialogSize(t *ui.Theme, w, h float32) (dw, dh float32) {
	mx := max(px(t, sp8), w*0.05)
	my := max(px(t, sp6), h*0.075)
	return min(w-2*mx, px(t, obMaxW)), min(h-2*my, px(t, obMaxH))
}

// obHeader is the steps as breadcrumbs (passed ones with a check), ×, and
// the progress line under them.
func (a *app) obHeader(c *ui.Context) {
	t := c.Theme()
	k := tok(t)
	steps := obStepTitles()
	gap, arrow := a.obStepsSpacing(c, steps)
	ui.Row(c).Padding(px(t, sp3), px(t, sp3), px(t, sp3), px(t, obPadX)).Gap(px(t, sp2)).AlignItems(ui.Center).Children(func() {
		ui.Row(c).GapX(gap).GapY(px(t, 2)).AlignItems(ui.Center).Grow(1).MinWidth(0).Wrap().Children(func() {
			for i, s := range steps {
				if i > 0 {
					ui.Icon(c, icons.AltArrowRight.Linear).Size(arrow, arrow).TextColor(k.Text3)
				}
				ui.Row(c.Key("ob-step-" + itoa(i))).Gap(px(t, sp1)).AlignItems(ui.Center).Children(func() {
					col := k.Text3
					switch {
					case i == a.ob.step:
						col = k.Text
					case a.ob.passed[i]:
						col = k.Text2
						ui.Icon(c, icons.CheckCircle.Bold).Size(px(t, 12), px(t, 12)).TextColor(k.Accent)
					case i < a.ob.step:
						col = k.Text2
					}
					tx := ui.Text(c, s).FontSize(px(t, fsCaption)).TextColor(col).SingleLine()
					if i == a.ob.step {
						tx.FontWeight(600)
					}
				})
			}
		})
		if iconButton(c.Key("ob-close"), icons.Close, i18n.T("Close the guide")).Tooltip(i18n.T("Close the guide (Settings → General opens it again)")).Clicked() {
			a.closeOnboarding()
		}
	})
	frac := float32(a.ob.step+1) / float32(obSteps)
	ui.Box(c).Height(2).Draw(func(p *ui.Painter, r ui.Rect) {
		p.Fill(r, k.Line, 0)
		p.Fill(ui.Rect{X: r.X, Y: r.Y, W: r.W * frac, H: r.H}, k.Ink, 0)
	})
}

// obStepsSpacing is the breadcrumbs' gap and arrow size: 6 and 12, or
// 3 and 10 when those would wrap them (English with the try steps
// checked); they wrap only if even that is too wide.
func (a *app) obStepsSpacing(c *ui.Context, steps []string) (gap, arrow float32) {
	t := c.Theme()
	w, h := c.Size()
	dw, _ := onboardingDialogSize(t, w, h)
	room := dw - px(t, obPadX+sp3+sp2+28+2) // the header's padding, gap, × and rounding
	var need float32
	for i, s := range steps {
		var weight int
		if i == a.ob.step {
			weight = 600
		}
		tw, _ := c.MeasureText(0, ui.Span{Text: s, Size: px(t, fsCaption), Weight: weight})
		need += tw
		if a.ob.passed[i] && i != a.ob.step {
			need += px(t, 12+sp1) // the check
		}
	}
	gaps := float32(len(steps) - 1)
	if need+gaps*px(t, 12+2*6) <= room {
		return px(t, 6), px(t, 12)
	}
	return px(t, 3), px(t, 10)
}

// obTitle is the step's big title and its line of explanation.
func (a *app) obTitle(c *ui.Context) {
	t := c.Theme()
	var title, sub string
	switch a.ob.step {
	case obWelcome:
		title = i18n.T("Welcome to VoxtNew")
		sub = i18n.T("Speak in any app and get text where your cursor is: transcribe, translate and rewrite with models that run on this Mac. A few short steps; you can skip any of them.")
	case obPermissions:
		title = i18n.T("Permissions")
	case obModels:
		title = i18n.T("Local models")
		sub = i18n.T("VoxtNew runs its models on this Mac. These two are the best start; the Local models page has the others.")
	case obTranscribe:
		title = i18n.T("Try Transcribe")
		sub = i18n.T("Press the shortcut (or Start), read the sentence below or say anything, then press it again to stop. The result shows here.")
	case obTranslate:
		title = i18n.T("Try Translate")
		sub = i18n.T("Translate selected text, or what you say. Translate the sample below, or press the shortcut (or Start) and say a sentence.")
	case obRewrite:
		title = i18n.T("Try Rewrite")
		sub = i18n.T("With nothing selected, say what to write; with text selected, say how to change it. Press the shortcut (or Start) and ask for something short.")
	default:
		title = i18n.T("More to explore")
		sub = i18n.T("VoxtNew is ready. Two more things make it yours.")
	}
	ui.Column(c).Gap(px(t, sp1)).Margin(px(t, sp3), 0, px(t, sp2)).Children(func() {
		ui.Text(c, title).FontSize(px(t, fsTitle)).FontWeight(600)
		if sub != "" {
			ui.Text(c, sub).TextColor(tok(t).Text2)
		}
	})
}

// obFooter is Back, Skip and Continue (Start using VoxtNew on the last
// step). A try step's Continue waits for its try to pass; Skip moves on
// without it. Nothing moves while the step's try runs. Continue is a
// secondary button while the models step's reuse card shows its Use
// them.
func (a *app) obFooter(c *ui.Context) {
	t := c.Theme()
	busy := a.obBusy()
	_, tryStep := obStepKind(a.ob.step)
	// Back's arrow in line with the body's text; Continue under ×.
	ui.Row(c).Padding(px(t, sp3), px(t, sp3), px(t, sp3), px(t, obPadX)-t.Space(2.5)).Gap(px(t, sp2)).AlignItems(ui.Center).Children(func() {
		if a.ob.step > 0 && ghostIconButton(c.Key("ob-back"), icons.AltArrowLeft, i18n.T("Back")).Label(i18n.T("Back")).Disabled(busy).Clicked() {
			a.obGo(a.ob.step - 1)
		}
		ui.Spacer(c)
		if busy {
			caption(c, i18n.T("Finish or cancel the try first.")).SingleLine().Ellipsis("…").MinWidth(0).Shrink(1)
		}
		switch {
		case a.ob.step == obMore:
			if primaryButton(c.Key("ob-finish"), i18n.T("Start using VoxtNew")).Clicked() {
				a.closeOnboarding()
				a.navigate("/home")
			}
		case a.ob.step == obWelcome:
			if ghostButton(c.Key("ob-later"), i18n.T("Set up later")).Tooltip(i18n.T("Close the guide (Settings → General opens it again)")).Clicked() {
				a.closeOnboarding()
			}
			if primaryButton(c.Key("ob-next"), i18n.T("Get started")).Clicked() {
				a.obGo(a.ob.step + 1)
			}
		case tryStep && !a.ob.passed[a.ob.step]:
			if ghostButton(c.Key("ob-skip"), i18n.T("Skip this step")).Disabled(busy).Clicked() {
				a.obGo(a.ob.step + 1)
			}
			primaryButton(c.Key("ob-next"), i18n.T("Continue")).Disabled(true).Tooltip(i18n.T("A try that gives a result unlocks Continue"))
		case a.obReuseShown(): // the reuse card's Use them is the step's primary button
			if ui.Button(c.Key("ob-next"), i18n.T("Continue")).Clicked() {
				a.obGo(a.ob.step + 1)
			}
		default:
			if primaryButton(c.Key("ob-next"), i18n.T("Continue")).Disabled(busy).Clicked() {
				a.obGo(a.ob.step + 1)
			}
		}
	})
}
