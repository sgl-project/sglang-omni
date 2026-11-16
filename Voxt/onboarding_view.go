package main

// The setup guide's dialog (onboarding.go): its frame, the steps' header,
// the title and the footer with Back and Continue.

import (
	"strconv"
	"time"

	"github.com/egoist/mygo/ui"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/icons"
)

type onboardingLocalUI struct {
	open bool
}

// onboardingView is the guide over the window: the steps as breadcrumbs over
// a thin ink progress line, a big title, one column, Back at the bottom
// left and Continue at the bottom right.
func onboardingView(c *ui.Context, state onboardingViewState, actions onboardingActions, local *onboardingLocalUI) {
	if !local.open {
		return
	}
	t := c.Theme()
	k := tok(t)
	w, h := c.Size()
	if state.Recording {
		c.After(250 * time.Millisecond)
	}
	ui.DialogBase(c.Key("onboarding"), &local.open, func(backdrop, panel ui.Element) {
		backdrop.Background(ui.RGBA(0, 0, 0, 0.35))
		dw, dh := onboardingDialogSize(t, w, h)
		panel.Width(dw).Height(dh).Radius(px(t, radiusCard)).Background(k.Bg).Border(1, k.Line).
			Shadow(0, 16, 40, 0, ui.RGBA(0, 0, 0, 0.3)).Label(i18n.T("Set up Voxt"))
		ui.Column(c).Fill().Children(func() {
			obHeader(c, state, actions)
			ui.Scroll(c.Key("ob-scroll-" + strconv.Itoa(state.Step))).Grow(1).MinHeight(0).Children(func() {
				ui.Column(c).Padding(px(t, sp2), px(t, obPadX), px(t, sp5)).Gap(px(t, sp3)).Children(func() {
					obTitle(c, state.Title, state.Subtitle)
					switch state.Step {
					case obWelcome:
						obWelcomeStep(c, state.Welcome, actions)
					case obPermissions:
						onboardingPermissionRows(c, state.Permissions, actions)
					case obModels:
						obModelsStep(c, state.Models, actions)
					case obTranscribe, obTranslate, obRewrite:
						obTryStep(c, state.Try, actions)
					default:
						obMoreStep(c, state.More, actions)
					}
				})
			})
			ui.Box(c).Height(1).Background(k.Line)
			obFooter(c, state, actions)
		})
	})
	if !local.open {
		actions.closeOnboarding()
	}
}

// onboardingBackBar is the way back to the guide on a page one of its
// links opened; × hides it (the guide still opens at that step).
func onboardingBackBar(c *ui.Context, state onboardingBackViewState, actions onboardingActions) {
	if !state.Visible {
		return
	}
	notice(c, "ob-back", noticeInfo, state.Text, "", func() {
		if ui.Button(c.Key("ob-back-open"), i18n.T("Back to the setup guide")).Clicked() {
			actions.openOnboarding()
		}
	}, actions.dismissOnboardingBack)
}

// onboardingRow is Settings → General's way back to the guide.
func onboardingRow(c *ui.Context, actions onboardingActions) {
	settingRow(c, "setup-guide", i18n.T("Setup guide"), "", func() {
		if ui.Button(c.Key("setup-guide-open"), i18n.T("Open setup guide")).Clicked() {
			actions.openOnboarding()
		}
	})
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
func obHeader(c *ui.Context, state onboardingViewState, actions onboardingActions) {
	t := c.Theme()
	k := tok(t)
	gap, arrow := obStepsSpacing(c, state)
	ui.Row(c).Padding(px(t, sp3), px(t, sp3), px(t, sp3), px(t, obPadX)).Gap(px(t, sp2)).AlignItems(ui.Center).Children(func() {
		ui.Row(c).GapX(gap).GapY(px(t, 2)).AlignItems(ui.Center).Grow(1).MinWidth(0).Wrap().Children(func() {
			for index, title := range state.StepTitles {
				if index > 0 {
					ui.Icon(c, icons.AltArrowRight.Linear).Size(arrow, arrow).TextColor(k.Text3)
				}
				ui.Row(c.Key("ob-step-" + strconv.Itoa(index))).Gap(px(t, sp1)).AlignItems(ui.Center).Children(func() {
					color := k.Text3
					switch {
					case index == state.Step:
						color = k.Text
					case state.Passed[index]:
						color = k.Text2
						ui.Icon(c, icons.CheckCircle.Bold).Size(px(t, 12), px(t, 12)).TextColor(k.Accent)
					case index < state.Step:
						color = k.Text2
					}
					text := ui.Text(c, title).FontSize(px(t, fsCaption)).TextColor(color).SingleLine()
					if index == state.Step {
						text.FontWeight(600)
					}
				})
			}
		})
		if iconButton(c.Key("ob-close"), icons.Close, i18n.T("Close guide")).Tooltip(i18n.T("Close guide (Settings → General opens it again)")).Clicked() {
			actions.closeOnboarding()
		}
	})
	fraction := float32(state.Step+1) / float32(len(state.StepTitles))
	ui.Box(c).Height(2).Draw(func(painter *ui.Painter, rect ui.Rect) {
		painter.Fill(rect, k.Line, 0)
		painter.Fill(ui.Rect{X: rect.X, Y: rect.Y, W: rect.W * fraction, H: rect.H}, k.Ink, 0)
	})
}

// obStepsSpacing is the breadcrumbs' gap and arrow size: 6 and 12, or
// 3 and 10 when those would wrap them (English with the try steps
// checked); they wrap only if even that is too wide.
func obStepsSpacing(c *ui.Context, state onboardingViewState) (gap, arrow float32) {
	t := c.Theme()
	w, h := c.Size()
	dw, _ := onboardingDialogSize(t, w, h)
	room := dw - px(t, obPadX+sp3+sp2+28+2)
	var needed float32
	for index, title := range state.StepTitles {
		weight := 0
		if index == state.Step {
			weight = 600
		}
		width, _ := c.MeasureText(0, ui.Span{Text: title, Size: px(t, fsCaption), Weight: weight})
		needed += width
		if state.Passed[index] && index != state.Step {
			needed += px(t, 12+sp1)
		}
	}
	gaps := float32(len(state.StepTitles) - 1)
	if needed+gaps*px(t, 12+2*6) <= room {
		return px(t, 6), px(t, 12)
	}
	return px(t, 3), px(t, 10)
}

// obTitle is the step's big title and its line of explanation.
func obTitle(c *ui.Context, title, subtitle string) {
	t := c.Theme()
	ui.Column(c).Gap(px(t, sp1)).Margin(px(t, sp3), 0, px(t, sp2)).Children(func() {
		ui.Text(c, title).FontSize(px(t, fsTitle)).FontWeight(600)
		if subtitle != "" {
			ui.Text(c, subtitle).TextColor(tok(t).Text2)
		}
	})
}

// obFooter is Back, Skip and Continue (Start using Voxt on the last
// step). A try step's Continue waits for its try to pass; Skip moves on
// without it. Nothing moves while the step's try runs. Continue is a
// secondary button while the models step's reuse card shows its Use
// them, and Download and continue (beside Skip) while the models step has
// only the download to offer.
func obFooter(c *ui.Context, state onboardingViewState, actions onboardingActions) {
	t := c.Theme()
	footer := state.Footer
	ui.Row(c).Padding(px(t, sp3), px(t, sp3), px(t, sp3), px(t, obPadX)-t.Space(2.5)).Gap(px(t, sp2)).AlignItems(ui.Center).Children(func() {
		if footer.Back && ghostIconButton(c.Key("ob-back"), icons.AltArrowLeft, i18n.T("Back")).Label(i18n.T("Back")).Disabled(footer.Busy).Clicked() {
			actions.obGo(state.Step - 1)
		}
		ui.Spacer(c)
		if footer.Busy {
			caption(c, i18n.T("Finish or cancel the test first.")).SingleLine().Ellipsis("…").MinWidth(0).Shrink(1)
		}
		skip := func() {
			if ghostButton(c.Key("ob-skip"), i18n.T("Skip this step")).Disabled(footer.Busy).Clicked() {
				actions.obGo(state.Step + 1)
			}
		}
		switch footer.Mode {
		case "finish":
			if primaryButton(c.Key("ob-finish"), footer.ContinueLabel).Clicked() {
				actions.finishOnboarding("/home")
			}
		case "welcome":
			if ghostButton(c.Key("ob-later"), i18n.T("Set up later")).Tooltip(i18n.T("Close guide (Settings → General opens it again)")).Clicked() {
				actions.closeOnboarding()
			}
			if primaryButton(c.Key("ob-next"), footer.ContinueLabel).Clicked() {
				actions.obGo(state.Step + 1)
			}
		case "try-wait":
			skip()
			primaryButton(c.Key("ob-next"), footer.ContinueLabel).Disabled(true).Tooltip(footer.ContinueTooltip)
		case "reuse":
			if ui.Button(c.Key("ob-next"), footer.ContinueLabel).Clicked() {
				actions.obGo(state.Step + 1)
			}
		case "download":
			skip()
			if primaryButton(c.Key("ob-next"), footer.ContinueLabel).Disabled(footer.ContinueDisabled).Clicked() {
				actions.onboardingDownloadAndContinue()
			}
		default:
			button := primaryButton(c.Key("ob-next"), footer.ContinueLabel).Disabled(footer.ContinueDisabled)
			if footer.ContinueTooltip != "" {
				button = button.Tooltip(footer.ContinueTooltip)
			}
			if button.Clicked() {
				actions.obGo(state.Step + 1)
			}
		}
	})
}
