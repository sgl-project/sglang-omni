package main

import (
	"github.com/egoist/mygo/ui"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/icons"
)

type tryLocalUI struct {
	transcript    transcriptRenderFunc
	customPalette bool
}

// tryCard draws one Try area from its display state and user-intent actions.
func tryCard(c *ui.Context, state *tryCardViewState, actions tryCardActions, local ...*tryLocalUI) {
	t := c.Theme()
	prefix := state.Prefix
	var render transcriptRenderFunc
	defaultPalette := true
	if len(local) > 0 && local[0] != nil {
		render = local[0].transcript
		defaultPalette = !local[0].customPalette
	}
	previewKey := prefix + "/" + state.Feature
	panel := card(c.Key(prefix + "-try"))
	if state.Mode == "recording" && state.Feature == "transcribe" {
		k := tok(t)
		if k.TranscriptFace != k.Subtle {
			panel.Background(k.TranscriptFace)
		}
	}
	panel.Children(func() {
		switch state.Mode {
		case "recording":
			tryRecording(c, state.Prefix, state.Recording, actions, defaultPalette)
			return
		case "working":
			ui.Row(c).Gap(px(t, sp2)).AlignItems(ui.Center).Children(func() {
				ui.Spinner(c).Size(px(t, 16), px(t, 16))
				ui.Text(c, state.WorkingText).Grow(1).MinWidth(0)
				if ghostButton(c.Key(prefix+"-try-cancel"), i18n.T("Cancel")).Clicked() {
					actions.cancelTry()
				}
			})
			if state.Draft != "" {
				sessionTextCard(c.Key(prefix+"-try-draft"), state.Feature).Padding(px(t, sp3)).Children(func() {
					transcriptResult(c, render, state.Feature, previewKey, state.Draft, true)
				})
			}
			return
		}
		if state.Ended != nil {
			tryEnded(c, prefix, previewKey, state.Feature, state.Ended, actions, render)
		} else if state.Intro != "" {
			caption(c, state.Intro)
		}
		tryNotice(c, prefix+"-try-block", noticeError, state.Block, actions)
		ui.Row(c).Gap(px(t, sp2)).AlignItems(ui.Center).Wrap().Children(func() {
			if tryRecordButton(c.Key(prefix+"-try-start"), state.StartLabel, defaultPalette).Label(i18n.T("Start a test")).Disabled(state.Busy).Clicked() {
				actions.startTry()
			}
			if state.Busy {
				caption(c, state.BusyText)
			}
		})
	})
}

func tryRecordButton(c *ui.Context, label string, defaultPalette bool) ui.Element {
	if defaultPalette {
		return recordButton(c, label)
	}
	return ui.PrimaryButton(c, label).Border(1, ui.Color{})
}

func tryNotice(c *ui.Context, key string, tone noticeKind, state *tryNoticeViewState, actions tryCardActions) {
	if state == nil {
		return
	}
	var dismiss func()
	if state.Dismiss {
		dismiss = actions.clearTryNotice
	}
	notice(c, key, tone, state.Text, state.More, func() func() {
		if state.ActionLabel == "" {
			return nil
		}
		return func() {
			if ui.Button(c.Key(key+"-action"), state.ActionLabel).Clicked() {
				actions.runTryNoticeAction(state.Action)
			}
		}
	}(), dismiss)
}

func tryRecording(c *ui.Context, prefix string, state *tryRecordingViewState, actions tryCardActions, defaultPalette bool) {
	t := c.Theme()
	k := tok(t)
	icon := transcribeIcon
	switch state.Kind {
	case "translate":
		icon = translateIcon
	case "rewrite":
		icon = rewriteIcon
	}
	ui.Row(c).Gap(px(t, sp3)).AlignItems(ui.Center).Children(func() {
		ui.Icon(c, icon.Bold).Size(px(t, 18), px(t, 18)).TextColor(k.Record).Shrink(0)
		levelBars(c, state.Levels, state.Heard)
		ui.Text(c, state.ElapsedText).FontSize(px(t, fsCaption)).Font("monospace").TextColor(k.Text2).Shrink(0)
	})
	if state.LiveText == "" {
		caption(c, i18n.T("Listening…"))
	} else {
		ui.Text(c, state.LiveText).MaxLines(4).Ellipsis("…")
	}
	ui.Row(c).Gap(px(t, sp2)).AlignItems(ui.Center).Children(func() {
		if tryRecordButton(c.Key(prefix+"-try-stop"), i18n.T("Stop"), defaultPalette).Label(i18n.T("Stop recording")).Clicked() {
			actions.stopTry()
		}
		if ghostButton(c.Key(prefix+"-try-cancel"), i18n.T("Cancel")).Clicked() {
			actions.cancelTry()
		}
	})
}

const tryLevelBarCount = 16

func levelBars(c *ui.Context, levels []float64, heard bool) {
	t := c.Theme()
	k := tok(t)
	ui.Row(c.Key("tr-wave")).Grow(1).MinWidth(0).Height(px(t, 24)).Gap(px(t, 3)).AlignItems(ui.Center).Label(i18n.T("Input level")).Children(func() {
		pad := tryLevelBarCount - len(levels)
		col := k.Waveform
		if !heard {
			col = k.G[5]
		}
		for index := range tryLevelBarCount {
			value := 0.0
			if index >= pad {
				value = levels[index-pad]
			}
			ui.Box(c).Width(px(t, 3)).Height(px(t, float32(3+min(max(value, 0), 1)*21))).Radius(px(t, 1.5)).Background(col)
		}
	})
}

func tryEnded(c *ui.Context, prefix, previewKey, feature string, state *tryEndedViewState, actions tryCardActions, render transcriptRenderFunc) {
	t := c.Theme()
	k := tok(t)
	if state.Cancelled {
		caption(c, i18n.T("Cancelled."))
		return
	}
	if state.Failure != nil {
		tryNotice(c, prefix+"-try-failed", noticeError, state.Failure, actions)
		return
	}
	if state.FailureText != "" {
		caption(c, state.FailureText)
		return
	}
	tryNotice(c, prefix+"-try-warn", noticeWarn, state.Warning, actions)
	ui.Row(c).Gap(px(t, sp2)).AlignItems(ui.Center).Children(func() {
		ui.Text(c, i18n.T("Result")).FontSize(px(t, fsCaption)).FontWeight(600).TextColor(k.Text2).Grow(1)
		if ghostIconButton(c.Key(prefix+"-try-copy"), icons.Copy, i18n.T("Copy")).Label(i18n.T("Copy result")).Disabled(state.Result == "").Clicked() {
			actions.copyTryResult(state.Result)
		}
	})
	sessionTextCard(c.Key(prefix+"-try-result"), feature).Padding(px(t, sp3)).Children(func() {
		transcriptResult(c, render, feature, previewKey, state.Result, false, func() {
			dictionaryText(c, state.Result, state.Highlights).Selectable()
		})
	})
	if state.Raw != "" {
		ui.Column(c).Gap(px(t, 4)).Children(func() {
			ui.Text(c, state.RawTitle).FontSize(px(t, fsCaption)).FontWeight(600).TextColor(k.Text2)
			ui.Text(c, state.Raw).FontSize(px(t, fsCaption)).TextColor(k.Text2).Selectable()
		})
	}
	if state.TimingsText != "" {
		caption(c, state.TimingsText)
	}
}
