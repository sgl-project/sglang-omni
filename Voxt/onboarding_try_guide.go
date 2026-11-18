package main

// What a setup-guide try step still needs before Start, and the try's
// recording, working and result states.

import (
	"github.com/egoist/mygo/ui"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/icons"
)

func obTryPrereqs(c *ui.Context, state onboardingTryViewState, actions onboardingActions) {
	if guide := state.LanguageGuide; guide != nil {
		notice(c, state.Prefix+"-llm-guide", noticeWarn, guide.Text, guide.More, func() {
			if ui.Button(c.Key(state.Prefix+"-llm-setup"), guide.ActionLabel).Clicked() {
				actions.onboardingTryLanguage()
			}
		}, nil)
		return
	}
	if guide := state.SpeechGuide; guide != nil {
		notice(c, state.Prefix+"-asr-dl", noticeInfo, guide.Text, guide.More, nil, nil)
	}
}

func onboardingNoticeButton(c *ui.Context, key string, state *onboardingNoticeViewState, actions onboardingActions) func() {
	if state == nil || state.ActionLabel == "" {
		return nil
	}
	return func() {
		if ui.Button(c.Key(key), state.ActionLabel).Clicked() {
			actions.onboardingNoticeAction(state.Action)
		}
	}
}

// obTryCard is a Try area: Start, the try while it records and works, its
// result, why it did not start.
func obTryCard(c *ui.Context, prefix string, state onboardingTryCardViewState, actions onboardingActions, render transcriptRenderFunc, defaultPalette bool) {
	t := c.Theme()
	card(c.Key(prefix + "-try")).Children(func() {
		switch {
		case state.Mine && state.Recording:
			obTryRecording(c, prefix, state, actions, defaultPalette)
			return
		case state.Mine && state.Live:
			ui.Row(c).Gap(px(t, sp2)).AlignItems(ui.Center).Children(func() {
				ui.Spinner(c).Size(px(t, 16), px(t, 16))
				ui.Text(c, state.WorkingText).Grow(1).MinWidth(0)
				if ghostButton(c.Key(prefix+"-try-cancel"), i18n.T("Cancel")).Clicked() {
					actions.cancelSession()
				}
			})
			if state.Draft != "" && state.Composing {
				sessionTextCard(c.Key(prefix+"-try-draft"), state.SessionKind).Padding(px(t, sp3)).Children(func() {
					transcriptResult(c, render, state.SessionKind, prefix, state.Draft, true)
				})
			}
			return
		}
		if state.Mine && state.Ended {
			obTryEnded(c, prefix, state, actions, render)
		}
		if state.Block != nil {
			notice(c, prefix+"-try-block", noticeError, state.Block.Text, state.Block.More,
				onboardingNoticeButton(c, prefix+"-try-block-action", state.Block, actions), actions.clearOnboardingBlock)
		}
		ui.Row(c).Gap(px(t, sp2)).AlignItems(ui.Center).Wrap().Children(func() {
			label := i18n.T("Start")
			if state.Mine && state.Ended {
				label = i18n.T("Try again")
			}
			if tryRecordButton(c.Key(prefix+"-try-start"), label, defaultPalette).Label(i18n.T("Start a test")).Disabled(state.Live).Clicked() {
				_ = actions.obTry("")
			}
			if state.Live {
				caption(c, state.BusyText)
			}
		})
	})
}

// obTryRecording is the Try area while it records: the levels, the time,
// the live text, Stop and Cancel.
func obTryRecording(c *ui.Context, prefix string, state onboardingTryCardViewState, actions onboardingActions, defaultPalette bool) {
	t := c.Theme()
	k := tok(t)
	ui.Row(c).Gap(px(t, sp3)).AlignItems(ui.Center).Children(func() {
		ui.Icon(c, onboardingKindIcon(state.SessionKind).Bold).Size(px(t, 18), px(t, 18)).TextColor(k.Record).Shrink(0)
		onboardingLevelBars(c, state.Levels, state.Heard)
		ui.Text(c, state.ElapsedText).FontSize(px(t, fsCaption)).Font("monospace").TextColor(k.Text2).Shrink(0)
	})
	if state.Partial == "" {
		caption(c, i18n.T("Listening…"))
	} else {
		ui.Text(c, state.Partial).MaxLines(4).Ellipsis("…")
	}
	ui.Row(c).Gap(px(t, sp2)).AlignItems(ui.Center).Children(func() {
		if tryRecordButton(c.Key(prefix+"-try-stop"), i18n.T("Stop"), defaultPalette).Label(i18n.T("Stop recording")).Clicked() {
			actions.stopSession()
		}
		if ghostButton(c.Key(prefix+"-try-cancel"), i18n.T("Cancel")).Clicked() {
			actions.cancelSession()
		}
	})
}

func onboardingKindIcon(kind string) icons.Icon {
	switch kind {
	case "translate":
		return translateIcon
	case "rewrite":
		return rewriteIcon
	}
	return transcribeIcon
}

// onboardingLevelBars draws the input levels newest on the right, in the
// recording colour once the microphone is heard.
func onboardingLevelBars(c *ui.Context, levels []float64, heard bool) {
	t := c.Theme()
	k := tok(t)
	const bars = 16
	ui.Row(c.Key("tr-wave")).Grow(1).MinWidth(0).Height(px(t, 24)).Gap(px(t, 3)).AlignItems(ui.Center).Label(i18n.T("Input level")).Children(func() {
		pad := bars - len(levels)
		col := k.Waveform
		if !heard {
			col = k.G[5]
		}
		for index := range bars {
			value := 0.0
			if index >= pad {
				value = levels[index-pad]
			}
			ui.Box(c).Width(px(t, 3)).Height(px(t, float32(3+min(max(value, 0), 1)*21))).Radius(px(t, 1.5)).Background(col)
		}
	})
}

// obTryEnded is a finished try: its result (or why there is none), what
// was recognised, the stage times and Copy.
func obTryEnded(c *ui.Context, prefix string, state onboardingTryCardViewState, actions onboardingActions, render transcriptRenderFunc) {
	t := c.Theme()
	k := tok(t)
	if state.Cancelled {
		caption(c, i18n.T("Cancelled."))
		return
	}
	if state.Failed {
		if state.Failure != nil {
			notice(c, prefix+"-try-failed", noticeError, state.Failure.Text, state.Failure.More,
				onboardingNoticeButton(c, prefix+"-try-failed-action", state.Failure, actions), nil)
			return
		}
		caption(c, state.FailureText)
		return
	}
	if state.Warning != nil {
		notice(c, prefix+"-try-warn", noticeWarn, state.Warning.Text, state.Warning.More,
			onboardingNoticeButton(c, prefix+"-try-warn-action", state.Warning, actions), nil)
	}
	if !state.ShowResult {
		return
	}
	ui.Row(c).Gap(px(t, sp2)).AlignItems(ui.Center).Children(func() {
		ui.Text(c, i18n.T("Result")).FontSize(px(t, fsCaption)).FontWeight(600).TextColor(k.Text2).Grow(1)
		if ghostIconButton(c.Key(prefix+"-try-copy"), icons.Copy, i18n.T("Copy")).Label(i18n.T("Copy result")).Disabled(state.Result == "").Clicked() {
			actions.copyOnboardingResult(state.Result)
		}
	})
	sessionTextCard(c.Key(prefix+"-try-result"), state.SessionKind).Padding(px(t, sp3)).Children(func() {
		transcriptResult(c, render, state.SessionKind, prefix, state.Result, false)
	})
	if state.ShowRaw {
		ui.Column(c).Gap(px(t, 4)).Children(func() {
			ui.Text(c, state.RawTitle).FontSize(px(t, fsCaption)).FontWeight(600).TextColor(k.Text2)
			ui.Text(c, state.Raw).FontSize(px(t, fsCaption)).TextColor(k.Text2).Selectable()
		})
	}
	if state.Timings != "" {
		caption(c, state.Timings)
	}
}
