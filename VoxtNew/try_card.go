package main

// The Try area: a session started from a page (a feature page, Home, the
// setup guide) whose result stays on it instead of going to the app in
// front, with its recording, its notice and its result's timings.

import (
	"strings"
	"time"

	"github.com/egoist/mygo/ui"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/hotkey"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/i18n"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/icons"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/session"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/store"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/uitext"
)

// ---- Try it ----

// trySessionOf reports whether the current session is a try of feature.
func (a *app) trySessionOf(feature string) bool {
	s := a.sess
	return s.ID != "" && s.Try && s.Kind == featureKind(feature)
}

// tryIntro is what a feature's Try area says before a try: its shortcut
// (pressed while VoxtNew is in front, it starts a try here) or Start.
func (a *app) tryIntro(feature string) string {
	keys := a.featureKeys(feature)
	switch feature {
	case hotkey.Translate:
		if keys != "" {
			return i18n.T("Press %s or Start and say a sentence or two: the translation shows here and is not typed anywhere.", keys)
		}
		return i18n.T("Say a sentence or two: the translation shows here and is not typed anywhere.")
	case hotkey.Rewrite:
		if keys != "" {
			return i18n.T("Press %s or Start and say what to write (\"a short note saying I will be late\"): the answer shows here and is not typed anywhere.", keys)
		}
		return i18n.T("Say what to write (\"a short note saying I will be late\"): the answer shows here and is not typed anywhere.")
	}
	if keys != "" {
		return i18n.T("Press %s or Start and say a sentence or two: the result shows here and is not typed anywhere.", keys)
	}
	return i18n.T("Say a sentence or two: the result shows here and is not typed anywhere.")
}

func (a *app) featureTry(c *ui.Context, feature string) {
	u := a.fui(feature)
	sectionTitle(c, i18n.T("Try it"))
	a.tryCard(c, tryOpts{
		prefix: uitext.FeaturePrefix(feature), feature: feature, mine: a.trySessionOf(feature),
		intro: a.tryIntro(feature), block: &u.block,
		start: func() *sessionNotice { _, n := a.startTry(featureKind(feature)); return n },
	})
}

// tryOpts describe a Try card: a feature page's, Home's, a step of the
// setup guide's.
type tryOpts struct {
	prefix  string // keys its controls
	feature string // what Start tries
	// mine: the current session is this card's try (its id, for the
	// setup guide: a late or another place's session never shows).
	mine  bool
	intro string // before a try
	// block is why Start (or the shortcut) did not start, nil for
	// nothing; start starts a try and says why not.
	block **sessionNotice
	start func() *sessionNotice
}

// tryCard is a Try area: Start, the try while it records and works, its
// result, why it did not start.
func (a *app) tryCard(c *ui.Context, o tryOpts) {
	t := c.Theme()
	p := o.prefix
	s := a.sess
	try := o.mine
	clear := func() { *o.block = nil }
	card(c.Key(p + "-try")).Children(func() {
		switch {
		case try && s.State == session.Recording:
			a.tryRecording(c, p, s)
			return
		case try && s.State.Live():
			ui.Row(c).Gap(px(t, sp2)).AlignItems(ui.Center).Children(func() {
				ui.Spinner(c).Size(px(t, 16), px(t, 16))
				ui.Text(c, uitext.WorkingText(s)).Grow(1).MinWidth(0)
				if ghostButton(c.Key(p+"-try-cancel"), i18n.T("Cancel")).Clicked() {
					a.cancelSession()
				}
			})
			if d := strings.TrimSpace(s.Draft); d != "" && s.State == session.Composing {
				fillCard(c.Key(p + "-try-draft")).Padding(px(t, sp3)).Children(func() {
					ui.Text(c, d).Selectable()
				})
			}
			return
		}
		if try && s.State.Ended() {
			a.tryEnded(c, p, s, clear)
		} else if o.intro != "" {
			caption(c, o.intro)
		}
		if n := *o.block; n != nil {
			notice(c, p+"-try-block", noticeError, n.text, n.more, a.noticeButton(c, p+"-try-block-action", n.action, clear), clear)
		}
		ui.Row(c).Gap(px(t, sp2)).AlignItems(ui.Center).Wrap().Children(func() {
			label := i18n.T("Start")
			if try && s.State.Ended() {
				label = i18n.T("Try again")
			}
			busy := s.State.Live()
			if recordButton(c.Key(p+"-try-start"), label).Label(i18n.T("Start a try")).Disabled(busy).Clicked() {
				*o.block = o.start()
			}
			if busy {
				if s.Kind == store.KindTranscribe {
					caption(c, i18n.T("A transcription is running."))
				} else {
					caption(c, i18n.T("Another session is running."))
				}
			}
		})
	})
}

// noticeButton is a notice's action button, nil for none; clear hides
// the notice.
func (a *app) noticeButton(c *ui.Context, key, action string, clear func()) func() {
	label := noticeActionLabel(action)
	if label == "" {
		return nil
	}
	return func() {
		if ui.Button(c.Key(key), label).Clicked() {
			clear()
			a.noticeAction(action)
		}
	}
}

// tryRecording is the Try area while it records: the levels, the time,
// the live text, Stop and Cancel.
func (a *app) tryRecording(c *ui.Context, p string, s session.Snapshot) {
	t := c.Theme()
	k := tok(t)
	ui.Row(c).Gap(px(t, sp3)).AlignItems(ui.Center).Children(func() {
		ui.Icon(c, kindIcon(s.Kind).Bold).Size(px(t, 18), px(t, 18)).TextColor(k.Record).Shrink(0)
		levelBars(c, s.Levels, !s.HeardAt.IsZero())
		ui.Text(c, clock(s.Elapsed(time.Now()))).FontSize(px(t, fsCaption)).Font("monospace").TextColor(k.Text2).Shrink(0)
	})
	live := strings.TrimSpace(s.Partial)
	if live == "" {
		caption(c, i18n.T("Listening…"))
	} else {
		ui.Text(c, live).MaxLines(4).Ellipsis("…")
	}
	ui.Row(c).Gap(px(t, sp2)).AlignItems(ui.Center).Children(func() {
		if recordButton(c.Key(p+"-try-stop"), i18n.T("Stop")).Label(i18n.T("Stop recording")).Clicked() {
			a.stopSession()
		}
		if ghostButton(c.Key(p+"-try-cancel"), i18n.T("Cancel")).Clicked() {
			a.cancelSession()
		}
	})
}

// levelBars draws the input levels as session.Bars bars, newest on the
// right, in the recording colour once the microphone is heard.
func levelBars(c *ui.Context, levels []float64, heard bool) {
	t := c.Theme()
	k := tok(t)
	ui.Row(c.Key("tr-wave")).Grow(1).MinWidth(0).Height(px(t, 24)).Gap(px(t, 3)).AlignItems(ui.Center).Label(i18n.T("Input level")).Children(func() {
		pad := session.Bars - len(levels)
		col := k.Record
		if !heard {
			col = k.G[5]
		}
		for i := range session.Bars {
			v := 0.0
			if i >= pad {
				v = levels[i-pad]
			}
			ui.Box(c).Width(px(t, 3)).Height(px(t, float32(3+min(max(v, 0), 1)*21))).Radius(px(t, 1.5)).Background(col)
		}
	})
}

// tryEnded is a finished try: its result (or why there is none), what
// was recognised, the stage times and Copy.
func (a *app) tryEnded(c *ui.Context, p string, s session.Snapshot, clear func()) {
	t := c.Theme()
	k := tok(t)
	switch s.State {
	case session.Cancelled:
		caption(c, i18n.T("Cancelled."))
		return
	case session.Failed:
		if s.Problem != nil {
			text, more, action := sessionProblemText(s.Problem)
			if text != "" {
				notice(c, p+"-try-failed", noticeError, text, more, a.noticeButton(c, p+"-try-failed-action", action, clear), nil)
				return
			}
		}
		caption(c, i18n.T("Nothing was heard."))
		return
	}
	if pr := s.Problem; pr != nil {
		if text, more, action := sessionProblemText(pr); text != "" {
			notice(c, p+"-try-warn", noticeWarn, text, more, a.noticeButton(c, p+"-try-warn-action", action, clear), nil)
		}
	}
	ui.Row(c).Gap(px(t, sp2)).AlignItems(ui.Center).Children(func() {
		ui.Text(c, i18n.T("Result")).FontSize(px(t, fsCaption)).FontWeight(600).TextColor(k.Text2).Grow(1)
		if ghostIconButton(c.Key(p+"-try-copy"), icons.Copy, i18n.T("Copy")).Label(i18n.T("Copy the result")).Disabled(s.Result == "").Clicked() {
			c.Services().WriteClipboard(s.Result)
			a.toast(i18n.T("Copied"))
		}
	})
	fillCard(c.Key(p + "-try-result")).Padding(px(t, sp3)).Children(func() {
		ui.Text(c, s.Result).Selectable()
	})
	if raw := strings.TrimSpace(s.Raw); raw != "" && raw != strings.TrimSpace(s.Result) {
		title := i18n.T("As recognised")
		if s.Kind == store.KindRewrite {
			title = i18n.T("Your instruction")
		}
		ui.Column(c).Gap(px(t, 4)).Children(func() {
			ui.Text(c, title).FontSize(px(t, fsCaption)).FontWeight(600).TextColor(k.Text2)
			ui.Text(c, raw).FontSize(px(t, fsCaption)).TextColor(k.Text2).Selectable()
		})
	}
	if line := uitext.TimingsText(s); line != "" {
		caption(c, line)
	}
}
