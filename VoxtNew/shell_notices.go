package main

// The notifications (the notification centre's, internal/notify), shown
// as a stack of cards at the foot of the sidebar, over the Settings row:
// the newest whole, up to two more peeking under it; × dismisses one and
// the next comes up. The icon rail shows a dot on its Settings icon, which
// then opens the same stack in a popover. There is no status bar: what it
// showed (the services Transcribe uses, the helper) becomes a card while
// something is wrong (setProblem), with the action that fixes it.

import (
	"context"
	"slices"

	"github.com/egoist/mygo/ui"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/core"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/helper"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/i18n"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/icons"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/local"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/uitext"
)

// The problems' notifications: one ID each, so a problem shows once.
const (
	problemModels = "problem/models"
	problemHelper = "problem/helper"
)

// linkRestartHelper is the link of the helper's card: its action
// restarts the helper (noticeAction), it is not a deep link.
const linkRestartHelper = "action:restart-helper"

// noticeLayers is how many cards show under the newest, peeking.
const noticeLayers = 2

// noticePosted adds a notification to the centre's snapshot: the sidebar
// shows it at once, the rail as a dot on Settings.
func (a *app) noticePosted(n core.Notification) {
	a.notices = slices.DeleteFunc(a.notices, func(x core.Notification) bool { return x.ID == n.ID })
	a.notices = slices.Insert(a.notices, 0, n)
}

// notify posts a notification (on any goroutine).
func (a *app) notify(sev core.Severity, title, body, link string) {
	a.notes.Post(core.Notification{Severity: sev, Title: title, Body: body, Link: link})
}

// dismissNotice removes one notification (its card's ×).
func (a *app) dismissNotice(id string) {
	a.notices = slices.DeleteFunc(a.notices, func(n core.Notification) bool { return n.ID == id })
	a.problemSeen(id)
	notes := a.notes
	a.async(func() { notes.Dismiss(context.Background(), id) })
}

// clearNotices removes them all.
func (a *app) clearNotices() {
	for _, n := range a.notices {
		a.problemSeen(n.ID)
	}
	a.notices = nil
	a.noticesOpen = false
	notes := a.notes
	a.async(func() { notes.Clear(context.Background()) })
}

// problemSeen keeps a problem's card the user dismissed away, this run,
// while the problem stays the same (setProblem).
func (a *app) problemSeen(id string) {
	code, ok := a.problemCode[id]
	if !ok {
		return
	}
	if a.problemDismissed == nil {
		a.problemDismissed = map[string]string{}
	}
	a.problemDismissed[id] = code
}

// setProblem shows a problem's card (n, under id; code says which form
// of the problem it is), unless the same shows already or the user
// dismissed this form of it this run; nil withdraws it (solved).
func (a *app) setProblem(id, code string, n *core.Notification) {
	i := slices.IndexFunc(a.notices, func(x core.Notification) bool { return x.ID == id })
	if n == nil {
		delete(a.problemDismissed, id)
		delete(a.problemCode, id)
		if i >= 0 {
			a.notices = slices.Delete(a.notices, i, i+1)
			notes := a.notes
			a.async(func() { notes.Dismiss(context.Background(), id) })
		}
		return
	}
	if a.problemCode == nil {
		a.problemCode = map[string]string{}
	}
	a.problemCode[id] = code
	if a.problemDismissed[id] == code {
		return
	}
	if i >= 0 && a.notices[i].Title == n.Title && a.notices[i].Body == n.Body && a.notices[i].Link == n.Link {
		return
	}
	n.ID = id
	a.notes.Post(*n) // noticePosted adds it
}

// checkModels shows a card while no speech model or no language model
// is installed (what the status bar said).
func (a *app) checkModels() {
	if a.notes == nil {
		return
	}
	asrOK := len(a.installed(local.ASR)) > 0
	llmOK := len(a.installed(local.LLM)) > 0
	var n *core.Notification
	code := ""
	switch {
	case !asrOK && !llmOK:
		code = "both"
		n = &core.Notification{Title: i18n.T("No models installed"), Body: i18n.T("Install a speech model and a language model to start.")}
	case !asrOK:
		code = "asr"
		n = &core.Notification{Title: i18n.T("No speech model"), Body: i18n.T("Transcribe, Translate and Rewrite need one to hear you.")}
	case !llmOK:
		code = "llm"
		n = &core.Notification{Title: i18n.T("No language model"), Body: i18n.T("Translate and Rewrite need one; Transcribe types what it hears.")}
	}
	if n != nil {
		n.Severity, n.Link = core.SeverityWarning, modelsPath
	}
	a.setProblem(problemModels, code, n)
}

// checkHelper shows a card while the helper does not run (it crashed,
// is missing or out of date, or did not restart).
func (a *app) checkHelper() {
	if a.notes == nil {
		return
	}
	st := a.helperStatus()
	if st.Reason == helper.FailUpdated {
		// Not the helper's fault: the update's card says what to do.
		a.setProblem(problemHelper, "", nil)
		a.appUpdated() // updates.go
		return
	}
	if st.State != helper.Failed {
		if st.State == helper.Ready {
			a.setProblem(problemHelper, "", nil)
		}
		return
	}
	a.setProblem(problemHelper, string(st.Reason), &core.Notification{Severity: core.SeverityError, Title: uitext.HelperFailText(st),
		Body: i18n.T("Recording and the shortcuts wait for it."), Link: linkRestartHelper})
}

// cardAction runs a card's action: restart the helper, or open its link.
func (a *app) cardAction(n core.Notification) {
	a.noticesOpen = false
	switch n.Link {
	case linkRestartHelper:
		a.restartHelper()
		return
	case linkRelaunch:
		a.relaunch() // updates.go
		return
	}
	a.openURL(n.Link)
}

// noticeStack is the cards, newest on top, the next ones peeking under it,
// with a count and Clear all over them when there are more than one.
func (a *app) noticeStack(c *ui.Context) {
	t := c.Theme()
	k := tok(t)
	list := a.notices // as this frame shows them: Clear all or × may change a.notices
	if len(list) == 0 {
		return
	}
	layers := min(len(list)-1, noticeLayers)
	step, inset := px(t, 6), px(t, 8)
	ui.Column(c.Key("notices")).Gap(px(t, sp1)).Children(func() {
		if len(list) > 1 {
			ui.Row(c).AlignItems(ui.Center).Padding(0, px(t, 2)).Children(func() {
				ui.Text(c, i18n.N(len(list), "%d notification", "%d notifications", len(list))).
					FontSize(px(t, fsMicro)).TextColor(k.Text3).SingleLine().Grow(1).MinWidth(0)
				b := ui.ButtonBase(c.Key("notices-clear")).Padding(px(t, 1), px(t, 4)).Radius(px(t, 4)).Label(i18n.T("Clear all"))
				if b.Hovered() {
					b.Background(k.Hover)
				}
				b.Children(func() {
					ui.Text(c, i18n.T("Clear all")).FontSize(px(t, fsMicro)).TextColor(k.Text2).SingleLine()
				})
				if b.Clicked() {
					a.clearNotices()
				}
			})
		}
		ui.Column(c).Children(func() {
			a.noticeCard(c, list[0])
			// The cards under it: their foot shows, a step lower and
			// narrower each (the bottom of a rounded card, clipped).
			r := px(t, radiusControl+2)
			for i := 1; i <= layers; i++ {
				f := float32(i)
				ui.Box(c).Height(step).Margin(0, inset*f).Clip().Shrink(0).Children(func() {
					ui.Box(c).Absolute().Left(0).Right(0).Bottom(0).Height(step+2*r).
						Radius(r).Background(k.Bg).Border(1, k.Line).Opacity(1 - 0.15*f)
				})
			}
		})
	})
}

// noticeCard is a notification: its severity's icon, title, one or two
// lines, the action, and × at the top right.
func (a *app) noticeCard(c *ui.Context, n core.Notification) {
	t := c.Theme()
	k := tok(t)
	ic, col := icons.InfoCircle, k.Text2
	switch n.Severity {
	case core.SeverityError:
		ic, col = icons.DangerCircle, k.Danger
	case core.SeverityWarning:
		ic, col = icons.DangerTriangle, k.Warning
	case core.SeveritySuccess:
		ic, col = icons.CheckCircle, k.Success
	}
	card := ui.Column(c.Key("notice-"+n.ID)).Gap(px(t, 2)).Padding(px(t, 6), px(t, sp1), px(t, sp2), px(t, sp2)).
		Radius(px(t, radiusControl+2)).Background(k.Bg).Border(1, k.Line).Shadow(0, 2, 8, 0, ui.RGBA(0, 0, 0, 0.06)).
		Tooltip(i18n.DateTime(n.Time))
	card.Children(func() {
		ui.Row(c).Gap(px(t, 6)).AlignItems(ui.Start).Children(func() {
			ui.Icon(c, ic.Linear).Size(px(t, 14), px(t, 14)).TextColor(col).Shrink(0).Margin(px(t, 3), 0, 0, 0)
			ui.Text(c, n.Title).FontSize(px(t, fsCaption)).FontWeight(600).MaxLines(2).Ellipsis("…").Grow(1).MinWidth(0).Margin(px(t, 2), 0, 0, 0)
			b := ghostBase(c.Key("notice-close")).Size(px(t, 20), px(t, 20)).Center().Shrink(0).Label(i18n.T("Dismiss")).Tooltip(i18n.T("Dismiss"))
			b.Children(func() {
				ui.Icon(c, icons.Close.Linear).Size(px(t, 12), px(t, 12)).TextColor(k.Text3)
			})
			if b.Clicked() {
				a.dismissNotice(n.ID)
			}
		})
		// Under the title, from its start.
		ui.Column(c).Gap(px(t, 2)).Padding(0, px(t, sp1), 0, px(t, 20)).Children(func() {
			if n.Body != "" {
				ui.Text(c, n.Body).FontSize(px(t, fsMicro)).TextColor(k.Text2).MaxLines(3).Ellipsis("…")
			}
			if n.Link != "" {
				label := cardActionLabel(n.Link)
				b := ui.ButtonBase(c.Key("notice-act")).Padding(px(t, 2), 0).Label(label).AlignSelf(ui.Start).MinWidth(0)
				b.Children(func() {
					txt := ui.Text(c, label).FontSize(px(t, fsMicro)).FontWeight(500).TextColor(k.AccentInk).SingleLine().Ellipsis("…").MinWidth(0)
					if b.Hovered() {
						txt.Underline()
					}
				})
				if b.Clicked() {
					a.cardAction(n)
				}
			}
		})
	})
}

// noticePopover is the rail's: the same stack, then Settings, which the
// rail's icon opens when there is no notification.
func (a *app) noticePopover(c *ui.Context) {
	t := c.Theme()
	ui.Column(c).Width(px(t, 260)).Gap(px(t, sp2)).Padding(px(t, sp1)).Children(func() {
		a.noticeStack(c)
		ui.Divider(c)
		b := ghostBase(c.Key("notices-settings")).Padding(px(t, 6), px(t, sp2)).Gap(px(t, sp2)).Justify(ui.Start).Label(i18n.T("Settings"))
		b.Children(func() {
			ui.Icon(c, icons.Settings.Linear).Size(px(t, 16), px(t, 16)).TextColor(tok(t).Text2)
			ui.Text(c, i18n.T("Settings")).SingleLine().Grow(1)
			ui.Text(c, "⌘,").FontSize(px(t, fsMicro)).TextColor(tok(t).Text3)
		})
		if b.Clicked() {
			a.noticesOpen = false
			a.navigate("/settings")
		}
	})
}

// cardActionLabel names a card's action by its link.
func cardActionLabel(link string) string {
	switch link {
	case linkRestartHelper:
		return i18n.T("Restart the helper")
	case linkRelaunch:
		return i18n.T("Restart VoxtNew")
	case modelsPath:
		return i18n.T("Open Local models")
	}
	return i18n.T("Open")
}
