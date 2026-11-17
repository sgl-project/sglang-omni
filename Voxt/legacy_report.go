package main

// What an import from the previous Voxt did (legacy_import.go), grouped
// so it reads at a glance: what came over, what was left out and why,
// and what is left to do, each to-do with the button that does it. The
// setup guide's card shows it; an import from elsewhere (the sidebar's
// card, the dictionary's ••• menu) shows it in a dialog.

import (
	"strings"

	"github.com/egoist/mygo/ui"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/icons"
	"github.com/sgl-project/sglang-omni/Voxt/internal/local"
)

// legacyReport is an import's result: failed says why nothing came over
// ("" when it did).
type legacyReport struct {
	failed   string
	imported []string
	skipped  []string
	todo     []legacyTodo
}

// legacyTodo is something left to do after an import: kind's setup id to
// complete in its dialog.
type legacyTodo struct {
	text string
	kind local.Kind
	id   string
}

// text is the report in one line, for the log and the smoke test.
func (r legacyReport) text() string {
	if r.failed != "" {
		return r.failed
	}
	parts := []string{i18n.T("Imported") + ": " + strings.Join(r.imported, "; ")}
	if len(r.skipped) > 0 {
		parts = append(parts, i18n.T("Left out")+": "+strings.Join(r.skipped, "; "))
	}
	if len(r.todo) > 0 {
		var todo []string
		for _, d := range r.todo {
			todo = append(todo, d.text)
		}
		parts = append(parts, i18n.T("To do")+": "+strings.Join(todo, "; "))
	}
	return strings.Join(parts, ". ")
}

// codexLeftOut is the result's line for the previous Voxt's Codex: "" when
// it had none.
func (l legacySettings) codexLeftOut() string {
	var used []string
	for i, name := range []string{i18n.T("Transcribe"), i18n.T("Translate"), i18n.T("Rewrite")} {
		if l.codexFeatures[i] {
			used = append(used, name)
		}
	}
	switch {
	case len(used) > 0:
		return i18n.T("Codex (used by %s): sign in to ChatGPT (Codex) or choose another language model", strings.Join(used, i18n.T(", ")))
	case l.codexSetup:
		return i18n.T("Codex setup: sign in to ChatGPT (Codex) on the Language models page")
	}
	return ""
}

// legacyReportView shows a report: a group per kind of result, under its
// heading; a to-do's Complete setup opens the setup's dialog on its page
// (leaving the guide with its way back, when it shows the report).
func (a *app) legacyReportView(c *ui.Context, r legacyReport) {
	t := c.Theme()
	k := tok(t)
	if r.failed != "" {
		statusMark(c, icons.DangerTriangle, k.Warning, r.failed)
		return
	}
	group := func(key string, ic icons.Icon, col ui.Color, title string, lines func()) {
		ui.Column(c.Key(key)).Gap(px(t, 2)).Children(func() {
			ui.Row(c).Gap(px(t, 6)).AlignItems(ui.Center).Children(func() {
				ui.Icon(c, ic.Linear).Size(px(t, 14), px(t, 14)).TextColor(col).Shrink(0)
				sectionTitleText(c, title)
			})
			ui.Column(c).Gap(px(t, 2)).Padding(0, 0, 0, px(t, 20)).Children(lines)
		})
	}
	group("legacy-imported", icons.CheckCircle, k.Success, i18n.T("Imported"), func() {
		for _, line := range r.imported {
			caption(c, line)
		}
	})
	if len(r.skipped) > 0 {
		group("legacy-skipped", icons.InfoCircle, k.Text2, i18n.T("Left out"), func() {
			for _, line := range r.skipped {
				caption(c, line)
			}
		})
	}
	if len(r.todo) > 0 {
		group("legacy-todo", icons.DangerTriangle, k.Warning, i18n.T("To do"), func() {
			for i, d := range r.todo {
				ui.Row(c.Key("todo-" + itoa(i))).Gap(px(t, sp2)).AlignItems(ui.Center).Children(func() {
					caption(c, d.text).Grow(1).MinWidth(0)
					if ghostButton(c.Key("complete"), i18n.T("Complete setup")).Label(i18n.T("Complete setup of %s", lookupService(d.kind, d.id).name)).Clicked() {
						a.completeLegacyTodo(d)
					}
				})
			}
		})
	}
}

// completeLegacyTodo opens the dialog of a to-do's setup: from the guide,
// which waits with its way back on that page, or from the report's dialog.
func (a *app) completeLegacyTodo(d legacyTodo) {
	a.ob.legacyOpen = false
	if a.ob.open {
		page := remotePath
		if d.kind == local.ASR {
			page = modelsPath
		}
		a.obLeave(page) // onboarding.go
	}
	a.completeSetup(d.kind, d.id) // remote_link.go
}

// legacyReportDialog is the report of an import from outside the guide.
func (a *app) legacyReportDialog(c *ui.Context) {
	if !a.ob.legacyOpen {
		return
	}
	dialogFrame(c, "legacy-report", &a.ob.legacyOpen, i18n.T("Imported from the previous Voxt"), i18n.T("The previous Voxt keeps all its data."), 520, func() {
		a.legacyReportView(c, a.ob.legacy)
	}, func() {
		ui.Spacer(c)
		if primaryButton(c.Key("legacy-report-done"), i18n.T("Done")).Clicked() {
			a.ob.legacyOpen = false
		}
	})
}
