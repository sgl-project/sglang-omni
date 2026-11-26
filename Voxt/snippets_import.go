package main

// The Snippets page's import dialog: a place to choose or drop a JSON
// file, and what such a file looks like, so that it is clear what to
// import before the file chooser opens.

import (
	"time"

	"github.com/egoist/mygo/ui"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/icons"
)

// snippetsImportDialog takes a file by a click (the file chooser) or a
// drop, and shows the problem the last file had; a file that imports
// closes it.
func snippetsImportDialog(c *ui.Context, actions snippetsActions, local *snippetsLocalUI) {
	if !local.importOpen {
		return
	}
	t := c.Theme()
	k := tok(t)
	dialogFrame(c, "snippet-import-dlg", &local.importOpen, i18n.T("Import snippets"), "", 560, func() {
		// Hover's grey at rest, a step darker under the pointer and another
		// pressed; the accent's tint while a file is dragged over it.
		zone := ui.ButtonBase(c.Key("snippet-import-zone")).Padding(px(t, sp8), px(t, sp4)).Radius(px(t, radiusCard)).
			Background(k.Hover).Justify(ui.Center).Label(i18n.T("Choose a .json file"))
		if files := zone.DroppedFiles(); len(files) > 0 {
			local.importErr = ""
			actions.importSnippetsFile(files[0])
		}
		switch {
		case zone.FileDragOver():
			zone.Background(k.AccentSoft)
		case zone.Pressed():
			zone.Background(k.G[4])
		case zone.Hovered():
			zone.Background(k.Pressed)
		}
		if zone.Children(func() {
			ui.Column(c).Gap(px(t, sp1)).AlignItems(ui.Center).Children(func() {
				ui.Icon(c, icons.UploadMinimalistic.Linear).Size(px(t, 32), px(t, 32)).TextColor(k.Text2).Margin(0, 0, px(t, sp2), 0)
				ui.Text(c, i18n.T("Choose a .json file")).FontWeight(600)
				ui.Text(c, i18n.T("Click to browse, or drop the file here.")).FontSize(px(t, fsCaption)).TextColor(k.Text2)
			})
		}).Clicked() {
			local.importErr = ""
			actions.importSnippets()
		}
		if local.importErr != "" {
			warnLine(c, local.importErr)
		}
		snippetsImportFormat(c, local)
	}, func() {
		ui.Spacer(c)
		if ui.Button(c.Key("snippet-import-cancel"), i18n.T("Cancel")).Clicked() {
			local.importOpen = false
		}
	})
}

// snippetsImportFormat is "File format", folded until a click (or a file
// that did not import) opens it on an example of two snippets.
func snippetsImportFormat(c *ui.Context, local *snippetsLocalUI) {
	t := c.Theme()
	k := tok(t)
	chevron := icons.AltArrowRight
	if local.importExample {
		chevron = icons.AltArrowDown
	}
	fold := ui.ButtonBase(c.Key("snippet-import-format")).Margin(px(t, sp2), 0, 0).Padding(px(t, 2), 0).Gap(px(t, sp1)).
		AlignItems(ui.Center).AlignSelf(ui.Start).Label(i18n.T("File format"))
	if fold.Children(func() {
		ui.Icon(c, chevron.Linear).Size(px(t, 12), px(t, 12)).TextColor(k.Text3)
		ui.Text(c, i18n.T("File format")).TextColor(k.Text2)
	}).Clicked() {
		local.importExample = !local.importExample
	}
	if !local.importExample {
		return
	}
	ui.Box(c.Key("snippet-import-example")).Padding(px(t, sp2), px(t, sp3)).Radius(px(t, radiusControl)).Background(k.Subtle).Children(func() {
		ui.Text(c, snippetsImportExample()).Font("monospace").FontSize(px(t, fsCaption)).Selectable()
	})
}

// snippetsImportExample is the file "File format" shows, in the UI's
// language; it imports as it is.
func snippetsImportExample() string {
	return "[\n" +
		`  { "trigger": "` + i18n.T("my work email") + `", "text": "name@company.com" },` + "\n" +
		`  { "trigger": "` + i18n.T("my meeting link") + `", "text": "zoom.us/j/123" }` + "\n]"
}

// How long an import's notice shows: longer when it skipped some, the
// part worth reading.
const (
	snippetsImportNoticeFor        = 2 * time.Second
	snippetsImportNoticeSkippedFor = 3 * time.Second
)

// snippetsImportNotice shows what an import did on a slightly
// see-through ink card that lets the pointer through, centred in its
// parent below top (the list, under the search), and fades it out after
// a moment. A toast at the top would cover Import….
func snippetsImportNotice(c *ui.Context, local *snippetsLocalUI, top float32) {
	if local.notice == "" {
		return
	}
	if local.noticeAt.IsZero() {
		local.noticeAt = c.Now()
	}
	left := local.noticeAt.Add(local.noticeFor).Sub(c.Now())
	if left <= 0 {
		local.notice, local.noticeAt = "", time.Time{}
		return
	}
	c.After(left)
	t := c.Theme()
	k := tok(t)
	ui.Box(c.Key("snippet-import-notice-layer")).Absolute().Left(0).Right(0).Top(top).Bottom(0).PassThrough().Justify(ui.Center).AlignItems(ui.Center).Children(func() {
		ui.Box(c.Key("snippet-import-notice")).Padding(px(t, 10), px(t, 16)).Radius(px(t, radiusControl)).
			MaxWidth(px(t, 420)).Background(k.Ink.Alpha(0.82)).TextColor(k.OnInk).Shadow(0, 6, 20, 0, ui.RGBA(0, 0, 0, 0.18)).
			Transition(ui.ElementTransition{Duration: 200 * time.Millisecond, Enter: &ui.Motion{Scale: 0.96}, Exit: &ui.Motion{}}).
			Children(func() { ui.Text(c, local.notice).Wrap() })
	})
}
