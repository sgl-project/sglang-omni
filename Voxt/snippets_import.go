package main

// The Snippets page's import dialog: a place to choose or drop a JSON
// file, and what such a file looks like, so that it is clear what to
// import before the file chooser opens.

import (
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
		zone := ui.ButtonBase(c.Key("snippet-import-zone")).Padding(px(t, sp8), px(t, sp4)).Radius(px(t, radiusCard)).
			Border(1, k.G[5]).BorderStyle(ui.BorderDashed).Justify(ui.Center).Label(i18n.T("Choose a .json file"))
		if files := zone.DroppedFiles(); len(files) > 0 {
			local.importErr = ""
			actions.importSnippetsFile(files[0])
		}
		switch {
		case zone.FileDragOver():
			zone.BorderColor(k.Accent).Background(k.AccentSoft)
		case zone.Hovered():
			zone.Background(k.Hover)
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
	example := "[\n" +
		`  { "trigger": "` + i18n.T("my work email") + `", "text": "name@company.com" },` + "\n" +
		`  { "trigger": "` + i18n.T("my meeting link") + `", "text": "zoom.us/j/123" }` + "\n]"
	ui.Box(c.Key("snippet-import-example")).Padding(px(t, sp2), px(t, sp3)).Radius(px(t, radiusControl)).Background(k.Subtle).Children(func() {
		ui.Text(c, example).Font("monospace").FontSize(px(t, fsCaption)).Selectable()
	})
}
