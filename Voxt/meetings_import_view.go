package main

import (
	"github.com/egoist/mygo/ui"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
)

// meetingImportRow is the Meetings page's progress while a recording imports.
func meetingImportRow(c *ui.Context, imp *meetingImport) {
	t := c.Theme()
	fillCard(c).Padding(px(t, sp3), px(t, sp4)).Children(func() {
		ui.Row(c).Gap(px(t, sp3)).AlignItems(ui.Center).Children(func() {
			ui.Column(c).Grow(1).MinWidth(0).Gap(px(t, sp1)).Children(func() {
				ui.Text(c, i18n.T("Importing %s…", imp.name)).SingleLine().Ellipsis("…")
				ui.Progress(c, imp.progress).Label(i18n.T("Importing %s…", imp.name))
			})
			if ui.Button(c.Key("mt-import-cancel"), i18n.T("Cancel")).Shrink(0).Clicked() {
				imp.cancel()
			}
		})
	})
}

// meetingImportDialog explains that importing needs MTD and opens its page.
func meetingImportDialog(c *ui.Context, actions meetingsActions, local *meetingsLocalUI) {
	n := local
	open := n.importNoMTD
	if open {
		dialogFrame(c, "mt-import-mtd", &open, i18n.T("MOSS-Transcribe-Diarize is not downloaded"), "", 420, func() {
			ui.Text(c, i18n.T("Importing a recording needs MOSS-Transcribe-Diarize to label its speakers. Download it on the Speech models page, then import again."))
		}, func() {
			ui.Spacer(c)
			if ui.Button(c.Key("mt-import-mtd-cancel"), i18n.T("Cancel")).AutoFocus().Clicked() {
				open = false
			}
			if primaryButton(c.Key("mt-import-mtd-open"), i18n.T("Open Speech models")).Clicked() {
				open = false
				actions.openMeetingSpeechModels()
			}
		})
		if !open {
			n.importNoMTD = false
			return
		}
	}
	open = n.importNoLanguage
	if open {
		dialogFrame(c, "mt-import-language", &open, i18n.T("Choose the recording's language"), "", 420, func() {
			ui.Text(c, i18n.T("MOSS-Transcribe-Diarize labels speakers only in Chinese or English. Choose Chinese or English as the spoken language in Meeting settings, then import again."))
		}, func() {
			ui.Spacer(c)
			if ui.Button(c.Key("mt-import-language-cancel"), i18n.T("Cancel")).AutoFocus().Clicked() {
				open = false
			}
			if primaryButton(c.Key("mt-import-language-open"), i18n.T("Open Meeting settings")).Clicked() {
				open = false
				n.setOpen = true
			}
		})
		if !open {
			n.importNoLanguage = false
			return
		}
	}
	open = n.importNoHistory
	if !open {
		return
	}
	dialogFrame(c, "mt-import-history", &open, i18n.T("Keep a history is off"), "", 420, func() {
		ui.Text(c, i18n.T("Importing a recording saves it with its whole transcript and summary. Turn on Keep a history in Settings → Privacy to import one."))
	}, func() {
		ui.Spacer(c)
		if ui.Button(c.Key("mt-import-history-cancel"), i18n.T("Cancel")).AutoFocus().Clicked() {
			open = false
		}
		if primaryButton(c.Key("mt-import-history-open"), i18n.T("Open Privacy settings")).Clicked() {
			open = false
			actions.openMeetingPrivacySettings()
		}
	})
	if !open {
		n.importNoHistory = false
	}
}
