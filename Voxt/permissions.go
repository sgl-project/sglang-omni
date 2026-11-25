package main

// Settings → Permissions (/settings/permissions): what macOS lets
// VoxtNew do, why each permission matters, and how to grant it. The
// helper, which records and posts keys, reads and asks for them
// (permissions.*): macOS checks its permissions against the app, so they
// are VoxtNew's. Speech Recognition is System dictation's, which the
// helper runs too. Before the helper is ready they are read here
// (internal/permission). The section reads
// them again when it opens, when VoxtNew comes to the front and with
// Check again. Browser addresses are read through Accessibility only, so
// there is no Automation permission to ask for.

import (
	"github.com/egoist/mygo/ui"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/icons"
)

type permissionsActions interface {
	askRequired()
	requestPermission(which string)
	recheckPermissions()
	openPrivacy(services permissionServices, url string)
}

// permissionsSettings is the Permissions section of the settings dialog.
// The dialog is too narrow for a row's status and buttons beside its name,
// so they go under it on every row, not only on rows that happen to wrap.
func permissionsSettings(c *ui.Context, state permissionsViewState, actions permissionsActions) {
	permissionRows(c, state, actions, settingRowUnder)
}

// permissionRows lists the permissions, each a row made by row.
func permissionRows(c *ui.Context, state permissionsViewState, actions permissionsActions, row func(c *ui.Context, key, title, desc string, control func())) {
	t := c.Theme()
	badge := func(permission permissionRowViewState) {
		if !permission.Known {
			statusMark(c, icons.InfoCircle, tok(t).Text2, permission.Status).Tooltip(permission.Tip)
		} else if permission.Allowed {
			statusMark(c, icons.CheckCircle, t.Success, permission.Status)
		} else {
			statusMark(c, icons.DangerTriangle, t.Warning, permission.Status)
		}
	}
	if state.AskRequired {
		ui.Row(c).Children(func() {
			if ui.Button(c.Key("perm-ask-required"), i18n.T("Ask for the required permissions")).Tooltip(i18n.T("macOS asks for each in turn: the microphone, then Accessibility.")).Clicked() {
				actions.askRequired()
			}
		})
	}
	sectionTitle(c, i18n.T("Required"))

	row(c, "perm-mic", i18n.T("Microphone"), "", func() {
		badge(state.Microphone)
		if state.Microphone.Ask {
			if ghostButton(c.Key("perm-mic-ask"), i18n.T("Ask")).Clicked() {
				actions.requestPermission(state.Microphone.PermissionID)
			}
		}
		if state.Microphone.Open {
			if ghostIconButton(c.Key("perm-mic-open"), icons.SquareArrowRightUp, i18n.T("Open System Settings")).Label(i18n.T("Open System Settings")).Clicked() {
				actions.openPrivacy(c.Services(), state.Microphone.SettingsURL)
			}
		}
	})
	row(c, "perm-access", i18n.T("Accessibility"), i18n.T("Lets Voxt type results, read the selected text and the browser tab's address, and use single keys like Right ⌥ as shortcuts."), func() {
		badge(state.Accessibility)
		if state.Accessibility.Ask {
			if ghostButton(c.Key("perm-access-ask"), i18n.T("Ask")).Tooltip(i18n.T("Adds Voxt to the list in System Settings")).Clicked() {
				actions.requestPermission(state.Accessibility.PermissionID)
			}
		}
		if state.Accessibility.Open {
			if ghostIconButton(c.Key("perm-access-open"), icons.SquareArrowRightUp, i18n.T("Open System Settings")).Label(i18n.T("Open System Settings")).Clicked() {
				actions.openPrivacy(c.Services(), state.Accessibility.SettingsURL)
			}
		}
	})
	sectionTitle(c, i18n.T("Optional"))
	row(c, "perm-system-audio", i18n.T("System audio recording"), i18n.T("Lets Voxt record sound from other apps during a meeting. macOS asks the first time a meeting records system audio."), func() {
		// macOS has no public read for this permission.
		badge(state.SystemAudio)
		if state.SystemAudio.Ask {
			if ghostButton(c.Key("perm-system-audio-ask"), i18n.T("Ask")).Tooltip(i18n.T("Adds Voxt to the list in System Settings")).Clicked() {
				actions.requestPermission(state.SystemAudio.PermissionID)
			}
		}
		if state.SystemAudio.Open {
			if ghostIconButton(c.Key("perm-system-audio-open"), icons.SquareArrowRightUp, i18n.T("Open System Settings")).Label(i18n.T("Open System Settings")).Clicked() {
				actions.openPrivacy(c.Services(), state.SystemAudio.SettingsURL)
			}
		}
	})
	row(c, "perm-speech", i18n.T("Speech Recognition"), i18n.T("Needed only for System dictation."), func() {
		badge(state.SpeechRecognition)
		if state.SpeechRecognition.Ask {
			if ghostButton(c.Key("perm-speech-ask"), i18n.T("Ask")).Clicked() {
				actions.requestPermission(state.SpeechRecognition.PermissionID)
			}
		}
		if state.SpeechRecognition.Open {
			if ghostIconButton(c.Key("perm-speech-open"), icons.SquareArrowRightUp, i18n.T("Open System Settings")).Label(i18n.T("Open System Settings")).Clicked() {
				actions.openPrivacy(c.Services(), state.SpeechRecognition.SettingsURL)
			}
		}
	})
	ui.Row(c).Gap(px(t, sp2)).AlignItems(ui.Center).Margin(px(t, sp2), 0, 0, 0).Children(func() {
		if secondaryIconButton(c.Key("perm-recheck"), icons.Refresh, i18n.T("Check again")).Label(i18n.T("Check again")).Clicked() {
			actions.recheckPermissions()
		}
		infoTip(c.Key("perm-note"), i18n.T("After changing a permission, you may need to reopen Voxt."))
	})
}
