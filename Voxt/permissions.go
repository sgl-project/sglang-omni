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
	"time"

	"github.com/egoist/mygo"
	"github.com/egoist/mygo/ui"
	"github.com/sgl-project/sglang-omni/Voxt/internal/helper/proto"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/icons"
	"github.com/sgl-project/sglang-omni/Voxt/internal/permission"
	"github.com/sgl-project/sglang-omni/Voxt/internal/uitext"
)


// permissionsState is what the section last read.
type permissionsState struct {
	permSys permission.System
	perms   struct {
		mic     permission.State
		speech  permission.State
		access  bool
		checked bool
	}
}

// permState is a permission.State from the helper's (proto.Perm*).
func permState(s string) permission.State {
	switch s {
	case proto.PermGranted:
		return permission.Granted
	case proto.PermDenied:
		return permission.Denied
	case proto.PermRestricted:
		return permission.Restricted
	case proto.PermNotDetermined:
		return permission.NotDetermined
	}
	return permission.Unknown
}

// helperPerms takes the helper's answer to permissions.*.
func (a *app) helperPerms(r proto.PermissionsResult, err error) {
	if err != nil {
		return // the section keeps what internal/permission read
	}
	a.perms.mic, a.perms.access, a.perms.checked = permState(r.Microphone), r.Accessibility, true
	if r.SpeechRecognition != "" { // else an older helper: keep what internal/permission read
		a.perms.speech = permState(r.SpeechRecognition)
	}
}

func (a *app) openPermissions() {
	if a.permSys == nil {
		a.permSys = permission.Native()
	}
}

// recheckPermissions reads the permissions: at once here (cheap system
// calls), then as the helper sees them.
func (a *app) recheckPermissions() {
	a.perms.mic, a.perms.access, a.perms.checked = a.permSys.Microphone(), a.permSys.AccessibilityTrusted(), true
	a.perms.speech = a.permSys.SpeechRecognition()
	if a.hstatus.Has(proto.MethodPermissionsStatus) {
		helperCall(a, proto.MethodPermissionsStatus, nil, 5*time.Second, a.helperPerms)
	}
}

// askPermission shows macOS's prompt for a permission
// (proto.Permission*), through the helper, else from here; then (nil:
// nothing) runs once it was answered.
func (a *app) askPermission(which string, then func()) {
	answered := func() {
		if then != nil {
			then()
		}
	}
	if a.hstatus.Has(proto.MethodPermissionsRequest) {
		helperCall(a, proto.MethodPermissionsRequest, proto.PermissionsRequestParams{Permission: which}, 5*time.Minute, func(r proto.PermissionsResult, err error) {
			a.helperPerms(r, err)
			answered()
		})
		return
	}
	sys := a.permSys
	recheck := func(bool) { a.post(func() { a.recheckPermissions(); answered() }) }
	switch which {
	case proto.PermissionAccessibility:
		sys.PromptAccessibility()
		a.recheckPermissions()
		answered()
	case proto.PermissionSpeechRecognition:
		a.async(func() { sys.RequestSpeechRecognition(recheck) })
	default:
		a.async(func() { sys.RequestMicrophone(recheck) })
	}
}

// askRequired asks for the permissions Voxt needs and lacks, in turn:
// the microphone's prompt, then, once it was answered, Accessibility's
// (which adds Voxt to its list in System Settings). One denied before
// shows no prompt: its row's Open System Settings is the way.
func (a *app) askRequired() {
	ask := func() {
		if !a.perms.access {
			a.askPermission(proto.PermissionAccessibility, nil)
		}
	}
	if a.perms.mic == permission.NotDetermined {
		a.askPermission(proto.PermissionMicrophone, ask)
		return
	}
	ask()
}

// openPrivacy opens a pane of System Settings.
func (a *app) openPrivacy(c *ui.Context, url string) {
	if a.win == nil {
		c.Services().OpenURL(url)
		return
	}
	go mygo.Shell.OpenExternal(url)
}

// permissionsSettings is the Permissions section of the settings dialog.
func (a *app) permissionsSettings(c *ui.Context) {
	t := c.Theme()
	if !a.perms.checked {
		a.recheckPermissions()
	}
	badge := func(text string, ok bool) {
		if ok {
			statusMark(c, icons.CheckCircle, t.Success, text)
		} else {
			statusMark(c, icons.DangerTriangle, t.Warning, text)
		}
	}
	if a.perms.mic == permission.NotDetermined || !a.perms.access {
		ui.Row(c).Children(func() {
			if ui.Button(c.Key("perm-ask-required"), i18n.T("Ask for the required permissions")).Tooltip(i18n.T("macOS asks for each in turn: the microphone, then Accessibility.")).Clicked() {
				a.askRequired()
			}
		})
	}
	sectionTitle(c, i18n.T("Required"))

	micText, micOK := uitext.PermissionText(a.perms.mic)
	settingRow(c, "perm-mic", i18n.T("Microphone"), "", func() {
		badge(micText, micOK)
		if a.perms.mic == permission.NotDetermined {
			if ghostButton(c.Key("perm-mic-ask"), i18n.T("Ask")).Clicked() {
				a.askPermission(proto.PermissionMicrophone, nil)
			}
		}
		if !micOK {
			if ghostIconButton(c.Key("perm-mic-open"), icons.SquareArrowRightUp, i18n.T("Open System Settings")).Label(i18n.T("Open System Settings")).Clicked() {
				a.openPrivacy(c, privacyMicrophone)
			}
		}
	})
	accText, accOK := i18n.T("Not allowed"), a.perms.access
	if accOK {
		accText = i18n.T("Allowed")
	}
	settingRow(c, "perm-access", i18n.T("Accessibility"), i18n.T("Lets Voxt type results, read the selected text and the browser tab's address, and use single keys like Right ⌥ as shortcuts."), func() {
		badge(accText, accOK)
		if !accOK {
			if ghostButton(c.Key("perm-access-ask"), i18n.T("Ask")).Tooltip(i18n.T("Adds Voxt to the list in System Settings")).Clicked() {
				a.askPermission(proto.PermissionAccessibility, nil)
			}
			if ghostIconButton(c.Key("perm-access-open"), icons.SquareArrowRightUp, i18n.T("Open System Settings")).Label(i18n.T("Open System Settings")).Clicked() {
				a.openPrivacy(c, privacyAccessibility)
			}
		}
	})
	sectionTitle(c, i18n.T("Optional"))
	settingRow(c, "perm-system-audio", i18n.T("System audio recording (for meetings)"), i18n.T("Lets Voxt record sound from other apps during a meeting. macOS asks the first time a meeting records system audio."), func() {
		if ghostIconButton(c.Key("perm-system-audio-open"), icons.SquareArrowRightUp, i18n.T("Open System Settings")).Label(i18n.T("Open System Settings")).Clicked() {
			a.openPrivacy(c, privacyAudioCapture)
		}
	})
	speechText, speechOK := uitext.PermissionText(a.perms.speech)
	settingRow(c, "perm-speech", i18n.T("Speech Recognition"), i18n.T("Needed only for System dictation."), func() {
		badge(speechText, speechOK)
		if a.perms.speech == permission.NotDetermined {
			if ghostButton(c.Key("perm-speech-ask"), i18n.T("Ask")).Clicked() {
				a.askPermission(proto.PermissionSpeechRecognition, nil)
			}
		}
		if !speechOK {
			if ghostIconButton(c.Key("perm-speech-open"), icons.SquareArrowRightUp, i18n.T("Open System Settings")).Label(i18n.T("Open System Settings")).Clicked() {
				a.openPrivacy(c, privacySpeech)
			}
		}
	})
	ui.Row(c).Gap(px(t, sp2)).AlignItems(ui.Center).Margin(px(t, sp2), 0, 0, 0).Children(func() {
		if secondaryIconButton(c.Key("perm-recheck"), icons.Refresh, i18n.T("Check again")).Label(i18n.T("Check again")).Clicked() {
			a.recheckPermissions()
		}
		infoTip(c.Key("perm-note"), i18n.T("After changing a permission, you may need to reopen Voxt."))
	})
}
