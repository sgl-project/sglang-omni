package main

// Settings → Permissions (/settings/permissions): what macOS lets
// VoxtNew do, why each permission matters, and how to grant it. The
// helper, which records and posts keys, reads and asks for them
// (permissions.*): macOS checks its permissions against the app, so they
// are VoxtNew's. Before the helper is ready the Microphone and
// Accessibility are read here (internal/permission). The section reads
// them again when it opens, when VoxtNew comes to the front and with
// Check again. Browser addresses are read through Accessibility only, so
// there is no Automation permission to ask for.

import (
	"time"

	"github.com/egoist/mygo"
	"github.com/egoist/mygo/ui"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/helper/proto"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/i18n"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/icons"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/permission"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/uitext"
)

// System Settings panes.
const (
	privacyMicrophone    = "x-apple.systempreferences:com.apple.preference.security?Privacy_Microphone"
	privacyAccessibility = "x-apple.systempreferences:com.apple.preference.security?Privacy_Accessibility"
)

// permissionsState is what the section last read.
type permissionsState struct {
	permSys permission.System
	perms   struct {
		mic     permission.State
		access  bool
		checked bool
	}
}

// micState is a permission.State from the helper's microphone state.
func micState(s string) permission.State {
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
	a.perms.mic, a.perms.access, a.perms.checked = micState(r.Microphone), r.Accessibility, true
}

func (a *app) openPermissions() {
	if a.permSys == nil {
		a.permSys = permission.Native()
	}
}

// recheckPermissions reads the Microphone and Accessibility: at once
// here (cheap system calls), then as the helper sees them.
func (a *app) recheckPermissions() {
	a.perms.mic, a.perms.access, a.perms.checked = a.permSys.Microphone(), a.permSys.AccessibilityTrusted(), true
	if a.hstatus.Has(proto.MethodPermissionsStatus) {
		helperCall(a, proto.MethodPermissionsStatus, nil, 5*time.Second, a.helperPerms)
	}
}

// askPermission shows macOS's prompt for the Microphone or Accessibility
// (proto.Permission*), through the helper, else from here.
func (a *app) askPermission(which string) {
	if a.hstatus.Has(proto.MethodPermissionsRequest) {
		helperCall(a, proto.MethodPermissionsRequest, proto.PermissionsRequestParams{Permission: which}, 5*time.Minute, a.helperPerms)
		return
	}
	sys := a.permSys
	if which == proto.PermissionAccessibility {
		sys.PromptAccessibility()
		a.recheckPermissions()
		return
	}
	a.async(func() { sys.RequestMicrophone(func(bool) { a.post(a.recheckPermissions) }) })
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
	caption(c, i18n.T("macOS asks you before VoxtNew may record or work with other apps. Each row says what needs it; you can change them any time in System Settings."))

	micText, micOK := uitext.PermissionText(a.perms.mic)
	settingRow(c, "perm-mic", i18n.T("Microphone"), i18n.T("Transcribe, translate and rewrite by voice record while you hold or tap their shortcut."), func() {
		badge(micText, micOK)
		if a.perms.mic == permission.NotDetermined {
			if ghostButton(c.Key("perm-mic-ask"), i18n.T("Ask")).Clicked() {
				a.askPermission(proto.PermissionMicrophone)
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
	settingRow(c, "perm-access", i18n.T("Accessibility"), i18n.T("Lets VoxtNew type results where your cursor is, read the selected text and the address of the browser tab in front, and use Right ⌥ and other single keys as shortcuts. Without it results are copied for you to paste."), func() {
		badge(accText, accOK)
		if !accOK {
			if ghostButton(c.Key("perm-access-ask"), i18n.T("Ask")).Tooltip(i18n.T("Adds VoxtNew to the list in System Settings")).Clicked() {
				a.askPermission(proto.PermissionAccessibility)
			}
			if ghostIconButton(c.Key("perm-access-open"), icons.SquareArrowRightUp, i18n.T("Open System Settings")).Label(i18n.T("Open System Settings")).Clicked() {
				a.openPrivacy(c, privacyAccessibility)
			}
		}
	})
	ui.Row(c).Gap(px(t, sp2)).AlignItems(ui.Center).Margin(px(t, sp2), 0, 0, 0).Children(func() {
		if secondaryIconButton(c.Key("perm-recheck"), icons.Refresh, i18n.T("Check again")).Label(i18n.T("Check again")).Clicked() {
			a.recheckPermissions()
		}
		infoTip(c.Key("perm-note"), i18n.T("After you change a permission, macOS may ask to quit and reopen VoxtNew. Updating an ad hoc signed build can make macOS forget Accessibility: turn it off and on again."))
	})
}
