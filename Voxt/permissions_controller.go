package main

import (
	"time"

	"github.com/egoist/mygo"
	"github.com/sgl-project/sglang-omni/Voxt/internal/helper/proto"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/permission"
)

type permissionServices interface {
	OpenURL(url string)
}

// permissionsState is what the section last read.
type permissionsState struct {
	permSys permission.System
	perms   struct {
		mic     permission.State
		speech  permission.State
		access  bool
		checked bool
	}
	permissionsCache struct {
		key   permissionsCacheKey
		state permissionsViewState
		valid bool
	}
}

type permissionsCacheKey struct {
	locale        string
	microphone    permission.State
	speech        permission.State
	accessibility bool
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

func (a *app) permissionsModel() permissionsViewState {
	key := permissionsCacheKey{
		locale: i18n.Locale(), microphone: a.perms.mic, speech: a.perms.speech, accessibility: a.perms.access,
	}
	if a.permissionsCache.valid && a.permissionsCache.key == key {
		return a.permissionsCache.state
	}
	state := buildPermissionsViewState(i18n.CurrentTranslator(), permissionsStateInput{
		Microphone: a.perms.mic, Accessibility: a.perms.access, SpeechRecognition: a.perms.speech,
	})
	a.permissionsCache.key, a.permissionsCache.state, a.permissionsCache.valid = key, state, true
	return state
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
	case proto.PermissionSystemAudio:
		answered() // only the helper can make the tap macOS asks on
	default:
		a.async(func() { sys.RequestMicrophone(recheck) })
	}
}

func (a *app) requestPermission(which string) {
	a.askPermission(which, nil)
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
func (a *app) openPrivacy(services permissionServices, url string) {
	if a.win == nil {
		services.OpenURL(url)
		return
	}
	go mygo.Shell.OpenExternal(url)
}
