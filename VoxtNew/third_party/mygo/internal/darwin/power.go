//go:build darwin

package darwin

import "time"

type power struct{ b *Backend }

// powerEvents maps the notifications the app delegate observes to power
// events.
var powerEvents = map[string]string{
	"NSWorkspaceWillSleepNotification": "suspend",
	"NSWorkspaceDidWakeNotification":   "resume",
	"com.apple.screenIsLocked":         "lock-screen",
	"com.apple.screenIsUnlocked":       "unlock-screen",
}

func (p power) Watch() {
	withPool(func() {
		observe := func(center id, name string) {
			send(center, "addObserver:selector:name:object:", uintptr(p.b.delegate), uintptr(sel("mygoPowerNotification:")), uintptr(nsString(name)), 0)
		}
		workspace := send(send(class("NSWorkspace"), "sharedWorkspace"), "notificationCenter")
		observe(workspace, "NSWorkspaceWillSleepNotification")
		observe(workspace, "NSWorkspaceDidWakeNotification")
		// Screen locking is only announced to other processes.
		distributed := send(class("NSDistributedNotificationCenter"), "defaultCenter")
		observe(distributed, "com.apple.screenIsLocked")
		observe(distributed, "com.apple.screenIsUnlocked")
	})
}

func (p power) KeepAwake(display bool, reason string) func() {
	const idleSystemSleepDisabled, idleDisplaySleepDisabled = 1 << 20, 1 << 40
	opts := uint64(idleSystemSleepDisabled)
	if display {
		opts |= idleDisplaySleepDisabled
	}
	var token id
	withPool(func() {
		info := send(class("NSProcessInfo"), "processInfo")
		token = retain(send(info, "beginActivityWithOptions:reason:", uintptr(opts), uintptr(nsString(reason))))
	})
	return func() {
		send(send(class("NSProcessInfo"), "processInfo"), "endActivity:", uintptr(token))
		release(token)
	}
}

func (p power) OnBattery() bool {
	if iopsCopyPowerSourcesInfo == nil {
		return false
	}
	info := iopsCopyPowerSourcesInfo()
	if info == 0 {
		return false
	}
	defer cfRelease(info)
	return goString(iopsGetProvidingPowerSourceType(info)) == "Battery Power"
}

func (p power) IdleTime() time.Duration {
	const combinedSessionState, anyInputEvent = 0, ^uint32(0)
	return time.Duration(cgEventSourceSecondsSinceLastType(combinedSessionState, anyInputEvent) * float64(time.Second))
}
