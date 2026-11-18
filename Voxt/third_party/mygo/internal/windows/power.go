//go:build windows && (amd64 || arm64)

package windows

import (
	"time"
	"unsafe"

	"github.com/egoist/mygo/internal/platform"
)

var (
	wtsapi32 = systemDLL("wtsapi32.dll")

	procWTSRegisterSessionNotification = wtsapi32.NewProc("WTSRegisterSessionNotification")
	procPowerCreateRequest             = kernel32.NewProc("PowerCreateRequest")
	procPowerSetRequest                = kernel32.NewProc("PowerSetRequest")
	procPowerClearRequest              = kernel32.NewProc("PowerClearRequest")
	procCloseHandle                    = kernel32.NewProc("CloseHandle")
	procGetSystemPowerStatus           = kernel32.NewProc("GetSystemPowerStatus")
	procGetTickCount                   = kernel32.NewProc("GetTickCount")
	procGetLastInputInfo               = user32.NewProc("GetLastInputInfo")
)

// Messages about power and the session, which the application window gets.
const (
	wmPowerBroadcast      = 0x0218
	wmWTSSessionChange    = 0x02B1
	pbtAPMSuspend         = 0x4
	pbtAPMResumeAutomatic = 0x12
	wtsSessionLock        = 0x7
	wtsSessionUnlock      = 0x8
)

func (b *Backend) Power() platform.Power     { return power{b} }
func (b *Backend) Secrets() platform.Secrets { return secrets{} }

type power struct{ b *Backend }

func (p power) Watch() {
	p.b.watchingPower = true
	const notifyForThisSession = 0
	procWTSRegisterSessionNotification.Call(p.b.appHwnd, notifyForThisSession)
}

// powerMessage reports the power and session messages of the application
// window.
func (b *Backend) powerMessage(m uint32, wp uintptr) {
	if !b.watchingPower {
		return
	}
	switch {
	case m == wmPowerBroadcast && wp == pbtAPMSuspend:
		b.h.PowerEvent("suspend")
	case m == wmPowerBroadcast && wp == pbtAPMResumeAutomatic:
		b.h.PowerEvent("resume")
	case m == wmWTSSessionChange && wp == wtsSessionLock:
		b.h.PowerEvent("lock-screen")
	case m == wmWTSSessionChange && wp == wtsSessionUnlock:
		b.h.PowerEvent("unlock-screen")
	}
}

// reasonContext mirrors REASON_CONTEXT with a simple reason string.
type reasonContext struct {
	version, flags uint32
	reason         *uint16
	_              [2]uintptr // the rest of the union
}

func (p power) KeepAwake(display bool, reason string) func() {
	const (
		contextSimpleString = 0x1
		requestDisplay      = 0 // PowerRequestDisplayRequired
		requestSystem       = 1 // PowerRequestSystemRequired
	)
	ctx := reasonContext{flags: contextSimpleString, reason: u16(reason)}
	h, _, _ := procPowerCreateRequest.Call(uintptr(unsafe.Pointer(&ctx)))
	if h == 0 || h == ^uintptr(0) {
		return nil
	}
	requests := []uintptr{requestSystem}
	if display {
		requests = append(requests, requestDisplay)
	}
	for _, r := range requests {
		procPowerSetRequest.Call(h, r)
	}
	return func() {
		for _, r := range requests {
			procPowerClearRequest.Call(h, r)
		}
		procCloseHandle.Call(h)
	}
}

func (p power) OnBattery() bool {
	var status struct {
		acLineStatus, batteryFlag, batteryLifePercent, systemStatusFlag uint8
		batteryLifeTime, batteryFullLifeTime                            uint32
	}
	if r, _, _ := procGetSystemPowerStatus.Call(uintptr(unsafe.Pointer(&status))); r == 0 {
		return false
	}
	return status.acLineStatus == 0
}

func (p power) IdleTime() time.Duration {
	info := struct{ size, time uint32 }{size: 8}
	if r, _, _ := procGetLastInputInfo.Call(uintptr(unsafe.Pointer(&info))); r == 0 {
		return 0
	}
	now, _, _ := procGetTickCount.Call()
	return time.Duration(uint32(now)-info.time) * time.Millisecond
}
