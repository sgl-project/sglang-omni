//go:build windows && (amd64 || arm64)

package windows

import (
	"syscall"
	"unsafe"

	"github.com/egoist/mygo/internal/platform"
)

var iphlpapi = systemDLL("iphlpapi.dll")

// Both are on Windows 10 2004 and later.
var (
	procGetNetworkConnectivityHint          = iphlpapi.NewProc("GetNetworkConnectivityHint")
	procNotifyNetworkConnectivityHintChange = iphlpapi.NewProc("NotifyNetworkConnectivityHintChange")
)

// NL_NETWORK_CONNECTIVITY_LEVEL_HINT and NL_NETWORK_CONNECTIVITY_COST_HINT.
const (
	nlConnectivityLevelUnknown = 0
	nlConnectivityLevelNone    = 1
	nlConnectivityLevelHidden  = 5

	nlConnectivityCostFixed    = 2
	nlConnectivityCostVariable = 3
)

// nlNetworkConnectivityHint mirrors NL_NETWORK_CONNECTIVITY_HINT of nldef.h:
// the connectivity level and cost are C enums (int, 4 bytes each), followed
// by three BOOLEANs.
type nlNetworkConnectivityHint struct {
	Level            int32
	Cost             int32
	ApproachingLimit uint8
	OverLimit        uint8
	Roaming          uint8
}

func (b *Backend) Network() platform.Network { return network{b} }

type network struct{ b *Backend }

// Status reads the system's connectivity hint, the aggregate of what the
// Network Connectivity Status Indicator found out on every interface.
func (network) Status() platform.NetworkStatus {
	var h nlNetworkConnectivityHint
	if !has(procGetNetworkConnectivityHint) {
		// An older Windows does not tell. Assume the connection works and
		// is not metered, as the browsers do.
		return platform.NetworkStatus{Online: true}
	}
	if r, _, _ := procGetNetworkConnectivityHint.Call(uintptr(unsafe.Pointer(&h))); r != 0 {
		return platform.NetworkStatus{Online: true}
	}
	return networkStatus(h)
}

// networkStatus reads a connectivity hint as macOS and Linux report
// theirs. Online is a connection beyond the computer, as their route out:
// local access (a network, which the system found no way to the internet
// from), internet access and constrained internet access (behind a
// captive portal) count, no connectivity and a hidden network, whose
// routes the system ignores, do not; an unknown level is taken to work,
// as on an older Windows. Metered follows Microsoft's guidance for
// metered connections (Windows.Networking.Connectivity's ConnectionCost):
// any cost but unrestricted, or roaming, over the data limit or
// approaching it, asks to hold back.
func networkStatus(h nlNetworkConnectivityHint) platform.NetworkStatus {
	return platform.NetworkStatus{
		Online: h.Level != nlConnectivityLevelNone && h.Level != nlConnectivityLevelHidden,
		Metered: h.Cost == nlConnectivityCostFixed || h.Cost == nlConnectivityCostVariable ||
			h.Roaming != 0 || h.OverLimit != 0 || h.ApproachingLimit != 0,
	}
}

// networkHandle is the registration of the hint's changes, which lives
// as long as the app.
var networkHandle uintptr

// Watch asks to be told when the connectivity hint changes. The core calls
// it once, after the application window exists.
func (n network) Watch() {
	if !has(procNotifyNetworkConnectivityHintChange) {
		return
	}
	// The callback runs on a thread of the system's, so it only posts to
	// the application window, and the main thread reads the hint again.
	// Its second parameter is the hint by value, which amd64 passes as a
	// pointer and arm64 in two registers, so it is not read.
	cb := syscall.NewCallback(func(ctx, hint uintptr) uintptr {
		postMessage(n.b.appHwnd, wmAppNetwork, 0, 0)
		return 0
	})
	procNotifyNetworkConnectivityHintChange.Call(cb, 0, 0, uintptr(unsafe.Pointer(&networkHandle)))
}
