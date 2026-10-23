//go:build linux && (amd64 || arm64)

package linux

import (
	"github.com/egoist/mygo/internal/platform"
)

// GNetworkMonitor is a singleton GLib keeps up to date, so a status is a
// plain read and its network-changed signal drives the event.
type network struct{ b *Backend }

func (b *Backend) Network() platform.Network { return network{b} }

// Status reads the monitor's current view. The first call creates the
// singleton, on the main thread.
func (network) Status() platform.NetworkStatus {
	m := gNetworkMonitorGetDefault()
	if m == 0 {
		return platform.NetworkStatus{}
	}
	// network-available is a default route, of IPv4 or IPv6: a
	// connectivity other than G_NETWORK_CONNECTIVITY_LOCAL, as macOS's
	// satisfied path.
	return platform.NetworkStatus{
		Online:  gNetworkMonitorGetNetworkAvailable(m),
		Metered: gNetworkMonitorGetNetworkMetered(m),
	}
}

// Watch connects network-changed. The core calls it once.
func (network) Watch() {
	if m := gNetworkMonitorGetDefault(); m != 0 {
		connect(m, "network-changed", cbNetworkChanged, 0)
	}
}
