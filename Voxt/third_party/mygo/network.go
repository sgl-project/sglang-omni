package mygo

import (
	"sync"

	"github.com/egoist/mygo/internal/platform"
)

// NetworkModule reports whether the computer is online. Use the Network
// singleton.
type NetworkModule struct {
	mu       sync.Mutex
	watching bool
	// last is the status listeners last saw, so they are only called when
	// something changed: the systems report changes of routes and
	// interfaces that leave the status as it was.
	last NetworkStatus

	onChanged listeners[func(NetworkStatus)]
}

// Network reports whether the computer is online.
var Network = &NetworkModule{}

// NetworkStatus is the state of the computer's network connection. Pages
// get it as {"online": …, "metered": …}.
type NetworkStatus struct {
	// Online reports whether the system has a connection to a network
	// beyond the computer: a route out on macOS (Network.framework's
	// satisfied path) and Linux (GNetworkMonitor's network-available, a
	// default route), and on Windows a connectivity level of local
	// access or more. A network behind a captive portal, or one whose
	// way to the internet is down, counts, as it does for the browsers'
	// navigator.onLine: it does not prove that a server answers.
	Online bool `json:"online"`
	// Metered reports whether the connection costs by the amount of data
	// or is limited, so the app should send less and ask the user before
	// it downloads something large: an expensive path (cellular, a
	// phone's hotspot) or Low Data Mode on macOS, a metered connection on
	// Linux, and on Windows a cost that is fixed or variable, or one
	// that roams, is over its data limit or approaches it. It is false
	// in browsers, which do not say.
	Metered bool `json:"metered"`
}

// Status returns the computer's network state. It is safe from any
// goroutine; like the other calls that need the running app, it panics on
// the main goroutine before App.Run, and waits for Run elsewhere. On
// macOS the first call waits for the system's first answer, which comes
// within milliseconds.
func (n *NetworkModule) Status() NetworkStatus {
	needsApp("Network.Status")
	return onMainValue(func() NetworkStatus { return networkStatus(backend().Network().Status()) })
}

// OnChanged is called on the main thread when the computer's network state
// changes, with the new state.
func (n *NetworkModule) OnChanged(fn func(status NetworkStatus)) (off func()) {
	off = n.onChanged.add(fn, false)
	n.mu.Lock()
	start := !n.watching
	n.watching = true
	n.mu.Unlock()
	if start {
		App.WhenReady(func() {
			b := backend().Network()
			b.Watch()
			s := networkStatus(b.Status())
			n.mu.Lock()
			n.last = s
			n.mu.Unlock()
		})
	}
	return off
}

// changed records a status from the backend and calls the listeners when
// it differs from the last one.
func (n *NetworkModule) changed(status NetworkStatus) {
	n.mu.Lock()
	same := status == n.last
	n.last = status
	n.mu.Unlock()
	if !same {
		fire1(&n.onChanged, status)
	}
}

func networkStatus(s platform.NetworkStatus) NetworkStatus {
	return NetworkStatus{Online: s.Online, Metered: s.Metered}
}

var _ platform.Network // the backends implement it
