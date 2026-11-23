//go:build js

package web

import (
	"syscall/js"

	"github.com/egoist/mygo/internal/platform"
)

func (b *Backend) Network() platform.Network { return network{b} }

// network is the browser's navigator.onLine, which tells no cost.
type network struct{ b *Backend }

func (network) Status() platform.NetworkStatus {
	return platform.NetworkStatus{Online: js.Global().Get("navigator").Get("onLine").Bool()}
}

// Watch listens to the window's online and offline events.
func (n network) Watch() {
	changed := n.b.funcOf(func(js.Value, []js.Value) any {
		n.b.sync(func() { n.b.h.NetworkChanged(n.Status()) })
		return nil
	})
	for _, event := range []string{"online", "offline"} {
		js.Global().Call("addEventListener", event, changed)
	}
}
