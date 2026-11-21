//go:build js

package mygo

import "github.com/egoist/mygo/internal/platform"

// ProtocolModule registers custom URL schemes served by Go, for web pages,
// which browsers do not show. Builds for them leave out net/http, and with
// it Handle, HandleFunc and FileServer: no scheme is handled.
type ProtocolModule struct{}

// Protocol serves custom URL schemes such as app:// from Go.
var Protocol = &ProtocolModule{}

// Unhandle removes the handler of scheme.
func (p *ProtocolModule) Unhandle(scheme string) {}

// IsHandled reports whether scheme has a handler: none has.
func (p *ProtocolModule) IsHandled(scheme string) bool { return false }

func (p *ProtocolModule) schemes() []string { return nil }

func (h *windowHandler) SchemeRequest(req *platform.SchemeRequest) {}

func (h *windowHandler) SchemeDownload(rawURL string) {}
