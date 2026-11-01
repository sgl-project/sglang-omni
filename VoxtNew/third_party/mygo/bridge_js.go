//go:build js

package mygo

import (
	"context"

	"github.com/egoist/mygo/internal/platform"
)

// Browsers show no web pages (backend_js.go), and so have no bridge to
// talk to: these stand for bridge.go, without the JSON it takes.

func (w *Window) bridgeScript() string { return "" }

func (w *Window) pageScripts() []platform.UserScript { return nil }

func pagesLocaleChanged(string) {}

func decodeResult(raw []byte, v any) error { return errNoPage }

func (w *Window) eval(ctx context.Context, code string) ([]byte, error) { return nil, errNoPage }

func (w *Window) sendTitleBar() {}

func (w *Window) handleMessage(msg string) {}

func findScript(text string, opts FindOptions) string { return "" }

func preferencesScript() string { return "" }

func sendPreferences() {}
