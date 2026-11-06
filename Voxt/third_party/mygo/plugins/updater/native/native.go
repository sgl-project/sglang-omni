// Package native is the update window of package updater drawn in native
// UI (package ui) instead of a web page, for apps whose windows all show
// native UI: it needs no webview, so neither WebKitGTK on Linux nor the
// WebView2 Runtime on Windows. Use it in place of updater.Plugin:
//
//	mygo.Use(native.Plugin) // or native.New(updater.Options{Interval: ..., Icon: ...})
//
// The rest is package updater's: updater.MenuItem and
// updater.CheckForUpdates check from the app's menu, and the preferences
// (updater.SetAutomaticChecks, updater.OnChange, …) work the same.
package native

import (
	"github.com/egoist/mygo"
	"github.com/egoist/mygo/plugins/updater"
	"github.com/egoist/mygo/plugins/updater/internal/frontend"
)

// Plugin is the plugin with the default options.
var Plugin = New(updater.Options{})

// New returns the plugin with options. Use one of them only, and neither
// updater.Plugin nor updater.New.
func New(opts updater.Options) mygo.Plugin { return frontend.Plugin(opts, open) }
