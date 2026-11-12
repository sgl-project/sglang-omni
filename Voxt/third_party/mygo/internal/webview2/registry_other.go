//go:build !windows

package webview2

// regString finds nothing: only Windows has a registry.
func regString(machine bool, path, name string) string { return "" }
