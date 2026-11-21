//go:build !windows

package historyapi

// sockTempBase is where socket tests make their short paths (the macOS
// 104-byte limit wants /tmp; Windows uses the system temp).
func sockTempBase() string { return "/tmp" }
