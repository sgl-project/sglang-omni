//go:build windows

package historyapi

// sockTempBase: the system temp is short enough on Windows.
func sockTempBase() string { return "" }
