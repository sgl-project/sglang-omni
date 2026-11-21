//go:build !windows

package fileasr

func sockTempBase() string { return "/tmp" }
