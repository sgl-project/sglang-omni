//go:build !(darwin || linux || windows)

package hfhub

func freeBytes(string) int64 { return -1 }
