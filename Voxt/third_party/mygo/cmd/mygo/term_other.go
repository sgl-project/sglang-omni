//go:build !darwin && !dragonfly && !freebsd && !linux && !netbsd && !openbsd && !windows

package main

import "os"

func termWidth(f *os.File) int { return 0 }

func enableVT(f *os.File) bool { return true }
