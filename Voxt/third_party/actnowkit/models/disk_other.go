//go:build !(darwin || linux || windows)

package models

import "os"

func freeSpace(string) int64 { return -1 }

func linkedElsewhere(string, os.FileInfo, uint32) bool { return false }
