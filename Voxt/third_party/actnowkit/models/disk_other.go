//go:build !(darwin || linux)

package models

import "os"

func freeSpace(string) int64 { return -1 }

func linkedElsewhere(os.FileInfo) bool { return false }
