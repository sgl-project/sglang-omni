//go:build windows

package models

import (
	"os"
	"syscall"
)

// SameVolume reports whether a and b are folders on one volume, where
// a model is hard-linked instead of copied. On Windows two folders share
// a volume when their volume serial numbers match.
func SameVolume(a, b string) bool {
	va := volumeSerial(a)
	vb := volumeSerial(b)
	return va != 0 && va == vb
}

// volumeSerial is the volume serial number of path's volume, or 0 when it
// cannot be read.
func volumeSerial(path string) uint32 {
	f, err := os.Open(path)
	if err != nil {
		return 0
	}
	defer f.Close()
	var info syscall.ByHandleFileInformation
	if err := syscall.GetFileInformationByHandle(syscall.Handle(f.Fd()), &info); err != nil {
		return 0
	}
	return info.VolumeSerialNumber
}
