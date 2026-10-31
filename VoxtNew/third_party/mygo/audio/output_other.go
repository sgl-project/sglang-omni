//go:build !darwin && !(linux && (amd64 || arm64)) && !(windows && (amd64 || arm64)) && !js

package audio

func openDevice() (device, int, error) { return nil, 0, errUnsupported }
