//go:build !darwin && !(linux && (amd64 || arm64)) && !(windows && (amd64 || arm64)) && !js

package gamepad

func scan(known map[string]device) []device { return nil }
