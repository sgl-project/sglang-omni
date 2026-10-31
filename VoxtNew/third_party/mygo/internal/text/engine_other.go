//go:build !(windows && (amd64 || arm64)) && !darwin && !(linux && (amd64 || arm64)) && !js

package text

func newEngine() engine { return &stubEngine{} }
