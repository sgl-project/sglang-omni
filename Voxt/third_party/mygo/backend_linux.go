//go:build linux && (amd64 || arm64)

package mygo

import (
	"github.com/egoist/mygo/internal/linux"
	"github.com/egoist/mygo/internal/platform"
)

func newBackend() platform.Backend { return linux.New() }
