//go:build js

package mygo

import (
	"github.com/egoist/mygo/internal/platform"
	"github.com/egoist/mygo/internal/web"
)

func newBackend() platform.Backend { return web.New() }
