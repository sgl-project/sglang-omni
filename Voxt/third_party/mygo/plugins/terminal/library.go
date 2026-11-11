package terminal

import (
	_ "embed"
	"fmt"
	"runtime"
	"sync"

	"github.com/egoist/mygo/plugins/terminal/internal/library"
	"github.com/egoist/mygo/plugins/terminal/internal/vt"
)

//go:generate go run ./internal/libbuild

// natives describes the builds of libghostty-vt this package binds, one
// per platform, which `mygo build` and `mygo dev` put into apps: the CLI
// reads this file from the package's directory.
//
//go:embed mygo-plugin.json
var natives []byte

var load struct {
	once sync.Once
	path string
	err  error
}

// Load loads libghostty-vt, once, from LibraryPath. New calls it; call it
// earlier to report a missing library before showing a terminal.
func Load() error {
	load.once.Do(func() {
		if !vt.Supported {
			load.err = fmt.Errorf("terminal: not supported on %s/%s", runtime.GOOS, runtime.GOARCH)
			return
		}
		load.path, load.err = LibraryPath()
		if load.err == nil {
			load.err = vt.Load(load.path)
		}
	})
	return load.err
}

// LibraryPath returns the libghostty-vt the terminal loads, the first of:
//
//   - the file $MYGO_GHOSTTY_VT names;
//   - the one among the app's resources (mygo.PathResources), where `mygo
//     build` and `mygo dev` put it, signed with the app on macOS;
//   - the one next to the executable;
//   - the one in the user's cache, where the CLI downloads it
//     (<cache>/mygo/natives/<sha256>/), which programs that are not
//     packaged apps, as under `go run` and `go test`, download there when
//     it is missing, checking its SHA-256.
//
// Packaged apps never download it: build them with the CLI.
func LibraryPath() (string, error) { return library.Find(natives) }
