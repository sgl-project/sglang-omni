//go:build mygo_noinspector

package ui

import (
	"time"

	"github.com/egoist/mygo/internal/platform"
)

// Production builds (mygo build, unless MYGO_INSPECTOR=1 or --debug) leave
// the inspector out, as their developer tools are off: this stands in for
// it, never open, so that it takes no room in the binary.
type inspector struct {
	enabled, open, changed bool
	selected               uint64
}

func (in *inspector) contentWidth(w float32) float32                                { return w }
func (in *inspector) lap(int)                                                       {}
func (in *inspector) repainted(time.Duration)                                       {}
func (in *inspector) noteSource()                                                   {}
func (in *inspector) noteKey(uint64, any)                                           {}
func (in *inspector) pointer(*engine, platform.SurfaceEvent, float32, float32) bool { return false }
func (in *inspector) snapshot(*engine, *node)                                       {}
func (in *inspector) paintHighlight(*engine, *Painter, float32)                     {}
func (rt *engine) toggleInspector()                                                 {}
func (rt *engine) inspectKey(Modifiers, Key) bool                                   { return false }
func (rt *engine) buildInspector(*context, float32, float32, float32)               {}
