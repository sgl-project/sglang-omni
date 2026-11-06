package ui

import (
	"fmt"
	"log"
	"runtime"
	"strconv"
	"strings"
)

// maxWarnings is how many warnings the inspector keeps, the latest.
const maxWarnings = 50

// duplicateKey reports an element given the key of another element of
// the pass under the same parent: they have one ID, and so share one
// state, so that the pointer, the focus and scrolling meant for one go to
// the other. A Tester panics (engine.strict), so that tests fail where the
// key was given; apps log it once, and the inspector lists it.
func (rt *engine) duplicateKey(id uint64, k any) {
	msg := fmt.Sprintf("ui: two elements under one parent have the key %#v, and share one state: give each a key of its own", k)
	if at := callSite(); at != "" {
		msg += " (" + at + ")"
	}
	if rt.strict {
		panic(msg)
	}
	if rt.dupKeys == nil {
		rt.dupKeys = map[uint64]bool{}
	}
	if rt.dupKeys[id] {
		return
	}
	rt.dupKeys[id] = true
	log.Print(msg)
	rt.warn(msg)
}

// warn notes a warning for the inspector.
func (rt *engine) warn(msg string) {
	if len(rt.warnings) == maxWarnings {
		copy(rt.warnings, rt.warnings[1:])
		rt.warnings = rt.warnings[:maxWarnings-1]
	}
	rt.warnings = append(rt.warnings, msg)
	rt.insp.changed = true
}

// uiPackage prefixes the names of the functions of package ui.
const uiPackage = "github.com/egoist/mygo/ui."

// callSite returns the file and line of the innermost call outside package
// ui on the stack: the app's code that built an element, or "".
func callSite() string {
	var pcs [32]uintptr
	n := runtime.Callers(2, pcs[:])
	frames := runtime.CallersFrames(pcs[:n])
	for {
		f, more := frames.Next()
		if f.Function != "" && !strings.HasPrefix(f.Function, uiPackage) && !strings.HasPrefix(f.Function, "runtime.") && !strings.HasPrefix(f.Function, "testing.") {
			return f.File + ":" + strconv.Itoa(f.Line)
		}
		if !more {
			return ""
		}
	}
}
