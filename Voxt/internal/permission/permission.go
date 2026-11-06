// Package permission reads and asks for the macOS privacy permissions
// VoxtNew's own process holds: the Microphone (the helper records under
// the app's identity) and Accessibility (pasting into other apps, reading
// the selection, the helper's event tap). On darwin it calls AVFoundation
// and HIServices through purego (no cgo); elsewhere Native knows nothing,
// so the app's tests run on Linux with Fake.
//
// There is no Automation permission: browser addresses (app
// enhancement's URL rules) are read through Accessibility.
//
// (The darwin code is AudioSmith's internal/dictation, permissions only.)
//
// Imports: nothing internal.
package permission

import "sync"

// State is the state of a privacy permission.
type State int

const (
	Unknown       State = iota // not on macOS, or the query failed
	NotDetermined              // never asked
	Denied
	Restricted // parental controls / MDM
	Granted
)

// System reads and asks for permissions. Methods are safe from any
// goroutine.
type System interface {
	// Microphone is the Microphone permission's state.
	Microphone() State
	// RequestMicrophone shows the system's prompt (once: afterwards
	// only System Settings changes it) and calls done with the outcome,
	// on a goroutine of its own.
	RequestMicrophone(done func(granted bool))
	// AccessibilityTrusted reports whether VoxtNew may send keys and
	// read other apps' text.
	AccessibilityTrusted() bool
	// PromptAccessibility shows the system's prompt, which adds VoxtNew
	// to Privacy & Security → Accessibility.
	PromptAccessibility()
}

// Fake is a System for tests: it reports what its fields say.
type Fake struct {
	mu       sync.Mutex
	Mic      State
	Trusted  bool
	Grant    bool // what RequestMicrophone answers
	Prompted int  // PromptAccessibility calls
}

func (f *Fake) Microphone() State { f.mu.Lock(); defer f.mu.Unlock(); return f.Mic }

func (f *Fake) RequestMicrophone(done func(bool)) {
	f.mu.Lock()
	g := f.Grant
	if g {
		f.Mic = Granted
	} else {
		f.Mic = Denied
	}
	f.mu.Unlock()
	done(g)
}

func (f *Fake) AccessibilityTrusted() bool { f.mu.Lock(); defer f.mu.Unlock(); return f.Trusted }
func (f *Fake) PromptAccessibility()       { f.mu.Lock(); f.Prompted++; f.mu.Unlock() }
