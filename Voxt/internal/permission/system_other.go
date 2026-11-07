//go:build !darwin

package permission

// Native is the system's permissions: unknown off macOS.
func Native() System { return other{} }

type other struct{}

func (other) Microphone() State                 { return Unknown }
func (other) RequestMicrophone(done func(bool)) { go done(false) }
func (other) AccessibilityTrusted() bool        { return false }
func (other) PromptAccessibility()              {}
