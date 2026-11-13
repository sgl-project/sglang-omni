//go:build !darwin

package sounds

// DefaultOutput is the platform's output: Silent off macOS.
func DefaultOutput() Output { return Silent{} }
