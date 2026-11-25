//go:build !darwin

package macx

import (
	"errors"
	"time"
)

// FadingSound is a sound that can end with a fade: macOS only.
type FadingSound struct{}

// NewFadingSound fails off macOS.
func NewFadingSound([]byte) (*FadingSound, error) {
	return nil, errors.New("fading sounds need macOS")
}

func (*FadingSound) Play(float32) (uint64, bool)   { return 0, false }
func (*FadingSound) Playing() bool                 { return false }
func (*FadingSound) Stop(uint64)                   {}
func (*FadingSound) FadeOut(uint64, time.Duration) {}
