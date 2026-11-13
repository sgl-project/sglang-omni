//go:build darwin

package darwin

import (
	"errors"
	"fmt"
	"runtime"
	"sync"
	"unsafe"

	"github.com/ebitengine/purego"
	"github.com/ebitengine/purego/objc"
)

var (
	avOnce sync.Once
	avErr  error
)

func loadAVFoundation() error {
	if err := loadFrameworks(); err != nil {
		return err
	}
	avOnce.Do(func() {
		if _, err := purego.Dlopen("/System/Library/Frameworks/AVFoundation.framework/AVFoundation", purego.RTLD_GLOBAL|purego.RTLD_NOW); err != nil {
			avErr = err
			return
		}
		if class("AVAudioPlayer") == 0 {
			avErr = errors.New("no AVAudioPlayer")
		}
	})
	return avErr
}

// AudioPlayer is a retained AVAudioPlayer (or NSSound, for system sounds)
// ready to play.
type AudioPlayer struct {
	id     objc.ID
	system bool
}

// NewAudioPlayer makes an AVAudioPlayer of an audio file's bytes (WAV,
// CAF, AIFF, M4A …) and prepares it (prepareToPlay). Its first play
// still takes 15–60 ms (Engine is the fast path; this one is for files
// Engine does not take).
func NewAudioPlayer(data []byte) (*AudioPlayer, error) {
	if err := loadAVFoundation(); err != nil {
		return nil, err
	}
	if len(data) == 0 {
		return nil, errors.New("no audio data")
	}
	var p objc.ID
	var errDesc string
	withPool(func() {
		d := class("NSData").Send(sel("dataWithBytes:length:"), unsafe.Pointer(&data[0]), uintptr(len(data)))
		var nsErr objc.ID
		p = class("AVAudioPlayer").Send(sel("alloc")).Send(sel("initWithData:error:"), d, unsafe.Pointer(&nsErr))
		if p == 0 {
			errDesc = "AVAudioPlayer initWithData failed"
			if nsErr != 0 {
				errDesc += ": " + goString(nsErr.Send(sel("localizedDescription")))
			}
			return
		}
		objc.Send[bool](p, sel("prepareToPlay"))
	})
	runtime.KeepAlive(data)
	if p == 0 {
		return nil, errors.New(errDesc)
	}
	return &AudioPlayer{id: p}, nil
}

// NewSystemSound is a copy of the system sound name (NSSound soundNamed:,
// copied so that copies play over each other).
func NewSystemSound(name string) (*AudioPlayer, error) {
	if err := loadFrameworks(); err != nil {
		return nil, err
	}
	var p objc.ID
	withPool(func() {
		if s := class("NSSound").Send(sel("soundNamed:"), nsString(name)); s != 0 {
			p = s.Send(sel("copy")) // +1, ours
		}
	})
	if p == 0 {
		return nil, fmt.Errorf("no system sound %q", name)
	}
	return &AudioPlayer{id: p, system: true}, nil
}

// Play starts the sound from its beginning at volume (0–1). A player that
// still plays is paused and rewound rather than stopped: stop undoes
// prepareToPlay, and the next start would wait for it again.
func (a *AudioPlayer) Play(volume float32) bool {
	ok := false
	withPool(func() {
		if a.system {
			a.id.Send(sel("stop"))
			a.id.Send(sel("setVolume:"), volume)
			ok = objc.Send[bool](a.id, sel("play"))
			return
		}
		if objc.Send[bool](a.id, sel("isPlaying")) {
			a.id.Send(sel("pause"))
		}
		a.id.Send(sel("setCurrentTime:"), float64(0))
		a.id.Send(sel("setVolume:"), volume)
		ok = objc.Send[bool](a.id, sel("play"))
	})
	return ok
}

// Stop stops the sound (an AVAudioPlayer is paused and rewound, so it
// stays ready to play).
func (a *AudioPlayer) Stop() {
	withPool(func() {
		if a.system {
			a.id.Send(sel("stop"))
			return
		}
		a.id.Send(sel("pause"))
		a.id.Send(sel("setCurrentTime:"), float64(0))
	})
}

// Playing reports whether the sound plays.
func (a *AudioPlayer) Playing() bool {
	var b bool
	withPool(func() { b = objc.Send[bool](a.id, sel("isPlaying")) })
	return b
}

// Duration is the sound's length in seconds.
func (a *AudioPlayer) Duration() float64 {
	var d float64
	withPool(func() { d = objc.Send[float64](a.id, sel("duration")) })
	return d
}

// Release lets the player go.
func (a *AudioPlayer) Release() {
	if a.id != 0 {
		withPool(func() {
			a.id.Send(sel("stop"))
			a.id.Send(sel("release"))
		})
		a.id = 0
	}
}
