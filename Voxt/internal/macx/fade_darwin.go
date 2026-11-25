//go:build darwin

package macx

import (
	"errors"
	"runtime"
	"sync"
	"time"
	"unsafe"

	"github.com/ebitengine/purego"
	"github.com/ebitengine/purego/objc"
)

var av struct {
	once sync.Once
	err  error
}

func loadAV() error {
	if err := load(); err != nil {
		return err
	}
	av.once.Do(func() {
		if _, err := purego.Dlopen("/System/Library/Frameworks/AVFoundation.framework/AVFoundation", purego.RTLD_GLOBAL|purego.RTLD_NOW); err != nil {
			av.err = err
			return
		}
		if class("AVAudioPlayer") == 0 {
			av.err = errors.New("no AVAudioPlayer")
		}
	})
	return av.err
}

var (
	selAlloc        = objc.RegisterName("alloc")
	selDataBytes    = objc.RegisterName("dataWithBytes:length:")
	selInitData     = objc.RegisterName("initWithData:error:")
	selPrepare      = objc.RegisterName("prepareToPlay")
	selIsPlaying    = objc.RegisterName("isPlaying")
	selPause        = objc.RegisterName("pause")
	selPlay         = objc.RegisterName("play")
	selSetTime      = objc.RegisterName("setCurrentTime:")
	selSetVolume    = objc.RegisterName("setVolume:")
	selFadeVolume   = objc.RegisterName("setVolume:fadeDuration:")
	selDescribeErr  = objc.RegisterName("localizedDescription")
	errNoAudioBytes = errors.New("no audio data")
)

// FadingSound is a sound file on an AVAudioPlayer, which can end with a
// fade (setVolume:fadeDuration:); actnowkit's engine can only cut a sound
// off. The player is prepared, so a start takes some 15–60 ms.
type FadingSound struct {
	mu  sync.Mutex
	id  objc.ID
	gen uint64 // counts the starts: a fade's stop ends only its own start
}

// NewFadingSound prepares a player of an audio file's bytes (WAV, CAF …).
func NewFadingSound(data []byte) (*FadingSound, error) {
	if err := loadAV(); err != nil {
		return nil, err
	}
	if len(data) == 0 {
		return nil, errNoAudioBytes
	}
	var p objc.ID
	var msg string
	pooled(func() {
		d := class("NSData").Send(selDataBytes, unsafe.Pointer(&data[0]), uintptr(len(data)))
		var nsErr objc.ID
		p = class("AVAudioPlayer").Send(selAlloc).Send(selInitData, d, unsafe.Pointer(&nsErr))
		if p == 0 {
			msg = "AVAudioPlayer initWithData failed"
			if nsErr != 0 {
				msg += ": " + goString(nsErr.Send(selDescribeErr))
			}
			return
		}
		objc.Send[bool](p, selPrepare)
	})
	runtime.KeepAlive(data)
	if p == 0 {
		return nil, errors.New(msg)
	}
	return &FadingSound{id: p}, nil
}

// Play starts the sound from its beginning at volume (0–1) and returns
// the start's number. A start that still plays is paused and rewound, not
// stopped (stop would undo prepareToPlay).
func (s *FadingSound) Play(volume float32) (uint64, bool) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.gen++
	ok := false
	pooled(func() {
		if objc.Send[bool](s.id, selIsPlaying) {
			s.id.Send(selPause)
		}
		s.id.Send(selSetTime, float64(0))
		s.id.Send(selSetVolume, volume)
		ok = objc.Send[bool](s.id, selPlay)
	})
	return s.gen, ok
}

// Playing reports whether the sound plays.
func (s *FadingSound) Playing() bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	var b bool
	pooled(func() { b = objc.Send[bool](s.id, selIsPlaying) })
	return b
}

// Stop ends start gen now.
func (s *FadingSound) Stop(gen uint64) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if gen == s.gen {
		s.rewind()
	}
}

// FadeOut takes start gen to silence over d, then stops it.
func (s *FadingSound) FadeOut(gen uint64, d time.Duration) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if gen != s.gen {
		return
	}
	pooled(func() { s.id.Send(selFadeVolume, float32(0), d.Seconds()) })
	time.AfterFunc(d+20*time.Millisecond, func() {
		s.mu.Lock()
		defer s.mu.Unlock()
		if gen == s.gen {
			s.rewind()
		}
	})
}

// rewind pauses the player at its beginning; the caller holds mu.
func (s *FadingSound) rewind() {
	pooled(func() {
		s.id.Send(selPause)
		s.id.Send(selSetTime, float64(0))
	})
}
