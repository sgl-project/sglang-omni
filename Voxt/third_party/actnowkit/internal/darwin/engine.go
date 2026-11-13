//go:build darwin

package darwin

import (
	"errors"
	"fmt"
	"sync"
	"time"
	"unsafe"

	"github.com/ebitengine/purego/objc"
)

// Engine plays short sounds with low latency: an AVAudioEngine with a few
// AVAudioPlayerNodes on its main mixer, all in one format (stereo float32
// at Rate). A start schedules a buffer on a free node, taking
// microseconds. The engine runs only while sounds play: it stops after
// IdleStop without one (releasing the audio hardware) and starts again
// with the next. A stop by the system (the output device changed) is
// noticed at the next play, which starts it again.
type Engine struct {
	Rate     float64
	IdleStop time.Duration

	mu      sync.Mutex
	eng     objc.ID
	format  objc.ID
	nodes   []objc.ID
	until   []time.Time // when each node's sound ends
	gen     []uint64
	next    int
	seq     uint64
	last    time.Time
	stopper *time.Timer
}

// EngineBuffer is a sound loaded into an Engine.
type EngineBuffer struct {
	id      objc.ID
	seconds float64
}

// NewEngine builds the engine with voices player nodes (not started).
func NewEngine(rate float64, voices int) (*Engine, error) {
	if err := loadAVFoundation(); err != nil {
		return nil, err
	}
	if class("AVAudioEngine") == 0 || class("AVAudioPlayerNode") == 0 {
		return nil, errors.New("no AVAudioEngine")
	}
	e := &Engine{Rate: rate, IdleStop: 5 * time.Second}
	withPool(func() {
		e.eng = class("AVAudioEngine").Send(sel("alloc")).Send(sel("init"))
		e.format = class("AVAudioFormat").Send(sel("alloc")).Send(sel("initStandardFormatWithSampleRate:channels:"), rate, uint32(2))
		mixer := e.eng.Send(sel("mainMixerNode"))
		for range voices {
			n := class("AVAudioPlayerNode").Send(sel("alloc")).Send(sel("init"))
			e.eng.Send(sel("attachNode:"), n)
			e.eng.Send(sel("connect:to:format:"), n, mixer, e.format)
			e.nodes = append(e.nodes, n)
			e.until = append(e.until, time.Time{})
			e.gen = append(e.gen, 0)
		}
		e.eng.Send(sel("prepare"))
	})
	if e.eng == 0 || e.format == 0 {
		return nil, errors.New("AVAudioEngine could not be made")
	}
	return e, nil
}

// NewBuffer copies two channels of samples (equal lengths, at the
// engine's rate) into a buffer of the engine's format.
func (e *Engine) NewBuffer(left, right []float32) (*EngineBuffer, error) {
	n := min(len(left), len(right))
	if n == 0 {
		return nil, errors.New("no samples")
	}
	var b objc.ID
	withPool(func() {
		b = class("AVAudioPCMBuffer").Send(sel("alloc")).Send(sel("initWithPCMFormat:frameCapacity:"), e.format, uint32(n))
		if b == 0 {
			return
		}
		b.Send(sel("setFrameLength:"), uint32(n))
		chans := objc.Send[**float32](b, sel("floatChannelData")) // float * const *
		if chans == nil {
			b.Send(sel("release"))
			b = 0
			return
		}
		ptrs := unsafe.Slice(chans, 2)
		copy(unsafe.Slice(ptrs[0], n), left[:n])
		copy(unsafe.Slice(ptrs[1], n), right[:n])
	})
	if b == 0 {
		return nil, errors.New("AVAudioPCMBuffer could not be made")
	}
	return &EngineBuffer{id: b, seconds: float64(n) / e.Rate}, nil
}

// ensureRunning starts the engine when it is not running. Caller holds mu,
// inside a pool.
func (e *Engine) ensureRunning() error {
	if objc.Send[bool](e.eng, sel("isRunning")) {
		return nil
	}
	var nsErr objc.ID
	if !objc.Send[bool](e.eng, sel("startAndReturnError:"), unsafe.Pointer(&nsErr)) {
		msg := "AVAudioEngine did not start"
		if nsErr != 0 {
			msg += ": " + goString(nsErr.Send(sel("localizedDescription")))
		}
		return errors.New(msg)
	}
	return nil
}

// Play starts b at volume (0–1) on a free node (or the one whose sound
// ends first, cut off) and returns which node and start that was.
func (e *Engine) Play(b *EngineBuffer, volume float32) (node int, gen uint64, err error) {
	e.mu.Lock()
	defer e.mu.Unlock()
	now := time.Now()
	node = -1
	for j := range e.nodes {
		k := (e.next + j) % len(e.nodes)
		if !e.until[k].After(now) {
			node = k
			break
		}
	}
	if node < 0 {
		node = 0
		for k := range e.nodes {
			if e.until[k].Before(e.until[node]) {
				node = k
			}
		}
	}
	withPool(func() {
		if err = e.ensureRunning(); err != nil {
			return
		}
		n := e.nodes[node]
		if e.until[node].After(now) {
			n.Send(sel("stop")) // drops what it had scheduled
		}
		n.Send(sel("setVolume:"), volume)
		n.Send(sel("scheduleBuffer:atTime:options:completionHandler:"), b.id, objc.ID(0), uint(0), objc.ID(0))
		// The engine may have been stopped by the system since the check
		// above (rare: a device change); a node then must not play.
		if !objc.Send[bool](e.eng, sel("isRunning")) {
			err = errors.New("AVAudioEngine stopped")
			return
		}
		n.Send(sel("play"))
	})
	if err != nil {
		return -1, 0, err
	}
	e.next = (node + 1) % len(e.nodes)
	e.until[node] = now.Add(time.Duration(b.seconds*float64(time.Second)) + 50*time.Millisecond)
	e.seq++
	e.gen[node] = e.seq
	e.last = now
	e.scheduleStop()
	return node, e.seq, nil
}

// scheduleStop stops the engine IdleStop after the last play. Caller
// holds mu.
func (e *Engine) scheduleStop() {
	if e.IdleStop <= 0 {
		return
	}
	if e.stopper != nil {
		e.stopper.Stop()
	}
	e.stopper = time.AfterFunc(e.IdleStop, func() {
		e.mu.Lock()
		defer e.mu.Unlock()
		now := time.Now()
		if now.Sub(e.last) < e.IdleStop {
			return
		}
		for _, u := range e.until {
			if u.After(now) {
				e.scheduleStop()
				return
			}
		}
		withPool(func() {
			for _, n := range e.nodes {
				n.Send(sel("stop"))
			}
			e.eng.Send(sel("stop"))
		})
	})
}

// Stop stops what node plays, if it still plays the start gen.
func (e *Engine) Stop(node int, gen uint64) {
	e.mu.Lock()
	defer e.mu.Unlock()
	if node < 0 || node >= len(e.nodes) || e.gen[node] != gen || !e.until[node].After(time.Now()) {
		return
	}
	withPool(func() { e.nodes[node].Send(sel("stop")) })
	e.until[node] = time.Time{}
}

// Running reports whether the engine runs (tests).
func (e *Engine) Running() bool {
	e.mu.Lock()
	defer e.mu.Unlock()
	var r bool
	withPool(func() { r = objc.Send[bool](e.eng, sel("isRunning")) })
	return r
}

func (b *EngineBuffer) String() string { return fmt.Sprintf("buffer %.3fs", b.seconds) }
