//go:build darwin

package sounds

import (
	"fmt"
	"sync"

	"github.com/hehehai/actnowkit/internal/darwin"
)

// DefaultOutput is the platform's output. On macOS: the catalogue's WAV
// files on an AVAudioEngine (a start takes microseconds; up to
// engineVoices sounds at once; the engine stops after 5 s without a
// sound), other files on AVAudioPlayers, system sounds on NSSound.
func DefaultOutput() Output { return &macOutput{} }

const (
	engineVoices = 8
	playerVoices = 3 // AVAudioPlayers per sound for files the engine does not take
)

type macOutput struct {
	mu       sync.Mutex
	engine   *darwin.Engine
	engErr   error
	engTried bool
	buffers  map[string]*darwin.EngineBuffer
	pools    map[string]*macPool // AVAudioPlayers (not WAV, or no engine)
	system   map[string]*macPool // NSSound copies
}

type macPool struct {
	players []*darwin.AudioPlayer
	gen     []uint64
	next    int
	seq     uint64
}

func (o *macOutput) eng() *darwin.Engine {
	if !o.engTried {
		o.engTried = true
		o.engine, o.engErr = darwin.NewEngine(EngineRate, engineVoices)
	}
	return o.engine
}

func (o *macOutput) Load(key string, data []byte) error {
	o.mu.Lock()
	defer o.mu.Unlock()
	if o.buffers[key] != nil || o.pools[key] != nil {
		return nil
	}
	if pcm, ok := DecodeWAV(data); ok {
		if e := o.eng(); e != nil {
			b, err := e.NewBuffer(pcm.Left, pcm.Right)
			if err == nil {
				if o.buffers == nil {
					o.buffers = map[string]*darwin.EngineBuffer{}
				}
				o.buffers[key] = b
				return nil
			}
		}
	}
	p := &macPool{}
	for range playerVoices {
		ap, err := darwin.NewAudioPlayer(data)
		if err != nil {
			for _, q := range p.players {
				q.Release()
			}
			return err
		}
		p.players = append(p.players, ap)
		p.gen = append(p.gen, 0)
	}
	if o.pools == nil {
		o.pools = map[string]*macPool{}
	}
	o.pools[key] = p
	return nil
}

func (o *macOutput) Play(key string, volume float64) (Voice, error) {
	o.mu.Lock()
	b, p := o.buffers[key], o.pools[key]
	e := o.engine
	o.mu.Unlock()
	v := float32(min(1, max(0, volume)))
	switch {
	case b != nil:
		node, gen, err := e.Play(b, v)
		if err != nil {
			return nil, fmt.Errorf("sounds: %s: %w", key, err)
		}
		return engineVoice{e, node, gen}, nil
	case p != nil:
		o.mu.Lock()
		defer o.mu.Unlock()
		return o.start(p, key, v)
	}
	return nil, fmt.Errorf("sounds: %s is not loaded", key)
}

func (o *macOutput) PlaySystem(name string, volume float64) (Voice, error) {
	o.mu.Lock()
	defer o.mu.Unlock()
	p, ok := o.system[name]
	if !ok {
		p = &macPool{}
		for range 2 {
			ap, err := darwin.NewSystemSound(name)
			if err != nil {
				return nil, err
			}
			p.players = append(p.players, ap)
			p.gen = append(p.gen, 0)
		}
		if o.system == nil {
			o.system = map[string]*macPool{}
		}
		o.system[name] = p
	}
	return o.start(p, SystemPrefix+name, float32(min(1, max(0, volume))))
}

// start plays the first idle player of p, or the next in turn. Caller
// holds mu.
func (o *macOutput) start(p *macPool, key string, volume float32) (Voice, error) {
	i := -1
	for j := range p.players {
		k := (p.next + j) % len(p.players)
		if !p.players[k].Playing() {
			i = k
			break
		}
	}
	if i < 0 {
		i = p.next
	}
	p.next = (i + 1) % len(p.players)
	p.seq++
	p.gen[i] = p.seq
	if !p.players[i].Play(volume) {
		return nil, fmt.Errorf("sounds: %s did not start", key)
	}
	return &poolVoice{o: o, p: p, i: i, gen: p.seq}, nil
}

type engineVoice struct {
	e    *darwin.Engine
	node int
	gen  uint64
}

func (v engineVoice) Stop() { v.e.Stop(v.node, v.gen) }

type poolVoice struct {
	o   *macOutput
	p   *macPool
	i   int
	gen uint64
}

func (v *poolVoice) Stop() {
	v.o.mu.Lock()
	defer v.o.mu.Unlock()
	if v.p.gen[v.i] == v.gen {
		v.p.players[v.i].Stop()
	}
}
