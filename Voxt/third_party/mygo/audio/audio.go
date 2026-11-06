// Package audio plays sound: effects decoded into memory, music streamed
// as it plays, and sound an app makes itself, mixed in Go and sent to the
// system's audio output:
//
//	click, err := audio.DecodeSound(clickWAV) // decoded once
//	...
//	click.Play() // as often as it sounds, over itself
//
//	music, err := audio.OpenStream(f) // decoded as it plays
//	music.SetLoop(true).SetVolume(0.6).Play()
//
// WAV decodes out of the box; import audio/vorbis for Ogg Vorbis and
// audio/mp3 for MP3, which register themselves as image's formats do.
//
// The output opens when something first plays and closes a few seconds
// after everything stopped. Where no output can open, players still play
// on time, unheard, and Err says why. Every function and method is safe
// to call from any goroutine.
package audio

import (
	"bufio"
	"bytes"
	"errors"
	"io"
	"math"
	"runtime"
	"sync"
	"time"
)

// The output's format: every player is resampled to it.
const (
	sampleRate = 48000
	channels   = 2
)

// mu guards the voices playing and the state of every player.
var mu sync.Mutex

// A Decoder decodes audio as interleaved float32 samples in [-1, 1], one
// per channel for each frame. Formats registered with RegisterFormat
// return one; implement one to play sound an app generates itself.
type Decoder interface {
	// SampleRate returns the frames per second.
	SampleRate() int
	// Channels returns the samples per frame: 1 for mono, 2 for stereo
	// (left, right); later channels of more are not played.
	Channels() int
	// Read decodes the next samples into buf, a whole number of frames,
	// and returns how many it wrote, with io.EOF at the end.
	Read(buf []float32) (int, error)
	// SetPosition moves to frame, counted from the start.
	SetPosition(frame int64) error
	// Len returns the frames from the start to the end, or -1 if unknown.
	Len() int64
}

type format struct {
	name, magic string
	open        func(io.ReadSeeker) (Decoder, error)
}

var (
	formatsMu sync.Mutex
	formats   []format
)

// RegisterFormat registers a format for DecodeSound, OpenStream and
// NewDecoder: magic is the start of its files, where '?' matches any
// byte, and open returns a decoder reading r from its start. A package
// for a format calls it in its init function.
func RegisterFormat(name, magic string, open func(r io.ReadSeeker) (Decoder, error)) {
	formatsMu.Lock()
	defer formatsMu.Unlock()
	formats = append(formats, format{name, magic, open})
}

// ErrFormat reports data of no format registered.
var ErrFormat = errors.New("audio: unknown format")

func match(magic string, b []byte) bool {
	if len(magic) > len(b) {
		return false
	}
	for i := 0; i < len(magic); i++ {
		if magic[i] != '?' && magic[i] != b[i] {
			return false
		}
	}
	return true
}

// NewDecoder returns a decoder of r's format, and the format's name.
func NewDecoder(r io.ReadSeeker) (Decoder, string, error) {
	start, err := r.Seek(0, io.SeekCurrent)
	if err != nil {
		return nil, "", err
	}
	var head [16]byte
	n, err := io.ReadFull(r, head[:])
	if err != nil && err != io.ErrUnexpectedEOF {
		return nil, "", err
	}
	if _, err := r.Seek(start, io.SeekStart); err != nil {
		return nil, "", err
	}
	formatsMu.Lock()
	fs := formats
	formatsMu.Unlock()
	for _, f := range fs {
		if match(f.magic, head[:n]) {
			d, err := f.open(r)
			if err != nil {
				return nil, f.name, err
			}
			return d, f.name, nil
		}
	}
	return nil, "", ErrFormat
}

// A Sound is audio decoded into memory, for effects that play often and
// over themselves: each Play is a new Player.
type Sound struct {
	samples  []float32
	channels int
	rate     int
}

// DecodeSound decodes data of a registered format into memory.
func DecodeSound(data []byte) (*Sound, error) {
	d, _, err := NewDecoder(bytes.NewReader(data))
	if err != nil {
		return nil, err
	}
	return ReadSound(d)
}

// ReadSound decodes everything d has into memory.
func ReadSound(d Decoder) (*Sound, error) {
	ch := d.Channels()
	if ch < 1 || d.SampleRate() <= 0 {
		return nil, errors.New("audio: decoder of no channels or sample rate")
	}
	keep := min(ch, 2)
	var samples []float32
	if n := d.Len(); n > 0 {
		samples = make([]float32, 0, n*int64(keep))
	}
	buf := make([]float32, 4096/ch*ch)
	for {
		n, err := d.Read(buf)
		n -= n % ch
		for i := 0; i < n; i += ch {
			samples = append(samples, buf[i:i+keep]...)
		}
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, err
		}
		if n == 0 {
			return nil, io.ErrNoProgress
		}
	}
	return &Sound{samples: samples, channels: keep, rate: d.SampleRate()}, nil
}

// NewSound returns a sound of samples, interleaved frames of 1 (mono) or 2
// (stereo) channels at sampleRate frames per second, as an app generates
// them. The sound keeps samples: don't change them afterwards.
func NewSound(sampleRate, channels int, samples []float32) *Sound {
	if channels != 1 && channels != 2 {
		panic("audio: NewSound takes 1 or 2 channels")
	}
	if sampleRate <= 0 {
		panic("audio: NewSound takes a positive sample rate")
	}
	return &Sound{samples: samples[:len(samples)/channels*channels], channels: channels, rate: sampleRate}
}

// Duration returns how long the sound plays.
func (s *Sound) Duration() time.Duration {
	return frames(int64(len(s.samples)/s.channels), s.rate)
}

// NewPlayer returns a paused player of the sound.
func (s *Sound) NewPlayer() *Player {
	src := &memSource{samples: s.samples, channels: s.channels}
	return newPlayer(src, s.rate, s.channels, int64(len(s.samples)/s.channels))
}

// Play plays the sound from its start, over any other of its players, and
// returns the player playing it.
func (s *Sound) Play() *Player {
	p := s.NewPlayer()
	p.Play()
	return p
}

// OpenStream returns a paused player decoding r as it plays, for music and
// other long audio: it decodes about a second ahead. Closing the player
// closes r if r is an io.Closer.
func OpenStream(r io.ReadSeeker) (*Player, error) {
	d, _, err := NewDecoder(r)
	if err != nil {
		return nil, err
	}
	if c, ok := r.(io.Closer); ok {
		d = closingDecoder{d, c}
	}
	return NewPlayer(d), nil
}

type closingDecoder struct {
	Decoder
	io.Closer
}

// NewPlayer returns a paused player decoding d as it plays, about a second
// ahead. d is used by one goroutine at a time. Closing the player closes d
// if d is an io.Closer.
func NewPlayer(d Decoder) *Player {
	ch := min(d.Channels(), 2)
	s := newStream(d)
	p := newPlayer(s, d.SampleRate(), ch, d.Len())
	// A player dropped while paused would leave its decoder waiting; one
	// dropped while playing plays to its end, as a sound's do.
	runtime.AddCleanup(p, func(v *voice) {
		mu.Lock()
		defer mu.Unlock()
		if v.active {
			v.orphan = true
		} else {
			v.close()
		}
	}, p.v)
	return p
}

// readerOf buffers r when it reads small amounts at a time, as a file.
func readerOf(r io.Reader) io.Reader {
	if _, ok := r.(*bytes.Reader); ok {
		return r
	}
	return bufio.NewReader(r)
}

// frames returns the duration of n frames at rate.
func frames(n int64, rate int) time.Duration {
	return time.Duration(float64(n) * float64(time.Second) / float64(rate))
}

// A Player plays a sound or a stream: its position, volume, pan, speed and
// looping are its own.
type Player struct {
	v *voice
}

func newPlayer(src source, rate, ch int, length int64) *Player {
	v := &voice{src: src, rate: float64(rate), mono: ch == 1, length: length, volume: 1, speed: 1}
	v.reset(0)
	return &Player{v}
}

// Play plays from where the player is, or from the start if it ended.
func (p *Player) Play() {
	mu.Lock()
	defer mu.Unlock()
	v := p.v
	if v.closed || v.playing {
		return
	}
	if v.ended() {
		v.src.seek(0)
		v.reset(0)
	}
	v.playing = true
	if !v.active {
		// A sound starting at its start starts at once; one resuming in
		// the middle fades in, as one pausing fades out, without a click.
		// One still fading out ramps back from where it is.
		v.snap = v.idx+3 <= 0
		v.fadeIn = !v.snap
		v.active = true
		voices = append(voices, v)
		wakeOutput()
	}
}

// Pause pauses the player where it is.
func (p *Player) Pause() {
	mu.Lock()
	defer mu.Unlock()
	p.v.playing = false
}

// Stop pauses the player and moves it to the start.
func (p *Player) Stop() {
	mu.Lock()
	defer mu.Unlock()
	v := p.v
	if v.closed {
		return
	}
	v.playing = false
	v.src.seek(0)
	v.reset(0)
}

// Playing reports whether the player plays: false once paused, stopped,
// closed, or at its end.
func (p *Player) Playing() bool {
	mu.Lock()
	defer mu.Unlock()
	return p.v.playing
}

// Seek moves the player to t from the start.
func (p *Player) Seek(t time.Duration) {
	mu.Lock()
	defer mu.Unlock()
	v := p.v
	if v.closed {
		return
	}
	f := max(int64(math.Round(t.Seconds()*v.rate)), 0)
	if v.length >= 0 {
		f = min(f, v.length)
	}
	v.src.seek(f)
	v.reset(f)
}

// Position returns how far the player is from the start.
func (p *Player) Position() time.Duration {
	mu.Lock()
	defer mu.Unlock()
	return time.Duration(p.v.position() / p.v.rate * float64(time.Second))
}

// Duration returns how long the player's audio is, or 0 if unknown.
func (p *Player) Duration() time.Duration {
	if p.v.length < 0 {
		return 0
	}
	return frames(p.v.length, int(p.v.rate))
}

// SetVolume sets the player's volume, a gain: 1 (the default) as decoded,
// 0 silent. Changes ramp over a few milliseconds, without a click.
func (p *Player) SetVolume(v float64) *Player {
	mu.Lock()
	defer mu.Unlock()
	p.v.volume = max(v, 0)
	return p
}

// Volume returns the player's volume.
func (p *Player) Volume() float64 {
	mu.Lock()
	defer mu.Unlock()
	return p.v.volume
}

// SetPan places the player between the left speaker (-1) and the right
// (1); 0, the default, is the middle. A mono sound pans keeping its
// power; a stereo one keeps both channels, as Web Audio's StereoPannerNode
// does.
func (p *Player) SetPan(pan float64) *Player {
	mu.Lock()
	defer mu.Unlock()
	p.v.pan = min(max(pan, -1), 1)
	return p
}

// Pan returns the player's pan.
func (p *Player) Pan() float64 {
	mu.Lock()
	defer mu.Unlock()
	return p.v.pan
}

// SetSpeed sets how fast the player plays, which changes its pitch with
// it, as a record's speed does: 1 (the default) as decoded, 2 twice as
// fast and an octave higher. Games vary it a little for effects that
// play often.
func (p *Player) SetSpeed(s float64) *Player {
	mu.Lock()
	defer mu.Unlock()
	if s > 0 && !math.IsInf(s, 0) {
		p.v.speed = min(s, 16)
	}
	return p
}

// Speed returns the player's speed.
func (p *Player) Speed() float64 {
	mu.Lock()
	defer mu.Unlock()
	return p.v.speed
}

// SetLoop sets whether the player starts again from the start when it
// ends, without a gap, or from its loop's start (SetLoopRange). A change
// is heard after what the player decoded ahead: a few milliseconds of a
// sound, up to a second of a stream.
func (p *Player) SetLoop(loop bool) *Player {
	mu.Lock()
	defer mu.Unlock()
	if !p.v.closed {
		p.v.loop = loop
		p.v.src.setLoop(loop, p.v.loopStart, p.v.loopEnd)
	}
	return p
}

// SetLoopRange loops the player between start and end, without a gap,
// end 0 being the end of its audio: what is before start, as a tune's
// intro, plays once, and playing past end goes on from start. It sets the
// player looping; SetLoop(false) plays on past end, as far as the end.
func (p *Player) SetLoopRange(start, end time.Duration) *Player {
	mu.Lock()
	defer mu.Unlock()
	v := p.v
	if v.closed {
		return p
	}
	s := max(int64(math.Round(start.Seconds()*v.rate)), 0)
	e := max(int64(math.Round(end.Seconds()*v.rate)), 0)
	if v.length >= 0 {
		s, e = min(s, v.length), min(e, v.length)
	}
	if e != 0 && e <= s {
		s, e = 0, 0 // nothing to loop: all of it
	}
	v.loopStart, v.loopEnd, v.loop = s, e, true
	v.src.setLoop(true, s, e)
	return p
}

// Loop reports whether the player loops.
func (p *Player) Loop() bool {
	mu.Lock()
	defer mu.Unlock()
	return p.v.loop
}

// Close stops the player for good and frees what it decodes with. A
// sound's players need no closing.
func (p *Player) Close() {
	mu.Lock()
	defer mu.Unlock()
	p.v.playing = false
	p.v.close()
}

var (
	volume = 1.0
	outErr error
	errMu  sync.Mutex
)

// SetVolume sets the volume of everything played, a gain: 1 (the default)
// as decoded, 0 silent.
func SetVolume(v float64) {
	mu.Lock()
	defer mu.Unlock()
	volume = max(v, 0)
}

// Volume returns the volume of everything played.
func Volume() float64 {
	mu.Lock()
	defer mu.Unlock()
	return volume
}

// Err returns why the last output to open is not the system's audio
// output, as on a machine with none, or nil.
func Err() error {
	errMu.Lock()
	defer errMu.Unlock()
	return outErr
}

func setErr(err error) {
	errMu.Lock()
	defer errMu.Unlock()
	outErr = err
}
