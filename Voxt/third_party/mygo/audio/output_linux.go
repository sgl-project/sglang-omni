//go:build linux && (amd64 || arm64)

package audio

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sync"
	"unsafe"

	"github.com/ebitengine/purego"
)

// Linux plays through PulseAudio's simple API, which PipeWire serves as
// well, or else ALSA's default device. Both block while the device holds
// enough, from the output's goroutine.

const linuxFrames = sampleRate / 100 // per write: 10 ms

var (
	pulseLoad sync.Once
	pulseErr  error

	paSimpleNew   func(server, name *byte, dir int32, dev, stream *byte, spec *paSampleSpec, chmap, attr unsafe.Pointer, err *int32) uintptr
	paSimpleWrite func(s uintptr, data unsafe.Pointer, bytes uintptr, err *int32) int32
	paSimpleFree  func(s uintptr)
	paStrerror    func(err int32) string

	alsaLoad sync.Once
	alsaErr  error

	sndPcmOpen      func(pcm *uintptr, name *byte, stream int32, mode int32) int32
	sndPcmSetParams func(pcm uintptr, format, access int32, channels, rate uint32, softResample int32, latency uint32) int32
	sndPcmWritei    func(pcm uintptr, buf unsafe.Pointer, frames uintptr) int
	sndPcmRecover   func(pcm uintptr, err int32, silent int32) int32
	sndPcmClose     func(pcm uintptr) int32
	sndStrerror     func(err int32) string
)

type paSampleSpec struct {
	format   int32
	rate     uint32
	channels uint8
}

type paBufferAttr struct {
	maxLength, tLength, preBuf, minReq, fragSize uint32
}

const (
	paStreamPlayback  = 1
	paSampleFloat32LE = 5

	sndPcmStreamPlayback      = 0
	sndPcmFormatFloatLE       = 14
	sndPcmAccessRWInterleaved = 3
)

func cstr(s string) *byte {
	b := append([]byte(s), 0)
	return &b[0]
}

func appName() string {
	if exe, err := os.Executable(); err == nil {
		return filepath.Base(exe)
	}
	return "mygo"
}

func openDevice() (device, int, error) {
	d, perr := openPulse()
	if perr == nil {
		return d, linuxFrames, nil
	}
	d, aerr := openALSA()
	if aerr == nil {
		return d, linuxFrames, nil
	}
	return nil, 0, errors.Join(perr, aerr)
}

type pulseDevice struct{ s uintptr }

func openPulse() (device, error) {
	pulseLoad.Do(func() {
		lib, err := purego.Dlopen("libpulse-simple.so.0", purego.RTLD_NOW|purego.RTLD_GLOBAL)
		if err != nil {
			pulseErr = err
			return
		}
		purego.RegisterLibFunc(&paSimpleNew, lib, "pa_simple_new")
		purego.RegisterLibFunc(&paSimpleWrite, lib, "pa_simple_write")
		purego.RegisterLibFunc(&paSimpleFree, lib, "pa_simple_free")
		purego.RegisterLibFunc(&paStrerror, lib, "pa_strerror")
	})
	if pulseErr != nil {
		return nil, pulseErr
	}
	spec := paSampleSpec{format: paSampleFloat32LE, rate: sampleRate, channels: channels}
	// About 40 ms ahead, asking for more every 10 ms.
	const frame = channels * 4
	attr := paBufferAttr{
		maxLength: ^uint32(0),
		tLength:   sampleRate / 25 * frame,
		preBuf:    ^uint32(0),
		minReq:    linuxFrames * frame,
		fragSize:  ^uint32(0),
	}
	var code int32
	name := appName()
	s := paSimpleNew(nil, cstr(name), paStreamPlayback, nil, cstr(name), &spec, nil, unsafe.Pointer(&attr), &code)
	if s == 0 {
		return nil, fmt.Errorf("audio: PulseAudio: %s", paStrerror(code))
	}
	return &pulseDevice{s}, nil
}

func (d *pulseDevice) write(samples []float32) error {
	var code int32
	if paSimpleWrite(d.s, unsafe.Pointer(&samples[0]), uintptr(len(samples)*4), &code) < 0 {
		return fmt.Errorf("audio: PulseAudio: %s", paStrerror(code))
	}
	return nil
}

func (d *pulseDevice) close() { paSimpleFree(d.s) }

type alsaDevice struct{ pcm uintptr }

func openALSA() (device, error) {
	alsaLoad.Do(func() {
		lib, err := purego.Dlopen("libasound.so.2", purego.RTLD_NOW|purego.RTLD_GLOBAL)
		if err != nil {
			alsaErr = err
			return
		}
		purego.RegisterLibFunc(&sndPcmOpen, lib, "snd_pcm_open")
		purego.RegisterLibFunc(&sndPcmSetParams, lib, "snd_pcm_set_params")
		purego.RegisterLibFunc(&sndPcmWritei, lib, "snd_pcm_writei")
		purego.RegisterLibFunc(&sndPcmRecover, lib, "snd_pcm_recover")
		purego.RegisterLibFunc(&sndPcmClose, lib, "snd_pcm_close")
		purego.RegisterLibFunc(&sndStrerror, lib, "snd_strerror")
	})
	if alsaErr != nil {
		return nil, alsaErr
	}
	var pcm uintptr
	if e := sndPcmOpen(&pcm, cstr("default"), sndPcmStreamPlayback, 0); e < 0 {
		return nil, fmt.Errorf("audio: ALSA: %s", sndStrerror(e))
	}
	if e := sndPcmSetParams(pcm, sndPcmFormatFloatLE, sndPcmAccessRWInterleaved, channels, sampleRate, 1, 40000); e < 0 {
		sndPcmClose(pcm)
		return nil, fmt.Errorf("audio: ALSA: %s", sndStrerror(e))
	}
	return &alsaDevice{pcm}, nil
}

func (d *alsaDevice) write(samples []float32) error {
	for len(samples) > 0 {
		n := sndPcmWritei(d.pcm, unsafe.Pointer(&samples[0]), uintptr(len(samples)/channels))
		if n < 0 {
			// An underrun or a suspend, recovered from by preparing the
			// device again.
			if e := sndPcmRecover(d.pcm, int32(n), 1); e < 0 {
				return fmt.Errorf("audio: ALSA: %s", sndStrerror(e))
			}
			continue
		}
		samples = samples[n*channels:]
	}
	return nil
}

func (d *alsaDevice) close() { sndPcmClose(d.pcm) }
