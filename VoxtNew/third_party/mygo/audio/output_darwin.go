package audio

import (
	"fmt"
	"sync"
	"sync/atomic"
	"time"
	"unsafe"

	"github.com/ebitengine/purego"
)

// macOS plays through an Audio Queue on the default output, which follows
// it when it changes.

type audioStreamBasicDescription struct {
	sampleRate       float64
	formatID         uint32
	formatFlags      uint32
	bytesPerPacket   uint32
	framesPerPacket  uint32
	bytesPerFrame    uint32
	channelsPerFrame uint32
	bitsPerChannel   uint32
	reserved         uint32
}

type audioQueueBuffer struct {
	audioDataBytesCapacity uint32
	audioData              unsafe.Pointer
	audioDataByteSize      uint32
	userData               uintptr
	packetDescCapacity     uint32
	packetDescs            uintptr
	packetDescCount        uint32
}

const (
	kAudioFormatLinearPCM   = 0x6C70636D // 'lpcm'
	kAudioFormatFlagIsFloat = 1
	aqBuffers               = 4
	aqFrames                = 512 // per buffer: 10.7 ms
)

var (
	aqLoad sync.Once
	aqErr  error

	audioQueueNewOutput      func(format *audioStreamBasicDescription, callback uintptr, user uintptr, runLoop, runLoopMode uintptr, flags uint32, out *uintptr) int32
	audioQueueAllocateBuffer func(q uintptr, size uint32, out **audioQueueBuffer) int32
	audioQueueEnqueueBuffer  func(q uintptr, b *audioQueueBuffer, n uint32, descs uintptr) int32
	audioQueueStart          func(q uintptr, at uintptr) int32
	audioQueueStop           func(q uintptr, immediate bool) int32
	audioQueueDispose        func(q uintptr, immediate bool) int32

	// aqCallback is created once; queues route buffers back to the device
	// open, through aqCurrent.
	aqCallback uintptr
	aqCurrent  atomic.Pointer[aqDevice]
)

func loadAudioToolbox() error {
	aqLoad.Do(func() {
		lib, err := purego.Dlopen("/System/Library/Frameworks/AudioToolbox.framework/AudioToolbox", purego.RTLD_LAZY|purego.RTLD_GLOBAL)
		if err != nil {
			aqErr = err
			return
		}
		purego.RegisterLibFunc(&audioQueueNewOutput, lib, "AudioQueueNewOutput")
		purego.RegisterLibFunc(&audioQueueAllocateBuffer, lib, "AudioQueueAllocateBuffer")
		purego.RegisterLibFunc(&audioQueueEnqueueBuffer, lib, "AudioQueueEnqueueBuffer")
		purego.RegisterLibFunc(&audioQueueStart, lib, "AudioQueueStart")
		purego.RegisterLibFunc(&audioQueueStop, lib, "AudioQueueStop")
		purego.RegisterLibFunc(&audioQueueDispose, lib, "AudioQueueDispose")
		// The queue's own thread calls it when it is done with a buffer.
		aqCallback = purego.NewCallback(func(user, q uintptr, b *audioQueueBuffer) {
			if d := aqCurrent.Load(); d != nil && d.q == q {
				select {
				case d.free <- b:
				default:
				}
			}
		})
	})
	return aqErr
}

type aqDevice struct {
	q       uintptr
	free    chan *audioQueueBuffer
	queued  int // buffers enqueued before the queue started
	started bool
}

func openDevice() (device, int, error) {
	if err := loadAudioToolbox(); err != nil {
		return nil, 0, err
	}
	desc := audioStreamBasicDescription{
		sampleRate:       sampleRate,
		formatID:         kAudioFormatLinearPCM,
		formatFlags:      kAudioFormatFlagIsFloat,
		bytesPerPacket:   channels * 4,
		framesPerPacket:  1,
		bytesPerFrame:    channels * 4,
		channelsPerFrame: channels,
		bitsPerChannel:   32,
	}
	d := &aqDevice{free: make(chan *audioQueueBuffer, aqBuffers)}
	if st := audioQueueNewOutput(&desc, aqCallback, 0, 0, 0, 0, &d.q); st != 0 {
		return nil, 0, fmt.Errorf("audio: AudioQueueNewOutput failed (OSStatus %d)", st)
	}
	for range aqBuffers {
		var b *audioQueueBuffer
		if st := audioQueueAllocateBuffer(d.q, aqFrames*channels*4, &b); st != 0 {
			audioQueueDispose(d.q, true)
			return nil, 0, fmt.Errorf("audio: AudioQueueAllocateBuffer failed (OSStatus %d)", st)
		}
		d.free <- b
	}
	aqCurrent.Store(d)
	return d, aqFrames, nil
}

func (d *aqDevice) write(samples []float32) error {
	var b *audioQueueBuffer
	select {
	case b = <-d.free:
	case <-time.After(time.Second):
		// The queue stopped taking buffers, as when its device went.
		return fmt.Errorf("audio: the audio queue stopped")
	}
	copy(unsafe.Slice((*float32)(b.audioData), len(samples)), samples)
	b.audioDataByteSize = uint32(len(samples) * 4)
	if st := audioQueueEnqueueBuffer(d.q, b, 0, 0); st != 0 {
		return fmt.Errorf("audio: AudioQueueEnqueueBuffer failed (OSStatus %d)", st)
	}
	if !d.started {
		// Start once every buffer holds something, not to begin with an
		// underrun.
		if d.queued++; d.queued == aqBuffers {
			if st := audioQueueStart(d.q, 0); st != 0 {
				return fmt.Errorf("audio: AudioQueueStart failed (OSStatus %d)", st)
			}
			d.started = true
		}
	}
	return nil
}

func (d *aqDevice) close() {
	aqCurrent.CompareAndSwap(d, nil)
	audioQueueStop(d.q, true)
	audioQueueDispose(d.q, true)
}
