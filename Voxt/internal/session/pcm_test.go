package session

import (
	"bytes"
	"errors"
	"io"
	"os"
	"testing"
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/asr"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
)

func waitForPCMSpill(t *testing.T, b *pcmBuffer) {
	t.Helper()
	b.mu.Lock()
	done := b.spillDone
	b.mu.Unlock()
	if done == nil {
		return
	}
	select {
	case <-done:
	case <-time.After(5 * time.Second):
		t.Fatal("the PCM spill did not end")
	}
}

func TestPCMSpillDoesNotBlockCapture(t *testing.T) {
	b := newPCMBuffer()
	diskUnavailable := errors.New("disk unavailable")
	t.Cleanup(func() {
		if err := b.discard(); !errors.Is(err, diskUnavailable) {
			t.Errorf("discard reported %v, want the spool error", err)
		}
	})
	opening, release := make(chan struct{}), make(chan struct{})
	defer close(release)
	b.openSpool = func() (pcmSpool, error) {
		close(opening)
		<-release
		return nil, diskUnavailable
	}
	b.append(make([]byte, pcmSpillAt))
	select {
	case <-opening:
	case <-time.After(5 * time.Second):
		t.Fatal("the spool was never opened")
	}

	result := make(chan error, 1)
	go func() {
		tail := []byte{21, 34}
		b.append(tail)
		got, closed, err := b.next(t.Context(), pcmSpillAt)
		if err == nil && (closed || !bytes.Equal(tail, got)) {
			err = errors.New("the feeder missed the PCM appended while opening the spool")
		}
		result <- err
	}()
	select {
	case err := <-result:
		if err != nil {
			t.Fatal(err)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("capture or the feeder blocked on disk I/O")
	}
}

type failingPCMSpool struct {
	pcmSpool
	err error
}

func (f failingPCMSpool) WriteAt(p []byte, off int64) (int, error) {
	if off == 0 {
		return f.pcmSpool.WriteAt(p, off)
	}
	n, err := f.pcmSpool.WriteAt(p[:min(len(p), 16)], off)
	return n, errors.Join(f.err, err)
}

func TestPCMSpillFailureKeepsEveryByte(t *testing.T) {
	diskFull := errors.New("disk full")
	b := newPCMBuffer()
	t.Cleanup(func() {
		if err := b.discard(); !errors.Is(err, diskFull) {
			t.Errorf("discard reported %v, want the spill error", err)
		}
	})
	b.openSpool = func() (pcmSpool, error) {
		f, err := openPCMSpool()
		if err != nil {
			return nil, err
		}
		return failingPCMSpool{f, diskFull}, nil
	}
	prefix := bytes.Repeat([]byte{11}, pcmSpillAt)
	suffix := bytes.Repeat([]byte{22}, pcmSpillBatch)
	b.append(prefix)
	waitForPCMSpill(t, b)
	b.append(suffix)
	waitForPCMSpill(t, b)
	b.append([]byte{33, 33})
	b.close()
	b.mu.Lock()
	err, stored := b.spillErr, b.stored
	b.mu.Unlock()
	if !errors.Is(err, diskFull) || stored != len(prefix) {
		t.Fatalf("spill state: %d bytes and %v", stored, err)
	}

	want := bytes.Join([][]byte{prefix, suffix, {33, 33}}, nil)
	got, err := b.bytes()
	if err != nil || !bytes.Equal(want, got) {
		t.Fatalf("file ASR read %d bytes, error %v", len(got), err)
	}
	streamed := make([]byte, 0, len(want))
	for {
		pcm, closed, err := b.next(t.Context(), len(streamed))
		if err != nil {
			t.Fatal(err)
		}
		if closed && len(pcm) == 0 {
			break
		}
		streamed = append(streamed, pcm...)
	}
	if !bytes.Equal(want, streamed) {
		t.Fatalf("streaming ASR read %d bytes of %d", len(streamed), len(want))
	}
}

type pcmRetryASR struct {
	*fakeStreamer
	*fakeFile
}

func (f pcmRetryASR) Caps() asr.Caps { return asr.Caps{Stream: true, File: true} }

func TestLongRecordingSpoolsAndCanRetry(t *testing.T) {
	open := make(chan struct{})
	provider := pcmRetryASR{
		fakeStreamer: &fakeStreamer{openGate: open, silent: true, finalErr: &asr.Error{Kind: asr.KindServer}},
		fakeFile:     &fakeFile{text: "recovered"},
	}
	h := newHarness(t, provider, nil, func(_ *Config, s *settings.Settings) {
		s.Audio.Clean.Off = true
		s.Audio.VAD.Off = true
		s.Privacy.KeepAudio = false
	})
	recording := tone(pcmSpillAt+pcmSpillBatch+asr.BytesPerSecond/10, 215, 0.3)
	h.audio.file, h.audio.drained = recording, make(chan struct{})
	id := h.start()
	t.Cleanup(func() { h.m.Cancel(id) })
	select {
	case <-h.audio.drained:
	case <-time.After(5 * time.Second):
		t.Fatal("the microphone did not deliver the recording")
	}
	h.m.mu.Lock()
	buffer := h.m.cur.pcm
	h.m.mu.Unlock()
	waitForPCMSpill(t, buffer)
	buffer.mu.Lock()
	stored, tailCapacity, file := buffer.stored, cap(buffer.data), buffer.file
	buffer.mu.Unlock()
	if stored < pcmSpillAt || tailCapacity >= pcmSpillAt || file == nil {
		t.Fatalf("before Stop, stored %d bytes with %d bytes of tail capacity", stored, tailCapacity)
	}
	if f := file.(*os.File); f.Name() == "" {
		t.Fatal("expected a temporary recording")
	} else if _, err := os.Stat(f.Name()); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("the temporary recording still has a directory entry: %v", err)
	}
	close(open)
	h.m.Stop(id)
	end := h.rec.wait(t)
	if end.State != Failed || !end.Recording || end.Timings.Audio != pcmDuration(len(recording)) {
		t.Fatalf("session did not preserve retry audio: %+v", end)
	}
	stream := provider.streams[0]
	stream.mu.Lock()
	sent := stream.data
	stream.mu.Unlock()
	if !bytes.Equal(sent, recording) {
		t.Fatalf("stream received %d bytes of %d", len(sent), len(recording))
	}
	entries := h.hist.all()
	if len(entries) != 1 || entries[0].Audio == "" || entries[0].AudioUntil.IsZero() {
		t.Fatalf("history did not keep the failed recording for Retry: %+v", entries)
	}
	wav, err := os.ReadFile(h.hist.AudioPath(entries[0].Audio))
	if err != nil || !bytes.Equal(wav, asr.WAV(recording, asr.SampleRate)) {
		t.Fatalf("saved WAV differs from the original recording: %v", err)
	}
	if _, err := file.ReadAt(make([]byte, 1), 0); !errors.Is(err, os.ErrClosed) {
		t.Fatalf("the spool is still open after publishing the result: %v", err)
	}
	if _, _, err := buffer.next(t.Context(), 0); !errors.Is(err, io.ErrClosedPipe) {
		t.Fatalf("read after session disposal: %v", err)
	}
	retried, err := h.m.Retry(t.Context(), RetryRequest{Entry: entries[0]})
	if err != nil || retried.Result != "recovered" || retried.Audio != "" {
		t.Fatalf("retry = %+v, %v", retried, err)
	}
	if !bytes.Equal(provider.audio, recording) {
		t.Fatalf("retry received %d bytes of %d", len(provider.audio), len(recording))
	}
}
