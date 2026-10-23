package helpertest

import (
	"bufio"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"slices"
	"strconv"
	"strings"
	"time"

	"github.com/sgl-project/sglang-omni/VoxtNew/internal/helper/proto"
)

// ---- capture ----

func (f *fake) cancelled(id string) bool {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.cancel[id]
}

// startCapture runs a capture.start job on a goroutine of its own; one at
// a time.
func (f *fake) startCapture(m proto.Message) {
	var p proto.CaptureParams
	if err := m.DecodeParams(&p); err != nil {
		f.fail(m.ID, proto.CodeBadRequest, "Bad params: %v", err)
		return
	}
	f.mu.Lock()
	if f.mic == proto.PermNotDetermined {
		f.mic = proto.PermGranted
	}
	mic := f.mic
	f.mu.Unlock()
	if mic != proto.PermGranted {
		f.failData(m.ID, proto.CodePermissionDenied, map[string]string{"permission": proto.PermissionMicrophone}, "VoxtNew may not use the microphone")
		return
	}
	if len(p.Devices) > 0 && p.Devices[0] == "gone" {
		f.failData(m.ID, proto.CodeDeviceUnavailable, map[string]string{"device": "gone"}, "No input gave audio")
		return
	}
	f.mu.Lock()
	if len(f.running) > 0 {
		f.mu.Unlock()
		f.fail(m.ID, proto.CodeBusy, "A recording is running already")
		return
	}
	f.running[m.ID] = true
	stop, done := make(chan struct{}), make(chan struct{})
	f.stop[m.ID], f.finished[m.ID] = stop, done
	f.mu.Unlock()
	go func() {
		defer func() {
			f.mu.Lock()
			delete(f.running, m.ID)
			delete(f.cancel, m.ID)
			delete(f.stop, m.ID)
			delete(f.finished, m.ID)
			f.mu.Unlock()
			close(done)
		}()
		f.capture(m.ID, p, stop)
	}()
}

// source is the audio a capture plays: pcm= at the requested rate, then
// silence; or a tone.
type source struct {
	rate int
	pcm  []byte // PCM16 LE at rate; nil: the tone
	pos  int    // samples given
}

func (s *source) next(n int) []byte {
	out := make([]byte, 2*n)
	if s.pcm == nil {
		for i := range n {
			v := 0.25 * math.Sin(2*math.Pi*440*float64(s.pos+i)/float64(s.rate))
			binary.LittleEndian.PutUint16(out[2*i:], uint16(int16(v*32767)))
		}
	} else if 2*s.pos < len(s.pcm) {
		copy(out, s.pcm[2*s.pos:])
	}
	s.pos += n
	return out
}

// level is the frame's RMS and peak in dBFS.
func level(frame []byte) (rms, peak float64) {
	var sum, max float64
	n := len(frame) / 2
	for i := range n {
		v := math.Abs(float64(int16(binary.LittleEndian.Uint16(frame[2*i:]))) / 32768)
		sum += v * v
		max = math.Max(max, v)
	}
	db := func(x float64) float64 {
		if x <= 1e-6 {
			return -120
		}
		return math.Max(-120, math.Round(200*math.Log10(x))/10)
	}
	if n == 0 {
		return -120, -120
	}
	return db(math.Sqrt(sum / float64(n))), db(max)
}

// capture records until capture.stop, job.cancel, maxSeconds or a fake
// device's end.
func (f *fake) capture(id string, p proto.CaptureParams, stop chan struct{}) {
	rate := p.SampleRate
	if rate <= 0 {
		rate = 16000
	}
	dev := Devices()[0]
	fellBack := false
	for _, d := range p.Devices {
		if i := slices.IndexFunc(Devices(), func(x proto.CaptureDevice) bool { return x.ID == d }); i >= 0 {
			dev = Devices()[i]
			break
		}
		if d != "crash" && d != "lost" {
			fellBack = true
		}
	}
	src := &source{rate: rate}
	if path := f.opts["pcm"]; path != "" {
		pcm, err := ReadPCM(path, rate)
		if err != nil {
			f.fail(id, proto.CodeInternal, "pcm: %v", err)
			return
		}
		src.pcm = pcm
	}
	started := proto.CaptureStartedParams{Job: id, Device: dev.ID, DeviceName: dev.Name, FellBack: fellBack,
		DeviceSampleRate: dev.SampleRate, DeviceChannels: dev.Channels, SampleRate: rate, StartMS: 5}
	if fellBack {
		started.FallbackReason = "input " + p.Devices[0] + " not found"
	}
	f.notify(proto.MethodCaptureStarted, started)
	frameMS := p.FrameMS
	if frameMS <= 0 {
		frameMS = 100
	}
	wallMS := frameMS
	if ms, err := strconv.Atoi(f.opts["frame-ms"]); err == nil && ms > 0 {
		wallMS = ms
	}
	samples := rate * frameMS / 1000
	tick := time.NewTicker(time.Duration(wallMS) * time.Millisecond)
	defer tick.Stop()
	seq, total, reason, peak := 0, 0, "", -120.0
	first := 0.0
	t0 := time.Now()
	for reason == "" {
		select {
		case <-stop:
			reason = proto.CaptureStopped
			continue
		case <-tick.C:
		}
		if f.cancelled(id) {
			f.fail(id, proto.CodeCancelled, "The recording was cancelled")
			return
		}
		frame := src.next(samples)
		rms, pk := level(frame)
		peak = math.Max(peak, pk)
		seconds := float64(total+samples) / float64(rate)
		f.notify(proto.MethodCaptureLevel, proto.LevelParams{Job: id, Seconds: seconds, RMS: rms, Peak: pk})
		f.notify(proto.MethodCapturePCM, proto.PCMParams{Job: id, Seq: seq, Offset: total, Data: frame})
		if seq == 0 {
			first = float64(time.Since(t0).Microseconds())/1000 + 1
		}
		seq++
		total += samples
		if len(p.Devices) > 0 && p.Devices[0] == "crash" {
			fmt.Fprintln(os.Stderr, "fatal: the audio engine stopped")
			os.Exit(3)
		}
		if len(p.Devices) > 0 && p.Devices[0] == "lost" && seq == 3 {
			reason = proto.CaptureDeviceLost
		}
		if p.MaxSeconds > 0 && seconds >= p.MaxSeconds {
			reason = proto.CaptureMaxSeconds
		}
	}
	f.result(id, proto.CaptureResult{Device: dev.ID, Reason: reason, Frames: seq, Samples: total,
		Seconds: float64(total) / float64(rate), Peak: peak, FirstFrameMS: first, CPUSeconds: 0.001})
}

// captureStop ends a recording; the job answers first.
func (f *fake) captureStop(m proto.Message) {
	var p proto.CaptureStopParams
	m.DecodeParams(&p)
	if p.Job == "" {
		f.fail(m.ID, proto.CodeBadRequest, "Missing job")
		return
	}
	f.mu.Lock()
	stop, ok := f.stop[p.Job]
	done := f.finished[p.Job]
	delete(f.stop, p.Job)
	f.mu.Unlock()
	if !ok {
		f.result(m.ID, proto.CaptureStopResult{Stopped: false})
		return
	}
	close(stop)
	go func() {
		<-done
		f.result(m.ID, proto.CaptureStopResult{Stopped: true})
	}()
}

// ReadPCM reads a WAV file (PCM16 mono) or raw PCM16 LE, resampled
// (linearly) from the WAV's rate to rate; raw PCM is taken to be at rate.
func ReadPCM(path string, rate int) ([]byte, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	if len(data) < 12 || string(data[:4]) != "RIFF" || string(data[8:12]) != "WAVE" {
		return data[:len(data)&^1], nil
	}
	var from, channels, bits int
	var pcm []byte
	for off := 12; off+8 <= len(data); {
		id, size := string(data[off:off+4]), int(binary.LittleEndian.Uint32(data[off+4:]))
		body := data[off+8 : min(len(data), off+8+size)]
		switch id {
		case "fmt ":
			if len(body) < 16 || binary.LittleEndian.Uint16(body) != 1 {
				return nil, fmt.Errorf("%s: not PCM", path)
			}
			channels, from, bits = int(binary.LittleEndian.Uint16(body[2:])), int(binary.LittleEndian.Uint32(body[4:])), int(binary.LittleEndian.Uint16(body[14:]))
		case "data":
			pcm = body
		}
		off += 8 + size + size%2
	}
	if channels != 1 || bits != 16 || from <= 0 {
		return nil, fmt.Errorf("%s: want 16-bit mono, have %d channels of %d bits", path, channels, bits)
	}
	if from == rate {
		return pcm[:len(pcm)&^1], nil
	}
	n := len(pcm) / 2
	m := int(int64(n) * int64(rate) / int64(from))
	out := make([]byte, 2*m)
	for i := range m {
		x := float64(i) * float64(from) / float64(rate)
		j := int(x)
		a := float64(int16(binary.LittleEndian.Uint16(pcm[2*j:])))
		b := a
		if j+1 < n {
			b = float64(int16(binary.LittleEndian.Uint16(pcm[2*j+2:])))
		}
		binary.LittleEndian.PutUint16(out[2*i:], uint16(int16(math.Round(a+(b-a)*(x-float64(j))))))
	}
	return out, nil
}

// WAV wraps mono PCM16 in a WAV header.
func WAV(pcm []byte, rate int) []byte {
	b := make([]byte, 44, 44+len(pcm))
	copy(b, "RIFF")
	binary.LittleEndian.PutUint32(b[4:], uint32(36+len(pcm)))
	copy(b[8:], "WAVEfmt ")
	binary.LittleEndian.PutUint32(b[16:], 16)
	binary.LittleEndian.PutUint16(b[20:], 1)
	binary.LittleEndian.PutUint16(b[22:], 1)
	binary.LittleEndian.PutUint32(b[24:], uint32(rate))
	binary.LittleEndian.PutUint32(b[28:], uint32(rate*2))
	binary.LittleEndian.PutUint16(b[32:], 2)
	binary.LittleEndian.PutUint16(b[34:], 16)
	copy(b[36:], "data")
	binary.LittleEndian.PutUint32(b[40:], uint32(len(pcm)))
	return append(b, pcm...)
}

// replay answers requests from a fixture transcript: each request takes
// the next unused fixture request of the same method, and the helper-bound
// lines that follow it are written with the fixture's ids replaced by the
// live ones. Hello is answered by the fake itself.
func (f *fake) replay(path string) {
	data, err := os.ReadFile(path)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(2)
	}
	type line struct {
		To  string          `json:"to"`
		Msg json.RawMessage `json:"msg"`
	}
	var lines []line
	for _, l := range strings.Split(strings.TrimSpace(string(data)), "\n") {
		var ln line
		if err := json.Unmarshal([]byte(l), &ln); err != nil || ln.Msg == nil {
			continue
		}
		lines = append(lines, ln)
	}
	used := make([]bool, len(lines))
	ids := map[string]string{} // fixture id → live id
	rewrite := func(raw json.RawMessage) string {
		var m proto.Message
		json.Unmarshal(raw, &m)
		if to, ok := ids[m.ID]; ok {
			m.ID = to
		}
		if len(m.Params) > 0 {
			var params map[string]any
			json.Unmarshal(m.Params, &params)
			if job, ok := params["job"].(string); ok {
				if to, ok := ids[job]; ok {
					params["job"] = to
				}
			}
			m.Params, _ = json.Marshal(params)
		}
		out, _ := json.Marshal(m)
		return string(out)
	}
	sc := bufio.NewScanner(os.Stdin)
	sc.Buffer(make([]byte, 64*1024), proto.MaxLine)
	for sc.Scan() {
		m, err := proto.Decode(sc.Bytes())
		if err != nil {
			continue
		}
		if m.Method == proto.MethodHello {
			f.result(m.ID, proto.HelloResult{Protocol: proto.Version, Version: "replay", PID: os.Getpid(), Methods: Methods()})
			continue
		}
		found := -1
		for i, ln := range lines {
			if used[i] || ln.To != "helper" {
				continue
			}
			var fm proto.Message
			json.Unmarshal(ln.Msg, &fm)
			if fm.Method == m.Method {
				found = i
				ids[fm.ID] = m.ID
				break
			}
		}
		if found < 0 {
			f.fail(m.ID, proto.CodeMethodNotFound, "not in the fixture: %s", m.Method)
			continue
		}
		used[found] = true
		for i := found + 1; i < len(lines) && lines[i].To == "app"; i++ {
			used[i] = true
			f.writeRaw(rewrite(lines[i].Msg))
		}
	}
}
